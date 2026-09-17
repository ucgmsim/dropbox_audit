"""Follow a tar's header chain, recording members without fetching their data.

Every header states its member's size, so the next header sits at
``offset + 512 + roundup(size, 512)``. Python's ``tarfile`` does that arithmetic and
the header parsing -- GNU long names, pax headers, base-256 sizes above 8 GiB -- and
because it seeks over member data rather than reading it, a walk over a seekable
file touches only headers.

One ``tarfile`` behaviour has to be corrected for: mid-archive it treats an *invalid*
header exactly like the end of the archive, stopping silently. So the end is verified
here rather than inferred from iteration finishing.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass

BLOCK = 512
TERMINATOR = BLOCK * 2
#: A GNU tar pads to at most one 10,240-byte record, so more zero bytes than this
#: after a terminator means the chain stopped somewhere unexpected.
TRAILING_LIMIT = 1 << 20

#: tarfile's wording when a header sequence runs off the end of the data, as opposed to
#: finding something there that is not a header. Stable across CPython versions; the tests
#: pin both branches.
_RAN_OUT = frozenset({"unexpected end of data", "empty header", "truncated header"})


@dataclass(frozen=True)
class Member:
    hdr_offset: int
    data_offset: int
    size: int
    type: str
    mode: int
    mtime: int
    uname: str
    gname: str
    dir: str
    name: str
    linkname: str

    @classmethod
    def from_tarinfo(cls, info) -> "Member":
        path = info.name.rstrip("/") if info.isdir() else info.name
        head, _, tail = path.rpartition("/")
        return cls(
            hdr_offset=info.offset, data_offset=info.offset_data, size=info.size,
            type=info.type.decode("ascii", "replace") if isinstance(info.type, bytes)
            else str(info.type),
            mode=info.mode or 0, mtime=int(info.mtime), uname=info.uname or "",
            gname=info.gname or "", dir=head, name=tail or path,
            linkname=info.linkname or "")


@dataclass(frozen=True)
class WalkResult:
    state: str              # complete | truncated | corrupt | crossed | stopped
    end_offset: int
    members: int
    detail: str = ""


def walk(concat, start_offset: int, commit, batch_size: int = 2000,
         stop_at: int | None = None, should_stop=None) -> WalkResult:
    """Walk from ``start_offset``, calling ``commit(members, next_offset)`` per batch.

    ``commit`` is expected to write the members and the cursor in one transaction, so
    that an interrupted walk resumes from exactly the last committed offset. It is
    called exactly once per batch and exactly once more at the end, on every return
    path -- including when the archive cannot even be opened at ``start_offset``.

    ``stop_at`` ends the walk once the next header lies at or past that offset, which
    is how one segment's chain confirms where its successor's chain must begin. A
    walk whose ``start_offset`` already lies at or past ``stop_at`` has nothing of its
    own left to record: that start IS the crossing, so it commits nothing and returns
    immediately.

    ``should_stop``, given a zero-argument predicate, is checked after every member is
    recorded -- not only at a batch boundary -- so a caller reacting to something like
    SIGINT never waits longer than one header hop before the walk commits what it has
    and returns with state ``stopped``.
    """
    # A chain whose start already lies at or past its boundary has nothing of its own to
    # record: that start IS the crossing, and the member beginning there belongs to the
    # next segment. Without this, two chains record the same member, and when one of them
    # is later re-walked, dropping its rows deletes the other's copy too.
    if stop_at is not None and start_offset >= stop_at:
        commit([], start_offset)
        return WalkResult("crossed", start_offset, 0)

    total = concat.archive.total_size
    concat.seek(start_offset)
    try:
        archive = tarfile.open(fileobj=concat, mode="r:")
    except tarfile.ReadError as exc:
        commit([], start_offset)
        return _classify_end(concat, start_offset, 0, str(exc))

    batch: list[Member] = []
    seen = 0
    while True:
        try:
            info = archive.next()
        except tarfile.ReadError as exc:
            commit(batch, archive.offset)
            if str(exc) in _RAN_OUT:
                return WalkResult("truncated", total, seen, str(exc))
            return WalkResult("corrupt", archive.offset, seen,
                              f"{exc} after the header at {archive.offset}")
        if info is None:
            break
        batch.append(Member.from_tarinfo(info))
        seen += 1
        archive.members.clear()             # the walk is a stream; do not accumulate
        if stop_at is not None and archive.offset >= stop_at:
            commit(batch, archive.offset)
            return WalkResult("crossed", archive.offset, seen)
        if should_stop is not None and should_stop():
            commit(batch, archive.offset)
            return WalkResult("stopped", archive.offset, seen)
        if len(batch) >= batch_size:
            commit(batch, archive.offset)
            batch = []

    end = archive.offset
    commit(batch, end)
    return _classify_end(concat, end, seen)


def _first_non_zero(buf: bytes) -> int | None:
    rest = buf.lstrip(b"\x00")
    return None if not rest else len(buf) - len(rest)


def _classify_end(concat, end: int, seen: int, detail: str = "") -> WalkResult:
    total = concat.archive.total_size
    remainder = total - end
    if remainder < TERMINATOR:
        return WalkResult("truncated", end, seen,
                          detail or f"{remainder} bytes left, need {TERMINATOR}")
    concat.seek(end)
    probe = concat.read(TERMINATOR)
    if any(probe):
        return WalkResult("corrupt", end, seen,
                          f"not a header and not a terminator at {end}: {probe.hex()}")
    if remainder > TRAILING_LIMIT:
        return WalkResult("corrupt", end, seen,
                          f"terminator at {end} with {remainder} bytes after it")
    concat.seek(end)
    tail = concat.read(remainder)
    bad = _first_non_zero(tail)
    if bad is not None:
        return WalkResult("corrupt", end, seen,
                          f"non-zero byte at {end + bad}, after the terminator at {end}")
    return WalkResult("complete", end, seen, detail)


def find_chain_start(concat, from_offset: int, scan_read: int = 16 << 20,
                     limit: int | None = None) -> int | None:
    """The first checksum-valid header at or after ``from_offset``.

    A part boundary falls wherever split(1) put it, which is almost always inside a
    member, so a chain that starts at one has to find its footing. Task 1 measured
    this: one seed found a header 21 MiB in, five others found none within 64 MiB
    because they sat inside larger members. So there is no default limit, and reads
    are large -- a request costs 1.60 s whether it carries 512 bytes or 16 MiB.

    ``from_offset`` is rounded up to the next block boundary before scanning: a
    header can only start on a 512-byte boundary, and rounding down could return one
    that starts before ``from_offset``, breaking the "at or after" contract.

    A valid header here is not proof: a tar stored inside the tar offers perfectly
    good ones. The joining rule in the CLI is what settles it.
    """
    total = concat.archive.total_size
    offset = -(-from_offset // BLOCK) * BLOCK   # round up: a header only starts on a block
    scanned = 0
    while offset < total and (limit is None or scanned < limit):
        length = min(scan_read, total - offset)
        if length < BLOCK:
            return None
        concat.seek(offset)
        buf = concat.read(length)
        if not buf:
            return None
        for position in range(0, len(buf) - BLOCK + 1, BLOCK):
            block = buf[position:position + BLOCK]
            if block[257:262] != b"ustar":
                continue
            try:
                tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
            except tarfile.HeaderError:
                continue
            return offset + position
        offset += len(buf) - (len(buf) % BLOCK)
        scanned += length
    return None
