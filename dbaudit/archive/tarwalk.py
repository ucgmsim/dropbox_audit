"""Follow a tar's header chain, recording members without fetching their data.

Every header states its member's size, so the next header sits at
``offset + 512 + roundup(size, 512)``. Python's ``tarfile`` does that arithmetic and
the header parsing -- GNU long names, base-256 sizes above 8 GiB -- and because it
seeks over member data rather than reading it, a walk over a seekable file touches
only headers.

GNU and ustar archives only. A pax archive can keep a member's name and size outside
the checksummed header, where a misread moves every header after it; the walk raises
UnsupportedArchive rather than index one without the guarantees it relies on.

One ``tarfile`` behaviour has to be corrected for: mid-archive it treats an *invalid*
header exactly like the end of the archive, stopping silently. So the end is verified
here rather than inferred from iteration finishing.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass, replace

BLOCK = 512
TERMINATOR = BLOCK * 2
#: A GNU tar pads to at most one 10,240-byte record, so more zero bytes than this
#: after a terminator means the chain stopped somewhere unexpected.
TRAILING_LIMIT = 1 << 20

#: tarfile's wording when a header sequence runs off the end of the data, as opposed to
#: finding something there that is not a header. Stable across CPython versions; the tests
#: pin both branches.
_RAN_OUT = frozenset({"unexpected end of data", "empty header", "truncated header"})

#: The verdicts that end a walk for good, and so are never believed on one read.
#: `crossed` and `stopped` are not final -- something walks on from them either way.
#: `truncated` means the chain itself ran off the end of the data, which takes a
#: checksum-valid header claiming too much; a bad read cannot manufacture one.
_FINAL = frozenset({"corrupt", "complete"})

#: How many times one walk will accept "that read was contradicted by the next" before
#: it gives up and raises UnsettledRead. A bad read is rare and independent; a bad
#: archive is neither, so a verdict that keeps changing is the server, not the archive.
REREAD_LIMIT = 3


class UnsettledRead(Exception):
    """Reads of the same bytes kept disagreeing, past REREAD_LIMIT.

    That is a fact about the server, not the archive, so it is raised rather than
    returned as a verdict. The CLI records it as the segment's error, and the next run
    reclaims the segment and resumes from its committed cursor.
    """


class UnsupportedArchive(Exception):
    """The archive is in a tar format this walker does not index: pax.

    Everything here rests on a member's name, size and type sitting in its 512-byte
    header, where a checksum vouches for them. In a pax archive any of those can live
    instead in an extended header's data blocks, which carry no checksum, and a size
    read wrongly there moves every header after it. Rather than index that with
    guarantees it does not have, the walk stops and says so.
    """


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

    No verdict that ends the walk -- ``corrupt`` or ``complete`` -- is believed on one
    read. Reaching one drops the reader's cache and walks again from where it landed,
    so the deciding bytes come from a second request; a verdict that repeats stands,
    and one that changes means the first read was wrong and the walk carries on, which
    the result's detail records. A genuinely damaged archive reads the same way twice
    and reaches the verdict it always did, one short pass later.

    ``commit`` is expected to write the members and the cursor in one transaction, so
    that an interrupted walk resumes from exactly the last committed offset. It is
    called exactly once per batch and exactly once more at the end of each pass -- so
    more than once at the end when a verdict was read twice, the extra calls carrying
    no members, and the last call always reflecting the offset finally reached.

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

    # No verdict that ends the walk is believed on one read. Reaching one drops the
    # cache and walks again from where it landed, which re-reads the deciding bytes
    # from the server: a verdict that repeats is the archive talking, and one that
    # changes means a read was wrong -- so the walk carries on and asks again until two
    # consecutive reads agree. That covers each of the three ways `_classify_end` says
    # corrupt, and `complete` too, which is the silent failure: a bad read of zeros would
    # otherwise end a six-hour walk claiming success, members missing, and `index` would
    # then refuse to walk it again.
    #
    # A verdict is its state and the offset it was reached at, not its wording: a pass
    # that walks into damage and one that opens on it describe the same bytes two ways
    # -- "bad checksum after the header at H" against "bad checksum at H" -- and that is
    # not the bytes changing. The first is the truer, which is why an agreeing retry
    # reports the first pass. The state matters as much as the offset: a lie and the
    # truth about a terminator land on the same one.
    #
    # Nothing here stops a retry for landing where it started. That is exactly where the
    # confirming read of a genuine `complete` lands, and where a walk resumed on its own
    # cursor meets its first read. The loop needs no such guard to end: after the first
    # final pass, each one either agrees with the last and stops, or disagrees and counts
    # against REREAD_LIMIT.
    position, seen, rescued, verdict, first = start_offset, 0, [], None, None
    while True:
        result, seen = _walk_chain(concat, position, commit, batch_size, stop_at,
                                   should_stop, seen)
        again = (result.state, result.end_offset)
        if verdict is not None and again != verdict:
            rescued.append(position)        # reading `position` again changed the answer
        if result.state not in _FINAL:
            break
        if again == verdict:
            result = first                  # two reads agree: keep the first of them,
            break                           # which says what actually stopped the walk
        if len(rescued) > REREAD_LIMIT:
            where = ", ".join(str(o) for o in dict.fromkeys(rescued))
            raise UnsettledRead(
                f"reads kept contradicting each other at {where}; the last said "
                f"{result.state} at {result.end_offset} and was never confirmed")
        verdict, first = again, result
        concat.drop_cache()
        position = result.end_offset
    if rescued:
        where = ", ".join(str(o) for o in dict.fromkeys(rescued))
        note = f"re-read at {where}: a read there was contradicted by the next"
        result = replace(result, detail=f"{result.detail}; {note}" if result.detail
                         else note)
    return result


def _walk_chain(concat, start_offset: int, commit, batch_size: int, stop_at, should_stop,
                seen: int):
    """One pass of the chain from ``start_offset``. Returns (result, members so far)."""
    total = concat.archive.total_size
    concat.seek(start_offset)
    try:
        archive = tarfile.open(fileobj=concat, mode="r:")
    except tarfile.ReadError as exc:
        commit([], start_offset)
        return _classify_end(concat, start_offset, seen, str(exc)), seen

    batch: list[Member] = []
    while True:
        try:
            info = archive.next()
        except tarfile.ReadError as exc:
            commit(batch, archive.offset)
            if str(exc) in _RAN_OUT:
                return WalkResult("truncated", total, seen, str(exc)), seen
            return WalkResult("corrupt", archive.offset, seen,
                              f"{exc} after the header at {archive.offset}"), seen
        if info is None:
            break
        if info.pax_headers:
            # tarfile has already applied an `x` or `g` extended header to this member.
            # Keep what was walked before it; the cursor stops short of it, not past.
            commit(batch, info.offset)
            raise UnsupportedArchive(
                f"a pax extended header applies to the member at {info.offset} "
                f"({info.name!r}): this is a pax-format archive, which dbaudit does not "
                f"index -- its names and sizes can sit outside the checksummed header")
        batch.append(Member.from_tarinfo(info))
        seen += 1
        archive.members.clear()             # the walk is a stream; do not accumulate
        if stop_at is not None and archive.offset >= stop_at:
            commit(batch, archive.offset)
            return WalkResult("crossed", archive.offset, seen), seen
        if should_stop is not None and should_stop():
            commit(batch, archive.offset)
            return WalkResult("stopped", archive.offset, seen), seen
        if len(batch) >= batch_size:
            commit(batch, archive.offset)
            batch = []

    end = archive.offset
    commit(batch, end)
    return _classify_end(concat, end, seen), seen


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
        # When tarfile has already said what is wrong -- a damaged header *after* a GNU
        # long name, say -- that is the diagnosis; the block here may be a perfectly
        # good header, and calling it "not a header" sends the reader the wrong way.
        what = detail or "not a header and not a terminator"
        return WalkResult("corrupt", end, seen, f"{what} at {end}: {probe.hex()}")
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
