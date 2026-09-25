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
from collections import Counter, deque
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

#: The outcomes that are read a second time before they are believed: all but `stopped`.
#: `corrupt`, `complete` and `truncated` end the walk for good, and `pax` makes the archive
#: `unsupported` for good. `crossed` does not end anything, but it is where the next
#: segment's chain is judged to begin, and a wrong one resets that segment onto bytes that
#: are not a header.
_CONFIRMED = frozenset({"corrupt", "complete", "truncated", "pax", "crossed"})

#: How many of the latest fetches that served a header a verdict is read again from. One
#: bad fetch serves every header in it, so the header that sent a walk astray is the first
#: one the bad fetch served -- not necessarily the last member recorded -- and where it
#: lands can be a real header too, inside a stored tarball, whose own fetch serves more
#: members before the verdict comes. Two fetches cover both.
CONFIRM_FILLS = 2

#: How many times a walk will accept "that read was contradicted by the next" while
#: confirming its verdict, before it gives up and raises UnsettledRead. A bad read is rare
#: and independent; a bad archive is neither, so a verdict that keeps changing is the
#: server, not the archive.
REREAD_LIMIT = 3

#: What a pass that crossed its boundary, or was stopped, read where it ended: nothing.
_NOT_READ = object()


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
                            # (and pax, which walk() raises as UnsupportedArchive)
    end_offset: int
    members: int
    detail: str = ""


def walk(concat, start_offset: int, commit, *, rewind, batch_size: int = 2000,
         stop_at: int | None = None, should_stop=None) -> WalkResult:
    """Walk from ``start_offset``, calling ``commit(members, next_offset)`` per batch.

    No outcome is believed on one read, except a stop. Reaching a verdict -- ``corrupt``,
    ``complete``, ``truncated``, a pax header, or a crossing of ``stop_at`` -- drops the
    reader's cache and walks again over the last CONFIRM_FILLS fetches that served a
    header, comparing every header sequence with the first reading of it. Two readings
    that agree, verdict included, stand. Where they part, the place is read again until
    two reads agree on it (`_tie_break`); a reading that loses there was a bad read, and
    the walk goes where the winner says. A genuinely damaged archive
    reads the same way twice and reaches the verdict it always did, a few fetches later.
    A verdict that keeps changing, past REREAD_LIMIT contradicted reads, raises
    UnsettledRead; a confirmed pax header raises UnsupportedArchive.

    ``commit`` is expected to write the members and the cursor in one transaction, so
    that an interrupted walk resumes from exactly the last committed offset. It is called
    exactly once per batch and exactly once more at the end of each pass, the last call
    reflecting the offset finally reached. A second reading that agrees with the first
    records nothing again.

    ``rewind(offset)`` must drop every member the caller holds for this chain at or past
    ``offset`` and put its cursor there, in one transaction. It is called when a second
    reading shows the first one recorded members past a header it read wrongly.

    ``stop_at`` ends the walk once the next header lies at or past that offset, which
    is how one segment's chain confirms where its successor's chain must begin. A
    walk whose ``start_offset`` already lies at or past ``stop_at`` has nothing of its
    own left to record: that start IS the crossing, so it commits nothing and returns
    immediately.

    ``should_stop``, given a zero-argument predicate, is checked after every member is
    recorded -- not only at a batch boundary -- so a caller reacting to something like
    SIGINT never waits longer than one header hop before the walk commits what it has
    and returns with state ``stopped``. The one exception is a second reading while it
    still agrees with the first: that is a few fetches long, and runs to its verdict.
    """
    # A chain whose start already lies at or past its boundary has nothing of its own to
    # record: that start IS the crossing, and the member beginning there belongs to the
    # next segment. Without this, two chains record the same member, and when one of them
    # is later re-walked, dropping its rows deletes the other's copy too.
    if stop_at is not None and start_offset >= stop_at:
        commit([], start_offset)
        return WalkResult("crossed", start_offset, 0)

    # A verdict read once is not believed. Reading again only where the walk landed is
    # not enough: a bad read that is valid tar from elsewhere passes its checksum and moves
    # the chain by its own size, and the landing, read truly, repeats the verdict every
    # time -- `corrupt` in member data, `unsupported` on a stored tarball's pax header, or
    # a crossing at the wrong header. What decided the verdict is the fetch that served the
    # wrong header, so the second reading starts where the last fetches that served a
    # header began, and follows the chain to the verdict again.
    #
    # A verdict is its state and the offset it was reached at, not its wording, and the
    # first of two agreeing readings is the one reported. The state matters as much as the
    # offset: a lie and the truth about a terminator land on the same one.
    position, seen, rescued, prior, budget = start_offset, 0, [], None, None
    while True:
        this = _walk_chain(concat, position, commit, rewind, batch_size, stop_at,
                           should_stop, seen, rescued, prior)
        seen, result = this.seen, this.result
        if result.state not in _CONFIRMED:
            break                           # stopped: the next run reads it all again
        if prior is not None:
            if (result.state, result.end_offset) == (prior.result.state,
                                                     prior.result.end_offset):
                result = replace(prior.result, members=seen)
                break
            if not this.diverged:
                rescued.append(this.vpos)   # the two readings parted here
        if budget is None:
            budget = len(rescued) + REREAD_LIMIT
        if len(rescued) > budget:
            where = ", ".join(str(o) for o in dict.fromkeys(rescued))
            raise UnsettledRead(
                f"reads kept contradicting each other at {where}; the last said "
                f"{result.state} at {result.end_offset} and was never confirmed")
        prior = this
        concat.drop_cache()
        position = this.rewind_to()
    if rescued:
        where = ", ".join(str(o) for o in dict.fromkeys(rescued))
        note = f"re-read at {where}: a read there was contradicted by the next"
        result = replace(result, detail=f"{result.detail}; {note}" if result.detail
                         else note)
    if result.state == "pax":
        raise UnsupportedArchive(result.detail)
    return result


@dataclass
class _Pass:
    """One pass along the chain, as the pass that reads it again needs it."""
    result: WalkResult
    start: int
    window: list            # (before, key) of the members its last CONFIRM_FILLS fetches
                            # served: where each header sequence begins, and what was there
    vpos: int               # where the header sequence it did not record begins
    version: object         # what it read at vpos: a key, None for no member, or _NOT_READ
    seen: int
    diverged: bool          # a third read took it off the path of the pass before it

    def rewind_to(self) -> int:
        return self.window[0][0] if self.window else self.start

    def versions(self) -> dict:
        read = dict(self.window)
        if self.version is not _NOT_READ:
            read[self.vpos] = self.version
        return read


def _key(info, after):
    """A member as two readings of it are compared: everything recorded, and where the
    next header sequence begins."""
    return _identity(info), after


def _walk_chain(concat, start_offset: int, commit, rewind, batch_size: int, stop_at,
                should_stop, seen: int, rescued: list, prior):
    """One pass of the chain from ``start_offset``; returns its `_Pass`.

    ``prior`` is None for a first pass. Otherwise it is the pass being read again: this
    one begins where that one's window does and compares each header sequence it reads
    with that pass's reading of the same offset. While they agree the pass is *matching*:
    the store already holds those rows, up to ``prior.vpos`` (the *floor*), so it records
    and counts nothing, and every commit carries no members and the floor as its cursor
    -- a cursor short of it would resume over rows already written. Where the readings
    differ, `_tie_break` reads the place again: if the earlier reading wins, the pass
    follows it and goes on matching; if this one wins, the earlier one's rows from there
    on were a wrong reading, `rewind` drops them, and this pass records from there as a
    first pass would. A pass that ends while matching, short of the floor, disagrees with
    the earlier reading about what lies there, and the rows past it go the same way.

    A member settled by a third read is added to ``rescued``.
    """
    total = concat.archive.total_size
    window = deque()                # (before, the fetch that finished its header, key)
    matching = prior is not None
    floor = prior.vpos if matching else None
    versions = prior.versions() if matching else {}
    diverged = False
    batch: list[Member] = []

    def cursor(offset):
        return max(offset, floor) if matching else offset

    def leave(at):
        nonlocal matching, seen
        if matching:
            if at < floor:
                rewind(at)
                seen -= sum(1 for before, _ in prior.window if before >= at)
            matching = False

    def finish(vpos, version, verdict):
        """End the pass at ``vpos``, having read ``version`` there; ``verdict(seen)``
        says what that means, and is asked only once the batch is committed."""
        if matching and vpos < floor:
            leave(vpos)
        commit(batch, cursor(vpos))
        return _Pass(verdict(seen), start_offset, [(b, k) for b, _, k in window], vpos,
                     version, seen, diverged)

    def found_nothing(before, how):
        """The verdict where no member begins at ``before``, as a walk arriving there
        reports it: ``how`` is tarfile's ReadError message, or None if it saw no member."""
        if how is None:
            return lambda n: _classify_end(concat, before, n)
        if how in _RAN_OUT:
            return lambda n: WalkResult("truncated", total, n, how)
        return lambda n: WalkResult("corrupt", before, n, f"{how} after the header at {before}")

    concat.seek(start_offset)
    try:
        archive = tarfile.open(fileobj=concat, mode="r:")
    except tarfile.ReadError as exc:
        detail = str(exc)
        return finish(start_offset, None,
                      lambda n: _classify_end(concat, start_offset, n, detail))

    # Where the next member's header sequence begins. Not always its own offset: tarfile
    # reports a member after a `g` header at the member's header, past the `g`.
    before = start_offset
    while True:
        try:
            info = archive.next()
        except tarfile.ReadError as exc:
            return finish(before, None, found_nothing(before, str(exc)))
        if info is None:
            break
        try:
            info = _settled(concat, archive, info, before, rescued)
        except UnsettledRead:
            commit(batch, cursor(before))   # a resume retries exactly this member
            raise
        if isinstance(info, _NoMember):
            return finish(before, None, found_nothing(before, info.how))
        key = _key(info, archive.offset)
        theirs = versions.get(before, _NOT_READ)
        if theirs is not _NOT_READ and theirs != key and not info.pax_headers:
            try:
                fresh = _tie_break(concat, archive, before, key, theirs, rescued)
            except UnsettledRead:
                commit(batch, cursor(before))
                raise
            if not isinstance(fresh, tuple):
                # Two reads say no member begins here, whatever this pass read.
                return finish(before, None, found_nothing(before, fresh))
            # No `g` can be in force here: tarfile gives a member after one pax headers,
            # and `_settled` has either cleared them with a fresh read or confirmed pax.
            info, after = fresh
            key = _key(info, after)
            archive.offset = after
            if key != theirs:
                diverged = True
                leave(before)
        if info.pax_headers:
            # Two reads agree an `x` or `g` extended header applies to this member. Keep
            # what was walked before it, with the cursor where its headers begin: a
            # resume from past a `g` would walk on with no global header in force.
            message = (f"a pax extended header applies to the member at {before} "
                       f"({info.name!r}): this is a pax-format archive, which dbaudit does "
                       f"not index -- its names and sizes can sit outside the checksummed "
                       f"header")
            return finish(before, key, lambda n: WalkResult("pax", before, n, message))
        window.append((before, concat.fills, key))
        while window[0][1] <= concat.fills - CONFIRM_FILLS:
            window.popleft()
        if not matching:
            batch.append(Member.from_tarinfo(info))
            seen += 1
        before = archive.offset
        archive.members.clear()             # the walk is a stream; do not accumulate
        if stop_at is not None and before >= stop_at:
            return finish(before, _NOT_READ, lambda n: WalkResult("crossed", before, n))
        # Not while matching: a reading that agrees is a few fetches long, and cut short
        # it would end the run with its verdict unconfirmed -- or, under --max-batches,
        # never reach the verdict at all.
        if not matching and should_stop is not None and should_stop():
            commit(batch, before)
            return _Pass(WalkResult("stopped", before, seen), start_offset, [], before,
                         _NOT_READ, seen, diverged)
        if len(batch) >= batch_size:
            commit(batch, before)
            batch = []

    return finish(before, None, found_nothing(before, None))


def _settled(concat, archive, info, before, rescued):
    """The member to record, once nothing about it rests on one read alone.

    Two things are taken on trust from blocks no checksum covers, and each is read
    again before it is believed. A GNU long name or link target: GNU tar (99 bytes in
    oldgnu, 100 in gnu) and Python copy its start into the checksummed header after it,
    so the two must agree. And a pax extended header: it makes the archive
    `unsupported` for good, and a bad read can be valid pax from elsewhere -- a stored
    tarball -- sitting where a GNU header belongs.

    A suspect member is read afresh from ``before``, where its header sequence begins
    (for a member after a `g` header, that is the `g`). A read with nothing suspect
    about it replaces the first; two reads in a row that agree are the archive as
    written -- a writer that fills those fields some other way costs one request, not a
    refusal, and pax confirmed twice is pax. The walk goes on from where the member it
    keeps says, since a bad read can bring a header of another size.

    Two reads in a row that find no member there at all agree too: the suspect member
    was a bad read at a place the chain should never have reached -- a bad read before
    it sent the walk there. That is returned as `_NoMember`, for the walk to end there
    and confirm the verdict from the fetches that led to it.
    """
    encoding, errors = archive.encoding, archive.errors
    if not info.pax_headers and _long_fields_agree(concat, info, encoding, errors):
        return info
    previous = _identity(info)
    for _ in range(REREAD_LIMIT):
        concat.drop_cache()
        fresh = _read_here(concat, before, encoding, errors)
        if not isinstance(fresh, tuple):
            if previous is None:
                rescued.append(before)
                return _NoMember(fresh)
            previous = None
            continue
        member, after = fresh
        believable = (not member.pax_headers
                      and _long_fields_agree(concat, member, encoding, errors))
        if believable or _identity(member) == previous:
            if _identity(member) != _identity(info):
                rescued.append(before)
            if not member.pax_headers:
                # A `g` header seen only by the bad read would go on applying to every
                # member after it: the walk's own TarFile keeps it.
                archive.pax_headers.clear()
            archive.offset = after
            return member
        previous = _identity(member)
    raise UnsettledRead(
        f"the member at {before} read differently every time, and never without a long "
        f"name, link target or pax header that no second read confirmed")


@dataclass(frozen=True)
class _NoMember:
    """Reads agree no member begins here; ``how`` is as `_read_here` reports it."""
    how: str | None


def _long_fields_agree(concat, info, encoding, errors) -> bool:
    """Whether a member's long name and link target begin as the checksummed header
    after their blocks says. True for a member with no such blocks. The header was
    read a moment ago, so it is still in the window: checking costs no request."""
    if info.offset_data - info.offset <= BLOCK or info.issparse():
        return True                 # one header; or sparse maps, not names, before it
    here = concat.tell()
    concat.seek(info.offset_data - BLOCK)
    header = concat.read(BLOCK)
    concat.seek(here)
    name_field = header[0:100].split(b"\0", 1)[0].rstrip(b"/")
    link_field = header[157:257].split(b"\0", 1)[0]
    return (info.name.encode(encoding, errors).startswith(name_field)
            and info.linkname.encode(encoding, errors).startswith(link_field))


def _identity(info):
    return (info.name, info.linkname, info.size, info.type, info.mode, info.mtime,
            info.uname, info.gname, info.offset_data)


def _read_here(concat, offset, encoding=tarfile.ENCODING, errors="surrogateescape"):
    """What a walk arriving at ``offset`` finds there, parsed from what the reader returns:
    (member, where the next header lies), or -- where no member begins -- the message of
    the ReadError tarfile raised, or None if it simply saw none."""
    concat.seek(offset)
    try:
        one = tarfile.open(fileobj=concat, mode="r:", encoding=encoding, errors=errors)
    except tarfile.ReadError as exc:
        return str(exc)
    try:
        # Not next(): with no member at `offset` that seeks back a byte and quietly
        # reads the same place again -- a second request, and a retry nobody counted.
        info = one.firstmember
        return None if info is None else (info, one.offset)
    finally:
        one.close()                 # the ConcatFile is the caller's; this leaves it open


def _read_one(concat, offset, encoding=tarfile.ENCODING, errors="surrogateescape"):
    """The member whose header sequence begins at ``offset``, parsed from what the
    reader returns, and where the next header lies -- or None if none begins there."""
    found = _read_here(concat, offset, encoding, errors)
    return found if isinstance(found, tuple) else None


def _tie_break(concat, archive, before, mine, theirs, rescued):
    """Two passes read the header sequence at ``before`` differently: this one as
    ``mine``, the one before it as ``theirs`` -- each a member's `_key`, or None for no
    member there. Read it afresh until one version has two reads behind it, and return
    what the deciding read found (as `_read_here` does).

    Either way, one of the two readings was wrong, so ``before`` joins ``rescued``.
    """
    rescued.append(before)
    votes = Counter((mine, theirs))
    for _ in range(REREAD_LIMIT):
        concat.drop_cache()
        found = _read_here(concat, before, archive.encoding, archive.errors)
        version = _key(*found) if isinstance(found, tuple) else None
        votes[version] += 1
        if votes[version] >= 2:
            return found
    raise UnsettledRead(
        f"the header at {before} read differently every time, and no two reads of it "
        f"agreed")


def read_member(concat, offset) -> Member | None:
    """The member whose header sequence begins at ``offset``, as the reader has it now
    -- how a recorded row is checked against a fresh read. None if none begins there.

    Reads through ``concat``'s window: a fresh ConcatFile gives a fresh read.
    """
    fresh = _read_one(concat, offset)
    return None if fresh is None else Member.from_tarinfo(fresh[0])


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
