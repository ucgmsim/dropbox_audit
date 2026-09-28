import io
import random
import tarfile

import pytest

from dbaudit.archive.reader import ConcatFile, LocalRangeReader, ReaderError
from dbaudit.archive.tarwalk import (CONFIRM_FILLS, REREAD_LIMIT, UnsettledRead,
                                     UnsupportedArchive, find_chain_start, walk)
from tests.archive_fakes import build_tar, write_parts

MEMBERS = [(f"run/file{i:03d}.bin", bytes([i % 251]) * (1000 + i)) for i in range(50)]


def collect(tmp_path, data, part_size=4096, start=0, batch_size=7, window=None):
    archive = write_parts(tmp_path, data, part_size=part_size)
    windows = {} if window is None else {"window_min": window, "window_max": window}
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive), **windows)
    seen, cursors = [], []

    def commit(members, next_offset):
        seen.extend(members)
        cursors.append(next_offset)

    result = walk(handle, start, commit, batch_size=batch_size)
    return seen, cursors, result


def test_the_walk_matches_what_tarfile_itself_reports(tmp_path):
    data = build_tar(MEMBERS)
    seen, _, result = collect(tmp_path, data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as reference:
        expected = [(m.name, m.size, m.offset, m.offset_data) for m in reference]
    assert [(f"{m.dir}/{m.name}" if m.dir else m.name, m.size, m.hdr_offset, m.data_offset)
            for m in seen] == expected
    assert result.state == "complete"
    assert result.members == len(MEMBERS)


def test_the_manifest_does_not_depend_on_where_the_parts_divide(tmp_path):
    data = build_tar(MEMBERS)
    baseline = None
    for index, part_size in enumerate((512, 1024, 4096, 100_000, len(data))):
        directory = tmp_path / f"p{index}"
        directory.mkdir()
        seen, _, result = collect(directory, data, part_size=part_size)
        manifest = [(m.name, m.size, m.hdr_offset) for m in seen]
        assert result.state == "complete"
        baseline = baseline if baseline is not None else manifest
        assert manifest == baseline


def test_gnu_long_names_survive_but_pax_is_refused(tmp_path):
    """The plan asked for both formats to walk. arr65 ruled on 2026-09-23 that pax is out
    of scope and should fail loudly instead: in pax, names and sizes can live outside the
    checksummed header, and a size misread there moves every later header while a retry
    re-reads only the wrong place. GNU long names keep the size in the header, so a
    misread can garble a name but never move the chain."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    gnu = tmp_path / "gnu"
    gnu.mkdir()
    seen, _, result = collect(gnu, build_tar([(long_name, b"x" * 10)], format=tarfile.GNU_FORMAT))
    assert result.state == "complete"
    assert f"{seen[0].dir}/{seen[0].name}" == long_name

    pax = tmp_path / "pax"
    pax.mkdir()
    with pytest.raises(UnsupportedArchive, match="pax"):
        collect(pax, build_tar([(long_name, b"x" * 10)], format=tarfile.PAX_FORMAT))


def test_directories_symlinks_and_empty_files(tmp_path):
    data = build_tar([
        ("run/", b"", {"type": tarfile.DIRTYPE}),
        ("run/link", b"", {"type": tarfile.SYMTYPE, "linkname": "file000.bin"}),
        ("run/empty.bin", b""),
    ])
    seen, _, result = collect(tmp_path, data)
    assert result.state == "complete"
    assert [m.type for m in seen] == ["5", "2", "0"]
    assert seen[1].linkname == "file000.bin"


def test_a_missing_final_part_reports_truncated(tmp_path):
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data[:len(data) - 4096], part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    result = walk(handle, 0, lambda members, offset: None)
    assert result.state == "truncated"


class _Counting(LocalRangeReader):
    """Counts the fetches that cover ``watch``."""

    def __init__(self, directory, archive, watch):
        super().__init__(directory, archive)
        self.watch, self.covering = watch, 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        start = self.archive.parts[part_idx].offset + offset
        self.covering += start <= self.watch < start + length
        return data


def test_a_truncated_verdict_is_read_twice_before_it_stands(tmp_path):
    """A chain runs off the end because the last header's size claims more than is
    left. That size is one read's word, so it is read again before `truncated` stands:
    one pass, which commits nothing it has read only once, then a second over the fetches
    that served the last headers -- which reads that header afresh, agrees, and commits."""
    data = build_tar(MEMBERS)
    last = _header_offsets(data)[-1]
    archive = write_parts(tmp_path, data[:len(data) - 4096], part_size=4096)
    assert last + 512 + 1049 > archive.total_size       # its member runs off the end
    reader = _Counting(tmp_path, archive, last)
    calls = []

    result = walk(ConcatFile(archive, reader), 0,
                  lambda members, offset: calls.append(len(members)), batch_size=1000)

    assert result.state == "truncated"
    assert "re-read at" not in result.detail
    # Each pass's end commits nothing read only once; then what both readings agreed on.
    assert calls == [0, 0, len(MEMBERS)]
    assert reader.covering == 2, "the header that ran out was read twice"


def test_a_well_formed_bad_read_cannot_fake_a_truncated_archive(tmp_path):
    """A bad read that is valid tar from elsewhere -- here, an early 40 KB member's
    header served where a late member's belongs -- passes its checksum and claims more
    bytes than the archive has left. Believed on one read, a complete archive would be
    reported truncated; read again, the true header is there and the walk carries on."""
    members = [("run/big.bin", b"b" * 40000)] + MEMBERS
    data = build_tar(members)
    offsets = _header_offsets(data)
    late = offsets[-3]                    # fewer than 40,000 bytes follow this header
    assert len(data) - late < 40000
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, late, fill=data[0:512], span=512)
    rows = {}

    def commit(batch, offset):
        rows.update((m.hdr_offset, m.name) for m in batch)      # INSERT OR REPLACE

    result = walk(ConcatFile(archive, reader), 0, commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert [rows[o] for o in offsets] == [n.rsplit("/", 1)[-1] for n, _ in members]
    assert result.members == len(members), "the member read twice was counted twice"
    assert f"re-read at {late}" in result.detail


def test_a_missing_middle_part_is_detected(tmp_path):
    """Dropping a part shifts every later offset, so the chain lands in member data.

    The set is built directly rather than through from_entries, which would refuse a
    gap outright -- this is testing what the walker does when one slips past.
    """
    from dbaudit.archive.parts import ArchiveSet, Part

    data = build_tar(MEMBERS)
    full = write_parts(tmp_path, data, part_size=4096)
    (tmp_path / full.parts[2].name).unlink()
    kept, offset = [], 0
    for idx, part in enumerate(p for p in full.parts if p.idx != 2):
        kept.append(Part(idx=idx, name=part.name, size=part.size, offset=offset,
                         content_hash=part.content_hash))
        offset += part.size
    archive = ArchiveSet(kept)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    result = walk(handle, 0, lambda members, offset: None)
    assert result.state in {"corrupt", "truncated"}


def test_a_corrupt_header_is_not_mistaken_for_the_end(tmp_path):
    data = bytearray(build_tar(MEMBERS))
    third = 512 + (-(-len(MEMBERS[0][1]) // 512) * 512)     # header of the second member
    data[third:third + 8] = b"\xff" * 8
    seen, _, result = collect(tmp_path, bytes(data))
    assert result.state == "corrupt"
    assert result.end_offset == third
    # Ruling D: the whole probe is recorded, not just that some detail exists.
    assert "ff" * 8 in result.detail


def test_a_corrupt_second_terminator_block_is_not_hidden_by_a_truncated_probe(tmp_path):
    """Ruling D: the whole 1024-byte probe belongs in the detail, not a 64-byte
    prefix. The test above corrupts the first 8 bytes of the probe, which a 64-byte
    prefix would also catch -- it doesn't discriminate the fix. A corruption sitting
    in the *second* terminator block, behind a first block that looks like a valid
    zero start of a terminator, is the case a truncated prefix hides completely."""
    data = bytearray(build_tar(MEMBERS))
    end = sum(512 + (-(-len(payload) // 512) * 512) for _, payload in MEMBERS)
    assert data[end:end + 1024] == b"\x00" * 1024        # build_tar's own terminator
    data[end + 512:end + 520] = b"\xff" * 8              # corrupt only the second block
    seen, _, result = collect(tmp_path, bytes(data))
    assert result.state == "corrupt"
    assert result.end_offset == end
    assert "ff" * 8 in result.detail


def test_a_corrupt_first_block_is_read_twice_and_reports_corrupt(tmp_path):
    """tarfile.open() itself raises when the very first header is invalid (offset 0
    is special-cased inside tarfile.next()). That path must still commit (Ruling H), and
    -- as for every final verdict -- it is read a second time before it stands, so it
    commits once per pass: twice, both empty, both at 0. A walk used to believe a bad
    first block outright, and segment 0 is the chain the whole join hangs off."""
    data = bytearray(build_tar(MEMBERS))
    data[0:8] = b"\xff" * 8
    archive = write_parts(tmp_path, bytes(data), part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    calls = []
    result = walk(handle, 0, lambda members, offset: calls.append((list(members), offset)))
    assert result.state == "corrupt"
    assert calls == [([], 0), ([], 0)]


def test_a_corrupt_header_after_a_long_name_is_corrupt_not_truncated(tmp_path):
    """tarfile raises SubsequentHeaderError -- wrapped as a plain ReadError -- when a
    GNU long-name header's follow-up header is corrupt. That is a corrupt archive,
    not a short one (Ruling E). Paired with test_a_missing_final_part_reports_truncated
    above, which is the genuinely-short half."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    data = bytearray(build_tar(
        [("run/first.bin", b"a" * 100), (long_name, b"x" * 10)], format=tarfile.GNU_FORMAT))
    first_span = 512 + (-(-100 // 512) * 512)
    longname_header = tarfile.TarInfo.frombuf(bytes(data[first_span:first_span + 512]),
                                              "utf-8", "surrogateescape")
    real_header = first_span + 512 + (-(-longname_header.size // 512) * 512)
    data[real_header:real_header + 8] = b"\xff" * 8
    seen, _, result = collect(tmp_path, bytes(data))
    assert result.state == "corrupt"
    assert "bad checksum" in result.detail
    assert [m.name for m in seen] == ["first.bin"]


def test_non_zero_bytes_after_the_terminator_are_corrupt_not_complete(tmp_path):
    """The spec's `complete` means zeros through to the end of the last part. Two
    tars concatenated (`cat a.tar b.tar`) put a terminator in the middle -- reporting
    that as complete would silently drop everything after it (Ruling C)."""
    data = bytearray(build_tar(MEMBERS) + b"\x00" * 3000)
    bad_at = len(data) - 500
    data[bad_at] = 0x7A
    _, _, result = collect(tmp_path, bytes(data))
    assert result.state == "corrupt"
    assert str(bad_at) in result.detail


def test_a_tar_inside_the_tar_is_skipped_as_data(tmp_path):
    inner = build_tar([("inner/a.bin", b"a" * 2000)])
    data = build_tar([("run/outer.tar", inner), ("run/after.bin", b"b" * 10)])
    seen, _, result = collect(tmp_path, data)
    assert result.state == "complete"
    assert [m.name for m in seen] == ["outer.tar", "after.bin"]


def test_a_walk_resumes_from_a_mid_archive_offset(tmp_path):
    data = build_tar(MEMBERS)
    seen, _, _ = collect(tmp_path, data)
    resume_at = seen[10].hdr_offset
    later, _, result = collect(tmp_path, data, start=resume_at)
    assert result.state == "complete"
    assert [m.name for m in later] == [m.name for m in seen[10:]]


def test_batches_commit_with_the_offset_to_resume_from(tmp_path):
    """One-block windows: every header is a fetch of its own, so members leave the
    held-back window a few headers behind the walk and are batched from there."""
    data = build_tar(MEMBERS)
    seen, cursors, _ = collect(tmp_path, data, batch_size=7, window=1024)
    assert len(cursors) >= len(MEMBERS) // 7
    # After the first batch the cursor is the header offset of the next member.
    assert cursors[0] == seen[7].hdr_offset


@pytest.mark.parametrize("window", [1024, 65536])
def test_should_stop_ends_the_walk_after_the_current_member(tmp_path, window):
    """A SIGINT handler wants the walk to stop within one header hop, not one whole
    batch: should_stop is checked after every member, not only at batch_size (Ruling
    G). It commits what has left the held-back window; the walk resuming from its
    end_offset reads the rest again -- together, every member, and each once."""
    data = build_tar(MEMBERS)
    reference, _, _ = collect(tmp_path, data)

    directory = tmp_path / "ruling_g"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive), window_min=window,
                        window_max=window)
    seen = []

    def commit(members, next_offset):
        seen.extend(members)

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] >= 5

    result = walk(handle, 0, commit, batch_size=2000, should_stop=should_stop)
    assert result.state == "stopped"
    assert calls["n"] == 5, "it stopped at the first check that said so"
    assert [m.name for m in seen] == [m.name for m in reference[:len(seen)]]
    assert result.end_offset == reference[len(seen)].hdr_offset

    resumed, _, resumed_result = collect(directory, data, start=result.end_offset)
    assert resumed_result.state == "complete"
    assert [m.name for m in seen + resumed] == [m.name for m in reference]


def test_a_walk_stops_at_a_segment_boundary_on_the_next_header(tmp_path):
    """A chain hands its successor a confirmed start: the first header at or past the
    boundary, which is what the joining rule compares against."""
    data = build_tar(MEMBERS)
    seen, _, _ = collect(tmp_path, data)
    boundary = seen[5].hdr_offset - 16          # mid-member, as a part boundary would be
    directory = tmp_path / "stop"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    result = walk(handle, 0, lambda members, offset: None, stop_at=boundary)
    assert result.state == "crossed"
    assert result.end_offset == seen[5].hdr_offset


def test_a_walk_starting_at_or_past_stop_at_records_nothing(tmp_path):
    """A chain whose start already lies at or past its boundary has nothing of its
    own to record: that start IS the crossing, and the member beginning there
    belongs to the next segment (Ruling A)."""
    data = build_tar(MEMBERS)
    seen, _, _ = collect(tmp_path, data)
    boundary = seen[5].hdr_offset

    directory = tmp_path / "ruling_a"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    calls = []
    result = walk(handle, boundary,
                  lambda members, offset: calls.append((list(members), offset)), stop_at=boundary)
    assert result.state == "crossed"
    assert result.end_offset == boundary
    assert result.members == 0
    assert calls == [([], boundary)]


def test_find_chain_start_locates_the_first_header_after_a_cold_offset(tmp_path):
    data = build_tar(MEMBERS)
    directory = tmp_path / "scan"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    seen, _, _ = collect(tmp_path, data)
    inside_member = seen[3].data_offset + 8
    assert find_chain_start(handle, inside_member, scan_read=4096) == seen[4].hdr_offset


def test_find_chain_start_rounds_up_never_returning_a_header_before_from_offset(tmp_path):
    """A down-rounding implementation, probed one byte past a real header, lands back
    on that same header's own offset -- before from_offset, breaking the "at or
    after" contract (Ruling B). Rounding up must skip past it to the next one."""
    data = build_tar(MEMBERS)
    directory = tmp_path / "scan_b"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    seen, _, _ = collect(tmp_path, data)
    h = seen[7].hdr_offset
    assert h % 512 == 0
    found = find_chain_start(handle, h + 1, scan_read=4096)
    assert found > h
    assert found == seen[8].hdr_offset


def test_find_chain_start_returns_none_when_it_runs_out(tmp_path):
    directory = tmp_path / "none"
    directory.mkdir()
    archive = write_parts(directory, b"\x00" * 20000, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    assert find_chain_start(handle, 0, scan_read=4096) is None


JUNK = bytes(range(256))


class _LiesOnce(LocalRangeReader):
    """Tells the truth except on the ``nth`` fetch that covers ``bad`` (the first, by
    default), which gets ``fill`` there instead -- what Dropbox did twice in 42,000 reads
    on 2026-09-21: a 206 whose body did not match its own Content-Range.

    ``nth`` matters because the dangerous bad read is not always the first. The walk
    reads a final verdict's bytes again to confirm it, and a lie on *that* read used to
    overturn a verdict the first read had right.

    ``span=None`` corrupts the rest of that read, which is how a bad read of zeros can
    reach `_classify_end`'s probe *and* its tail check out of one cached window.
    """

    once = True

    def __init__(self, directory, archive, bad, fill=JUNK, span=1024, nth=1):
        super().__init__(directory, archive)
        self.bad, self.fill, self.span, self.nth = bad, fill, span, nth
        self.covering = self.lied = 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        start = self.archive.parts[part_idx].offset + offset
        if not start <= self.bad < start + length:
            return data
        self.covering += 1
        if not self._lies_now():
            return data
        self.lied += 1
        cut = self.bad - start
        n = length - cut if self.span is None else min(self.span, length - cut)
        fill = self.fill * (n // len(self.fill) + 1)
        return data[:cut] + fill[:n] + data[cut + n:]


    def _lies_now(self):
        return not self.once or self.covering == self.nth


class _AlwaysLies(_LiesOnce):
    """Lies every time, with different bytes each time -- so the verdict's wording
    changes but its substance does not."""

    once = False

    def read_range(self, part_idx, offset, length):
        self.fill = bytes((b + self.lied) % 256 for b in JUNK)
        return super().read_range(part_idx, offset, length)


def _header_offsets(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as reference:
        return [m.offset for m in reference]


def test_one_bad_read_does_not_condemn_the_whole_archive(tmp_path):
    """A read that came back wrong must not be able to end the walk.

    The walk stops on junk, re-reads that block from the server, finds a good header
    and carries on -- so the manifest is complete and the archive is not called corrupt
    on the strength of a single read.
    """
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, _header_offsets(data)[30])
    handle = ConcatFile(archive, reader)
    seen = []

    result = walk(handle, 0, lambda members, offset: seen.extend(members))

    assert reader.lied == 1, "the test did not actually inject a bad read"
    assert result.state == "complete"
    assert result.members == len(MEMBERS)
    assert [m.name for m in seen] == [n.rsplit("/", 1)[-1] for n, _ in MEMBERS]
    assert "re-read at" in result.detail


def test_bytes_that_are_wrong_twice_are_still_corrupt(tmp_path):
    """The guard above must not swallow real damage: junk that is really in the file
    reads the same way every time, so the verdict stands."""
    data = bytearray(build_tar(MEMBERS))
    bad = _header_offsets(bytes(data))[30]
    data[bad:bad + 1024] = (bytes(range(256)) * 4)
    seen, _, result = collect(tmp_path, bytes(data), part_size=len(data))

    assert result.state == "corrupt"
    assert str(bad) in result.detail
    assert "re-read at" not in result.detail
    assert len(seen) == 30


def test_dropping_the_cache_makes_the_next_read_ask_again(tmp_path):
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = LocalRangeReader(tmp_path, archive)
    handle = ConcatFile(archive, reader)
    handle.seek(0)
    handle.read(512)
    served_from_cache = reader.requests
    handle.seek(0)
    handle.read(512)
    assert reader.requests == served_from_cache, "the second read should hit the cache"
    handle.drop_cache()
    handle.seek(0)
    handle.read(512)
    assert reader.requests == served_from_cache + 1


#: Big enough that the bytes after a mid-archive injection exceed TRAILING_LIMIT,
#: which is the branch `_classify_end` takes when a bad read looks like a terminator.
BIG = [(f"run/big{i:03d}.bin", bytes([i % 251]) * 120_000) for i in range(40)]


def test_a_bad_read_of_zeros_is_not_mistaken_for_the_end_of_the_archive(tmp_path):
    """The 2026-09-21 bad reads carried float data, but nothing says the next one will.
    Zeros where a header belongs look like a terminator, which is a *different* corrupt
    verdict -- and one bad read must not reach it either.
    """
    data = build_tar(BIG)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, _header_offsets(data)[10], fill=b"\x00")
    handle = ConcatFile(archive, reader)
    seen = []

    result = walk(handle, 0, lambda members, offset: seen.extend(members))

    assert reader.lied == 1, "the test did not actually inject a bad read"
    assert result.state == "complete"
    assert result.members == len(BIG)
    assert len(seen) == len(BIG)


def test_a_bad_read_of_zeros_cannot_end_the_walk_claiming_success(tmp_path):
    """The worst outcome is not a false `corrupt` -- it is a false `complete`, because
    members go missing silently and `archive index` then refuses to walk the archive
    again. One read must not be able to say the archive ended here.
    """
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    # span=None zeroes the rest of that read, so the probe and the tail check both
    # come back clean out of the one bad window.
    reader = _LiesOnce(tmp_path, archive, _header_offsets(data)[40], fill=b"\x00",
                       span=None)
    handle = ConcatFile(archive, reader)
    seen = []

    result = walk(handle, 0, lambda members, offset: seen.extend(members))

    assert reader.lied == 1
    assert result.members == len(MEMBERS), "members were silently dropped"
    assert result.state == "complete"
    assert "re-read at" in result.detail


def _long_name_archive():
    """A tar whose second member carries a GNU long name, and where its real header is.

    Built the same way as test_a_corrupt_header_after_a_long_name_is_corrupt_not_truncated
    above, which is the damaged-for-real half of this pair.
    """
    long_name = "run/" + "d" * 150 + "/deep.bin"
    data = bytearray(build_tar([("run/first.bin", b"a" * 100), (long_name, b"x" * 10)],
                               format=tarfile.GNU_FORMAT))
    first_span = 512 + (-(-100 // 512) * 512)
    longname = tarfile.TarInfo.frombuf(bytes(data[first_span:first_span + 512]),
                                       "utf-8", "surrogateescape")
    return data, first_span + 512 + (-(-longname.size // 512) * 512)


# ---- GNU long names and link targets are checked against the header after them ---------
#
# Their blocks carry no checksum, but GNU tar (99 bytes in oldgnu, 100 in gnu) and Python
# copy the start of each into the checksummed header that follows. In
# `_long_name_archive` the `L` header is at 1024, its name block at 1536, and the
# member's real header at 2048.

def _with_field(data, header, start, length, value):
    """`data` with one field of the header at `header` rewritten and its checksum redone
    -- how another tar writer might have filled that field in."""
    block = bytearray(data[header:header + 512])
    block[start:start + length] = value.ljust(length, b"\0")
    block[148:156] = b" " * 8
    block[148:156] = b"%06o\0 " % sum(block)
    return bytes(data[:header]) + bytes(block) + bytes(data[header + 512:])


def _long_link_archive():
    """Like `_long_name_archive`, but the second member is a symlink whose target needs
    a GNU `K` block: `K` header at 1024, target block at 1536, real header at 2048."""
    target = "../" + "t" * 150 + "/target.bin"
    return build_tar([("run/first.bin", b"a" * 100),
                      ("run/link", b"", {"type": tarfile.SYMTYPE, "linkname": target})],
                     format=tarfile.GNU_FORMAT), target


def _rows(tmp_path, reader_class=LocalRangeReader, data=None, **reader_kw):
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = reader_class(tmp_path, archive, **reader_kw)
    rows = {}

    def commit(batch, offset):
        rows.update((m.hdr_offset, m) for m in batch)            # INSERT OR REPLACE

    result = walk(ConcatFile(archive, reader), 0, commit)
    return rows, result, reader


def test_a_bad_read_of_only_a_long_name_block_is_read_again(tmp_path):
    """The one bad read no header checksum sees: the name block garbled, the headers
    around it intact. Unchecked, the member is recorded under a garbled path."""
    data, _ = _long_name_archive()
    long_name = "run/" + "d" * 150 + "/deep.bin"
    rows, result, reader = _rows(tmp_path, _LiesOnce, bytes(data), bad=1536,
                                 fill=b"garbled/", span=512)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert f"{rows[1024].dir}/{rows[1024].name}" == long_name
    assert "re-read at 1024" in result.detail


def test_a_bad_read_of_only_a_long_link_target_block_is_read_again(tmp_path):
    data, target = _long_link_archive()
    rows, result, reader = _rows(tmp_path, _LiesOnce, data, bad=1536,
                                 fill=b"garbled/", span=512)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert rows[1024].linkname == target
    assert "re-read at 1024" in result.detail


@pytest.mark.parametrize("form", ["gnu name (100 bytes)", "oldgnu name (99 bytes)",
                                  "gnu link target (100 bytes)"])
def test_the_check_costs_nothing_when_the_writer_follows_gnu(tmp_path, form):
    """The header after the long blocks is still in the window that just served it, so
    checking against it takes no request: one fetch for the whole small archive, and one
    for the second reading from the start of the fetch that served its last header --
    two, as for an archive with no long names at all."""
    if form.startswith("oldgnu"):
        data, _ = _long_name_archive()
        long_name = "run/" + "d" * 150 + "/deep.bin"
        data = _with_field(bytes(data), 2048, 0, 100, long_name.encode()[:99])
    elif "link" in form:
        data, _ = _long_link_archive()
    else:
        data = bytes(_long_name_archive()[0])
    rows, result, reader = _rows(tmp_path, data=data)

    assert result.state == "complete"
    assert reader.requests == 2
    assert "re-read" not in result.detail


def test_a_writer_that_does_not_copy_the_name_costs_one_read_not_a_refusal(tmp_path):
    """A writer is free to put anything in the field the long name overrides. The two
    disagree, so the member is read again; the second read agrees with the first, which
    makes it the archive, not a bad read -- recorded as read, one request dearer than
    the two a conforming archive costs, and one more when the verdict is read again."""
    data, _ = _long_name_archive()
    data = _with_field(bytes(data), 2048, 0, 100, b"placeholder")
    rows, result, reader = _rows(tmp_path, data=data)

    assert result.state == "complete"
    assert rows[1024].name == "deep.bin" and rows[1024].dir == "run/" + "d" * 150
    assert reader.requests == 4
    assert "re-read" not in result.detail


def test_a_long_name_that_reads_differently_every_time_gives_up(tmp_path):
    """A server that never tells the same story twice is not an archive fact. The walk
    gives up, with the cursor where the members it has read only once begin -- here the
    first -- so the retry that `UnsettledRead` invites reads them again."""
    data, _ = _long_name_archive()
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    reader = _AlwaysLies(tmp_path, archive, 1536, span=512)
    calls = []

    with pytest.raises(UnsettledRead, match="1024"):
        walk(ConcatFile(archive, reader), 0,
             lambda members, offset: calls.append(([m.name for m in members], offset)))
    assert calls[-1] == ([], 0)


def test_a_wrong_header_after_true_long_blocks_is_replaced_and_the_chain_follows_it(
        tmp_path):
    """A well-formed bad read can put another member's valid header where this one's
    belongs, with a different size. The fresh read's member is recorded -- and the walk
    must go on from where *its* size says, not the wrong header's."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    members = [("run/first.bin", b"a" * 100), (long_name, b"x" * 10),
               ("run/big.bin", b"b" * 5000), ("run/last.bin", b"z" * 10)]
    data = build_tar(members, format=tarfile.GNU_FORMAT)
    offsets = _header_offsets(data)
    assert offsets == [0, 1024, 3072, 8704]
    rows, result, reader = _rows(tmp_path, _LiesOnce, data, bad=2048,
                                 fill=data[3072:3584], span=512)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert sorted(rows) == offsets
    assert rows[1024].size == 10
    assert [rows[o].name for o in offsets] == ["first.bin", "deep.bin", "big.bin",
                                                "last.bin"]


def _gnu_sparse_archive():
    """A GNU sparse member whose map runs on past its header into an extension block,
    as GNU tar writes for a file with more than four holes -- built by hand, because
    tarfile cannot write one. Header at 0, map block at 512, 512 stored bytes at 1024,
    terminator at 1536; the file it describes is 4096 bytes."""
    def octal(value, width):
        return b"%0*o" % (width - 1, value)

    header = bytearray(512)
    header[0:14] = b"run/sparse.bin"
    header[100:107], header[108:115], header[116:123] = b"0000644", b"0001750", b"0001750"
    header[124:135], header[136:147] = octal(512, 12), octal(1_700_000_000, 12)
    header[156:157] = tarfile.GNUTYPE_SPARSE
    header[257:265] = tarfile.GNU_MAGIC
    header[386:397], header[398:409] = octal(0, 12), octal(256, 12)
    header[482] = 1                                             # the map continues
    header[483:494] = octal(4096, 12)
    header[148:156] = b" " * 8
    header[148:156] = b"%06o\0 " % sum(header)
    extension = bytearray(512)
    extension[0:11], extension[12:23] = octal(3840, 12), octal(256, 12)
    return bytes(header) + bytes(extension) + b"s" * 512 + b"\0" * 1024


def test_a_gnu_sparse_member_is_not_mistaken_for_a_long_name(tmp_path):
    """A sparse member's header sequence is long too, but what follows its header is
    more of the sparse map -- not a header whose name field could vouch for anything.
    Read as one, it would never match, and every sparse member would cost a re-read."""
    rows, result, reader = _rows(tmp_path, data=_gnu_sparse_archive())

    assert result.state == "complete"
    assert (rows[0].name, rows[0].type, rows[0].size, rows[0].data_offset) == (
        "sparse.bin", "S", 4096, 1024)
    assert reader.requests == 2


class _Scripted(LocalRangeReader):
    """Spoils the fetches named in ``script`` -- {fetch number: (offset, what)} -- so each
    bad read lands exactly where a test needs it. ``what`` is a byte count to garble
    there, or the exact bytes to put there instead."""

    def __init__(self, directory, archive, script):
        super().__init__(directory, archive)
        self.script, self.fetches = dict(script), 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        self.fetches += 1
        if self.fetches not in self.script:
            return data
        bad, what = self.script[self.fetches]
        cut = bad - (self.archive.parts[part_idx].offset + offset)
        assert 0 <= cut < length, "the scripted fetch does not cover its offset"
        fill = what if isinstance(what, bytes) else (b"garbled/" * 128)[:what]
        return data[:cut] + fill + data[cut + len(fill):]


def test_a_fresh_read_that_also_comes_back_wrong_is_simply_read_again(tmp_path):
    """Fetch 1 garbles the name block; fetch 2, the first fresh read, garbles the long
    name's own header so no member starts there at all. That is one more bad read, not
    a verdict: fetch 3 is read, agrees with its header, and is kept."""
    data, _ = _long_name_archive()
    rows, result, reader = _rows(tmp_path, _Scripted, bytes(data),
                                 script={1: (1536, 512), 2: (1024, 512)})

    assert result.state == "complete"
    assert rows[1024].name == "deep.bin" and rows[1024].dir == "run/" + "d" * 150
    assert "re-read at 1024" in result.detail


def test_a_fresh_read_carrying_pax_is_not_believed_on_its_own_either(tmp_path):
    """The read that settles a garbled long name is a read like any other, and can be
    wrong too -- here valid pax from elsewhere, which would make the archive
    `unsupported` for good. It is not believed alone: a third read agrees with the long
    name's own header, and the walk carries on."""
    data, _ = _long_name_archive()
    pax = _pax_tar([("x/" + "p" * 120 + ".bin", b"q" * 700, None)])
    assert pax[156:157] == b"x"
    rows, result, reader = _rows(tmp_path, _Scripted, bytes(data),
                                 script={1: (1536, 512), 2: (1024, pax[:1536])})

    assert result.state == "complete"
    assert rows[1024].name == "deep.bin" and rows[1024].dir == "run/" + "d" * 150


def test_a_directory_named_in_exactly_100_bytes_costs_nothing(tmp_path):
    """GNU tar writes a long-name block for any name of 100 bytes or more, so it gives
    a directory named in exactly 100 one too -- and a header whose name field holds all
    100, trailing slash included, which tarfile strips from the member it returns."""
    name = "d" * 99 + "/"
    longlink = tarfile.TarInfo("././@LongLink")
    longlink.type, longlink.size = tarfile.GNUTYPE_LONGNAME, len(name) + 1
    directory = tarfile.TarInfo(name)
    directory.type = tarfile.DIRTYPE
    data = (longlink.tobuf(tarfile.GNU_FORMAT) + (name.encode() + b"\0").ljust(512, b"\0")
            + directory.tobuf(tarfile.GNU_FORMAT) + b"\0" * 1024)
    rows, result, reader = _rows(tmp_path, data=data)

    assert result.state == "complete"
    assert rows[0].name == "d" * 99 and rows[0].type == "5"
    assert reader.requests == 2


def test_a_damaged_long_name_member_is_diagnosed_once_and_not_called_transient(tmp_path):
    """A checksum-valid header that tarfile still cannot chain through used to spin:
    the block re-read fine every time, so the walk resumed on it over and over and
    reported a rescue that never happened. A verdict is its state and its offset, so
    reaching the same one twice settles it -- and the first pass's diagnosis is kept.
    The confirming pass opens a fresh tarfile *on* the long-name header, so it blames
    that header; only the first pass saw that the damage is in the one after it.
    """
    data, real_header = _long_name_archive()
    data[real_header:real_header + 8] = b"\xff" * 8
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    calls = []

    result = walk(handle, 0, lambda members, offset: calls.append((len(members), offset)))

    assert result.state == "corrupt"
    assert "re-read at" not in result.detail, "reported a rescue that did not happen"
    long_name_header = 1024                 # after run/first.bin's header and data block
    assert f"bad checksum after the header at {long_name_header}" in result.detail, (
        f"reported the confirming pass, which blames the long-name header: {result.detail}")
    assert len(calls) == 3, f"each pass's end, then what both agreed on: {calls}"


def test_bytes_that_are_wrong_differently_every_time_are_still_corrupt(tmp_path):
    """A server that lies afresh on every read changes the verdict's wording but not
    its substance, so retrying must stay bounded rather than run to REREAD_LIMIT."""
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _AlwaysLies(tmp_path, archive, _header_offsets(data)[30])
    handle = ConcatFile(archive, reader)
    calls = []

    result = walk(handle, 0, lambda members, offset: calls.append(offset))

    assert result.state == "corrupt"
    assert reader.lied >= 2, "the test did not actually keep lying"
    assert len(calls) == 3, f"each pass's end, then what both agreed on: {calls}"
    assert "re-read at" not in result.detail


def test_a_bad_read_after_a_long_name_header_is_also_re_read(tmp_path):
    """The block tarfile rejects is not always the one the walk stopped on: after a GNU
    long-name header it is the real header that follows. Re-walking from the member's
    own offset re-reads both, so a transient there is rescued like any other.
    """
    data, real_header = _long_name_archive()
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, real_header, fill=b"\xff", span=8)
    handle = ConcatFile(archive, reader)
    seen = []

    result = walk(handle, 0, lambda members, offset: seen.extend(members))

    assert reader.lied == 1, "the test did not actually inject a bad read"
    assert result.state == "complete"
    assert [m.name for m in seen] == ["first.bin", "deep.bin"]


class _LiesAtEach(LocalRangeReader):
    """Lies once at each of several offsets: a walk that meets one transient after
    another, which is what a long run on a flaky account looks like."""

    def __init__(self, directory, archive, offsets):
        super().__init__(directory, archive)
        self.pending = set(offsets)
        self.lied = 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        start = self.archive.parts[part_idx].offset + offset
        for bad in sorted(self.pending):
            if start <= bad < start + length:
                self.pending.discard(bad)
                self.lied += 1
                cut = bad - start
                n = min(1024, length - cut)
                return data[:cut] + (JUNK * 4)[:n] + data[cut + n:]
        return data


def _walk_with_transients(tmp_path, count):
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    offsets = _header_offsets(data)[10:10 + count]
    reader = _LiesAtEach(tmp_path, archive, offsets)
    handle = ConcatFile(archive, reader)
    seen = []
    return (walk(handle, 0, lambda members, offset: seen.extend(members)), seen, reader)


def test_a_walk_rescues_up_to_the_reread_limit(tmp_path):
    result, seen, reader = _walk_with_transients(tmp_path, REREAD_LIMIT)
    assert reader.lied == REREAD_LIMIT
    assert result.state == "complete"
    assert len(seen) == len(MEMBERS)


class _LiesOnEach(_LiesOnce):
    """Lies on every fetch covering ``bad`` whose number is in ``nths``."""

    def __init__(self, directory, archive, bad, nths, **kw):
        super().__init__(directory, archive, bad, **kw)
        self.nths = set(nths)

    def _lies_now(self):
        return self.covering in self.nths


def test_a_verdict_that_keeps_changing_gives_up_loudly(tmp_path):
    """Every other read of the terminator comes back as junk: each reading of the verdict
    contradicts the one before. One more contradiction than REREAD_LIMIT allows, and the
    walk raises rather than report a verdict it never confirmed. The segment then lands
    in `error`, which the next run reclaims and resumes from its committed cursor."""
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnEach(tmp_path, archive, _terminator(data), nths=range(1, 100, 2))

    with pytest.raises(UnsettledRead):
        walk(ConcatFile(archive, reader), 0, lambda members, offset: None)
    # One reading of the terminator per pass: the first, and one contradicting the one
    # before it for each of REREAD_LIMIT + 1 more.
    assert reader.covering == REREAD_LIMIT + 2


class _NeverTheSameTwice(LocalRangeReader):
    """Every fetch covering ``bad`` carries a different member's valid header there."""

    def __init__(self, directory, archive, bad, donors):
        super().__init__(directory, archive)
        self.bad, self.donors, self.lied = bad, list(donors), 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        start = self.archive.parts[part_idx].offset + offset
        if not start <= self.bad < start + length:
            return data
        donor = self.donors[self.lied % len(self.donors)]
        self.lied += 1
        cut = self.bad - start
        return data[:cut] + donor[:length - cut] + data[cut + 512:]


def test_a_header_that_never_reads_the_same_way_twice_gives_up_loudly(tmp_path):
    """Two readings of a header disagree, and no third read agrees with either -- nor any
    read after it. No version has two reads behind it, so none is recorded: the walk
    raises, and an honest retry from where it left the cursor finishes the job."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _NeverTheSameTwice(tmp_path, archive, offsets[30],
                                [data[o:o + 512] for o in offsets[1:10]])
    rows = _Rows()

    with pytest.raises(UnsettledRead, match=str(offsets[30])):
        walk(ConcatFile(archive, reader), 0, rows.commit)
    assert rows.consistent()
    assert all(key < offsets[30] for key in rows.rows)

    # The retry UnsettledRead invites starts where the members read only once begin.
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), rows.cursor,
                  rows.commit)
    assert result.state == "complete"
    assert rows.names() == _names_of(MEMBERS)


# ---- no single read decides a final verdict -----------------------------------------
#
# Each test lies exactly once, on the Nth fetch that covers the offset deciding the
# verdict, for each N that such a fetch exists -- measured, and asserted, so a change to
# the read pattern fails here rather than leaving a case that quietly tests nothing.
# Which fetch is the dangerous one depends on window sizes and pass structure, so the
# tests do not guess it: whichever read the lie lands on, the verdict must be the truth.
# On a walk from 0, N = 2 is the read that confirms the verdict, and a lie there used to
# overturn a correct one.

def _terminator(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        last = t.getmembers()[-1]
    return last.offset_data + (-(-last.size // 512)) * 512


def _walk_lying_once(tmp_path, data, bad, nth, fill=JUNK, start=0, span=1024):
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, bad, fill=fill, span=span, nth=nth)
    seen = []
    result = walk(ConcatFile(archive, reader), start,
                  lambda members, offset: seen.extend(members))
    return result, seen, reader


@pytest.mark.parametrize("nth", [1, 2])
def test_one_bad_read_cannot_turn_a_complete_archive_corrupt(tmp_path, nth):
    data = build_tar(MEMBERS)
    result, seen, reader = _walk_lying_once(tmp_path, data, _terminator(data), nth)
    assert result.state == "complete", f"a lie on covering read #{nth} decided the verdict"
    assert len(seen) == len(MEMBERS)
    assert reader.lied == 1, "the test did not actually inject a bad read"


@pytest.mark.parametrize("nth", [1, 2])
def test_one_bad_read_cannot_pass_a_damaged_archive_as_complete(tmp_path, nth):
    """The false `complete` is the silent failure: members go missing and `index`
    refuses to walk the archive again. Here the archive really is damaged -- junk after
    its terminator -- and one read of zeros there must not hide it."""
    data = bytearray(build_tar(MEMBERS))
    junk_at = _terminator(bytes(data)) + 1536                # inside the record padding
    assert junk_at + 8 <= len(data)
    data[junk_at:junk_at + 8] = b"\xff" * 8
    result, _, reader = _walk_lying_once(tmp_path, bytes(data), junk_at, nth, fill=b"\x00")
    assert result.state == "corrupt", f"a lie on covering read #{nth} decided the verdict"
    assert reader.lied == 1, "the test did not actually inject a bad read"


def test_a_bad_read_of_the_very_first_block_is_read_again(tmp_path):
    """Segment 0 starts at 0 on a fresh reader, so nothing has validated its first block.
    A verdict reached at a walk's own start offset used to be believed outright. (Only
    one read covers offset 0 in a walk that reads it right, so there is one case.)"""
    data = build_tar(MEMBERS)
    result, seen, reader = _walk_lying_once(tmp_path, data, 0, 1)
    assert reader.lied == 1
    assert result.state == "complete"
    assert len(seen) == len(MEMBERS)


@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_a_walk_resumed_at_the_terminator_confirms_complete(tmp_path, nth):
    """A stop landing between the last batch and `finish_segment` leaves the cursor on
    the terminator, so a resumed walk's own start decides the verdict. The read that
    matters here is #2, not #1: tarfile meets the lie, re-seeks to `offset - 1`, and that
    cache miss fetches what `_classify_end` judges. Lie and truth land on the same
    offset, so this also pins that a verdict is compared by state, not offset alone."""
    data = build_tar(MEMBERS)
    t = _terminator(data)
    result, _, reader = _walk_lying_once(tmp_path, data, t, nth, start=t)
    assert result.state == "complete", f"a lie on covering read #{nth} decided the verdict"
    assert reader.lied == 1


@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_a_walk_resumed_at_the_terminator_still_finds_real_damage(tmp_path, nth):
    """The same resume on an archive that really is damaged after its terminator: a
    read of zeros over the damage must not let it pass as complete."""
    data = bytearray(build_tar(MEMBERS))
    t = _terminator(bytes(data))
    data[t + 1536:t + 1544] = b"\xff" * 8
    result, _, reader = _walk_lying_once(tmp_path, bytes(data), t + 1536, nth, fill=b"\x00",
                                         start=t)
    assert result.state == "corrupt", f"a lie on covering read #{nth} decided the verdict"
    assert reader.lied == 1


def test_a_walk_resumed_on_a_long_name_member_re_reads_its_real_header(tmp_path):
    """A cursor can land on a GNU long-name member. tarfile rejects the whole thing when
    the real header after the name reads badly, at the walk's own start offset."""
    data, real_header = _long_name_archive()
    first_span = 512 + (-(-100 // 512) * 512)                # where the long name starts
    result, seen, reader = _walk_lying_once(tmp_path, bytes(data), real_header, 1,
                                            fill=b"\xff", start=first_span, span=8)
    assert reader.lied == 1
    assert result.state == "complete"
    assert [m.name for m in seen] == ["deep.bin"]


def test_a_rescue_that_ends_in_a_crossing_keeps_its_note(tmp_path):
    """Seventeen of v01p0's eighteen segments end `crossed`, not `complete`, so this is
    where a rescue's note usually has to survive -- it is the only record of it."""
    data = build_tar(MEMBERS)
    offs = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, offs[10])
    result = walk(ConcatFile(archive, reader), 0, lambda members, offset: None, stop_at=offs[30])
    assert reader.lied == 1
    assert result.state == "crossed"
    assert result.end_offset == offs[30]
    assert f"re-read at {offs[10]}" in result.detail


def test_the_note_names_each_re_read_offset_once(tmp_path):
    """Re-reading one offset until two reads agree can take several passes there; the
    note is for a person, and should name the offset once."""
    data = build_tar(MEMBERS)
    t = _terminator(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnEach(tmp_path, archive, t, nths={1, 3})
    result = walk(ConcatFile(archive, reader), 0, lambda members, offset: None)
    assert reader.lied == 2
    assert result.state == "complete"
    assert result.detail.count(str(t)) == 1, result.detail


def test_a_walk_starting_on_a_damaged_long_name_member_says_what_is_wrong(tmp_path):
    """tarfile rejects the member for its damaged *second* header and says so, but
    `_classify_end` used to discard that and describe the first -- a real, checksum-valid
    `././@LongLink` header -- as "not a header". An operator reads this detail to decide
    whether to re-run six hours of walking; it should name the actual fault."""
    data, real_header = _long_name_archive()
    data[real_header:real_header + 8] = b"\xff" * 8
    first_span = 512 + (-(-100 // 512) * 512)
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), first_span,
                  lambda members, offset: None)
    assert result.state == "corrupt"
    assert result.end_offset == first_span
    assert "bad checksum" in result.detail
    assert "not a header" not in result.detail


# ---- pax is refused, loudly ----------------------------------------------------------
#
# In a pax archive a member's name, link target, size and times can live in an extended
# header's data blocks, which carry no checksum -- the guarantees this walker rests on do
# not hold there. dbaudit indexes GNU and ustar tars only, and says so rather than
# producing a verdict it cannot stand behind.

def _pax_tar(members, pax_global=None):
    buf = io.BytesIO()
    kw = {"format": tarfile.PAX_FORMAT}
    if pax_global:
        kw["pax_headers"] = pax_global
    with tarfile.open(fileobj=buf, mode="w", **kw) as tf:
        for name, payload, extra in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            if extra:
                info.pax_headers = extra
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _walk_all(tmp_path, data):
    archive = write_parts(tmp_path, data, part_size=len(data))
    calls = []
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), 0,
                  lambda members, offset: calls.append(([m.name for m in members], offset)))
    return result, calls


def test_an_archive_with_pax_headers_on_every_member_is_refused_at_once(tmp_path):
    """What GNU tar --format=posix writes: an `x` header before every entry."""
    data = _pax_tar([(f"f{i}.bin", b"x" * 100, {"mtime": "1700000000.5"}) for i in range(5)])
    with pytest.raises(UnsupportedArchive, match="pax"):
        _walk_all(tmp_path, data)


def test_a_pax_header_deep_in_the_archive_stops_the_walk_where_it_is(tmp_path):
    """tarfile's own pax writer adds an `x` header only where it needs one -- here, for
    a long name -- so detection can come deep into an archive. What was walked before it
    is committed, and the cursor stops before the pax member, never past it."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    data = _pax_tar([("a.bin", b"a" * 700, None), ("b.bin", b"b" * 700, None),
                     (long_name, b"y" * 700, None), ("z.bin", b"z" * 700, None)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as reference:
        pax_member = [m for m in reference if m.pax_headers][0]
    archive = write_parts(tmp_path, data, part_size=len(data))
    calls = []
    with pytest.raises(UnsupportedArchive, match=str(pax_member.offset)):
        walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), 0,
             lambda members, offset: calls.append(([m.name for m in members], offset)))
    assert [name for names, _ in calls for name in names] == ["a.bin", "b.bin"]
    assert calls[-1][1] == pax_member.offset


def test_a_global_pax_header_is_refused_with_the_cursor_before_it(tmp_path):
    """A `g` header applies to every member after it, and tarfile reports those members
    at their own offsets, past it. A cursor saved there resumes with no global header in
    force -- and a later `index` walks the rest as plain ustar, to `complete`. The cursor
    goes where the header sequence began: 0, not 1024."""
    data = _pax_tar([("a.bin", b"a" * 100, None), ("b.bin", b"b" * 100, None)],
                    pax_global={"comment": "archived by some other tool"})
    archive = write_parts(tmp_path, data, part_size=len(data))
    calls = []
    with pytest.raises(UnsupportedArchive, match="is a pax-format archive"):
        walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), 0,
             lambda members, offset: calls.append((len(members), offset)))
    assert calls[-1] == (0, 0)


def test_one_bad_read_carrying_a_pax_header_is_read_again_not_believed(tmp_path):
    """`unsupported` is for good, so it is not decided on one read. A bad read can be
    valid pax from elsewhere -- a stored tarball, say -- sitting where a GNU member's
    header belongs. Read again, the GNU header is there, and the walk carries on."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    at = offsets[20]
    pax = _pax_tar([("x/" + "p" * 120 + ".bin", b"q" * 700, None)])
    assert pax[156:157] == b"x"
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=pax[:1536], span=1536)
    rows = {}

    def commit(batch, offset):
        rows.update((m.hdr_offset, m.name) for m in batch)

    result = walk(ConcatFile(archive, reader), 0, commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert [rows[o] for o in offsets] == [n.rsplit("/", 1)[-1] for n, _ in MEMBERS]
    assert f"re-read at {at}" in result.detail


def test_one_bad_read_carrying_a_global_pax_header_leaves_nothing_behind(tmp_path):
    """A `g` header changes the walk's own state: tarfile applies it to every member
    after it. Read again and found not to be there, it must stop applying -- otherwise
    every later member would look like pax and be read again, one by one."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    at = offsets[20]
    glob = _pax_tar([("a.bin", b"a", None)], pax_global={"comment": "elsewhere"})
    assert glob[156:157] == b"g"
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=glob[:1024] + data[at:at + 512],
                       span=1536)
    rows = {}

    def commit(batch, offset):
        rows.update((m.hdr_offset, m.name) for m in batch)

    result = walk(ConcatFile(archive, reader), 0, commit)
    clean = LocalRangeReader(tmp_path, archive)
    walk(ConcatFile(archive, clean), 0, lambda members, offset: None)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert [rows[o] for o in offsets] == [n.rsplit("/", 1)[-1] for n, _ in MEMBERS]
    assert f"re-read at {at}: " in result.detail
    # One read again, and the window that refills after it -- not one per later member.
    assert reader.requests <= clean.requests + 2, (reader.requests, clean.requests)


@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.USTAR_FORMAT])
def test_gnu_and_ustar_archives_are_not_mistaken_for_pax(tmp_path, fmt):
    """GNU long names and link targets also live outside the header, but they are GNU
    format, and ustar's 155-byte name prefix is inside the checksummed header -- neither
    may trip the pax check."""
    buf = io.BytesIO()
    long_path = "sub/" + "p" * 120 + "/b.bin"   # GNU: an `L` header; ustar: a prefix
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for name in ("a.bin", long_path):
            info = tarfile.TarInfo(name)
            info.size = 10
            tf.addfile(info, io.BytesIO(b"q" * 10))
    result, calls = _walk_all(tmp_path, buf.getvalue())
    assert result.state == "complete"
    assert [name for names, _ in calls for name in names] == ["a.bin", "b.bin"]


def test_giving_up_on_a_member_after_a_g_header_keeps_the_cursor_before_the_g(tmp_path):
    """Review 5's K8. The member after a `g` header is suspect (pax); if its re-reads never
    agree, the walk raises UnsettledRead -- leaving the cursor where the member's header
    sequence begins, at the `g`, or a resume walks on with no global header in force.
    Here every read after the first has a valid header there, named differently each
    time: no two reads agree on what the member is."""
    data = _pax_tar([(f"g/f{i}", bytes([i]) * 600, None) for i in range(3)],
                    pax_global={"comment": "0123abcd"})
    assert data[156:157] == b"g" and data[1024:1027] == b"g/f"
    archive = write_parts(tmp_path, data, part_size=len(data))

    class EveryRereadDiffers(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            true = super().read_range(part_idx, offset, length)
            if self.requests == 1:
                return true                   # the first read: the g and its member
            other = _with_field(data, 1024, 0, 100, b"g/other%d" % self.requests)
            return other[offset:offset + length]

    reader = EveryRereadDiffers(tmp_path, archive)
    calls = []
    with pytest.raises(UnsettledRead):
        walk(ConcatFile(archive, reader, window_min=4_096, window_max=4_096), 0,
             lambda members, offset: calls.append((len(members), offset)))
    assert calls[-1] == (0, 0), calls


# ---- a verdict is read again from the fills that led to it -----------------------------
#
# Review 5's P-a. A bad read that is valid tar from elsewhere -- a real header, just not the
# one at this offset -- passes its checksum and moves the chain by its own size. Where it
# lands is read truly, so reading the landing again only repeats the verdict. What decided
# it is the fill that served the wrong header, so that is what is read again: every final
# verdict, and every crossing, is re-walked from the start of the last fills that served a
# header. Where the two readings part, a third read of that header settles which was
# right. (Review 6 replaced dropping the wrong reading's rows with never committing them:
# the members of those fills are held back until two readings agree.)


class _Rows:
    """What the store holds after a walk: rows by header offset (a commit replaces a row
    at the same offset), the cursor, and every commit in order."""

    def __init__(self):
        self.rows, self.cursor, self.calls = {}, None, []

    def commit(self, members, next_offset):
        self.calls.append(("commit", [m.hdr_offset for m in members], next_offset))
        self.rows.update((m.hdr_offset, m) for m in members)
        self.cursor = next_offset

    def names(self):
        return [self.rows[k].name for k in sorted(self.rows)]

    def consistent(self):
        return all(k < self.cursor for k in self.rows)


def _names_of(members):
    return [name.rsplit("/", 1)[-1] for name, *_ in members]


def _header_with_size(data, header, size):
    """The header at ``header`` claiming ``size`` bytes, checksum redone: a valid header
    that sends the chain wherever a test needs it -- as a real one from elsewhere would."""
    return _with_field(data, header, 124, 12, b"%011o" % size)[header:header + 512]


def test_a_header_copied_from_elsewhere_cannot_condemn_a_sound_archive(tmp_path):
    """The big member's header served where member 30's belongs sends the chain 6,656
    bytes on, into member data. Reading that landing again finds the same data, so it
    used to stand as `corrupt`. Read again from the fill that served the wrong header,
    member 30 is there, and the walk carries on to the end."""
    members = [("run/big.bin", b"b" * 6000)] + MEMBERS
    data = build_tar(members)
    offsets = _header_offsets(data)
    at = offsets[31]
    landing = at + 512 + 6144
    assert landing not in offsets and any(data[landing:landing + 512])
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=data[0:512], span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert result.members == len(members)
    # Where the readings parted, and nothing else: the second's own verdict, reached
    # after it left the first reading's path, is not a place two reads disagreed.
    assert result.detail == f"re-read at {at}: a read there was contradicted by the next"
    assert rows.consistent()


def _gnu_holding_a_pax_tarball():
    """GNU members around a tarball Python wrote -- pax, an `x` header before each member
    -- and where the second of its members' header sequences begins."""
    inner = _pax_tar([(f"inner/p{i}.dat", bytes([i]) * 2000, {"mtime": "1700000000.5"})
                      for i in range(4)])
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        second = stored + t.getmembers()[1].offset
    return members, data, second


def test_a_header_copied_from_elsewhere_cannot_condemn_an_archive_as_pax(tmp_path):
    """The wrong header's size lands the chain on a real pax header inside a stored
    tarball. Every read of it says pax, so reading it twice confirmed `unsupported` -- for
    good -- on the strength of one bad read of the header before it."""
    members, data, pax_at = _gnu_holding_a_pax_tarball()
    offsets = _header_offsets(data)
    at = offsets[1]
    wrong = _header_with_size(data, at, pax_at - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert f"re-read at {at}" in result.detail


def test_a_header_copied_from_elsewhere_cannot_move_where_a_chain_crosses(tmp_path):
    """A crossing is what the next segment's start is judged by: a wrong one resets that
    segment onto bytes that are not a header. The wrong header here skips three members
    and crosses on a real header -- the wrong one."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    boundary = offsets[30] - 16
    at = offsets[28]
    wrong = _header_with_size(data, at, offsets[33] - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit,
                  stop_at=boundary)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "crossed"
    assert result.end_offset == offsets[30]
    assert sorted(rows.rows) == offsets[:30]
    assert rows.cursor == offsets[30]


def test_a_copied_header_that_rejoins_the_chain_near_its_end_is_read_again(tmp_path):
    """Member 5's header served where member 20's belongs: the same padded size, so the
    chain rejoins and the walk ends `complete`, one row wrong. It lies within the fills
    re-read to confirm the verdict, so that reading finds it."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, offsets[20], fill=data[offsets[5]:offsets[5] + 512],
                       span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert rows.names() == _names_of(MEMBERS)
    assert result.members == len(MEMBERS)
    assert f"re-read at {offsets[20]}" in result.detail


def test_a_verdict_is_committed_once_two_readings_agree(tmp_path):
    """Nothing read only once is committed: the first pass holds back what its last
    fetches served -- here, the whole small archive -- and commits it only once a second
    reading has agreed with it, once."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows = _Rows()

    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), 0, rows.commit)

    t = _terminator(data)
    assert result.state == "complete"
    assert rows.calls == [("commit", [], 0), ("commit", [], 0), ("commit", offsets, t)]
    assert result.members == len(MEMBERS)


class _LiesOnceArmed(LocalRangeReader):
    """Truthful until ``armed``; then the first fetch covering ``bad`` carries ``fill``
    there -- so a lie lands on the first read of a place after some event, whatever the
    read pattern before it."""

    def __init__(self, directory, archive, bad, fill):
        super().__init__(directory, archive)
        self.bad, self.fill, self.armed, self.lied, self.covering = bad, fill, False, 0, 0

    def read_range(self, part_idx, offset, length):
        data = super().read_range(part_idx, offset, length)
        start = self.archive.parts[part_idx].offset + offset
        self.covering += start <= self.bad < start + length
        if not self.armed or self.lied or not start <= self.bad < start + length:
            return data
        self.lied += 1
        cut = self.bad - start
        return data[:cut] + self.fill[:length - cut] + data[cut + len(self.fill):]


def test_a_wrong_header_read_by_the_confirming_pass_is_outvoted(tmp_path):
    """The archive really is damaged at member 30's header. The second reading of it comes
    back as a valid header from elsewhere -- one whose size rejoins the chain, so, believed,
    that pass walks on to `complete` and its own confirmation, from fills near the end,
    agrees. A third read of the header settles it: two reads say no header there."""
    damaged = bytearray(build_tar(MEMBERS))
    offsets = _header_offsets(bytes(damaged))
    at = offsets[30]
    donor = bytes(damaged[offsets[26]:offsets[26] + 512])       # the same padded size
    damaged[at:at + 1024] = JUNK * 4
    archive = write_parts(tmp_path, bytes(damaged), part_size=len(damaged))
    reader = _LiesOnceArmed(tmp_path, archive, at, donor)
    rows = _Rows()

    def commit(members, next_offset):
        rows.commit(members, next_offset)
        reader.armed = True                                   # the first pass has ended

    # Two-block windows: the first pass's probe at the damage comes from the window that
    # read the header there, so the first fetch covering it after that pass is the second
    # reading's -- and the last fills before the end hold only the last few members.
    result = walk(ConcatFile(archive, reader, window_min=2048, window_max=2048), 0,
                  commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "corrupt", result.detail
    assert result.end_offset == at
    assert sorted(rows.rows) == offsets[:30]
    # The first reading's vote counts: one more read, agreeing with it, settles it.
    assert reader.covering == 3, "one read per pass, and one to break the tie"


@pytest.mark.parametrize("stop_at_check", range(1, 58))
def test_a_stop_anywhere_loses_nothing_and_repeats_nothing(tmp_path, stop_at_check):
    """A stop at any check -- in the first pass, or while its verdict is read again --
    commits only members read twice or far enough behind; the walk resumed from where it
    says ends with every member, each committed once. One-block windows, so members
    leave the held-back window mid-walk too."""
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows = _Rows()
    checks = {"n": 0}

    def should_stop():
        checks["n"] += 1
        return checks["n"] == stop_at_check

    def walk_from(start, stop):
        return walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive),
                               window_min=1024, window_max=1024), start, rows.commit,
                    should_stop=stop)

    first = walk_from(0, should_stop)
    if first.state == "stopped":
        assert rows.cursor == first.end_offset
        assert all(k < rows.cursor for k in rows.rows)
        first = walk_from(first.end_offset, None)
    assert first.state == "complete"
    assert rows.names() == _names_of(MEMBERS)
    committed = [o for _, offsets, _ in rows.calls for o in offsets]
    assert len(committed) == len(set(committed)) == len(MEMBERS), "a member committed twice"


class _LiesAfter(_AlwaysLies):
    """Truthful for the first ``first`` fetches covering ``bad``, then wrong -- and wrong
    differently -- every time after."""

    def __init__(self, directory, archive, bad, first, **kw):
        super().__init__(directory, archive, bad, **kw)
        self.first = first

    def _lies_now(self):
        return self.covering > self.first


def test_giving_up_during_a_second_reading_leaves_it_to_be_read_again(tmp_path):
    """The second reading gives up on a long name that never reads the same way twice.
    Nothing either reading saw was committed, and the cursor is where they begin: the
    retry that `UnsettledRead` invites reads them again, and an honest one finishes."""
    data, _ = _long_name_archive()
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    reader = _LiesAfter(tmp_path, archive, 1536, first=1, span=512)
    rows = _Rows()

    with pytest.raises(UnsettledRead, match="1024"):
        walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied >= 2, "the test did not reach the second reading"
    assert (rows.names(), rows.cursor) == ([], 0)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), rows.cursor,
                  rows.commit)
    assert result.state == "complete"
    assert rows.names() == ["first.bin", "deep.bin"]


def test_reads_that_agree_no_member_is_there_are_an_answer(tmp_path):
    """A bad fetch holding a stretch of a stored pax tarball, served where z.bin's header
    belongs: its first member sends the chain into z.bin's data, where the same fetch has
    that tarball's next pax header. Read again, there is no member there at all -- and
    two reads agreeing on that is an answer, not a reason to give up. Giving up left the
    cursor there, on the wrong path, and the resume condemned the archive."""
    inner = _pax_tar([(f"inner/p{i}.dat", bytes([i]) * 2000, {"mtime": "1700000000.5"})
                      for i in range(4)])
    members = [("run/a.bin", b"a" * 3000), ("run/results.tar", inner),
               ("run/z.bin", b"z" * 30000), ("run/last.bin", b"l" * 100)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    at = offsets[2]
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        real = t.getmembers()[1].offset_data - 512          # member 1's own header
    chunk = inner[real:real + 512 + 2048 + 1536]            # ... and member 2's `x` on
    assert chunk[2560 + 156:2560 + 157] == b"x"
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=chunk, span=len(chunk))
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    landing = at + 512 + 2048
    assert f"re-read at {landing}" in result.detail, "the pax read there was a rescue"


def test_a_landing_on_a_real_header_in_its_own_fetch_is_read_back_past(tmp_path):
    """The wrong header's size lands the chain on a real header -- the last member of a
    tarball stored in the archive -- in a fetch of its own. The chain walks that member and
    meets the stored tarball's terminator mid-archive, and reading the landing's fetch
    again finds the same. The fetch before it, which served the wrong header, is the one
    that has to be read again: why a verdict is read again from more than one fetch."""
    inner = build_tar([(f"inner/g{i}.dat", bytes([i]) * 600) for i in range(3)])
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    landing = stored + _header_offsets(inner)[2]        # the stored tarball's last member
    at = offsets[1]
    assert landing - at > 4096                            # beyond the fetch that lies
    wrong = _header_with_size(data, at, landing - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader, window_min=4096, window_max=4096), 0,
                  rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)


def test_a_bad_fetch_is_read_again_from_the_first_header_it_served(tmp_path):
    """A bad fetch holding a stretch of the archive from elsewhere serves two wrong headers
    in a row before the chain leaves it, landing in member data. The last of them is not
    where the walk went wrong: it is read again from the first, and the rows from there
    on are replaced."""
    members = [("run/a.bin", b"a" * 100), ("run/b.bin", b"b" * 3000),
               ("run/c.bin", b"c" * 100), ("run/d.bin", b"d" * 20000),
               ("run/e.bin", b"e" * 100)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    at = offsets[2]
    stretch = data[offsets[0]:offsets[1] + 512]     # a.bin's header and data, b.bin's header
    landing = at + 1024 + 512 + 3072
    assert landing not in offsets and data[landing:landing + 1] == b"d"
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=stretch, span=len(stretch))
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert rows.consistent()


def test_damage_read_as_a_valid_header_is_found_by_the_second_reading(tmp_path):
    """The archive really is damaged at member 30's header, and the first pass read it as a
    valid header from elsewhere that rejoins the chain, walking on to `complete`. The
    second reading finds the damage -- and the rows the first pass recorded past it go, or
    the index would hold members past its own verdict."""
    damaged = bytearray(build_tar(MEMBERS))
    offsets = _header_offsets(bytes(damaged))
    at = offsets[30]
    donor = bytes(damaged[offsets[26]:offsets[26] + 512])       # the same padded size
    damaged[at:at + 1024] = JUNK * 4
    archive = write_parts(tmp_path, bytes(damaged), part_size=len(damaged))
    reader = _LiesOnce(tmp_path, archive, at, fill=donor, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert (result.state, result.end_offset) == ("corrupt", at), result.detail
    assert sorted(rows.rows) == offsets[:30]
    assert rows.consistent()


class _GarblesEachOnce(LocalRangeReader):
    """Garbles 512 bytes at each of ``offsets`` in turn, at most one per fetch: successive
    transients, each met only after the walk has survived the one before."""

    def __init__(self, directory, archive, offsets):
        super().__init__(directory, archive)
        self.pending, self.lied = sorted(offsets), 0

    def read_range(self, part_idx, offset, length):
        data = bytearray(super().read_range(part_idx, offset, length))
        start = self.archive.parts[part_idx].offset + offset
        if self.pending and start <= self.pending[0] and self.pending[0] + 512 <= start + length:
            bad = self.pending.pop(0)
            self.lied += 1
            data[bad - start:bad - start + 512] = b"garbled/" * 64
        return bytes(data)


def test_rescues_on_the_way_do_not_use_up_the_verdicts_own_re_reads(tmp_path):
    """A long walk meets independent transients far apart -- here a garbled long name for
    each of REREAD_LIMIT + 1 members. Each is settled where it is met. None of them is
    the verdict's, and the verdict's own re-reads are counted from where it is reached:
    a six-hour walk must not end in UnsettledRead for having survived four bad reads."""
    long_named = [(f"run/{'d' * 120}/f{i}.bin", bytes([i]) * 700)
                  for i in range(REREAD_LIMIT + 1)]
    members = long_named + list(MEMBERS[:5])
    data = build_tar(members, format=tarfile.GNU_FORMAT)
    blocks = [o + 512 for o in _header_offsets(data)[:REREAD_LIMIT + 1]]
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _GarblesEachOnce(tmp_path, archive, blocks)
    rows = _Rows()

    # One-block windows keep each garble in a fetch the walk parses: a larger fetch made
    # to re-read one member would carry the next garble, and be dropped unread.
    result = walk(ConcatFile(archive, reader, window_min=1024, window_max=1024), 0,
                  rows.commit)

    assert reader.lied == REREAD_LIMIT + 1
    assert result.state == "complete", result.detail
    assert all(f"{o - 512}" in result.detail for o in blocks), "each one was a rescue"
    assert [f"{rows.rows[k].dir}/{rows.rows[k].name}" for k in sorted(rows.rows)] == [
        name for name, _ in members]


def test_confirming_a_verdict_reads_its_last_fetches_again_not_the_walk(tmp_path):
    """One-block windows make every header a fetch of its own, so what a confirmation
    costs can be counted: the fetches that served the last CONFIRM_FILLS headers, the one
    that finds the terminator, and the two that check the zeros after it -- not a second
    walk of fifty."""
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = LocalRangeReader(tmp_path, archive)
    at_commit = []

    def commit(members, next_offset):
        at_commit.append(reader.requests)

    result = walk(ConcatFile(archive, reader, window_min=1024, window_max=1024), 0, commit)

    assert result.state == "complete"
    first = at_commit[0] + 2            # the first pass's own two checks of the zeros
    assert reader.requests - first == CONFIRM_FILLS + 3


def test_a_copied_header_spanning_two_members_is_read_again_and_counted(tmp_path):
    """A wrong header whose size covers members 20 and 21 together: the chain rejoins at
    22 with 21 never read, and the walk ends `complete`, one member short. The second
    reading records both -- and the result's count is what the second reading found."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    at = offsets[20]
    wrong = _header_with_size(data, at, offsets[22] - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete"
    assert rows.names() == _names_of(MEMBERS)
    assert result.members == len(MEMBERS)


def test_a_wrong_header_read_by_the_second_reading_of_a_sound_archive_is_outvoted(tmp_path):
    """The first pass reads member 45 right; the second reading gets a valid header there
    claiming another size. A third read agrees with the first, and the walk must go on
    from where *that* reading says, not the wrong one -- or it leaves the chain."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    at = offsets[45]
    t = _terminator(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnceArmed(tmp_path, archive, at, _header_with_size(data, at, 5000))
    rows = _Rows()

    def commit(members, next_offset):
        rows.commit(members, next_offset)
        reader.armed = True                                   # the first pass has ended

    result = walk(ConcatFile(archive, reader), 0, commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(MEMBERS)
    assert f"re-read at {at}" in result.detail


def test_a_global_header_only_the_second_readings_bad_read_saw_stops_applying(tmp_path):
    """The second reading of member 48 comes back as a `g` header and a copy of it. The `g`
    makes the member suspect, so it is read again at once, and that read agrees with the
    first reading; the `g` must stop applying there, or every member after it looks like
    pax and is read again, one request each. (Two-block windows: the second reading covers
    only the last two members.)"""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    at = offsets[48]
    t = _terminator(data)
    glob = _pax_tar([("a.bin", b"a", None)], pax_global={"comment": "elsewhere"})
    assert glob[156:157] == b"g"
    archive = write_parts(tmp_path, data, part_size=len(data))

    def run(lie):
        reader = _LiesOnceArmed(tmp_path, archive, at, glob[:1024] + data[at:at + 512])
        rows = _Rows()

        def commit(members, next_offset):
            rows.commit(members, next_offset)
            reader.armed = lie                                # the first pass has ended

        result = walk(ConcatFile(archive, reader, window_min=2048, window_max=2048), 0,
                      commit)
        return result, rows, reader

    clean, _, honest = run(lie=False)
    result, rows, reader = run(lie=True)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(MEMBERS)
    # The tie-break's one fresh read, and nothing per member after it.
    assert reader.requests == honest.requests + 1, (reader.requests, honest.requests)


def _cut_after_a_long_name(damaged=False):
    """MEMBERS[:5], then a long-named member whose real header is cut short by the end of
    the archive -- or, if ``damaged``, present but corrupt. Returns the data and where the
    long-named member's header sequence begins."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    data = build_tar(list(MEMBERS[:5]) + [(long_name, b"x" * 700)], format=tarfile.GNU_FORMAT)
    start = _header_offsets(data)[-1]
    if damaged:
        data = bytearray(data)
        data[start + 1024:start + 1032] = b"\xff" * 8
        return bytes(data), start
    return data[:start + 1024 + 100], start


def test_a_tie_break_that_finds_the_archive_cut_short_says_truncated(tmp_path):
    """The archive ends inside the real header of a long-named member: `truncated`. The
    second reading gets a valid header there instead; two reads settle that none begins
    there -- and say so as the first pass did, `truncated`, so the readings agree at once
    instead of disagreeing about wording for two more passes."""
    data, start = _cut_after_a_long_name()
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnceArmed(tmp_path, archive, start, data[0:512])
    rows = _Rows()

    def commit(members, next_offset):
        rows.commit(members, next_offset)
        reader.armed = True                                   # the first pass has ended

    result = walk(ConcatFile(archive, reader), 0, commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "truncated"
    assert reader.covering == 3, "one read per pass, and one to break the tie"


def test_reads_that_agree_no_member_is_there_say_what_tarfile_found(tmp_path):
    """A pax header served where a long-named member's own sequence begins -- whose real
    header is damaged. Read again, tarfile rejects the sequence for that damage, twice; the
    verdict names it, as a walk arriving there would, rather than "not a header"."""
    data, start = _cut_after_a_long_name(damaged=True)
    pax = _pax_tar([("x/" + "p" * 120 + ".bin", b"q" * 700, None)])
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, start, fill=pax[:1536], span=1536)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert (result.state, result.end_offset) == ("corrupt", start)
    assert "bad checksum" in result.detail, result.detail


def test_a_second_reading_that_moves_a_sparse_member_is_outvoted(tmp_path):
    """A GNU sparse member records the file's expanded size, so where its stored bytes end
    -- where the chain goes next -- is not among the fields recorded. A second reading that
    agrees on every recorded field but stores a different amount would move the chain; it
    is compared too, outvoted by a third read, and costs no extra pass."""
    data = _gnu_sparse_archive()
    t = 1536
    wrong = _with_field(data, 0, 124, 12, b"%011o" % 1024)[0:512]
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnceArmed(tmp_path, archive, 0, wrong)
    rows = _Rows()

    def commit(members, next_offset):
        rows.commit(members, next_offset)
        reader.armed = True                                   # the first pass has ended

    result = walk(ConcatFile(archive, reader), 0, commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert (rows.rows[0].size, rows.rows[0].data_offset) == (4096, 1024)
    assert reader.covering == 3, "one read per pass, and one to break the tie"


# ---- review 6: the last fetches' members are held back; a proven-bad fetch's are re-read

class _LiesThenFails(_LiesOnce):
    """`_LiesOnce`, and then, once ``armed``, a fetch that fails for good -- as five 429s
    in a row would, or a response whose Content-Range is not the range asked for."""

    armed = False

    def read_range(self, part_idx, offset, length):
        if self.armed:
            self.armed = False
            raise ReaderError("simulated: the fetch after the first pass failed for good")
        return super().read_range(part_idx, offset, length)


def test_a_walk_interrupted_before_its_verdict_is_read_again_reads_it_again_on_resume(
        tmp_path):
    """Review 6's #1. The first pass is derailed by a copied header and lands in member
    data; the read that would confirm it fails. The resume must start where the members
    read only once begin -- so it reads the copied header again -- not where the first
    pass landed, where re-reading only repeats `corrupt`."""
    members = [("run/big.bin", b"b" * 6000)] + MEMBERS
    data = build_tar(members)
    at = _header_offsets(data)[31]
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesThenFails(tmp_path, archive, at, fill=data[0:512], span=512)
    rows = _Rows()

    def commit(batch, next_offset):
        rows.commit(batch, next_offset)
        reader.armed = True                     # the first pass has ended

    with pytest.raises(ReaderError):
        walk(ConcatFile(archive, reader), 0, commit)
    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert rows.cursor <= at and all(k < rows.cursor for k in rows.rows)

    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive)), rows.cursor,
                  rows.commit)
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)


def test_re_reading_one_member_does_not_push_the_bad_fetch_out_of_what_is_read_again(
        tmp_path):
    """Review 6's #4. The wrong header's size lands in a stored tarball whose members' long
    names its writer did not copy into their headers, so each of the three it walks is
    read twice -- fetches that re-read one place, not the walk moving on. Counted as the
    walk's own, they pushed the fetch that served the wrong header out of those read
    again."""
    inner = build_tar([("ginner/" + "d" * 110 + f"/g{i:02d}.dat", bytes([i]) * 600)
                       for i in range(4)])
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        sequences = [m.offset for m in t]
        reals = [m.offset_data - 512 for m in t]
    for real in reals:                              # a writer that does not copy the name
        inner = _with_field(inner, real, 0, 100, b"elsewhere")
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    at = offsets[1]
    wrong = _header_with_size(data, at, stored + sequences[1] - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader, window_min=4096, window_max=4096), 0,
                  rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)


def test_a_fetch_caught_lying_has_the_headers_it_served_before_read_again(tmp_path):
    """Review 6's #5. A bad fetch holding a stretch of a stored pax tarball serves, where
    b.bin belongs, that tarball's member of the same padded size -- the chain rejoins,
    nothing looks wrong -- and then the next member's pax header, which a second read
    shows is not there. That proves the fetch bad, so the header it served before is read
    again as well, not kept on the one read it had: it is far from the end, and no
    verdict's second reading would reach it."""
    inner = _pax_tar([(f"inner/p{i}.dat", bytes([i]) * 744, {"mtime": "1700000000.5"})
                      for i in range(4)])
    members = ([("run/a.bin", b"a" * 1000), ("run/b.bin", b"b" * 1000),
                ("run/c.bin", b"c" * 1000)]
               + [(f"run/m{i:02d}.bin", bytes([i]) * 1000) for i in range(30)]
               + [("run/results.tar", inner), ("run/z.bin", b"z" * 1000)])
    data = build_tar(members)
    at = _header_offsets(data)[1]                   # b.bin pads to 1024, as 744 does
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        real = t.getmembers()[1].offset_data - 512  # inner member 1's own header
    stretch = inner[real:real + 512 + 1024 + 1536]  # it, its data, and member 2's `x` on
    assert stretch[1536 + 156:1536 + 157] == b"x"
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=stretch, span=len(stretch))
    rows = _Rows()

    result = walk(ConcatFile(archive, reader, window_min=8192, window_max=8192), 0,
                  rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert f"re-read at {at + 1536}" in result.detail


def test_a_second_reading_in_a_dense_run_starts_at_the_window_it_had_grown_to(tmp_path):
    """Review 6's #6. In a run of small members the window has grown to its cap, each
    fetch serving hundreds of headers. The second reading of the verdict starts at that
    size too: from the floor it would take several fetches just to grow back."""
    members = [(f"run/s{i:05d}.bin", b"s" * 100) for i in range(2000)]
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = LocalRangeReader(tmp_path, archive)
    at_commit = []

    def commit(batch, next_offset):
        at_commit.append(reader.requests)

    result = walk(ConcatFile(archive, reader, window_min=65536, window_max=262144), 0,
                  commit)

    assert result.state == "complete"
    # The fetches that held the last CONFIRM_FILLS fetches' headers, read again at the
    # size they were read at; the terminator's check comes out of the last of them.
    assert reader.requests - at_commit[0] == CONFIRM_FILLS


def test_two_readings_that_run_out_in_different_places_do_not_agree(tmp_path):
    """Review 6's #8. Two bad reads, each a header claiming more than the archive has left:
    one in the first pass at member 45, and one met only by the second reading -- which
    rightly reads 45 -- at member 47. Both run off the end and say `truncated` at its
    size, but they stopped reading headers in different places: that is a disagreement,
    and it is read again until two readings stop in the same one."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))
    wrong = {at: _header_with_size(data, at, len(data) * 2) for at in offsets[45:48:2]}
    covering = dict.fromkeys(wrong, 0)

    class RunsOffTwice(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            got = super().read_range(part_idx, offset, length)
            start = self.archive.parts[part_idx].offset + offset
            for at, header in wrong.items():
                if start <= at and at + 512 <= start + length:
                    covering[at] += 1
                    if covering[at] == 1:
                        got = got[:at - start] + header + got[at - start + 512:]
            return got

    rows = _Rows()
    result = walk(ConcatFile(archive, RunsOffTwice(tmp_path, archive), window_min=1024,
                             window_max=1024), 0, rows.commit)

    assert all(covering.values()), "the test did not actually inject both bad reads"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(MEMBERS)


def test_rescues_after_the_walk_moves_on_do_not_use_up_the_verdicts_re_reads(tmp_path):
    """Review 6's #9. A derailment early on, rightly undone by the second reading -- which
    then walks on through the rest of the archive and meets REREAD_LIMIT garbled long names
    on the way. Those are transients met on the way, not the verdict disagreeing with
    itself: the walk ends `complete`, not in UnsettledRead."""
    long_named = [(f"run/{'d' * 120}/f{i}.bin", bytes([i]) * 700)
                  for i in range(REREAD_LIMIT)]
    members = list(MEMBERS[:6]) + long_named + list(MEMBERS[6:12])
    data = build_tar(members, format=tarfile.GNU_FORMAT)
    offsets = _header_offsets(data)
    at = offsets[1]
    wrong = _header_with_size(data, at, offsets[3] + 1024 - at - 512)   # into 3's data
    pending = [offsets[6 + i] + 512 for i in range(REREAD_LIMIT)]       # the name blocks
    state = {"lied": 0}
    archive = write_parts(tmp_path, data, part_size=len(data))

    class DerailsThenGarbles(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            got = bytearray(super().read_range(part_idx, offset, length))
            start = self.archive.parts[part_idx].offset + offset
            if not state["lied"] and start <= at and at + 512 <= start + length:
                state["lied"] = 1
                got[at - start:at - start + 512] = wrong
            elif (state["lied"] and pending and start <= pending[0]
                  and pending[0] + 512 <= start + length):
                block = pending.pop(0)
                got[block - start:block - start + 512] = b"garbled/" * 64
            return bytes(got)

    rows = _Rows()
    result = walk(ConcatFile(archive, DerailsThenGarbles(tmp_path, archive),
                             window_min=1024, window_max=1024), 0, rows.commit)

    assert state["lied"] and not pending, "the test did not inject every bad read"
    assert result.state == "complete", result.detail
    assert [f"{rows.rows[k].dir}/{rows.rows[k].name}" for k in sorted(rows.rows)] == [
        name for name, _ in members]


def test_a_landing_that_reads_two_more_fetches_of_real_headers_is_read_back_past(
        tmp_path):
    """The wrong header's size lands on a real header in a stored tarball whose members are
    each a fetch of their own, and the chain walks two of them -- two fetches -- before
    that tarball's terminator gives the verdict. The fetch that served the wrong header is
    the third back, which is why a verdict is read again from three."""
    inner = build_tar([(f"inner/g{i}.dat", bytes([i]) * 5000) for i in range(3)])
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    at = offsets[1]
    wrong = _header_with_size(data, at, stored + _header_offsets(inner)[1] - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader, window_min=4096, window_max=4096), 0,
                  rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)


# ---- review 7: every pass that goes against the one before it counts --------------------

@pytest.mark.parametrize("settles_on", ["another member", "no member"])
def test_a_header_that_flips_from_pass_to_pass_gives_up_rather_than_loop(tmp_path,
                                                                         settles_on):
    """Review 7's B1. The server serves member 20's header one way on one pass and another
    way on the next, each version holding for the whole pass, tie-break included. Where
    each version leads it serves a pax header, and a read starting there finds none. So
    each pass goes against the one before it at 20, then is caught reading wrongly where
    the other never read, and ends `rescued`, with no verdict to compare. Rescued passes
    were never counted, so the walk read those places forever. A pass that goes against
    the one before it is a disagreement, whether it reaches a verdict or not -- and
    whether the tie-break takes it to another member or, as fresh reads that find
    nothing there would, to none."""
    data = build_tar(MEMBERS)
    offsets = _header_offsets(data)
    flip, lands_a, lands_b = offsets[20], offsets[21], offsets[23]
    version_b = _header_with_size(data, flip, lands_b - flip - 512)
    suspect = _pax_tar([("elsewhere/p.dat", b"q" * 10, {"mtime": "1700000000.5"})])[:1536]
    assert suspect[156:157] == b"x"
    archive = write_parts(tmp_path, data, part_size=len(data))
    state = {"flipped": False, "passes": 1, "fetches": 0}

    class Flipping(ConcatFile):
        def drop_cache(self, keep_window=False):
            if keep_window:                 # the walk starting its next pass
                state["flipped"] = not state["flipped"]
                state["passes"] += 1
            super().drop_cache(keep_window)

    class Reader(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            state["fetches"] += 1
            if state["fetches"] > 500:
                raise RuntimeError("still reading: the walk is going round in circles")
            got = bytearray(super().read_range(part_idx, offset, length))
            start = self.archive.parts[part_idx].offset + offset
            if state["flipped"] and start <= flip and flip + 512 <= start + length:
                nothing = settles_on == "no member" and start == flip
                got[flip - start:flip - start + 512] = JUNK * 2 if nothing else version_b
            for at in (lands_a, lands_b):
                if start < at and at + len(suspect) <= start + length:
                    got[at - start:at - start + len(suspect)] = suspect
            return bytes(got)

    rows = _Rows()
    with pytest.raises(UnsettledRead, match="kept contradicting") as raised:
        walk(Flipping(archive, Reader(tmp_path, archive), window_min=8192,
                      window_max=8192), 0, rows.commit)
    # The first pass, and one going against the one before it for each of REREAD_LIMIT + 1.
    assert state["passes"] == REREAD_LIMIT + 2
    assert (f"the last pass was cut short at {lands_a}, where a fresh read contradicted it"
            in str(raised.value))
    assert rows.consistent() and rows.cursor <= flip


# ---- review 7: what is held back costs no more than it must ------------------------------

def _dense(count=600, where="run"):
    return [(f"{where}/s{i:05d}.bin", b"s" * 100) for i in range(count)]


def test_no_commit_carries_more_than_a_batch(tmp_path):
    """Review 7's B3. In a run of small members one fetch serves hundreds of headers, and
    they leave the held-back window together; and once a verdict's two readings agree,
    everything held back is committed at once. Neither may pass `batch_size` in one
    transaction: `--batch` is what bounds one, and `--max-batches` counts them."""
    members = _dense()
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows = _Rows()

    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=65536,
                             window_max=65536), 0, rows.commit, batch_size=50)

    assert result.state == "complete"
    assert rows.names() == _names_of(members)
    assert max(len(offsets) for _, offsets, _ in rows.calls) == 50
    headers = _header_offsets(data) + [_terminator(data)]
    for _, offsets, cursor in rows.calls:
        # Each commit's cursor is where the first member it did not carry begins.
        assert cursor == (headers[headers.index(offsets[-1]) + 1] if offsets else cursor)


def test_a_crossing_or_a_stop_anywhere_in_a_dense_run_commits_no_more_than_a_batch(
        tmp_path):
    """A crossing or a stop ends the pass on the member that reaches the boundary, or the
    one read when the stop came. When that member is the first a new fetch served, a whole
    fetch's worth leaves the held-back window with it, and the pass's last commit carries
    them: it too goes a batch at a time."""
    members = _dense()
    data = build_tar(members)
    headers = _header_offsets(data)
    archive = write_parts(tmp_path, data, part_size=len(data))

    def fresh():
        return ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=65536,
                          window_max=65536)

    largest = set()
    for stop_at in headers[300:400]:
        rows = _Rows()
        result = walk(fresh(), 0, rows.commit, batch_size=10, stop_at=stop_at)
        assert (result.state, result.end_offset) == ("crossed", stop_at)
        assert rows.names() == _names_of(members[:headers.index(stop_at)])
        largest.add(max(len(offsets) for _, offsets, _ in rows.calls))
    for when in range(300, 400):
        rows, checks = _Rows(), iter(range(10**6))
        result = walk(fresh(), 0, rows.commit, batch_size=10,
                      should_stop=lambda: next(checks) == when)
        assert result.state == "stopped" and rows.consistent()
        largest.add(max(len(offsets) for _, offsets, _ in rows.calls))
    assert largest == {10}


@pytest.mark.parametrize("refused", range(1, 30))
def test_a_commit_refused_anywhere_in_a_dense_run_resumes_exactly(tmp_path, refused):
    """Committed a batch at a time, the members of an agreeing second reading carry the
    cursor of the next one in each commit. Refuse any one commit and resume from the
    cursor the store holds: nothing lost, nothing repeated."""
    members = _dense()
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows, calls = _Rows(), {"n": 0}

    def commit(batch, next_offset):
        calls["n"] += 1
        if calls["n"] == refused:
            raise RuntimeError("simulated: the store refused this commit")
        rows.commit(batch, next_offset)

    def fresh():
        return ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=65536,
                          window_max=65536)

    try:
        walk(fresh(), 0, commit, batch_size=50)
    except RuntimeError:
        assert rows.cursor is None or rows.consistent()
        walk(fresh(), rows.cursor or 0, rows.commit, batch_size=50)
    assert rows.names() == _names_of(members)


def test_a_second_reading_is_not_handed_the_members_the_first_one_held_back(tmp_path,
                                                                           monkeypatch):
    """Review 7's B3. The second reading compares each header with the first reading's --
    for which it needs where each member begins and how it read, not the members: those
    it commits are its own. In a dense run the first reading holds back tens of
    thousands, and every chain in the pool keeps them while it reads them again."""
    import gc

    import dbaudit.archive.tarwalk as tarwalk

    data = build_tar(_dense(where="run/held"))
    archive = write_parts(tmp_path, data, part_size=len(data))
    alive = []
    original = tarwalk._walk_chain

    def counting(*args, **kwargs):
        alive.append(sum(isinstance(o, tarwalk.Member) and o.dir == "run/held"
                         for o in gc.get_objects()))
        return original(*args, **kwargs)

    monkeypatch.setattr(tarwalk, "_walk_chain", counting)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=65536,
                             window_max=65536), 0, lambda members, offset: None)

    assert result.state == "complete"
    assert alive == [0, 0]
    assert hasattr(tarwalk.Member, "__slots__"), "and each one carries no dict of its own"


def test_a_second_reading_lets_go_of_the_first_as_it_passes_it(tmp_path, monkeypatch):
    """Review 7's B3. Each header sequence is compared once, so the second reading drops
    the first reading's version of each as it passes it: the two readings' held-back
    windows are never both held whole."""
    import dbaudit.archive.tarwalk as tarwalk

    data = build_tar(_dense())
    archive = write_parts(tmp_path, data, part_size=len(data))
    handed = []
    original = tarwalk._walk_chain

    def keeping(*args):
        handed.append((args[-1], len(args[-1])))
        return original(*args)

    monkeypatch.setattr(tarwalk, "_walk_chain", keeping)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=65536,
                             window_max=65536), 0, lambda members, offset: None)

    assert result.state == "complete"
    (_, none), (versions, compared) = handed
    assert none == 0 and compared > 100
    # All but the terminator's: where the pass ends, it reads no member to compare.
    assert list(versions) == [_terminator(data)]


# ---- review 8: the window ages by how far the walk moves on, not by what it re-reads ---

def _uncopied_long_names(count):
    """``count`` small members with GNU long names, from a writer that does not copy the
    start of the name into the header after the name's blocks -- which `_settled`
    accepts, at the cost of one more read each."""
    data = bytearray(build_tar([("run/" + "d" * 120 + f"/f{i:05d}.dat", b"z" * 100)
                                for i in range(count)]))
    with tarfile.open(fileobj=io.BytesIO(bytes(data)), mode="r:") as t:
        reals = [m.offset_data - 512 for m in t]
    for real in reals:
        block = data[real:real + 512]
        block[0:100] = b"elsewhere".ljust(100, b"\0")
        block[148:156] = b" " * 8
        block[148:156] = b"%06o\0 " % sum(block)
        data[real:real + 512] = block
    return bytes(data)


def test_a_run_of_members_each_read_twice_still_commits_as_it_goes(tmp_path, monkeypatch):
    """Review 8's #1. A member that needs a second read costs a fetch of its own, and the
    walk goes on reading the headers after it out of that fetch. Those re-reading fetches
    are not the walk's own, so the headers read from them never counted as moving on: in a
    run of such members the held-back window never aged. Nothing was committed until the
    verdict -- a stop or a crash walked the run again from its start -- and the verdict's
    second reading read every member of it twice again. How far the walk has read decides
    what it holds back, whichever fetch the headers came out of."""
    import dbaudit.archive.tarwalk as tarwalk

    count = 2000
    data = _uncopied_long_names(count)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = LocalRangeReader(tmp_path, archive)
    rows, held = _Rows(), []
    original = tarwalk._walk_chain

    def recording(*args):
        this = original(*args)
        held.append((len(rows.rows), len(this.window)))
        return this

    monkeypatch.setattr(tarwalk, "_walk_chain", recording)
    result = walk(ConcatFile(archive, reader), 0, rows.commit)

    assert result.state == "complete"
    assert len(rows.rows) == count
    (committed, held_back), _ = held
    assert held_back < count // 10, "the first reading held the whole run back"
    assert committed + held_back == count
    # A read of each member, once more for each one held back, and a handful of the walk's.
    assert reader.requests < count * 11 // 10


def test_a_server_that_lies_about_most_headers_is_given_up_on(tmp_path, monkeypatch):
    """Review 8's #2. The server serves a pax header in place of every header a read does
    not start at -- as a first read of each does not -- and the truth to a read that
    starts there, as a re-read does. So every pass ends `rescued` one member further on,
    having read every member it held back again: 601 passes and 181,499 requests for a
    0.9 MiB archive. Rescues one after another inside one held-back window, with nothing
    committed in between, are a server lying too often to walk through, and count."""
    import dbaudit.archive.tarwalk as tarwalk

    data = build_tar([(f"d/f{i:06d}", b"z" * 1000) for i in range(200)])
    offsets = _header_offsets(data)
    suspect = _pax_tar([("elsewhere/p.dat", b"", {"mtime": "1700000000.5"})])[:1536]
    assert suspect[156:157] == b"x"
    archive = write_parts(tmp_path, data, part_size=len(data))

    class LiesUnlessAskedThere(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            got = bytearray(super().read_range(part_idx, offset, length))
            start = self.archive.parts[part_idx].offset + offset
            for at in offsets:
                if start < at and at + len(suspect) <= start + length:
                    got[at - start:at - start + len(suspect)] = suspect
            return bytes(got)

    passes = []
    original = tarwalk._walk_chain
    monkeypatch.setattr(tarwalk, "_walk_chain",
                        lambda *args: passes.append(1) or original(*args))
    reader = LiesUnlessAskedThere(tmp_path, archive)
    rows = _Rows()

    with pytest.raises(UnsettledRead, match="was cut short"):
        walk(ConcatFile(archive, reader, window_min=65536, window_max=65536), 0,
             rows.commit)
    # The first pass, and REREAD_LIMIT + 1 rescued one after another with nothing between.
    assert len(passes) == REREAD_LIMIT + 2
    assert reader.requests < 30
    assert rows.consistent()


def test_a_header_sequence_two_fetches_serve_counts_as_two(tmp_path, monkeypatch):
    """A long name's blocks and the header after them can take two fetches of a small
    window, and both are the walk's own. Counted as one, the walk holds back a fetch more
    than it says -- and reads it again, at a request's cost, before every verdict."""
    import dbaudit.archive.tarwalk as tarwalk

    members = [("run/a.bin", b"a" * 100), ("run/" + "d" * 120 + "/b.bin", b"b" * 100),
               ("run/c.bin", b"c" * 100)]
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    held = []
    original = tarwalk._walk_chain

    def recording(*args):
        this = original(*args)
        held.append([member.name for _, _, member in this.window])
        return this

    monkeypatch.setattr(tarwalk, "_walk_chain", recording)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=1024,
                             window_max=1024), 0, lambda members, offset: None)

    assert result.state == "complete"
    # The last three fetches: two for b.bin's header sequence, one for c.bin's.
    assert held[0] == ["b.bin", "c.bin"]


def test_reading_on_out_of_re_reads_fetches_moves_on_a_fetch_at_a_time(tmp_path):
    """Review 6's #4 case, one stored member longer: the wrong header lands in a stored
    tarball whose members each take a re-read, and the walk reads on out of those re-reads'
    fetches through four of them before the verdict. That is a fetch's worth of reading on,
    not four: counted a header at a time, the fetch that served the wrong header left what
    is read again first, and the verdict stood."""
    inner = build_tar([("ginner/" + "d" * 110 + f"/g{i:02d}.dat", bytes([i]) * 600)
                       for i in range(5)])
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        sequences = [m.offset for m in t]
        reals = [m.offset_data - 512 for m in t]
    for real in reals:                              # a writer that does not copy the name
        inner = _with_field(inner, real, 0, 100, b"elsewhere")
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    at = offsets[1]
    wrong = _header_with_size(data, at, stored + sequences[1] - at - 512)
    archive = write_parts(tmp_path, data, part_size=len(data))
    reader = _LiesOnce(tmp_path, archive, at, fill=wrong, span=512)
    rows = _Rows()

    result = walk(ConcatFile(archive, reader, window_min=4096, window_max=4096), 0,
                  rows.commit)

    assert reader.lied == 1, "the test did not actually inject the bad read"
    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)


def test_a_verdict_reached_after_rescues_does_not_count_as_one(tmp_path):
    """Rescues one after another count toward REREAD_LIMIT; the pass that then walks on to
    the verdict was not cut short, and does not. Here REREAD_LIMIT + 1 headers in a row
    read wrongly unless read from exactly where they begin: the first rescue is the walk's
    first reading, the rest count, and the archive is still read to its end."""
    members = [(f"run/m{i:02d}.bin", bytes([i]) * 1000) for i in range(30)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    lying = offsets[20:20 + REREAD_LIMIT + 1]
    suspect = _pax_tar([("elsewhere/p.dat", b"", {"mtime": "1700000000.5"})])[:1536]
    archive = write_parts(tmp_path, data, part_size=len(data))

    class LiesUnlessAskedThere(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            got = bytearray(super().read_range(part_idx, offset, length))
            start = self.archive.parts[part_idx].offset + offset
            for at in lying:
                if start < at and at + len(suspect) <= start + length:
                    got[at - start:at - start + len(suspect)] = suspect
            return bytes(got)

    rows = _Rows()
    result = walk(ConcatFile(archive, LiesUnlessAskedThere(tmp_path, archive),
                             window_min=65536, window_max=65536), 0, rows.commit)

    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert all(f"{at}" in result.detail for at in lying), "each one was a rescue"


# ---- review 9: what the counted fetches hold is not counted again; rescues far apart -----

def _derailed_into_a_stored_tarball(directory, inner_members, uncopied, landing, window):
    """Walk an archive whose second header is read once wrongly, sending the walk to the
    header sequence ``landing`` of a stored tarball of ``inner_members``; those at
    ``uncopied`` come from a writer that does not copy a long name into the header, so
    each takes a re-read. Fetches are ``window`` bytes. Returns what a test needs to see."""
    directory.mkdir()
    inner = build_tar(inner_members)
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as t:
        sequences = [(m.offset, m.offset_data) for m in t]
    for i in uncopied:
        inner = _with_field(inner, sequences[i][1] - 512, 0, 100, b"elsewhere")
    members = [("run/a.bin", b"a" * 3000), ("run/b.bin", b"b" * 3000),
               ("run/results.tar", inner), ("run/z.bin", b"z" * 3000)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        stored = t.getmember("run/results.tar").offset_data
    at = offsets[1]
    wrong = _header_with_size(data, at, stored + sequences[landing][0] - at - 512)
    archive = write_parts(directory, data, part_size=len(data))
    reader = _LiesOnce(directory, archive, at, fill=wrong, span=512)
    rows = _Rows()
    result = walk(ConcatFile(archive, reader, window_min=window, window_max=window), 0,
                  rows.commit)
    assert reader.lied == 1, "the test did not actually inject the bad read"
    # Where each stored member's header sequence begins and ends, in the archive.
    inside = [(stored + begins, stored + ends) for begins, ends in sequences]
    return result, rows.names() == _names_of(members), inside


def test_a_fetch_made_where_a_re_read_counted_one_does_not_count_again(tmp_path):
    """Review 9's #1. The wrong header lands in a stored tarball, at L, read out of a fetch
    C of the walk's own. A member S inside C takes a re-read, whose fetch R reaches past
    C's end, so the header H after C's end is read out of R and counts one: the fetch the
    walk would have made there. The header K after H lies past R's end and takes a fetch of
    the walk's own -- that same fetch, made for real, as it would have served both. Counted
    again, the fetch that served the wrong header left what is read again a fetch early,
    and the verdict stood. Without the re-read the derailment is caught, and so with it."""
    window = 4096
    inner = [("i/first.dat", b"f" * 100), ("i/l.dat", b"l" * 100),
             ("i/" + "s" * 110 + "/s.dat", b"s" * 100), ("i/m.dat", b"m" * 500),
             ("i/h.dat", b"h" * 600), ("i/k.dat", b"k" * 100)]
    for uncopied in ((), (2,)):
        result, exact, inside = _derailed_into_a_stored_tarball(
            tmp_path / f"uncopied{len(uncopied)}", inner, uncopied, 1, window)
        (l, _), (s, _), _, (h, _), (k, _) = inside[1:]
        # C is [l - 1, l - 1 + window) and R is [s, s + window): H past C, in R; K past R.
        assert l - 1 + window <= h and h + 512 <= s + window <= k
        assert result.state == "complete", result.detail
        assert exact


def test_fetching_again_what_a_re_read_took_out_of_the_cache_is_not_reading_on(tmp_path):
    """Review 9's #3. A member's re-read starts where its header sequence does -- before the
    fetch of the walk's own that served the rest of it -- so it ends before that fetch did.
    The next header sequence, which that fetch held whole, then takes a fetch of the walk's
    own for its tail: the walk has not read on, only lost what the re-read replaced.
    Counted, the fetch that served the wrong header left what is read again a fetch early,
    and the verdict stood. Without the re-reads the derailment is caught, and so with them."""
    window = 4096
    inner = [("g/" + "d" * 110 + f"/m{i:02d}", bytes([i + 1]) * size)
             for i, size in enumerate([500, 2500, 1500, 1500, 1000])]
    for uncopied in ((), (0, 1, 3)):
        result, exact, inside = _derailed_into_a_stored_tarball(
            tmp_path / f"uncopied{len(uncopied)}", inner, uncopied, 2, window)
        (landing, _), (third, third_ends), (_, fourth_ends) = inside[2:]
        # The landing's fetch ends inside the third's header sequence; the fetch for the
        # rest of it begins at the block that runs past, before which its re-read begins.
        rest = (landing - 1 + window) // 512 * 512
        assert third < rest < third_ends <= rest + window
        # That fetch held the fourth's sequence whole; the re-read's ends inside it.
        assert third + window < fourth_ends <= rest + window
        assert result.state == "complete", result.detail
        assert exact


def test_a_header_sequence_that_runs_on_into_the_next_fetch_counts_it(tmp_path, monkeypatch):
    """A header sequence that begins in one fetch and ends in the next rests on the next:
    that is the walk reading on, and counts. Judged by where it begins, it would be held
    back only as long as the members before it -- though the fetch that served the rest of
    it may be the bad one."""
    import dbaudit.archive.tarwalk as tarwalk

    members = [("run/a.bin", b""), ("run/" + "d" * 120 + "/b.bin", b"b" * 100),
               ("run/c.bin", b"c" * 100), ("run/d.bin", b"d" * 100)]
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    held = []
    original = tarwalk._walk_chain

    def recording(*args):
        this = original(*args)
        held.append([member.name for _, _, member in this.window])
        return this

    monkeypatch.setattr(tarwalk, "_walk_chain", recording)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=1024,
                             window_max=1024), 0, lambda members, offset: None)

    assert result.state == "complete"
    # The last three fetches: the one b.bin's header sequence runs on into, c.bin's, d.bin's.
    assert held[0] == ["b.bin", "c.bin", "d.bin"]


def test_a_header_where_what_was_counted_ends_lies_past_it(tmp_path, monkeypatch):
    """A fetch holds the bytes up to where it ends, not the one there: a header that begins
    exactly where what was counted ends takes another fetch's worth. Out of re-reads'
    fetches, a run of members two to a window then counts one every two members, as the
    walk's own fetches would; taken as inside, it counts one every three, holding back half
    as much again -- and reading it all again before the verdict."""
    import dbaudit.archive.tarwalk as tarwalk

    count, window = 42, 4096                    # each member 2,048 bytes: two to a window
    data = _uncopied_long_names(count)
    archive = write_parts(tmp_path, data, part_size=len(data))
    held = []
    original = tarwalk._walk_chain

    def recording(*args):
        this = original(*args)
        held.append(len(this.window))
        return this

    monkeypatch.setattr(tarwalk, "_walk_chain", recording)
    result = walk(ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=window,
                             window_max=window), 0, lambda members, offset: None)

    assert result.state == "complete"
    assert held[0] <= 2 * CONFIRM_FILLS         # the last three windows' worth, no more


class _LiesUnlessAskedThere(LocalRangeReader):
    """Serves ``suspect`` -- a pax header from elsewhere -- in place of each header at
    ``lying`` that a read covers without starting there, as a first read of it does not
    and a re-read does."""

    def __init__(self, directory, archive, lying, suspect):
        super().__init__(directory, archive)
        self.lying, self.suspect = lying, suspect

    def read_range(self, part_idx, offset, length):
        got = bytearray(super().read_range(part_idx, offset, length))
        start = self.archive.parts[part_idx].offset + offset
        for at in self.lying:
            if start < at and at + len(self.suspect) <= start + length:
                got[at - start:at - start + len(self.suspect)] = self.suspect
        return bytes(got)


def test_rescues_far_apart_do_not_add_up_to_giving_up(tmp_path):
    """Review 9's #2. Two headers in a row that read wrongly unless read from exactly where
    they begin cut two passes short, the second with nothing committed since the first,
    and that counts toward REREAD_LIMIT. REREAD_LIMIT + 1 such pairs far apart -- as a
    walk of hours meets bad reads -- are not a server that keeps changing its story: only
    a run of passes cut short with nothing committed between them is given up on."""
    gap = 200                                   # ~300 KiB: well past what is held back
    members = [(f"run/m{i:04d}.bin", bytes([i % 251]) * 1000)
               for i in range(30 + (REREAD_LIMIT + 1) * gap)]
    data = build_tar(members)
    offsets = _header_offsets(data)
    lying = [offsets[20 + pair * gap + k] for pair in range(REREAD_LIMIT + 1) for k in (0, 1)]
    suspect = _pax_tar([("elsewhere/p.dat", b"", {"mtime": "1700000000.5"})])[:1536]
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows = _Rows()

    result = walk(ConcatFile(archive, _LiesUnlessAskedThere(tmp_path, archive, lying, suspect),
                             window_min=65536, window_max=65536), 0, rows.commit)

    assert result.state == "complete", result.detail
    assert rows.names() == _names_of(members)
    assert all(f"{at}" in result.detail for at in lying), "each one was a rescue"


# ---- review 10: a walk on from a chain-start scan counts from where the scan ends --------

def test_a_walk_on_from_a_chain_start_scan_counts_from_where_the_scan_ends(tmp_path,
                                                                          monkeypatch):
    """Review 10's #1. A segment's first walk starts out of find_chain_start's read, which
    can begin up to its whole length before the first header. Taken as the fetch the walk
    would have made at that header, it stood for bytes it never held: the walk's own
    fetches after it went uncounted for up to that length, and nothing the scan served
    left what is held back before the verdict. The scan's read holds what it holds; the
    walk's own fetches past it count."""
    import dbaudit.archive.tarwalk as tarwalk

    window, scan, scan_from = 4096, 65536, 1024
    members = ([("run/big.bin", b"\1" * 40_000)]
               + [(f"run/s/f{i:04d}", b"z" * 100) for i in range(60)])
    data = build_tar(members)
    archive = write_parts(tmp_path, data, part_size=len(data))
    concat = ConcatFile(archive, LocalRangeReader(tmp_path, archive), window_min=window,
                        window_max=window)
    start = find_chain_start(concat, scan_from, scan_read=scan)
    assert concat.cached() == (scan_from, scan_from + scan), "the scan is one read"
    passes = []
    original = tarwalk._walk_chain
    monkeypatch.setattr(tarwalk, "_walk_chain",
                        lambda *args: passes.append(original(*args)) or passes[-1])

    result = walk(concat, start, lambda members, offset: None)

    assert result.state == "complete"
    # The walk read on well past the scan's read, a window at a time, before the verdict.
    assert len(data) - (scan_from + scan) > 4 * window
    assert passes[0].held_from >= scan_from + scan, "the scan's members were held to the end"
