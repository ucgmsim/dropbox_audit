import io
import random
import tarfile

import pytest

from dbaudit.archive.reader import ConcatFile, LocalRangeReader
from dbaudit.archive.tarwalk import (REREAD_LIMIT, UnsettledRead, UnsupportedArchive,
                                     find_chain_start, walk)
from tests.archive_fakes import build_tar, write_parts

MEMBERS = [(f"run/file{i:03d}.bin", bytes([i % 251]) * (1000 + i)) for i in range(50)]


def collect(tmp_path, data, part_size=4096, start=0, batch_size=7):
    archive = write_parts(tmp_path, data, part_size=part_size)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
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


def test_a_truncated_verdict_is_read_twice_before_it_stands(tmp_path):
    """A chain runs off the end because the last header's size claims more than is
    left. That size is one read's word, so it is read again before `truncated` stands:
    one pass, then a confirming pass that records that last member a second time."""
    data = build_tar(MEMBERS)
    archive = write_parts(tmp_path, data[:len(data) - 4096], part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    calls = []

    result = walk(handle, 0, lambda members, offset: calls.append(len(members)),
                  batch_size=1000)

    assert result.state == "truncated"
    assert "re-read at" not in result.detail
    assert len(calls) == 2, f"one pass and one confirming pass, got {calls}"
    assert calls[1] == 1, "the confirming pass re-reads the header that ran out"


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
    data = build_tar(MEMBERS)
    seen, cursors, _ = collect(tmp_path, data, batch_size=7)
    assert len(cursors) >= len(MEMBERS) // 7
    # After the first batch the cursor is the header offset of the next member.
    assert cursors[0] == seen[7].hdr_offset


def test_should_stop_ends_the_walk_after_the_current_member(tmp_path):
    """A SIGINT handler wants the walk to stop within one header hop, not one whole
    batch: should_stop is checked after every member, not only at batch_size (Ruling
    G)."""
    data = build_tar(MEMBERS)
    reference, _, _ = collect(tmp_path, data)

    directory = tmp_path / "ruling_g"
    directory.mkdir()
    archive = write_parts(directory, data, part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(directory, archive))
    seen = []

    def commit(members, next_offset):
        seen.extend(members)

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] >= 5

    result = walk(handle, 0, commit, batch_size=2000, should_stop=should_stop)
    assert result.state == "stopped"
    assert [m.name for m in seen] == [m.name for m in reference[:5]]
    assert result.end_offset == reference[5].hdr_offset

    resumed, _, resumed_result = collect(directory, data, start=result.end_offset)
    assert resumed_result.state == "complete"
    assert [m.name for m in resumed] == [m.name for m in reference[5:]]


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
                  lambda members, offset: calls.append((list(members), offset)),
                  stop_at=boundary)
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
        if self.once and self.covering != self.nth:
            return data
        self.lied += 1
        cut = self.bad - start
        n = length - cut if self.span is None else min(self.span, length - cut)
        fill = self.fill * (n // len(self.fill) + 1)
        return data[:cut] + fill[:n] + data[cut + n:]


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
    checking against it takes no request: one fetch for the whole small archive, and two
    for the pass confirming the terminator (tarfile opens there, then seeks back a byte
    to check the data before it) -- three, as before the check existed."""
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
    assert reader.requests == 3
    assert "re-read" not in result.detail


def test_a_writer_that_does_not_copy_the_name_costs_one_read_not_a_refusal(tmp_path):
    """A writer is free to put anything in the field the long name overrides. The two
    disagree, so the member is read again; the second read agrees with the first, which
    makes it the archive, not a bad read -- recorded as read, one request dearer than
    the three a conforming archive costs."""
    data, _ = _long_name_archive()
    data = _with_field(bytes(data), 2048, 0, 100, b"placeholder")
    rows, result, reader = _rows(tmp_path, data=data)

    assert result.state == "complete"
    assert rows[1024].name == "deep.bin" and rows[1024].dir == "run/" + "d" * 150
    assert reader.requests == 4
    assert "re-read" not in result.detail


def test_a_long_name_that_reads_differently_every_time_gives_up(tmp_path):
    """A server that never tells the same story twice is not an archive fact. The walk
    gives up -- keeping what it walked before, with the cursor on the member it could
    not settle, so the retry that `UnsettledRead` invites starts exactly there."""
    data, _ = _long_name_archive()
    archive = write_parts(tmp_path, bytes(data), part_size=len(data))
    reader = _AlwaysLies(tmp_path, archive, 1536, span=512)
    calls = []

    with pytest.raises(UnsettledRead, match="1024"):
        walk(ConcatFile(archive, reader), 0,
             lambda members, offset: calls.append(([m.name for m in members], offset)))
    assert calls[-1] == (["first.bin"], 1024)


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
    assert reader.requests == 3


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
    assert reader.requests == 3


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
    assert len(calls) == 2, f"expected one pass plus one confirming pass, got {calls}"


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
    assert len(calls) == 2, f"expected one pass plus one confirming pass, got {calls}"
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
    return walk(handle, 0, lambda members, offset: seen.extend(members)), seen, reader


def test_a_walk_rescues_up_to_the_reread_limit(tmp_path):
    result, seen, reader = _walk_with_transients(tmp_path, REREAD_LIMIT)
    assert reader.lied == REREAD_LIMIT
    assert result.state == "complete"
    assert len(seen) == len(MEMBERS)


def test_a_walk_that_cannot_get_two_reads_to_agree_gives_up_loudly(tmp_path):
    """One more disagreement than REREAD_LIMIT allows, and the walk raises rather than
    report a verdict it never confirmed. The segment then lands in `error`, which the
    next run reclaims and resumes from its committed cursor."""
    with pytest.raises(UnsettledRead):
        _walk_with_transients(tmp_path, REREAD_LIMIT + 1)


# ---- no single read decides a final verdict -----------------------------------------
#
# Each test lies exactly once, on the Nth fetch that covers the offset deciding the
# verdict, for each N that such a fetch exists -- measured, and asserted, so a change to
# the read pattern fails here rather than leaving a case that quietly tests nothing.
# Which fetch is the dangerous one depends on window sizes and pass structure, so the
# tests do not guess it: whichever read the lie lands on, the verdict must be the truth.
# On a walk from 0, N = 3 is the read that confirms the verdict, and a lie there used to
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


@pytest.mark.parametrize("nth", [1, 2, 3])
def test_one_bad_read_cannot_turn_a_complete_archive_corrupt(tmp_path, nth):
    data = build_tar(MEMBERS)
    result, seen, reader = _walk_lying_once(tmp_path, data, _terminator(data), nth)
    assert result.state == "complete", f"a lie on covering read #{nth} decided the verdict"
    assert len(seen) == len(MEMBERS)
    assert reader.lied == 1, "the test did not actually inject a bad read"


@pytest.mark.parametrize("nth", [1, 2, 3])
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
    result = walk(ConcatFile(archive, reader), 0, lambda members, offset: None,
                  stop_at=offs[30])
    assert reader.lied == 1
    assert result.state == "crossed"
    assert result.end_offset == offs[30]
    assert f"re-read at {offs[10]}" in result.detail


def test_the_note_names_each_re_read_offset_once(tmp_path):
    """Re-reading one offset until two reads agree can take several passes there; the
    note is for a person, and should name the offset once."""
    data = build_tar(MEMBERS)
    t = _terminator(data)
    result, _, _ = _walk_lying_once(tmp_path, data, t, 3)
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
    assert calls[-1] == (["a.bin", "b.bin"], pax_member.offset)


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
    sequence begins, at the `g`, or a resume walks on with no global header in force."""
    data = _pax_tar([(f"g/f{i}", bytes([i]) * 600, None) for i in range(3)],
                    pax_global={"comment": "0123abcd"})
    archive = write_parts(tmp_path, data, part_size=len(data))

    class EveryRereadDiffers(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            true = super().read_range(part_idx, offset, length)
            if self.requests == 1:
                return true                   # the first read: the g and its member
            return random.Random(self.requests).randbytes(length)

    reader = EveryRereadDiffers(tmp_path, archive)
    calls = []
    with pytest.raises(UnsettledRead):
        walk(ConcatFile(archive, reader, window_min=4_096, window_max=4_096), 0,
             lambda members, offset: calls.append((len(members), offset)))
    assert calls[-1] == (0, 0), calls
