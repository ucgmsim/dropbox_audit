import io
import tarfile

from dbaudit.archive.reader import ConcatFile, LocalRangeReader
from dbaudit.archive.tarwalk import find_chain_start, walk
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


def test_long_names_and_pax_headers_survive(tmp_path):
    long_name = "run/" + "d" * 150 + "/deep.bin"
    for fmt in (tarfile.GNU_FORMAT, tarfile.PAX_FORMAT):
        data = build_tar([(long_name, b"x" * 10)], format=fmt)
        directory = tmp_path / f"f{fmt}"
        directory.mkdir()
        seen, _, result = collect(directory, data)
        assert result.state == "complete"
        assert f"{seen[0].dir}/{seen[0].name}" == long_name


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


def test_a_corrupt_first_block_commits_once_and_reports_corrupt(tmp_path):
    """tarfile.open() itself raises when the very first header is invalid (offset 0
    is special-cased inside tarfile.next()). That path must still commit -- Task 7
    relies on every return path committing exactly once (Ruling H)."""
    data = bytearray(build_tar(MEMBERS))
    data[0:8] = b"\xff" * 8
    archive = write_parts(tmp_path, bytes(data), part_size=4096)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    calls = []
    result = walk(handle, 0, lambda members, offset: calls.append((list(members), offset)))
    assert result.state == "corrupt"
    assert calls == [([], 0)]


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
