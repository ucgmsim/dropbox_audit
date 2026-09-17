import pytest

from dbaudit.archive.reader import ConcatFile, LocalRangeReader
from tests.archive_fakes import write_parts


def test_reads_across_parts_return_the_original_bytes(tmp_path):
    data = bytes(range(256)) * 40                      # 10,240 bytes
    archive = write_parts(tmp_path, data, part_size=1024)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    handle.seek(1000)
    assert handle.read(100) == data[1000:1100]         # straddles a boundary
    handle.seek(0)
    assert handle.read(len(data)) == data


def test_small_hops_coalesce_into_few_requests(tmp_path):
    """A run of small members must not cost one round trip per header."""
    archive = write_parts(tmp_path, b"x" * 200_000, part_size=100_000)
    reader = LocalRangeReader(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=4096, window_max=1 << 16)
    for offset in range(0, 60_000, 512):
        handle.seek(offset)
        handle.read(512)
    assert reader.requests <= 8


def test_long_jumps_fetch_only_the_minimum(tmp_path):
    """A run of large members must not drag a full window behind each header."""
    archive = write_parts(tmp_path, b"y" * 400_000, part_size=400_000)
    reader = LocalRangeReader(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=1024, window_max=1 << 16)
    for offset in range(0, 400_000, 50_000):
        handle.seek(offset)
        handle.read(512)
    assert reader.requests == 8
    assert reader.bytes_fetched == 8 * 1024


def test_reading_past_the_end_returns_what_exists(tmp_path):
    archive = write_parts(tmp_path, b"z" * 100, part_size=100)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    handle.seek(90)
    assert handle.read(512) == b"z" * 10
    assert handle.tell() == 100


def test_unbounded_read_is_refused(tmp_path):
    archive = write_parts(tmp_path, b"z" * 100, part_size=100)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    with pytest.raises(ValueError):
        handle.read(-1)


def test_the_default_window_bounds_are_the_measured_ones(tmp_path):
    """64 KiB to 16 MiB, from Task 1: a request costs 1.60 s + 0.044 s/MiB, so the cap
    is generous and the floor is small because it is paid on every jump."""
    archive = write_parts(tmp_path, b"z" * 100, part_size=100)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    assert handle._window_min == 65536
    assert handle._window_max == 16 << 20


def test_seek_from_end_and_current(tmp_path):
    archive = write_parts(tmp_path, bytes(range(100)), part_size=40)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive))
    handle.seek(-10, 2)
    assert handle.tell() == 90
    handle.seek(-5, 1)
    assert handle.tell() == 85
