import random
from types import SimpleNamespace

import pytest

from dbaudit.archive.reader import LEAD_STEP, SHAPES, ConcatFile, LocalRangeReader
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
    """A run of large members must not drag a full window behind each header. (Shape 0
    has no lead, so the bytes counted are the windows' alone.)"""
    archive = write_parts(tmp_path, b"y" * 400_000, part_size=400_000)
    reader = LocalRangeReader(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=1024, window_max=1 << 16, shape=0)
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


# ---- request shapes: a re-read asks differently ------------------------------------------

DATA = random.Random(7).randbytes(300_000)


class _Recording(LocalRangeReader):
    """Records every request as (part, offset within the part, length)."""

    def __init__(self, directory, archive):
        super().__init__(directory, archive)
        self.asked = []

    def read_range(self, part_idx, offset, length):
        self.asked.append((part_idx, offset, length))
        return super().read_range(part_idx, offset, length)


def _replay(handle):
    """Reads, drops and window-keeping drops across three parts; what each read saw."""
    seen = []
    for step in [(1000, 512), (150_000, 70_000), (99_990, 20), "drop", (250_000, 4096),
                 "keep", (1000, 512), (299_000, 5000)]:
        if step in ("drop", "keep"):
            handle.drop_cache(keep_window=step == "keep")
            continue
        offset, size = step
        handle.seek(offset)
        assert handle.read(size) == DATA[offset:offset + size]
        seen.append((handle.cached(), handle.fills))
    return seen


def test_what_a_read_returns_does_not_depend_on_the_shape(tmp_path):
    """Bytes, the cached span and the fill count are what they were before shapes, in
    every shape: only the requests change."""
    archive = write_parts(tmp_path, DATA, part_size=100_000)

    def run(shape):
        return _replay(ConcatFile(archive, LocalRangeReader(tmp_path, archive),
                                  window_min=4096, window_max=65_536, shape=shape))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("dbaudit.archive.reader.LEAD_STEP", 0)
        unshaped = run(0)
    for shape in (0, 1, 2, 30_599, 34_937, SHAPES - 1):
        assert run(shape) == unshaped


def test_a_re_read_after_a_drop_starts_somewhere_else(tmp_path):
    """Each drop moves the request's start by 40,503 or 25,033 bytes -- never by one,
    which the block-edge guard could undo by moving one of them back a byte."""
    archive = write_parts(tmp_path, DATA, part_size=len(DATA))
    reader = _Recording(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=4096, window_max=4096, shape=0)
    starts = []
    for _ in range(64):
        handle.seek(200_000)
        assert handle.read(512) == DATA[200_000:200_512]
        starts.append(reader.asked[-1][1])
        handle.drop_cache()
    assert starts[0] == 200_000                     # shape 0 has no lead
    assert {abs(b - a) for a, b in zip(starts, starts[1:])} == {25_033, 40_503}


def test_a_re_read_at_a_parts_first_byte_asks_for_a_different_length(tmp_path):
    """A lead cannot reach back past the start of its part, so there the start stays put
    and only the length asked for past the end changes."""
    archive = write_parts(tmp_path, DATA, part_size=100_000)
    reader = _Recording(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=4096, window_max=4096, shape=1)
    for _ in range(8):
        handle.seek(100_000)                        # part 1's first byte
        assert handle.read(512) == DATA[100_000:100_512]
        handle.drop_cache()
    # 4,096 wanted, and past that the lead of shapes 1 to 8.
    assert reader.asked == [(1, 0, 44_599), (1, 0, 19_566), (1, 0, 60_069), (1, 0, 35_036),
                            (1, 0, 10_003), (1, 0, 50_506), (1, 0, 25_473), (1, 0, 65_976)]


def test_a_re_read_crossing_into_the_next_part_asks_it_for_a_different_length(tmp_path):
    """A read that crosses a part boundary is two requests, and the second starts at the
    next part's first byte whatever the shape: it is the end that moves."""
    archive = write_parts(tmp_path, DATA, part_size=100_000)
    reader = _Recording(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=4096, window_max=4096, shape=1)
    for _ in range(8):
        handle.seek(99_000)
        assert handle.read(2000) == DATA[99_000:101_000]
        handle.drop_cache()
    # 3,096 of the window fall in part 1, and past that the lead of shapes 1 to 8.
    assert reader.asked[1::2] == [(1, 0, 43_599), (1, 0, 18_566), (1, 0, 59_069),
                                  (1, 0, 34_036), (1, 0, 9_003), (1, 0, 49_506),
                                  (1, 0, 24_473), (1, 0, 64_976)]


def test_a_lead_never_leaves_its_part_or_adds_a_request(tmp_path):
    archive = write_parts(tmp_path, DATA, part_size=100_000)
    for pos in (0, 95_000, 99_000, 100_005, 199_990, 299_500):
        wanted = archive.slices(pos, min(4096, len(DATA) - pos))
        for shape in (1, 2, 3, 30_599, 34_937, SHAPES - 1):
            reader = _Recording(tmp_path, archive)
            handle = ConcatFile(archive, reader, window_min=4096, window_max=4096,
                                shape=shape)
            handle.seek(pos)
            assert handle.read(512) == DATA[pos:pos + 512]
            assert [part for part, _, _ in reader.asked] == [part for part, _, _ in wanted]
            assert all(0 <= offset and offset + length <= archive.parts[part].size
                       for part, offset, length in reader.asked)
            assert reader.bytes_fetched == sum(length for _, _, length in reader.asked)


def test_a_part_smaller_than_a_request_is_asked_for_whole_every_time(tmp_path):
    """Where shapes stop: a request that already runs from a part's first byte to its
    last cannot move or grow without asking a second part, so it is the same every time.
    In practice only a small last part is like this."""
    archive = write_parts(tmp_path, DATA[:103_000], part_size=100_000)   # 3,000 left
    reader = _Recording(tmp_path, archive)
    handle = ConcatFile(archive, reader, window_min=4096, window_max=4096, shape=1)
    for _ in range(3):
        handle.seek(100_000)
        assert handle.read(512) == DATA[100_000:100_512]
        handle.drop_cache()
    assert reader.asked == [(1, 0, 3000)] * 3


def test_a_new_reader_starts_at_a_random_shape_unless_given_one(tmp_path, monkeypatch):
    archive = write_parts(tmp_path, b"z" * 100, part_size=100)
    drawn = []
    monkeypatch.setattr("dbaudit.archive.reader.secrets",
                        SimpleNamespace(randbelow=lambda n: drawn.append(n) or 12_345))
    assert ConcatFile(archive, LocalRangeReader(tmp_path, archive)).shape == 12_345
    assert ConcatFile(archive, LocalRangeReader(tmp_path, archive), shape=7).shape == 7
    assert drawn == [SHAPES]


def test_a_drop_moves_to_the_next_shape_with_or_without_its_window(tmp_path):
    archive = write_parts(tmp_path, b"z" * 100, part_size=100)
    handle = ConcatFile(archive, LocalRangeReader(tmp_path, archive), shape=SHAPES - 2)
    handle.drop_cache()
    assert handle.shape == SHAPES - 1
    handle.drop_cache(keep_window=True)
    assert handle.shape == 0


def test_every_lead_comes_once_and_near_shapes_are_never_a_byte_apart():
    leads = [shape * LEAD_STEP % SHAPES for shape in range(SHAPES)]
    assert sorted(leads) == list(range(SHAPES))
    assert {abs(leads[(s + 1) % SHAPES] - leads[s]) for s in range(SHAPES)} == {25_033, 40_503}
    # leads[d] is how far apart, mod SHAPES, the leads of two shapes d apart are.
    assert min(d for d in range(1, SHAPES) if leads[d] in (1, SHAPES - 1)) == 30_599
