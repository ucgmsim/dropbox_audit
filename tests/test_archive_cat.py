"""`archive cat`: retrieve one member's bytes out of a tar archive, by offset, in as
few range reads as possible. Everything here is local or monkeypatched -- see
tests/test_live_archive.py for the one live check, gated behind DBAUDIT_LIVE=1.
"""

import sys
import tarfile

import pytest

from dbaudit.archive.reader import LocalRangeReader, ShortRead
from dbaudit.archive.store import ArchiveStore
from dbaudit.cli import main
from tests.archive_fakes import build_tar, write_parts

PAYLOAD = bytes(range(256)) * 30          # 7,680 bytes: spans a 4 KiB part boundary


def setup_archive(tmp_path):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/small.txt", b"hello\n"), ("run/big.bin", PAYLOAD)])
    write_parts(source, data, part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    return db


def test_cat_extracts_a_member_that_spans_parts(tmp_path):
    db = setup_archive(tmp_path)
    out = tmp_path / "big.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 0
    assert out.read_bytes() == PAYLOAD


def test_cat_refuses_an_ambiguous_path(tmp_path, capsys):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/dup.txt", b"first\n"), ("run/dup.txt", b"second\n")])
    write_parts(source, data, part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/dup.txt"]) != 0
    assert "--offset" in capsys.readouterr().err


# ---- T8-1: stdout is never closed -----------------------------------------------


def test_cat_to_stdout_leaves_stdout_usable_afterwards(tmp_path, capsys):
    """Discriminates closing sys.stdout.buffer (T8-1): `with sys.stdout.buffer as
    sink:` closes it on the way out, so a later write to stdout would raise."""
    db = setup_archive(tmp_path)
    capsys.readouterr()

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/small.txt"]) == 0
    assert capsys.readouterr().out.encode() == b"hello\n"

    sys.stdout.write("still usable\n")
    sys.stdout.flush()
    assert "still usable" in capsys.readouterr().out


# ---- T8-2: usage errors exit 2, not a traceback ---------------------------------


def test_cat_an_unknown_archive_or_a_missing_db_exits_2(tmp_path, capsys):
    db = setup_archive(tmp_path)
    capsys.readouterr()

    assert main(["archive", "cat", "--db", db, "--archive", "nope.tar",
                 "--member", "run/small.txt"]) == 2
    assert "nope.tar" in capsys.readouterr().err

    missing = str(tmp_path / "missing.db")
    assert main(["archive", "cat", "--db", missing, "--archive", "a.tar",
                 "--member", "run/small.txt"]) == 2


# ---- T8-3: --offset is an int, and a bad one is a clean error -------------------


def test_cat_an_offset_naming_no_member_exits_2_and_lists_candidates(tmp_path, capsys):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/dup.txt", b"first\n"), ("run/dup.txt", b"second\n")])
    write_parts(source, data, part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    capsys.readouterr()

    store = ArchiveStore(db)
    matches = store.find_members(store.get("a.tar")["id"], "run/dup.txt")
    assert len(matches) == 2

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/dup.txt", "--offset", "999999999"]) == 2
    err = capsys.readouterr().err
    assert "999999999" in err
    for match in matches:
        assert str(match["hdr_offset"]) in err


def test_cat_with_offset_selects_the_named_duplicate(tmp_path):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/dup.txt", b"first\n"), ("run/dup.txt", b"second\n")])
    write_parts(source, data, part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    store = ArchiveStore(db)
    matches = store.find_members(store.get("a.tar")["id"], "run/dup.txt")
    second = matches[1]

    out = tmp_path / "dup.txt"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/dup.txt", "--offset", str(second["hdr_offset"]),
                 "--out", str(out)]) == 0
    assert out.read_bytes() == b"second\n"


# ---- T8-4: a stale archive refuses to cat ---------------------------------------


def test_cat_refuses_a_stale_archive_and_writes_nothing(tmp_path):
    db = setup_archive(tmp_path)
    store = ArchiveStore(db)
    row = store.get("a.tar")
    store.mark_stale(row["id"], "parts changed underneath the index")

    out = tmp_path / "small.txt"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/small.txt", "--out", str(out)]) == 1
    assert not out.exists()


# ---- T8-5: only a regular file may be cat'd -------------------------------------


def test_cat_refuses_a_directory_naming_its_type(tmp_path, capsys):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/adir", b"", {"type": tarfile.DIRTYPE})])
    write_parts(source, data, part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    capsys.readouterr()

    out = tmp_path / "adir.out"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/adir", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert f"{tarfile.DIRTYPE.decode()!r}" in err
    assert not out.exists()


# ---- T8-6: cat reads exactly the member, not the window -------------------------


def test_cat_fetches_the_member_and_its_anchors_not_the_window(tmp_path, monkeypatch):
    """With the real CAT_CHUNK (16 MiB) a member this size is one request regardless
    of the window, so CAT_CHUNK is shrunk here to force several fills -- the only way
    the window's floor and doubling would diverge from an exact read at unit-test
    scale. T8-6's pass-through window means cat reads what it plans and nothing more.

    What it plans is no longer the member alone: one block either side as anchors, and
    CAT_OVERLAP between consecutive reads, so that a read whose body is some other range
    cannot go unnoticed. That is exactly the span plus the overlaps -- a window-driven
    over-read would show up as more.
    """
    payload = b"y" * 100_000                  # > WINDOW_MIN (65,536)
    tail = b"z" * 50_000                      # room after the member to over-read into
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/small.txt", b"x" * 10),
                       ("run/big.bin", payload),
                       ("run/after.bin", tail)])
    write_parts(source, data, part_size=len(data) + 10)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    chunk, overlap = 8192, 512
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", chunk)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", overlap)
    fetched = {"bytes": 0, "reads": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        block = real_read(self, part_idx, offset, length)
        fetched["bytes"] += len(block)
        fetched["reads"] += 1
        return block

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    out = tmp_path / "big.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 0
    assert out.read_bytes() == payload

    span = -(-len(payload) // 512) * 512 + 2 * 512      # header block + data + next header
    reads = 1 + -(-(span - chunk) // (chunk - overlap))
    assert fetched["reads"] == reads
    assert fetched["bytes"] == span + (reads - 1) * overlap


# ---- T8-8: a short read is a hard failure ---------------------------------------


def test_cat_a_member_running_past_the_end_of_the_parts_exits_1(tmp_path):
    """Truncate a part *after* indexing, without re-registering: the DB still believes
    the part is its original size, so LocalRangeReader raises ShortRead reading past
    what is actually there -- a real exception, not a clean empty chunk.
    """
    source = tmp_path / "parts"
    source.mkdir()
    payload = b"z" * 2000
    data = build_tar([("run/small.txt", b"hi\n"), ("run/big.bin", payload)])
    write_parts(source, data, part_size=len(data))          # a single part
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    store = ArchiveStore(db)
    row = store.get("a.tar")
    member = store.find_members(row["id"], "run/big.bin")[0]
    assert member["size"] == len(payload)

    part_path = sorted(source.iterdir())[0]
    truncated = part_path.read_bytes()[:member["data_offset"] + member["size"] - 50]
    part_path.write_bytes(truncated)

    out = tmp_path / "big.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 1


# ---- T8-9: a missing --out directory is a clean error ---------------------------


def test_cat_out_in_a_missing_directory_exits_2(tmp_path, capsys):
    db = setup_archive(tmp_path)
    capsys.readouterr()

    out = tmp_path / "no" / "such" / "dir" / "small.txt"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/small.txt", "--out", str(out)]) == 2
    assert str(out) in capsys.readouterr().err


# ---- fix round 1: a multi-chunk --out failure must not touch the target path ----


def _setup_for_a_mid_stream_failure(tmp_path, monkeypatch):
    """A member big enough, with CAT_CHUNK shrunk, that a --out cat needs several
    chunks -- and a reader that raises ShortRead on the *second* read_range call, so
    at least one chunk has already reached `sink.write()` before the failure. Task
    11's real use (a large multi-part database) needs many CAT_CHUNK-sized reads at
    the real 16 MiB size, so this is the ordinary failure shape, not an edge case.
    """
    payload = b"z" * 50_000
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar([("run/big.bin", payload)])
    write_parts(source, data, part_size=len(data))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 8192)
    calls = {"n": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ShortRead("simulated mid-stream corruption")
        return real_read(self, part_idx, offset, length)

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    return db, calls


def test_cat_out_a_multi_chunk_failure_leaves_no_file_at_the_target(tmp_path, monkeypatch):
    db, calls = _setup_for_a_mid_stream_failure(tmp_path, monkeypatch)

    out = tmp_path / "big.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 1
    assert calls["n"] >= 2                       # proves this really was multi-chunk
    assert not out.exists()
    # No temp file left behind alongside it either.
    assert list(tmp_path.glob(".cat-*")) == []


def test_cat_out_a_multi_chunk_failure_preserves_an_existing_file(tmp_path, monkeypatch):
    db, calls = _setup_for_a_mid_stream_failure(tmp_path, monkeypatch)

    out = tmp_path / "big.bin"
    original = b"this is the previous good copy -- a failed cat must not destroy it\n"
    out.write_bytes(original)

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 1
    assert calls["n"] >= 2
    assert out.read_bytes() == original


# ---- a read whose body is some other range -----------------------------------------
#
# On 2026-09-21 Dropbox twice returned a 206 whose body was not the range its own
# Content-Range named. cat used to write such a body straight into the operator's file
# with exit 0. It now anchors each member between its own header and whatever follows
# it, overlaps consecutive reads, and re-reads anything that does not line up.

class _Swaps(LocalRangeReader):
    """Truthful, except on the read_range calls numbered in `lie_on` (1-based), which
    come back wrong with the right length, as the real failure did. `lie_on=None` lies
    on every call. By default the lie is the same length from elsewhere in the part;
    `take_from` names where, `fill` repeats a pattern instead, and `back_half` keeps the
    first half of the read true -- a read that goes wrong partway through."""

    def __init__(self, directory, archive, lie_on, fill=None, take_from=None,
                 back_half=False):
        super().__init__(directory, archive)
        self.lie_on, self.fill, self.take_from = lie_on, fill, take_from
        self.back_half, self.calls, self.lied = back_half, 0, 0

    def read_range(self, part_idx, offset, length):
        self.calls += 1
        true = super().read_range(part_idx, offset, length)
        if self.lie_on is not None and self.calls not in self.lie_on:
            return true
        self.lied += 1
        if self.back_half:
            cut = length // 2
            return true[:cut] + (bytes(range(256)) * (length // 256 + 1))[:length - cut]
        if self.fill is not None:
            return (self.fill * (length // len(self.fill) + 1))[:length]
        size = self.archive.parts[part_idx].size
        at = (self.take_from if self.take_from is not None
              else (offset + size // 2) % max(size - length, 1))
        return super().read_range(part_idx, at, length)


def _single_part_archive(tmp_path, members):
    source = tmp_path / "parts"
    source.mkdir()
    data = build_tar(members)
    write_parts(source, data, part_size=len(data))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    return db, source


def _with_reader(monkeypatch, make):
    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            make(row["folder"], archive_set))


RANDOMISH = bytes((i * 7919 + (i >> 8) * 104729) % 251 for i in range(40_000))
AROUND = [("run/before.bin", b"b" * 3000), ("run/data.bin", RANDOMISH),
          ("run/after.bin", b"a" * 3000)]


def test_cat_repairs_a_read_whose_body_is_another_range(tmp_path, monkeypatch):
    db, _ = _single_part_archive(tmp_path, AROUND)
    readers = []
    _with_reader(monkeypatch, lambda d, a: readers.append(_Swaps(d, a, {1})) or readers[-1])
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1, "the test did not actually inject a bad read"
    assert out.read_bytes() == RANDOMISH


@pytest.mark.parametrize("liar", [1, 2, 3])
def test_cat_repairs_a_bad_read_on_any_chunk_of_a_large_member(tmp_path, monkeypatch, liar):
    """Several reads per member, and whichever of them comes back wrong, the file is
    right. Before, only the chunk boundaries of the *index* were checked: none."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    db, _ = _single_part_archive(tmp_path, AROUND)
    readers = []
    _with_reader(monkeypatch, lambda d, a: readers.append(_Swaps(d, a, {liar})) or readers[-1])
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == RANDOMISH


def test_cat_gives_up_on_reads_that_never_line_up_and_writes_nothing(tmp_path, monkeypatch):
    db, _ = _single_part_archive(tmp_path, AROUND)
    _with_reader(monkeypatch, lambda d, a: _Swaps(d, a, None))
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 1
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".cat-")]


def test_cat_notices_a_part_that_changed_under_the_index(tmp_path):
    """A part rewritten since `index` ran, without a re-register: the stored offsets now
    point at other bytes, and nothing but the anchors can tell."""
    db, source = _single_part_archive(tmp_path, AROUND)
    store = ArchiveStore(db)
    member = store.find_members(store.get("a.tar")["id"], "run/data.bin")[0]
    part = sorted(source.iterdir())[0]
    raw = bytearray(part.read_bytes())
    shift = 512
    raw[member["hdr_offset"]:] = raw[member["hdr_offset"] + shift:] + b"\0" * shift
    part.write_bytes(bytes(raw))

    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 1
    assert not out.exists()


def test_cat_trusts_an_all_zero_overlap_only_when_two_reads_agree(tmp_path, monkeypatch):
    """Zeros where two reads overlap agree with anything that is also zeros there -- so
    a wrong body of zeros would sail through. Such a read is taken only once a second
    fresh read of it matches."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = b"\0" * 20_000 + RANDOMISH[:20_000]       # zeros across the first boundary
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3000)])
    readers = []
    _with_reader(monkeypatch,
                 lambda d, a: readers.append(_Swaps(d, a, {2}, fill=b"\0")) or readers[-1])
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


def test_cat_checks_a_zero_length_member_against_its_neighbours(tmp_path, monkeypatch):
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3000),
                                            ("run/empty", b""),
                                            ("run/after.bin", b"a" * 3000)])
    readers = []
    _with_reader(monkeypatch, lambda d, a: readers.append(_Swaps(d, a, {1})) or readers[-1])
    out = tmp_path / "empty"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/empty", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == b""


def test_cat_checks_the_last_member_against_the_terminator(tmp_path, monkeypatch):
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3000),
                                            ("run/last.bin", RANDOMISH[:5000])])
    readers = []
    _with_reader(monkeypatch, lambda d, a: readers.append(_Swaps(d, a, {1})) or readers[-1])
    out = tmp_path / "last.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/last.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == RANDOMISH[:5000]


def test_cat_catches_a_read_that_goes_wrong_partway_through(tmp_path, monkeypatch):
    """Right at its head, wrong from the middle on: only the trailing anchor -- the next
    member's header, one block past this one -- can see that."""
    db, _ = _single_part_archive(tmp_path, AROUND)
    readers = []
    _with_reader(monkeypatch,
                 lambda d, a: readers.append(_Swaps(d, a, {1}, back_half=True)) or readers[-1])
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == RANDOMISH


def test_cat_checks_a_bad_read_over_the_terminator(tmp_path, monkeypatch):
    """The same failure on the archive's last member, whose trailing anchor is the
    terminator: those bytes must be zeros, not merely present."""
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3000),
                                            ("run/last.bin", RANDOMISH[:5000])])
    readers = []
    _with_reader(monkeypatch,
                 lambda d, a: readers.append(_Swaps(d, a, {1}, back_half=True)) or readers[-1])
    out = tmp_path / "last.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/last.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == RANDOMISH[:5000]


def test_cat_rejects_another_members_header_before_writing_a_byte(tmp_path, monkeypatch):
    """The likeliest wrong body on a busy account is some other read's range -- and
    reads start on headers, so it opens with a perfectly valid one. Only its size gives
    it away. Caught at the head, the read is simply made again; caught any later, its
    bytes are already written and cat can only give up."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    db, _ = _single_part_archive(tmp_path, AROUND)
    readers = []
    _with_reader(monkeypatch,          # the lie is the same length, from before.bin's header
                 lambda d, a: readers.append(_Swaps(d, a, {1}, take_from=0)) or readers[-1])
    out = tmp_path / "data.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin", "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == RANDOMISH
