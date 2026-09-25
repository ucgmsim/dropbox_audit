"""`archive cat`: retrieve one member's bytes out of a tar archive, by offset, in as
few range reads as possible. Everything here is local or monkeypatched -- see
tests/test_live_archive.py for the one live check, gated behind DBAUDIT_LIVE=1.
"""

import io
import random
import sys
import tarfile
from pathlib import Path

import pytest

from dbaudit.archive.reader import LocalRangeReader, ShortRead
from dbaudit.archive.store import ArchiveStore
from dbaudit.archive.tarwalk import Member, WalkResult
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
    payload = random.Random(1).randbytes(100_000)    # > WINDOW_MIN (65,536)
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


# Random, so no stretch of it recurs elsewhere: two reads agreeing on a piece of it says
# something. (Repetitive data makes cat read twice; see the fill and zeros tests below.)
RANDOMISH = random.Random(20260924).randbytes(40_000)
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


# ---- review 4: every byte written is one that two requests agree on -----------------------
#
# The anchored reads above catch a body that is some other range entirely. Review 4 found
# the shapes they missed: another member of the same size, a read wrong only in its back
# half, overlaps of zeros or fill, a read crossing a part boundary, an archive that ends
# at the member, a long-name header of the right size. Each must now be repaired or
# refused -- never written, with exit 0.

class _Planned(LocalRangeReader):
    """Truthful, except on the read_range calls numbered in ``plan`` (1-based), whose
    body is ``plan[n](reader, offset, length, true)`` instead -- always the right length,
    as the real failure was. ``offset`` is the archive-wide one. Every call is logged."""

    def __init__(self, directory, archive, plan=None):
        super().__init__(directory, archive)
        self.plan, self.calls, self.lied, self.log = dict(plan or {}), 0, 0, []

    def raw(self, offset, length):
        """The true bytes at an archive-wide offset, straight from the part files."""
        out = b""
        for idx, within, take in self.archive.slices(offset, length):
            with open(self.directory / self.archive.parts[idx].name, "rb") as handle:
                handle.seek(within)
                out += handle.read(take)
        return out

    def read_range(self, part_idx, offset, length):
        self.calls += 1
        true = super().read_range(part_idx, offset, length)
        at = self.archive.parts[part_idx].offset + offset
        self.log.append((at, length))
        make = self.plan.get(self.calls)
        if make is None:
            return true
        self.lied += 1
        lie = make(self, at, length, true)
        assert len(lie) == length, "a bad read keeps its length, as the real one did"
        return lie


def _random(seed, n):
    return random.Random(seed).randbytes(n)


def _split_archive(tmp_path, members, part_size):
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(members), part_size=part_size)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    return db, source


def _row(db, path):
    store = ArchiveStore(db)
    return store.find_members(store.get("a.tar")["id"], path)[0]


def _cat(db, member, out):
    return main(["archive", "cat", "--db", db, "--archive", "a.tar", "--member", member,
                 "--out", str(out)])


def _planned(monkeypatch, plan):
    readers = []
    _with_reader(monkeypatch,
                 lambda d, a: readers.append(_Planned(d, a, plan)) or readers[-1])
    return readers


def _back_half(reader, at, length, true):
    cut = length // 2
    return true[:cut] + (bytes(range(256)) * (length // 256 + 1))[:length - cut]


def test_cat_is_not_fooled_by_another_member_of_the_same_size(tmp_path, monkeypatch):
    """root_params.yaml in two realisations: same size, other bytes. The body of a
    concurrent `cat` of the other one opens on a valid header stating the same size --
    all the anchor used to ask for -- and ends on a valid header too. Its path gives it
    away."""
    mine, theirs = _random(12, 1_359), _random(13, 1_359)
    db, _ = _single_part_archive(tmp_path, [("run/r01/root_params.yaml", mine),
                                            ("run/r02/root_params.yaml", theirs),
                                            ("run/r02/after.bin", b"a" * 100)])
    other = _row(db, "run/r02/root_params.yaml")
    readers = _planned(monkeypatch, {1: lambda r, at, n, t: r.raw(other["hdr_offset"], n)})
    out = tmp_path / "root_params.yaml"

    assert _cat(db, "run/r01/root_params.yaml", out) == 0
    assert readers[0].lied == 1, "the test did not actually inject the bad read"
    assert out.read_bytes() == mine


def test_cat_is_not_fooled_by_a_long_name_header_of_the_right_size(tmp_path, monkeypatch):
    """A GNU ././@LongLink header states the name's length + 1: for a 200-byte member, a
    body starting on it passes a size check, the name block standing in for the data."""
    long_name = "d/" + "n" * 197
    small = _random(6, 200)
    db, _ = _single_part_archive(tmp_path, [("run/x.bin", small), (long_name, b"y" * 10)])
    other = _row(db, long_name)
    assert other["data_offset"] - other["hdr_offset"] > 512
    readers = _planned(monkeypatch, {1: lambda r, at, n, t: r.raw(other["hdr_offset"], n)})
    out = tmp_path / "x.bin"

    assert _cat(db, "run/x.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == small


def test_cat_refuses_a_part_rewritten_with_the_same_layout(tmp_path, capsys):
    """Re-tarred under the same names and sizes, one file changed: every header is valid
    and the right size, so only the fields the index recorded -- here the mtime -- can
    tell. The file is refused, naming what differs, and nothing is written."""
    old, new = _random(7, 40_000), _random(8, 40_000)
    db, source = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                                 ("run/data.bin", old),
                                                 ("run/after.bin", b"a" * 3_000)])
    [part] = sorted(source.iterdir())
    part.write_bytes(build_tar([("run/before.bin", b"b" * 3_000),
                                ("run/data.bin", new, {"mtime": 1_800_000_000}),
                                ("run/after.bin", b"a" * 3_000)]))
    capsys.readouterr()
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 1
    assert "mtime 1800000000, not 1700000000" in capsys.readouterr().err
    assert not out.exists()


def test_cat_repairs_a_read_wrong_in_its_back_half_before_writing_it(
        tmp_path, monkeypatch, capsysbinary):
    """Right at its head, wrong from halfway: on a read that is not the last, only the
    next read can see it, so no read is written until the next has agreed with its tail.
    On stdout -- which cannot be taken back -- not one wrong byte goes out."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _random(9, 40_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])
    readers = _planned(monkeypatch, {1: _back_half})
    capsysbinary.readouterr()

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/data.bin"]) == 0
    assert readers[0].lied == 1
    assert capsysbinary.readouterr().out == payload


def test_cat_does_not_take_zeros_at_a_reads_tail_on_trust(tmp_path, monkeypatch):
    """The first read replaced by another same-size member's range, both zero where the
    first and second reads meet: the second read's head agrees with the wrong first read,
    because zeros agree with zeros. Agreement on zeros vouches for neither side."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)

    def content(seed):
        body = bytearray(_random(seed, 40_000))
        body[14_000:16_500] = bytes(2_500)          # the first reads meet in here
        return bytes(body)

    a, b = content(1), content(2)
    db, _ = _single_part_archive(tmp_path, [("run/r01/out.bin", a), ("run/r02/out.bin", b)])
    other = _row(db, "run/r02/out.bin")
    readers = _planned(monkeypatch, {1: lambda r, at, n, t: r.raw(other["hdr_offset"], n)})
    out = tmp_path / "out.bin"

    assert _cat(db, "run/r01/out.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == a


def test_cat_does_not_take_fill_at_both_ends_of_a_read_on_trust(tmp_path, monkeypatch):
    """Zeros have a twin: any overlap whose bytes recur elsewhere. Both ends of the
    second read fall in 0xFF fill, and the wrong body is fill from another member -- it
    agrees with both neighbours while its own middle, which is data, is lost."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = bytearray(b"\xff" * 50_000)
    body[17_000:29_000] = _random(3, 12_000)
    body = bytes(body)
    db, _ = _single_part_archive(tmp_path, [("run/fill.bin", b"\xff" * 40_000),
                                            ("run/data.bin", body)])
    fill = _row(db, "run/fill.bin")
    readers = _planned(monkeypatch, {2: lambda r, at, n, t: r.raw(fill["data_offset"] + 100, n)})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == body


@pytest.mark.parametrize("extra", [0, 300])
def test_cat_confirms_the_last_read_when_nothing_follows_the_member(
        tmp_path, monkeypatch, extra):
    """An archive that ends at the member, or within a block of it, offers no block
    after it to vouch for the read's tail -- so the read is taken only once a second
    fetch agrees with it."""
    payload = _random(4, 5_000)
    full = build_tar([("run/before.bin", b"b" * 3_000), ("run/last.bin", payload)])
    with tarfile.open(fileobj=io.BytesIO(full), mode="r:") as t:
        last = t.getmembers()[-1]
    end = last.offset_data + -(-last.size // 512) * 512
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, full[:end + extra], part_size=end + extra)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    assert ArchiveStore(db).get("a.tar")["state"] == "truncated"
    readers = _planned(monkeypatch, {1: _back_half})
    out = tmp_path / "last.bin"

    assert _cat(db, "run/last.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


def test_cat_checks_every_request_when_a_member_spans_small_parts(tmp_path, monkeypatch):
    """A read over three parts is three requests, and the middle one touches neither end
    of the read -- so no read may cross a part boundary, and a short read straddles each
    boundary to vouch for the reads on both sides."""
    payload = _random(5, 12_000)
    db, _ = _split_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                      ("run/data.bin", payload),
                                      ("run/after.bin", b"a" * 3_000)], part_size=4_096)
    other = bytes((i * 7 + 3) % 256 for i in range(256))
    readers = _planned(monkeypatch, {2: lambda r, at, n, t: (other * (n // 256 + 1))[:n]})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


def test_cat_checks_both_ends_of_a_read_that_meets_a_part_boundary(tmp_path, monkeypatch):
    """The observed failure's own shape -- real content from 777 bytes away -- on the
    part of the first read that lies past a boundary, where the member is zeros around
    the read's tail. Such a request used to have no head check at all, and its zero tail
    agreed with the next read's."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = bytearray(_random(20, 40_000))
    body[12_000:17_000] = bytes(5_000)
    body = bytes(body)
    data = build_tar([("run/before.bin", _random(21, 30_000)), ("run/data.bin", body),
                      ("run/after.bin", _random(22, 3_000))])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        start = t.getmember("run/data.bin").offset_data
    boundary = start - 512 + 5_000
    source = tmp_path / "parts"
    source.mkdir()
    (source / "a.tar.aa").write_bytes(data[:boundary])
    (source / "a.tar.ab").write_bytes(data[boundary:])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    def shifted(reader, at, length, true):
        return reader.raw(at - 777, length)

    readers = _planned(monkeypatch, {2: shifted})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == body


def test_cat_vouches_for_every_request_at_both_ends(tmp_path, monkeypatch):
    """The rule behind the boundary handling, checked directly: every request cat makes
    shares its first and its last byte with some other request -- or has them vouched for
    by the member's own headers, or by the block after the member."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    db, _ = _split_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                      ("run/data.bin", _random(23, 40_000)),
                                      ("run/after.bin", b"a" * 3_000)], part_size=10_000)
    member = _row(db, "run/data.bin")
    stop = member["data_offset"] + -(-member["size"] // 512) * 512 + 512
    readers = _planned(monkeypatch, {})
    assert _cat(db, "run/data.bin", tmp_path / "data.bin") == 0

    spans = [(at, at + length) for at, length in readers[0].log]
    assert len({lo for lo, _ in spans}) > 4, "the member should cross several parts"
    for k, (lo, hi) in enumerate(spans):
        others = spans[:k] + spans[k + 1:]
        head = lo < member["data_offset"] or any(a <= lo < b for a, b in others)
        tail = hi == stop or any(a <= hi - 1 < b for a, b in others)
        assert head and tail, f"request [{lo}, {hi}) is checked by nothing at one end"


def test_cat_refuses_a_pax_member_left_in_an_older_index(tmp_path, capsys):
    """Indexes built before pax was refused may still hold pax members, recorded as
    tarfile applied their extended headers. cat reads a member's whole header sequence
    now, so it sees the pax header itself -- and refuses, saying why."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        info = tarfile.TarInfo("run/" + "p" * 120 + ".bin")
        info.size = 700
        tf.addfile(info, io.BytesIO(b"q" * 700))
    data = buf.getvalue()
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, data, part_size=len(data))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        [info] = t.getmembers()
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    store = ArchiveStore(db)
    row = store.get("a.tar")
    segment = store.segments(row["id"])[0]["id"]
    end = info.offset_data + 1024
    store.commit_batch(row["id"], segment, [Member.from_tarinfo(info)], end)
    store.finish(row["id"], WalkResult("complete", end, 1, ""))
    capsys.readouterr()
    out = tmp_path / "p.bin"

    assert _cat(db, info.name, out) == 1
    assert "pax" in capsys.readouterr().err
    assert not out.exists()


def test_cat_refuses_a_member_the_archive_ends_inside(tmp_path, capsys):
    """A truncated archive's last member is indexed at its full size, but its bytes stop
    short. Extracting it must fail, not hand over a short file with exit 0."""
    payload = _random(24, 20_000)
    data = build_tar([("run/before.bin", b"b" * 3_000), ("run/cut.bin", payload)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        start = t.getmember("run/cut.bin").offset_data
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, data[:start + 10_000], part_size=start + 10_000)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    assert ArchiveStore(db).get("a.tar")["state"] == "truncated"
    capsys.readouterr()
    out = tmp_path / "cut.bin"

    assert _cat(db, "run/cut.bin", out) == 1
    assert "ends" in capsys.readouterr().err
    assert not out.exists()


def test_cat_reads_repetitive_data_twice_and_distinctive_data_once(tmp_path, monkeypatch):
    """What the ambiguity rule costs, pinned: a member of zeros is read twice over, one
    of random bytes once. (For a real 16 MiB read the price is 1.6 s per extra request.)"""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    costs = {}
    for name, payload in (("zeros", bytes(40_000)), ("random", _random(25, 40_000))):
        directory = tmp_path / name
        directory.mkdir()
        db, _ = _single_part_archive(directory, [("run/before.bin", b"b" * 3_000),
                                                 ("run/data.bin", payload),
                                                 ("run/after.bin", b"a" * 3_000)])
        readers = _planned(monkeypatch, {})
        out = directory / "data.bin"
        assert _cat(db, "run/data.bin", out) == 0
        assert out.read_bytes() == payload
        costs[name] = readers[0].calls
    assert costs == {"random": 3, "zeros": 6}, costs


@pytest.mark.parametrize("where", ["in the block after the member",
                                   "in the member's own headers"])
def test_cat_extracts_a_member_whose_anchor_block_a_part_boundary_splits(
        tmp_path, monkeypatch, where):
    """Parts need not be a multiple of 512 bytes (`split -b 1GB` is 10**9), so a
    boundary can fall inside a block. Inside the one after the member, or between a long
    name's header and the member's own, the anchor still reads the block whole -- it is
    checked by what it says, not by an overlap -- and a healthy member is extracted."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _random(26, 20_000)
    name = "run/" + "d" * 150 + "/data.bin"
    data = build_tar([("run/before.bin", b"b" * 3_000), (name, payload),
                      ("run/after.bin", b"a" * 3_000)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        info = t.getmember(name)
    end = info.offset_data + -(-len(payload) // 512) * 512
    cut = end + 200 if where.startswith("in the block") else info.offset + 700
    source = tmp_path / "parts"
    source.mkdir()
    (source / "a.tar.aa").write_bytes(data[:cut])
    (source / "a.tar.ab").write_bytes(data[cut:])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    out = tmp_path / "data.bin"

    assert _cat(db, name, out) == 0
    assert out.read_bytes() == payload


# ---- each safeguard, pinned: the shapes that only it catches ------------------------------

def _zeros_at(payload_len, start, stop, seed):
    body = bytearray(_random(seed, payload_len))
    body[start:stop] = bytes(stop - start)
    return bytes(body)


def test_cat_reads_twice_a_read_whose_head_only_zeros_vouch_for(tmp_path, monkeypatch):
    """The second read begins in zeros, so the first read agreeing with its head proves
    nothing -- and this bad read is zeros where its data begins and right from halfway
    on, so its tail agrees with the third read. Only reading it twice shows it wrong."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _zeros_at(40_000, 14_000, 18_000, 27)      # reads 1 and 2 meet in here
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])

    def front_half_zeros(reader, at, length, true):
        return bytes(length // 2) + true[length // 2:]

    readers = _planned(monkeypatch, {2: front_half_zeros})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


def test_cat_holds_a_re_read_to_its_own_head_too(tmp_path, monkeypatch):
    """The first read comes back wrong in its back half; two reads of the second agree
    with each other and not with it, so the first is read again -- and that re-read is
    wrong too, everywhere but the bytes it shares with the second. It must meet its own
    head, the member's headers, before it replaces anything."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _random(28, 40_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])

    def all_but_its_tail(reader, at, length, true):
        return b"x" * (length - 1_024) + true[length - 1_024:]

    readers = _planned(monkeypatch, {1: _back_half, 4: all_but_its_tail})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 2
    assert out.read_bytes() == payload


def test_cat_refuses_rather_than_take_a_thrice_served_body_its_neighbour_denies(
        tmp_path, monkeypatch, capsys):
    """If a server ever served the same wrong body three times running, even a lead of two
    would not be enough: here the second read's tail is zeros, so it is settled -- and
    every re-read brings the same wrong bytes. Its head, shared with the first read,
    still says otherwise, so nothing is written."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _zeros_at(40_000, 28_000, 33_000, 29)      # reads 2 and 3 meet in here
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])
    other = _row(db, "run/before.bin")

    def cached_wrong(reader, at, length, true):
        return reader.raw(other["hdr_offset"], length)

    readers = _planned(monkeypatch, {4: cached_wrong, 5: cached_wrong,
                                     6: cached_wrong})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 1
    assert readers[0].lied == 3
    assert "never settled" in capsys.readouterr().err
    assert not out.exists()


def test_cat_does_not_take_the_terminators_zeros_as_proof(tmp_path, monkeypatch):
    """The archive's last member ends on the terminator: zeros. A read wrong from halfway
    -- zeros from there on -- agrees with that as well as the true one does, so the last
    read is taken only once a second fetch agrees with it."""
    payload = _random(30, 5_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/last.bin", payload)])

    def zeros_from_halfway(reader, at, length, true):
        return true[:length // 2] + bytes(length - length // 2)

    readers = _planned(monkeypatch, {1: zeros_from_halfway})
    out = tmp_path / "last.bin"

    assert _cat(db, "run/last.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


def test_cat_rereads_zeros_where_the_index_says_a_header_follows(tmp_path, monkeypatch):
    """What follows a member that is not the last is another header. Zeros there are not
    what the index says, so the read is made again -- one request -- rather than merely
    confirmed by two more."""
    payload = _random(31, 5_000)
    db, _ = _single_part_archive(tmp_path, [("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])

    def zeros_from_halfway(reader, at, length, true):
        return true[:length // 2] + bytes(length - length // 2)

    readers = _planned(monkeypatch, {1: zeros_from_halfway})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert out.read_bytes() == payload
    assert readers[0].calls == 2


def test_cat_refuses_a_member_whose_data_moved_under_the_same_header(tmp_path, capsys):
    """GNU tar gives a 100-character name a long-name block; Python does not. A part
    re-written by the other tool keeps every header field the same and moves the data by
    two blocks -- which only the check on where the data begins can see."""
    name = "run/" + "n" * 96                                 # exactly 100 characters
    assert len(name) == 100
    payload = _random(32, 1_000)
    tail = [("run/next.bin", b"x" * 400), ("run/after.bin", b"a" * 3_000)]
    plain = build_tar([(name, payload)] + tail)             # what Python writes
    longlink = tarfile.TarInfo("././@LongLink")
    longlink.type, longlink.size = tarfile.GNUTYPE_LONGNAME, len(name) + 1
    gnu = (longlink.tobuf(tarfile.GNU_FORMAT)
           + (name.encode() + b"\0").ljust(512, b"\0") + plain)   # what GNU tar writes
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, gnu, part_size=len(gnu))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    assert _row(db, name)["data_offset"] == 1536
    [part] = sorted(source.iterdir())
    part.write_bytes(plain + bytes(len(gnu) - len(plain)))   # same size, as a re-write
    capsys.readouterr()
    out = tmp_path / "n.bin"

    assert _cat(db, name, out) == 1
    assert "data begins" in capsys.readouterr().err
    assert not out.exists()


def _cat_across_parts(tmp_path, monkeypatch, part_size):
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    payload = _random(33, 40_000)
    db, _ = _split_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                      ("run/data.bin", payload),
                                      ("run/after.bin", b"a" * 3_000)], part_size=part_size)
    readers = _planned(monkeypatch, {})
    out = tmp_path / "data.bin"
    assert _cat(db, "run/data.bin", out) == 0
    assert out.read_bytes() == payload
    return readers[0].log


def test_cat_reads_each_part_once_when_the_reads_vouch_for_each_other(
        tmp_path, monkeypatch):
    """Random data across 10,000-byte parts: the member's span [3584, 45056) touches five
    parts, read once each, and a short read straddles each of the four boundaries -- two
    requests apiece, one either side. Every read shares distinctive bytes with its
    neighbours, so none is read twice: 5 + 4 x 2 = 13 requests."""
    log = _cat_across_parts(tmp_path, monkeypatch, 10_000)
    assert len(log) == 13
    assert len(set(log)) == len(log), "a read was confirmed twice over"


def test_cat_does_not_let_a_few_shared_bytes_vouch_for_a_read(tmp_path, monkeypatch):
    """Consecutive reads share 4 KiB; shared bytes much fewer -- 64 here -- are too few
    to go on: a read right at both ends and wrong in between agrees with both its
    neighbours. Under 512 bytes, a read is taken only once a second fetch agrees."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 64)
    payload = _random(34, 40_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", b"b" * 3_000),
                                            ("run/data.bin", payload),
                                            ("run/after.bin", b"a" * 3_000)])

    def wrong_between_its_ends(reader, at, length, true):
        return true[:64] + b"x" * (length - 128) + true[length - 64:]

    readers = _planned(monkeypatch, {2: wrong_between_its_ends})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == payload


# ---- review 5: the block after the member must be the member the index says follows ----

TWINS = [("run/r01/out.bin", _random(40, 5_000)), ("run/r01/next.bin", b"n" * 100),
         ("run/r02/out.bin", _random(41, 5_000)), ("run/r02/next.bin", b"m" * 100)]


def _twins_back_half(me, twin):
    """A read crossed halfway with the same range of an equal-size twin: its second half
    is the twin's data, then the header after the twin -- a valid header, not this one's."""
    shift = twin["hdr_offset"] - me["hdr_offset"]

    def lie(reader, at, length, true):
        half = length // 2
        return true[:half] + reader.raw(at + shift + half, length - half)

    return lie


def test_cat_is_not_fooled_by_a_twins_back_half(tmp_path, monkeypatch):
    """Two realisations' out.bin: same size, same header layout. Half of this read is the
    twin's, ending on the header after the twin -- valid, so a check for "some header"
    passed it. What follows this member is run/r01/next.bin, and the index says so."""
    db, _ = _single_part_archive(tmp_path, TWINS)
    me, twin = _row(db, "run/r01/out.bin"), _row(db, "run/r02/out.bin")
    readers = _planned(monkeypatch, {1: _twins_back_half(me, twin)})
    out = tmp_path / "out.bin"

    assert _cat(db, "run/r01/out.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == TWINS[0][1]


def test_cat_is_not_fooled_by_a_twins_back_half_on_the_last_of_several_reads(
        tmp_path, monkeypatch):
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 2_048)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 512)
    db, _ = _single_part_archive(tmp_path, TWINS)
    me, twin = _row(db, "run/r01/out.bin"), _row(db, "run/r02/out.bin")
    clean = _planned(monkeypatch, {})
    assert _cat(db, "run/r01/out.bin", tmp_path / "clean.bin") == 0
    last = clean[0].calls
    readers = _planned(monkeypatch, {last: _twins_back_half(me, twin)})
    out = tmp_path / "out.bin"

    assert _cat(db, "run/r01/out.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == TWINS[0][1]


def test_cat_on_a_partial_index_does_not_take_any_header_after_the_member_on_trust(
        tmp_path, monkeypatch):
    """A walk stopped early knows nothing past its last member, so what follows the
    member it last recorded could be anything -- and so it vouches for nothing: the read
    is taken once a second fetch agrees with it."""
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(TWINS), part_size=len(build_tar(TWINS)))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    # One-block windows: members leave the held-back window, and are committed, a few
    # headers behind the walk -- so one batch of one is the first member alone.
    main(["archive", "index", "--db", db, "--archive", "a.tar", "--workers", "1",
          "--batch", "1", "--max-batches", "1", "--window-min", "1024",
          "--window-max", "1024"])
    store = ArchiveStore(db)
    assert store.get("a.tar")["state"] == "walking"
    assert store.member_at(store.get("a.tar")["id"], _row(db, "run/r01/out.bin")["data_offset"]
                           + 5_120) is None, "the member after out.bin is not indexed yet"
    raw = build_tar(TWINS)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as t:
        twin_offset = t.getmember("run/r02/out.bin").offset
    me = _row(db, "run/r01/out.bin")
    readers = _planned(monkeypatch, {1: _twins_back_half(me, {"hdr_offset": twin_offset})})
    out = tmp_path / "out.bin"

    assert _cat(db, "run/r01/out.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == TWINS[0][1]


def test_cat_reads_a_long_named_successors_headers_whole_wherever_a_read_ends(
        tmp_path, monkeypatch):
    """A member followed by one with a GNU long name: what must follow it is three
    blocks, longer than two reads share. A read ending inside those blocks must not leave
    the last read starting past their start -- the member here is healthy, and was
    refused."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 512)
    payload = _random(42, 14_848)                 # data [512, 15360): a read ends at 16384
    db, _ = _single_part_archive(tmp_path, [("run/data.bin", payload),
                                            ("run/" + "l" * 150 + "/next.bin", b"n" * 100)])
    member = _row(db, "run/data.bin")
    assert (member["hdr_offset"], member["data_offset"] + 14_848) == (0, 15_360)
    readers = _planned(monkeypatch, {})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert out.read_bytes() == payload
    # One request, [0, 16896): the member, then its successor's three header blocks,
    # which vouch for the read's tail -- so it is not read a second time.
    assert readers[0].log == [(0, 16_896)]


def test_cat_does_not_let_a_few_zero_bytes_of_a_header_vouch_for_a_request(
        tmp_path, monkeypatch):
    """A part boundary 48 bytes before a long-named member's data: the request after it
    holds the end of the member's header -- zeros -- and then data. The anchor checks
    those 48 bytes, but zeros agree with zeros: a bad read that is zeros there, and wrong
    after, passes the anchor. So that read is taken only once a second fetch agrees."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    name = "run/" + "d" * 150 + "/data.bin"
    payload = _random(43, 12_000)
    data = build_tar([("run/before.bin", b"b" * 3_000), (name, payload),
                      ("run/after.bin", b"a" * 3_000)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        start = t.getmember(name).offset_data
    source = tmp_path / "parts"
    source.mkdir()
    cuts = [0, start - 48, start + 6_000, len(data)]
    for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
        (source / f"a.tar.a{'abc'[k]}").write_bytes(data[a:b])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    def zeros_then_true(reader, at, length, true):
        half = length // 2
        return bytes(half) + true[half:]

    readers = _planned(monkeypatch, {2: zeros_then_true})
    out = tmp_path / "data.bin"

    assert _cat(db, name, out) == 0
    assert readers[0].lied == 1
    assert readers[0].log[1][0] == start - 48, "the lie is on the request after the boundary"
    assert out.read_bytes() == payload


def test_cat_does_not_let_a_few_bytes_of_the_next_header_vouch_for_a_request(
        tmp_path, monkeypatch):
    """The mirror image: a part boundary 200 bytes into the header after the member. The
    request before it ends on those 200 bytes -- too few to vouch for it -- so a read
    right at its head and right in those last 200 bytes, wrong in between, is caught
    only by a second fetch."""
    payload = _random(44, 5_000)
    data = build_tar([("run/data.bin", payload), ("run/next.bin", b"n" * 100),
                      ("run/after.bin", b"a" * 1_000)])
    end = 512 + 5_120
    source = tmp_path / "parts"
    source.mkdir()
    (source / "a.tar.aa").write_bytes(data[:end + 200])
    (source / "a.tar.ab").write_bytes(data[end + 200:])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])

    def right_at_both_ends(reader, at, length, true):
        return true[:600] + b"x" * (length - 800) + true[length - 200:]

    readers = _planned(monkeypatch, {1: right_at_both_ends})
    out = tmp_path / "data.bin"

    assert _cat(db, "run/data.bin", out) == 0
    assert readers[0].lied == 1
    assert readers[0].log[0] == (0, end + 200), "the lie is on the request that ends there"
    assert out.read_bytes() == payload


def _split_at(tmp_path, data, cuts):
    source = tmp_path / "parts"
    source.mkdir()
    edges = [0, *cuts, len(data)]
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        (source / f"a.tar.a{'abcdefghij'[k]}").write_bytes(data[a:b])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    return db


def test_cat_repairs_a_twins_body_behind_five_bytes_of_header_padding(tmp_path, monkeypatch):
    """Review 5's T3, real constants: a part boundary 5 bytes before the member's data,
    and the request after it replaced by the same shape of range around an equal-size
    twin -- whose first 5 bytes are header padding like these, and whose tail lands on
    the twin's successor. Five zero bytes vouch for nothing; the read is fetched again."""
    mine, theirs = _random(30, 1_300), _random(31, 1_400)
    data = build_tar([("run/r01/x.bin", mine), ("run/r01/y.bin", b"y" * 50),
                      ("run/r02/x.bin", theirs), ("run/r02/y.bin", b"z" * 50)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        d1 = t.getmember("run/r01/x.bin").offset_data
        d2 = t.getmember("run/r02/x.bin").offset_data
    db = _split_at(tmp_path, data, [d1 - 5])
    readers = _planned(monkeypatch, {2: lambda r, at, n, t: r.raw(d2 - 5, n)})
    out = tmp_path / "x.bin"

    assert _cat(db, "run/r01/x.bin", out) == 0
    assert readers[0].lied == 1
    assert readers[0].log[1][0] == d1 - 5
    assert out.read_bytes() == mine


def test_cat_repairs_a_twins_body_that_ends_one_byte_into_the_next_header(
        tmp_path, monkeypatch):
    """Review 5's T4: a boundary one byte past the member's padded end, so the request
    before it ends on a single byte of the next header -- 'r', as in every sibling's
    "run/...". A body right at its head and a twin's after it passes that one byte.
    One byte vouches for nothing; the read is fetched again."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    a, b = _random(40, 40_000), _random(41, 40_000)
    data = build_tar([("run/before.bin", _random(42, 3_000)), ("run/r01/m.bin", a),
                      ("run/r01/n.bin", _random(43, 900)), ("run/r02/m.bin", b),
                      ("run/r02/n.bin", _random(44, 900))])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        i1, i2 = t.getmember("run/r01/m.bin"), t.getmember("run/r02/m.bin")
    end1 = i1.offset_data + 40_448
    shift = i2.offset - i1.offset
    db = _split_at(tmp_path, data, [end1 + 1])

    def splice(reader, at, length, true):
        return true[:2_048] + reader.raw(at + shift + 2_048, length - 2_048)

    readers = _planned(monkeypatch, {3: splice})
    out = tmp_path / "m.bin"

    assert _cat(db, "run/r01/m.bin", out) == 0
    assert readers[0].lied == 1
    assert sum(readers[0].log[2]) == end1 + 1, "the lie is on the request ending there"
    assert out.read_bytes() == a


# ---- review 5: a wrong body served twice, and rows no confirmed chain vouches for -------

class _ServesTwice(LocalRangeReader):
    """Truthful, except that the 2nd and 3rd fetches of one exact range both return the
    same wrong body -- a server that, once, got the same range wrong twice running."""

    def __init__(self, directory, archive, target, wrong):
        super().__init__(directory, archive)
        self.target, self.wrong, self.seen, self.lied = target, wrong, 0, 0

    def read_range(self, part_idx, offset, length):
        true = super().read_range(part_idx, offset, length)
        if (self.archive.parts[part_idx].offset + offset, length) != self.target:
            return true
        self.seen += 1
        if self.seen not in (2, 3):
            return true
        self.lied += 1
        return self.wrong(true)


def test_cat_does_not_let_a_body_served_twice_outvote_a_right_one(tmp_path, monkeypatch):
    """Review 5's T2: zeros at both of the middle read's overlaps, data only inside it.
    Its first fetch is right, but nothing distinctive vouches for it, so it is fetched
    again -- and the next two fetches bring the same wrong body. Two against one used to
    win. Once fetches have disagreed, a version must lead every other by two."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = bytearray(40_000)
    body[20_000:24_000] = _random(5, 4_000)
    body = bytes(body)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", _random(6, 3_000)),
                                            ("run/m.bin", body),
                                            ("run/after.bin", _random(7, 3_000))])
    wrong = _random(8, 16_384)
    readers = []
    _with_reader(monkeypatch, lambda d, a: readers.append(_ServesTwice(
        d, a, (18_944, 16_384), lambda t: t[:4_000] + wrong[4_000:12_000] + t[12_000:]))
        or readers[-1])
    out = tmp_path / "m.bin"

    assert _cat(db, "run/m.bin", out) == 0
    assert readers[0].lied == 2, "the test did not serve the wrong body twice"
    assert out.read_bytes() == body


def test_cat_refuses_a_row_no_confirmed_chain_vouches_for(tmp_path, capsys):
    """Review 5's T5: a walk stopped early can hold rows from a chain whose start nothing
    has confirmed -- here one that began inside a stored tarball and walked its members.
    Its run/x.bin is the tarball's old copy, not the archive's. `cat` refuses such a row
    and says how to finish the index, rather than extract it with exit 0."""
    import argparse
    import threading
    import dbaudit.cli as cli

    old, new = _random(50, 9_000), _random(51, 9_000)
    inner = build_tar([("run/x.bin", old), ("run/y.bin", _random(52, 4_000))])
    data = build_tar([("run/a.bin", _random(53, 20_000)), ("backup/run.tar", inner),
                      ("run/x.bin", new)])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        start = t.getmember("backup/run.tar").offset_data
    source = tmp_path / "parts"
    source.mkdir()
    edges = [0, start - 100, len(data) - 5_000, len(data)]
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        (source / f"a.tar.a{'abc'[k]}").write_bytes(data[a:b])
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    store = ArchiveStore(db)
    row = store.get("a.tar")
    args = argparse.Namespace(window_min=65_536, window_max=16 << 20, batch=2_000,
                              max_batches=None)
    run = cli._Run(args, store, row, cli._stored_archive_set(store, row), None, None,
                   threading.Event())
    segment = store.claim_segment(row["id"], "test")           # segment 0 ...
    store.release_segment(segment["id"])
    store.connect().execute("UPDATE segments SET state='walking' WHERE id=?",
                            (segment["id"],))                  # ... held elsewhere
    claimed = store.claim_segment(row["id"], "test")
    assert claimed["idx"] == 1
    cli._walk_segment(run, claimed)
    [hit] = store.find_members(row["id"], "run/x.bin")
    assert hit["hdr_offset"] >= start, "the row is the stored tarball's copy"
    capsys.readouterr()
    out = tmp_path / "x.bin"

    assert _cat(db, "run/x.bin", out) == 1
    assert "not yet confirmed" in capsys.readouterr().err
    assert not out.exists()


# ---- review 5's mutant-killers: each pins a check no earlier test could see --------------

def _clean_log(monkeypatch, db, path):
    """The requests a clean `cat` of ``path`` makes, to aim a bad read at one of them."""
    readers = _planned(monkeypatch, {})
    assert _cat(db, path, Path(db).parent / "clean.bin") == 0
    return readers[0].log


def test_cat_bridges_a_part_boundary_exactly_at_the_members_end(tmp_path, monkeypatch):
    """`split -b 300G` boundaries are 512-aligned, so a part can end exactly where a
    member does. The read ending there still has its tail checked by a bridge: a body
    right at its head and wrong from halfway is repaired, not written."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = _random(1, 30_000)
    data = build_tar([("run/before.bin", _random(2, 3_000)), ("run/m.bin", body),
                      ("run/after.bin", _random(3, 3_000))])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        info = t.getmember("run/m.bin")
    end = info.offset_data + -(-info.size // 512) * 512
    db = _split_at(tmp_path, data, [end])
    log = _clean_log(monkeypatch, db, "run/m.bin")
    ending_there = max(i for i, (at, n) in enumerate(log) if at + n == end) + 1
    readers = _planned(monkeypatch, {ending_there: _back_half})
    out = tmp_path / "m.bin"

    assert _cat(db, "run/m.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == body


def test_cat_expects_the_terminator_after_the_last_member(tmp_path, monkeypatch):
    """The archive's last member is followed by the terminator. A last read whose tail
    is a valid header -- spliced from a same-shape member's range -- must not stand."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    twin, last = _random(4, 20_000), _random(5, 20_000)
    db, _ = _single_part_archive(tmp_path, [("run/r01/m.bin", twin),
                                            ("run/r01/n.bin", _random(6, 700)),
                                            ("run/r02/m.bin", last)])
    m1, m2 = _row(db, "run/r01/m.bin"), _row(db, "run/r02/m.bin")
    stop = m2["data_offset"] + -(-m2["size"] // 512) * 512 + 512
    log = _clean_log(monkeypatch, db, "run/r02/m.bin")
    last_read = [i for i, (at, n) in enumerate(log) if at + n == stop][0] + 1
    shift = m1["hdr_offset"] - m2["hdr_offset"]

    def splice(reader, at, length, true):
        return true[:2_048] + reader.raw(at + shift + 2_048, length - 2_048)

    readers = _planned(monkeypatch, {last_read: splice})
    out = tmp_path / "m.bin"

    assert _cat(db, "run/r02/m.bin", out) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == last


def test_cat_tells_two_copies_of_a_path_apart_by_owner_and_mode(tmp_path, monkeypatch):
    """`tar -r` after a chown: the same path, size and mtime twice, owners and modes
    differ. A body from the other copy's range is caught by the header anchor."""
    a, b = _random(7, 1_359), _random(8, 1_359)
    db, _ = _single_part_archive(tmp_path, [
        ("run/p.yaml", a, {"uname": "alice", "gname": "g1", "mode": 0o644}),
        ("run/q.bin", _random(9, 600)),
        ("run/p.yaml", b, {"uname": "bob", "gname": "g2", "mode": 0o600}),
        ("run/z.bin", _random(10, 600))])
    store = ArchiveStore(db)
    first, second = store.find_members(store.get("a.tar")["id"], "run/p.yaml")
    shift = second["hdr_offset"] - first["hdr_offset"]
    readers = _planned(monkeypatch, {1: lambda r, at, n, t: r.raw(at + shift, n)})
    out = tmp_path / "p.yaml"

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar", "--member",
                 "run/p.yaml", "--offset", str(first["hdr_offset"]),
                 "--out", str(out)]) == 0
    assert readers[0].lied == 1
    assert out.read_bytes() == a


@pytest.mark.parametrize("part", [300, 511, 700, 1_000])
def test_cat_extracts_every_member_across_parts_smaller_than_the_overlap(
        tmp_path, monkeypatch, part):
    """A bridge across a boundary must stop at the next one: parts smaller than the
    bytes two reads share are otherwise straddled twice by one bridge."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    members = [(f"run/f{i}.bin", _random(20 + i, s))
               for i, s in enumerate([0, 1, 513, 3_000, 9_000, 20_000])]
    data = build_tar(members)
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, data, part_size=part)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    for name, payload in members:
        out = tmp_path / "out.bin"
        assert _cat(db, name, out) == 0, name
        assert out.read_bytes() == payload, name


def test_cat_holds_a_re_read_last_read_to_its_own_head(tmp_path, monkeypatch):
    """The last read's tail fails the check after it; its re-read comes back wrong at its
    head -- a second bad read. That re-read must not replace it unless its head agrees
    with the read before."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = _random(11, 40_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", _random(12, 3_000)),
                                            ("run/m.bin", body),
                                            ("run/after.bin", _random(13, 3_000))])
    last = len(_clean_log(monkeypatch, db, "run/m.bin"))
    readers = _planned(monkeypatch, {
        last: lambda r, at, n, t: t[:-600] + _random(14, 600),
        last + 1: lambda r, at, n, t: _random(15, 2_000) + t[2_000:]})
    out = tmp_path / "m.bin"

    assert _cat(db, "run/m.bin", out) == 0
    assert readers[0].lied == 2
    assert out.read_bytes() == body


def test_cat_checks_a_settled_last_read_against_the_read_before_it(tmp_path, monkeypatch):
    """The archive's last member ends on the terminator's zeros, so its last read is
    settled. A wrong body served three times running wins that vote -- and must still
    agree with the read before it. Here it does not, so nothing is written."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    body = _random(16, 40_000)
    db, _ = _single_part_archive(tmp_path, [("run/before.bin", _random(17, 3_000)),
                                            ("run/last.bin", body)])
    m = _row(db, "run/last.bin")
    stop = m["data_offset"] + -(-m["size"] // 512) * 512 + 512
    log = _clean_log(monkeypatch, db, "run/last.bin")
    first_last = [i for i, (at, n) in enumerate(log) if at + n == stop][0] + 1
    wrong = {}

    def served_thrice(reader, at, length, true):
        wrong.setdefault("body", _random(18, length - 1_024) + true[length - 1_024:])
        return wrong["body"]

    readers = _planned(monkeypatch, {first_last + k: served_thrice for k in (1, 2, 3)})
    out = tmp_path / "last.bin"

    assert _cat(db, "run/last.bin", out) == 1
    assert readers[0].lied == 3
    assert not out.exists()


def test_cat_rereads_a_header_where_the_terminator_belongs_even_served_thrice(
        tmp_path, monkeypatch):
    """After the archive's last member come the terminator's zeros, so a header there is
    wrong on its face and the read is made again. Were it only confirmed by vote, the
    same spliced body served three times running would win it."""
    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 16_384)
    monkeypatch.setattr("dbaudit.cli.CAT_OVERLAP", 1_024)
    twin, last = _random(4, 20_000), _random(5, 20_000)
    db, _ = _single_part_archive(tmp_path, [("run/r01/m.bin", twin),
                                            ("run/r01/n.bin", _random(6, 700)),
                                            ("run/r02/m.bin", last)])
    m1, m2 = _row(db, "run/r01/m.bin"), _row(db, "run/r02/m.bin")
    stop = m2["data_offset"] + -(-m2["size"] // 512) * 512 + 512
    log = _clean_log(monkeypatch, db, "run/r02/m.bin")
    first = [i for i, (at, n) in enumerate(log) if at + n == stop][0] + 1
    shift = m1["hdr_offset"] - m2["hdr_offset"]

    def splice(reader, at, length, true):
        return true[:2_048] + reader.raw(at + shift + 2_048, length - 2_048)

    readers = _planned(monkeypatch, {first + k: splice for k in (0, 1, 2)})
    out = tmp_path / "m.bin"

    assert _cat(db, "run/r02/m.bin", out) == 0
    assert readers[0].lied == 3
    assert out.read_bytes() == last
