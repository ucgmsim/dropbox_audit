"""`archive cat`: retrieve one member's bytes out of a tar archive, by offset, in as
few range reads as possible. Everything here is local or monkeypatched -- see
tests/test_live_archive.py for the one live check, gated behind DBAUDIT_LIVE=1.
"""

import sys
import tarfile

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


def test_cat_fetches_exactly_the_member_size_not_the_window(tmp_path, monkeypatch):
    """With the real CAT_CHUNK (16 MiB) a member this size is one request regardless
    of the window, so CAT_CHUNK is shrunk here to force several fills -- the only way
    the window's floor and doubling would diverge from an exact read at unit-test
    scale. Without T8-6's pass-through window this reads more than the member's own
    size; with it, `bytes_fetched` matches exactly.
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

    monkeypatch.setattr("dbaudit.cli.CAT_CHUNK", 8192)
    fetched = {"bytes": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        block = real_read(self, part_idx, offset, length)
        fetched["bytes"] += len(block)
        return block

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    out = tmp_path / "big.bin"
    assert main(["archive", "cat", "--db", db, "--archive", "a.tar",
                 "--member", "run/big.bin", "--out", str(out)]) == 0
    assert out.read_bytes() == payload
    assert fetched["bytes"] == len(payload)


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
