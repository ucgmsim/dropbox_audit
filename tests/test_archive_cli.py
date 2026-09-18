"""The `archive` commands end to end, over real tars split into real part files.

Nothing here reaches Dropbox. A "Dropbox" archive is one registered through a
monkeypatched `build_lister`, and where such a test has to read bytes it substitutes a
`LocalRangeReader` through `_reader_for` -- which is exactly the seam Task 8 reuses.
"""

import logging
import os
import signal
import threading

from dbaudit.api import Page
from dbaudit.archive.reader import LocalRangeReader
from dbaudit.archive.store import ArchiveStore
from dbaudit.cli import main
from dbaudit.lock import InstanceLock
from tests.archive_fakes import build_tar, write_parts

MEMBERS = [(f"run/file{i:02d}.bin", b"x" * (700 + i)) for i in range(30)]
# One member big enough to swallow whole parts: split at 4096, parts 1-4 hold no header
# at all, so those chains cross the moment they start and commit only a cursor.
SWALLOWED = [("run/a.bin", b"a" * 1000), ("run/big.bin", b"b" * 20000),
             ("run/c.bin", b"c" * 1000)]


def local_archive(tmp_path, data=None, part_size=4096, name="parts"):
    """Split a tar into part files and return the directory holding them."""
    source = tmp_path / name
    source.mkdir()
    write_parts(source, data if data is not None else build_tar(MEMBERS), part_size)
    return source


def dropbox_entries(archive_set, override=None):
    """The `files/list_folder` entries a folder of these parts would return."""
    entries = [{".tag": "file", "name": p.name, "size": p.size,
                "path_display": p.path_display, "id": p.dbx_id, "rev": p.rev,
                "content_hash": p.content_hash} for p in archive_set.parts]
    if override is not None:
        index, content_hash = override
        entries[index]["content_hash"] = content_hash
    return entries


def fake_build_lister(pages):
    """A `build_lister` whose nth listing returns `pages[n]` (the last one repeats)."""
    calls = {"n": 0}

    class Lister:
        def list_folder(self, path, recursive=False):
            entries = pages[min(calls["n"], len(pages) - 1)]
            calls["n"] += 1
            return Page(entries=entries, cursor=None, has_more=False)

        def continue_(self, cursor):
            raise AssertionError("these fakes never paginate")

    def build(remote, include_deleted=False):
        return object(), Lister()

    build.calls = calls
    return build


def run_starts(db, archive_id):
    """How many walks this archive has begun, from the events `start_walk` writes."""
    return ArchiveStore(db).connect().execute(
        "SELECT COUNT(*) FROM events WHERE archive_id=? AND kind='run_start'",
        (archive_id,)).fetchone()[0]


def has_analysis_indexes(db):
    return bool(ArchiveStore(db).connect().execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='members_dir'").fetchone())


def register_dropbox(monkeypatch, db, archive_set, folder="/d"):
    monkeypatch.setattr("dbaudit.cli.build_lister",
                        fake_build_lister([dropbox_entries(archive_set)]))
    assert main(["archive", "register", "--db", db, "--folder", folder,
                 "--name", "a.tar"]) == 0


# ---- the four from the brief -------------------------------------------------


def test_register_then_index_a_local_archive(tmp_path, capsys):
    source = local_archive(tmp_path)
    (source / "notes.txt").write_text("a real folder holds other files too")
    db = str(tmp_path / "archives.db")

    assert main(["archive", "register", "--db", db, "--local-dir", str(source),
                 "--name", "a.tar"]) == 0
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0
    assert main(["archive", "status", "--db", db]) == 0

    out = capsys.readouterr().out
    assert "complete" in out
    assert str(len(MEMBERS)) in out


def test_an_interrupted_index_resumes(tmp_path):
    """Kill the walk halfway, re-run, and the manifest must be identical."""
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source),
          "--name", "a.tar"])

    store = ArchiveStore(db)
    archive_id = store.get("a.tar")["id"]

    # Walk only the first two batches, as an interrupted run would have.
    main(["archive", "index", "--db", db, "--archive", "a.tar", "--batch", "5",
          "--max-batches", "2"])
    partial = store.stats(archive_id)["n_members"]
    assert 0 < partial < len(MEMBERS)

    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    assert store.stats(archive_id)["n_members"] == len(MEMBERS)
    assert store.get("a.tar")["state"] == "complete"


def test_parallel_chains_produce_the_same_manifest_as_one(tmp_path):
    """The whole point of segments: four chains must agree with one, member for member."""
    data = build_tar(MEMBERS)
    manifests = {}
    for workers in (1, 4):
        source = local_archive(tmp_path, data, name=f"parts{workers}")
        db = str(tmp_path / f"archives{workers}.db")
        main(["archive", "register", "--db", db, "--local-dir", str(source),
              "--name", "a.tar"])
        main(["archive", "index", "--db", db, "--archive", "a.tar",
              "--workers", str(workers)])
        store = ArchiveStore(db)
        archive_id = store.get("a.tar")["id"]
        assert store.get("a.tar")["state"] == "complete"
        manifests[workers] = [(r["hdr_offset"], r["dir"], r["name"], r["size"])
                              for r in store.query_members(archive_id)]
    assert manifests[1] == manifests[4]


def test_a_tar_inside_the_tar_at_a_boundary_is_repaired(tmp_path):
    """A scan starting mid-member can land on an inner tar's header. The join must see
    that the chains do not meet, drop that segment's members and re-walk it."""
    inner = build_tar([(f"inner/deep{i:02d}.bin", b"i" * 400) for i in range(6)])
    data = build_tar([("run/first.bin", b"a" * 5000), ("run/nested.tar", inner),
                      ("run/last.bin", b"z" * 5000)])
    # Split so a part begins inside the nested tar, among the inner headers.
    nested_at = data.find(b"inner/deep00.bin") - 512
    source = local_archive(tmp_path, data, part_size=(nested_at + 1024) // 512 * 512)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar", "--workers", "2"])
    store = ArchiveStore(db)
    names = [r["name"] for r in store.query_members(store.get("a.tar")["id"])]
    assert names == ["first.bin", "nested.tar", "last.bin"]
    assert store.get("a.tar")["state"] == "complete"


# ---- stopping a run ----------------------------------------------------------


def test_a_signal_commits_the_batch_in_flight_and_leaves_the_walk_resumable(
        tmp_path, monkeypatch):
    """SIGINT mid-walk: exit 0, keep what was committed, hand the segment back, and
    put pytest's own SIGINT handler back where it was."""
    data = build_tar(MEMBERS)
    source = local_archive(tmp_path, data, part_size=len(data))   # one long segment
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    handled = threading.Event()

    class Watch(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("signal "):
                handled.set()

    watch = Watch()
    logging.getLogger("dbaudit").addHandler(watch)
    real_commit = ArchiveStore.commit_batch
    sent = threading.Event()

    def commit_batch(self, *args, **kwargs):
        result = real_commit(self, *args, **kwargs)
        if not sent.is_set():
            sent.set()
            os.kill(os.getpid(), signal.SIGINT)
            # The handler runs on the main thread; the log line it writes is the
            # proof that the stop flag is already set.
            handled.wait(10.0)
        return result

    monkeypatch.setattr(ArchiveStore, "commit_batch", commit_batch)
    before = signal.getsignal(signal.SIGINT)
    try:
        assert main(["archive", "index", "--db", db, "--archive", "a.tar",
                     "--workers", "1", "--batch", "5"]) == 0
    finally:
        logging.getLogger("dbaudit").removeHandler(watch)
    monkeypatch.undo()

    assert handled.is_set()
    assert signal.getsignal(signal.SIGINT) is before

    store = ArchiveStore(db)
    row = store.get("a.tar")
    partial = store.stats(row["id"])["n_members"]
    assert 0 < partial < len(MEMBERS)
    assert row["state"] == "walking"
    segment = store.segments(row["id"])[0]
    assert segment["state"] == "pending"
    assert segment["cursor_offset"] is not None
    # Analysis indexes are built after a walk completes, never during one.
    assert not has_analysis_indexes(db)
    # A walk in progress is what `status` exists for: rate and ETA only have a value
    # to print while the archive is `walking`.
    assert main(["archive", "status", "--db", db, "--archive", "a.tar"]) == 0

    reads = []
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        reads.append(offset)
        return real_read(self, part_idx, offset, length)

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0
    assert store.stats(row["id"])["n_members"] == len(MEMBERS)
    assert store.get("a.tar")["state"] == "complete"
    assert has_analysis_indexes(db)
    # Resuming means resuming. Re-walking from the segment's start would reach the same
    # manifest -- the rows replace cleanly -- while quietly throwing away every hour the
    # interrupted run had already paid for. The first byte asked for says which it did.
    assert reads[0] == segment["cursor_offset"]


def test_max_batches_commits_exactly_that_many_batches(tmp_path, monkeypatch):
    """--max-batches N means N batches, not N plus whatever was in flight when the
    limit was reached."""
    data = build_tar(MEMBERS)
    source = local_archive(tmp_path, data, part_size=len(data))   # one long segment
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    reads = {"n": 0, "bytes": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        block = real_read(self, part_idx, offset, length)
        reads["n"] += 1
        reads["bytes"] += len(block)
        return block

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar", "--workers", "1",
                 "--batch", "5", "--max-batches", "2",
                 "--window-min", "1024", "--window-max", "4096"]) == 0
    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert store.stats(row["id"])["n_members"] == 10
    # The refused batch was still read before it was refused. Releasing a segment has to
    # charge those reads or an interrupted run -- the normal workflow -- undercounts.
    assert reads["n"] > 1
    assert row["requests"] == reads["n"]
    assert row["bytes_fetched"] == reads["bytes"]


def test_an_empty_crossing_commit_does_not_consume_max_batches(tmp_path):
    """A part lying wholly inside one member has no header of its own: its chain crosses
    at once, committing a cursor and nothing else. Charging that to the budget buys no
    members, which is the opposite of what a smoke test wants -- Task 11 reads
    `--batch 500 --max-batches 4` as 2,000 members.
    """
    source = local_archive(tmp_path, build_tar(SWALLOWED), part_size=4096)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    assert main(["archive", "index", "--db", db, "--archive", "a.tar",
                 "--workers", "1", "--batch", "5", "--max-batches", "2"]) == 0

    store = ArchiveStore(db)
    row = store.get("a.tar")
    empty_crossings = [s for s in store.segments(row["id"])
                       if s["state"] == "crossed" and s["members"] == 0]
    assert len(empty_crossings) == 4        # the parts swallowed by big.bin
    # Two batches carrying members: the one before those crossings and the one after.
    assert store.stats(row["id"])["n_members"] == len(SWALLOWED)
    assert row["state"] == "complete"


# ---- fingerprints ------------------------------------------------------------


def test_a_part_that_changed_before_the_walk_is_stale_and_nothing_is_read(
        tmp_path, monkeypatch):
    """The stored offsets describe bytes that no longer exist, so the walk must not
    read a single range."""
    source = tmp_path / "parts"
    source.mkdir()
    archive_set = write_parts(source, build_tar(MEMBERS), 4096)
    db = str(tmp_path / "archives.db")
    register_dropbox(monkeypatch, db, archive_set)

    monkeypatch.setattr("dbaudit.cli.build_lister",
                        fake_build_lister([dropbox_entries(archive_set, (1, "ff" * 32))]))

    def landmine(*args, **kwargs):
        raise AssertionError("a stale archive must not be read")

    monkeypatch.setattr("dbaudit.cli._reader_for", landmine)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 1

    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] == "stale"
    assert store.stats(row["id"])["n_members"] == 0
    # The landmine alone does not prove this: a chain that hit it would be recorded as
    # a failed segment and the *after* fingerprint would flag the archive stale anyway.
    # What says the check ran before the walk is that no walk was ever started.
    assert run_starts(db, row["id"]) == 0
    assert row["started_at"] is None


def test_a_part_that_changes_during_the_walk_is_stale_but_keeps_its_members(
        tmp_path, monkeypatch):
    """The spec's error table: the walk's results are kept, the index is flagged."""
    source = tmp_path / "parts"
    source.mkdir()
    archive_set = write_parts(source, build_tar(MEMBERS), 4096)
    db = str(tmp_path / "archives.db")
    register_dropbox(monkeypatch, db, archive_set)

    # The before-check sees what was registered; the after-check sees a changed part.
    monkeypatch.setattr("dbaudit.cli.build_lister", fake_build_lister(
        [dropbox_entries(archive_set), dropbox_entries(archive_set, (0, "ff" * 32))]))
    monkeypatch.setattr(
        "dbaudit.cli._reader_for",
        lambda row, archive_set, tokens=None, limiter=None:
            LocalRangeReader(source, archive_set))

    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 1
    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] == "stale"
    assert store.stats(row["id"])["n_members"] == len(MEMBERS)


# ---- outcomes ----------------------------------------------------------------


def test_indexing_a_finished_archive_reads_nothing(tmp_path):
    """Re-running `index` is how you resume, so resuming a finished archive is a no-op."""
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0

    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] == "complete"
    before = row["requests"]
    assert before > 0
    assert run_starts(db, row["id"]) == 1

    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0
    assert store.get("a.tar")["requests"] == before
    # An idle re-walk would also read nothing -- every segment is already finished, so
    # nothing is claimable. What says the archive was recognised as done is that the
    # second run never started a walk at all.
    assert run_starts(db, row["id"]) == 1


def test_the_reported_cost_is_the_delta_and_not_a_running_total(tmp_path, monkeypatch):
    """`commit_batch` adds what it is handed, so a chain reports what it has spent
    *since its last report*. Handing over the running total every time compounds it,
    and bytes_fetched / member_bytes is this project's headline number."""
    data = build_tar(MEMBERS)
    source = local_archive(tmp_path, data, part_size=len(data))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    reads = {"n": 0, "bytes": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        block = real_read(self, part_idx, offset, length)
        reads["n"] += 1
        reads["bytes"] += len(block)
        return block

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    # A window small enough to force many reads, and a batch small enough to force
    # many reports: with one of each, a running total and a delta are the same number.
    assert main(["archive", "index", "--db", db, "--archive", "a.tar", "--workers", "1",
                 "--batch", "5", "--window-min", "1024", "--window-max", "4096"]) == 0

    row = ArchiveStore(db).get("a.tar")
    assert reads["n"] > 1
    assert row["requests"] == reads["n"]
    assert row["bytes_fetched"] == reads["bytes"]


def test_a_missing_final_part_is_truncated_and_keeps_what_it_found(tmp_path):
    source = local_archive(tmp_path)
    for part in sorted(source.iterdir())[8:]:      # the tail of the tar never arrived
        part.unlink()
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 1
    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] == "truncated"
    assert 0 < store.stats(row["id"])["n_members"] < len(MEMBERS)


# ---- status and usage --------------------------------------------------------


def test_status_shows_every_segment_and_the_efficiency_ratio(tmp_path, capsys):
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    capsys.readouterr()

    assert main(["archive", "status", "--db", db]) == 0
    out = capsys.readouterr().out
    store = ArchiveStore(db)
    segments = store.segments(store.get("a.tar")["id"])
    assert len(segments) == 13
    for segment in segments:
        assert f"[{segment['idx']:2d}]" in out
    # The trailing padding part lies past where the chain ended.
    assert segments[-1]["state"] == "beyond"
    assert "beyond" in out
    assert "ratio" in out
    assert "a.tar" in out and "complete" in out


def test_usage_errors_exit_2_and_a_held_lock_exits_3(tmp_path, capsys):
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    capsys.readouterr()

    assert main(["archive", "index", "--db", db, "--archive", "nope.tar"]) == 2
    assert "nope.tar" in capsys.readouterr().err

    missing = str(tmp_path / "missing.db")
    assert main(["archive", "index", "--db", missing, "--archive", "a.tar"]) == 2
    assert main(["archive", "status", "--db", missing]) == 2
    capsys.readouterr()

    with InstanceLock(f"{db}.lock"):
        assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 3
    assert "held by" in capsys.readouterr().err
