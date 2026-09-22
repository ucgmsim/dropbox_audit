"""The `archive` commands end to end, over real tars split into real part files.

Nothing here reaches Dropbox. A "Dropbox" archive is one registered through a
monkeypatched `build_lister`, and where such a test has to read bytes it substitutes a
`LocalRangeReader` through `_reader_for` -- which is exactly the seam Task 8 reuses.
"""

import contextlib
import tarfile
import io
import logging
import os
import signal
import threading

import pytest

from dbaudit.api import Page
from dbaudit.archive.reader import LocalRangeReader
from dbaudit.archive.store import ArchiveStore
from dbaudit.archive.tarwalk import REREAD_LIMIT, find_chain_start
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


def event_count(db, archive_id, kind):
    return ArchiveStore(db).connect().execute(
        "SELECT COUNT(*) FROM events WHERE archive_id=? AND kind=?",
        (archive_id, kind)).fetchone()[0]


def run_starts(db, archive_id):
    """How many walks this archive has begun, from the events `start_walk` writes."""
    return event_count(db, archive_id, "run_start")


@contextlib.contextmanager
def signal_handled():
    """Yields an Event set once the CLI's signal handler has logged.

    The handler sets its stop flag *before* logging, so the line arriving means the walk
    is already winding down -- which is what lets a test send SIGINT from a worker thread
    and know when the main thread has acted on it, instead of sleeping on a guess.
    """
    handled = threading.Event()

    class Watch(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("signal "):
                handled.set()

    watch = Watch()
    logging.getLogger("dbaudit").addHandler(watch)
    try:
        yield handled
    finally:
        logging.getLogger("dbaudit").removeHandler(watch)


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
    assert "429s" in out          # T7-16: throttling is the first thing to look at

    store = ArchiveStore(db)
    archive_id = store.get("a.tar")["id"]
    # Task 11's acceptance check, and the one behaviour an operator is asked to verify:
    # every segment on the chain had its start confirmed by the chain before it. The
    # trailing padding parts are `beyond` -- they were never on the chain.
    on_chain = [s for s in store.segments(archive_id) if s["state"] != "beyond"]
    assert len(on_chain) == 12
    assert all(s["joined"] for s in on_chain)
    # A local run has no 429s to report, so it should not leave an event saying "0".
    assert event_count(db, archive_id, "rate_limited") == 0


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

    real_commit = ArchiveStore.commit_batch
    sent = threading.Event()
    before = signal.getsignal(signal.SIGINT)
    with signal_handled() as handled:
        def commit_batch(self, *args, **kwargs):
            result = real_commit(self, *args, **kwargs)
            if not sent.is_set():
                sent.set()
                os.kill(os.getpid(), signal.SIGINT)
                handled.wait(10.0)
            return result

        monkeypatch.setattr(ArchiveStore, "commit_batch", commit_batch)
        assert main(["archive", "index", "--db", db, "--archive", "a.tar",
                     "--workers", "1", "--batch", "5"]) == 0
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


def test_a_stop_between_claim_and_scan_never_starts_the_scan(tmp_path, monkeypatch):
    """The scan for a chain's first header is deliberately unbounded -- Task 1 measured
    seeds that found nothing within 64 MiB -- so a worker that claimed a segment a moment
    before the stop must hand it straight back rather than begin one.
    """
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    scans = {"n": 0}
    real_scan = find_chain_start

    def counting_scan(*args, **kwargs):
        scans["n"] += 1
        return real_scan(*args, **kwargs)

    monkeypatch.setattr("dbaudit.cli.find_chain_start", counting_scan)
    real_claim = ArchiveStore.claim_segment

    with signal_handled() as handled:
        def claim_segment(self, archive_id, owner):
            segment = real_claim(self, archive_id, owner)
            # Segment 0 starts at offset 0 and needs no scan. Stop the run exactly as a
            # segment that *would* scan is handed to a worker.
            if (segment is not None and segment["first_header"] is None
                    and not handled.is_set()):
                os.kill(os.getpid(), signal.SIGINT)
                handled.wait(10.0)
            return segment

        monkeypatch.setattr(ArchiveStore, "claim_segment", claim_segment)
        assert main(["archive", "index", "--db", db, "--archive", "a.tar",
                     "--workers", "1"]) == 0

    assert handled.is_set()
    assert scans["n"] == 0


def test_a_chain_that_dies_is_still_charged_for_what_it_read(tmp_path, monkeypatch):
    """One bad part must not kill the run -- but the reads it made before it died are
    real, and every other exit path charges them."""
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    reads = {"n": 0}
    real_read = LocalRangeReader.read_range

    def read_range(self, part_idx, offset, length):
        reads["n"] += 1
        if reads["n"] == 3:
            raise RuntimeError("the part went away mid-read")
        return real_read(self, part_idx, offset, length)

    monkeypatch.setattr(LocalRangeReader, "read_range", read_range)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar", "--workers", "1",
                 "--window-min", "1024", "--window-max", "4096"]) == 1

    row = ArchiveStore(db).get("a.tar")
    assert reads["n"] > 3                       # the dead chain was not the only one
    assert row["requests"] == reads["n"] - 1    # every read but the one that raised


def test_a_worker_that_blows_up_does_not_look_like_a_clean_stop(tmp_path, monkeypatch):
    """On a nohup'd run the exit code is most of what anyone sees, so a chain dying of
    something unexpected must not report the same 0 as Ctrl-C."""
    source = local_archive(tmp_path)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])

    def boom(self, archive_id, owner):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(ArchiveStore, "claim_segment", boom)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar",
                 "--workers", "1"]) == 1


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


# ---- a bad read through the real pool --------------------------------------------------

class _SharedLies:
    """One schedule of bad reads for every reader the pool builds, counted across all
    readers and threads -- so a lie can land on whichever chain, and whichever pass,
    happens to read there.

    `plan` maps an offset to the covering fetch that lies about it. `sequence` instead
    lies about each offset in turn, once, at most one per fetch: successive transients,
    each met only after the walk has survived the one before. (With `plan`, one window
    covering several offsets would carry every lie in a single fetch.)"""

    def __init__(self, source, plan=None, sequence=None):
        self.source, self.plan = source, dict(plan or {})
        self.sequence = list(sequence or [])
        self.covering = {bad: 0 for bad in self.plan}
        self.lied = 0
        self.lock = threading.Lock()

    def reader_for(self, row, archive_set, tokens=None, limiter=None):
        shared = self

        class _Reader(LocalRangeReader):
            def read_range(self, part_idx, offset, length):
                data = super().read_range(part_idx, offset, length)
                start = self.archive.parts[part_idx].offset + offset
                with shared.lock:
                    if shared.sequence and start <= shared.sequence[0] < start + length:
                        bad = shared.sequence.pop(0)
                        shared.lied += 1
                        cut = bad - start
                        n = min(1024, length - cut)
                        return data[:cut] + (bytes(range(256)) * 4)[:n] + data[cut + n:]
                    for bad, nth in shared.plan.items():
                        if start <= bad < start + length:
                            shared.covering[bad] += 1
                            if shared.covering[bad] == nth:
                                shared.lied += 1
                                cut = bad - start
                                n = min(1024, length - cut)
                                data = data[:cut] + (bytes(range(256)) * 4)[:n] + data[cut + n:]
                return data

        return _Reader(self.source, archive_set)


def _terminator_of(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        last = t.getmembers()[-1]
    return last.offset_data + (-(-last.size // 512)) * 512


@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_one_bad_read_at_the_terminator_cannot_change_the_archive_verdict(
        tmp_path, monkeypatch, nth):
    """End to end, through the 8-worker pool, the join and the store: one bad read at the
    terminator -- on whichever fetch it lands, including the one that confirms `complete`
    -- must leave a complete archive complete, every member in place, exit 0. A lie on
    the confirming read used to turn the whole archive `corrupt`, and an honest re-run
    then refused to walk it again.

    One part, so one chain. With many, every other chain's cold scan reads to the end of
    the archive and covers the terminator first, and the lie never reaches the read that
    confirms the verdict -- which is how this test once passed against the broken code."""
    data = build_tar(MEMBERS)
    source = local_archive(tmp_path, data, part_size=len(data))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    liar = _SharedLies(source, {_terminator_of(data): nth})
    monkeypatch.setattr("dbaudit.cli._reader_for", liar.reader_for)

    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0

    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] == "complete", f"a lie on covering read #{nth} decided the verdict"
    assert store.stats(row["id"])["n_members"] == len(MEMBERS)
    if nth <= 3:
        assert liar.lied == 1, "the test did not actually inject a bad read"


def test_a_server_that_will_not_settle_leaves_a_retryable_error_not_a_verdict(
        tmp_path, monkeypatch):
    """More contradicted reads in one walk than REREAD_LIMIT allows: the segment must land
    in `error`, the archive must not be condemned, and an honest re-run must finish it.
    Believing the last read instead left the archive `corrupt`, and a re-run refused."""
    data = build_tar(MEMBERS)
    source = local_archive(tmp_path, data, part_size=len(data))    # one segment
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        offsets = [m.offset for m in t]
    liar = _SharedLies(source, sequence=offsets[10:10 + REREAD_LIMIT + 1])
    monkeypatch.setattr("dbaudit.cli._reader_for", liar.reader_for)

    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 1
    assert liar.lied == REREAD_LIMIT + 1, "the test did not inject every transient"
    store = ArchiveStore(db)
    row = store.get("a.tar")
    assert row["state"] not in ("corrupt", "complete", "truncated")
    [segment] = store.segments(row["id"])
    assert segment["state"] == "error"
    assert "UnsettledRead" in (segment["error"] or "")

    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            LocalRangeReader(source, archive_set))
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0
    row = store.get("a.tar")
    assert row["state"] == "complete"
    assert store.stats(row["id"])["n_members"] == len(MEMBERS)
