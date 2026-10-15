"""`archive run` end to end: a listing, a folder-by-folder fake of Dropbox, real tars.

The audit database is built the way a crawl builds it. Listing a folder goes through a
fake `build_lister` that answers from the same entries, and every read through a fake
`_reader_for` that serves the parts from local files -- the seams the other archive tests
use. Names are made up.
"""

import gzip
import logging
import os
import random
import signal
import sqlite3
import tarfile
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from dbaudit import cli
from dbaudit.api import RateLimited, TransientError
from dbaudit.archive.reader import LocalRangeReader, ReaderError
from dbaudit.archive.store import ArchiveStore
from dbaudit.auth import AuthError
from dbaudit.cli import LISTING_ATTEMPTS, _run_limiter, main
from dbaudit.limiter import AdaptiveLimiter
from dbaudit.lock import InstanceLock
from tests.archive_fakes import World, build_tar, members


def rows(db):
    store = ArchiveStore(db)
    return {r["folder"]: r for r in store.connect().execute("SELECT * FROM archives")}


def run(db, audit, *extra):
    return main(["archive", "run", "--db", db, "--audit-db", audit, *extra])


def three(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/a", "one.tar", build_tar(members(5, 1)))
    world.add("/R/b", "two.tar", build_tar(members(30, 2)), part_size=40_000)
    world.add("/R/c", "three.tar", build_tar(members(8, 3)))
    world.install(monkeypatch)
    return world, str(tmp_path / "archives.db"), world.audit_db()


def test_every_archive_the_listing_names_is_walked(tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    assert run(db, audit) == 0
    found = rows(db)
    assert {f: (r["state"], r["n_members"]) for f, r in found.items()} == {
        "/R/a": ("complete", 5), "/R/b": ("complete", 30), "/R/c": ("complete", 8)}
    assert found["/R/b"]["n_parts"] > 1 and found["/R/a"]["source"] == "dropbox"


def test_a_dry_run_says_what_it_would_walk_and_reads_nothing(tmp_path, monkeypatch, capsys):
    world, db, audit = three(tmp_path, monkeypatch)

    def landmine(*args, **kwargs):
        raise AssertionError("a dry run must not contact Dropbox")

    monkeypatch.setattr("dbaudit.cli.build_lister", landmine)
    monkeypatch.setattr("dbaudit.cli._reader_for", landmine)
    assert run(db, audit, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "3 archives to walk" in out and "3 new, 0 resuming" in out
    assert not os.path.exists(db)                # and writes nothing


def test_a_second_run_finds_nothing_left_to_walk(tmp_path, monkeypatch, capsys):
    world, db, audit = three(tmp_path, monkeypatch)
    assert run(db, audit) == 0
    capsys.readouterr()
    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda *a, **k: pytest.fail("a walked archive was read again"))
    assert run(db, audit) == 0
    out = capsys.readouterr().out
    assert "0 archives to walk" in out and "already walked   : 3" in out


def test_a_copy_elsewhere_is_walked_once(tmp_path, monkeypatch, capsys):
    world = World(tmp_path)
    data = build_tar(members(4, 9))
    world.add("/R/a", "x.tar", data)
    world.add("/R/b", "x.tar", data)
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 0
    assert "copies           : 1" in capsys.readouterr().out
    assert list(rows(db)) == ["/R/a"]


def test_an_archive_gone_from_its_folder_is_recorded_and_the_run_goes_on(tmp_path,
                                                                         monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    world.gone.add("/R/a")
    assert run(db, audit) == 0
    found = rows(db)
    assert "/R/a" not in found and found["/R/b"]["state"] == "complete"
    events = ArchiveStore(db).connect().execute(
        "SELECT archive_id, detail FROM events WHERE kind='gone'").fetchall()
    assert [(e["archive_id"], e["detail"].split(":")[0]) for e in events] == [
        (None, "/R/a/one.tar")]


def test_a_registered_archive_that_has_gone_is_recorded_under_its_id(tmp_path, monkeypatch):
    """Every part gone, or the whole folder: an event under the archive's id, and its row
    left as it was for a later run to try again."""
    world, db, audit = three(tmp_path, monkeypatch)
    for folder, name in (("/R/a", "one.tar"), ("/R/c", "three.tar")):
        assert main(["archive", "register", "--db", db, "--folder", folder,
                     "--name", name]) == 0
    before = rows(db)
    world.folders["/R/a"] = []
    world.gone.add("/R/c")
    assert run(db, audit) == 0
    after = rows(db)
    assert [after[f]["state"] for f in ("/R/a", "/R/c")] == [
        before[f]["state"] for f in ("/R/a", "/R/c")]
    events = ArchiveStore(db).connect().execute(
        "SELECT archive_id, detail FROM events WHERE kind='gone' ORDER BY archive_id"
        ).fetchall()
    assert [(e["archive_id"], e["detail"].split(":")[0]) for e in events] == [
        (before["/R/a"]["id"], "/R/a/one.tar"), (before["/R/c"]["id"], "/R/c/three.tar")]


def test_a_pax_archive_is_refused_and_every_other_walk_goes_on(tmp_path, monkeypatch):
    """A pax header stops that archive's walk, never the run: the five behind it are still
    walked."""
    world = World(tmp_path)
    world.add("/R/p", "pax.tar", build_tar(
        [(n, d, {"pax_headers": {"mtime": "1700000000.25"}}) for n, d in members(30, 1)],
        format=tarfile.PAX_FORMAT))
    for i in range(5):
        world.add(f"/R/s{i}", f"s{i}.tar", build_tar(members(2, 10 + i)))
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db(), "--streams", "1", "--max-streams", "1") == 0
    states = {f: r["state"] for f, r in rows(db).items()}
    assert states.pop("/R/p") == "unsupported"
    assert states == {f"/R/s{i}": "complete" for i in range(5)}


def test_a_failed_compression_check_leaves_its_archive_in_error_and_the_run_goes_on(
        tmp_path, monkeypatch):
    """Gzip files named .tar were 7% of a sample, and the read that recognises one meets
    Dropbox's passing failures like any other read: three must not stop the run as bugs."""
    world = World(tmp_path)
    for i in range(3):
        world.add(f"/R/g{i}", f"g{i}.tar", gzip.compress(build_tar(members(5, i)), mtime=0))
        world.add(f"/R/t{i}", f"t{i}.tar", build_tar(members(3, 20 + i)))
    world.install(monkeypatch)

    class Failing(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            if length <= 6:                     # only the compression check asks so little
                raise ReaderError("rate limited")
            return super().read_range(part_idx, offset, length)

    monkeypatch.setattr("dbaudit.cli._reader_for", lambda row, archive_set, tokens=None,
                        limiter=None: Failing(world.local[row["folder"]], archive_set))
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 0
    found = {f: (r["state"], r["error"] or "") for f, r in rows(db).items()}
    assert {f: s for f, (s, _) in found.items() if f.startswith("/R/t")} == {
        f"/R/t{i}": "complete" for i in range(3)}
    for i in range(3):
        state, error = found[f"/R/g{i}"]
        assert state == "error" and "could not read its first bytes" in error


def test_an_archive_that_moved_since_registration_is_walked_once_where_it_is_now(
        tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/old", "m.tar", build_tar(members(6, 5)))
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert main(["archive", "register", "--db", db, "--folder", "/R/old",
                 "--name", "m.tar"]) == 0
    # The same files, under the same ids, in another folder; the old one is empty.
    world.folders["/R/new"] = [dict(e, path_display=e["path_display"].replace("old", "new"))
                               for e in world.folders.pop("/R/old")]
    world.local["/R/new"] = world.local.pop("/R/old")
    tries, real = Counter(), cli._walk_archive

    def counted(args, store, row, *rest, **kw):
        tries[row["id"]] += 1
        return real(args, store, row, *rest, **kw)

    monkeypatch.setattr("dbaudit.cli._walk_archive", counted)
    assert run(db, world.audit_db()) == 0
    [row] = rows(db).values()
    assert (row["folder"], row["state"], row["n_members"]) == ("/R/new", "complete", 6)
    assert list(tries.values()) == [1]
    assert ArchiveStore(db).connect().execute(
        "SELECT COUNT(*) FROM events WHERE kind='gone'").fetchone()[0] == 0


def test_an_archive_registered_by_hand_outside_the_listing_is_walked_too(tmp_path,
                                                                         monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    world.add("/elsewhere", "extra.tar", build_tar(members(4, 8)))   # not in the listing
    assert main(["archive", "register", "--db", db, "--folder", "/elsewhere",
                 "--name", "extra.tar"]) == 0
    assert run(db, audit) == 0
    assert (rows(db)["/elsewhere"]["state"], rows(db)["/elsewhere"]["n_members"]) == (
        "complete", 4)


def test_without_credentials_the_third_archive_stops_the_run(tmp_path, monkeypatch,
                                                              capsys):
    """The lister hands on a token failure as a failed request. Asking again cannot help,
    so each archive that meets it is a bug, and the third stops the run."""
    world, db, audit = three(tmp_path, monkeypatch)
    for folder in world.folders:
        failed = TransientError("request failed: AuthError: rclone has no token")
        failed.__cause__ = AuthError("rclone has no token: run `rclone config reconnect`")
        world.faults[folder] = [failed]
    assert run(db, audit) == 1
    assert "repeated unexpected failures" in capsys.readouterr().err
    assert rows(db) == {}
    details = [e[0] for e in ArchiveStore(db).connect().execute(
        "SELECT detail FROM events WHERE kind='fail'")]
    assert len(details) == 3 and all("AuthError: rclone has no token" in d for d in details)


def test_a_folder_listed_page_by_page_is_read_whole(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/s", "split.tar", build_tar(members(40, 4)), part_size=60_000)
    world.folders["/R/s"].insert(3, {".tag": "file", "name": "notes.txt", "size": 5,
                                     "path_display": "/R/s/notes.txt", "id": "id:notes",
                                     "rev": "r", "content_hash": "00" * 32,
                                     "client_modified": "2026-01-01T00:00:00Z",
                                     "server_modified": "2026-01-01T00:00:00Z"})
    world.page_size = 1
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 0
    [row] = rows(db).values()
    assert row["n_parts"] > 4 and (row["state"], row["n_members"]) == ("complete", 40)


def test_archives_of_one_name_in_different_folders_are_walked_apart(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/a", "x.tar", build_tar(members(3, 1)))
    world.add("/R/b", "x.tar", build_tar(members(7, 2)))
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 0
    assert {f: (r["state"], r["n_members"]) for f, r in rows(db).items()} == {
        "/R/a": ("complete", 3), "/R/b": ("complete", 7)}


def test_parts_that_change_during_the_walk_leave_it_stale_and_the_run_goes_on(
        tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    changed = []

    def reader(row, archive_set, tokens=None, limiter=None):
        if row["folder"] == "/R/b" and not changed:
            changed.append(True)              # someone re-uploads a part mid-walk
            world.folders["/R/b"][0]["content_hash"] = "ff" * 32
        return world.reader(row, archive_set)

    monkeypatch.setattr("dbaudit.cli._reader_for", reader)
    assert run(db, audit) == 0
    assert {f: r["state"] for f, r in rows(db).items()} == {
        "/R/a": "complete", "/R/b": "stale", "/R/c": "complete"}


def test_a_set_of_parts_with_a_gap_at_its_turn_is_refused_and_the_run_goes_on(
        tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    del world.folders["/R/b"][1]                 # a part deleted since the listing
    assert run(db, audit) == 0
    assert {f: r["state"] for f, r in rows(db).items()} == {
        "/R/a": "complete", "/R/c": "complete"}
    [detail] = [e[0] for e in ArchiveStore(db).connect().execute(
        "SELECT detail FROM events WHERE kind='refused'")]
    assert detail.startswith("/R/b/two.tar: ")


def test_an_archive_mid_walk_resumes_and_finishes(tmp_path, monkeypatch, capsys):
    world, db, audit = three(tmp_path, monkeypatch)
    assert main(["archive", "register", "--db", db, "--folder", "/R/b",
                 "--name", "two.tar"]) == 0
    assert main(["archive", "index", "--db", db, "--archive", "two.tar", "--batch", "5",
                 "--max-batches", "1"]) == 0
    assert rows(db)["/R/b"]["state"] == "walking"
    capsys.readouterr()
    assert run(db, audit) == 0
    assert "2 new, 1 resuming" in capsys.readouterr().out
    assert rows(db)["/R/b"]["state"] == "complete"
    assert rows(db)["/R/b"]["n_members"] == 30


def test_parts_that_changed_since_registration_are_walked_as_new_content(tmp_path,
                                                                         monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    assert main(["archive", "register", "--db", db, "--folder", "/R/a",
                 "--name", "one.tar"]) == 0
    old = rows(db)["/R/a"]["id"]
    # The folder now holds other content under the same name.
    world.folders["/R/a"] = []
    world.local.pop("/R/a")
    world.add("/R/a", "one.tar", build_tar(members(6, 7)))
    assert run(db, audit) == 0
    store = ArchiveStore(db)
    by_id = {r["id"]: r for r in store.connect().execute("SELECT * FROM archives")}
    assert by_id[old]["state"] == "stale"
    [new] = [r for r in by_id.values() if r["folder"] == "/R/a" and r["id"] != old]
    assert (new["state"], new["n_members"]) == ("complete", 6)


def test_a_held_database_refuses_with_status_3(tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    with InstanceLock(f"{db}.lock"):
        assert run(db, audit) == 3


def test_streams_beyond_their_ceiling_are_a_usage_error(tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    assert run(db, audit, "--streams", "30", "--max-streams", "24") == 2
    assert run(db, str(tmp_path / "no-such.db")) == 2


def test_a_db_that_is_not_an_archive_database_is_refused_untouched(tmp_path, monkeypatch):
    """The audit database given as --db by mistake gains no tables."""
    world, db, audit = three(tmp_path, monkeypatch)

    def tables(path):
        return {r[0] for r in sqlite3.connect(path).execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

    before = tables(audit)
    for extra in ((), ("--dry-run",)):
        assert run(audit, audit, *extra) == 2
    assert tables(audit) == before


def logged(monkeypatch, text):
    """An Event set once a `dbaudit` log message holds ``text``: the signal handler logs
    after it sets the stop, so its line means every walk is already winding down."""
    seen = threading.Event()

    class Watch(logging.Handler):
        def emit(self, record):
            if text in record.getMessage():
                seen.set()

    logger = logging.getLogger("dbaudit")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, Watch()])
    return seen


def signalling(world, monkeypatch, signum):
    """Readers that send ``signum`` on the run's third read, once, and hold that read until
    the main thread has acted on it, so its archive is stopped mid-walk."""
    reads, sent, lock = [0], threading.Event(), threading.Lock()
    handled = logged(monkeypatch, "committing every batch")

    class Signalling(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            with lock:
                reads[0] += 1
                third = reads[0] == 3
            if third:
                sent.set()
                signal.raise_signal(signum)
                handled.wait(timeout=10)
            return super().read_range(part_idx, offset, length)

    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            Signalling(world.local[row["folder"]], archive_set))
    return sent


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_stops_every_walk_and_a_second_run_finishes_them(tmp_path, monkeypatch,
                                                                  signum):
    world, db, audit = three(tmp_path, monkeypatch)
    sent = signalling(world, monkeypatch, signum)
    before = signal.getsignal(signum)
    assert run(db, audit, "--streams", "1", "--max-streams", "1", "--workers", "1") == 0
    assert signal.getsignal(signum) is before
    assert sent.is_set()
    assert any(r["state"] != "complete" for r in rows(db).values())

    world.install(monkeypatch)
    assert run(db, audit) == 0
    assert {(r["state"], r["n_members"]) for r in rows(db).values()} == {
        ("complete", 5), ("complete", 30), ("complete", 8)}


def test_a_second_signal_exits_at_once_with_status_1(tmp_path, monkeypatch):
    """The second signal is sent once the first is handled: two at once arrive as one."""
    world, db, audit = three(tmp_path, monkeypatch)
    exits, lock, reads = [], threading.Lock(), [0]
    monkeypatch.setattr("os._exit", exits.append)   # the real one would end pytest too
    handled = logged(monkeypatch, "committing every batch")

    class Twice(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            with lock:
                reads[0] += 1
                third = reads[0] == 3
            if third:
                signal.raise_signal(signal.SIGTERM)
                assert handled.wait(timeout=10)
                signal.raise_signal(signal.SIGTERM)
                deadline = time.monotonic() + 10
                while not exits and time.monotonic() < deadline:
                    time.sleep(0.01)
            return super().read_range(part_idx, offset, length)

    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            Twice(world.local[row["folder"]], archive_set))
    run(db, audit, "--streams", "1", "--max-streams", "1", "--workers", "1")
    assert exits == [1]


def test_three_unexpected_failures_stop_the_run_with_status_1(tmp_path, monkeypatch,
                                                              capsys):
    """Each failure is recorded on its archive; the third stops the run."""
    world = World(tmp_path)
    for i in range(3):
        world.add(f"/R/bad{i}", f"bad{i}.tar", build_tar(members(3, i)))
        world.add(f"/R/good{i}", f"good{i}.tar", build_tar(members(3, 10 + i)))
    world.install(monkeypatch)
    real = cli._walk_archive

    def failing(args, store, row, *rest, **kw):
        if row["name"].startswith("bad"):
            raise RuntimeError(f"injected into {row['name']}")
        return real(args, store, row, *rest, **kw)

    monkeypatch.setattr("dbaudit.cli._walk_archive", failing)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 1
    assert "repeated unexpected failures" in capsys.readouterr().err
    found = rows(db)
    assert {f: (found[f]["state"], found[f]["error"]) for f in found if "bad" in f} == {
        f"/R/bad{i}": ("error", f"RuntimeError: injected into bad{i}.tar") for i in range(3)}


def test_an_archive_in_error_is_tried_once_more_then_left_while_the_rest_finish(
        tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)

    class Unreadable(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            raise OSError("the disk went away")

    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            (Unreadable if row["folder"] == "/R/a" else LocalRangeReader)(
                                world.local[row["folder"]], archive_set))
    tries, real = Counter(), cli._walk_archive

    def counted(args, store, row, *rest, **kw):
        tries[row["folder"]] += 1
        return real(args, store, row, *rest, **kw)

    monkeypatch.setattr("dbaudit.cli._walk_archive", counted)
    assert run(db, audit) == 0
    assert tries == {"/R/a": 2, "/R/b": 1, "/R/c": 1}
    assert {f: r["state"] for f, r in rows(db).items()} == {
        "/R/a": "error", "/R/b": "complete", "/R/c": "complete"}


def test_a_split_archive_reads_with_no_more_chains_than_workers(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/s", "split.tar", build_tar(members(40, 4)), part_size=60_000)
    world.install(monkeypatch)
    lock, flying, most = threading.Lock(), [0], [0]

    class Counting(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            with lock:
                flying[0] += 1
                most[0] = max(most[0], flying[0])
            try:
                time.sleep(0.01)
                return super().read_range(part_idx, offset, length)
            finally:
                with lock:
                    flying[0] -= 1

    monkeypatch.setattr("dbaudit.cli._reader_for",
                        lambda row, archive_set, tokens=None, limiter=None:
                            Counting(world.local[row["folder"]], archive_set))
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db(), "--workers", "2") == 0
    [row] = rows(db).values()
    assert row["n_parts"] > 4 and (row["state"], row["n_members"]) == ("complete", 40)
    assert most[0] == 2


def test_a_split_archive_capped_at_its_workers_leaves_streams_for_others(tmp_path,
                                                                         monkeypatch):
    """One stream and a margin of two: a split archive capped at two chains leaves room
    for a small archive to walk beside it, and does not wait for it to finish first."""
    world = World(tmp_path)
    world.add("/R/s", "split.tar", build_tar(members(40, 4)), part_size=60_000)
    world.add("/R/t", "tiny.tar", build_tar(members(2, 5)))
    world.install(monkeypatch)
    tiny_read, waited, lock = threading.Event(), [], threading.Lock()

    def reader(row, archive_set, tokens=None, limiter=None):
        if row["folder"] == "/R/t":
            tiny_read.set()
        else:
            with lock:                       # only the split archive's first chain waits
                first = not waited
                waited.append(None)
            if first:
                waited[0] = tiny_read.wait(timeout=5)
        return world.reader(row, archive_set)

    monkeypatch.setattr("dbaudit.cli._reader_for", reader)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db(), "--streams", "1", "--max-streams", "1",
               "--workers", "2") == 0
    assert waited[0] is True
    assert {r["state"] for r in rows(db).values()} == {"complete"}


def test_content_walked_before_under_another_name_is_not_walked_again(tmp_path,
                                                                      monkeypatch):
    """The listing is older than the folder: what the folder holds now was walked
    elsewhere already."""
    world = World(tmp_path)
    data = build_tar(members(4, 6))
    world.add("/R/a", "a.tar", build_tar(members(3, 7)))
    world.add("/R/b", "b.tar", data)
    world.install(monkeypatch)
    audit = world.audit_db()
    db = str(tmp_path / "archives.db")
    assert main(["archive", "register", "--db", db, "--folder", "/R/b",
                 "--name", "b.tar"]) == 0
    assert main(["archive", "index", "--db", db, "--archive", "b.tar"]) == 0
    world.folders["/R/a"] = []
    world.local.pop("/R/a")
    world.add("/R/a", "a.tar", data)
    walked, real = [], cli._walk_archive

    def counted(args, store, row, *rest, **kw):
        walked.append(row["folder"])
        return real(args, store, row, *rest, **kw)

    monkeypatch.setattr("dbaudit.cli._walk_archive", counted)
    assert run(db, audit) == 0
    assert walked == []


def test_two_places_that_come_to_hold_the_same_content_walk_it_once(tmp_path, monkeypatch):
    """The listing saw two archives; by their turns one place holds the other's bytes. Both
    jobs meet the same archive, which one walks, not both at once."""
    world = World(tmp_path)
    data = build_tar(members(40, 4))
    world.add("/R/a", "a.tar", data, part_size=60_000)
    world.add("/R/b", "b.tar", build_tar(members(40, 5)), part_size=60_000)
    world.install(monkeypatch)
    audit = world.audit_db()
    world.folders["/R/b"] = []
    world.add("/R/b", "b.tar", data, part_size=60_000)     # b.tar now holds a.tar's bytes
    flying, most, lock, real = Counter(), Counter(), threading.Lock(), cli._walk_archive

    def counted(args, store, row, *rest, **kw):
        with lock:
            flying[row["id"]] += 1
            most[row["id"]] = max(most[row["id"]], flying[row["id"]])
        try:
            return real(args, store, row, *rest, **kw)
        finally:
            with lock:
                flying[row["id"]] -= 1

    class Slow(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            time.sleep(0.02)                    # so that the two walks would overlap
            return super().read_range(part_idx, offset, length)

    monkeypatch.setattr("dbaudit.cli._walk_archive", counted)
    monkeypatch.setattr("dbaudit.cli._reader_for", lambda row, archive_set, tokens=None,
                        limiter=None: Slow(world.local[row["folder"]], archive_set))
    db = str(tmp_path / "archives.db")
    assert run(db, audit) == 0
    assert list(most.values()) == [1]
    assert [r["state"] for r in rows(db).values()] == ["complete"]


@pytest.mark.parametrize("how", ["gone", "gap"])
def test_a_copy_is_walked_when_the_archive_standing_for_it_has_gone_or_lost_a_part(
        tmp_path, monkeypatch, how):
    """One place stands for each content, and its copies wait. If that place has gone by its
    turn, or lost a part, the content is walked where a copy of it is."""
    world = World(tmp_path)
    data = build_tar(members(30, 9))
    for folder in ("/R/a", "/R/b"):             # /R/a stands for both: it sorts first
        world.add(folder, "x.tar", data, part_size=40_000)
    world.install(monkeypatch)
    audit = world.audit_db()
    if how == "gone":
        world.gone.add("/R/a")
    else:
        del world.folders["/R/a"][1]            # a part deleted after the listing
    db = str(tmp_path / "archives.db")
    assert run(db, audit) == 0
    assert {f: (r["state"], r["n_members"]) for f, r in rows(db).items()} == {
        "/R/b": ("complete", 30)}
    events = ArchiveStore(db).connect().execute("SELECT kind, detail FROM events").fetchall()
    assert any(kind == ("gone" if how == "gone" else "refused") and detail.startswith("/R/a/")
               for kind, detail in events)


def test_a_run_goes_on_with_a_walk_from_its_cursors_not_from_the_start(tmp_path, monkeypatch):
    """What the switch rests on: a walk half done in the database goes on from where it
    stopped. A walk from scratch would read the first header again."""
    world = World(tmp_path)
    world.add("/R/a", "a.tar", build_tar(members(120, 8)))
    world.install(monkeypatch)
    audit = world.audit_db()
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--folder", "/R/a", "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar", "--batch", "10",
          "--max-batches", "3"])
    asked = []

    class Recording(LocalRangeReader):
        def read_range(self, part_idx, offset, length):
            asked.append(offset)
            return super().read_range(part_idx, offset, length)

    monkeypatch.setattr("dbaudit.cli._reader_for", lambda row, archive_set, tokens=None,
                        limiter=None: Recording(world.local[row["folder"]], archive_set))
    assert run(db, audit) == 0
    assert (rows(db)["/R/a"]["state"], rows(db)["/R/a"]["n_members"]) == ("complete", 120)
    assert asked and min(asked) > 0


def test_several_archives_in_one_folder_are_each_walked(tmp_path, monkeypatch):
    """The usual shape: one folder holds many archives, plain and split, side by side."""
    world = World(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    for name, data, part_size in (("a.tar", build_tar(members(5, 1)), None),
                                  ("b.tar", build_tar(members(30, 2)), 40_000),
                                  ("c.tar", build_tar(members(8, 3)), None)):
        for path in list(world.add("/R/x", name, data, part_size).iterdir()):
            path.rename(shared / path.name)
    world.local["/R/x"] = shared
    world.install(monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, world.audit_db()) == 0
    found = ArchiveStore(db).connect().execute(
        "SELECT name, state, n_members, n_parts > 1 FROM archives ORDER BY name").fetchall()
    assert [tuple(r) for r in found] == [("a.tar", "complete", 5, 0),
                                         ("b.tar", "complete", 30, 1),
                                         ("c.tar", "complete", 8, 0)]


def test_every_walk_and_listing_takes_its_turn_at_one_limiter(tmp_path, monkeypatch):
    """So one walk's rate-limit response slows every walk: the limiter pauses every
    request it gates."""
    world, db, audit = three(tmp_path, monkeypatch)
    made, seen, fast = [], [], cli._run_limiter      # the World's, quick for tests

    def limiter_for(args):
        made.append(fast(args))
        return made[-1]

    def reader(row, archive_set, tokens=None, limiter=None):
        seen.append(limiter)
        return world.reader(row, archive_set)

    monkeypatch.setattr("dbaudit.cli._run_limiter", limiter_for)
    monkeypatch.setattr("dbaudit.cli._reader_for", reader)
    assert run(db, audit) == 0
    assert len(made) == 1 and len(seen) >= 3 and all(limiter is made[0] for limiter in seen)
    assert made[0].granted_total == 6            # each archive's two listings


def test_the_runs_limiter_starts_at_the_streams_and_may_climb_to_the_ceiling():
    limiter = _run_limiter(SimpleNamespace(streams=12, max_streams=24))
    assert (limiter.concurrency, limiter.max_concurrency, limiter.rps, limiter.max_rps) == (
        12, 24, 8.0, 16.0)


def test_a_rate_limited_listing_pauses_every_walk_and_is_asked_again(tmp_path, monkeypatch):
    world, db, audit = three(tmp_path, monkeypatch)
    world.faults["/R/a"] = [RateLimited("too_many_requests", retry_after=0.01)]
    made, fast = [], cli._run_limiter
    monkeypatch.setattr("dbaudit.cli._run_limiter",
                        lambda args: made.append(fast(args)) or made[-1])
    assert run(db, audit) == 0
    assert {r["state"] for r in rows(db).values()} == {"complete"}
    assert (made[0].rate_limit_events, made[0].concurrency) == (1, 11)


def test_a_folder_that_cannot_be_listed_is_left_in_error_and_the_run_goes_on(tmp_path,
                                                                            monkeypatch):
    """Failures that outlast the retries: tried once more at the end of the queue, then
    left in error -- not a bug, so the run exits 0."""
    world, db, audit = three(tmp_path, monkeypatch)
    monkeypatch.setattr("dbaudit.cli.LISTING_BACKOFF", 0)
    world.faults["/R/a"] = [TransientError("request failed: connection reset")] * 100
    assert run(db, audit) == 0
    assert set(rows(db)) == {"/R/b", "/R/c"}
    details = [e[0] for e in ArchiveStore(db).connect().execute(
        "SELECT detail FROM events WHERE kind='fail'")]
    assert details == ["/R/a/one.tar: could not list its folder: TransientError: request "
                       "failed: connection reset"] * 2
    assert len(world.faults["/R/a"]) == 100 - 2 * LISTING_ATTEMPTS


def mixed(rng, i, round_):
    """One archive of the equivalence check's mix: mostly plain tars, some split into
    parts, some pax, some gzip files named .tar."""
    data = build_tar(members(rng.randint(1, 12), seed=round_ * 1000 + i))
    kind = rng.choices(["plain", "split", "pax", "gzip"], weights=[60, 25, 10, 5])[0]
    if kind == "pax":
        data = build_tar([(n, d, {"pax_headers": {"mtime": "1700000000.25"}})
                          for n, d in members(rng.randint(1, 6), seed=i)],
                         format=tarfile.PAX_FORMAT)
    elif kind == "gzip":
        data = gzip.compress(data, mtime=0)
    split = kind == "split"
    return data, max(len(data) // rng.randint(2, 4), 1024) if split else None


@pytest.mark.parametrize("round_", range(3))
def test_many_archives_walked_at_once_give_exactly_the_rows_walked_one_at_a_time(
        tmp_path, monkeypatch, round_):
    """The equivalence check: threads, a shared limiter and one database change nothing
    about what any archive's walk records. Each round has its own mix, and DBAUDIT_SEED
    gives every round another."""
    rng = random.Random(f"{os.environ.get('DBAUDIT_SEED', '')}/{round_}")
    world = World(tmp_path)
    names = {}
    for i in range(200):
        data, part_size = mixed(rng, i, round_)
        names[f"/R/f{i:03d}"] = f"a{i:03d}.tar"
        world.add(f"/R/f{i:03d}", f"a{i:03d}.tar", data, part_size=part_size)
    world.install(monkeypatch)
    many, single = str(tmp_path / "many.db"), str(tmp_path / "single.db")
    assert run(many, world.audit_db(), "--streams", "24", "--max-streams", "24") == 0
    for folder, name in names.items():
        assert main(["archive", "register", "--db", single, "--folder", folder,
                     "--name", name]) == 0
        assert main(["archive", "index", "--db", single, "--archive", name,
                     "--rps", "1000"]) in (0, 1)

    def manifest(db):
        store = ArchiveStore(db)
        out = {}
        for archive in store.connect().execute(
                "SELECT id, set_hash, state, detail, n_members FROM archives"):
            out[archive["set_hash"]] = (
                archive["state"], archive["detail"], archive["n_members"],
                [tuple(r) for r in store.connect().execute(
                    "SELECT hdr_offset, data_offset, size, type, mode, mtime, uname, gname, "
                    "dir, name, linkname FROM members WHERE archive_id=? ORDER BY hdr_offset",
                    (archive["id"],))])
        return out

    walked_many, walked_single = manifest(many), manifest(single)
    assert len(walked_many) == 200
    assert {state for state, *_ in walked_many.values()} >= {"complete", "unsupported"}
    assert walked_many == walked_single
