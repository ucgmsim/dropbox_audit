"""`archive import` and `archive forget`: walks moved from one database into another,
and an archive dropped so that the next run walks it from scratch."""

import shutil
import sqlite3

import pytest

from dbaudit.archive.store import ArchiveStore
from dbaudit.cli import main
from dbaudit.lock import InstanceLock
from tests.archive_fakes import World, build_tar, members, write_parts


def three(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.add("/R/a", "one.tar", build_tar(members(5, 1)))
    world.add("/R/b", "two.tar", build_tar(members(30, 2)), part_size=40_000)
    world.add("/R/c", "three.tar", build_tar(members(8, 3)))
    world.install(monkeypatch)
    return world, world.audit_db()


def run(db, audit, *extra):
    return main(["archive", "run", "--db", db, "--audit-db", audit, *extra])


def db_import(target, source):
    return main(["archive", "import", "--db", target, "--from", source])


def forget(db, name, folder):
    return main(["archive", "forget", "--db", db, "--archive", name, "--folder", folder])


def archives(db):
    return {r["folder"]: r for r in ArchiveStore(db).connect().execute(
        "SELECT * FROM archives")}


def manifest(db):
    """Each archive's row, parts, segments, members and events -- all but the import's
    own event -- keyed by its content rather than its id."""
    conn = ArchiveStore(db).connect()
    skip, out = {"archive_id", "id", "owner"}, {}
    for archive in conn.execute("SELECT * FROM archives"):
        rows = {}
        for table, order in (("parts", "idx"), ("segments", "idx"), ("members", "hdr_offset"),
                             ("events", "ts")):
            found = conn.execute(f"SELECT * FROM {table} WHERE archive_id=? ORDER BY {order}",
                                 (archive["id"],))
            rows[table] = [tuple(v for k, v in zip(r.keys(), r) if k not in skip)
                           for r in found if table != "events" or r["kind"] != "import"]
        out[archive["set_hash"]] = ({k: archive[k] for k in archive.keys() if k != "id"}, rows)
    return out


def test_import_copies_every_archive_under_new_ids_with_everything_recorded(tmp_path,
                                                                            monkeypatch,
                                                                            capsys):
    world, audit = three(tmp_path, monkeypatch)
    source, target = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    assert run(source, audit) == 0
    # The target already holds an archive, under an id the source also uses.
    world.add("/R/x", "x.tar", build_tar(members(2, 9)))
    assert main(["archive", "register", "--db", target, "--folder", "/R/x",
                 "--name", "x.tar"]) == 0
    one = archives(source)["/R/a"]["id"]
    with ArchiveStore(source).connect() as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)",
                     (f"recheck:{one}", '{"from": 0, "verdict": [0, "corrupt", 0]}'))
        conn.execute("UPDATE segments SET owner='4242:1'")   # as a killed walk leaves them
    capsys.readouterr()

    assert db_import(target, source) == 0
    assert "imported 3 archives" in capsys.readouterr().out
    found = archives(target)
    assert set(found) == {"/R/a", "/R/b", "/R/c", "/R/x"}
    assert len({r["id"] for r in found.values()}) == 4
    copied = manifest(target)
    for set_hash, recorded in manifest(source).items():
        assert copied[set_hash] == recorded
    conn = ArchiveStore(target).connect()
    assert conn.execute("SELECT COUNT(*) FROM segments WHERE owner IS NOT NULL"
                        ).fetchone()[0] == 0
    assert conn.execute("SELECT value FROM meta WHERE key=?",
                        (f"recheck:{found['/R/a']['id']}",)).fetchone()[0].startswith('{"from"')
    events = conn.execute("SELECT archive_id FROM events WHERE kind='import'").fetchall()
    assert sorted(e[0] for e in events) == sorted(found[f]["id"] for f in ("/R/a", "/R/b",
                                                                            "/R/c"))


def test_import_replaces_an_archive_of_the_same_content_and_everything_it_held(
        tmp_path, monkeypatch):
    world, audit = three(tmp_path, monkeypatch)
    source, target = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    assert run(source, audit) == 0
    assert main(["archive", "register", "--db", target, "--folder", "/R/b",
                 "--name", "two.tar"]) == 0
    assert main(["archive", "index", "--db", target, "--archive", "two.tar", "--batch", "5",
                 "--max-batches", "1"]) == 0

    assert db_import(target, source) == 0
    row = archives(target)["/R/b"]
    assert (row["state"], row["n_members"]) == ("complete", 30)
    conn = ArchiveStore(target).connect()

    def count(sql, *args):
        return conn.execute(sql, args).fetchone()[0]

    assert count("SELECT COUNT(*) FROM members WHERE archive_id=?", row["id"]) == 30
    assert count("SELECT COUNT(*) FROM segments WHERE archive_id=?", row["id"]) == row["n_parts"]
    for table in ("parts", "segments", "members", "events"):
        assert count(f"SELECT COUNT(*) FROM {table} WHERE archive_id IS NOT NULL AND "
                     f"archive_id NOT IN (SELECT id FROM archives)") == 0


def test_an_imported_mid_walk_archive_finishes_with_the_members_a_fresh_walk_records(
        tmp_path, monkeypatch):
    world, audit = three(tmp_path, monkeypatch)
    source, target, fresh = (str(tmp_path / n) for n in ("source.db", "target.db",
                                                          "fresh.db"))
    assert main(["archive", "register", "--db", source, "--folder", "/R/b",
                 "--name", "two.tar"]) == 0
    assert main(["archive", "index", "--db", source, "--archive", "two.tar", "--batch", "5",
                 "--max-batches", "1"]) == 0
    assert archives(source)["/R/b"]["state"] == "walking"

    assert db_import(target, source) == 0
    assert run(target, audit) == 0
    assert run(fresh, audit) == 0

    def members_of(db):
        conn = ArchiveStore(db).connect()
        return {r["set_hash"]: [tuple(m) for m in conn.execute(
            "SELECT hdr_offset, data_offset, size, name FROM members WHERE archive_id=? "
            "ORDER BY hdr_offset", (r["id"],))]
            for r in conn.execute("SELECT id, set_hash FROM archives")}

    assert archives(target)["/R/b"]["state"] == "complete"
    assert members_of(target) == members_of(fresh)


def test_import_refuses_while_either_database_is_held(tmp_path, monkeypatch, capsys):
    world, audit = three(tmp_path, monkeypatch)
    source, target = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    assert run(source, audit) == 0
    for held in (source, target):
        with InstanceLock(f"{held}.lock"):
            assert db_import(target, source) == 3
    assert "held by" in capsys.readouterr().err
    assert not (tmp_path / "target.db").exists() or archives(target) == {}


def test_import_refuses_databases_whose_tables_differ(tmp_path, monkeypatch, capsys):
    world, audit = three(tmp_path, monkeypatch)
    source, target = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    assert run(source, audit) == 0
    assert main(["archive", "register", "--db", target, "--folder", "/R/a",
                 "--name", "one.tar"]) == 0
    with sqlite3.connect(source) as conn:
        conn.execute("ALTER TABLE members ADD COLUMN checksum TEXT")
    capsys.readouterr()
    assert db_import(target, source) == 2
    err = capsys.readouterr().err
    assert "differ" in err and "members" in err and "nothing was imported" in err
    assert set(archives(target)) == {"/R/a"} and archives(target)["/R/a"]["n_members"] == 0


@pytest.mark.parametrize("which", ["missing", "same", "not an archive database"])
def test_import_from_nothing_or_itself_or_into_another_kind_is_a_usage_error(
        tmp_path, monkeypatch, which):
    world, audit = three(tmp_path, monkeypatch)
    target = str(tmp_path / "target.db")
    assert run(target, audit) == 0
    if which == "not an archive database":
        source, target = target, audit
    else:
        source = str(tmp_path / "no-such.db") if which == "missing" else target
    tables = sqlite3.connect(target).execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert db_import(target, source) == 2
    assert sqlite3.connect(target).execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall() == tables


def test_forget_drops_an_archive_so_the_next_run_walks_it_from_scratch(tmp_path,
                                                                       monkeypatch, capsys):
    world, audit = three(tmp_path, monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, audit) == 0
    gone = archives(db)["/R/a"]["id"]
    with ArchiveStore(db).connect() as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, '{}')", (f"recheck:{gone}",))
    capsys.readouterr()

    assert forget(db, "one.tar", "/R/a") == 0
    assert "forgot archive" in capsys.readouterr().out
    conn = ArchiveStore(db).connect()
    for table in ("parts", "segments", "members", "events"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE archive_id=?",
                            (gone,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM meta WHERE key=?",
                        (f"recheck:{gone}",)).fetchone()[0] == 0
    [event] = conn.execute("SELECT archive_id, detail FROM events WHERE kind='forget'")
    assert event["archive_id"] is None and "/R/a/one.tar" in event["detail"]
    assert set(archives(db)) == {"/R/b", "/R/c"}

    read = set()

    def watched(row, archive_set, tokens=None, limiter=None):
        read.add(row["folder"])
        return world.reader(row, archive_set)

    monkeypatch.setattr("dbaudit.cli._reader_for", watched)
    assert run(db, audit) == 0
    assert read == {"/R/a"}
    assert (archives(db)["/R/a"]["state"], archives(db)["/R/a"]["n_members"]) == (
        "complete", 5)


def test_forget_refuses_an_archive_that_is_not_there_and_a_held_database(tmp_path,
                                                                         monkeypatch):
    world, audit = three(tmp_path, monkeypatch)
    db = str(tmp_path / "archives.db")
    assert run(db, audit) == 0
    assert forget(db, "one.tar", "/R/elsewhere") == 2
    assert forget(str(tmp_path / "no-such.db"), "one.tar", "/R/a") == 2
    with InstanceLock(f"{db}.lock"):
        assert forget(db, "one.tar", "/R/a") == 3
    assert set(archives(db)) == {"/R/a", "/R/b", "/R/c"}


@pytest.mark.parametrize("imported", ["newer", "older"])
def test_import_leaves_only_the_latest_content_of_one_name_in_one_folder(tmp_path, imported):
    """As register does: of the archives of one name in one folder, the latest registered
    is what the folder holds, and other content there is stale -- whichever database held
    which."""
    folder = tmp_path / "parts"
    folder.mkdir()
    first, second = str(tmp_path / "first.db"), str(tmp_path / "second.db")
    for db, seed in ((first, 1), (second, 2)):            # one name, one folder, new content
        for old in folder.iterdir():
            old.unlink()
        write_parts(folder, build_tar(members(5, seed)), 1 << 20)
        assert main(["archive", "register", "--db", db, "--local-dir", str(folder),
                     "--name", "a.tar"]) == 0
    target, source = (first, second) if imported == "newer" else (second, first)
    assert db_import(target, source) == 0
    older, latest = ArchiveStore(target).connect().execute(
        "SELECT id, state, detail FROM archives ORDER BY created_at").fetchall()
    assert (older["state"], older["detail"]) == ("stale", f"superseded by archive "
                                                          f"{latest['id']}")
    assert latest["state"] != "stale"


def test_import_and_forget_take_the_lock_a_run_was_given(tmp_path, monkeypatch):
    """A run started with --lock PATH holds PATH, not DB.lock: import and forget take
    --lock too, so they wait for that run rather than write beside it."""
    world, audit = three(tmp_path, monkeypatch)
    db, other = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    assert run(db, audit) == 0 and run(other, audit) == 0
    lock = str(tmp_path / "elsewhere.lock")
    with InstanceLock(lock):
        assert main(["archive", "import", "--db", db, "--from", other, "--lock", lock]) == 3
        assert main(["archive", "forget", "--db", db, "--archive", "one.tar", "--folder",
                     "/R/a", "--lock", lock]) == 3


def test_import_keeps_the_live_content_of_a_folder_that_went_back(tmp_path):
    """A fourth part appears, then goes again: X, then Y, then X once more. register leaves
    X live and Y stale, though X was registered first; an import must agree."""
    folder = tmp_path / "parts"
    folder.mkdir()
    data = build_tar(members(40, 7))
    write_parts(folder, data, len(data) // 4 + 1)
    last = sorted(folder.iterdir())[-1]
    aside = tmp_path / last.name
    db, fresh = str(tmp_path / "a.db"), str(tmp_path / "fresh.db")
    for move in ((last, aside), (aside, last), (last, aside)):  # X: 3 parts, Y: 4, X again
        shutil.move(*move)
        assert main(["archive", "register", "--db", db, "--local-dir", str(folder),
                     "--name", "a.tar"]) == 0
    assert db_import(fresh, db) == 0
    states = {row["n_parts"]: row["state"] for row in ArchiveStore(fresh).connect().execute(
        "SELECT n_parts, state FROM archives")}
    assert states[3] != "stale" and states[4] == "stale"


def test_import_waits_for_a_source_run_started_with_its_own_lock(tmp_path, monkeypatch):
    world, audit = three(tmp_path, monkeypatch)
    db, other = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    assert run(db, audit) == 0 and run(other, audit) == 0
    lock = str(tmp_path / "elsewhere.lock")
    with InstanceLock(lock):
        assert main(["archive", "import", "--db", db, "--from", other,
                     "--from-lock", lock]) == 3


def test_a_lock_in_a_folder_that_does_not_exist_is_a_usage_error(tmp_path, monkeypatch,
                                                                 capsys):
    """Said plainly, not as a traceback from opening the lock file."""
    world, audit = three(tmp_path, monkeypatch)
    db, other = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    assert run(db, audit) == 0 and run(other, audit) == 0
    lock = str(tmp_path / "nowhere" / "x.lock")
    assert run(db, audit, "--lock", lock) == 2
    assert main(["archive", "index", "--db", db, "--archive", "one.tar", "--lock", lock]) == 2
    assert main(["archive", "import", "--db", db, "--from", other, "--lock", lock]) == 2
    assert main(["archive", "import", "--db", db, "--from", other, "--from-lock", lock]) == 2
    assert main(["archive", "forget", "--db", db, "--archive", "one.tar", "--folder", "/R/a",
                 "--lock", lock]) == 2
    assert capsys.readouterr().err.count("error: no folder") == 5
