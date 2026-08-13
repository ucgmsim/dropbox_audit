import pytest

from dbaudit.cli import human_bytes, human_duration, main
from dbaudit.store import Store


def seeded_db(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    store.init_schema()
    store.set_meta("root", "/R")
    store.add_shard("/R/a", 1, "recursive")
    store.add_shard("/R/b", 1, "recursive")
    store.claim_shard("w")
    store.finish_shard(1)
    return db, store


def test_status_reports_progress(tmp_path, capsys):
    db, _ = seeded_db(tmp_path)
    assert main(["status", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "1/2" in out
    assert "shards" in out.lower()
    assert "/R" in out


def test_status_lists_failed_shards(tmp_path, capsys):
    db, store = seeded_db(tmp_path)
    store.fail_shard(2, "transient failures exhausted")
    main(["status", "--db", str(db)])
    out = capsys.readouterr().out
    assert "/R/b" in out and "transient" in out


def test_status_on_uninitialised_db_is_an_error(tmp_path, capsys):
    assert main(["status", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_init_refuses_when_disk_too_small(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("dbaudit.cli.free_bytes", lambda p: 1024)
    rc = main(["init", "--db", str(tmp_path / "a.db"), "--root", "/R", "--min-free-gb", "50"])
    assert rc == 2
    assert "free" in capsys.readouterr().err


def test_run_on_uninitialised_db_is_an_error(tmp_path, capsys):
    assert main(["run", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_run_refuses_when_another_instance_holds_the_lock(tmp_path, capsys):
    from dbaudit.lock import InstanceLock

    db = tmp_path / "a.db"
    store = Store(db)
    store.init_schema()
    store.set_meta("root", "/R")
    with InstanceLock(f"{db}.lock"):
        assert main(["run", "--db", str(db)]) == 3
    assert "held by" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value,expected",
    [(0, "0 B"), (1536, "1.5 KiB"), (250 * 1024**4, "250.0 TiB")],
)
def test_human_bytes(value, expected):
    assert human_bytes(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(45, "45s"), (3600 * 6 + 60, "6h 1m"), (86400 * 2, "2d 0h"), (float("inf"), "unknown")],
)
def test_human_duration(value, expected):
    assert human_duration(value) == expected
