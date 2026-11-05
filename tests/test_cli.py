import pytest

import dbaudit.cli as cli
from dbaudit.archive.reader import WINDOW_MAX, WINDOW_MIN
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
    rc = main(
        ["init", "--db", str(tmp_path / "a.db"), "--root", "/R", "--min-free-gb", "50"]
    )
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
    [
        (45, "45s"),
        (3600 * 6 + 60, "6h 1m"),
        (86400 * 2, "2d 0h"),
        (float("inf"), "unknown"),
    ],
)
def test_human_duration(value, expected):
    assert human_duration(value) == expected


def test_report_calls_a_crawl_with_failed_shards_incomplete(
    tmp_path, capsys, monkeypatch
):
    """Nothing is pending or running, but /R/b failed: its files are missing."""
    db, store = seeded_db(tmp_path)
    store.fail_shard(2, "transient failures exhausted")

    def offline(remote):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("dbaudit.cli._space_usage", offline)
    main(["report", "--db", str(db)])
    assert "incomplete" in capsys.readouterr().out


def test_init_says_when_dropbox_cannot_look_up_the_account_just_now(
    tmp_path, monkeypatch, capsys
):
    """A 429 or a server error on the account lookup is not a listing that failed."""

    class Tokens:
        def account(self):
            raise RuntimeError(
                "users/get_current_account: 429 from Dropbox; ask again later"
            )

    monkeypatch.setattr("dbaudit.cli.free_bytes", lambda path: 1 << 50)
    monkeypatch.setattr(
        "dbaudit.cli.build_lister", lambda remote, tokens=None: (Tokens(), None)
    )
    assert main(["init", "--db", str(tmp_path / "a.db"), "--root", "/R"]) == 2
    err = capsys.readouterr().err
    assert "could not look up the account" in err and "could not list" not in err


COMMANDS = [
    ["init"],
    ["run"],
    ["status"],
    ["verify"],
    ["report"],
    ["export"],
    ["index"],
    ["tree"],
    ["kinds"],
    *(
        ["archive", c]
        for c in (
            "register",
            "index",
            "run",
            "import",
            "forget",
            "status",
            "report",
            "export",
            "cat",
        )
    ),
]


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_command_has_help(command, capsys):
    assert main([*command, "--help"]) == 0
    assert "Usage: dbaudit " + " ".join(command) in capsys.readouterr().out


# What each body got from argparse, so scripts that run dbaudit keep working.
BEFORE = [
    (
        ["init", "--db", "a.db", "--root", "/R"],
        "cmd_init",
        dict(db="a.db", root="/R", remote="dropbox", min_free_gb=30.0),
    ),
    (
        ["run", "--db", "a.db"],
        "cmd_run",
        dict(
            db="a.db",
            workers=8,
            rps=5.0,
            max_rps=20.0,
            split_depth=6,
            queue_target=None,
            max_shards=2_000_000,
            progress_interval=30.0,
            no_index=False,
            log=None,
            lock=None,
        ),
    ),
    (["status", "--db", "a.db"], "cmd_status", dict(db="a.db")),
    (["verify", "--db", "a.db", "/R/x"], "cmd_verify", dict(db="a.db", path="/R/x")),
    (
        ["report", "--db", "a.db"],
        "cmd_report",
        dict(
            db="a.db",
            top=25,
            hotspot_min_files=100_000,
            hotspot_max_mean=65536,
            archive_db=None,
            rules=None,
        ),
    ),
    (["export", "--db", "a.db", "--out", "d"], "cmd_export", dict(db="a.db", out="d")),
    (["index", "--db", "a.db"], "cmd_index", dict(db="a.db")),
    (
        ["tree", "--db", "a.db", "--out", "t.html", "--min-size", "100MiB"],
        "cmd_tree",
        dict(
            db="a.db",
            out="t.html",
            subtree=None,
            min_size=100 << 20,
            max_rows=100_000,
            archive_db=None,
            rules=None,
        ),
    ),
    (
        ["kinds", "--db", "a.db"],
        "cmd_kinds",
        dict(db="a.db", archive_db=None, rules=None, subtree=None, out=None),
    ),
    (
        ["archive", "register", "--db", "x.db", "--name", "a.tar", "--folder", "/R"],
        "cmd_archive_register",
        dict(db="x.db", name="a.tar", folder="/R", local_dir=None, remote="dropbox"),
    ),
    (
        ["archive", "index", "--db", "x.db", "--archive", "a.tar", "--recheck"],
        "cmd_archive_index",
        dict(
            db="x.db",
            archive="a.tar",
            remote="dropbox",
            workers=8,
            rps=5.0,
            window_min=WINDOW_MIN,
            window_max=WINDOW_MAX,
            batch=2000,
            max_batches=0,
            recheck=True,
            lock=None,
        ),
    ),
    (
        [
            "archive",
            "run",
            "--db",
            "x.db",
            "--audit-db",
            "a.db",
            "--streams",
            "6",
            "--max-streams",
            "9",
            "--limit",
            "3",
            "--dry-run",
            "--lock",
            "l",
        ],
        "cmd_archive_run",
        dict(
            db="x.db",
            audit_db="a.db",
            remote="dropbox",
            streams=6,
            max_streams=9,
            workers=8,
            limit=3,
            dry_run=True,
            window_min=WINDOW_MIN,
            window_max=WINDOW_MAX,
            batch=2000,
            lock="l",
            max_batches=0,
        ),
    ),
    (
        [
            "archive",
            "import",
            "--db",
            "x.db",
            "--from",
            "y.db",
            "--from-lock",
            "y.lock",
        ],
        "cmd_archive_import",
        dict(db="x.db", source="y.db", lock=None, from_lock="y.lock"),
    ),
    (
        ["archive", "forget", "--db", "x.db", "--archive", "a.tar", "--folder", "/R"],
        "cmd_archive_forget",
        dict(db="x.db", archive="a.tar", folder="/R", lock=None),
    ),
    (
        ["archive", "status", "--db", "x.db"],
        "cmd_archive_status",
        dict(db="x.db", archive=None),
    ),
    (
        ["archive", "report", "--db", "x.db", "--archive", "a.tar"],
        "cmd_archive_report",
        dict(db="x.db", archive="a.tar", top=25),
    ),
    (
        ["archive", "export", "--db", "x.db", "--archive", "a.tar", "--out", "d"],
        "cmd_archive_export",
        dict(db="x.db", archive="a.tar", out="d"),
    ),
    (
        [
            "archive",
            "cat",
            "--db",
            "x.db",
            "--archive",
            "a.tar",
            "--member",
            "m",
            "--offset",
            "512",
        ],
        "cmd_archive_cat",
        dict(
            db="x.db",
            archive="a.tar",
            member="m",
            offset=512,
            out=None,
            remote="dropbox",
        ),
    ),
]


@pytest.mark.parametrize(
    "argv,body,expected", BEFORE, ids=[" ".join(argv[:2]) for argv, _, _ in BEFORE]
)
def test_each_command_gets_the_arguments_it_got_before(
    monkeypatch, argv, body, expected
):
    got = []
    monkeypatch.setattr(cli, body, lambda args: got.append(args) or 0)
    assert main(argv) == 0
    assert vars(got[0]) == {**expected, "verbose": False}


def test_verbose_goes_before_the_command(monkeypatch):
    got = []
    monkeypatch.setattr(cli, "cmd_status", lambda args: got.append(args) or 0)
    assert main(["--verbose", "status", "--db", "a.db"]) == 0
    assert got[0].verbose is True


def test_a_body_s_exit_code_is_the_command_s(monkeypatch):
    monkeypatch.setattr(cli, "cmd_status", lambda args: 3)
    assert main(["status", "--db", "a.db"]) == 3


def test_a_missing_option_is_a_usage_error(capsys):
    assert main(["status"]) == 2
    assert "Missing option '--db'" in capsys.readouterr().err


def test_a_word_where_a_number_goes_is_a_usage_error(capsys):
    assert main(["report", "--db", "a.db", "--top", "many"]) == 2
    assert "'many' is not a valid int" in capsys.readouterr().err


def test_no_command_shows_the_commands(capsys):
    assert main([]) == 2
    shown = capsys.readouterr()
    assert "archive" in shown.out + shown.err


@pytest.mark.parametrize("where", [[], ["--folder", "/R", "--local-dir", "d"]])
def test_register_takes_exactly_one_place(where, capsys):
    assert main(["archive", "register", "--db", "x.db", "--name", "a.tar", *where]) == 2
    assert "exactly one of --folder and --local-dir" in capsys.readouterr().err


def test_arguments_fills_in_a_commands_defaults():
    args = cli.arguments(
        cli.archive_index_command, db="x.db", archive="a.tar", workers=4
    )
    assert (
        args.db,
        args.archive,
        args.workers,
        args.batch,
        args.recheck,
        args.verbose,
    ) == ("x.db", "a.tar", 4, 2000, False, False)
