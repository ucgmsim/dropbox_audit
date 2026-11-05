import json
import logging
import os
import re
import runpy
import signal
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

import pytest

from dbaudit import cli
from dbaudit.api import ApiError, Page
from dbaudit.archive.reader import WINDOW_MAX, WINDOW_MIN
from dbaudit.auth import AuthError, TokenProvider
from dbaudit.cli import human_bytes, human_duration, main
from dbaudit.store import Store
from dbaudit.tree import Page as TreePage
from dbaudit.tree import Summary
from dbaudit.verify import VerifyResult, verify_subtree
from tests.fakes import TREE, FakeLister
from tests.listing_fakes import LISTED, archive, archive_index, file, listing

# GitHub Actions sets GITHUB_ACTIONS=true on every runner. Typer reads that
# variable when it is imported and, if it is set, styles its help and error
# messages with terminal codes for colour and bold, as if writing to a terminal.
# Those codes can fall inside the phrases these tests look for, so the tests
# strip them first. Only runs under GitHub Actions are affected (or runs with
# FORCE_COLOR or PY_COLORS set). Anywhere else Typer prints plain text, and
# plain() returns it unchanged.
_STYLE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Return ``text`` without terminal styling, as a reader of the help sees it."""
    return _STYLE.sub("", text)


def seeded_db(tmp_path: Path):
    db = tmp_path / "a.db"
    store = Store(db)
    store.init_schema()
    store.set_meta("root", "/R")
    store.add_shard("/R/a", 1, "recursive")
    store.add_shard("/R/b", 1, "recursive")
    store.claim_shard("w")
    store.finish_shard(1)
    return db, store


def test_status_reports_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db, _ = seeded_db(tmp_path)
    assert main(["status", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "1/2" in out
    assert "shards" in out.lower()
    assert "/R" in out


def test_status_lists_failed_shards(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db, store = seeded_db(tmp_path)
    store.fail_shard(2, "transient failures exhausted")
    main(["status", "--db", str(db)])
    out = capsys.readouterr().out
    assert "/R/b" in out and "transient" in out


def test_status_on_uninitialised_db_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["status", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_verify_on_a_database_with_no_root_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An `init` stopped after the schema and before the root leaves such a database."""
    store = Store(tmp_path / "a.db")
    store.init_schema()
    assert main(["verify", "--db", str(tmp_path / "a.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_init_refuses_when_disk_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr("dbaudit.cli.free_bytes", lambda p: 1024)
    rc = main(
        ["init", "--db", str(tmp_path / "a.db"), "--root", "/R", "--min-free-gb", "50"]
    )
    assert rc == 2
    assert "free" in capsys.readouterr().err


def test_run_on_uninitialised_db_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["run", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_run_refuses_when_another_instance_holds_the_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
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
def test_human_bytes(value: int, expected: str):
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
def test_human_duration(value: float, expected: str):
    assert human_duration(value) == expected


def test_report_calls_a_crawl_with_failed_shards_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Nothing is pending or running, but /R/b failed: its files are missing."""
    db, store = seeded_db(tmp_path)
    store.fail_shard(2, "transient failures exhausted")

    def offline(remote: str):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("dbaudit.cli._space_usage", offline)
    main(["report", "--db", str(db)])
    assert "incomplete" in capsys.readouterr().out


def test_init_says_when_dropbox_cannot_look_up_the_account_just_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
def test_every_command_has_help(
    command: tuple[str, ...], capsys: pytest.CaptureFixture[str]
):
    assert main([*command, "--help"]) == 0
    assert "Usage: dbaudit " + " ".join(command) in plain(capsys.readouterr().out)


# What each body got from argparse, so scripts that run dbaudit keep working.
BEFORE = [
    (
        ["init", "--db", "a.db", "--root", "/R"],
        "cmd_init",
        {"db": "a.db", "root": "/R", "remote": "dropbox", "min_free_gb": 30.0},
    ),
    (
        ["run", "--db", "a.db"],
        "cmd_run",
        {
            "db": "a.db",
            "workers": 8,
            "rps": 5.0,
            "max_rps": 20.0,
            "split_depth": 6,
            "queue_target": None,
            "max_shards": 2_000_000,
            "progress_interval": 30.0,
            "no_index": False,
            "log": None,
            "lock": None,
        },
    ),
    (["status", "--db", "a.db"], "cmd_status", {"db": "a.db"}),
    (["verify", "--db", "a.db", "/R/x"], "cmd_verify", {"db": "a.db", "path": "/R/x"}),
    (
        ["report", "--db", "a.db"],
        "cmd_report",
        {
            "db": "a.db",
            "top": 25,
            "hotspot_min_files": 100_000,
            "hotspot_max_mean": 65536,
            "archive_db": None,
            "rules": None,
        },
    ),
    (
        ["export", "--db", "a.db", "--out", "d"],
        "cmd_export",
        {"db": "a.db", "out": "d"},
    ),
    (["index", "--db", "a.db"], "cmd_index", {"db": "a.db"}),
    (
        ["tree", "--db", "a.db", "--out", "t.html", "--min-size", "100MiB"],
        "cmd_tree",
        {
            "db": "a.db",
            "out": "t.html",
            "subtree": None,
            "min_size": 100 << 20,
            "max_rows": 100_000,
            "archive_db": None,
            "rules": None,
        },
    ),
    (
        ["kinds", "--db", "a.db"],
        "cmd_kinds",
        {"db": "a.db", "archive_db": None, "rules": None, "subtree": None, "out": None},
    ),
    (
        ["archive", "register", "--db", "x.db", "--name", "a.tar", "--folder", "/R"],
        "cmd_archive_register",
        {
            "db": "x.db",
            "name": "a.tar",
            "folder": "/R",
            "local_dir": None,
            "remote": "dropbox",
        },
    ),
    (
        ["archive", "index", "--db", "x.db", "--archive", "a.tar", "--recheck"],
        "cmd_archive_index",
        {
            "db": "x.db",
            "archive": "a.tar",
            "remote": "dropbox",
            "workers": 8,
            "rps": 5.0,
            "window_min": WINDOW_MIN,
            "window_max": WINDOW_MAX,
            "batch": 2000,
            "max_batches": 0,
            "recheck": True,
            "lock": None,
        },
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
        {
            "db": "x.db",
            "audit_db": "a.db",
            "remote": "dropbox",
            "streams": 6,
            "max_streams": 9,
            "workers": 8,
            "limit": 3,
            "dry_run": True,
            "window_min": WINDOW_MIN,
            "window_max": WINDOW_MAX,
            "batch": 2000,
            "lock": "l",
            "max_batches": 0,
        },
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
        {"db": "x.db", "source": "y.db", "lock": None, "from_lock": "y.lock"},
    ),
    (
        ["archive", "forget", "--db", "x.db", "--archive", "a.tar", "--folder", "/R"],
        "cmd_archive_forget",
        {"db": "x.db", "archive": "a.tar", "folder": "/R", "lock": None},
    ),
    (
        ["archive", "status", "--db", "x.db"],
        "cmd_archive_status",
        {"db": "x.db", "archive": None},
    ),
    (
        ["archive", "report", "--db", "x.db", "--archive", "a.tar"],
        "cmd_archive_report",
        {"db": "x.db", "archive": "a.tar", "top": 25},
    ),
    (
        ["archive", "export", "--db", "x.db", "--archive", "a.tar", "--out", "d"],
        "cmd_archive_export",
        {"db": "x.db", "archive": "a.tar", "out": "d"},
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
        {
            "db": "x.db",
            "archive": "a.tar",
            "member": "m",
            "offset": 512,
            "out": None,
            "remote": "dropbox",
        },
    ),
]


@pytest.mark.parametrize(
    "argv,body,expected", BEFORE, ids=[" ".join(argv[:2]) for argv, _, _ in BEFORE]
)
def test_each_command_gets_the_arguments_it_got_before(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], body: str, expected: dict
):
    got = []
    monkeypatch.setattr(cli, body, lambda args: got.append(args) or 0)
    assert main(argv) == 0
    assert vars(got[0]) == {**expected, "verbose": False}


def test_verbose_goes_before_the_command(monkeypatch: pytest.MonkeyPatch):
    got = []
    monkeypatch.setattr(cli, "cmd_status", lambda args: got.append(args) or 0)
    assert main(["--verbose", "status", "--db", "a.db"]) == 0
    assert got[0].verbose is True


def test_a_body_s_exit_code_is_the_command_s(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "cmd_status", lambda args: 3)
    assert main(["status", "--db", "a.db"]) == 3


def test_a_missing_option_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    assert main(["status"]) == 2
    assert "Missing option '--db'" in plain(capsys.readouterr().err)


def test_a_word_where_a_number_goes_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
):
    assert main(["report", "--db", "a.db", "--top", "many"]) == 2
    assert "'many' is not a valid int" in plain(capsys.readouterr().err)


def test_no_command_shows_the_commands(capsys: pytest.CaptureFixture[str]):
    assert main([]) == 2
    shown = capsys.readouterr()
    assert "archive" in shown.out + shown.err


@pytest.mark.parametrize("where", [[], ["--folder", "/R", "--local-dir", "d"]])
def test_register_takes_exactly_one_place(
    where: list[str], capsys: pytest.CaptureFixture[str]
):
    assert main(["archive", "register", "--db", "x.db", "--name", "a.tar", *where]) == 2
    assert "exactly one of --folder and --local-dir" in plain(capsys.readouterr().err)


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


def test_main_module_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """``python -m dbaudit`` hands sys.argv to the CLI and exits with its code."""
    monkeypatch.setattr(sys, "argv", ["dbaudit", "--help"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("dbaudit", run_name="__main__")
    assert raised.value.code == 0
    assert "Usage:" in capsys.readouterr().out


# ---- the crawl commands: init, run, status, verify, report, export, tree, index ----


class StubTokens:
    """What `init` asks of a token provider: the account and the team space root."""

    def __init__(self, account: dict[str, object]) -> None:
        self._account = account

    def account(self) -> dict[str, object]:
        return self._account

    def root_namespace_id(self) -> str:
        return "ns-4242"


class ListerFor:
    """A `build_lister` handing out `lister`, noting the remote it was asked for."""

    def __init__(self, lister: FakeLister) -> None:
        self.lister = lister
        self.remotes: list[str] = []

    def __call__(
        self, remote: str, tokens: TokenProvider | None = None
    ) -> tuple[StubTokens, FakeLister]:
        self.remotes.append(remote)
        return StubTokens({}), self.lister


def initialised(tmp_path: Path, remote: str = "dropbox") -> str:
    """An audit database `init` has made: schema, root /R, one pending shard."""
    db = str(tmp_path / "a.db")
    store = Store(db)
    store.init_schema()
    store.set_meta("root", "/R")
    store.set_meta("remote", remote)
    store.add_shard("/R", depth=0)
    store.close()
    return db


def free_terabyte(path: str) -> int:
    """A `free_bytes` that finds a terabyte free."""
    return 1 << 40


def free_two_gib(path: str) -> int:
    """A `free_bytes` that finds 2 GiB free."""
    return 2 * 1024**3


def never_a_dir(path: str) -> bool:
    """An `os.path.isdir` that finds no folder anywhere."""
    return False


def index_count(store: Store) -> int:
    """How many indexes the database holds."""
    return store.query("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")[0][0]


RUN_FAST = ["--workers", "2", "--rps", "1000", "--max-rps", "1000"]


@pytest.fixture
def keep_signal_handlers() -> Iterator[None]:
    """`run` installs signal handlers and root log handlers; put the old ones back."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    root = logging.getLogger()
    handlers = list(root.handlers)
    yield
    for number, handler in saved.items():
        signal.signal(number, handler if handler is not None else signal.SIG_DFL)
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)


def test_free_bytes_measures_the_nearest_folder_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    measured: list[str] = []

    def statvfs(path: str) -> SimpleNamespace:
        measured.append(path)
        return SimpleNamespace(f_bavail=10, f_frsize=4096)

    monkeypatch.setattr("dbaudit.cli.os.statvfs", statvfs)
    assert cli.free_bytes(str(tmp_path / "not" / "yet" / "a.db")) == 40960
    assert measured == [str(tmp_path)]


def test_free_bytes_stops_at_the_top_of_the_tree(
    monkeypatch: pytest.MonkeyPatch,
):
    measured: list[str] = []

    def statvfs(path: str) -> SimpleNamespace:
        measured.append(path)
        return SimpleNamespace(f_bavail=1, f_frsize=512)

    monkeypatch.setattr("dbaudit.cli.os.path.isdir", never_a_dir)
    monkeypatch.setattr("dbaudit.cli.os.statvfs", statvfs)
    assert cli.free_bytes("/TeamSpace/proj00001/a.db") == 512
    assert measured == ["/"]


def test_setup_logging_also_writes_to_a_file(tmp_path: Path):
    logfile = tmp_path / "run.log"
    try:
        cli.setup_logging(verbose=True, logfile=str(logfile))
        cli.log.debug("a line for the file")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "a line for the file" in logfile.read_text()
    finally:
        cli.setup_logging()  # replaces, and closes, the file handler


def test_build_lister_makes_a_provider_for_the_remote_or_reuses_one(
    monkeypatch: pytest.MonkeyPatch,
):
    class Provider:
        def __init__(self, remote: str) -> None:
            self.remote = remote

    class Lister:
        def __init__(self, tokens: Provider) -> None:
            self.tokens = tokens

    monkeypatch.setattr("dbaudit.cli.TokenProvider", Provider)
    monkeypatch.setattr("dbaudit.cli.HttpLister", Lister)
    tokens, lister = cli.build_lister("TeamSpace")
    assert isinstance(tokens, Provider) and tokens.remote == "TeamSpace"
    assert isinstance(lister, Lister) and lister.tokens is tokens
    again, relister = cli.build_lister("other", tokens)
    assert again is tokens and isinstance(relister, Lister)
    assert relister.tokens is tokens


def test_init_seeds_the_root_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    account: dict[str, object] = {
        "account_id": "dbid:proj00001",
        "email": "user@example.org",
        "team": {"name": "TeamSpace"},
    }
    monkeypatch.setattr("dbaudit.cli.free_bytes", free_terabyte)

    def build(
        remote: str, tokens: TokenProvider | None = None
    ) -> tuple[StubTokens, FakeLister]:
        return StubTokens(account), FakeLister(TREE)

    monkeypatch.setattr("dbaudit.cli.build_lister", build)
    db = str(tmp_path / "a.db")
    assert main(["init", "--db", db, "--root", "R/", "--remote", "TeamSpace"]) == 0
    out = capsys.readouterr().out
    assert f"initialised {db}" in out
    assert "root      : /R (2+ entries at the top level)" in out
    assert "account   : user@example.org / team TeamSpace" in out
    assert "namespace : ns-4242" in out
    assert f"next: python -m dbaudit run --db {db}" in out
    store = Store(db)
    assert store.get_meta("root") == "/R"
    assert store.get_meta("remote") == "TeamSpace"
    assert store.get_meta("schema_version") == "2"
    assert store.get_meta("root_namespace_id") == "ns-4242"
    assert store.get_meta("account_id") == "dbid:proj00001"
    assert store.get_meta("team") == "TeamSpace"
    assert store.get_meta("created_at")
    assert store.query("SELECT path, depth, state FROM shards") == [
        ("/R", 0, "pending")
    ]
    store.close()


def test_init_reports_an_auth_error_and_makes_no_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def build(
        remote: str, tokens: TokenProvider | None = None
    ) -> tuple[StubTokens, FakeLister]:
        raise AuthError("rclone has no remote called TeamSpace")

    monkeypatch.setattr("dbaudit.cli.free_bytes", free_terabyte)
    monkeypatch.setattr("dbaudit.cli.build_lister", build)
    db = tmp_path / "a.db"
    assert main(["init", "--db", str(db), "--root", "/R"]) == 2
    assert "error: rclone has no remote called TeamSpace" in capsys.readouterr().err
    assert not db.exists()


def test_init_fails_cleanly_when_the_root_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    lister = FakeLister(TREE, faults={"/R": RuntimeError("connection reset")})
    monkeypatch.setattr("dbaudit.cli.free_bytes", free_terabyte)
    monkeypatch.setattr("dbaudit.cli.build_lister", ListerFor(lister))
    db = tmp_path / "a.db"
    assert main(["init", "--db", str(db), "--root", "/R"]) == 2
    assert (
        "error: could not list /R: RuntimeError: connection reset"
        in capsys.readouterr().err
    )
    assert not db.exists()


def test_init_names_the_free_space_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr("dbaudit.cli.free_bytes", free_two_gib)
    assert (
        main(
            ["init", "--db", str(tmp_path / "a.db"), "--root", "/R"]
            + ["--min-free-gb", "5"]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "only 2.0 GiB free" in err and "requires 5.0 GiB" in err


def test_run_crawls_a_fake_tree_to_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_signal_handlers: None,
):
    db = initialised(tmp_path, remote="TeamSpace")
    build = ListerFor(FakeLister(TREE))
    monkeypatch.setattr("dbaudit.cli.build_lister", build)
    probe = Store(db)
    indexes_before = index_count(probe)
    probe.close()
    assert main(["run", "--db", db, *RUN_FAST]) == 0
    err = capsys.readouterr().err
    assert "crawling /R with 2 workers at 1000.0 req/s" in err
    assert "finished in" in err and "5 files, 4 dirs" in err
    assert "crawl complete; building analysis indexes" in err and "indexes built" in err
    assert build.remotes == ["TeamSpace"]
    store = Store(db)
    assert store.query("SELECT COUNT(*) FROM files") == [(5,)]
    assert store.get_meta("last_pass_completed_at")
    assert store.query("SELECT finished_at IS NOT NULL FROM runs") == [(1,)]
    assert index_count(store) > indexes_before
    store.close()


def test_run_with_no_index_leaves_the_indexes_unbuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_signal_handlers: None,
):
    db = initialised(tmp_path)
    monkeypatch.setattr("dbaudit.cli.build_lister", ListerFor(FakeLister(TREE)))
    assert main(["run", "--db", db, "--no-index", *RUN_FAST]) == 0
    err = capsys.readouterr().err
    assert "finished in" in err and "building analysis indexes" not in err
    store = Store(db)
    assert store.get_meta("last_pass_completed_at")
    store.close()


def test_run_stops_when_the_credentials_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_signal_handlers: None,
):
    def build(
        remote: str, tokens: TokenProvider | None = None
    ) -> tuple[StubTokens, FakeLister]:
        raise AuthError("the refresh token was revoked")

    db = initialised(tmp_path)
    monkeypatch.setattr("dbaudit.cli.build_lister", build)
    assert main(["run", "--db", db]) == 2
    assert "error: the refresh token was revoked" in capsys.readouterr().err


def test_run_exits_1_when_a_shard_ends_in_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_signal_handlers: None,
):
    db = initialised(tmp_path)
    lister = FakeLister(TREE, faults={"/R": ApiError("access denied")})
    monkeypatch.setattr("dbaudit.cli.build_lister", ListerFor(lister))
    assert main(["run", "--db", db, *RUN_FAST]) == 1
    err = capsys.readouterr().err
    assert "1 shard(s) ended in error; see `status`" in err
    store = Store(db)
    assert store.query("SELECT state, error FROM shards") == [
        ("error", "ApiError: access denied")
    ]
    assert store.get_meta("last_pass_completed_at") is None
    store.close()


class SignallingLister(FakeLister):
    """On its first listing, sends the process SIGINT and then SIGTERM.

    It waits after each for the CLI's handler to have logged, so the handlers have run
    before the crawl is allowed on.
    """

    def __init__(self) -> None:
        super().__init__(TREE)
        self.logged = {
            "signal 2": threading.Event(),
            "second signal": threading.Event(),
        }
        self.sent = False

    def list_folder(self, path: str, recursive: bool) -> Page:
        if not self.sent:
            self.sent = True
            os.kill(os.getpid(), signal.SIGINT)
            assert self.logged["signal 2"].wait(10)
            os.kill(os.getpid(), signal.SIGTERM)
            assert self.logged["second signal"].wait(10)
        return super().list_folder(path, recursive)


def test_run_finishes_the_page_on_a_signal_and_leaves_at_once_on_a_second(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_signal_handlers: None,
):
    db = initialised(tmp_path)
    lister = SignallingLister()
    exits: list[int] = []

    class Watch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            for prefix, seen in lister.logged.items():
                if record.getMessage().startswith(prefix):
                    seen.set()

    watch = Watch()
    logging.getLogger("dbaudit").addHandler(watch)
    monkeypatch.setattr("dbaudit.cli.os._exit", exits.append)
    monkeypatch.setattr("dbaudit.cli.build_lister", ListerFor(lister))
    try:
        assert main(["run", "--db", db, "--workers", "1", *RUN_FAST[2:]]) == 0
    finally:
        logging.getLogger("dbaudit").removeHandler(watch)
    err = capsys.readouterr().err
    assert "signal 2: finishing the page in flight, then exiting" in err
    assert "second signal: exiting immediately" in err
    assert exits == [1]
    store = Store(db)
    left = store.pending_count() + store.running_count()
    assert left > 0
    assert f"interrupted with {left} shard(s) outstanding; re-run to resume" in err
    assert "building analysis indexes" not in err
    assert store.get_meta("last_pass_completed_at") is None
    store.close()


class Ticks(threading.Event):
    """An Event whose `wait` runs the next step and returns False, then returns True."""

    def __init__(self, steps: list[Callable[[], None]]) -> None:
        super().__init__()
        self.steps = steps

    def wait(self, timeout: float | None = None) -> bool:
        if not self.steps:
            return True
        self.steps.pop(0)()
        return False


class Clock:
    """Stands in for the `time` module: each `time()` is the next of the given readings."""

    def __init__(self, *readings: float) -> None:
        self.readings = iter(readings)

    def time(self) -> float:
        return next(self.readings)


class FlakyStore(Store):
    """A store whose first progress read fails, as it does when the crawl holds a lock."""

    failures = 1

    def progress(self) -> dict:
        if self.failures:
            self.failures -= 1
            raise sqlite3.OperationalError("database is locked")
        return super().progress()


def test_the_progress_reporter_logs_counts_then_rate_and_eta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    db, writer = seeded_db(tmp_path)

    def files(count: int) -> Callable[[], None]:
        def bump() -> None:
            writer.connect().execute(
                "UPDATE shards SET n_files=?, n_dirs=2, bytes=2048 WHERE id=1", (count,)
            )

        return bump

    # Ticks: a failed read (skipped), 100 files (no rate yet), then 700 ten seconds on.
    stop = Ticks([lambda: None, files(100), files(700)])
    monkeypatch.setattr(cli, "time", Clock(1000.0, 1010.0, 1020.0))
    reader = FlakyStore(db)
    with caplog.at_level(logging.INFO, logger="dbaudit"):
        cli._progress_reporter(reader, stop, 30.0)
    reader.close()
    lines = [r.getMessage() for r in caplog.records if r.name == "dbaudit"]
    assert lines == [
        "shards 1/2 done (0 error) | 100 files, 2 dirs, 2.0 KiB",
        "shards 1/2 done (0 error) | 700 files, 2 dirs, 2.0 KiB | 60 files/s | eta~11s",
    ]
    writer.close()


def test_the_progress_reporter_reports_no_eta_while_nothing_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    db, writer = seeded_db(tmp_path)
    writer.connect().execute("UPDATE shards SET state='pending' WHERE id=1")

    def files(count: int) -> Callable[[], None]:
        def bump() -> None:
            writer.connect().execute("UPDATE shards SET n_files=? WHERE id=1", (count,))

        return bump

    reader = Store(db)
    stop = Ticks([files(10), files(50)])
    monkeypatch.setattr(cli, "time", Clock(0.0, 1.0, 2.0))
    with caplog.at_level(logging.INFO, logger="dbaudit"):
        cli._progress_reporter(reader, stop, 30.0)
    reader.close()
    last = [r.getMessage() for r in caplog.records if r.name == "dbaudit"][-1]
    assert last == "shards 0/2 done (0 error) | 50 files, 0 dirs, 0 B | 40 files/s"
    writer.close()


def test_status_says_how_long_ago_the_last_pass_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    db, store = seeded_db(tmp_path)
    store.set_meta("last_pass_completed_at", 10_000 - 3700)
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=lambda: 10_000.0))
    assert main(["status", "--db", str(db)]) == 0
    assert "last pass : completed 1h 1m ago" in capsys.readouterr().out
    store.close()


VERIFY_FILES = (file("/R/a.bin", 10), file("/R/c.bin", 7))


def verifying(
    output: str | Exception,
) -> Callable[[Store, str, Callable[[str], str] | None, str | None], VerifyResult]:
    """A `verify_subtree` over a listing it is given, so no rclone runs."""

    def verify(
        store: Store,
        path: str,
        rclone_fn: Callable[[str], str] | None = None,
        remote: str | None = None,
    ) -> VerifyResult:
        def listed(subtree: str) -> str:
            if isinstance(output, Exception):
                raise output
            return output

        return verify_subtree(store, path, rclone_fn=listed, remote=remote)

    return verify


def test_verify_on_a_missing_database_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["verify", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_verify_passes_when_rclone_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, *VERIFY_FILES)
    agreed = json.dumps([{"Path": "a.bin", "Size": 10}, {"Path": "c.bin", "Size": 7}])
    monkeypatch.setattr("dbaudit.cli.verify_subtree", verifying(agreed))
    assert main(["verify", "--db", store.path]) == 0
    out = capsys.readouterr().out
    assert "verifying /R against an independent rclone walk" in out
    assert "database : 2 files, 17 B" in out and "rclone   : 2 files, 17 B" in out
    assert "0 differences" in out
    store.close()


def test_verify_reports_paths_missing_from_the_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, *VERIFY_FILES, file("/R/d.bin", 3))
    listed = json.dumps(
        [
            {"Path": "a.bin", "Size": 10},
            {"Path": "b.bin", "Size": 5},
            {"Path": "c.bin", "Size": 9},
        ]
    )
    monkeypatch.setattr("dbaudit.cli.verify_subtree", verifying(listed))
    assert main(["verify", "--db", store.path, "/R"]) == 1
    out = capsys.readouterr().out
    assert "3 differences" in out
    assert "missing from audit : b.bin" in out
    assert "not seen by rclone : d.bin" in out
    assert "size differs       : c.bin (audit 7, rclone 9)" in out
    store.close()


def test_verify_exits_2_when_the_rclone_walk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, *VERIFY_FILES)
    failure = RuntimeError("rclone lsjson failed: directory not found")
    monkeypatch.setattr("dbaudit.cli.verify_subtree", verifying(failure))
    assert main(["verify", "--db", store.path]) == 2
    assert "error: rclone lsjson failed: directory not found" in (
        capsys.readouterr().err
    )
    store.close()


def space_of(
    account: int, team: int | None, allocated: int | None
) -> Callable[[str], tuple[int, int | None, int | None]]:
    """A `_space_usage` that reports these figures."""

    def space(remote: str) -> tuple[int, int | None, int | None]:
        return account, team, allocated

    return space


def test_space_usage_reads_the_account_and_team_figures(
    monkeypatch: pytest.MonkeyPatch,
):
    posted: list[tuple[str, dict[str, str]]] = []

    class Tokens:
        def __init__(self, remote: str) -> None:
            self.remote = remote

        def access_token(self) -> str:
            return f"token-for-{self.remote}"

    class Reply:
        def __init__(self, body: dict[str, object]) -> None:
            self.body = body

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return self.body

    replies: list[dict[str, object]] = [
        {"used": 500, "allocation": {"used": 900, "allocated": 2000}},
        {"used": 500},
        {},
    ]
    bodies = iter(replies)

    def post(
        url: str, data: object = None, json: object = None, **kwargs: object
    ) -> Reply:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        posted.append((url, headers))
        return Reply(next(bodies))

    monkeypatch.setattr("dbaudit.cli.TokenProvider", Tokens)
    monkeypatch.setattr("requests.post", post)
    assert cli._space_usage("TeamSpace") == (500, 900, 2000)
    assert cli._space_usage("TeamSpace") == (500, None, None)
    assert cli._space_usage("TeamSpace") == (0, None, None)
    assert posted[0] == (
        "https://api.dropboxapi.com/2/users/get_space_usage",
        {"Authorization": "Bearer token-for-TeamSpace"},
    )


def crawled(tmp_path: Path):
    """A finished audit with two copies of one file, and two small files in /R/a."""
    return listing(
        tmp_path,
        file("/R/a/x1.bin", 1000, "same"),
        file("/R/a/x2.bin", 1000, "same"),
        file("/R/b/y.txt", 500),
    )


def test_report_on_a_missing_database_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["report", "--db", str(tmp_path / "missing.db")]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_report_prints_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    store = crawled(tmp_path)
    monkeypatch.setattr("dbaudit.cli._space_usage", space_of(5000, 8000, 16000))
    assert (
        main(
            ["report", "--db", store.path]
            + ["--hotspot-min-files", "2", "--hotspot-max-mean", "100000"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "# Storage audit of /R" in out
    assert "3 files," in out and "2.4 KiB of live data" in out
    assert "measured live bytes under /R: 2.4 KiB" in out
    assert "Dropbox reports for this account : 4.9 KiB" in out
    assert "team space used / allocated      : 7.8 KiB / 15.6 KiB" in out
    assert "unaccounted for                  : 2.4 KiB" in out
    assert "1000 B reclaimable" in out
    assert any(
        "x2" in line and "1000 B" in line and "/R/a/" in line
        for line in out.splitlines()
    )
    hotspot = out.split("## Small-file hotspots")[1].split("##")[0]
    assert "2 files" in hotspot and "/R/a" in hotspot
    assert "WARNING" not in out
    store.close()


def test_report_without_team_figures_leaves_out_the_team_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    store = crawled(tmp_path)
    monkeypatch.setattr("dbaudit.cli._space_usage", space_of(5000, None, None))
    assert main(["report", "--db", store.path]) == 0
    out = capsys.readouterr().out
    assert "Dropbox reports for this account : 4.9 KiB" in out
    assert "team space used" not in out
    assert "none with >=100,000 files averaging <=64.0 KiB" in out
    store.close()


def test_report_goes_on_when_live_space_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def offline(remote: str) -> tuple[int, int | None, int | None]:
        raise RuntimeError("no network in tests")

    store = crawled(tmp_path)
    monkeypatch.setattr("dbaudit.cli._space_usage", offline)
    assert main(["report", "--db", store.path]) == 0
    out = capsys.readouterr().out
    assert "(could not read live space usage: RuntimeError: no network in tests)" in out
    assert "## Reclaimable duplicates" in out
    store.close()


def test_export_writes_the_csvs(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    store = crawled(tmp_path)
    out_dir = tmp_path / "csv"
    assert main(["export", "--db", store.path, "--out", str(out_dir)]) == 0
    wrote = capsys.readouterr().out.splitlines()
    assert wrote and all(line.startswith("wrote ") for line in wrote)
    assert f"wrote {out_dir / 'duplicates.csv'}" in wrote
    assert "same" not in (out_dir / "top_dirs.csv").read_text()
    assert "/R/a" in (out_dir / "top_dirs.csv").read_text()
    store.close()


def test_export_on_a_missing_database_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["export", "--db", str(tmp_path / "x.db"), "--out", "d"]) == 2
    assert "not initialised" in capsys.readouterr().err


def test_index_builds_the_indexes(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    store = crawled(tmp_path)
    before = index_count(store)
    try:
        assert main(["index", "--db", store.path]) == 0
        root_handlers = list(logging.getLogger().handlers)
    finally:
        cli.setup_logging()
    assert index_count(store) > before
    assert root_handlers  # `index` set the logging up itself
    store.close()


def test_tree_reads_the_archive_index_when_given_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, file("/R/t/a.tar", 700, "A"))
    index = archive_index(
        tmp_path,
        archive(
            "a.tar", "/R/t", [("a.tar", 700, "A")], [("", "vp3dfile.p", 700, LISTED)]
        ),
    )
    out = tmp_path / "t.html"
    assert (
        main(
            ["tree", "--db", store.path, "--out", str(out), "--min-size", "1"]
            + ["--archive-db", index]
        )
        == 0
    )
    assert f"wrote {out}: " in capsys.readouterr().out
    page = out.read_text(encoding="utf-8")
    row = page[page.index('data-path="t"') :]
    assert '<span class="badge kind">vm 100%</span>' in row[: row.index("</summary>")]
    store.close()


def test_a_failed_render_leaves_no_half_written_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def render(out: TextIO, page: TreePage, summary: Summary, chosen: dict) -> None:
        raise OSError("disk full")

    store = crawled(tmp_path)
    out = tmp_path / "t.html"
    monkeypatch.setattr("dbaudit.cli.tree.render", render)
    with pytest.raises(OSError, match="disk full"):
        main(["tree", "--db", store.path, "--out", str(out)])
    assert not out.exists() and not list(tmp_path.glob(".tree-*"))
    store.close()


def test_main_prints_a_message_exit_and_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def exits_with_a_message(args: SimpleNamespace) -> int:
        raise SystemExit("the disk is on fire")

    monkeypatch.setattr(cli, "cmd_status", exits_with_a_message)
    assert main(["status", "--db", "a.db"]) == 1
    assert "the disk is on fire" in capsys.readouterr().err


def test_main_returns_0_when_a_command_exits_with_no_code(
    monkeypatch: pytest.MonkeyPatch,
):
    def exits_quietly(args: SimpleNamespace) -> int:
        raise SystemExit

    monkeypatch.setattr(cli, "cmd_status", exits_quietly)
    assert main(["status", "--db", "a.db"]) == 0
