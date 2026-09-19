"""`dbaudit.archive.report`: rollups over one archive's manifest, and CSV export.

Everything here is local and offline (T9-10): archives are built with `build_tar` and
split into part files with `write_parts`, exactly as the other archive tests do. The
two brief tests (`test_summary_rolls_up_by_directory_and_extension`,
`test_export_writes_the_manifest_and_rollups`) are Task 9's own spec; the rest are the
task-9-rulings.md tests, numbered by the ruling they discriminate.
"""

import csv
import tarfile
import time as time_mod
from datetime import datetime, timezone

from dbaudit.archive.report import summary, write_csv
from dbaudit.archive.store import ArchiveStore
from dbaudit.cli import main
from tests.archive_fakes import build_tar, write_parts

MEMBERS = ([(f"run/fault_a/rel{i:02d}/out.bin", b"x" * 1000) for i in range(5)]
           + [(f"run/fault_b/rel{i:02d}/out.bin", b"y" * 2000) for i in range(3)]
           + [("run/notes.txt", b"hello\n")])


def indexed(tmp_path):
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(MEMBERS), part_size=8192)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar"])
    store = ArchiveStore(db)
    return store, store.get("a.tar")["id"], db


def build_and_index(tmp_path, members, part_size=8192, name="a.tar", db_name="archives.db"):
    """Like `indexed`, but with a caller-supplied member list."""
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(members), part_size, base=name)
    db = str(tmp_path / db_name)
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", name])
    main(["archive", "index", "--db", db, "--archive", name])
    store = ArchiveStore(db)
    return store, store.get(name)["id"], db


# ---- the brief's own tests (Step 1) ----------------------------------------------


def test_summary_rolls_up_by_directory_and_extension(tmp_path):
    store, archive_id, _ = indexed(tmp_path)
    result = summary(store, archive_id)
    assert result["n_members"] == len(MEMBERS)
    tops = dict(result["by_top_dir"])
    assert tops["run"]["bytes"] == 5 * 1000 + 3 * 2000 + 6
    by_ext = dict(result["by_extension"])
    assert by_ext["bin"]["count"] == 8 and by_ext["txt"]["count"] == 1
    assert result["largest"][0]["size"] == 2000


def test_export_writes_the_manifest_and_rollups(tmp_path):
    store, archive_id, db = indexed(tmp_path)
    out = tmp_path / "exports"
    write_csv(store, archive_id, out)
    names = {p.name for p in out.iterdir()}
    assert {"manifest.csv", "by_dir.csv", "by_extension.csv"} <= names
    assert len((out / "manifest.csv").read_text().splitlines()) == len(MEMBERS) + 1


# ---- T9-7: unknown --archive or a missing --db exit 2, for both commands --------


def test_report_and_export_on_unknown_archive_or_missing_db_exit_2(tmp_path, capsys):
    store, archive_id, db = indexed(tmp_path)
    capsys.readouterr()

    assert main(["archive", "report", "--db", db, "--archive", "nope.tar"]) == 2
    assert "nope.tar" in capsys.readouterr().err

    missing = str(tmp_path / "missing.db")
    assert main(["archive", "report", "--db", missing, "--archive", "a.tar"]) == 2

    out_dir = tmp_path / "out1"
    assert main(["archive", "export", "--db", db, "--archive", "nope.tar",
                 "--out", str(out_dir)]) == 2
    assert "nope.tar" in capsys.readouterr().err
    assert not out_dir.exists()

    out_dir2 = tmp_path / "out2"
    assert main(["archive", "export", "--db", missing, "--archive", "a.tar",
                 "--out", str(out_dir2)]) == 2


# ---- T9-8: a non-complete or stale archive warns before the figures --------------


def test_report_on_a_partially_walked_archive_warns_before_the_figures(tmp_path, capsys):
    # A deliberately interrupted walk: real, non-zero figures attached to a state that
    # is not 'complete' -- the exact shape of Task 11's smoke test (a manifest that is
    # 12% walked must not read like a finished archive's).
    members = [(f"run/file{i:02d}.bin", b"x" * 100) for i in range(40)]
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(members), part_size=8192)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    main(["archive", "index", "--db", db, "--archive", "a.tar",
          "--batch", "5", "--max-batches", "1"])
    store = ArchiveStore(db)
    row = store.get("a.tar")
    n_members = store.stats(row["id"])["n_members"]
    assert row["state"] != "complete"          # sanity: the walk really did stop early
    assert n_members > 0                       # ... and it isn't merely unindexed
    capsys.readouterr()

    assert main(["archive", "report", "--db", db, "--archive", "a.tar"]) == 0
    out = capsys.readouterr().out
    assert "lower bound" in out
    warn_at = out.index("lower bound")
    figures_at = out.index(str(n_members))
    assert warn_at < figures_at


def test_report_on_a_stale_archive_says_so(tmp_path, capsys):
    store, archive_id, db = indexed(tmp_path)
    store.mark_stale(archive_id, "parts changed underneath the index")
    capsys.readouterr()

    assert main(["archive", "report", "--db", db, "--archive", "a.tar"]) == 0
    out = capsys.readouterr().out
    assert "lower bound" in out
    # Not just "assert 'stale' in out": the generic not-complete warning already
    # interpolates the raw state name ("state: stale"), so that alone would pass even
    # without a dedicated stale note. Pin the text that only the dedicated note has.
    assert "underneath this index" in out.lower()
    assert "no longer there" in out.lower()


# ---- T9-9: a registered-but-unindexed archive reports cleanly -------------------


def test_report_on_an_unindexed_archive_says_so_and_returns_0(tmp_path, capsys):
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, build_tar(MEMBERS), part_size=8192)
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    capsys.readouterr()

    assert main(["archive", "report", "--db", db, "--archive", "a.tar"]) == 0
    out = capsys.readouterr().out
    assert "not indexed" in out.lower()

    # summary() itself must not crash on an empty manifest either -- guard every
    # denominator, not just the CLI's own printing.
    store = ArchiveStore(db)
    archive_id = store.get("a.tar")["id"]
    result = summary(store, archive_id)
    assert result["n_members"] == 0
    assert result["mtime_span"] is None
    assert result["largest"] == []


# ---- T9-3: mtime_span is UTC, not the process's local timezone ------------------


def test_mtime_span_is_utc_even_when_the_local_timezone_is_not(tmp_path, monkeypatch):
    """Discriminates T9-3. `datetime.fromtimestamp(ts)` without `tz=utc` silently reads
    the epoch second through the process's local timezone; `utcfromtimestamp` would
    instead fail the whole suite outright (pytest.ini turns its DeprecationWarning into
    an error). Forcing the local zone to UTC+14 and choosing timestamps at 23:30 UTC
    makes the naive-local mistake roll the date over to the next day, so comparing
    against the UTC-computed date catches it even on a test machine whose own local
    zone happens to already be UTC.
    """
    earliest_ts = int(datetime(2001, 9, 9, 23, 30, 0, tzinfo=timezone.utc).timestamp())
    latest_ts = int(datetime(2020, 1, 1, 23, 30, 0, tzinfo=timezone.utc).timestamp())
    members = [("run/a.bin", b"a", {"mtime": earliest_ts}),
               ("run/b.bin", b"b", {"mtime": latest_ts})]
    store, archive_id, db = build_and_index(tmp_path, members)

    monkeypatch.setenv("TZ", "Pacific/Kiritimati")  # UTC+14, no DST
    time_mod.tzset()
    try:
        result = summary(store, archive_id)
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time_mod.tzset()

    expected = (datetime.fromtimestamp(earliest_ts, tz=timezone.utc).date().isoformat(),
                datetime.fromtimestamp(latest_ts, tz=timezone.utc).date().isoformat())
    assert result["mtime_span"] == expected


# ---- T9-4: owners and types cover directories, symlinks and two owners ----------


def test_owners_and_types_report_directories_symlinks_and_two_owners(tmp_path):
    members = [
        ("run/file_a.bin", b"a" * 100, {"uname": "alice", "gname": "teamA"}),
        ("run/file_b.bin", b"b" * 200, {"uname": "alice", "gname": "teamA"}),
        ("run/file_c.bin", b"c" * 300, {"uname": "bob", "gname": "teamB"}),
        ("run/subdir/", b"", {"type": tarfile.DIRTYPE, "uname": "bob", "gname": "teamB"}),
        ("run/link_to_a", b"", {"type": tarfile.SYMTYPE, "linkname": "file_a.bin",
                                 "uname": "alice", "gname": "teamA"}),
    ]
    store, archive_id, _ = build_and_index(tmp_path, members)

    result = summary(store, archive_id)
    owners = dict(result["owners"])
    assert owners[("alice", "teamA")]["count"] == 3
    assert owners[("bob", "teamB")]["count"] == 2

    types = dict(result["types"])
    assert types[tarfile.REGTYPE.decode()]["count"] == 3
    assert types[tarfile.DIRTYPE.decode()]["count"] == 1
    assert types[tarfile.SYMTYPE.decode()]["count"] == 1


# ---- T9-6: a comma-and-quote path round-trips through manifest.csv --------------


def test_manifest_csv_round_trips_a_path_with_comma_and_quote(tmp_path):
    tricky_name = 'odd, name "with quotes".bin'
    members = [("run/" + tricky_name, b"z" * 42)]
    store, archive_id, _ = build_and_index(tmp_path, members)

    out = tmp_path / "exports"
    write_csv(store, archive_id, out)

    with open(out / "manifest.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    paths = [row[0] for row in rows[1:]]
    assert f"run/{tricky_name}" in paths


# ---- T9-5: by_depth groups by the first 1, 2 and 3 path components --------------


def test_by_depth_groups_by_first_1_2_3_path_components(tmp_path):
    members = [
        ("run/fault_a/rel01/output/seismo.bin", b"x" * 1000),
        ("run/fault_a/rel02/output/seismo.bin", b"y" * 500),
        ("run/fault_b/rel01/output/seismo.bin", b"z" * 250),
        ("top_level.txt", b"q" * 10),  # dir == "" -> "(root)" at every depth
    ]
    store, archive_id, _ = build_and_index(tmp_path, members)

    result = summary(store, archive_id)
    depth1 = dict(result["by_depth"][1])
    depth2 = dict(result["by_depth"][2])
    depth3 = dict(result["by_depth"][3])

    assert depth1["run"]["bytes"] == 1000 + 500 + 250
    assert depth1["(root)"]["bytes"] == 10

    assert depth2["run/fault_a"]["bytes"] == 1500
    assert depth2["run/fault_b"]["bytes"] == 250
    assert depth2["(root)"]["bytes"] == 10

    assert depth3["run/fault_a/rel01"]["bytes"] == 1000
    assert depth3["run/fault_a/rel02"]["bytes"] == 500
    assert depth3["run/fault_b/rel01"]["bytes"] == 250
    assert depth3["(root)"]["bytes"] == 10


# ---- T9-4: --top limits `largest` ------------------------------------------------


def test_top_limits_the_largest_list(tmp_path):
    store, archive_id, _ = indexed(tmp_path)  # sizes: 1000 x5, 2000 x3, 6 x1
    result = summary(store, archive_id, top=2)
    assert len(result["largest"]) == 2
    assert all(m["size"] == 2000 for m in result["largest"])


# ---- T9-6: export creates a missing --out dir and names what it wrote -----------


def test_export_command_creates_the_out_dir_and_names_what_it_wrote(tmp_path, capsys):
    store, archive_id, db = indexed(tmp_path)
    out_dir = tmp_path / "new_export_dir"
    assert not out_dir.exists()
    capsys.readouterr()

    assert main(["archive", "export", "--db", db, "--archive", "a.tar",
                 "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert out_dir.is_dir()
    for filename in ("manifest.csv", "by_dir.csv", "by_extension.csv"):
        assert (out_dir / filename).exists()
        assert filename in out
