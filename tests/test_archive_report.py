"""`dbaudit.archive.report`: rollups over one archive's manifest, and CSV export.

Everything here is local and offline (T9-10): archives are built with `build_tar` and
split into part files with `write_parts`, exactly as the other archive tests do. The
two brief tests (`test_summary_rolls_up_by_directory_and_extension`,
`test_export_writes_the_manifest_and_rollups`) are Task 9's own spec; the middle
section is the task-9-rulings.md tests, numbered by the ruling they discriminate; the
"fix round 1" section at the bottom covers the two Important findings from that
review (materialising the whole manifest into memory, and `export` carrying no
completeness warning).
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
    assert "may be missing members" in out
    warn_at = out.index("may be missing members")
    figures_at = out.index(str(n_members))
    assert warn_at < figures_at


def test_report_on_a_stale_archive_says_so(tmp_path, capsys):
    store, archive_id, db = indexed(tmp_path)
    store.mark_stale(archive_id, "parts changed underneath the index")
    capsys.readouterr()

    assert main(["archive", "report", "--db", db, "--archive", "a.tar"]) == 0
    out = capsys.readouterr().out
    assert "may be missing members" in out
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


# =====================================================================================
# Fix round 1: two Important findings from review.
#
# 1. summary()/write_csv() used to read every member into one Python list
#    (`query_members`'s `fetchall`) before rolling it up -- measured at 521 MB / 0.79 s
#    for a 900,000-row manifest, and `largest = sorted(members, ...)[:top]` sorted the
#    whole thing in Python where `members_size` (an index `build_indexes` already
#    creates) answers the same query directly. Fixed by streaming the per-row pass
#    (`ArchiveStore.stream`, selecting only the columns actually read) and moving
#    `largest` into an indexed SQL query, both inside one `ArchiveStore.read_transaction()`
#    so the snapshot guarantee is explicit rather than an accident of reading everything
#    in one call.
# 2. `export` printed no lower-bound/stale warning, unlike `report`. Fixed by sharing
#    `_print_index_completeness_warning` between the two commands.
#
# Per the fix-round ruling: the streaming change is pinned first against the exact
# output the pre-fix implementation produced (byte-identical), then separately proven
# to actually stream via a row-count check -- not a functional one, since the outputs
# either side of that particular change are supposed to be identical.
# =====================================================================================

# A size tie (dup1.bin/dup2.bin, both 1500 bytes) is deliberate: it is what makes the
# `largest` SQL query's tie-break order (`hdr_offset ASC`) an actual behavioural claim
# rather than an untested assumption that it matches the old Python stable sort.
GOLDEN_MEMBERS = (
    [(f"run/fault_a/rel{i:02d}/out.bin", b"x" * 1000) for i in range(5)]
    + [(f"run/fault_b/rel{i:02d}/out.bin", b"y" * 2000) for i in range(3)]
    + [("run/notes.txt", b"hello\n")]
    + [("run/dup1.bin", b"d" * 1500), ("run/dup2.bin", b"e" * 1500)]
    + [("run/link_to_notes", b"", {"type": tarfile.SYMTYPE, "linkname": "notes.txt",
                                    "uname": "bob", "gname": "teamB"})]
)

# Captured verbatim from `summary(store, archive_id, top=5)` against GOLDEN_MEMBERS
# under the pre-streaming-refactor implementation (commit 38fae10) -- see
# task-9-report.md's fix-round-1 section for the capture script and its output.
# `largest`'s sqlite3.Row objects are reduced to (dir, name, size) tuples, the only
# fields anything reads off them, both here and in the real code after the fix.
EXPECTED_SUMMARY = {
    "state": "complete",
    "n_parts": 4,
    "n_members": 12,
    "member_bytes": 14006,
    "by_top_dir": [("run", {"count": 12, "bytes": 14006})],
    "by_depth": {
        1: [("run", {"count": 12, "bytes": 14006})],
        2: [("run/fault_b", {"count": 3, "bytes": 6000}),
            ("run/fault_a", {"count": 5, "bytes": 5000}),
            ("run", {"count": 4, "bytes": 3006})],
        3: [("run", {"count": 4, "bytes": 3006}),
            ("run/fault_b/rel00", {"count": 1, "bytes": 2000}),
            ("run/fault_b/rel01", {"count": 1, "bytes": 2000}),
            ("run/fault_b/rel02", {"count": 1, "bytes": 2000}),
            ("run/fault_a/rel00", {"count": 1, "bytes": 1000}),
            ("run/fault_a/rel01", {"count": 1, "bytes": 1000}),
            ("run/fault_a/rel02", {"count": 1, "bytes": 1000}),
            ("run/fault_a/rel03", {"count": 1, "bytes": 1000}),
            ("run/fault_a/rel04", {"count": 1, "bytes": 1000})],
    },
    "by_extension": [("bin", {"count": 10, "bytes": 14000}),
                      ("txt", {"count": 1, "bytes": 6}),
                      ("(none)", {"count": 1, "bytes": 0})],
    "largest": [("run/fault_b/rel00", "out.bin", 2000),
                ("run/fault_b/rel01", "out.bin", 2000),
                ("run/fault_b/rel02", "out.bin", 2000),
                ("run", "dup1.bin", 1500),
                ("run", "dup2.bin", 1500)],
    "mtime_span": ("2023-11-14", "2023-11-14"),
    "owners": [(("user", "proj00001"), {"count": 11, "bytes": 14006}),
               (("bob", "teamB"), {"count": 1, "bytes": 0})],
    "types": [("0", {"count": 11, "bytes": 14006}),
              ("2", {"count": 1, "bytes": 0})],
}

# Same capture, for write_csv(). Transcribed via repr() to pin whitespace/newlines
# exactly (T9-6 already requires newline="" and csv.writer; this additionally pins
# the actual bytes so the streamed rewrite cannot silently change them).
EXPECTED_MANIFEST_CSV = (
    "path,size,mtime,mode,uname,gname,type,hdr_offset,data_offset\n"
    "run/fault_a/rel00/out.bin,1000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,0,512\n"
    "run/fault_a/rel01/out.bin,1000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,1536,2048\n"
    "run/fault_a/rel02/out.bin,1000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,3072,3584\n"
    "run/fault_a/rel03/out.bin,1000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,4608,5120\n"
    "run/fault_a/rel04/out.bin,1000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,6144,6656\n"
    "run/fault_b/rel00/out.bin,2000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,7680,8192\n"
    "run/fault_b/rel01/out.bin,2000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,10240,10752\n"
    "run/fault_b/rel02/out.bin,2000,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,12800,13312\n"
    "run/notes.txt,6,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,15360,15872\n"
    "run/dup1.bin,1500,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,16384,16896\n"
    "run/dup2.bin,1500,2023-11-14T22:13:20+00:00,0644,user,proj00001,0,18432,18944\n"
    "run/link_to_notes,0,2023-11-14T22:13:20+00:00,0644,bob,teamB,2,20480,20992\n"
)
EXPECTED_BY_DIR_CSV = (
    "dir,count,bytes\n"
    "run,4,3006\n"
    "run/fault_b/rel00,1,2000\n"
    "run/fault_b/rel01,1,2000\n"
    "run/fault_b/rel02,1,2000\n"
    "run/fault_a/rel00,1,1000\n"
    "run/fault_a/rel01,1,1000\n"
    "run/fault_a/rel02,1,1000\n"
    "run/fault_a/rel03,1,1000\n"
    "run/fault_a/rel04,1,1000\n"
)
EXPECTED_BY_EXTENSION_CSV = (
    "extension,count,bytes\n"
    "bin,10,14000\n"
    "txt,1,6\n"
    "(none),1,0\n"
)


def test_summary_and_csv_output_is_unchanged_by_the_streaming_refactor(tmp_path):
    """Pins summary()/write_csv() to the exact output captured from the pre-fix
    implementation. Discriminates any behavioural drift from switching `members`
    reads from one `fetchall()` to a streamed, column-limited, transaction-scoped
    pass -- including, specifically, `largest`'s tie-break order once it moves from a
    Python stable sort to `ORDER BY size DESC, hdr_offset ASC`.
    """
    store, archive_id, _ = build_and_index(tmp_path, GOLDEN_MEMBERS)

    result = summary(store, archive_id, top=5)
    plain = dict(result)
    plain["largest"] = [(m["dir"], m["name"], m["size"]) for m in result["largest"]]
    assert plain == EXPECTED_SUMMARY

    out = tmp_path / "exports"
    write_csv(store, archive_id, out)
    assert (out / "manifest.csv").read_text() == EXPECTED_MANIFEST_CSV
    assert (out / "by_dir.csv").read_text() == EXPECTED_BY_DIR_CSV
    assert (out / "by_extension.csv").read_text() == EXPECTED_BY_EXTENSION_CSV


def test_summary_and_write_csv_stream_members_instead_of_materialising_them(
    tmp_path, monkeypatch
):
    """Row-count proof, not a functional one -- the fix-round ruling is explicit that
    the honest check here is about *how much* a call reads, not what it returns
    (that's the golden test above).

    `sqlite3.Cursor`/`Connection` are immutable C types (`monkeypatch.setattr(
    sqlite3.Cursor, "fetchall", ...)` raises `TypeError: cannot set 'fetchall'
    attribute of immutable type`), and subclassing them does not help either: the
    `conn.execute(...)` shorthand `ArchiveStore` uses everywhere is implemented at the
    C level and does not route through an overridden `Connection.cursor()` factory
    (confirmed empirically -- a subclassed cursor's `fetchall` is never reached
    through `conn.execute(...).fetchall()`). So this wraps `store.connect()`'s
    *return value* in a plain Python proxy instead: real behaviour is preserved
    (`execute`/iteration/`fetchall` all delegate straight to the real connection and
    cursor), but every `fetchall()` call's row count is recorded on the way past.

    Asserts no single call during `summary()`/`write_csv()` returns anywhere near the
    full member count. The small, legitimate `fetchall()` calls this design still
    makes -- the one-row archive lookup, the `top`-bounded `largest` query -- are
    still allowed; a `fetchall()` call returning all N members would mean the
    per-member rollup went back to materialising the whole table, which is exactly
    the regression this guards against. `store.stream()` itself never calls
    `fetchall` at all (it yields from the cursor), so this also indirectly confirms
    `summary`/`write_csv` are using it rather than `query_members`.
    """
    n = 2000
    members = [(f"run/file{i:05d}.bin", b"x" * 10) for i in range(n)]
    store, archive_id, _ = build_and_index(tmp_path, members, part_size=1 << 20)

    sizes = []
    real_connect = store.connect

    class _CursorSpy:
        def __init__(self, cursor):
            self._cursor = cursor

        def fetchall(self):
            result = self._cursor.fetchall()
            sizes.append(len(result))
            return result

        def __iter__(self):
            return iter(self._cursor)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class _ConnectionSpy:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            return _CursorSpy(self._conn.execute(sql, params))

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "connect", lambda: _ConnectionSpy(real_connect()))

    result = summary(store, archive_id, top=5)
    assert result["n_members"] == n  # the streamed pass still saw every row
    write_csv(store, archive_id, tmp_path / "exports")

    assert sizes, "expected at least the small archive-row/largest fetchall() calls"
    assert max(sizes) < 50, (
        f"a fetchall() call returned {max(sizes)} row(s) out of {n} members -- "
        f"the member read is materialising instead of streaming"
    )


def test_largest_query_uses_the_size_index_not_a_full_scan(tmp_path):
    """Confirms `summary()`'s own `largest` query is answered by the `members_size`
    index `build_indexes` creates, not a full scan -- the specific, measured claim in
    `summary()`'s docstring (~0.0001 s against 300,000 rows including a worst-case
    100,000-way tie, versus ~0.19 s to materialise and sort the same rows in Python).

    Traces the real SQL `summary()` executes (`Connection.set_trace_callback`) rather
    than re-typing a query that merely resembles it: a hand-written duplicate would
    keep passing even if `summary()`'s own query drifted to a different, unindexed
    shape, which is exactly the kind of test the brief warns does not discriminate.
    """
    store, archive_id, _ = indexed(tmp_path)

    statements = []
    store.connect().set_trace_callback(statements.append)
    try:
        summary(store, archive_id, top=5)
    finally:
        store.connect().set_trace_callback(None)

    candidates = [sql for sql in statements if "order by size desc" in sql.lower()]
    assert candidates, f"expected a query ordering by size descending; saw: {statements}"

    # The traced text already has its `?` placeholders expanded to literal values
    # (that is what `set_trace_callback` hands back), so it is run as-is.
    plan = store.query("EXPLAIN QUERY PLAN " + candidates[0])
    plan_text = " ".join(row["detail"] for row in plan).lower()
    assert "members_size" in plan_text


def test_summary_and_write_csv_open_one_explicit_read_transaction(tmp_path):
    """Confirms `summary()`/`write_csv()` genuinely wrap their reads in a `BEGIN`/
    `COMMIT` pair (`ArchiveStore.read_transaction`), traced from the real connection,
    rather than relying on "reads everything in one call" being an accidental
    snapshot -- the fix-round ruling's own reasoning for why this has to be explicit.
    Also checks the member reads land *between* the `BEGIN` and the `COMMIT`, not
    merely that both appear somewhere in the trace.
    """
    store, archive_id, _ = indexed(tmp_path)

    statements = []
    store.connect().set_trace_callback(statements.append)
    try:
        summary(store, archive_id, top=5)
    finally:
        store.connect().set_trace_callback(None)

    upper = [s.strip().upper() for s in statements]
    assert "BEGIN" in upper, f"expected an explicit BEGIN; saw {statements}"
    assert "COMMIT" in upper, f"expected an explicit COMMIT; saw {statements}"
    begin_at = upper.index("BEGIN")
    commit_at = len(upper) - 1 - upper[::-1].index("COMMIT")
    member_reads = [i for i, s in enumerate(upper) if "FROM MEMBERS" in s]
    assert member_reads, f"expected at least one read of members; saw {statements}"
    assert all(begin_at < i < commit_at for i in member_reads), (
        "the member reads must happen strictly between BEGIN and COMMIT, not "
        f"outside the transaction they are meant to be inside: {statements}"
    )

    statements.clear()
    store.connect().set_trace_callback(statements.append)
    try:
        write_csv(store, archive_id, tmp_path / "exports_txn_check")
    finally:
        store.connect().set_trace_callback(None)
    upper = [s.strip().upper() for s in statements]
    assert "BEGIN" in upper and "COMMIT" in upper, (
        f"expected write_csv to also open an explicit read transaction; saw {statements}"
    )


# ---- export shares report's completeness warning ---------------------------------


def test_export_warns_on_a_non_complete_archive_but_not_on_a_complete_one(
    tmp_path, capsys
):
    store, archive_id, db = indexed(tmp_path)
    capsys.readouterr()

    out_complete = tmp_path / "out_complete"
    assert main(["archive", "export", "--db", db, "--archive", "a.tar",
                 "--out", str(out_complete)]) == 0
    assert "may be missing members" not in capsys.readouterr().out.lower()

    store.mark_stale(archive_id, "parts changed underneath the index")
    out_stale = tmp_path / "out_stale"
    assert main(["archive", "export", "--db", db, "--archive", "a.tar",
                 "--out", str(out_stale)]) == 0
    out = capsys.readouterr().out
    assert "may be missing members" in out
    assert "no longer there" in out.lower()


def test_export_warns_on_a_partially_walked_archive_before_the_wrote_lines(
    tmp_path, capsys
):
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
    assert row["state"] != "complete"          # sanity: the walk really did stop early
    capsys.readouterr()

    out_dir = tmp_path / "out"
    assert main(["archive", "export", "--db", db, "--archive", "a.tar",
                 "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "may be missing members" in out
    assert "wrote " in out
    assert out.index("may be missing members") < out.index("wrote ")
