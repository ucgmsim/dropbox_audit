from dbaudit.report import (
    bytes_older_than,
    cold_bytes,
    duplicate_report,
    export_csv,
    extension_profile,
    reconcile,
    small_file_hotspots,
    top_dirs,
    top_level_summary,
    total_reclaimable_duplicates,
)
from dbaudit.store import Store


def mk(name, size, digest, ts="2020-01-01T00:00:00Z", d="/R"):
    import hashlib

    content_hash = hashlib.sha256(digest.encode()).hexdigest()
    return {
        ".tag": "file", "id": f"id:{d}/{name}", "name": name, "size": size,
        "path_display": f"{d}/{name}", "path_lower": f"{d}/{name}".lower(),
        "content_hash": content_hash[:64], "rev": "r",
        "client_modified": ts, "server_modified": ts,
    }


def seed(tmp_path, entries):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    sid = store.add_shard("/R", 0, "recursive")
    store.claim_shard("w")
    store.commit_page(sid, entries, cursor=None, has_more=False)
    store.build_indexes()
    return store


def test_duplicates_report_reclaimable_bytes(tmp_path):
    store = seed(tmp_path, [mk("a", 100, "a"), mk("b", 100, "a"), mk("c", 100, "a"),
                            mk("d", 7, "b")])
    rows = duplicate_report(store, limit=10)
    assert len(rows) == 1
    assert rows[0].copies == 3 and rows[0].reclaimable == 200
    assert rows[0].example.startswith("/R/")
    assert total_reclaimable_duplicates(store) == 200


def test_files_with_no_hash_are_not_treated_as_duplicates(tmp_path):
    no_hash = dict(mk("x", 5, "a"))
    no_hash.pop("content_hash")
    store = seed(tmp_path, [no_hash, dict(mk("y", 5, "a"), content_hash=None)])
    assert duplicate_report(store, limit=10) == []


def test_cold_bytes_buckets_by_year(tmp_path):
    store = seed(tmp_path, [mk("old", 10, "a", "2015-06-01T00:00:00Z"),
                            mk("new", 20, "b", "2026-01-01T00:00:00Z")])
    by_year = {r.year: r.bytes for r in cold_bytes(store)}
    assert by_year[2015] == 10 and by_year[2026] == 20


def test_bytes_older_than(tmp_path):
    import calendar

    store = seed(tmp_path, [mk("old", 10, "a", "2015-06-01T00:00:00Z"),
                            mk("new", 20, "b", "2026-01-01T00:00:00Z")])
    now = calendar.timegm((2026, 6, 1, 0, 0, 0, 0, 0, 0))
    count, total = bytes_older_than(store, years=5, now=now)
    assert (count, total) == (1, 10)


def test_top_dirs_rolls_up_recursively(tmp_path):
    store = seed(tmp_path, [mk("x", 5, "a", d="/R/sub"), mk("y", 7, "b", d="/R/sub/deeper")])
    top = {r.path: r.bytes for r in top_dirs(store, limit=10)}
    assert top["/R"] == 12
    assert top["/R/sub"] == 12
    assert top["/R/sub/deeper"] == 7


def test_top_dirs_counts_files_recursively(tmp_path):
    store = seed(tmp_path, [mk("x", 5, "a", d="/R/sub"), mk("y", 7, "b", d="/R/sub/deeper")])
    counts = {r.path: r.files for r in top_dirs(store, limit=10)}
    assert counts["/R"] == 2 and counts["/R/sub/deeper"] == 1


def test_top_dirs_on_empty_database(tmp_path):
    store = seed(tmp_path, [])
    assert top_dirs(store, limit=10) == []


def test_reconcile_reports_residual(tmp_path):
    store = seed(tmp_path, [mk("a", 100, "a")])
    assert reconcile(store, reported_used_bytes=250) == (100, 250, 150)


def test_small_file_hotspots(tmp_path):
    entries = [mk(f"f{i}", 1, f"{i:064d}", d="/R/junk") for i in range(10)]
    entries.append(mk("big", 10_000_000, "z", d="/R/ok"))
    store = seed(tmp_path, entries)
    hits = [r.path for r in small_file_hotspots(store, min_files=5, max_mean_size=100)]
    assert hits == ["/R/junk"]


def test_extension_profile_ranks_by_bytes(tmp_path):
    store = seed(tmp_path, [mk("a.tar", 1000, "a"), mk("b.txt", 5, "b"), mk("c.tar", 500, "c"),
                            mk("noext", 1, "d")])
    rows = extension_profile(store)
    assert rows[0].ext == "tar" and rows[0].files == 2 and rows[0].bytes == 1500
    assert any(r.ext == "(none)" for r in rows)


def test_extension_ignores_dotted_names_that_are_not_extensions(tmp_path):
    store = seed(tmp_path, [mk("run.2024-01-01-backup-archive", 10, "a")])
    assert extension_profile(store)[0].ext == "(none)"


def test_top_level_summary_is_per_immediate_child(tmp_path):
    store = seed(tmp_path, [mk("x", 5, "a", d="/R/alice"), mk("y", 7, "b", d="/R/bob/deep")])
    rows = {r.path: r.bytes for r in top_level_summary(store)}
    assert rows == {"/R/alice": 5, "/R/bob": 7}


def test_export_csv_writes_every_report(tmp_path):
    store = seed(tmp_path, [mk("a", 100, "a"), mk("b", 100, "a")])
    written = export_csv(store, str(tmp_path / "out"))
    assert len(written) == 5
    names = {p.rsplit("/", 1)[-1] for p in written}
    assert "duplicates.csv" in names and "top_dirs.csv" in names
    body = open(tmp_path / "out" / "duplicates.csv").read()
    assert "reclaimable" in body and "100" in body


def test_top_dirs_survives_rows_appearing_mid_report(tmp_path):
    """Reports are run against live crawls, so the snapshot must be self-consistent.

    Reading MAX(id) separately from the rows raced with the crawler and crashed with
    IndexError on a real 12.6M-file database.
    """
    store = seed(tmp_path, [mk("x", 5, "a", d="/R/sub")])
    conn = store.connect()
    # A file whose directory was created after our snapshot of `dirs`.
    conn.execute(
        "INSERT INTO files(dbx_id, dir_id, name, size, seen_run) VALUES('id:new', 9999, 'n', 3, 1)"
    )
    rows = {r.path: r.bytes for r in top_dirs(store, limit=10)}
    assert rows["/R/sub"] == 5  # known directories still roll up correctly


def test_top_dirs_ignores_parent_ids_beyond_the_snapshot(tmp_path):
    store = seed(tmp_path, [mk("x", 5, "a", d="/R/sub")])
    conn = store.connect()
    conn.execute(
        "INSERT INTO dirs(id, parent_id, name, path_display, path_lower, depth, seen_run) "
        "VALUES(500, 9999, 'orphan', '/R/orphan', '/r/orphan', 2, 1)"
    )
    assert top_dirs(store, limit=10)  # must not raise
