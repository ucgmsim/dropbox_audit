import calendar
import hashlib

from dbaudit import report, tree
from dbaudit.store import Store

LISTED = calendar.timegm((2026, 10, 1, 0, 0, 0, 0, 0, 0))


def file(path, size, content=None, modified="2026-01-01T00:00:00Z", uploaded=None):
    """A file as Dropbox lists it. Files with the same `content` are copies."""
    return {
        ".tag": "file", "id": f"id:{path}", "name": path.rsplit("/", 1)[1],
        "path_display": path, "size": size, "rev": "r",
        "content_hash": hashlib.sha256((content or path).encode()).hexdigest(),
        "client_modified": modified, "server_modified": uploaded or modified,
    }


def folder(path):
    return {".tag": "folder", "id": f"id:{path}", "name": path.rsplit("/", 1)[1],
            "path_display": path}


def listing(tmp_path, *entries):
    store = Store(tmp_path / "audit.db")
    store.init_schema()
    store.set_meta("root", "/R")
    shard = store.add_shard("/R", 1)
    store.claim_shard("test")
    store.commit_page(shard, [folder("/R"), *entries], cursor=None, has_more=False)
    return store


def at(store, path):
    return tree.find_dir(store, path)[0]


def measured(store, page="/R", min_size=1):
    return tree.measure(store, LISTED, min_size, at(store, page))


# ---- folder totals ----------------------------------------------------------


def test_folder_totals_cover_the_whole_subtree(tmp_path):
    store = listing(tmp_path, file("/R/a/x", 5), file("/R/a/b/y", 7), file("/R/c/z", 11),
                    file("/R/top", 2), folder("/R/empty"))
    f = measured(store)
    r, a = at(store, "/R"), at(store, "/R/a")
    assert (f.size[r], f.files[r], f.folders[r], f.kids[r], f.own_files[r]) == (25, 4, 4, 3, 1)
    assert (f.size[a], f.files[a], f.folders[a], f.kids[a], f.own_files[a]) == (12, 2, 1, 1, 1)
    assert f.size[at(store, "/R/a/b")] == 7
    assert f.size[at(store, "/R/empty")] == 0


def test_dates_are_the_files_own_not_dropboxs(tmp_path):
    store = listing(tmp_path,
                    file("/R/old/f", 10, modified="2020-01-01T00:00:00Z",
                         uploaded="2026-09-01T00:00:00Z"),
                    file("/R/new/g", 20, modified="2026-01-01T00:00:00Z"))
    f = measured(store)
    r, old = at(store, "/R"), at(store, "/R/old")
    assert f.cold[r] == 10 and f.cold[old] == 10 and f.cold[at(store, "/R/new")] == 0
    assert f.newest[old] == calendar.timegm((2020, 1, 1, 0, 0, 0, 0, 0, 0))
    assert f.newest[r] == calendar.timegm((2026, 1, 1, 0, 0, 0, 0, 0, 0))


def test_a_date_after_the_listing_is_never_a_folders_newest(tmp_path):
    store = listing(tmp_path, file("/R/a/x", 1, modified="2035-07-30T00:00:00Z"),
                    file("/R/a/y", 1, modified="2026-01-01T00:00:00Z"))
    newest = measured(store).newest[at(store, "/R/a")]
    assert newest == calendar.timegm((2026, 1, 1, 0, 0, 0, 0, 0, 0))


def test_the_page_is_its_root_and_everything_below_it(tmp_path):
    store = listing(tmp_path, file("/R/a/b/y", 7), file("/R/c/z", 11))
    f = measured(store, page="/R/a")
    assert f.on_page[at(store, "/R/a")] and f.on_page[at(store, "/R/a/b")]
    assert not f.on_page[at(store, "/R")] and not f.on_page[at(store, "/R/c")]


def test_folders_are_found_without_regard_to_case(tmp_path):
    store = listing(tmp_path, file("/R/Mixed/x", 1))
    assert tree.find_dir(store, "/r/mixed/") == (at(store, "/R/Mixed"), "/R/Mixed")
    assert tree.find_dir(store, "/R/missing") is None


def test_the_listing_time_is_when_a_pass_finished_or_else_began(tmp_path):
    store = listing(tmp_path)
    store.set_meta("created_at", 1000.0)
    assert tree.crawl_time(store) == (1000.0, False)
    store.set_meta("last_pass_completed_at", 2000.0)
    assert tree.crawl_time(store) == (2000.0, True)


# ---- duplicates -------------------------------------------------------------


def test_files_in_folders_newer_than_the_snapshot_are_left_out(tmp_path):
    """A page can be made while a crawl is still writing: rows that arrive mid-read
    reference folders the snapshot never saw."""
    store = listing(tmp_path, file("/R/a/x", 5, "X"))
    store.connect().execute(
        "INSERT INTO files(dbx_id, dir_id, name, size, content_hash) "
        "VALUES('id:late', 9999, 'late', 3, ?)", (hashlib.sha256(b"X").digest(),))
    f = measured(store)
    assert f.size[at(store, "/R")] == 5 and f.held[at(store, "/R/a")] == 0


def test_a_copy_elsewhere_counts_in_both_folders_and_not_their_parent(tmp_path):
    store = listing(tmp_path, file("/R/P/A/x/f", 100, "X"), file("/R/P/B/x/g", 100, "X"))
    f = measured(store)
    for path in ("/R/P/A", "/R/P/A/x", "/R/P/B", "/R/P/B/x"):
        assert f.held[at(store, path)] == 100, path
    assert f.held[at(store, "/R/P")] == 0 and f.held[at(store, "/R")] == 0


def test_two_copies_in_one_folder_count_nowhere(tmp_path):
    store = listing(tmp_path, file("/R/P/A/f", 100, "X"), file("/R/P/A/g", 100, "X"))
    f = measured(store)
    assert f.held[at(store, "/R/P/A")] == 0 and f.held[at(store, "/R/P")] == 0


def test_a_copy_in_a_subfolder_counts_only_there(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 100, "X"), file("/R/A/sub/g", 100, "X"))
    f = measured(store)
    assert f.held[at(store, "/R/A/sub")] == 100 and f.held[at(store, "/R/A")] == 0


def test_empty_files_are_never_duplicates(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 0, "E"), file("/R/B/g", 0, "E"))
    f = measured(store)
    assert f.held[at(store, "/R/A")] == 0 and f.copies == {} and f.reclaimable == 0


def test_copies_are_counted_for_files_that_can_get_a_row(tmp_path):
    store = listing(tmp_path, file("/R/a", 100, "X"), file("/R/b", 100, "X"),
                    file("/R/c", 100, "X"), file("/R/d", 5, "Y"), file("/R/e", 5, "Y"))
    f = measured(store, min_size=50)
    assert sorted(f.copies.values()) == [3, 3, 3]


def test_reclaimable_on_the_whole_listing_matches_the_report(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 100, "X"), file("/R/B/g", 100, "X"),
                    file("/R/B/h", 100, "X"), file("/R/C/i", 7, "Y"), file("/R/C/j", 7, "Y"),
                    file("/R/D/k", 50))
    assert measured(store).reclaimable == report.total_reclaimable_duplicates(store) == 207


def test_on_a_subtree_page_a_copy_outside_frees_every_copy_inside(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 100, "X"), file("/R/B/g", 100, "X"),
                    file("/R/A/h", 7, "Y"), file("/R/A/i", 7, "Y"))
    assert measured(store, page="/R/A").reclaimable == 100 + 7


# ---- layout -----------------------------------------------------------------


def sample(tmp_path):
    return listing(tmp_path, file("/R/big/f1", 50), file("/R/big/f2", 30),
                   file("/R/big/small/s", 4), file("/R/mid/m", 20), file("/R/t1", 1),
                   file("/R/t2", 2))


def shape(row):
    """(kind, name or entries, size, children) for every row, to compare layouts whole."""
    label = row.entries if row.kind == "fold" else row.name
    return (row.kind, label, row.size, [shape(c) for c in row.children])


def test_small_entries_fold_into_one_row_per_folder(tmp_path):
    store = sample(tmp_path)
    page = tree.layout(store, measured(store), 10, 1000)
    assert shape(page.root) == (
        "dir", "R", 107, [
            ("dir", "big", 84, [("file", "f1", 50, []), ("file", "f2", 30, []),
                                ("fold", 1, 4, [])]),
            ("dir", "mid", 20, [("file", "m", 20, [])]),
            ("fold", 2, 3, []),
        ])
    assert page.rows == 8 and page.threshold == 10


def test_fold_rows_and_shown_rows_add_up_to_each_folder(tmp_path):
    store = sample(tmp_path)
    page = tree.layout(store, measured(store), 10, 1000)

    def check(row):
        assert sum(c.size for c in row.children) == row.size
        assert sum(c.files for c in row.children) == row.files
        assert sum(c.folders + (c.kind == "dir") for c in row.children) == row.folders
        for c in row.children:
            if c.kind == "dir":
                check(c)

    check(page.root)


def test_a_larger_threshold_folds_more(tmp_path):
    store = sample(tmp_path)
    page = tree.layout(store, measured(store), 25, 1000)
    assert [shape(c)[:3] for c in page.root.children] == [("dir", "big", 84), ("fold", 3, 23)]


def test_a_subtree_page_starts_at_that_folder(tmp_path):
    store = sample(tmp_path)
    page = tree.layout(store, measured(store, page="/R/big"), 10, 1000)
    assert (page.root.name, page.root.path, page.root.size) == ("big", "big", 84)
    assert [c.path for c in page.root.children] == ["big/f1", "big/f2", "big"]


def test_too_many_rows_doubles_the_threshold(tmp_path):
    store = sample(tmp_path)
    page = tree.layout(store, measured(store), 10, 5)
    assert (page.threshold, page.rows) == (40, 5)
    assert [shape(c)[:3] for c in page.root.children] == [("dir", "big", 84), ("fold", 3, 23)]


def test_file_rows_carry_their_copy_count(tmp_path):
    store = listing(tmp_path, file("/R/a/f", 50, "X"), file("/R/b/g", 50, "X"))
    page = tree.layout(store, measured(store, min_size=10), 10, 1000)
    rows = [c for folder in page.root.children for c in folder.children]
    assert [(r.name, r.copies) for r in rows] == [("f", 2), ("g", 2)]


# ---- tables -----------------------------------------------------------------


def chosen(store, threshold=1, page="/R"):
    f = measured(store, page=page)
    return tree.tables(store, f, tree.layout(store, f, threshold, 1000))


def test_listed_folders_never_overlap(tmp_path):
    store = listing(tmp_path,
                    # D holds 15 bytes held elsewhere: X outside A, Y elsewhere in A. A holds
                    # only X's 10, yet qualifies: so D is taken first and A, above it, skipped.
                    file("/R/A/D/f1", 10, "X"), file("/R/Z/z1", 10, "X"),
                    file("/R/A/D/f2", 5, "Y"), file("/R/A/E/e1", 5, "Y"),
                    file("/R/C/old", 9, modified="2001-01-01T00:00:00Z"),
                    file("/R/C/sub/old", 9, modified="2001-01-01T00:00:00Z"))
    lists = chosen(store)
    assert [e.path for e in lists["dup"]] == ["A/D", "Z", "A/E"]
    assert [e.path for e in lists["cold"]] == ["C"]


def test_the_tiny_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(tree, "TINY_FILES", 3)
    monkeypatch.setattr(tree, "TINY_MEAN", 10)
    store = listing(tmp_path, *(file(f"/R/T/{i}", 10) for i in range(3)),
                    file("/R/U/0", 10), file("/R/U/1", 10), file("/R/U/2", 11),
                    file("/R/V/0", 1), file("/R/V/1", 1))
    assert [e.path for e in chosen(store)["tiny"]] == ["T"]
    rows = {c.name: c.tiny for c in tree.layout(store, measured(store), 1, 1000).root.children}
    assert rows == {"T": True, "U": False, "V": False}


def test_the_counterpart_is_where_the_other_copies_mostly_are(tmp_path):
    store = listing(tmp_path,
                    file("/R/P/A/f", 30, "X"), file("/R/P/B/x/f", 30, "X"),
                    file("/R/P/A/g", 10, "Y"), file("/R/Q/g", 10, "Y"))
    (entry,) = [e for e in chosen(store)["dup"] if e.path == "P/A"]
    assert (entry.place, entry.place_share) == ("P/B", 0.75)


def test_the_counterpart_is_never_shallower_than_two_below_the_root(tmp_path):
    store = listing(tmp_path, file("/R/S/sub/f", 30, "X"), file("/R/T/deep/x/f", 30, "X"))
    (entry,) = [e for e in chosen(store)["dup"] if e.path == "S"]
    assert (entry.place, entry.place_share) == ("T/deep", 1.0)


def test_a_folded_folder_links_to_its_nearest_row(tmp_path, monkeypatch):
    monkeypatch.setattr(tree, "TINY_FILES", 3)
    monkeypatch.setattr(tree, "TINY_MEAN", 10)  # big averages 25 bytes, so only big/t is tiny
    store = listing(tmp_path, file("/R/big/f", 100), *(file(f"/R/big/t/{i}", 1) for i in range(3)))
    (entry,) = chosen(store, threshold=50)["tiny"]
    assert (entry.path, entry.dir_id, entry.target) == ("big/t", at(store, "/R/big/t"),
                                                        at(store, "/R/big"))
