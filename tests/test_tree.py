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
