import calendar
import hashlib
import io
from html.parser import HTMLParser

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


# ---- the page ---------------------------------------------------------------


class Outline(HTMLParser):
    """What a page is made of: nesting, ids, in-page links, section headings."""

    def __init__(self, text):
        super().__init__()
        self.open = self.deepest = 0
        self.ids, self.links, self.headings = set(), [], []
        self._heading = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details":
            self.open += 1
            self.deepest = max(self.deepest, self.open)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.links.append(attrs["href"][1:])
        if tag == "h2":
            self._heading = ""

    def handle_endtag(self, tag):
        if tag == "details":
            self.open -= 1
            assert self.open >= 0, "a </details> without its <details>"
        if tag == "h2":
            self.headings.append(self._heading)
            self._heading = None

    def handle_data(self, data):
        if self._heading is not None:
            self._heading += data


def page_text(store, page="/R", min_size=1, max_rows=1000):
    out = io.StringIO()
    tree.render(out, *tree.build(store, page, min_size, max_rows, "audit.db"))
    return out.getvalue()


def test_sizes_are_shown_compactly():
    assert [tree.compact(n) for n in (0, 999, 1024, 85.9 * 1024, 341 * 2**30, 111.4 * 2**40)] \
        == ["0B", "999B", "1.0K", "85.9K", "341G", "111T"]


def test_bars_fill_in_eighths():
    assert tree.bar(0) == "" and tree.bar(1) == "█" * tree.BAR_WIDTH
    assert tree.bar(1 / tree.BAR_WIDTH + 3 / 8 / tree.BAR_WIDTH) == "█▍"


def test_names_are_escaped(tmp_path):
    store = listing(tmp_path, file('/R/a<b&"c/a<b&"c', 5))
    text = page_text(store)
    assert 'a<b&"c' not in text
    assert '>a&lt;b&amp;&quot;c</span>' in text and 'data-path="a&lt;b&amp;&quot;c/' in text


def test_the_page_holds_together(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 100, "X"), file("/R/B/g", 100, "X"),
                    file("/R/C/old", 50, modified="2001-01-01T00:00:00Z"))
    outline = Outline(page_text(store))
    assert outline.open == 0 and outline.deepest == 2
    assert outline.links and all(link in outline.ids for link in outline.links)
    assert outline.headings == [
        "Biggest duplicated folders", "Coldest big folders", "Folders made of tiny files",
        "Where the bytes are, by file type", "Bytes by year last modified", "Reading this"]


def test_an_incomplete_listing_says_so(tmp_path):
    store = listing(tmp_path, file("/R/A/f", 5))
    assert "The listing is incomplete" not in page_text(store)
    store.add_shard("/R/later", 2)
    assert "The listing is incomplete: 1 pending" in page_text(store)


def test_a_raised_threshold_is_stated(tmp_path):
    store = sample(tmp_path)
    assert "entries under 40B are folded" in page_text(store, min_size=10, max_rows=5)
    assert "are folded.</div>" not in page_text(store, min_size=10)


def test_names_beyond_ascii_come_through_intact(tmp_path):
    store = listing(tmp_path, file("/R/Ngā kōrero/日本 é.txt", 5))
    text = page_text(store)
    assert ">Ngā kōrero</span>" in text and 'data-path="Ngā kōrero/日本 é.txt"' in text


def test_a_page_of_empty_files_divides_by_nothing(tmp_path):
    store = listing(tmp_path, file("/R/a/x", 0), file("/R/b/y", 0))
    text = page_text(store, min_size=0)
    assert Outline(text).open == 0 and '<span class="pct">0.0%</span>' in text


def test_an_empty_listing_is_a_page_of_zeros(tmp_path):
    store = Store(tmp_path / "audit.db")
    store.init_schema()
    store.set_meta("root", "/R")
    outline = Outline(page_text(store))
    assert outline.open == 0 and "d0" in outline.ids


def test_a_crawl_writing_while_the_page_is_built_cannot_reach_it(tmp_path, monkeypatch):
    """Every query reads one snapshot. Read separately, a big file committed after
    measuring lands in a folder measured as small, which has no row to hold it."""
    store = listing(tmp_path, file("/R/a/x", 5), file("/R/b/y", 50))
    crawler = Store(tmp_path / "audit.db")
    measure = tree.measure

    def measure_then_crawl(*args):
        figures = measure(*args)
        crawler.commit_page(1, [file("/R/a/late", 1000)], cursor=None, has_more=False)
        return figures

    monkeypatch.setattr(tree, "measure", measure_then_crawl)
    page, _, _ = tree.build(store, "/R", 10, 1000, "audit.db")
    assert (page.root.size, page.rows) == (55, 4)


# ---- the command ------------------------------------------------------------


def test_the_command_writes_the_page_and_says_so(tmp_path, capsys):
    from dbaudit.cli import main

    sample(tmp_path)
    out = tmp_path / "tree.html"
    assert main(["tree", "--db", str(tmp_path / "audit.db"), "--out", str(out),
                 "--min-size", "10"]) == 0
    assert "8 rows" in capsys.readouterr().out
    assert Outline(out.read_text(encoding="utf-8")).open == 0
    assert not list(tmp_path.glob(".tree-*"))


def test_the_command_refuses_a_missing_database_or_folder(tmp_path, capsys):
    from dbaudit.cli import main

    out = tmp_path / "tree.html"
    assert main(["tree", "--db", str(tmp_path / "none.db"), "--out", str(out)]) == 2
    assert "not initialised" in capsys.readouterr().err
    sample(tmp_path)
    assert main(["tree", "--db", str(tmp_path / "audit.db"), "--out", str(out),
                 "--subtree", "/R/nowhere"]) == 2
    assert "/R/nowhere is not in" in capsys.readouterr().err
    assert not out.exists()


def test_sizes_are_read_in_binary_units():
    from dbaudit.cli import parse_size

    assert [parse_size(s) for s in ("1GiB", "100MiB", "512kib", "2048", "1.5KiB", "7B")] \
        == [1 << 30, 100 << 20, 512 << 10, 2048, 1536, 7]


def test_a_size_that_cannot_be_read_is_refused(capsys):
    import pytest

    from dbaudit.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["tree", "--db", "a.db", "--out", "t.html", "--min-size", "1GB"])
    assert stopped.value.code == 2
    assert "is not a size" in capsys.readouterr().err


def test_the_command_will_not_overwrite_the_database_or_write_nowhere(tmp_path, capsys):
    from dbaudit.cli import main

    sample(tmp_path)
    db = str(tmp_path / "audit.db")
    assert main(["tree", "--db", db, "--out", db]) == 2
    assert "names the database itself" in capsys.readouterr().err
    assert main(["tree", "--db", db, "--out", str(tmp_path / "no" / "t.html")]) == 2
    assert "is not a folder" in capsys.readouterr().err
    assert tree.find_dir(Store(db), "/R/big") is not None  # the database is untouched
