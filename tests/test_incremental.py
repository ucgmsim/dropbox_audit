import pytest

from dbaudit.api import Page
from dbaudit.crawler import Crawler
from dbaudit.limiter import AdaptiveLimiter
from dbaudit.store import Store
from tests.fakes import ALL_FILES, TREE, FakeLister


def limiter():
    return AdaptiveLimiter(rps=1e6, max_concurrency=4, sleep=lambda d: None)


def full_pass(tmp_path, tree=TREE, **kw):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    crawler = Crawler(store, FakeLister(tree), limiter(), workers=1, **kw)
    crawler.seed("/R")
    crawler.seed_delta_cursor("/R")
    crawler.run()
    return store


def files_in(store):
    return {
        row[0]
        for row in store.query(
            "SELECT dirs.path_display || '/' || files.name FROM files "
            "JOIN dirs ON dirs.id = files.dir_id"
        )
    }


class DeltaLister(FakeLister):
    """list_folder/continue after a completed pass: only what changed.

    The delta is delivered once, by whichever shard asks first -- a real change is
    reported by exactly the one shard whose subtree contains it.
    """

    def __init__(self, tree, delta):
        super().__init__(tree)
        self.pending_delta = list(delta)

    def continue_(self, cursor):
        self.calls.append(("cont", cursor, None))
        entries, self.pending_delta = self.pending_delta, []
        return Page(entries=entries, cursor=cursor, has_more=False)


NEW_FILE = {
    ".tag": "file", "id": "id:/R/b/new.txt", "name": "new.txt", "size": 4,
    "path_display": "/R/b/new.txt", "path_lower": "/r/b/new.txt",
    "content_hash": "ef" * 32, "rev": "r2",
    "client_modified": "2026-01-01T00:00:00Z", "server_modified": "2026-01-01T00:00:00Z",
}
DELETED_FILE = {".tag": "deleted", "name": "b1.txt",
                "path_display": "/R/b/b1.txt", "path_lower": "/r/b/b1.txt"}


def test_incremental_applies_adds_and_deletes(tmp_path):
    store = full_pass(tmp_path)
    assert files_in(store) == ALL_FILES

    crawler = Crawler(store, DeltaLister(TREE, [DELETED_FILE, NEW_FILE]), limiter(), workers=1)
    crawler.run(incremental=True)

    names = files_in(store)
    assert "/R/b/new.txt" in names
    assert "/R/b/b1.txt" not in names
    assert store.query("SELECT COUNT(*) FROM tombstones")[0][0] == 1


def test_incremental_uses_one_cursor_not_one_per_shard(tmp_path):
    """The whole point: cost scales with changes, not with the number of shards."""
    store = full_pass(tmp_path)
    shards = store.query("SELECT COUNT(*) FROM shards")[0][0]
    assert shards > 2, "need several shards for this to mean anything"
    lister = DeltaLister(TREE, [])
    Crawler(store, lister, limiter(), workers=1).run(incremental=True)
    assert sum(1 for c in lister.calls if c[0] == "cont") == 1


def test_incremental_without_a_delta_cursor_is_refused(tmp_path):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    crawler = Crawler(store, FakeLister(TREE), limiter(), workers=1)
    crawler.seed("/R")
    crawler.run()
    with pytest.raises(RuntimeError, match="delta cursor"):
        Crawler(store, DeltaLister(TREE, []), limiter(), workers=1).run(incremental=True)


def test_incremental_requires_a_completed_full_pass(tmp_path):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    store.add_shard("/R", 0, "recursive")
    crawler = Crawler(store, FakeLister(TREE), limiter(), workers=1)
    with pytest.raises(RuntimeError, match="full pass"):
        crawler.run(incremental=True)


def test_incremental_bumps_the_pass_number(tmp_path):
    store = full_pass(tmp_path)
    assert store.get_meta("seen_run", "1") in ("1", None)
    Crawler(store, DeltaLister(TREE, []), limiter(), workers=1).run(incremental=True)
    assert int(store.get_meta("seen_run")) == 2


def test_incremental_with_no_changes_leaves_data_intact(tmp_path):
    store = full_pass(tmp_path)
    before = files_in(store)
    Crawler(store, DeltaLister(TREE, []), limiter(), workers=1).run(incremental=True)
    assert files_in(store) == before


def test_deleting_a_directory_removes_everything_beneath_it(tmp_path):
    store = full_pass(tmp_path)
    deleted_dir = {".tag": "deleted", "name": "deep",
                   "path_display": "/R/a/deep", "path_lower": "/r/a/deep"}
    Crawler(store, DeltaLister(TREE, [deleted_dir]), limiter(), workers=1).run(incremental=True)
    names = files_in(store)
    assert "/R/a/deep/d1.txt" not in names
    assert "/R/a/a1.txt" in names
    assert store.query("SELECT COUNT(*) FROM dirs WHERE path_lower='/r/a/deep'")[0][0] == 0


def test_incremental_can_run_twice(tmp_path):
    store = full_pass(tmp_path)
    Crawler(store, DeltaLister(TREE, []), limiter(), workers=1).run(incremental=True)
    Crawler(store, DeltaLister(TREE, [NEW_FILE]), limiter(), workers=1).run(incremental=True)
    assert "/R/b/new.txt" in files_in(store)
    assert int(store.get_meta("seen_run")) == 3
