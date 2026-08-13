import threading

from dbaudit.api import CursorReset, RateLimited
from dbaudit.crawler import Crawler
from dbaudit.limiter import AdaptiveLimiter
from dbaudit.store import Store
from tests.fakes import ALL_FILES, TREE, FakeLister


def limiter():
    return AdaptiveLimiter(rps=1e6, max_concurrency=4, sleep=lambda d: None)


def build(tmp_path, lister, workers=2, **kw):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    return store, Crawler(store, lister, limiter(), workers=workers, **kw)


def files_in(store):
    return {
        row[0]
        for row in store.query(
            "SELECT dirs.path_display || '/' || files.name FROM files "
            "JOIN dirs ON dirs.id = files.dir_id"
        )
    }


def test_full_crawl_finds_every_file(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE))
    crawler.seed("/R")
    crawler.run()
    assert files_in(store) == ALL_FILES
    assert store.pending_count() == 0 and store.running_count() == 0


def test_crawl_without_splitting_finds_every_file(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), split_depth=0)
    crawler.seed("/R")
    crawler.run()
    assert files_in(store) == ALL_FILES
    assert store.query("SELECT COUNT(*) FROM shards")[0][0] == 1


def test_resume_after_interruption_is_exact(tmp_path):
    """Stop mid-crawl, restart on the same DB: every file present exactly once."""
    store, crawler = build(tmp_path, FakeLister(TREE, page_size=1), workers=1)
    crawler.seed("/R")
    stop = threading.Event()
    crawler.run(stop_event=stop, _after_page=lambda n: stop.set() if n >= 5 else None)
    partial = store.stats()["files"]
    assert 0 < partial < 5, f"expected a partial crawl, got {partial}"

    resumed = Store(tmp_path / "a.db")
    Crawler(resumed, FakeLister(TREE, page_size=1), limiter(), workers=1).run()
    assert files_in(resumed) == ALL_FILES
    duplicates = resumed.query(
        "SELECT COUNT(*) FROM (SELECT dbx_id FROM files GROUP BY dbx_id HAVING COUNT(*) > 1)"
    )[0][0]
    assert duplicates == 0


def test_resume_preserves_the_cursor_rather_than_restarting(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE, page_size=1), workers=1, split_depth=0)
    crawler.seed("/R")
    stop = threading.Event()
    crawler.run(stop_event=stop, _after_page=lambda n: stop.set() if n >= 3 else None)
    shard = store.get_shard(1)
    assert shard.state == "pending" and shard.cursor is not None and shard.pages == 3


def test_starving_queue_splits_shard(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=4, split_depth=1)
    crawler.seed("/R")
    crawler.run()
    modes = dict(store.query("SELECT path, mode FROM shards"))
    assert modes["/R"] == "split"
    assert files_in(store) == ALL_FILES


def test_split_mode_is_persisted_so_resume_stays_consistent(tmp_path):
    """A cursor from a non-recursive listing must never be resumed as a recursive one."""
    store, crawler = build(tmp_path, FakeLister(TREE, page_size=1), workers=1, split_depth=1)
    crawler.seed("/R")
    stop = threading.Event()
    crawler.run(stop_event=stop, _after_page=lambda n: stop.set() if n >= 1 else None)
    assert store.get_shard(1).mode == "split"


def test_healthy_queue_does_not_keep_splitting(tmp_path):
    tree = {"/R": [f"d{i}/" for i in range(20)]}
    tree.update({f"/R/d{i}": ["f.txt"] for i in range(20)})
    store, crawler = build(tmp_path, FakeLister(tree), workers=1, split_depth=1)
    crawler.seed("/R")
    crawler.run()
    assert store.query("SELECT COUNT(*) FROM shards WHERE mode='split'")[0][0] <= 1
    assert store.stats()["files"] == 20


def test_max_shards_caps_splitting(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=4, split_depth=3, max_shards=1)
    crawler.seed("/R")
    crawler.run()
    assert store.query("SELECT COUNT(*) FROM shards")[0][0] == 1
    assert files_in(store) == ALL_FILES


def test_cursor_reset_reseeds_shard(tmp_path):
    lister = FakeLister(TREE, page_size=1, continue_faults={1: CursorReset("reset")})
    store, crawler = build(tmp_path, lister, workers=1)
    crawler.seed("/R")
    crawler.run()
    assert files_in(store) == ALL_FILES
    assert store.query("SELECT COUNT(*) FROM api_events WHERE kind='cursor_reset'")[0][0] == 1


def test_rate_limit_is_retried_not_fatal(tmp_path):
    lister = FakeLister(TREE, page_size=1)
    lister.faults["/R"] = RateLimited("429", retry_after=0.05)
    store, crawler = build(tmp_path, lister, workers=1)
    crawler.seed("/R")
    crawler.run()
    assert files_in(store) == ALL_FILES
    assert store.query("SELECT COUNT(*) FROM api_events WHERE kind='rate_limited'")[0][0] == 1


def test_vanished_path_is_not_an_error(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=1, split_depth=0)
    crawler.seed("/R")
    store.add_shard("/R/does-not-exist", 2, "recursive")
    summary = crawler.run()
    assert summary.errors == 0
    assert store.query(
        "SELECT state FROM shards WHERE path='/R/does-not-exist'"
    )[0][0] == "done"


def test_transient_errors_eventually_fail_the_shard_not_the_run(tmp_path):
    from dbaudit.api import TransientError

    class AlwaysBroken(FakeLister):
        def list_folder(self, path, recursive):
            raise TransientError("nope")

    store, crawler = build(tmp_path, AlwaysBroken(TREE), workers=1, split_depth=0,
                           max_page_attempts=2, backoff_base=0.01)
    crawler.seed("/R")
    summary = crawler.run()
    assert summary.errors == 1
    assert store.query("SELECT state FROM shards")[0][0] == "error"


def test_summary_reports_totals(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE))
    crawler.seed("/R")
    summary = crawler.run()
    assert summary.files == 5
    assert summary.bytes == 5
    assert summary.shards_done == store.query("SELECT COUNT(*) FROM shards")[0][0]


def test_concurrent_workers_do_not_duplicate_or_lose_files(tmp_path):
    tree = {"/R": [f"d{i}/" for i in range(12)]}
    for i in range(12):
        tree[f"/R/d{i}"] = [f"f{j}.txt" for j in range(15)]
    store, crawler = build(tmp_path, FakeLister(tree, page_size=3), workers=6, split_depth=1)
    crawler.seed("/R")
    crawler.run()
    assert store.stats()["files"] == 180
    duplicates = store.query(
        "SELECT COUNT(*) FROM (SELECT dbx_id FROM files GROUP BY dbx_id HAVING COUNT(*) > 1)"
    )[0][0]
    assert duplicates == 0


def test_shallow_shards_split_even_when_the_queue_is_busy(tmp_path):
    """The straggler fix: a huge subtree must not open a recursive cursor early.

    Starvation-gated splitting alone never fires for shards claimed while the queue
    is healthy -- which is exactly when the biggest subtrees get claimed.
    """
    tree = {"/R": [f"d{i}/" for i in range(50)]}
    tree.update({f"/R/d{i}": ["f.txt"] for i in range(50)})
    store, crawler = build(tmp_path, FakeLister(tree), workers=1,
                           split_depth=6, min_split_depth=2)
    crawler.seed("/R")
    crawler.run()
    modes = dict(store.query("SELECT path, mode FROM shards"))
    assert modes["/R"] == "split", "root must split despite 50 pending shards"
    assert store.stats()["files"] == 50


def test_min_split_depth_zero_restores_pure_starvation_gating(tmp_path):
    tree = {"/R": [f"d{i}/" for i in range(50)]}
    tree.update({f"/R/d{i}": ["f.txt"] for i in range(50)})
    store, crawler = build(tmp_path, FakeLister(tree), workers=1,
                           split_depth=6, min_split_depth=0)
    crawler.seed("/R")
    crawler.run()
    assert store.stats()["files"] == 50


def test_min_split_depth_cannot_exceed_the_hard_bound(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=1,
                           split_depth=0, min_split_depth=5)
    crawler.seed("/R")
    crawler.run()
    assert store.query("SELECT COUNT(*) FROM shards")[0][0] == 1  # never split
    assert files_in(store) == ALL_FILES
