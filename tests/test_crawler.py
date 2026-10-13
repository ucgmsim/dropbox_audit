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
    store, crawler = build(tmp_path, FakeLister(tree), workers=1, split_depth=1, queue_target=1)
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


class StallsAfterOnePage(FakeLister):
    """What Dropbox did on a 13.8M-file folder in October 2026: the first page of a
    listing came back, then every continue timed out."""

    def __init__(self, tree, stalled, **kw):
        super().__init__(tree, **kw)
        self.stalled = stalled

    def continue_(self, cursor):
        if cursor.startswith(self.stalled + "|"):
            from dbaudit.api import TransientError

            self.calls.append(("cont", cursor, None))
            raise TransientError("read timed out")
        return super().continue_(cursor)


def crawl_with_a_failed_shard(tmp_path):
    """/R/a fails holding a cursor: with the queue short it is listed one level at a
    time, and the continue after its first page keeps failing."""
    store, crawler = build(tmp_path, StallsAfterOnePage(TREE, "/R/a", page_size=2),
                           workers=1, max_page_attempts=2, backoff_base=0.01)
    crawler.seed("/R")
    assert crawler.run().errors == 1
    return store


def test_a_failed_shard_is_retried_on_the_next_run(tmp_path):
    store = crawl_with_a_failed_shard(tmp_path)
    summary = Crawler(store, FakeLister(TREE, page_size=2), limiter(), workers=1).run()
    assert summary.errors == 0
    assert files_in(store) == ALL_FILES
    assert store.query("SELECT COUNT(*) FROM files") == [(len(ALL_FILES),)]
    assert store.query("SELECT COUNT(*) FROM shards WHERE state='error'") == [(0,)]


def test_a_retried_shard_starts_over_instead_of_resuming_its_cursor(tmp_path):
    """Resuming would repeat the continue that kept failing; starting over lists the
    folder again from its first page."""
    store = crawl_with_a_failed_shard(tmp_path)
    lister = FakeLister(TREE, page_size=2)
    Crawler(store, lister, limiter(), workers=1).run()
    calls = [call for call in lister.calls if call[1].startswith("/R/a")]
    assert calls and calls[0][0] == "list"


class FullListingsFail(FakeLister):
    """Dropbox in October 2026: a recursive listing of a big enough subtree failed with
    HTTP 500 after about 180 s, whatever page size was asked for, while one level of the
    same folder came back in about a second."""

    def __init__(self, tree, failing, **kw):
        super().__init__(tree, **kw)
        self.failing = set(failing)

    def list_folder(self, path, recursive):
        if recursive and path in self.failing:
            from dbaudit.api import TransientError

            self.calls.append(("list", path, recursive))
            raise TransientError("500 from Dropbox")
        return super().list_folder(path, recursive)


def test_a_full_listing_that_keeps_failing_is_split_at_its_second_failure(tmp_path):
    """split_depth=0 asks for full listings everywhere; /R and /R/a fail as one."""
    lister = FullListingsFail(TREE, {"/R", "/R/a"})
    store, crawler = build(tmp_path, lister, workers=1, split_depth=0, backoff_base=0.01)
    crawler.seed("/R")
    assert crawler.run().errors == 0
    assert files_in(store) == ALL_FILES
    assert lister.calls.count(("list", "/R", True)) == 2
    assert store.query("SELECT COUNT(*) FROM api_events WHERE kind='split_after_failure'") \
        == [(2,)]


class FailsOnce(FakeLister):
    """A momentary failure: the first full listing of `path` fails, then all is well."""

    def __init__(self, tree, path, **kw):
        super().__init__(tree, **kw)
        self.path, self.failed = path, False

    def list_folder(self, path, recursive):
        if recursive and path == self.path and not self.failed:
            from dbaudit.api import TransientError

            self.failed = True
            raise TransientError("connection reset")
        return super().list_folder(path, recursive)


def test_one_failure_of_a_full_listing_is_retried_not_split(tmp_path):
    store, crawler = build(tmp_path, FailsOnce(TREE, "/R"), workers=1, split_depth=0,
                           backoff_base=0.01)
    crawler.seed("/R")
    assert crawler.run().errors == 0
    assert files_in(store) == ALL_FILES
    assert store.query("SELECT mode FROM shards WHERE path='/R'") == [("recursive",)]


def test_a_full_listing_failing_after_its_first_page_is_split_only_when_retries_run_out(tmp_path):
    """Splitting drops the pages it holds, so a blip must not do it: only where the shard
    would otherwise fail. The first page holds /R/top.txt; the split lists it again, once."""
    lister = StallsAfterOnePage(TREE, "/R", page_size=4)
    store, crawler = build(tmp_path, lister, workers=1, split_depth=0, max_page_attempts=3,
                           backoff_base=0.01)
    crawler.seed("/R")
    summary = crawler.run()
    assert summary.errors == 0
    assert summary.files == len(ALL_FILES)      # what `status` reports, from shard counters
    assert summary.dirs == store.stats()["dirs"]
    assert files_in(store) == ALL_FILES
    assert sum(1 for call in lister.calls if call[0] == "cont" and call[1].startswith("/R|1|")) \
        == 3


def test_a_split_interrupted_mid_listing_resumes_as_a_split(tmp_path):
    """Its cursor is now a one-level cursor; resuming it as a full listing would stop
    queueing the subfolders on its later pages, and whole subtrees would go missing."""
    store, crawler = build(tmp_path, FullListingsFail(TREE, {"/R"}, page_size=1), workers=1,
                           split_depth=0, backoff_base=0.01)
    crawler.seed("/R")
    stop = threading.Event()
    crawler.run(stop_event=stop, _after_page=lambda n: stop.set())
    assert store.query("SELECT mode, cursor IS NOT NULL FROM shards WHERE path='/R'") \
        == [("split", 1)]
    Crawler(Store(tmp_path / "a.db"), FakeLister(TREE, page_size=1), limiter(), workers=1,
            split_depth=0).run()
    assert files_in(store) == ALL_FILES


def test_a_failing_full_listing_is_not_split_past_max_shards(tmp_path):
    store, crawler = build(tmp_path, FullListingsFail(TREE, {"/R"}), workers=1, split_depth=0,
                           max_shards=1, max_page_attempts=2, backoff_base=0.01)
    crawler.seed("/R")
    assert crawler.run().errors == 1
    assert store.query("SELECT COUNT(*) FROM shards") == [(1,)]


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


def test_splitting_stays_far_below_splitting_everything(tmp_path):
    """Self-limiting, and that is the point.

    Splitting every directory is rclone's cost model: one API call per directory.
    Unbounded splitting produced 17,908 shards on a real subtree and collapsed
    throughput to 4 files/s, so the tree here (341 splittable directories) must be
    covered by far fewer shards than that.
    """
    tree = {"/R": [f"d{i}/" for i in range(30)]}
    for i in range(30):
        tree[f"/R/d{i}"] = [f"s{j}/" for j in range(10)]
        for j in range(10):
            tree[f"/R/d{i}/s{j}"] = ["f.txt"]
    store, crawler = build(tmp_path, FakeLister(tree), workers=2,
                           split_depth=6, queue_target=4)
    crawler.seed("/R")
    crawler.run()
    shards = store.query("SELECT COUNT(*) FROM shards")[0][0]
    assert shards < 150, f"split too eagerly: {shards} shards for 341 directories"
    assert store.stats()["files"] == 300


def test_queue_target_defaults_to_twice_the_workers(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=5)
    assert crawler.queue_target == 10


def test_a_busy_queue_still_splits_when_it_drains(tmp_path):
    """Self-arming: the queue target is re-checked on every claim, so a long shard
    cannot leave workers idle."""
    tree = {"/R": [f"d{i}/" for i in range(6)]}
    for i in range(6):
        tree[f"/R/d{i}"] = [f"s{j}/" for j in range(4)]
        for j in range(4):
            tree[f"/R/d{i}/s{j}"] = ["f.txt"]
    store, crawler = build(tmp_path, FakeLister(tree), workers=1,
                           split_depth=6, queue_target=100)
    crawler.seed("/R")
    crawler.run()
    # With a target it can never reach, every splittable level gets split.
    assert store.query("SELECT COUNT(*) FROM shards WHERE mode='split'")[0][0] >= 7
    assert store.stats()["files"] == 24


def test_queue_target_respects_the_depth_bound(tmp_path):
    store, crawler = build(tmp_path, FakeLister(TREE), workers=1,
                           split_depth=0, queue_target=10_000)
    crawler.seed("/R")
    crawler.run()
    assert store.query("SELECT COUNT(*) FROM shards")[0][0] == 1  # never split
    assert files_in(store) == ALL_FILES


def wide_tree(children=30, with_subfolder=()):
    """/R/wide holds subfolders c00, c01, ..., each with two files; the ones numbered in
    `with_subfolder` also hold a folder `sub` with one file."""
    tree = {"/R": ["wide/"], "/R/wide": [f"c{i:02}/" for i in range(children)]}
    for i in range(children):
        path = f"/R/wide/c{i:02}"
        tree[path] = ["x.txt", "y.txt"] + (["sub/"] if i in with_subfolder else [])
        if i in with_subfolder:
            tree[path + "/sub"] = ["z.txt"]
    return tree


def crawl_wide(tmp_path, tree):
    """split_depth=2 splits /R and /R/wide, never the c-folders; with one worker they are
    listed in order. A full listing of a c-folder takes 3 requests at page_size=1."""
    lister = FakeLister(tree, page_size=1)
    store, crawler = build(tmp_path, lister, workers=1, split_depth=2)
    crawler.seed("/R")
    assert crawler.run().errors == 0
    full = [c[1] for c in lister.calls if c[0] == "list" and c[1].count("/") == 3 and c[2]]
    one_level = [c[1] for c in lister.calls if c[0] == "list" and c[1].count("/") == 3 and not c[2]]
    return store, full, one_level


def files_of(tree):
    return {f"{path}/{name}" for path, names in tree.items() for name in names
            if not name.endswith("/")}


def test_once_20_siblings_had_no_subfolders_the_rest_are_listed_one_level(tmp_path):
    tree = wide_tree()
    store, full, one_level = crawl_wide(tmp_path, tree)
    assert full == [f"/R/wide/c{i:02}" for i in range(20)]
    assert one_level == [f"/R/wide/c{i:02}" for i in range(20, 30)]
    assert files_in(store) == files_of(tree)


def test_siblings_with_subfolders_are_all_listed_in_full(tmp_path):
    tree = wide_tree(with_subfolder=range(30))
    store, full, one_level = crawl_wide(tmp_path, tree)
    assert len(full) == 30 and one_level == []
    assert files_in(store) == files_of(tree)


def test_more_than_one_in_ten_siblings_with_folders_turns_one_level_listing_off(tmp_path):
    """c20 and c21 hold folders and leave it on (1 in 21, 2 in 22); c22 makes 3 in 23."""
    tree = wide_tree(with_subfolder={20, 21, 22})
    store, full, one_level = crawl_wide(tmp_path, tree)
    assert one_level == ["/R/wide/c20", "/R/wide/c21", "/R/wide/c22"]
    assert full == [f"/R/wide/c{i:02}" for i in range(30) if i not in (20, 21, 22)]
    assert files_in(store) == files_of(tree)


def test_a_listing_resumed_from_a_cursor_does_not_count_as_having_no_folders(tmp_path):
    """c00's full listing is stopped just after the page showing its folder `sub`
    (pages 1-6 split /R and /R/wide; c00's are 7-11, `sub` on page 10). Resumed in a
    new process, its remaining page shows no folder -- which must not make c00 count
    as folder-free, so with one_level_after=1 its sibling c01 still gets a full
    listing."""
    tree = wide_tree(children=5, with_subfolder={0})
    store, crawler = build(tmp_path, FakeLister(tree, page_size=1), workers=1, split_depth=2,
                           one_level_after=1)
    crawler.seed("/R")
    stop = threading.Event()
    crawler.run(stop_event=stop, _after_page=lambda n: stop.set() if n >= 10 else None)
    assert store.query("SELECT state, cursor IS NOT NULL FROM shards "
                       "WHERE path='/R/wide/c00'") == [("pending", 1)]
    assert store.query("SELECT COUNT(*) FROM dirs WHERE path_lower='/r/wide/c00/sub'") \
        == [(1,)]                               # the stop came after `sub` was seen

    lister = FakeLister(tree, page_size=1)
    Crawler(Store(tmp_path / "a.db"), lister, limiter(), workers=1, split_depth=2,
            one_level_after=1).run()
    assert ("list", "/R/wide/c01", True) in lister.calls
    assert files_in(store) == files_of(tree)
