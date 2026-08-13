"""The crawl loop: claim a shard, list it, commit each page with its cursor.

Two ideas carry the whole design.

*Shards.* A shard is a subtree listed under one cursor. Because the cursor is
committed with the rows it follows, an interrupted crawl resumes exactly, losing at
most the page in flight.

*Splitting.* One worker grinding through a huge subtree while the rest idle is the
main way this crawl wastes time -- in TeamSpace, `Public` alone holds 17,906
directories at depth 3. A worker can therefore spend one API call splitting a shard
into its children instead of listing it recursively. The choice is always made
*before* a cursor exists, so splitting never throws away work.

The rule is **keep just `queue_target` shards pending** (default: twice the worker
count): a worker splits what it claims while the queue is below target, and lists
recursively once it is stocked.

Splitting is kept deliberately rare. Measurement showed that splitting harder to
keep workers busy is a trap -- it trades 2000-entry recursive calls for many
near-empty ones, and collapsed throughput by two orders of magnitude on a real
subtree. `_should_split` carries the numbers.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass

from .api import (
    ApiError,
    AuthExpired,
    CursorReset,
    PathNotFound,
    RateLimited,
    TransientError,
)

log = logging.getLogger("dbaudit.crawler")


@dataclass
class RunSummary:
    files: int = 0
    dirs: int = 0
    bytes: int = 0
    pages: int = 0
    shards_done: int = 0
    errors: int = 0
    rate_limit_events: int = 0


class Crawler:
    def __init__(
        self,
        store,
        lister,
        limiter,
        workers: int = 8,
        split_depth: int = 6,
        queue_target: int | None = None,
        max_shards: int = 2_000_000,
        max_page_attempts: int = 8,
        backoff_base: float = 2.0,
        idle_poll: float = 0.2,
    ):
        self.store = store
        self.lister = lister
        self.limiter = limiter
        self.workers = max(1, int(workers))
        self.split_depth = int(split_depth)
        # Default 2 x workers: just enough pending shards to keep every worker fed,
        # and deliberately no more. See _should_split for why more is worse.
        self.queue_target = max(1, int(queue_target)) if queue_target else 2 * self.workers
        self.max_shards = int(max_shards)
        self.max_page_attempts = int(max_page_attempts)
        self.backoff_base = float(backoff_base)
        self.idle_poll = float(idle_poll)

        self._page_counter = itertools.count(1)
        self._counter_lock = threading.Lock()
        self._errors = 0

    # ---- seeding --------------------------------------------------------

    def seed_delta_cursor(self, root: str) -> str | None:
        """Take the whole-tree cursor before crawling, if the lister supports it."""
        getter = getattr(self.lister, "get_latest_cursor", None)
        if getter is None:
            return None
        cursor = getter(root, True, True)
        if cursor:
            self.store.set_delta_cursor(cursor)
        return cursor

    def seed(self, root: str) -> int:
        """Create the single root shard. Everything else grows from splits.

        Shard depth is measured *relative to the crawl root*, so ``--split-depth 2``
        means "split the root and its children", leaving recursive shards two levels
        below the root regardless of how deep the root itself sits.
        """
        root = "/" + root.strip("/")
        self.store.set_meta("root", root)
        return self.store.add_shard(root, depth=0, mode="recursive")

    # ---- the decision that keeps workers busy ---------------------------

    def _should_split(self, shard) -> bool:
        if shard.mode == "delta":
            return False  # a whole-tree change feed, never a listing to split
        # The persisted mode is checked first and wins unconditionally. A cursor
        # belongs to the kind of listing that created it, so a resumed split shard
        # must stay split: continuing a non-recursive cursor in recursive mode would
        # stop enqueueing child shards and silently drop whole subtrees.
        if shard.mode == "split":
            return True
        if shard.cursor is not None:
            return False  # a recursive cursor is already open
        if shard.depth >= self.split_depth:
            return False

        # Split only enough to keep the queue barely stocked -- that is, split as
        # LITTLE as possible.
        #
        # This is counter-intuitive, and three A/B runs on /TeamSpace/Public were
        # needed to learn it. The instinct is that more shards means more
        # parallelism, so a straggler leaving workers idle should be fixed by
        # splitting harder. Both attempts at that were much worse, on the same
        # subtree against a 5-minute baseline of ~319,000 files:
        #
        #   split the top 2 levels unconditionally -> 17,908 shards; 18k files in the
        #                                             first 30s against 53k
        #   keep 200 shards queued                 -> 17,878 shards; throughput
        #                                             collapsed to 4 files/s
        #
        # The scarce resource is *requests*, not workers. A recursive listing of a
        # large subtree returns 2000 entries per call; a shard covering one small
        # directory returns a handful. Splitting therefore trades one high-yield call
        # for many low-yield ones -- which is exactly rclone's per-directory cost
        # model, the thing this crawler exists to avoid.
        #
        # An idle worker costs one concurrent request. An over-split tree costs the
        # entire request budget. So the target stays low, and stragglers are accepted.
        if self.store.pending_count() >= self.queue_target:
            return False
        # Checked last on purpose: it is a full count, and it only matters on the
        # rare occasions we are actually about to split.
        return self.store.shard_count() < self.max_shards

    # ---- per-shard work -------------------------------------------------

    def _process(self, shard, stop: threading.Event, after_page) -> None:
        if shard.mode == "delta":
            mode = "delta"
            if not shard.cursor:
                self._fail(shard, "delta shard has no cursor; run a full pass")
                return
        else:
            mode = "split" if self._should_split(shard) else "recursive"
            if mode != shard.mode:
                self.store.set_shard_mode(shard.id, mode)

        cursor = shard.cursor
        attempts = 0

        while not stop.is_set():
            try:
                with self.limiter.slot():
                    if stop.is_set():
                        break
                    page = (
                        self.lister.continue_(cursor)
                        if cursor
                        else self.lister.list_folder(shard.path, recursive=(mode == "recursive"))
                    )
            except RateLimited as exc:
                self.limiter.on_rate_limited(exc.retry_after)
                self.store.log_event("rate_limited", f"retry_after={exc.retry_after}")
                continue
            except CursorReset:
                self.store.log_event("cursor_reset", shard.path)
                if mode == "delta":
                    # Re-listing is not an option here: this cursor stands for the
                    # whole tree, so recovery means a fresh full pass.
                    self._fail(shard, "delta cursor invalidated by Dropbox; "
                                      "re-run a full pass to re-establish it")
                    return
                self.store.clear_cursor(shard.id)
                cursor = None
                continue
            except PathNotFound as exc:
                self.store.log_event("path_gone", f"{shard.path}: {exc}")
                self.store.finish_shard(shard.id, note="path vanished during crawl")
                return
            except AuthExpired:
                # The token provider refreshes on the next headers() call; one retry.
                self.store.log_event("auth_expired", shard.path)
                attempts += 1
                if attempts >= self.max_page_attempts:
                    self._fail(shard, "auth kept expiring")
                    return
                continue
            except TransientError as exc:
                attempts += 1
                if attempts >= self.max_page_attempts:
                    self._fail(shard, f"transient failures exhausted: {exc}")
                    return
                self.store.log_event("transient", f"{shard.path}: {exc}")
                time.sleep(min(self.backoff_base ** attempts, 60.0))
                continue
            except ApiError as exc:
                self._fail(shard, f"{type(exc).__name__}: {exc}")
                return

            attempts = 0
            self.limiter.on_success()

            # A non-recursive listing does not return the folder itself, but guard
            # anyway: enqueueing the shard's own path would spawn a shard that
            # re-lists the same directory forever.
            children = (
                [
                    (entry["path_display"], shard.depth + 1, "recursive")
                    for entry in page.entries
                    if entry.get(".tag") == "folder"
                    and entry.get("path_display")
                    and entry["path_display"].rstrip("/").lower() != shard.path.rstrip("/").lower()
                ]
                if mode == "split"
                else ()
            )

            self.store.commit_page(
                shard.id,
                page.entries,
                cursor=page.cursor,
                has_more=page.has_more,
                child_shards=children,
                shard_path=shard.path,
            )
            cursor = page.cursor

            if after_page is not None:
                with self._counter_lock:
                    n = next(self._page_counter)
                after_page(n)

            if not page.has_more:
                return

        # Interrupted: hand the shard back with its cursor intact.
        self.store.release_shard(shard.id)

    def _fail(self, shard, message: str) -> None:
        log.error("shard %s failed: %s", shard.path, message)
        self.store.log_event("shard_error", f"{shard.path}: {message}")
        self.store.fail_shard(shard.id, message)
        with self._counter_lock:
            self._errors += 1

    # ---- worker pool ----------------------------------------------------

    def _worker(self, name: str, stop: threading.Event, after_page) -> None:
        store = self.store
        try:
            while not stop.is_set():
                shard = store.claim_shard(name)
                if shard is None:
                    if store.running_count() == 0:
                        return
                    time.sleep(self.idle_poll)
                    continue
                try:
                    self._process(shard, stop, after_page)
                except Exception as exc:  # never let one shard kill the pool
                    log.exception("unhandled error on shard %s", shard.path)
                    self._fail(shard, f"unhandled: {type(exc).__name__}: {exc}")
        finally:
            store.close()  # one connection per worker, closed once at the end

    def run(self, stop_event=None, incremental: bool = False, _after_page=None) -> RunSummary:
        stop = stop_event or threading.Event()
        if incremental:
            self.store.begin_incremental_pass()

        # Any shard still marked running belongs to a process that died; the
        # single-instance lock guarantees no live owner. Cursors are preserved.
        reclaimed = self.store.reset_stale_shards()
        if reclaimed:
            log.info("reclaimed %d shard(s) from a previous run", reclaimed)

        threads = [
            threading.Thread(target=self._worker, args=(f"w{i}", stop, _after_page),
                             name=f"dbaudit-w{i}", daemon=True)
            for i in range(self.workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        progress = self.store.progress()
        return RunSummary(
            files=progress["files"],
            dirs=progress["dirs"],
            bytes=progress["bytes"],
            pages=progress["pages"],
            shards_done=progress["shards_done"],
            errors=self._errors,
            rate_limit_events=getattr(self.limiter, "rate_limit_events", 0),
        )
