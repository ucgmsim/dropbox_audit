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
        one_level_after: int = 20,
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
        # How many of a folder's subfolders must turn out to hold no folders of their
        # own before the rest are listed one level at a time; see _one_level_by_siblings.
        self.one_level_after = int(one_level_after)

        self._page_counter = itertools.count(1)
        self._counter_lock = threading.Lock()
        self._errors = 0
        self._siblings = {}             # parent path -> (subfolders listed, any had folders)
        self._siblings_lock = threading.Lock()

    # ---- seeding --------------------------------------------------------

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
        # The persisted mode is checked first and wins unconditionally. A cursor
        # belongs to the kind of listing that created it, so a resumed split shard
        # must stay split: continuing a non-recursive cursor in recursive mode would
        # stop enqueueing child shards and silently drop whole subtrees.
        if shard.mode == "split":
            return True
        if shard.cursor is not None:
            return False  # a recursive cursor is already open
        if self._one_level_by_siblings(shard.path):
            return True
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

    # ---- learning from siblings -----------------------------------------

    def _one_level_by_siblings(self, path) -> bool:
        """Whether to list this folder one level at a time, judging by its siblings.

        For a folder with no subfolders, a one-level listing returns everything a
        recursive one would, in one request per 2000 entries -- while Dropbox can take
        several requests over a recursive listing of even a few files (about three for
        nine files, measured on 260k such siblings in October 2026). Whether a folder
        has subfolders is only known once it is listed, so the crawler predicts it from
        the folder's siblings: once `one_level_after` of them have turned out to hold
        no folders, the rest are listed one level. One that does hold folders turns
        this off for its parent. The guess costs requests either way, never entries:
        a one-level listing queues any subfolders it finds as shards of their own.
        """
        parent = path.rstrip("/").rsplit("/", 1)[0]
        with self._siblings_lock:
            listed, had_folders = self._siblings.get(parent, (0, False))
        return listed >= self.one_level_after and not had_folders

    def _record_listed(self, path, had_folders) -> None:
        parent = path.rstrip("/").rsplit("/", 1)[0]
        with self._siblings_lock:
            listed, any_folders = self._siblings.get(parent, (0, False))
            before = listed >= self.one_level_after and not any_folders
            listed, any_folders = listed + 1, any_folders or had_folders
            self._siblings[parent] = (listed, any_folders)
            after = listed >= self.one_level_after and not any_folders
        if after and not before:
            self.store.log_event("one_level_on", f"{parent}: {listed} subfolders held "
                                                 f"no folders; listing the rest one level")
        elif before and not after:
            self.store.log_event("one_level_off", f"{parent}: {path} holds folders")

    # ---- per-shard work -------------------------------------------------

    def _process(self, shard, stop: threading.Event, after_page) -> None:
        mode = "split" if self._should_split(shard) else "recursive"
        if mode != shard.mode:
            self.store.set_shard_mode(shard.id, mode)

        cursor = shard.cursor
        attempts = 0
        # What the listing shows about the folder, for _one_level_by_siblings. "No
        # folders" is only known from a listing begun in this process.
        listed_from_start = cursor is None
        had_folders = False

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
                # Dropbox gives up on a recursive listing of a big enough subtree -- an
                # HTTP 500 after about 180 s, whatever the page size -- while one level of
                # the same folder comes back in a second. Retrying cannot help, so list it
                # one level at a time instead: its children are smaller and try a
                # recursive listing of their own. That failure repeats every time, while
                # one blip would cost a listing Dropbox can manage a split into many: so
                # at a second failure in a row if it has no cursor, and only where it
                # would otherwise fail if it has, since splitting drops the pages it
                # holds. This ignores split_depth, which bounds only the splitting done
                # to keep workers busy.
                if (mode == "recursive"
                        and attempts >= (2 if cursor is None else self.max_page_attempts)
                        and self.store.shard_count() < self.max_shards):
                    self.store.log_event("split_after_failure", f"{shard.path}: {exc}")
                    if cursor is not None:
                        self.store.clear_cursor(shard.id)
                    mode, cursor, attempts = "split", None, 0
                    self.store.set_shard_mode(shard.id, mode)
                    continue
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

            # A recursive listing returns the folder itself first. A non-recursive one
            # does not, but guard anyway: enqueueing the shard's own path would spawn a
            # shard that re-lists the same directory forever.
            subfolders = [
                entry["path_display"]
                for entry in page.entries
                if entry.get(".tag") == "folder"
                and entry.get("path_display")
                and entry["path_display"].rstrip("/").lower() != shard.path.rstrip("/").lower()
            ]
            had_folders = had_folders or bool(subfolders)
            children = (
                [(sub, shard.depth + 1, "recursive") for sub in subfolders]
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
                if had_folders or listed_from_start:
                    self._record_listed(shard.path, had_folders)
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

    def run(self, stop_event=None, _after_page=None) -> RunSummary:
        stop = stop_event or threading.Event()

        # Any shard still marked running belongs to a process that died; the
        # single-instance lock guarantees no live owner. Cursors are preserved.
        reclaimed = self.store.reset_stale_shards()
        if reclaimed:
            log.info("reclaimed %d shard(s) from a previous run", reclaimed)
        retried = self.store.retry_failed_shards()
        if retried:
            log.info("retrying %d shard(s) that failed in a previous run, from scratch",
                     retried)

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
