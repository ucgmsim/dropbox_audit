"""Command line: init, run, status, index.

`run` is the long-lived command. It is designed to be started under nohup, tmux or
systemd and left alone: it takes an exclusive lock, reclaims any shards orphaned by
a previous crash, logs progress on an interval, and on SIGINT/SIGTERM finishes the
page in flight, commits it, and exits 0. Re-running `run` is how you resume.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time

from .api import HttpLister
from .auth import AuthError, TokenProvider
from .crawler import Crawler
from .limiter import AdaptiveLimiter
from .lock import InstanceLock, LockHeld
from .store import Store

log = logging.getLogger("dbaudit")

# Measured on the live account 2026-08-13: 8 concurrent listers sustained
# ~4,600 entries/s with zero 429s; 16 started drawing them.
DEFAULT_WORKERS = 8
DEFAULT_RPS = 5.0
DEFAULT_SPLIT_DEPTH = 2


def free_bytes(path: str) -> int:
    target = os.path.dirname(os.path.abspath(path)) or "."
    while not os.path.isdir(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    stat = os.statvfs(target)
    return stat.f_bavail * stat.f_frsize


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n} B"


def human_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def setup_logging(verbose: bool = False, logfile: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_lister(remote: str, include_deleted: bool = False):
    tokens = TokenProvider(remote=remote)
    return tokens, HttpLister(tokens, include_deleted=include_deleted)


# ---- subcommands --------------------------------------------------------


def cmd_init(args) -> int:
    projected = args.min_free_gb * 1024**3
    available = free_bytes(args.db)
    if available < projected:
        print(
            f"error: only {human_bytes(available)} free where the database would live, "
            f"but --min-free-gb {args.min_free_gb} requires {human_bytes(projected)}.\n"
            f"       A 100M-file audit needs roughly 20 GB including indexes.",
            file=sys.stderr,
        )
        return 2

    root = "/" + args.root.strip("/")
    try:
        tokens, lister = build_lister(args.remote)
        account = tokens.account()
        page = lister.list_folder(root, recursive=False)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: could not list {root}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    store = Store(args.db)
    store.init_schema()
    store.set_meta("root", root)
    store.set_meta("remote", args.remote)
    store.set_meta("schema_version", "1")
    store.set_meta("root_namespace_id", tokens.root_namespace_id())
    store.set_meta("account_id", account.get("account_id", ""))
    store.set_meta("team", (account.get("team") or {}).get("name", ""))
    store.set_meta("created_at", time.time())
    store.set_meta("seen_run", store.get_meta("seen_run", "1"))

    crawler = Crawler(store, lister, AdaptiveLimiter(), split_depth=args.split_depth)
    crawler.seed(root)

    print(f"initialised {args.db}")
    print(f"  root      : {root} ({len(page.entries)}+ entries at the top level)")
    print(f"  account   : {account.get('email', '?')} / team {(account.get('team') or {}).get('name', '?')}")
    print(f"  namespace : {tokens.root_namespace_id()}")
    print(f"  free disk : {human_bytes(available)}")
    print(f"\nnext: python -m dbaudit run --db {args.db}")
    return 0


def _progress_reporter(store, stop: threading.Event, interval: float) -> None:
    last_files, last_time = None, time.time()
    while not stop.wait(interval):
        try:
            prog = store.progress()
        except Exception:
            continue
        now = time.time()
        rate = ""
        eta = ""
        if last_files is not None and now > last_time:
            per_sec = (prog["files"] - last_files) / (now - last_time)
            rate = f" | {per_sec:,.0f} files/s"
            done, total = prog["shards_done"], prog["shards_total"]
            if per_sec > 0 and done:
                remaining = (total - done) / done * prog["files"]
                eta = f" | eta {human_duration(remaining / per_sec)}"
        last_files, last_time = prog["files"], now
        log.info(
            "shards %d/%d done (%d error) | %s files, %s dirs, %s%s%s",
            prog["shards_done"], prog["shards_total"], prog["shards_error"],
            f"{prog['files']:,}", f"{prog['dirs']:,}", human_bytes(prog["bytes"]), rate, eta,
        )
        store.close()


def cmd_run(args) -> int:
    setup_logging(args.verbose, args.log)
    store = Store(args.db)
    if not store.is_initialised() or not store.get_meta("root"):
        print(f"error: {args.db} is not initialised; run `init` first", file=sys.stderr)
        return 2

    lockfile = args.lock or f"{args.db}.lock"
    try:
        lock = InstanceLock(lockfile)
        lock.__enter__()
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    try:
        remote = store.get_meta("remote", "dropbox")
        try:
            _, lister = build_lister(remote)
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        limiter = AdaptiveLimiter(rps=args.rps, max_concurrency=args.workers)
        crawler = Crawler(
            store, lister, limiter,
            workers=args.workers,
            split_depth=args.split_depth,
            max_shards=args.max_shards,
        )

        stop = threading.Event()
        hard = {"count": 0}

        def handle(signum, frame):
            hard["count"] += 1
            if hard["count"] == 1:
                log.warning("signal %d: finishing the page in flight, then exiting", signum)
                stop.set()
            else:
                log.warning("second signal: exiting immediately")
                os._exit(1)

        signal.signal(signal.SIGINT, handle)
        signal.signal(signal.SIGTERM, handle)

        store.connect().execute(
            "INSERT INTO runs(started_at, mode, host) VALUES(?, ?, ?)",
            (time.time(), "incremental" if args.incremental else "full", socket.gethostname()),
        )

        reporter_stop = threading.Event()
        reporter = threading.Thread(
            target=_progress_reporter,
            args=(Store(args.db), reporter_stop, args.progress_interval),
            daemon=True,
        )
        reporter.start()

        started = time.time()
        log.info("crawling %s with %d workers at %.1f req/s",
                 store.get_meta("root"), args.workers, args.rps)
        try:
            summary = crawler.run(stop_event=stop, incremental=args.incremental)
        finally:
            reporter_stop.set()

        elapsed = time.time() - started
        store.connect().execute(
            "UPDATE runs SET finished_at=? WHERE id=(SELECT MAX(id) FROM runs)", (time.time(),)
        )

        log.info(
            "finished in %s: %s files, %s dirs, %s across %d pages (%d rate-limit events)",
            human_duration(elapsed), f"{summary.files:,}", f"{summary.dirs:,}",
            human_bytes(summary.bytes), summary.pages, summary.rate_limit_events,
        )

        remaining = store.pending_count() + store.running_count()
        if stop.is_set():
            log.info("interrupted with %d shard(s) outstanding; re-run to resume", remaining)
            return 0
        if summary.errors:
            log.warning("%d shard(s) ended in error; see `status`", summary.errors)
            return 1
        if remaining == 0:
            store.set_meta("last_full_pass_completed_at", time.time())
            if not args.no_index:
                log.info("crawl complete; building analysis indexes")
                store.build_indexes()
                log.info("indexes built")
        return 0
    finally:
        lock.__exit__(None, None, None)


def cmd_status(args) -> int:
    store = Store(args.db)
    root = store.get_meta("root") if store.is_initialised() else None
    if not root:
        print(f"error: {args.db} is not initialised", file=sys.stderr)
        return 2

    prog = store.progress()
    pending = store.pending_count()
    running = store.running_count()
    recent_429 = store.query(
        "SELECT COUNT(*) FROM api_events WHERE kind='rate_limited' AND ts > ?",
        (time.time() - 3600,),
    )[0][0]
    errors = store.query(
        "SELECT path, error FROM shards WHERE state='error' ORDER BY id LIMIT 5"
    )

    print(f"root      : {root}")
    print(f"shards    : {prog['shards_done']}/{prog['shards_total']} done "
          f"({pending} pending, {running} running, {prog['shards_error']} error)")
    print(f"files     : {prog['files']:,}")
    print(f"dirs      : {prog['dirs']:,}")
    print(f"bytes     : {human_bytes(prog['bytes'])}")
    print(f"pages     : {prog['pages']:,}")
    print(f"429s (1h) : {recent_429}")

    completed = store.get_meta("last_full_pass_completed_at")
    if completed:
        age = time.time() - float(completed)
        print(f"last pass : completed {human_duration(age)} ago")
    if errors:
        print("\nshards in error:")
        for path, message in errors:
            print(f"  {path}: {(message or '')[:120]}")
    return 0


def cmd_index(args) -> int:
    setup_logging(args.verbose)
    store = Store(args.db)
    log.info("building analysis indexes (this scans the whole table once per index)")
    started = time.time()
    store.build_indexes()
    log.info("done in %s", human_duration(time.time() - started))
    return 0


# ---- argument parsing ---------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbaudit", description="Read-only audit of a Dropbox subtree."
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the database and seed the root shard")
    p_init.add_argument("--db", required=True)
    p_init.add_argument("--root", required=True, help="e.g. /TeamSpace")
    p_init.add_argument("--remote", default="dropbox", help="rclone remote to borrow credentials from")
    p_init.add_argument("--split-depth", type=int, default=DEFAULT_SPLIT_DEPTH)
    p_init.add_argument("--min-free-gb", type=float, default=30.0)
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="crawl until finished; safe to interrupt and re-run")
    p_run.add_argument("--db", required=True)
    p_run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_run.add_argument("--rps", type=float, default=DEFAULT_RPS)
    p_run.add_argument("--split-depth", type=int, default=DEFAULT_SPLIT_DEPTH)
    p_run.add_argument("--max-shards", type=int, default=2_000_000)
    p_run.add_argument("--incremental", action="store_true",
                       help="re-audit changes only, using the cursors from a completed pass")
    p_run.add_argument("--progress-interval", type=float, default=30.0)
    p_run.add_argument("--no-index", action="store_true",
                       help="skip building analysis indexes when the crawl completes")
    p_run.add_argument("--log", help="also write logs to this file")
    p_run.add_argument("--lock", help="lockfile path (default: <db>.lock)")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="show progress")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(func=cmd_status)

    p_index = sub.add_parser("index", help="build analysis indexes")
    p_index.add_argument("--db", required=True)
    p_index.set_defaults(func=cmd_index)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "verbose"):
        args.verbose = False
    return args.func(args)
