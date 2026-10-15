"""Command line: init, run, status, index, and the `archive` group.

`run` is the long-lived command. It is designed to be started under nohup, tmux or
systemd and left alone: it takes an exclusive lock, reclaims any shards orphaned by
a previous crash, logs progress on an interval, and on SIGINT/SIGTERM finishes the
page in flight, commits it, and exits 0. Re-running `run` is how you resume.

`archive index` is the same shape one layer down. It walks a tar's header chain over
Dropbox byte ranges, one chain per part, and re-running it resumes from each chain's
committed cursor. A chain that does not start at offset 0 has to scan for its first
header, and a valid-looking header proves nothing -- a tar stored inside the tar hands
a cold scan perfectly good ones. `_join` is what settles it: a chain is believed only
once the chain before it walks into exactly the offset it started from.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import logging
import os
import re
import signal
import socket
import sqlite3
import sys
import tarfile
import tempfile
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .api import ApiError, AuthExpired, HttpLister, PathNotFound, RateLimited, TransientError
from .archive.candidates import find_candidates
from .archive.parts import ArchiveSet, ArchiveSetError
from .archive.reader import (WINDOW_MAX, WINDOW_MIN, ConcatFile, DropboxRangeReader,
                             LocalRangeReader, ReaderError)
from .archive import report as archive_report
from .archive.runner import Done, Job, Runner
from .archive.store import ArchiveStore
from .archive.tarwalk import (BLOCK, REREAD_LIMIT, Member, UnsupportedArchive,
                              WalkResult, find_chain_start, read_member, walk)
from .auth import AuthError, TokenProvider
from .crawler import Crawler
from .limiter import AdaptiveLimiter
from .lock import InstanceLock, LockHeld
from .store import Store
from .units import human_bytes, human_duration
from .verify import verify_subtree
from . import report as reports
from . import tree

log = logging.getLogger("dbaudit")

# Measured on the live account 2026-08-13: 8 concurrent listers sustained
# ~4,600 entries/s with zero 429s; 16 started drawing them.
DEFAULT_WORKERS = 8
DEFAULT_RPS = 5.0
# The ceiling the request rate may climb to while Dropbox answers without a 429.
DEFAULT_MAX_RPS = 20.0
# Chains for `archive index`. Deliberately NOT DEFAULT_WORKERS: that figure measures
# `files/list_folder`, a different endpoint class, and conflating the two is a mistake
# this project has already made once and corrected. The closest measured analogue for
# scattered, latency-bound `files/download` reads is a sibling project's commit
# fccd755, where 8 concurrent streams ran clean. The owner fixed this at 8 on 2026-09-18:
# nothing here raises it or adapts it upward.
ARCHIVE_WORKERS = 8
# Dropbox's own content hash: SHA-256 over the concatenated SHA-256 digests of 4 MiB
# blocks. Local archives are hashed the same way so both sources share one identity.
DROPBOX_HASH_BLOCK = 4 << 20
# `archive cat` (Task 8) already knows the member's exact data_offset and size from the
# index, so it reads through a pass-through ConcatFile window (window_min=window_max=1)
# instead of the walker's adaptive one, and chunks its own reads at this size. 16 MiB is
# the measured point where transfer starts to dominate the 1.60 s round trip, and it is
# already this codebase's window ceiling (WINDOW_MAX) for a single read -- so an EMOD3D
# output file costs 2 requests, anything under 16 MiB costs 1, and a small file like
# root_params.yaml is one small read rather than a padded minimum.
CAT_CHUNK = 16 << 20
# Consecutive `cat` reads overlap by this much and must agree on it. A 206 whose body
# is some other range -- Dropbox returned two in ~42,000 reads on 2026-09-21 -- cannot
# also reproduce the bytes its neighbouring read saw there. 4 KiB against a 16 MiB read
# is 0.02%, and a member under 16 MiB still costs one request.
CAT_OVERLAP = 4 << 10
# Two reads agreeing on bytes this regular says nothing about either: a wrong body of
# zeros, 0xFF fill or a repeated pattern agrees with any other. Shared bytes vouch for a
# read only if they compress less than this many times over (a tar header manages 5,
# text about 3, float output about 1; zeros 100, a 256-byte cycle 13)...
AMBIGUOUS_RATIO = 8
# ...and are at least this many.
AMBIGUOUS_BELOW = 512
# The most rows under repeated paths one audit reads again. A bad read leaves a handful;
# an archive grown by `tar -r` can repeat thousands of paths legitimately, and at one
# request per scattered row an unbounded audit could cost hours.
AUDIT_LIMIT = 1000
# The file types `cat` will extract, decoded the same way Member.from_tarinfo
# (tarwalk.py) decodes them -- a single header byte through decode("ascii", "replace").
# Mirrors tarfile.REGULAR_TYPES minus GNUTYPE_SPARSE: a sparse member's data region is
# not a plain byte range (it interleaves a sparse map with the real data), so reading
# [data_offset, data_offset + size) for one would not reconstruct the file's bytes --
# excluding it is a correctness fix, not a narrowing for its own sake.
REGULAR_MEMBER_TYPES = frozenset(
    b.decode("ascii", "replace")
    for b in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE))
# A bound, not the primary control. Splitting is gated on the queue actually
# starving, so it is self-limiting; the depth only needs to be deep enough to let
# that gate act. At 2, big subtrees could not be split at all and the tail of a
# crawl ran on 2 of 8 workers (measured on /TeamSpace/Public: 2,400 -> 1,163 files/s).
DEFAULT_SPLIT_DEPTH = 6
# How many shards to keep queued. High enough that a long-running shard never
# leaves the other workers idle; low enough that per-shard API calls stay a rounding
# error. 17,908 shards on one subtree was far too many; a few hundred is right.
DEFAULT_QUEUE_TARGET = None  # => 2 x workers; see Crawler._should_split
# `tree`: at 1 GiB the October 2026 listing made a page of about 51k rows, and 100,000
# rows keeps a page to roughly 50 MB.
DEFAULT_TREE_MIN_SIZE = 1 << 30
DEFAULT_TREE_MAX_ROWS = 100_000


def free_bytes(path: str) -> int:
    target = os.path.dirname(os.path.abspath(path)) or "."
    while not os.path.isdir(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    stat = os.statvfs(target)
    return stat.f_bavail * stat.f_frsize


SIZE_UNITS = {"": 1, "b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30, "tib": 1 << 40}


def parse_size(text: str) -> int:
    """'1GiB', '100MiB' or a plain byte count, for argparse."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]iB|B)?\s*", text, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a size: give a byte count or a number with B, KiB, MiB, GiB "
            "or TiB")
    number, unit = match.groups()
    return int(float(number) * SIZE_UNITS[(unit or "").lower()])


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


def build_lister(remote: str, tokens=None):
    """A lister, with its own HTTP session, over ``tokens`` or a new provider. Pass the
    provider a run already holds: each new one shells out to rclone for its token."""
    tokens = tokens or TokenProvider(remote=remote)
    return tokens, HttpLister(tokens)


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
    store.set_meta("schema_version", "2")
    store.set_meta("root_namespace_id", tokens.root_namespace_id())
    store.set_meta("account_id", account.get("account_id", ""))
    store.set_meta("team", (account.get("team") or {}).get("name", ""))
    store.set_meta("created_at", time.time())

    store.add_shard(root, depth=0)

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
                # Rough: shards vary enormously in size, so this is a hint, not a promise.
                eta = f" | eta~{human_duration(remaining / per_sec)}"
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

        limiter = AdaptiveLimiter(rps=args.rps, max_rps=args.max_rps,
                                  max_concurrency=args.workers)
        crawler = Crawler(
            store, lister, limiter,
            workers=args.workers,
            split_depth=args.split_depth,
            queue_target=args.queue_target,
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
            "INSERT INTO runs(started_at, host) VALUES(?, ?)",
            (time.time(), socket.gethostname()),
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
            summary = crawler.run(stop_event=stop)
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
            log.warning("%d shard(s) ended in error; see `status`, and run again to "
                        "retry them from scratch", summary.errors)
            return 1
        if remaining == 0:
            store.set_meta("last_pass_completed_at", time.time())
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

    completed = store.get_meta("last_pass_completed_at")
    if completed:
        age = time.time() - float(completed)
        print(f"last pass : completed {human_duration(age)} ago")
    if errors:
        print("\nshards in error:")
        for path, message in errors:
            print(f"  {path}: {(message or '')[:120]}")
    return 0


def cmd_verify(args) -> int:
    store = Store(args.db)
    if not store.is_initialised():
        print(f"error: {args.db} is not initialised", file=sys.stderr)
        return 2
    target = args.path or store.get_meta("root")
    print(f"verifying {target} against an independent rclone walk ...")
    try:
        result = verify_subtree(store, target)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"  database : {result.db_files:,} files, {human_bytes(result.db_bytes)}")
    print(f"  rclone   : {result.rclone_files:,} files, {human_bytes(result.rclone_bytes)}")
    print(f"  {result.differences} differences")
    for name in result.only_in_rclone[:20]:
        print(f"    missing from audit : {name}")
    for name in result.only_in_db[:20]:
        print(f"    not seen by rclone : {name}")
    for name, db_size, remote_size in result.size_mismatches[:20]:
        print(f"    size differs       : {name} (audit {db_size}, rclone {remote_size})")
    return 0 if result.ok else 1


def _space_usage(remote: str):
    """Dropbox's own view of usage, for the reconciliation residual."""
    import requests

    tokens = TokenProvider(remote=remote)
    resp = requests.post(
        "https://api.dropboxapi.com/2/users/get_space_usage",
        headers={"Authorization": f"Bearer {tokens.access_token()}"}, timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    allocation = body.get("allocation") or {}
    return body.get("used", 0), allocation.get("used"), allocation.get("allocated")


def cmd_report(args) -> int:
    store = Store(args.db)
    if not store.is_initialised():
        print(f"error: {args.db} is not initialised", file=sys.stderr)
        return 2

    prog = store.progress()
    unfinished = store.pending_count() + store.running_count() + prog["shards_error"]
    root = store.get_meta("root")
    print(f"# Storage audit of {root}")
    if unfinished:
        print(f"\n  WARNING: the crawl is incomplete -- {unfinished} shard(s) unfinished "
              f"or failed.")
        print("  Every figure below is a lower bound.")
    print(f"\n{prog['files']:,} files, {prog['dirs']:,} directories, "
          f"{human_bytes(prog['bytes'])} of live data\n")

    print("## Reconciliation")
    live = store.query("SELECT COALESCE(SUM(size), 0) FROM files")[0][0]
    try:
        account_used, team_used, team_allocated = _space_usage(store.get_meta("remote", "dropbox"))
        _, _, residual = reports.reconcile(store, account_used)
        print(f"  measured live bytes under {root}: {human_bytes(live)}")
        print(f"  Dropbox reports for this account : {human_bytes(account_used)}")
        if team_used:
            print(f"  team space used / allocated      : {human_bytes(team_used)} / "
                  f"{human_bytes(team_allocated or 0)}")
        print(f"  unaccounted for                  : {human_bytes(residual)}")
        print("  The unaccounted figure is version history, deleted-but-retained data,")
        print("  AND anything outside this crawl root. It isolates version/deleted")
        print("  overhead only when the root covers everything the account owns.")
    except Exception as exc:
        print(f"  (could not read live space usage: {type(exc).__name__}: {exc})")
        print(f"  measured live bytes under {root}: {human_bytes(live)}")

    print("\n## Reclaimable duplicates")
    total_dupe = reports.total_reclaimable_duplicates(store)
    print(f"  identical content stored more than once: {human_bytes(total_dupe)} reclaimable")
    for row in reports.duplicate_report(store, limit=args.top)[:args.top]:
        print(f"  {human_bytes(row.reclaimable):>12}  x{row.copies:<4} "
              f"{human_bytes(row.size):>10} each  {row.example[:80]}")

    print("\n## Cold data")
    for years in (2, 5, 10):
        count, total = reports.bytes_older_than(store, years)
        print(f"  untouched for {years:>2}y+ : {human_bytes(total):>12} in {count:,} files")
    print("  by year last modified:")
    for row in reports.cold_bytes(store):
        print(f"    {row.year}  {human_bytes(row.bytes):>12}  {row.files:,} files")

    print("\n## Largest directories (recursive)")
    for row in reports.top_dirs(store, limit=args.top):
        print(f"  {human_bytes(row.bytes):>12}  {row.files:>10,} files  {row.path[:90]}")

    print("\n## Per top-level folder")
    for row in reports.top_level_summary(store):
        print(f"  {human_bytes(row.bytes):>12}  {row.files:>10,} files  {row.path[:90]}")

    print("\n## Small-file hotspots")
    hotspots = reports.small_file_hotspots(
        store, min_files=args.hotspot_min_files, max_mean_size=args.hotspot_max_mean, limit=args.top)
    if not hotspots:
        print(f"  none with >={args.hotspot_min_files:,} files averaging "
              f"<={human_bytes(args.hotspot_max_mean)}")
    for row in hotspots:
        print(f"  {row.files:>10,} files  mean {human_bytes(row.mean_size):>10}  "
              f"{human_bytes(row.bytes):>12}  {row.path[:70]}")

    print("\n## By file type")
    for row in reports.extension_profile(store, limit=args.top):
        label = "(no extension)" if row.ext == "(none)" else f".{row.ext}"
        print(f"  {human_bytes(row.bytes):>12}  {row.files:>10,} files  {label}")
    return 0


def cmd_export(args) -> int:
    store = Store(args.db)
    if not store.is_initialised():
        print(f"error: {args.db} is not initialised", file=sys.stderr)
        return 2
    for path in reports.export_csv(store, args.out):
        print(f"wrote {path}")
    return 0


def cmd_tree(args) -> int:
    store = Store(args.db)
    if not store.is_initialised():
        print(f"error: {args.db} is not initialised", file=sys.stderr)
        return 2
    if args.subtree and tree.find_dir(store, args.subtree) is None:
        print(f"error: {args.subtree} is not in {args.db}", file=sys.stderr)
        return 2
    # samefile also catches another name for the same file, e.g. AUDIT.DB for audit.db
    # on a case-insensitive disk, which realpath leaves alone.
    if os.path.realpath(args.out) == os.path.realpath(args.db) or (
            os.path.exists(args.out) and os.path.samefile(args.out, args.db)):
        print("error: --out names the database itself", file=sys.stderr)
        return 2
    folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(folder):
        print(f"error: {folder} is not a folder", file=sys.stderr)
        return 2
    started = time.time()
    page, summary, chosen = tree.build(store, args.subtree or store.get_meta("root") or "/",
                                       args.min_size, args.max_rows, os.path.basename(args.db))
    # Written beside the target and renamed into place, so a failed run never leaves
    # half a page behind.
    fd, partial = tempfile.mkstemp(dir=folder, prefix=".tree-", suffix=".html")
    try:
        with open(fd, "w", encoding="utf-8") as out:
            tree.render(out, page, summary, chosen)
        # mkstemp makes the file 0600; give it what any new file would get under the umask.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(partial, 0o666 & ~umask)
        os.replace(partial, args.out)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(partial)
        raise
    print(f"wrote {args.out}: {page.rows:,} rows, {human_bytes(os.path.getsize(args.out))}, "
          f"entries under {human_bytes(page.threshold)} folded "
          f"({human_duration(time.time() - started)})")
    return 0


def cmd_index(args) -> int:
    setup_logging(args.verbose)
    store = Store(args.db)
    log.info("building analysis indexes (this scans the whole table once per index)")
    started = time.time()
    store.build_indexes()
    log.info("done in %s", human_duration(time.time() - started))
    return 0


# ---- archive: what is inside the tars ------------------------------------


def _local_hash(path) -> str:
    """Dropbox's content hash for a local file.

    Computed here so a local archive and a Dropbox one land on the same identity
    (`ArchiveSet.set_hash`): registering the parts from disk and then re-registering
    the same parts from the folder they were uploaded to updates one row rather than
    creating a second archive.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(DROPBOX_HASH_BLOCK)
            if not block:
                break
            digest.update(hashlib.sha256(block).digest())
    return digest.hexdigest()


def _folder_entries(lister, folder: str, name: str) -> list[dict]:
    """Every file in `folder` whose name starts with `name`, across all pages.

    The prefix is the filter, not a guess at the part naming: `ArchiveSet.from_entries`
    is what decides whether those files actually form a gapless part sequence.
    """
    page = lister.list_folder(folder, recursive=False)
    entries = [e for e in page.entries
               if e.get(".tag") == "file" and e["name"].startswith(name)]
    while page.has_more:
        page = lister.continue_(page.cursor)
        entries += [e for e in page.entries
                    if e.get(".tag") == "file" and e["name"].startswith(name)]
    return entries


#: How many times a folder listing that failed on an expired token or a failed request is
#: asked for, backing off LISTING_BACKOFF ** n seconds (at most a minute) between tries. A
#: 429 is asked again however often it comes: the limiter first pauses every request for
#: its Retry-After.
LISTING_ATTEMPTS = 8
LISTING_BACKOFF = 2.0


def _listed(lister, limiter, folder, name, stop=None):
    """`_folder_entries` through the shared limiter, the way the crawl lists, or None if
    ``stop`` is set first."""
    tries = 0
    while stop is None or not stop.is_set():
        try:
            with limiter.slot():
                entries = _folder_entries(lister, folder, name)
        except RateLimited as exc:
            limiter.on_rate_limited(exc.retry_after)
            continue
        except (AuthExpired, TransientError) as exc:
            if isinstance(exc.__cause__, AuthError):
                raise exc.__cause__ from None   # no credentials: asking again cannot help
            tries += 1
            if tries >= LISTING_ATTEMPTS:
                raise
            time.sleep(min(LISTING_BACKOFF ** tries, 60.0))
            continue
        limiter.on_success()
        return entries
    return None


def _archive_set(args):
    """Build an ArchiveSet from Dropbox or from a local directory."""
    if args.local_dir:
        directory = Path(args.local_dir)
        entries = [{"name": p.name, "size": p.stat().st_size,
                    "path_display": str(p), "id": "", "rev": "",
                    "content_hash": _local_hash(p)}
                   for p in sorted(directory.iterdir())
                   # Same filter as the Dropbox branch: an unrelated file sitting in
                   # the directory is not a part of this archive.
                   if p.is_file() and p.name.startswith(args.name)]
        return ArchiveSet.from_entries(entries, args.name), "local", str(directory)
    _, lister = build_lister(args.remote)
    return (ArchiveSet.from_entries(_folder_entries(lister, args.folder, args.name),
                                    args.name),
            "dropbox", args.folder)


def _stored_archive_set(store, row) -> ArchiveSet:
    """Rebuild the coordinate space from what `register` recorded."""
    return ArchiveSet(store.parts_of(row["id"]))


def _reader_for(row, archive_set, tokens=None, limiter=None):
    """A reader per caller: a `requests.Session` is not thread-safe, so two chains
    cannot share one.

    Where a local archive's parts live is the folder recorded at registration -- `index`
    has no `--local-dir` of its own. Task 8's `cat` builds its `ConcatFile` from this
    helper and `_stored_archive_set`, so `cat` and `index` reach the bytes the same way.
    """
    if row["source"] == "local":
        return LocalRangeReader(row["folder"], archive_set)
    return DropboxRangeReader(archive_set, tokens, limiter)


def _fingerprint(store, row, remote, tokens, limiter, stop=None):
    """Compare every registered part against what the folder holds right now.

    Matching is by file id first and by name second: all 18 parts of v01p0_incomplete
    moved between folders in five weeks, and an id survives that where a path does not.
    Returns (missing, changed) part names, or None if ``stop`` is set before the folder
    is listed.
    """
    _, lister = build_lister(remote, tokens)
    entries = _listed(lister, limiter, row["folder"], row["name"], stop)
    if entries is None:
        return None
    by_id = {e["id"]: e for e in entries if e.get("id")}
    by_name = {e["name"]: e for e in entries}
    missing, changed = [], []
    for part in store.parts_of(row["id"]):
        entry = by_id.get(part.dbx_id) or by_name.get(part.name)
        if entry is None:
            missing.append(part.name)
        elif (entry.get("content_hash") or "") != part.content_hash:
            changed.append(part.name)
    return missing, changed


def _open_archive_store(path):
    """The archive store at `path`, or None if it holds no archive index.

    `index` and `status` never create one: `register` initialises the schema, so a typo
    in `--db` is a usage error rather than a silently created empty database.
    """
    if not os.path.exists(path):
        return None
    store = ArchiveStore(path)
    try:
        store.connect().execute("SELECT 1 FROM archives LIMIT 1").fetchone()
    except sqlite3.DatabaseError:
        # No such table, or the file is not a database at all -- both mean the same
        # thing to a caller who mistyped --db.
        return None
    return store


def _other_kind_of_database(path) -> bool:
    """A file at ``path`` that is not an archive database -- the audit database, given by
    mistake -- which `run` and `import` must not add their tables to."""
    if not os.path.exists(path):
        return False
    store = _open_archive_store(path)
    if store is None:
        return True
    store.close()
    return False


def cmd_archive_register(args) -> int:
    try:
        archive_set, source, folder = _archive_set(args)
    except ArchiveSetError as exc:
        # These messages already name the offending part.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                    # a missing directory, or a failed listing
        print(f"error: could not read {args.local_dir or args.folder}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    store = ArchiveStore(args.db)
    store.init_schema()
    archive_id = store.register(args.name, "tar", folder, source, archive_set.parts)
    store.seed_segments(archive_id, archive_set.parts)

    print(f"registered {args.name} from {source}")
    print(f"  folder : {folder}")
    print(f"  parts  : {len(archive_set.parts)}")
    print(f"  size   : {human_bytes(archive_set.total_size)}")
    print(f"\nnext: python -m dbaudit archive index --db {args.db} --archive {args.name}")
    return 0


class _Stopped(Exception):
    """The run is over. Raised out of a commit callback so the batch is NOT written."""


class _Run:
    """What every chain in the pool shares.

    One `TokenProvider` and one `AdaptiveLimiter` for the whole command: Dropbox limits
    per account, and the provider caches its token behind a lock -- one per worker would
    each shell out to rclone to refresh. Everything else (reader, window, cache) is per
    chain, because a `requests.Session` is not thread-safe.
    """

    def __init__(self, args, store, row, archive_set, tokens, limiter, stop):
        self.args = args
        self.store = store
        self.row = row
        self.archive_id = row["id"]
        self.archive_set = archive_set
        self.tokens = tokens
        self.limiter = limiter
        #: The caller's stop, which a signal sets -- in `archive run`, every archive's --
        #: and this archive's own, which a pax header or `--max-batches` sets.
        self.stop = stop
        self.halt = threading.Event()
        self.lock = threading.Lock()
        self.batches = 0
        #: Set by a worker whose own machinery failed, as opposed to a bad part. A run
        #: that ends this way must not exit 0: on a nohup'd walk the exit code is most
        #: of what anyone sees, and "stopped cleanly" and "a chain blew up" are not the
        #: same answer.
        self.failed = False
        #: Set by the first chain to meet a pax header. The format is the archive's, not
        #: one part's, so every other chain would meet it too: the walk stops.
        self.unsupported = None

    def stopping(self) -> bool:
        return self.stop.is_set() or self.halt.is_set()

    def commit(self, segment_id, members, next_offset, requests, bytes_fetched) -> None:
        """Write one batch, or refuse to once `--max-batches` has been reached.

        Refusing *before* the write is what makes `--max-batches N` mean exactly N
        batches rather than N plus whatever was in flight. Holding the lock across the
        write is fine: SQLite serialises writers anyway.

        Only a commit that carries members is refused. An empty one buys nothing, so
        refusing it cannot help the budget, and it costs a cursor. A verdict the limit
        cuts short -- the walk stops at the next member once it is reached, a verdict's
        second reading too, and the commit that confirms a verdict carries the members
        held back for it -- is read again, from where those members begin, on the next
        run.

        Only a batch that carries members counts against the budget. A part lying
        wholly inside one member has no header of its own, so its chain crosses the
        moment it starts and commits nothing but a cursor -- which it still must do,
        because that cursor is how the segment records that it crossed. Charging those
        to the budget would buy no members at all, and `--max-batches` exists to bound
        how much gets walked, which is measured in members.
        """
        with self.lock:
            limit = self.args.max_batches
            if members and limit and self.batches >= limit:
                raise _Stopped(f"--max-batches {limit} reached")
            self.store.commit_batch(self.archive_id, segment_id, members, next_offset,
                                    requests, bytes_fetched)
            if members:
                self.batches += 1
                if limit and self.batches >= limit:
                    self.halt.set()


def _walk_segment(run, segment) -> None:
    """One chain: find where it starts if it has none, then walk to its boundary."""
    args = run.args
    reader = _reader_for(run.row, run.archive_set, run.tokens, run.limiter)
    concat = ConcatFile(run.archive_set, reader, window_min=args.window_min,
                        window_max=args.window_max)
    spent = {"requests": 0, "bytes": 0}

    def delta():
        """What this reader has cost *since the last report*. The store adds what it is
        given, so handing it the running total every time would compound it.
        """
        return (reader.requests - spent["requests"],
                reader.bytes_fetched - spent["bytes"])

    def charged():
        spent["requests"], spent["bytes"] = reader.requests, reader.bytes_fetched

    def commit(members, next_offset):
        requests, fetched = delta()
        run.commit(segment["id"], members, next_offset, requests, fetched)
        charged()          # only once the write went through

    # Every exit below charges what the reader spent, so the try covers the scan too:
    # this is the last place that still knows what it cost.
    try:
        start = segment["first_header"]
        if start is None:
            start = find_chain_start(concat, segment["scan_from"])
            if start is None:
                # Not a failure in itself -- the join resets such a segment to its
                # predecessor's exit like any other disagreement.
                requests, fetched = delta()
                run.store.fail_segment(
                    segment["id"], "no header between here and the end of the archive",
                    requests, fetched)
                return
            run.store.set_segment_start(segment["id"], start)
        cursor = segment["cursor_offset"]
        if cursor is None:
            cursor = start
        result = walk(concat, cursor, commit, batch_size=args.batch,
                      stop_at=segment["stop_at"], should_stop=run.stopping)
    except _Stopped:
        # The refused batch was read before it was refused.
        requests, fetched = delta()
        run.store.release_segment(segment["id"], requests, fetched)
        return
    except UnsupportedArchive as exc:
        requests, fetched = delta()
        if segment["joined"]:
            # A chain whose start is confirmed walks the archive's own headers, so
            # this is the archive's format, which every chain would meet: stop the
            # pool and let `_run_index` mark the archive for good.
            with run.lock:
                if run.unsupported is None:
                    run.unsupported = str(exc)
            run.halt.set()
            log.error("segment %d: %s", segment["idx"], exc)
        else:
            # A chain found by a cold scan may be walking a tarball stored inside the
            # archive -- run folders hold them, and Python writes pax by default. The
            # join settles it: it resets this chain if it is not on the archive's own,
            # and refuses the archive if it is.
            log.warning("segment %d: %s -- not yet on the confirmed chain",
                        segment["idx"], exc)
        run.store.fail_segment(segment["id"], f"{_PAX_SIGHTING}{exc}", requests, fetched)
        return
    except Exception as exc:
        # One bad part must not kill the run: record why it died and what it cost, and
        # let the worker take the next segment.
        requests, fetched = delta()
        log.warning("segment %d failed: %s: %s", segment["idx"], type(exc).__name__, exc)
        run.store.fail_segment(segment["id"], f"{type(exc).__name__}: {exc}",
                               requests, fetched)
        return
    if result.state == "stopped":
        # Back to pending with its committed cursor; the next run picks it up there.
        requests, fetched = delta()
        run.store.release_segment(segment["id"], requests, fetched)
        return
    # The cost that lands after the last batch -- the terminator probe, or a scan that
    # found nothing -- would otherwise never be counted.
    requests, fetched = delta()
    run.store.finish_segment(segment["id"], result, requests, fetched)


def _worker(run) -> None:
    """Claim segments until there are none left, or until the run is stopping."""
    owner = f"{os.getpid()}:{threading.get_ident()}"
    try:
        while not run.stopping():
            segment = run.store.claim_segment(run.archive_id, owner)
            if segment is None:
                return
            if run.stopping():
                # Claimed a moment before the stop. Hand it straight back rather than
                # begin a scan, which has no bound of its own -- Task 1 measured seeds
                # that found no header within 64 MiB.
                run.store.release_segment(segment["id"])
                return
            _walk_segment(run, segment)
    except Exception:
        # Not a bad part -- `_walk_segment` records those itself and carries on. This is
        # the pool's own machinery failing, so the run must not report success.
        run.failed = True
        log.exception("worker stopping after an unexpected failure")
    finally:
        run.store.close()          # this thread's own connection


_OUTCOME, _REPAIRED, _BLOCKED = "outcome", "repaired", "blocked"
#: How a segment's error records a pax sighting, so the join can tell it from a failure.
_PAX_SIGHTING = "UnsupportedArchive: "

_ROW_FIELDS = ("hdr_offset", "data_offset", "size", "type", "mode", "mtime", "uname",
               "gname", "dir", "name", "linkname")


def _row_of(item):
    """A stored row or a freshly read Member, as one comparable tuple."""
    if item is None:
        return None
    if isinstance(item, sqlite3.Row):
        return tuple(item[field] for field in _ROW_FIELDS)
    return tuple(getattr(item, field) for field in _ROW_FIELDS)


def _audit_repeated_paths(run) -> list[int]:
    """Read again every row whose path the index holds more than once, and walk again
    each segment holding a row that disagrees. Returns those segments' indexes.

    A path can repeat legitimately (`tar -r`), but a bad read that is valid tar from
    elsewhere in the archive usually makes one: it records another member's header a
    second time, at an offset that was never that member's. When the member it replaced
    pads to the same size, the chain rejoins straight after, and nothing else notices --
    the walk ends `complete` with one member twice and one never. A fresh read of each
    such row says which is which, and a row that disagrees is read once more, asked
    differently, before a segment is walked again: one bad read by the audit must not
    cost a 300 GiB re-walk.
    An archive with no repeated path pays nothing.

    What it cannot see: a copied header whose path is recorded nowhere else -- a member
    of a tarball stored inside the archive, or a long-named member's header read
    without its long-name block.
    """
    store, archive_id = run.store, run.archive_id
    rows = store.repeated_paths(archive_id, AUDIT_LIMIT + 1)
    if not rows:
        return []
    if len(rows) > AUDIT_LIMIT:
        rows = rows[:AUDIT_LIMIT]
        detail = f"read again only the first {AUDIT_LIMIT} rows under repeated paths"
        log.warning("audit: %s", detail)
        store.log_event(archive_id, "audit_partial", detail)
    # Its own reader and window: every byte compared comes from a request of its own,
    # in a shape of its own -- not the walk's.
    reader = _reader_for(run.row, run.archive_set, run.tokens, run.limiter)
    concat = ConcatFile(run.archive_set, reader, window_min=run.args.window_min,
                        window_max=run.args.window_max)
    def disagrees(row):
        return _row_of(read_member(concat, row["hdr_offset"])) != _row_of(row)

    def confirmed(row):
        concat.drop_cache()                 # a second request, asked differently
        return disagrees(row)

    try:
        wrong = [row["hdr_offset"] for row in rows if disagrees(row) and confirmed(row)]
    finally:
        store.charge(archive_id, reader.requests, reader.bytes_fetched)
    store.log_event(archive_id, "audit", f"read {len(rows)} rows under repeated paths "
                    f"again; {len(wrong)} disagreed with the index")
    redo = []
    for segment in store.segments(archive_id):
        end = segment["stop_at"]
        if any(segment["scan_from"] <= o and (end is None or o < end) for o in wrong):
            store.reset_segment(segment["id"], segment["first_header"])
            store.log_event(archive_id, "audit_repair",
                            f"segment {segment['idx']} walked again: a row in it "
                            f"disagreed with a fresh read of its header")
            redo.append(segment["idx"])
    return redo


def _join(store, archive_id):
    """Confirm each chain's start against its predecessor's exit; repair what disagrees.

    Walk from segment 0, which is joined by construction: a tar begins at offset 0, so
    it needs no scan and no confirmation. After that a segment is believed only once the
    chain before it walks into *exactly* the offset it started from. Equality is
    stronger than the spec's "an offset that chain visited", and deliberately so: a tar
    stored inside the tar hands a cold scan perfectly valid headers, and equality is the
    only thing that tells the two apart. At worst it re-walks a chain that had resynced
    on its own; it never accepts a false one.

    Returns ("repaired", idx) after the first repair -- that segment is pending again,
    so the caller runs the pool and joins again -- or ("outcome", WalkResult) once the
    chain reaches a verdict, or ("blocked", segment) while some chain has none yet.

    Run with no workers running, so nothing else is writing these rows.
    """
    segments = store.segments(archive_id)
    index = 0
    while True:
        segment = segments[index]
        if segment["state"] == "crossed":
            # A chain only crosses at its own stop_at, and the last segment has none,
            # so a crossed segment always has a successor.
            following = segments[index + 1]
            if (following["state"] != "beyond"
                    and following["first_header"] == segment["exit_offset"]):
                if not following["joined"]:
                    store.mark_joined(following["id"])
                index += 1
                continue
            # Disagreement, or a chain that found no header at all -- or the chain has
            # crossed into a segment retired past an ending that no longer stands (after
            # `--recheck`), so everything retired after it is back in play. reset_segment
            # drops that segment's rows and re-arms it from the offset handed down.
            if following["state"] == "beyond":
                store.rearm_beyond(archive_id, following["idx"] + 1)
            store.reset_segment(following["id"], segment["exit_offset"])
            store.log_event(archive_id, "repair",
                            f"segment {following['idx']} restarted at "
                            f"{segment['exit_offset']}")
            return _REPAIRED, following["idx"]
        if segment["state"] in ("complete", "truncated", "corrupt"):
            # The outcome is the *ending* segment's, not the last segment's: a tar's
            # terminator is usually not in its last part. Everything after it was never
            # on the chain, so its rows go.
            for later in segments[index + 1:]:
                if later["state"] != "beyond":
                    store.retire_segment(later["id"])
            # `members` is unused: `finish` recounts the rows it actually has.
            return _OUTCOME, WalkResult(segment["state"], segment["exit_offset"], 0,
                                        segment["detail"] or "")
        return _BLOCKED, segment        # pending | walking | error: no verdict yet


def cmd_archive_index(args) -> int:
    setup_logging(args.verbose)
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives; "
              f"run `archive register` first", file=sys.stderr)
        return 2
    row = store.get(args.archive)
    if row is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2

    # Re-running `index` is how a walk resumes, so resuming a finished one reads
    # nothing and writes nothing.
    recheck = ("\n       if that is not what you expect, `archive index --recheck` walks "
               "the chain there again")
    if args.recheck and row["state"] not in ("corrupt", "truncated", "unsupported"):
        print(f"error: {row['name']} is {row['state']}; --recheck asks again a verdict a "
              f"walk reached -- corrupt, truncated or unsupported"
              + ("" if row["state"] in ("complete", "stale")
                 else ", and `index` alone resumes this one"), file=sys.stderr)
        return 1
    if row["state"] == "complete":
        print(f"{row['name']}: complete ({row['n_members']:,} members)")
        return 0
    if row["state"] == "stale":
        print(f"error: {row['name']} is stale ({row['detail'] or ''}); re-register it "
              f"to index the parts as they are now", file=sys.stderr)
        return 1
    if row["state"] in ("truncated", "corrupt") and not args.recheck:
        print(f"error: {row['name']} is {row['state']}: {row['detail'] or ''}{recheck}",
              file=sys.stderr)
        return 1
    if row["state"] == "unsupported" and not args.recheck:
        print(f"error: {row['name']} is a pax-format archive, which dbaudit does not "
              f"index: {row['detail'] or ''}{recheck}", file=sys.stderr)
        return 1

    try:
        lock = InstanceLock(args.lock or f"{args.db}.lock")
        lock.__enter__()
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    try:
        return _run_index(args, store, row)
    finally:
        lock.__exit__(None, None, None)


#: The first bytes of the compressed files people name `.tar`. Such a file has no
#: headers to walk; calling it `corrupt` would send its owner looking for damage.
COMPRESSION = (("gzip", b"\x1f\x8b"), ("bzip2", b"BZh"), ("xz", b"\xfd7zXZ\x00"),
               ("zstd", b"\x28\xb5\x2f\xfd"))


def _compression(row, archive_set, tokens, limiter) -> str | None:
    """The compression format the archive's first bytes announce, or None."""
    first = archive_set.parts[0]
    head = _reader_for(row, archive_set, tokens, limiter).read_range(0, 0, min(6, first.size))
    return next((name for name, magic in COMPRESSION if head.startswith(magic)), None)


@dataclass
class _Walked:
    """How one archive's walk ended: what `archive index` reports, and `archive run`
    counts. ``started`` is False for a walk refused before it began, which has no
    figures worth printing; ``bug`` marks the machinery failing, not the archive."""
    state: str
    code: int
    errors: list = field(default_factory=list)
    started: bool = True
    stopped: bool = False
    bug: bool = False


def _walk_archive(args, store, row, tokens, limiter, stop, check_before=True,
                  recheck=False) -> _Walked:
    """Walk one registered archive to a verdict, or until ``stop`` is set.

    The pipeline `archive index` and `archive run` share: the parts checked against the
    folder, the chains walked and joined, the audit, the parts checked again, and the
    outcome recorded. The caller owns the lock, the signals and the printing.
    ``check_before`` is False when the caller registered the parts from a listing a
    moment ago, which is that same check.
    """
    archive_id = row["id"]
    archive_set = _stored_archive_set(store, row)
    dropbox = row["source"] == "dropbox"

    # A before/after fingerprint of every part, as the spec requires. Local archives
    # are not fingerprinted: there is no cheap hash to ask for, and re-hashing 5 TiB
    # would cost far more than the walk it is meant to protect.
    if dropbox and check_before:
        try:
            fingerprinted = _fingerprint(store, row, args.remote, tokens, limiter, stop)
        except AuthError as exc:
            return _Walked(row["state"], 2, [f"error: {exc}"], started=False)
        except Exception as exc:
            return _Walked(row["state"], 1,
                           [f"error: could not list {row['folder']}: "
                            f"{type(exc).__name__}: {exc}"], started=False)
        if fingerprinted is None:
            return _Walked(row["state"], 0, started=False, stopped=True)
        missing, changed = fingerprinted
        if missing:
            detail = (f"{len(missing)} part(s) missing from {row['folder']}: "
                      f"{', '.join(missing[:5])}")
            store.fail(archive_id, detail)
            return _Walked("error", 1, [f"error: {detail}\n       re-register the archive "
                                        f"with its new --folder"], started=False)
        if changed:
            detail = (f"{len(changed)} part(s) changed since registration: "
                      f"{', '.join(changed[:5])}")
            store.mark_stale(archive_id, detail)
            return _Walked("stale", 1, [f"error: {detail}\n       the stored offsets "
                                        f"describe bytes that are no longer there; "
                                        f"re-register to walk it again"], started=False)

    if recheck:
        # Only now, under the lock and with every part unchanged: rows are dropped to be
        # walked again, and an archive whose parts changed keeps its rows as evidence.
        rechecked = store.recheck(archive_id)
        if rechecked is None:
            return _Walked("stale", 1, [f"error: {row['name']} is stale; re-register it"],
                           started=False)
        again, ending = rechecked
        further = (" -- a segment further back than the last --recheck, which ended the "
                   "same way" if again["idx"] < ending["idx"] - 1 else "")
        print(f"{row['name']}: walking the chain again from offset {again['first_header']:,} "
              f"(segment {again['idx']}; the walk had ended in segment {ending['idx']})"
              f"{further}")

    workers = max(args.workers, 1)
    store.start_walk(archive_id)
    run = _Run(args, store, row, archive_set, tokens, limiter, stop)
    outcome = blocked = audit_failed = None
    exhausted = False
    log.info("walking %s (%d parts, %s) with %d chains",
             row["name"], row["n_parts"], human_bytes(row["total_size"]), workers)
    # Each repair advances the confirmed frontier by at least one segment, so this is
    # bounded by the segment count; in the worst case it degrades to a sequential walk.
    # The bound is asserted rather than assumed.
    for _round in range(len(store.segments(archive_id)) + 2):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(_worker, run) for _ in range(workers)]:
                future.result()
        if run.unsupported:
            break                     # no verdict to join towards: see below
        kind, payload = _join(store, archive_id)
        if (kind == _BLOCKED and payload["state"] == "error"
                and (payload["error"] or "").startswith(_PAX_SIGHTING)):
            # The join only stops at a segment on the confirmed chain, so this pax
            # sighting is the archive's own.
            run.unsupported = payload["error"][len(_PAX_SIGHTING):]
            break
        if kind == _OUTCOME and payload.state == "complete":
            try:
                redo = _audit_repeated_paths(run)
            except Exception as exc:
                audit_failed = (f"could not read again the rows under repeated paths: "
                                f"{type(exc).__name__}: {exc}")
                break
            if redo:
                # No stop check needed: a stopping pool claims nothing, so the next join
                # finds them pending and the run ends there, resumable.
                log.warning("segment(s) %s held a row that disagreed with a fresh read; "
                            "walking them again", ", ".join(map(str, redo)))
                continue
        if kind == _OUTCOME:
            outcome = payload
            break
        if kind == _BLOCKED:
            blocked = payload
            break
        log.info("segment %d did not meet the chain before it; re-walking it", payload)
        if run.stopping():
            break
    else:
        # Unreachable by construction; if it ever happens it is a bug, not a stop.
        exhausted = True
        log.error("giving up after too many join rounds; see `archive status`")

    stale = None
    if dropbox:
        try:
            # Listed even when stopping: a verdict is recorded only after this check.
            missing, changed = _fingerprint(store, row, args.remote, tokens, limiter)
        except Exception as exc:
            log.warning("could not re-list %s to fingerprint it: %s: %s",
                        row["folder"], type(exc).__name__, exc)
        else:
            if missing or changed:
                stale = (f"parts changed while the walk ran: "
                         f"{', '.join((missing + changed)[:5])}")
                # Before the outcome is written: `finish` defends this flag and its
                # reason, so the walk's own bookkeeping cannot erase it.
                store.mark_stale(archive_id, stale)

    compressed = None
    if run.unsupported:
        store.mark_unsupported(archive_id, run.unsupported)
    elif audit_failed is not None:
        # Every chain is walked, so the retry this invites goes straight to the audit.
        store.fail(archive_id, audit_failed)
    elif outcome is not None:
        if outcome.state in ("corrupt", "truncated") and outcome.end_offset == 0:
            # No header at the very start: perhaps no tar at all.
            compressed = _compression(row, archive_set, tokens, limiter)
        if compressed:
            store.mark_unsupported(archive_id, f"{compressed}-compressed: not a tar, so it "
                                               f"cannot be walked header by header")
        else:
            store.finish(archive_id, outcome)
            store.build_indexes(archive_id)    # only once a walk is over, never during
    elif blocked is not None and blocked["state"] == "error":
        store.fail(archive_id, f"segment {blocked['idx']}: {blocked['error'] or 'failed'}")

    state = store.connect().execute(
        "SELECT state FROM archives WHERE id=?", (archive_id,)).fetchone()[0]
    if stale is not None:
        return _Walked(state, 1, [f"error: {stale}; the members found are kept, but "
                                  f"re-register before trusting them"])
    if run.unsupported:
        return _Walked(state, 1, [f"error: {row['name']} is a pax-format archive, which "
                                  f"dbaudit does not index. {run.unsupported}"])
    if compressed:
        return _Walked(state, 1, [f"error: {row['name']} is {compressed}-compressed, not "
                                  f"a tar: it cannot be walked header by header"])
    if audit_failed is not None:
        return _Walked(state, 1, [f"error: the walk finished, but the audit {audit_failed}; "
                                  f"run `index` again to retry it"])
    if run.failed:
        return _Walked(state, 1, ["error: a chain stopped on an unexpected failure; see "
                                  "the log and `archive status`"], bug=True)
    if outcome is not None:
        if outcome.state == "complete":
            return _Walked(state, 0)
        return _Walked(state, 1, [f"error: {row['name']} is {outcome.state}: "
                                  f"{outcome.detail}"])
    if blocked is not None and blocked["state"] == "error":
        return _Walked(state, 1, [f"error: segment {blocked['idx']}: {blocked['error']}"])
    if exhausted:
        return _Walked(state, 1, [f"error: {row['name']} did not settle; see "
                                  f"`archive status`"], bug=True)
    return _Walked(state, 0, stopped=True)


def _run_index(args, store, row) -> int:
    dropbox = row["source"] == "dropbox"
    workers = max(args.workers, 1)
    tokens = limiter = None
    if dropbox:
        # One provider and one limiter for the whole command: Dropbox limits per
        # account, and the provider caches its token behind a lock.
        tokens = TokenProvider(remote=args.remote)
        limiter = AdaptiveLimiter(rps=args.rps, max_concurrency=workers)

    stop = threading.Event()
    hard = {"count": 0}

    def handle(signum, frame):
        hard["count"] += 1
        if hard["count"] == 1:
            # Flag first, then log: the log line is then proof the walk is already
            # winding down rather than a promise that it is about to.
            stop.set()
            log.warning("signal %d: committing the batch in flight, then exiting", signum)
        else:
            log.warning("second signal: exiting immediately")
            os._exit(1)

    # Unlike `run`, these are put back: the tests call main() in-process and must not
    # be left with pytest's SIGINT handling replaced.
    previous = {}
    started = time.time()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle)
        walked = _walk_archive(args, store, row, tokens, limiter, stop,
                               recheck=args.recheck)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    if walked.started:
        archive_id = row["id"]
        stats = store.stats(archive_id)
        rate_limited = limiter.rate_limit_events if limiter else 0
        if rate_limited:
            store.log_event(archive_id, "rate_limited", str(rate_limited))
        ratio = ((stats["bytes_fetched"] / stats["member_bytes"])
                 if stats["member_bytes"] else 0.0)
        print(f"{row['name']}: {stats['state']}")
        print(f"  members       : {stats['n_members']:,} "
              f"({human_bytes(stats['member_bytes'])} of member data)")
        print(f"  requests      : {stats['requests']:,}")
        print(f"  bytes fetched : {human_bytes(stats['bytes_fetched'])}")
        print(f"  ratio         : {ratio:.3f} bytes fetched per member byte")
        print(f"  429s          : {rate_limited}")
        print(f"  elapsed       : {human_duration(time.time() - started)}")
    for line in walked.errors:
        print(line, file=sys.stderr)
    if walked.stopped:
        log.info("stopped before the chain reached a verdict; re-run to resume")
    return walked.code


def _print_archive_status(store, row) -> None:
    stats = store.stats(row["id"])
    total = stats["total_size"] or 1
    ratio = (stats["bytes_fetched"] / stats["member_bytes"]) if stats["member_bytes"] else 0.0
    print(f"{row['name']}  [{stats['state']}]")
    print(f"  parts         : {row['n_parts']} ({human_bytes(stats['total_size'])})")
    print(f"  members       : {stats['n_members']:,} "
          f"({human_bytes(stats['member_bytes'])} of member data)")
    print(f"  covered       : {human_bytes(stats['covered'])} "
          f"({100.0 * stats['covered'] / total:.1f}%)")
    print(f"  confirmed to  : {stats['confirmed']:,}")
    print(f"  requests      : {stats['requests']:,}")
    print(f"  bytes fetched : {human_bytes(stats['bytes_fetched'])}")
    print(f"  ratio         : {ratio:.3f} bytes fetched per member byte")
    if stats["state"] == "walking":
        # On a multi-hour walk this is the only view of progress there is.
        elapsed = (stats["updated_at"] or 0) - (stats["run_started_at"] or 0)
        gained = stats["covered"] - (stats["run_start_covered"] or 0)
        if elapsed > 0 and gained > 0:
            rate = gained / elapsed
            print(f"  rate          : {human_bytes(rate)}/s this run")
            print(f"  eta           : "
                  f"{human_duration((stats['total_size'] - stats['covered']) / rate)}")
    print("  segments:")
    for segment in store.segments(row["id"]):
        span_end = segment["stop_at"]
        if span_end is None:
            span_end = stats["total_size"]
        span = max(span_end - segment["scan_from"], 1)
        cursor = segment["cursor_offset"]
        if cursor is None:
            progress = "-"
        else:
            done = min(max(cursor - segment["scan_from"], 0), span)
            progress = f"{100.0 * done / span:.0f}%"
        first = segment["first_header"]
        note = (segment["error"] or segment["detail"] or "")[:60]
        print((f"    [{segment['idx']:2d}] {segment['state']:<9}"
               f" {'joined' if segment['joined'] else '      '}"
               f"  first={'-' if first is None else first:>12}"
               f"  cursor {progress:>5}"
               f"  members={segment['members']:<7}{note}").rstrip())


#: States that end a walk. An archive in one is not walked again by `archive run`.
VERDICTS = frozenset({"complete", "truncated", "corrupt", "unsupported"})


def _run_jobs(store, listing, workers):
    """What `archive run` has to walk, largest first, and a count of everything else.

    Every valid archive the listing names that has no verdict yet, and every archive the
    database holds without one -- imported mid-walk, or registered by hand -- once per
    folder and name. Returns (jobs, counts, sizes), the last two keyed by kind.
    """
    rows = store.connect().execute(
        "SELECT a.*, (SELECT COUNT(*) FROM segments s WHERE s.archive_id = a.id"
        "             AND s.state IN ('pending', 'walking', 'error')) AS unfinished "
        "FROM archives a").fetchall()
    by_hash = {row["set_hash"]: row for row in rows}
    jobs, counts, sizes, places, ids = [], Counter(), Counter(), set(), set()

    def add(kind, folder, name, size, chains, archive_id):
        # An archive that moved is listed where it is now, and is in the database where
        # it was registered: one walk, from where it is now, which comes first.
        if (folder, name) in places or archive_id in ids:
            return
        places.add((folder, name))
        if archive_id is not None:
            ids.add(archive_id)
        jobs.append(Job(key=f"{folder}/{name}", label=f"{folder}/{name}", size=size,
                        chains=max(1, min(chains, workers)), folder=folder, name=name,
                        archive_id=archive_id))
        counts[kind] += 1
        sizes[kind] += size

    for candidate in listing.candidates:
        if candidate.status != "ok":
            counts[candidate.status] += 1
            sizes[candidate.status] += candidate.size
            continue
        row = by_hash.get(candidate.set_hash)
        if row is not None and row["state"] in VERDICTS:
            places.add((candidate.folder, candidate.name))
            counts["done"] += 1
            sizes["done"] += candidate.size
        elif row is None:
            add("new", candidate.folder, candidate.name, candidate.size, candidate.n_parts,
                None)
        else:
            add("resuming", candidate.folder, candidate.name, candidate.size,
                row["unfinished"] or candidate.n_parts, row["id"])
    for row in rows:
        if row["state"] not in VERDICTS and row["source"] == "dropbox":
            add("resuming", row["folder"], row["name"], row["total_size"],
                row["unfinished"] or row["n_parts"], row["id"])
    counts["compressed names"], sizes["compressed names"] = listing.compressed
    counts["other names"], sizes["other names"] = listing.other
    return sorted(jobs, key=lambda job: -job.size), counts, sizes


def _walk_job(args, store, job, tokens, limiter, stop) -> Done:
    """One archive at its turn. One listing of its folder registers what is there now:
    the same row, with its walk so far, if the content is unchanged -- wherever it was
    registered before -- or a new one if not. Then the walk."""
    try:
        _, lister = build_lister(args.remote, tokens)
        try:
            entries = _listed(lister, limiter, job.folder, job.name, stop)
        except PathNotFound as exc:
            store.log_event(job.archive_id, "gone", f"{job.label}: {exc}")
            return Done("gone")
        except ApiError as exc:
            store.log_event(job.archive_id, "fail", f"{job.label}: could not list its "
                                                    f"folder: {type(exc).__name__}: {exc}")
            return Done("error")
        if entries is None:
            return Done("stopped")
        if not entries:
            store.log_event(job.archive_id, "gone", f"{job.label}: no longer in its folder")
            return Done("gone")
        try:
            archive_set = ArchiveSet.from_entries(entries, job.name)
        except ArchiveSetError as exc:
            store.log_event(job.archive_id, "refused", f"{job.label}: {exc}")
            return Done("refused")
        job.archive_id = store.register(job.name, "tar", job.folder, "dropbox",
                                        archive_set.parts)
        store.seed_segments(job.archive_id, archive_set.parts)
        row = store.connect().execute(
            "SELECT * FROM archives WHERE id=?", (job.archive_id,)).fetchone()
        if row["state"] in VERDICTS:
            return Done("done")                 # this content was walked before
        walked = _walk_archive(args, store, row, tokens, limiter, stop, check_before=False)
        for line in walked.errors:
            log.warning("%s: %s", job.label, line.removeprefix("error: "))
        stats = store.stats(row["id"])
        return Done("stopped" if walked.stopped else walked.state,
                    members=stats["n_members"], requests=stats["requests"], bug=walked.bug)
    finally:
        store.close()                           # this thread's own connection


def _record_bug(store, job, exc) -> None:
    """Leave the archive an unexpected failure stopped in `error`, saying why."""
    detail = f"{type(exc).__name__}: {exc}"
    try:
        if job.archive_id is not None:
            store.fail(job.archive_id, detail)
        else:
            store.log_event(None, "fail", f"{job.label}: {detail}")
    finally:
        store.close()                           # this thread's own connection


def _print_run_plan(jobs, counts, sizes) -> None:
    total = sum(job.size for job in jobs)
    print(f"archive run: {len(jobs):,} archives to walk ({human_bytes(total)}): "
          f"{counts['new']:,} new, {counts['resuming']:,} resuming")
    for kind, label in (("done", "already walked"), ("copy", "copies"),
                        ("refused", "refused"), ("compressed names", "compressed names"),
                        ("other names", "other names")):
        print(f"  {label:17}: {counts[kind]:,} ({human_bytes(sizes[kind])})")
    if jobs:
        print("  first in line:")
        for job in jobs[:5]:
            print(f"    {human_bytes(job.size):>10}  {job.label}")


def cmd_archive_run(args) -> int:
    setup_logging(args.verbose)
    if not 1 <= args.streams <= args.max_streams:
        print(f"error: --streams {args.streams} must lie between 1 and --max-streams "
              f"{args.max_streams}", file=sys.stderr)
        return 2
    if not os.path.exists(args.audit_db):
        print(f"error: no audit database at {args.audit_db}", file=sys.stderr)
        return 2
    if _other_kind_of_database(args.db):
        print(f"error: {args.db} is not an archive database", file=sys.stderr)
        return 2
    try:
        listing = find_candidates(args.audit_db)
    except sqlite3.DatabaseError as exc:
        print(f"error: {args.audit_db} is not an audit database: {exc}", file=sys.stderr)
        return 2
    workers = max(args.workers, 1)
    if args.dry_run:
        if os.path.exists(args.db):
            store = ArchiveStore(args.db)
        else:                       # nothing is written: a new database starts empty
            store = ArchiveStore(":memory:")
            store.init_schema()
        _print_run_plan(*_run_jobs(store, listing, workers))
        return 0
    try:
        lock = InstanceLock(args.lock or f"{args.db}.lock")
        lock.__enter__()
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    try:
        # The work list is read under the lock, so no other walk changes it after.
        store = ArchiveStore(args.db)
        store.init_schema()
        jobs, counts, sizes = _run_jobs(store, listing, workers)
        _print_run_plan(jobs, counts, sizes)
        return _run_archives(args, store, jobs)
    finally:
        lock.__exit__(None, None, None)


def _run_limiter(args) -> AdaptiveLimiter:
    """One limiter for every walk and listing: Dropbox limits per account. The rate starts
    where the streams need it, a request costing about 1.5 s, so the stream count is what
    Dropbox's pushback adjusts."""
    return AdaptiveLimiter(rps=args.streams / 1.5, max_rps=args.max_streams / 1.5,
                           max_concurrency=args.max_streams, start_concurrency=args.streams)


def _run_archives(args, store, jobs) -> int:
    tokens = TokenProvider(remote=args.remote)       # one, so its token is cached once
    limiter = _run_limiter(args)
    stop = threading.Event()
    hard = {"count": 0}

    def handle(signum, frame):
        hard["count"] += 1
        if hard["count"] == 1:
            stop.set()
            log.warning("signal %d: committing every batch in flight, then exiting", signum)
        else:
            log.warning("second signal: exiting immediately")
            os._exit(1)

    previous = {}
    started = time.time()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle)
        report = Runner(lambda job: _walk_job(args, store, job, tokens, limiter, stop),
                        limiter, stop, limit=args.limit,
                        on_bug=lambda job, exc: _record_bug(store, job, exc)).run(jobs)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(f"archive run: {report.started:,} archives started, "
          f"{sum(report.outcomes.values()):,} finished, in "
          f"{human_duration(time.time() - started)}")
    for state, count in sorted(report.outcomes.items()):
        print(f"  {state:11}: {count:,}")
    if report.failed:
        print("error: stopped after repeated unexpected failures; see the log",
              file=sys.stderr)
        return 1
    return 0


def cmd_archive_import(args) -> int:
    source = _open_archive_store(args.source)
    if source is None:
        print(f"error: {args.source} holds no registered archives", file=sys.stderr)
        return 2
    source.close()
    if _other_kind_of_database(args.db):
        print(f"error: {args.db} is not an archive database", file=sys.stderr)
        return 2
    if os.path.exists(args.db) and os.path.samefile(args.db, args.source):
        print(f"error: --from names {args.db} itself", file=sys.stderr)
        return 2
    store = ArchiveStore(args.db)
    try:
        # The source's lock too: a walk still writing to it would be copied half-done.
        with InstanceLock(f"{args.db}.lock"), InstanceLock(f"{args.source}.lock"):
            store.init_schema()
            moved = store.import_from(args.source)
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"error: {exc}; nothing was imported", file=sys.stderr)
        return 2
    replaced = sum(old is not None for _, _, old in moved)
    print(f"imported {len(moved):,} archives from {args.source} into {args.db}, "
          f"{replaced:,} of them replacing one of the same content")
    return 0


def cmd_archive_forget(args) -> int:
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives", file=sys.stderr)
        return 2
    try:
        with InstanceLock(f"{args.db}.lock"):
            found = store.query("SELECT id, state, n_members FROM archives "
                                "WHERE name=? AND folder=?", (args.archive, args.folder))
            if not found:
                print(f"error: no archive named {args.archive!r} in {args.folder}",
                      file=sys.stderr)
                return 2
            for row in found:
                what = (f"archive {row['id']}: {args.folder}/{args.archive}, {row['state']}, "
                        f"{row['n_members']:,} members")
                store.forget(row["id"], what)
                print(f"forgot {what}")
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


def cmd_archive_status(args) -> int:
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives", file=sys.stderr)
        return 2
    if not args.archive:
        _print_archive_summary(store, args.db)
        return 0
    row = store.get(args.archive)
    if row is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2
    _print_archive_status(store, row)
    return 0


def _print_archive_summary(store, db) -> None:
    """Every archive in the database, by state, and the ones mid-walk: what `archive
    run` has done so far, readable from any shell. One archive's full status is
    `--archive`."""
    conn = store.connect()
    states = conn.execute(
        "SELECT state, COUNT(*), COALESCE(SUM(total_size), 0), COALESCE(SUM(n_members), 0) "
        "FROM archives GROUP BY state ORDER BY state").fetchall()
    if not states:
        print(f"{db}: no archives registered yet")
        return
    count = sum(s[1] for s in states)
    print(f"{db}: {count:,} archive{'' if count == 1 else 's'}, "
          f"{human_bytes(sum(s[2] for s in states))}")
    for state, count, size, members in states:
        extra = f", {members:,} members" if members else ""
        print(f"  {state:12}: {count:,} ({human_bytes(size)}{extra})")
    walking = conn.execute(
        "SELECT id, name, folder, total_size FROM archives WHERE state='walking' "
        "ORDER BY total_size DESC").fetchall()
    if walking:
        print("mid-walk:")
        for row in walking:
            covered = store.stats(row["id"])["covered"] / max(row["total_size"], 1)
            print(f"  {row['name']}  {human_bytes(row['total_size'])}  "
                  f"covered {100 * covered:.1f}%  ({row['folder']})")


def _print_index_completeness_warning(row) -> None:
    """The "this index may not be the whole archive" warning, shared by `report` and
    `export` rather than restated by each (T9-8; extended from `report` alone to
    `export` too in Task 9's fix round 1).

    `cat` (T8-4) refuses outright on a stale archive, because extracting one would
    write *wrong bytes* -- content that may no longer be what is actually on Dropbox
    -- into a file the operator goes on to trust. `report` and `export` never write
    archive content at all: they describe the index we genuinely have, which may
    simply be partial (a walk still in progress) or out of date (stale), and a
    warning is the honest response to that, not a refusal -- a partial or stale index
    is still real, useful information about what has been walked so far. `export`'s
    entire product is files meant to be handed to someone else, so it needs this
    caveat at least as much as a printed report does; printed before its "wrote ..."
    lines, the same way `report` prints it before a single figure (T9-8).
    """
    if row["state"] != "complete":
        # Task 11 runs `report`/`export` against a walk in progress, and a manifest
        # that is 12% walked must never read like a finished archive's.
        print(f"\n  WARNING: {row['name']} is not fully indexed (state: {row['state']}).")
        # Not "a lower bound": rows from chains the join has not yet confirmed are
        # counted too, and a cold scan can lock onto a tar stored inside the archive.
        print("  It may be missing members, and until every chain is confirmed it may also")
        print("  list members of a tar stored inside the archive: treat these figures as")
        print("  provisional.")
    if row["state"] == "unsupported":
        print("  It is a pax-format archive, which dbaudit does not index: the walk")
        print("  stopped at the first pax member.")
    if row["state"] == "stale":
        print(f"  It is also stale ({row['detail'] or 'parts changed since indexing'}):")
        print("  the parts changed underneath this index, so it may describe bytes")
        print("  that are no longer there.")


def cmd_archive_report(args) -> int:
    """Print the rollups `archive_report.summary` computes over one archive's
    manifest, laid out the way `cmd_report` lays out the audit's own numbers.

    Read-only, and offline (T9-10): everything comes out of the local database, so
    there is no `--remote` here and nothing above this function touches Dropbox.
    """
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives; "
              f"run `archive register` first", file=sys.stderr)
        return 2
    row = store.get(args.archive)
    if row is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2

    print(f"# {row['name']}  [{row['state']}]")
    _print_index_completeness_warning(row)

    result = archive_report.summary(store, row["id"], top=args.top)
    if result["n_members"] == 0:
        # T9-9: say so plainly and stop here, rather than printing a page of empty
        # sections or dividing by a member count of zero.
        print("\n  not indexed: 0 members recorded. Run `archive index` first.")
        return 0

    print(f"\n{result['n_members']:,} members, {human_bytes(result['member_bytes'])}, "
          f"{result['n_parts']} part(s)\n")

    print("## By top-level directory")
    for key, agg in result["by_top_dir"][:args.top]:
        print(f"  {human_bytes(agg['bytes']):>12}  {agg['count']:>10,} files  {key}")

    print("\n## By depth")
    for depth in (1, 2, 3):
        print(f"  depth {depth}:")
        for key, agg in result["by_depth"][depth][:args.top]:
            print(f"    {human_bytes(agg['bytes']):>12}  {agg['count']:>10,} files  {key}")

    print("\n## By extension")
    for key, agg in result["by_extension"][:args.top]:
        label = "(no extension)" if key == "(none)" else f".{key}"
        print(f"  {human_bytes(agg['bytes']):>12}  {agg['count']:>10,} files  {label}")

    print(f"\n## Largest {len(result['largest'])} member(s)")
    for member in result["largest"]:
        path = f"{member['dir']}/{member['name']}" if member["dir"] else member["name"]
        print(f"  {human_bytes(member['size']):>12}  {path}")

    if result["mtime_span"] is not None:
        earliest, latest = result["mtime_span"]
        print(f"\n## mtime span\n  {earliest} .. {latest}")

    print("\n## Owners")
    for (uname, gname), agg in result["owners"]:
        print(f"  {agg['count']:>10,} members  {uname or '(none)'}/{gname or '(none)'}")

    print("\n## Types")
    for type_code, agg in result["types"]:
        print(f"  {agg['count']:>10,} members  type {type_code!r}")

    return 0


def cmd_archive_export(args) -> int:
    """Write one archive's manifest and rollups as CSV, and print each path written --
    the same convention `cmd_export` uses for the audit's own CSV export.
    """
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives; "
              f"run `archive register` first", file=sys.stderr)
        return 2
    row = store.get(args.archive)
    if row is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2
    _print_index_completeness_warning(row)
    for path in archive_report.write_csv(store, row["id"], args.out):
        print(f"wrote {path}")
    return 0


def _print_candidates(matches) -> None:
    """The offset/size/mtime of every member a path matched, so `--offset` can name
    one of them. Shared by the two `cat` outcomes that need it: ambiguous (no
    `--offset` given) and a given `--offset` that names none of them.
    """
    for match in matches:
        print(f"  --offset {match['hdr_offset']}  size={match['size']}  "
              f"mtime={match['mtime']}", file=sys.stderr)


class _CatError(Exception):
    """A member could not be read in a form its surroundings vouch for."""


def _ambiguous(shared: bytes) -> bool:
    """Whether two reads agreeing on ``shared`` vouches for neither of them."""
    return (len(shared) < AMBIGUOUS_BELOW
            or len(zlib.compress(shared, 1)) * AMBIGUOUS_RATIO < len(shared))


def _plan_reads(archive_set, hdr_offset, data_offset, end, stop):
    """The reads that cover [hdr_offset, stop), as (offset, length) pairs.

    Each is at most CAT_CHUNK, and consecutive ones share bytes, so every read is checked
    at both ends by another request. That holds only if no read crosses a part boundary:
    a read that does is two requests, and the two ends that meet at the boundary share
    nothing. So reads stop at each boundary, and a short read straddling it shares bytes
    with the reads either side. The exceptions are the two anchors, which are checked by
    what they say rather than by an overlap, and so are read whole wherever a boundary
    falls -- parts need not be a multiple of 512 bytes (`split -b 1GB`): the member's own
    header sequence [hdr_offset, data_offset), and what follows it [end, stop).
    """
    def part_end(offset):
        idx, _ = archive_set.locate(offset)
        part = archive_set.parts[idx]
        return part.offset + part.size

    reads, at, first = [], hdr_offset, True
    while True:
        limit = part_end(max(data_offset - 1, at) if first else at)
        end_of_read = min(at + CAT_CHUNK, stop, limit)
        if first:
            end_of_read, first = max(end_of_read, min(data_offset, stop)), False
        if end < end_of_read < stop:
            # What follows the member is read whole, whether a part boundary or the read
            # size would cut it: it is several blocks when the next member has a long
            # name, more than two reads share.
            end_of_read = stop
        reads.append((at, end_of_read - at))
        if end_of_read >= stop:
            return reads
        if end_of_read == limit:
            boundary = end_of_read
            left = min(CAT_OVERLAP, boundary - at)
            right = min(CAT_OVERLAP, part_end(boundary) - boundary, stop - boundary)
            reads.append((boundary - left, left + right))
            at = boundary
        else:
            at = end_of_read - CAT_OVERLAP


def _headers_mismatch(block, member):
    """Why ``block`` -- the bytes from a member's first header to its data -- is not the
    member its index row describes, or None if it is. Every field the row holds is
    compared: a read of another member's headers is caught even at the same size, and so
    is a part rewritten under the index."""
    try:
        with tarfile.open(fileobj=io.BytesIO(block), mode="r:") as sequence:
            info = sequence.firstmember
    except tarfile.ReadError as exc:
        return f"no header there ({exc})"
    if info is None:
        return "no header there"
    if info.pax_headers:
        return "a pax extended header, which dbaudit does not extract"
    if info.offset_data != len(block):
        return f"its data begins {info.offset_data - len(block):+,} bytes from the index's"
    found = Member.from_tarinfo(info)
    for field in ("dir", "name", "size", "type", "mode", "mtime", "uname", "gname",
                  "linkname"):
        if getattr(found, field) != member[field]:
            return f"{field} {getattr(found, field)!r}, not {member[field]!r}"
    return None


def _verified_member_bytes(concat, member, total, follows):
    """Yield one member's bytes, none of them before a second request vouches for them.

    A tar keeps no checksum of member data, and Dropbox has answered range requests with
    some other range's bytes (twice in ~42,000 reads on 2026-09-21). So a byte is written
    only once two independent requests agree on it, or an anchor confirms it:

    - the first read starts on the member's first header, and that whole header sequence
      must parse to exactly the member its index row describes -- every field, and its
      data where the row says it begins;
    - the last read ends on what the index says follows the member (``follows``): the
      next member's whole header sequence, which must parse to exactly that member's row
      -- not merely to some header, which a read crossed with an equal-size twin's range
      supplies too; or the terminator's zeros; or, where the index does not know (a walk
      stopped early), a header or zeros, vouching for nothing;
    - consecutive reads share bytes and must agree on them, and no read crosses a part
      boundary (`_plan_reads`), so every request is checked at both ends;
    - a read is written only once the read after it agrees with its tail, so a read wrong
      in its back half is read again rather than half-written;
    - bytes too regular to tell one region from another -- zeros, fill, a short repeated
      pattern (`_ambiguous`) -- or too few, vouch for nothing. A read vouched for at one
      end only by such bytes, or last before the terminator's zeros, the archive's end or
      an unknown successor, is taken once a second fetch agrees with it byte for byte. So
      zero-heavy data costs about twice the requests.

    A failed check is met by reading again, fresh, up to REREAD_LIMIT times; past that
    this raises _CatError, having written nothing it could not vouch for. What it cannot
    catch: a read wrong only between two ends that were each right, or a wrong body the
    server repeats on consecutive fetches of the same range where only regular bytes
    surround it -- twice running from the start, or three times after a right one.
    Memory: the read held and the next, plus up to two more while a disagreement is
    settled -- at most five reads (80 MiB), usually two.
    """
    hdr, data_offset, size = member["hdr_offset"], member["data_offset"], member["size"]
    if data_offset + size > total:
        raise _CatError(f"the archive ends {data_offset + size - total:,} byte(s) "
                        f"before this member does")
    end = data_offset + -(-size // BLOCK) * BLOCK      # where the next header sits
    known = follows not in ("zeros", None)             # the next member's index row
    stop = min(follows["data_offset"] if known else end + BLOCK, total)
    plan = _plan_reads(concat.archive, hdr, data_offset, end, stop)
    last = len(plan) - 1
    # Part boundaries inside the two anchors, where a read spans them. The bytes of the
    # anchor on the far side of one are all that vouches for that request's end.
    bounds = [part.offset for part in concat.archive.parts[1:]]
    head_cuts = [b for b in bounds if hdr < b < data_offset]
    tail_cuts = [b for b in bounds if end < b < stop]

    def fetch(i):
        at, length = plan[i]
        concat.drop_cache()                            # a re-read must be a new request
        concat.seek(at)
        buf = concat.read(length)
        if len(buf) != length:
            raise _CatError(f"read {len(buf):,} of {length:,} byte(s) at {at:,}")
        return buf

    def shared(i, buf, side):
        """The bytes read i has in common with read i+1 ("tail") or read i-1 ("head")."""
        j = i if side == "tail" else i - 1
        lo, hi = plan[j + 1][0], plan[j][0] + plan[j][1]
        return buf[lo - plan[i][0]:hi - plan[i][0]]

    def head_ok(i, buf, head):
        """Read i agrees with the read written before it -- or, the first read, with
        the member's own headers."""
        if i == 0:
            return _headers_mismatch(buf[:data_offset - hdr], member) is None
        return shared(i, buf, "head") == head

    def trailer(buf):
        """What follows the member, as the last read has it: the next member's header
        sequence, or one block. None if the archive ends first."""
        at = plan[last][0]
        if known:
            return buf[end - at:stop - at]      # short only if the parts shrank: refused
        return buf[end - at:end - at + BLOCK] if stop >= end + BLOCK else None

    def trailer_ok(block):
        if block is None:
            return True
        if known:
            return _headers_mismatch(block, follows) is None
        if not any(block[:BLOCK]):
            return True
        if follows == "zeros":
            return False
        try:
            tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        except tarfile.HeaderError:
            return False
        return True

    def settle(i, buf):
        """Read i as the fetches settle it: the version that leads every other by two.
        Normally that is ``buf`` and one fresh fetch agreeing with it. Once fetches have
        disagreed it takes more, so a wrong body served twice cannot outvote a right one
        served first. What no vote can catch is a server repeating one wrong answer --
        ``buf`` and the fetch after it, or three fetches in a row. Versions are counted
        by digest; only the leader's bytes are kept."""
        votes = Counter({hashlib.sha256(buf).digest(): 1})
        kept = {hashlib.sha256(buf).digest(): buf}
        for _ in range(2 * REREAD_LIMIT):
            again = fetch(i)
            digest = hashlib.sha256(again).digest()
            votes[digest] += 1
            kept.setdefault(digest, again)
            (lead, top), *rest = votes.most_common(2) + [(None, 0)]
            if top - rest[0][1] >= 2:
                return kept[lead]
            leader = votes.most_common(1)[0][0]
            kept = {d: b for d, b in kept.items() if d == leader or d == digest}
        raise _CatError(f"the reads at {plan[i][0]:,} never settled on one version")

    written = data_offset

    def release(i, buf):
        nonlocal written
        at = plan[i][0]
        lo, hi = max(written, at), min(data_offset + size, at + plan[i][1])
        if hi <= lo:
            return b""
        written = hi
        return buf[lo - at:hi - at]

    held = fetch(0)
    for _ in range(REREAD_LIMIT):
        if head_ok(0, held, None):
            break
        held = fetch(0)
    why = _headers_mismatch(held[:data_offset - hdr], member)
    if why is not None:
        raise _CatError(f"the headers at {hdr:,} never read as the member's index row "
                        f"says: {why}")
    # What the held read's head must equal, and whether that vouches for it. The member's
    # headers do -- except that past a part boundary inside them, the request's head is
    # vouched for only by the header bytes after the boundary, and a header's last bytes
    # are mostly zeros.
    head = None
    head_sure = not any(_ambiguous(held[b - hdr:data_offset - hdr]) for b in head_cuts)

    for i in range(1, last + 1):
        nxt, tries = fetch(i), 0
        while shared(i - 1, held, "tail") != shared(i, nxt, "head"):
            tries += 1
            if tries > REREAD_LIMIT:
                raise _CatError(f"the reads at {plan[i - 1][0]:,} and {plan[i][0]:,} "
                                f"never agreed on the bytes they share")
            again = fetch(i)
            if again == nxt:
                # Two reads of the new one agree, so the held one is the likelier wrong:
                # read it again, held to its own head as before.
                redo = fetch(i - 1)
                if head_ok(i - 1, redo, head):
                    held = redo
            nxt = again
        link_sure = not _ambiguous(shared(i - 1, held, "tail"))
        if not (head_sure and link_sure):
            sure = settle(i - 1, held)
            if sure != held:
                # The held read was the wrong one after all; what replaces it must still
                # agree with both of its neighbours.
                if (not head_ok(i - 1, sure, head)
                        or shared(i - 1, sure, "tail") != shared(i, nxt, "head")):
                    raise _CatError(f"the reads around {plan[i - 1][0]:,} never settled")
                held = sure
        piece = release(i - 1, held)
        if piece:
            yield piece
        head, head_sure, held = shared(i - 1, held, "tail"), link_sure, nxt

    tries = 0
    while not trailer_ok(trailer(held)):
        tries += 1
        if tries > REREAD_LIMIT:
            raise _CatError(f"what follows the member, at {end:,}, never read as the "
                            f"index says it should")
        redo = fetch(last)
        if head_ok(last, redo, head):
            held = redo
    # Only the member the index says comes next vouches for the read's tail -- and where a
    # part boundary falls inside its headers, only if their bytes before the boundary say
    # something. The terminator's zeros vouch for no more than zeros anywhere; a header
    # the index cannot name could be anyone's; an archive that ends at the member offers
    # nothing at all.
    at = plan[last][0]
    tail_sure = known and not any(_ambiguous(held[end - at:b - at]) for b in tail_cuts)
    if not (head_sure and tail_sure):
        sure = settle(last, held)
        if sure != held:
            if not head_ok(last, sure, head) or not trailer_ok(trailer(sure)):
                raise _CatError(f"the last read, at {plan[last][0]:,}, never settled")
            held = sure
    piece = release(last, held)
    if piece:
        yield piece


def cmd_archive_cat(args) -> int:
    """Extract one member's bytes by path, addressed directly through the index
    rather than a search: a walk already recorded `data_offset` and `size` for every
    member, so retrieving one is a direct read, not a scan. Each read is checked against
    the member's own header, the block that follows it and the read beside it
    (`_verified_member_bytes`), because Dropbox has returned 206 bodies that were some
    other range -- and here there is no second walk to catch it.

    Task 11 uses this to pull a run's own management database out of a 5.08 TiB tar
    and ask it what "incomplete" means. Everything before this command indexes; this
    is the first one that retrieves, so a wrong answer here is a wrong artefact the
    operator goes on to trust -- a database they query, a log they quote -- not just
    a bad report, which is why every ambiguity below is refused rather than guessed.
    """
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives; "
              f"run `archive register` first", file=sys.stderr)
        return 2
    row = store.get(args.archive)
    if row is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2
    if row["state"] == "stale":
        # The parts changed since this archive was indexed (Task 6's flag), so the
        # stored offsets may no longer describe the bytes actually on Dropbox now --
        # writing whatever currently sits at those offsets into a file the operator
        # will trust is worse than refusing outright.
        print(f"error: {row['name']} is stale ({row['detail'] or ''}); its offsets may "
              f"no longer match the parts -- re-register it before trusting them",
              file=sys.stderr)
        return 1

    if row["state"] == "unsupported":
        # Its index stops at the first pax member, and pax is where a member's name and
        # size can sit outside the checksummed header the read checks below rely on.
        print(f"error: {row['name']} is a pax-format archive, which dbaudit does not "
              f"index or extract: {row['detail'] or ''}", file=sys.stderr)
        return 1

    matches = store.find_members(row["id"], args.member)
    if not matches:
        print(f"error: no member {args.member!r} in {args.archive}", file=sys.stderr)
        return 1

    if args.offset is not None:
        chosen = next((m for m in matches if m["hdr_offset"] == args.offset), None)
        if chosen is None:
            print(f"error: no member {args.member!r} at --offset {args.offset} in "
                  f"{args.archive}; candidates:", file=sys.stderr)
            _print_candidates(matches)
            return 2
    elif len(matches) > 1:
        print(f"error: {len(matches)} members match {args.member!r}; choose one with "
              f"--offset:", file=sys.stderr)
        _print_candidates(matches)
        return 2
    else:
        chosen = matches[0]

    if row["state"] != "complete":
        # A walk not yet finished can hold rows from a chain whose start nothing has
        # confirmed -- one a cold scan began inside a tarball stored in the archive,
        # listing that tarball's members as if they were the archive's own.
        segment = next(s for s in store.segments(row["id"])
                       if s["scan_from"] <= chosen["hdr_offset"]
                       and (s["stop_at"] is None or chosen["hdr_offset"] < s["stop_at"]))
        if not segment["joined"]:
            print(f"error: {args.member!r} was found by a chain whose start is not yet "
                  f"confirmed (segment {segment['idx']}); it may belong to a tar stored "
                  f"inside the archive. Run `archive index` to finish the walk first.",
                  file=sys.stderr)
            return 1

    if chosen["type"] not in REGULAR_MEMBER_TYPES:
        print(f"error: {args.member!r} is not a regular file (type "
              f"{chosen['type']!r}) in {args.archive}", file=sys.stderr)
        return 1

    archive_set = _stored_archive_set(store, row)
    tokens = limiter = None
    if row["source"] == "dropbox":
        try:
            tokens = TokenProvider(remote=args.remote)
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        # A single stream, not a pool -- but Dropbox's budget is account-wide
        # regardless, so a 429 must still park this reader for the full Retry-After
        # rather than a lone stream hammering straight past it (T8-7).
        limiter = AdaptiveLimiter(rps=DEFAULT_RPS, max_concurrency=1)
    reader = _reader_for(row, archive_set, tokens, limiter)
    # A pass-through window (T8-6): the walker's adaptive window guesses generously
    # because it does not know where the next header is, but cat already knows
    # exactly what it wants. With the default window, the final fill of anything not
    # finished in one request would drag part of the *next* member along -- real
    # Dropbox traffic that never reaches the output. window_min=window_max=1 makes
    # every fill want exactly what is asked, clamped only by the archive's own end; the
    # request adds only the lead and tail of its shape (reader.py), up to 64 KiB each
    # side within its part, which are thrown away.
    concat = ConcatFile(archive_set, reader, window_min=1, window_max=1)
    # What must follow the member, for the trailing anchor: the member the index says
    # starts there, or the terminator where a complete walk ended there. A walk stopped
    # early knows neither, and then nothing there vouches for the read.
    end = chosen["data_offset"] + -(-chosen["size"] // BLOCK) * BLOCK
    follows = store.member_at(row["id"], end)
    if follows is None and row["state"] == "complete" and row["end_offset"] == end:
        follows = "zeros"

    # --out is written through a temp file in the same directory, promoted onto the
    # target only once every byte is confirmed written -- never opened (let alone
    # truncated) directly. Fix round 1: at CAT_CHUNK granularity a multi-part database
    # like Task 11's needs many reads, so a ReaderError or short read on any read but
    # the first is the ordinary failure shape, not an edge case, and it must neither
    # leave a partial file at the target nor destroy a good one already there. The
    # same directory keeps the final os.replace atomic rather than a cross-filesystem
    # copy; mkstemp's own uniqueness keeps two concurrent `cat`s from colliding.
    # stdout has no such seam -- bytes already written to it cannot be recalled, so a
    # failure partway through is reported but not undone.
    tmp_path = None
    if args.out:
        out_dir = os.path.dirname(args.out) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".cat-", suffix=".tmp")
        except OSError as exc:
            print(f"error: could not open --out {args.out!r}: {exc}", file=sys.stderr)
            return 2
        sink_cm = os.fdopen(fd, "wb")
    else:
        sink_cm = contextlib.nullcontext(sys.stdout.buffer)

    written = 0
    try:
        with sink_cm as sink:
            try:
                for piece in _verified_member_bytes(concat, chosen,
                                                    archive_set.total_size, follows):
                    sink.write(piece)
                    written += len(piece)
            except _CatError as exc:
                # Not one byte that failed a check was written -- but to stdout, what
                # passed before the failure is already out.
                print(f"error: could not read {args.member!r} from {args.archive} "
                      f"reliably: {exc}", file=sys.stderr)
                return 1
            except ReaderError as exc:
                # A part is shorter than the index believes -- data lost or corrupted
                # at the storage layer since this archive was indexed. Surfacing it as
                # a clean failure beats letting a raw reader exception traceback out
                # of a command whose whole point is to hand the caller a trustworthy
                # file.
                print(f"error: could not read {args.member!r} from {args.archive}: "
                      f"{exc}", file=sys.stderr)
                return 1

        # By construction the loop above only exits normally once written == the size
        # it started from; kept as an explicit check rather than trusted implicitly,
        # because a truncated file returned with exit 0 is the one failure mode here
        # that leaves the operator with nothing to say why (T8-8).
        if written != chosen["size"]:
            print(f"error: wrote {written} of {chosen['size']} byte(s) for "
                  f"{args.member!r} in {args.archive}", file=sys.stderr)
            return 1

        if tmp_path is not None:
            try:
                os.replace(tmp_path, args.out)
            except OSError as exc:
                print(f"error: could not write --out {args.out!r}: {exc}", file=sys.stderr)
                return 2
            tmp_path = None      # now lives at args.out; nothing left to clean up
        return 0
    finally:
        # Reached on every failure return above (the temp file was never promoted) and
        # on nothing else: the target is left exactly as it was, good copy or none.
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
    p_init.add_argument("--min-free-gb", type=float, default=30.0)
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="crawl until finished; safe to interrupt and re-run")
    p_run.add_argument("--db", required=True)
    p_run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_run.add_argument("--rps", type=float, default=DEFAULT_RPS,
                       help="request rate to start at")
    p_run.add_argument("--max-rps", type=float, default=DEFAULT_MAX_RPS,
                       help="ceiling the request rate climbs to while Dropbox allows it")
    p_run.add_argument("--split-depth", type=int, default=DEFAULT_SPLIT_DEPTH,
                       help="max levels a shard may be split into children to keep "
                            "workers busy (a bound, not a target; a listing Dropbox "
                            "keeps failing is split regardless)")
    p_run.add_argument("--queue-target", type=int, default=DEFAULT_QUEUE_TARGET,
                       help="shards to keep pending (default: 2 x workers). Raising "
                            "this splits more aggressively and measured much slower")
    p_run.add_argument("--max-shards", type=int, default=2_000_000)
    p_run.add_argument("--progress-interval", type=float, default=30.0)
    p_run.add_argument("--no-index", action="store_true",
                       help="skip building analysis indexes when the crawl completes")
    p_run.add_argument("--log", help="also write logs to this file")
    p_run.add_argument("--lock", help="lockfile path (default: <db>.lock)")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="show progress")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(func=cmd_status)

    p_verify = sub.add_parser(
        "verify", help="re-walk a subtree with rclone and diff it against the database")
    p_verify.add_argument("--db", required=True)
    p_verify.add_argument("path", nargs="?", help="subtree to check (default: the crawl root)")
    p_verify.set_defaults(func=cmd_verify)

    p_report = sub.add_parser("report", help="storage-reduction report")
    p_report.add_argument("--db", required=True)
    p_report.add_argument("--top", type=int, default=25)
    p_report.add_argument("--hotspot-min-files", type=int, default=100_000)
    p_report.add_argument("--hotspot-max-mean", type=int, default=65536)
    p_report.set_defaults(func=cmd_report)

    p_export = sub.add_parser("export", help="write the reports as CSV")
    p_export.add_argument("--db", required=True)
    p_export.add_argument("--out", required=True)
    p_export.set_defaults(func=cmd_export)

    p_index = sub.add_parser("index", help="build analysis indexes")
    p_index.add_argument("--db", required=True)
    p_index.set_defaults(func=cmd_index)

    p_tree = sub.add_parser("tree", help="write a browsable HTML page of where the space goes")
    p_tree.add_argument("--db", required=True)
    p_tree.add_argument("--out", required=True, help="the HTML file to write")
    p_tree.add_argument("--subtree", help="cover this folder instead of the whole crawl root")
    p_tree.add_argument("--min-size", type=parse_size, default=DEFAULT_TREE_MIN_SIZE,
                        help="fold entries smaller than this, e.g. 100MiB (default 1GiB)")
    p_tree.add_argument("--max-rows", type=int, default=DEFAULT_TREE_MAX_ROWS,
                        help="double --min-size until the page has at most this many rows")
    p_tree.set_defaults(func=cmd_tree)

    p_archive = sub.add_parser("archive", help="index what is inside tar archives")
    asub = p_archive.add_subparsers(dest="archive_cmd", required=True)

    p_reg = asub.add_parser("register", help="record an archive, its parts and its segments")
    p_reg.add_argument("--db", required=True)
    p_reg.add_argument("--name", required=True, help="the archive's own name, e.g. big.tar")
    where = p_reg.add_mutually_exclusive_group(required=True)
    where.add_argument("--folder", help="Dropbox folder holding the parts")
    where.add_argument("--local-dir", help="local directory holding the parts")
    p_reg.add_argument("--remote", default="dropbox")
    p_reg.set_defaults(func=cmd_archive_register)

    p_idx = asub.add_parser("index", help="walk the header chains; safe to interrupt and re-run")
    p_idx.add_argument("--db", required=True)
    p_idx.add_argument("--archive", required=True)
    p_idx.add_argument("--remote", default="dropbox")
    p_idx.add_argument("--workers", type=int, default=ARCHIVE_WORKERS)
    p_idx.add_argument("--rps", type=float, default=DEFAULT_RPS,
                       help="request rate to start at")
    # Measured in Task 1: a request costs 1.60 s + 0.044 s/MiB, so the cap is generous
    # and the floor is small because it is paid on every jump.
    p_idx.add_argument("--window-min", type=int, default=WINDOW_MIN)
    p_idx.add_argument("--window-max", type=int, default=WINDOW_MAX)
    p_idx.add_argument("--batch", type=int, default=2000)
    p_idx.add_argument("--max-batches", type=int, default=0, help="stop early; 0 means no limit")
    p_idx.add_argument("--recheck", action="store_true",
                       help="ask a corrupt, truncated or unsupported verdict again: walk the "
                            "chain again from the start of the segment before it -- and, "
                            "asked again about the same verdict, a segment further back "
                            "each time")
    p_idx.add_argument("--lock")
    p_idx.set_defaults(func=cmd_archive_index)

    p_arun = asub.add_parser(
        "run", help="walk every archive still to be walked, many at once; safe to "
                    "interrupt and re-run")
    p_arun.add_argument("--db", required=True)
    p_arun.add_argument("--audit-db", required=True,
                        help="the audit database whose tar archives to walk")
    p_arun.add_argument("--remote", default="dropbox")
    p_arun.add_argument("--streams", type=int, default=12, help="streams to start at")
    p_arun.add_argument("--max-streams", type=int, default=24,
                        help="most streams to climb to while Dropbox allows")
    p_arun.add_argument("--workers", type=int, default=ARCHIVE_WORKERS,
                        help="most chains one archive may have")
    p_arun.add_argument("--limit", type=int, default=0,
                        help="most archives to start; 0 means no limit")
    p_arun.add_argument("--dry-run", action="store_true",
                        help="print what would be walked, without contacting Dropbox")
    p_arun.add_argument("--window-min", type=int, default=WINDOW_MIN)
    p_arun.add_argument("--window-max", type=int, default=WINDOW_MAX)
    p_arun.add_argument("--batch", type=int, default=2000)
    p_arun.add_argument("--lock")
    p_arun.set_defaults(func=cmd_archive_run, max_batches=0)

    p_aimp = asub.add_parser(
        "import", help="copy every archive of another archive database into this one")
    p_aimp.add_argument("--db", required=True)
    p_aimp.add_argument("--from", dest="source", required=True,
                        help="the archive database to copy from; it is left as it is")
    p_aimp.set_defaults(func=cmd_archive_import)

    p_afgt = asub.add_parser(
        "forget", help="drop an archive and all its rows, so the next run walks it afresh")
    p_afgt.add_argument("--db", required=True)
    p_afgt.add_argument("--archive", required=True)
    p_afgt.add_argument("--folder", required=True)
    p_afgt.set_defaults(func=cmd_archive_forget)

    p_ast = asub.add_parser("status", help="show archive indexing progress")
    p_ast.add_argument("--db", required=True)
    p_ast.add_argument("--archive")
    p_ast.set_defaults(func=cmd_archive_status)

    p_areport = asub.add_parser("report", help="rollups over one archive's manifest")
    p_areport.add_argument("--db", required=True)
    p_areport.add_argument("--archive", required=True)
    p_areport.add_argument("--top", type=int, default=25)
    p_areport.set_defaults(func=cmd_archive_report)

    p_aexport = asub.add_parser(
        "export", help="write one archive's manifest and rollups as CSV")
    p_aexport.add_argument("--db", required=True)
    p_aexport.add_argument("--archive", required=True)
    p_aexport.add_argument("--out", required=True)
    p_aexport.set_defaults(func=cmd_archive_export)

    p_cat = asub.add_parser(
        "cat", help="extract one member's bytes by path, in the fewest range reads")
    p_cat.add_argument("--db", required=True)
    p_cat.add_argument("--archive", required=True)
    p_cat.add_argument("--member", required=True, help="the member's path inside the archive")
    p_cat.add_argument("--offset", type=int, default=None,
                       help="disambiguate a path matching more than one member, by "
                            "hdr_offset (a bare ambiguous cat prints the candidates)")
    p_cat.add_argument("--out", help="write to this file instead of stdout")
    p_cat.add_argument("--remote", default="dropbox")
    p_cat.set_defaults(func=cmd_archive_cat)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "verbose"):
        args.verbose = False
    return args.func(args)
