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
import logging
import os
import signal
import socket
import sqlite3
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .api import HttpLister
from .archive.parts import ArchiveSet, ArchiveSetError
from .archive.reader import (WINDOW_MAX, WINDOW_MIN, ConcatFile, DropboxRangeReader,
                             LocalRangeReader, ReaderError)
from .archive import report as archive_report
from .archive.store import ArchiveStore
from .archive.tarwalk import (BLOCK, REREAD_LIMIT, UnsupportedArchive, WalkResult,
                              find_chain_start, read_member, walk)
from .auth import AuthError, TokenProvider
from .crawler import Crawler
from .limiter import AdaptiveLimiter
from .lock import InstanceLock, LockHeld
from .store import Store
from .verify import verify_subtree
from . import report as reports

log = logging.getLogger("dbaudit")

# Measured on the live account 2026-08-13: 8 concurrent listers sustained
# ~4,600 entries/s with zero 429s; 16 started drawing them.
DEFAULT_WORKERS = 8
DEFAULT_RPS = 5.0
# Chains for `archive index`. Deliberately NOT DEFAULT_WORKERS: that figure measures
# `files/list_folder`, a different endpoint class, and conflating the two is a mistake
# this project has already made once and corrected. The closest measured analogue for
# scattered, latency-bound `files/download` reads is alpine_simulation_workflow commit
# fccd755, where 8 concurrent streams ran clean. arr65 fixed this at 8 on 2026-09-18:
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
    # Taken before the crawl starts, so the first incremental pass also catches
    # anything that changed while the full crawl was running.
    delta = None
    try:
        delta = crawler.seed_delta_cursor(root)
    except Exception as exc:
        print(f"  warning: could not record a delta cursor ({exc}); "
              f"incremental passes will be unavailable", file=sys.stderr)

    print(f"initialised {args.db}")
    print(f"  root      : {root} ({len(page.entries)}+ entries at the top level)")
    print(f"  account   : {account.get('email', '?')} / team {(account.get('team') or {}).get('name', '?')}")
    print(f"  namespace : {tokens.root_namespace_id()}")
    print(f"  free disk : {human_bytes(available)}")
    print(f"  delta     : {'recorded' if delta else 'unavailable'} "
          f"(enables `run --incremental` later)")
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

        limiter = AdaptiveLimiter(rps=args.rps, max_concurrency=args.workers)
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
    pending = store.pending_count() + store.running_count()
    root = store.get_meta("root")
    print(f"# Storage audit of {root}")
    if pending:
        print(f"\n  WARNING: the crawl is incomplete -- {pending} shard(s) outstanding.")
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


def _fingerprint(store, row, remote):
    """Compare every registered part against what the folder holds right now.

    Matching is by file id first and by name second: all 18 parts of v01p0_incomplete
    moved between folders in five weeks, and an id survives that where a path does not.
    Returns (missing, changed) part names.
    """
    _, lister = build_lister(remote)
    entries = _folder_entries(lister, row["folder"], row["name"])
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
        self.stop = stop
        self.lock = threading.Lock()
        self.batches = 0
        #: Set by a worker whose own machinery failed, as opposed to a bad part. A run
        #: that ends this way must not exit 0: on a nohup'd walk the exit code is most
        #: of what anyone sees, and "stopped cleanly" and "a chain blew up" are not the
        #: same answer.
        self.failed = False
        #: Set by the first chain to meet a pax header. The format is the archive's, not
        #: one part's, so every other chain would meet it too: the run stops.
        self.unsupported = None

    def commit(self, segment_id, members, next_offset, requests, bytes_fetched) -> None:
        """Write one batch, or refuse to once `--max-batches` has been reached.

        Refusing *before* the write is what makes `--max-batches N` mean exactly N
        batches rather than N plus whatever was in flight. Holding the lock across the
        write is fine: SQLite serialises writers anyway.

        Only a commit that carries members is refused. An empty one buys nothing, so
        refusing it cannot help the budget, and it costs: a cursor goes unwritten, and
        the walk's confirming second read of a final verdict ends as a stop instead of
        the verdict it went to fetch.

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
                    self.stop.set()


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
                      stop_at=segment["stop_at"], should_stop=run.stop.is_set)
    except _Stopped:
        # The refused batch was read before it was refused.
        requests, fetched = delta()
        run.store.release_segment(segment["id"], requests, fetched)
        return
    except UnsupportedArchive as exc:
        # Not a bad part: the archive's format, which every chain would meet. Stop the
        # pool and let `_run_index` mark the archive for good.
        requests, fetched = delta()
        with run.lock:
            if run.unsupported is None:
                run.unsupported = str(exc)
        run.stop.set()
        log.error("segment %d: %s", segment["idx"], exc)
        run.store.fail_segment(segment["id"], f"UnsupportedArchive: {exc}",
                               requests, fetched)
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
        while not run.stop.is_set():
            segment = run.store.claim_segment(run.archive_id, owner)
            if segment is None:
                return
            if run.stop.is_set():
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
    elsewhere in the archive always makes one: it records another member's header a
    second time, at an offset that was never that member's. When the member it replaced
    pads to the same size, the chain rejoins straight after, and nothing else notices --
    the walk ends `complete` with one member twice and one never. A fresh read of each
    such row says which is which. An archive with no repeated path pays nothing.
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
    # Its own reader and window: every byte compared comes from a request of its own.
    reader = _reader_for(run.row, run.archive_set, run.tokens, run.limiter)
    concat = ConcatFile(run.archive_set, reader, window_min=run.args.window_min,
                        window_max=run.args.window_max)
    try:
        wrong = [row["hdr_offset"] for row in rows
                 if _row_of(read_member(concat, row["hdr_offset"])) != _row_of(row)]
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
            if following["first_header"] == segment["exit_offset"]:
                if not following["joined"]:
                    store.mark_joined(following["id"])
                index += 1
                continue
            # Disagreement, or a chain that found no header at all. reset_segment drops
            # that segment's rows and re-arms it from the offset handed down, atomically.
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
    if row["state"] == "complete":
        print(f"{row['name']}: complete ({row['n_members']:,} members)")
        return 0
    if row["state"] in ("truncated", "corrupt"):
        print(f"error: {row['name']} is {row['state']}: {row['detail'] or ''}",
              file=sys.stderr)
        return 1
    if row["state"] == "stale":
        print(f"error: {row['name']} is stale ({row['detail'] or ''}); re-register it "
              f"to index the parts as they are now", file=sys.stderr)
        return 1
    if row["state"] == "unsupported":
        print(f"error: {row['name']} is a pax-format archive, which dbaudit does not "
              f"index: {row['detail'] or ''}", file=sys.stderr)
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


def _run_index(args, store, row) -> int:
    archive_id = row["id"]
    archive_set = _stored_archive_set(store, row)
    dropbox = row["source"] == "dropbox"

    # A before/after fingerprint of every part, as the spec requires. Local archives
    # are not fingerprinted: there is no cheap hash to ask for, and re-hashing 5 TiB
    # would cost far more than the walk it is meant to protect.
    if dropbox:
        try:
            missing, changed = _fingerprint(store, row, args.remote)
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: could not list {row['folder']}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        if missing:
            detail = (f"{len(missing)} part(s) missing from {row['folder']}: "
                      f"{', '.join(missing[:5])}")
            store.fail(archive_id, detail)
            print(f"error: {detail}\n       re-register the archive with its new "
                  f"--folder", file=sys.stderr)
            return 1
        if changed:
            detail = (f"{len(changed)} part(s) changed since registration: "
                      f"{', '.join(changed[:5])}")
            store.mark_stale(archive_id, detail)
            print(f"error: {detail}\n       the stored offsets describe bytes that are "
                  f"no longer there; re-register to walk it again", file=sys.stderr)
            return 1

    workers = max(args.workers, 1)
    tokens = limiter = None
    if dropbox:
        # One provider and one limiter for the whole command: Dropbox limits per
        # account, and the provider caches its token behind a lock.
        tokens = TokenProvider(remote=args.remote)
        limiter = AdaptiveLimiter(rps=args.rps, max_concurrency=workers)

    store.start_walk(archive_id)
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
    run = _Run(args, store, row, archive_set, tokens, limiter, stop)
    started = time.time()
    outcome = blocked = audit_failed = None
    exhausted = False
    log.info("walking %s (%d parts, %s) with %d chains",
             row["name"], row["n_parts"], human_bytes(row["total_size"]), workers)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle)
        # Each repair advances the confirmed frontier by at least one segment, so this
        # is bounded by the segment count; in the worst case it degrades to a
        # sequential walk. The bound is asserted rather than assumed.
        for _round in range(len(store.segments(archive_id)) + 2):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in [pool.submit(_worker, run) for _ in range(workers)]:
                    future.result()
            if run.unsupported:
                break                     # no verdict to join towards: see below
            kind, payload = _join(store, archive_id)
            if kind == _OUTCOME and payload.state == "complete":
                try:
                    redo = _audit_repeated_paths(run)
                except Exception as exc:
                    audit_failed = (f"could not read again the rows under repeated paths: "
                                    f"{type(exc).__name__}: {exc}")
                    break
                if redo:
                    # No stop check needed: a stopping pool claims nothing, so the next
                    # join finds them pending and the run ends there, resumable.
                    log.warning("segment(s) %s held a row that disagreed with a fresh "
                                "read; walking them again", ", ".join(map(str, redo)))
                    continue
            if kind == _OUTCOME:
                outcome = payload
                break
            if kind == _BLOCKED:
                blocked = payload
                break
            log.info("segment %d did not meet the chain before it; re-walking it",
                     payload)
            if stop.is_set():
                break
        else:
            # Unreachable by construction; if it ever happens it is a bug, not a stop.
            exhausted = True
            log.error("giving up after too many join rounds; see `archive status`")
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    stale = None
    if dropbox:
        try:
            missing, changed = _fingerprint(store, row, args.remote)
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

    if run.unsupported:
        store.mark_unsupported(archive_id, run.unsupported)
    elif audit_failed is not None:
        # Every chain is walked, so the retry this invites goes straight to the audit.
        store.fail(archive_id, audit_failed)
    elif outcome is not None:
        store.finish(archive_id, outcome)
        store.build_indexes(archive_id)        # only once a walk is over, never during
    elif blocked is not None and blocked["state"] == "error":
        store.fail(archive_id, f"segment {blocked['idx']}: {blocked['error'] or 'failed'}")

    stats = store.stats(archive_id)
    rate_limited = limiter.rate_limit_events if limiter else 0
    if rate_limited:
        store.log_event(archive_id, "rate_limited", str(rate_limited))
    ratio = (stats["bytes_fetched"] / stats["member_bytes"]) if stats["member_bytes"] else 0.0
    print(f"{row['name']}: {stats['state']}")
    print(f"  members       : {stats['n_members']:,} "
          f"({human_bytes(stats['member_bytes'])} of member data)")
    print(f"  requests      : {stats['requests']:,}")
    print(f"  bytes fetched : {human_bytes(stats['bytes_fetched'])}")
    print(f"  ratio         : {ratio:.3f} bytes fetched per member byte")
    print(f"  429s          : {rate_limited}")
    print(f"  elapsed       : {human_duration(time.time() - started)}")

    if stale is not None:
        print(f"error: {stale}; the members found are kept, but re-register before "
              f"trusting them", file=sys.stderr)
        return 1
    if run.unsupported:
        print(f"error: {row['name']} is a pax-format archive, which dbaudit does not index. "
              f"{run.unsupported}", file=sys.stderr)
        return 1
    if audit_failed is not None:
        print(f"error: the walk finished, but the audit {audit_failed}; run `index` again "
              f"to retry it", file=sys.stderr)
        return 1
    if run.failed:
        print(f"error: a chain stopped on an unexpected failure; see the log and "
              f"`archive status`", file=sys.stderr)
        return 1
    if outcome is not None:
        if outcome.state == "complete":
            return 0
        print(f"error: {row['name']} is {outcome.state}: {outcome.detail}",
              file=sys.stderr)
        return 1
    if blocked is not None and blocked["state"] == "error":
        print(f"error: segment {blocked['idx']}: {blocked['error']}", file=sys.stderr)
        return 1
    if exhausted:
        print(f"error: {row['name']} did not settle; see `archive status`",
              file=sys.stderr)
        return 1
    log.info("stopped before the chain reached a verdict; re-run to resume")
    return 0


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


def cmd_archive_status(args) -> int:
    store = _open_archive_store(args.db)
    if store is None:
        print(f"error: {args.db} holds no registered archives", file=sys.stderr)
        return 2
    if args.archive:
        row = store.get(args.archive)
        if row is None:
            print(f"error: no archive named {args.archive!r} in {args.db}",
                  file=sys.stderr)
            return 2
        rows = [row]
    else:
        rows = store.connect().execute("SELECT * FROM archives ORDER BY id").fetchall()
    if not rows:
        print(f"{args.db}: no archives registered yet")
        return 0
    for row in rows:
        _print_archive_status(store, row)
    return 0


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


def _verified_member_bytes(concat, data_offset, size, total, next_is):
    """Yield one member's bytes, each read checked against what surrounds it.

    A tar keeps no checksum of member data, so no read can be proved right -- but a
    body that is some other range entirely, which is how Dropbox got it wrong on
    2026-09-21, is caught for almost nothing by reading a little either side:

    - the first read starts one block early, on the member's own header, which must be
      checksum-valid and state this member's size;
    - the last read ends one block late, on whatever follows, which must be what the
      index says is there (`next_is`: "header", "zeros" for the terminator, or "either"
      when the index cannot say);
    - consecutive reads overlap by CAT_OVERLAP and must agree on it, so every read's
      head is checked before any of its bytes are yielded.

    A read failing a check is read again, fresh, up to REREAD_LIMIT times. An overlap
    that is all zeros agrees with any other zeros, so a read starting on one counts only
    once two fresh reads of it match. Past the limit this raises _CatError, having
    yielded nothing that failed. A read wrong only in its interior, right at both ends,
    is not caught; that is not how this has been seen to fail.
    """
    if data_offset + size > total:
        raise _CatError(f"the archive ends {data_offset + size - total:,} byte(s) "
                        f"before this member does")
    end = data_offset + -(-size // BLOCK) * BLOCK      # where the next header sits
    stop = min(end + BLOCK, total)
    has_trailer = stop >= end + BLOCK

    def fetch(at, length):
        concat.drop_cache()                            # a re-read must be a new request
        concat.seek(at)
        buf = concat.read(length)
        if len(buf) != length:
            raise _CatError(f"read {len(buf):,} of {length:,} byte(s) at {at:,}")
        return buf

    def is_header(block, want_size=None):
        try:
            info = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        except tarfile.HeaderError:
            return False
        return want_size is None or info.size == want_size

    def trailer_ok(block):
        if next_is == "header":
            return is_header(block)
        if next_is == "zeros":
            return not any(block)
        return not any(block) or is_header(block)

    at, written, seen_tail = data_offset - BLOCK, data_offset, None
    while True:
        length = min(CAT_CHUNK, stop - at)
        last = at + length >= stop
        for _ in range(REREAD_LIMIT + 1):
            buf = fetch(at, length)
            ok = (is_header(buf[:BLOCK], size) if seen_tail is None
                  else buf[:len(seen_tail)] == seen_tail)
            if ok and last and has_trailer:
                ok = trailer_ok(buf[end - at:end - at + BLOCK])
            if ok and seen_tail is not None and not any(seen_tail):
                ok = fetch(at, length) == buf
            if ok:
                break
        else:
            raise _CatError(f"{REREAD_LIMIT + 1} reads at {at:,} never lined up with "
                            f"what surrounds them")
        lo, hi = max(written, at), min(data_offset + size, at + length)
        if hi > lo:
            yield buf[lo - at:hi - at]
            written = hi
        if last:
            return
        seen_tail = buf[-CAT_OVERLAP:]
        at += length - CAT_OVERLAP


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
    # every fill read exactly what is asked, clamped only by the archive's own end.
    concat = ConcatFile(archive_set, reader, window_min=1, window_max=1)
    # What the block after the member must be, for the trailing anchor. The index knows
    # when another member starts there, and when a complete walk put the terminator
    # there; a partial index knows neither, and then either will do.
    end = chosen["data_offset"] + -(-chosen["size"] // BLOCK) * BLOCK
    if store.member_at(row["id"], end) is not None:
        next_is = "header"
    elif row["state"] == "complete" and row["end_offset"] == end:
        next_is = "zeros"
    else:
        next_is = "either"

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
                for piece in _verified_member_bytes(concat, chosen["data_offset"],
                                                    chosen["size"],
                                                    archive_set.total_size, next_is):
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
    p_init.add_argument("--split-depth", type=int, default=DEFAULT_SPLIT_DEPTH)
    p_init.add_argument("--min-free-gb", type=float, default=30.0)
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="crawl until finished; safe to interrupt and re-run")
    p_run.add_argument("--db", required=True)
    p_run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_run.add_argument("--rps", type=float, default=DEFAULT_RPS)
    p_run.add_argument("--split-depth", type=int, default=DEFAULT_SPLIT_DEPTH,
                       help="max levels a shard may be split into children (bound, "
                            "not a target: splitting only happens when workers starve)")
    p_run.add_argument("--queue-target", type=int, default=DEFAULT_QUEUE_TARGET,
                       help="shards to keep pending (default: 2 x workers). Raising "
                            "this splits more aggressively and measured much slower")
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
    p_idx.add_argument("--rps", type=float, default=DEFAULT_RPS)
    # Measured in Task 1: a request costs 1.60 s + 0.044 s/MiB, so the cap is generous
    # and the floor is small because it is paid on every jump.
    p_idx.add_argument("--window-min", type=int, default=WINDOW_MIN)
    p_idx.add_argument("--window-max", type=int, default=WINDOW_MAX)
    p_idx.add_argument("--batch", type=int, default=2000)
    p_idx.add_argument("--max-batches", type=int, default=0, help="stop early; 0 means no limit")
    p_idx.add_argument("--lock")
    p_idx.set_defaults(func=cmd_archive_index)

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
