"""SQLite store. The atomic page commit lives here.

The one invariant everything else depends on: **a page's rows and its shard's cursor
are written in a single transaction**. Kill the process at any instant and SQLite
rolls back the partial page, so a restart re-fetches exactly that page and nothing
is lost or duplicated. There is no reconciliation pass because there is nothing to
reconcile.
"""

from __future__ import annotations

import calendar
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DIR_CACHE_LIMIT = 200_000
TIME_CACHE_LIMIT = 50_000

# Built after the first full pass, never during it: a random-order index over 100M
# rows would slow the crawl for no benefit while it runs.
ANALYSIS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS files_dir ON files(dir_id)",
    "CREATE INDEX IF NOT EXISTS files_hash ON files(content_hash)",
    "CREATE INDEX IF NOT EXISTS files_size ON files(size)",
    "CREATE INDEX IF NOT EXISTS files_smod ON files(server_modified)",
    "CREATE INDEX IF NOT EXISTS dirs_parent ON dirs(parent_id)",
    "CREATE INDEX IF NOT EXISTS dirs_pathlower ON dirs(path_lower)",
]


@dataclass
class Shard:
    id: int
    path: str
    depth: int
    mode: str
    state: str
    cursor: str | None
    pages: int
    entries: int


@dataclass
class PageStats:
    files: int = 0
    dirs: int = 0
    bytes: int = 0
    deleted: int = 0


def _epoch(stamp: str | None) -> int | None:
    """Parse Dropbox's ``2020-01-02T03:04:05Z`` stamps without strptime overhead."""
    if not stamp:
        return None
    try:
        return calendar.timegm(
            (
                int(stamp[0:4]), int(stamp[5:7]), int(stamp[8:10]),
                int(stamp[11:13]), int(stamp[14:16]), int(stamp[17:19]),
                0, 0, 0,
            )
        )
    except (ValueError, IndexError):
        return None


def _depth_of(path: str) -> int:
    return path.rstrip("/").count("/")


class Store:
    def __init__(self, path):
        self.path = str(path)
        self._local = threading.local()
        self._meta_lock = threading.Lock()

    # ---- connections ---------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """One connection per thread. WAL lets readers run while a worker writes."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA cache_size=-262144")  # 256 MB page cache
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
            self._local.dir_cache = {}
            self._local.principal_cache = {}
            self._local.time_cache = {}
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _caches(self):
        self.connect()
        return self._local.dir_cache, self._local.principal_cache, self._local.time_cache

    def _drop_caches(self) -> None:
        """Called after a rollback: cached ids may reference rows that no longer exist."""
        self._local.dir_cache = {}
        self._local.principal_cache = {}

    # ---- schema and metadata -------------------------------------------

    def init_schema(self) -> None:
        conn = self.connect()
        conn.executescript(SCHEMA_PATH.read_text())

    def set_meta(self, key: str, value) -> None:
        with self._meta_lock:
            self.connect().execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def is_initialised(self) -> bool:
        """True once `init` has created the schema. Probing must not raise."""
        if not os.path.exists(self.path):
            return False
        row = self.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        return row is not None

    def get_meta(self, key: str, default=None):
        try:
            row = self.connect().execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
        except sqlite3.OperationalError:
            return default  # schema not created yet
        return default if row is None else row[0]

    def query(self, sql: str, params=()) -> list[tuple]:
        return self.connect().execute(sql, params).fetchall()

    def log_event(self, kind: str, detail: str = "") -> None:
        self.connect().execute(
            "INSERT INTO api_events(ts, kind, detail) VALUES(?, ?, ?)",
            (time.time(), kind, detail[:500]),
        )

    # ---- shards ---------------------------------------------------------

    def add_shard(self, path: str, depth: int, mode: str = "recursive") -> int:
        conn = self.connect()
        conn.execute(
            "INSERT OR IGNORE INTO shards(path, depth, mode, state, created_at) "
            "VALUES(?, ?, ?, 'pending', ?)",
            (path, depth, mode, time.time()),
        )
        return conn.execute("SELECT id FROM shards WHERE path=?", (path,)).fetchone()[0]

    def get_shard(self, shard_id: int) -> Shard | None:
        row = self.connect().execute(
            "SELECT id, path, depth, mode, state, cursor, pages, entries FROM shards WHERE id=?",
            (shard_id,),
        ).fetchone()
        return Shard(*row) if row else None

    def claim_shard(self, owner: str) -> Shard | None:
        """Atomically take the shallowest pending shard, so progress stays breadth-first."""
        conn = self.connect()
        now = time.time()
        row = conn.execute(
            "UPDATE shards SET state='running', owner=?, attempts=attempts+1, "
            "  started_at=COALESCE(started_at, ?), heartbeat_at=? "
            "WHERE id = (SELECT id FROM shards WHERE state='pending' "
            "            ORDER BY depth, id LIMIT 1) "
            "RETURNING id, path, depth, mode, state, cursor, pages, entries",
            (owner, now, now),
        ).fetchone()
        return Shard(*row) if row else None

    def finish_shard(self, shard_id: int, note: str | None = None) -> None:
        self.connect().execute(
            "UPDATE shards SET state='done', finished_at=?, owner=NULL, note=COALESCE(?, note) "
            "WHERE id=?",
            (time.time(), note, shard_id),
        )

    def fail_shard(self, shard_id: int, error: str) -> None:
        self.connect().execute(
            "UPDATE shards SET state='error', finished_at=?, owner=NULL, error=? WHERE id=?",
            (time.time(), error[:1000], shard_id),
        )

    def release_shard(self, shard_id: int) -> None:
        """Hand an unfinished shard back to the queue, keeping its cursor."""
        self.connect().execute(
            "UPDATE shards SET state='pending', owner=NULL WHERE id=? AND state='running'",
            (shard_id,),
        )

    def set_shard_mode(self, shard_id: int, mode: str) -> None:
        """Persist the recursive/split choice *before* the first page.

        A cursor belongs to the kind of listing that produced it, so a resumed shard
        must continue in the same mode. Persisting the decision is what stops a
        restart from resuming a non-recursive cursor as a recursive one.
        """
        self.connect().execute("UPDATE shards SET mode=? WHERE id=?", (mode, shard_id))

    def clear_cursor(self, shard_id: int) -> None:
        """After a cursor reset: forget the cursor and drop the rows it produced.

        `files.dbx_id` is UNIQUE, so a re-list would update rather than duplicate --
        but rows for entries deleted since the reset would linger, so clear them.

        Directory rows are deliberately left alone. A directory may have been created
        as an ancestor of another shard's files, and deleting it here would orphan
        their dir_id. Re-listing upserts directories anyway.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM files WHERE shard_id=?", (shard_id,))
            conn.execute(
                "UPDATE shards SET cursor=NULL, pages=0, entries=0, n_files=0, n_dirs=0, "
                "bytes=0 WHERE id=?",
                (shard_id,),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            self._drop_caches()

    def begin_incremental_pass(self) -> int:
        """Re-open every completed shard at its stored cursor.

        Dropbox keeps cursors valid after a listing completes, so continuing from one
        returns only what changed since. This is what makes a repeat audit cost
        minutes instead of hours.
        """
        conn = self.connect()
        unfinished = conn.execute(
            "SELECT COUNT(*) FROM shards WHERE state != 'done'"
        ).fetchone()[0]
        if unfinished:
            raise RuntimeError(
                f"cannot start an incremental pass: {unfinished} shard(s) from the "
                f"full pass are unfinished -- run a normal pass to completion first"
            )
        run = int(self.get_meta("seen_run", 1)) + 1
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('seen_run', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(run),),
            )
            conn.execute(
                "UPDATE shards SET state='pending', owner=NULL, finished_at=NULL "
                "WHERE cursor IS NOT NULL"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        self.refresh_seen_run()
        return run

    def reset_stale_shards(self) -> int:
        """Reclaim shards owned by a process that died. Cursors are preserved."""
        cur = self.connect().execute(
            "UPDATE shards SET state='pending', owner=NULL WHERE state='running'"
        )
        return cur.rowcount

    def pending_count(self) -> int:
        return self.connect().execute(
            "SELECT COUNT(*) FROM shards WHERE state='pending'"
        ).fetchone()[0]

    def running_count(self) -> int:
        return self.connect().execute(
            "SELECT COUNT(*) FROM shards WHERE state='running'"
        ).fetchone()[0]

    def shard_count(self) -> int:
        return self.connect().execute("SELECT COUNT(*) FROM shards").fetchone()[0]

    # ---- the hot path ---------------------------------------------------

    def _principal_id(self, conn, cache, account_id):
        if not account_id:
            return None
        cached = cache.get(account_id)
        if cached is not None:
            return cached
        conn.execute(
            "INSERT OR IGNORE INTO principals(dbx_account_id) VALUES(?)", (account_id,)
        )
        pid = conn.execute(
            "SELECT id FROM principals WHERE dbx_account_id=?", (account_id,)
        ).fetchone()[0]
        cache[account_id] = pid
        return pid

    def ensure_dir(self, conn, path_display: str, cache, shard_id=None, created=None) -> int:
        """Return the id of ``path_display``, creating it and any missing ancestors.

        Called with the parent path of every file, so it must be cheap; the cache
        makes it roughly free within a page, since recursive listings are local.

        ``created`` is a one-element list used as a counter of rows actually inserted.
        Counting insertions rather than folder entries is what keeps SUM(shards.n_dirs)
        equal to COUNT(*) FROM dirs: a recursive listing repeats its own root, and
        ancestors are created from child paths without ever appearing as an entry.
        """
        path_display = path_display.rstrip("/")
        key = path_display.lower()
        cached = cache.get(key)
        if cached is not None:
            return cached

        parent_path = path_display.rsplit("/", 1)[0]
        parent_id = (
            self.ensure_dir(conn, parent_path, cache, shard_id, created)
            if parent_path and parent_path != path_display
            else None
        )
        name = path_display.rsplit("/", 1)[-1]
        before = conn.total_changes
        conn.execute(
            "INSERT OR IGNORE INTO dirs(parent_id, name, path_display, path_lower, depth, "
            "                           shard_id, seen_run) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (parent_id, name, path_display, key, _depth_of(path_display), shard_id,
             self._seen_run(conn)),
        )
        if created is not None and conn.total_changes > before:
            created[0] += 1
        dir_id = conn.execute("SELECT id FROM dirs WHERE path_lower=?", (key,)).fetchone()[0]
        if len(cache) >= DIR_CACHE_LIMIT:
            cache.clear()
        cache[key] = dir_id
        return dir_id

    def _seen_run(self, conn) -> int:
        run = getattr(self._local, "seen_run", None)
        if run is None:
            row = conn.execute("SELECT value FROM meta WHERE key='seen_run'").fetchone()
            run = int(row[0]) if row else 1
            self._local.seen_run = run
        return run

    def refresh_seen_run(self) -> None:
        self._local.seen_run = None

    def commit_page(
        self,
        shard_id: int,
        entries: list[dict],
        cursor: str | None,
        has_more: bool,
        child_shards=(),
        shard_path: str = "",
    ) -> PageStats:
        """Write one page and its cursor in a single transaction.

        Anything raised in here rolls the whole page back, cursor included, so the
        page is simply re-fetched on restart.
        """
        conn = self.connect()
        dir_cache, principal_cache, time_cache = self._caches()
        seen_run = self._seen_run(conn)
        stats = PageStats()
        created_dirs = [0]

        try:
            conn.execute("BEGIN IMMEDIATE")

            file_rows = []
            deleted_paths = []
            for entry in entries:
                tag = entry.get(".tag")
                path_display = entry.get("path_display")
                if not path_display:
                    # Rare, but seen with some shared-folder entries. Place it under
                    # the shard root rather than dropping it, and leave a trace.
                    name = entry.get("name") or "<unnamed>"
                    path_display = f"{shard_path.rstrip('/')}/{name}" if shard_path else f"/{name}"
                    conn.execute(
                        "INSERT INTO api_events(ts, kind, detail) VALUES(?, 'no_path_display', ?)",
                        (time.time(), path_display[:500]),
                    )

                if tag == "file":
                    parent = path_display.rsplit("/", 1)[0] or "/"
                    dir_id = self.ensure_dir(conn, parent, dir_cache, shard_id, created_dirs)
                    size = int(entry["size"])
                    content_hash = entry.get("content_hash")
                    sharing = entry.get("sharing_info") or {}
                    file_rows.append(
                        (
                            entry.get("id"),
                            dir_id,
                            entry.get("name") or path_display.rsplit("/", 1)[-1],
                            size,
                            bytes.fromhex(content_hash) if content_hash else None,
                            entry.get("rev"),
                            self._cached_epoch(entry.get("client_modified"), time_cache),
                            self._cached_epoch(entry.get("server_modified"), time_cache),
                            self._principal_id(conn, principal_cache, sharing.get("modified_by")),
                            1 if entry.get("is_downloadable", True) else 0,
                            shard_id,
                            seen_run,
                        )
                    )
                    stats.files += 1
                    stats.bytes += size

                elif tag == "folder":
                    dir_id = self.ensure_dir(conn, path_display, dir_cache, shard_id, created_dirs)
                    sharing = entry.get("sharing_info") or {}
                    shared_folder_id = entry.get("shared_folder_id")
                    conn.execute(
                        "UPDATE dirs SET dbx_id=?, shared_folder_id=?, parent_shared_folder_id=?, "
                        "  is_mount=?, seen_run=? WHERE id=?",
                        (
                            entry.get("id"),
                            shared_folder_id,
                            entry.get("parent_shared_folder_id")
                            or sharing.get("parent_shared_folder_id"),
                            1 if shared_folder_id else 0,
                            seen_run,
                            dir_id,
                        ),
                    )

                elif tag == "deleted":
                    deleted_paths.append(path_display)
                    stats.deleted += 1

            stats.dirs = created_dirs[0]

            if file_rows:
                conn.executemany(
                    "INSERT INTO files(dbx_id, dir_id, name, size, content_hash, rev, "
                    "  client_modified, server_modified, modified_by, is_downloadable, "
                    "  shard_id, seen_run) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(dbx_id) DO UPDATE SET "
                    "  dir_id=excluded.dir_id, name=excluded.name, size=excluded.size, "
                    "  content_hash=excluded.content_hash, rev=excluded.rev, "
                    "  client_modified=excluded.client_modified, "
                    "  server_modified=excluded.server_modified, "
                    "  modified_by=excluded.modified_by, shard_id=excluded.shard_id, "
                    "  seen_run=excluded.seen_run",
                    file_rows,
                )

            for path in deleted_paths:
                lowered = path.lower()
                conn.execute(
                    "DELETE FROM files WHERE dir_id IN (SELECT id FROM dirs WHERE path_lower=?) "
                    "  AND lower(name)=?",
                    (lowered.rsplit("/", 1)[0] or "/", lowered.rsplit("/", 1)[-1]),
                )
                conn.execute(
                    "INSERT INTO tombstones(path_lower, seen_run, ts) VALUES(?, ?, ?)",
                    (lowered, seen_run, time.time()),
                )

            for path, depth, mode in child_shards:
                conn.execute(
                    "INSERT OR IGNORE INTO shards(path, depth, mode, state, created_at) "
                    "VALUES(?, ?, ?, 'pending', ?)",
                    (path, depth, mode, time.time()),
                )

            conn.execute(
                "UPDATE shards SET cursor=?, pages=pages+1, entries=entries+?, "
                "  n_files=n_files+?, n_dirs=n_dirs+?, bytes=bytes+?, heartbeat_at=?, "
                "  state=CASE WHEN ? THEN state ELSE 'done' END, "
                "  finished_at=CASE WHEN ? THEN finished_at ELSE ? END "
                "WHERE id=?",
                (
                    cursor, len(entries), stats.files, stats.dirs, stats.bytes, time.time(),
                    1 if has_more else 0, 1 if has_more else 0, time.time(), shard_id,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            self._drop_caches()
            raise

        return stats

    def _cached_epoch(self, stamp, cache):
        if not stamp:
            return None
        hit = cache.get(stamp)
        if hit is None:
            hit = _epoch(stamp)
            if len(cache) >= TIME_CACHE_LIMIT:
                cache.clear()
            cache[stamp] = hit
        return hit

    # ---- reporting helpers ----------------------------------------------

    def stats(self) -> dict:
        conn = self.connect()
        files, size = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM files"
        ).fetchone()
        dirs = conn.execute("SELECT COUNT(*) FROM dirs").fetchone()[0]
        done, total = conn.execute(
            "SELECT COALESCE(SUM(state='done'), 0), COUNT(*) FROM shards"
        ).fetchone()
        return {
            "files": files,
            "dirs": dirs,
            "bytes": size,
            "shards_done": done,
            "shards_total": total,
        }

    def progress(self) -> dict:
        """Cheap counters straight off `shards`, for frequent progress logging."""
        conn = self.connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(n_files),0), COALESCE(SUM(n_dirs),0), "
            "       COALESCE(SUM(bytes),0), COALESCE(SUM(pages),0), "
            "       COALESCE(SUM(state='done'),0), COUNT(*), "
            "       COALESCE(SUM(state='error'),0) FROM shards"
        ).fetchone()
        return dict(
            zip(("files", "dirs", "bytes", "pages", "shards_done", "shards_total", "shards_error"), row)
        )

    def build_indexes(self) -> None:
        conn = self.connect()
        for statement in ANALYSIS_INDEXES:
            conn.execute(statement)
        conn.execute("ANALYZE")

    def free_space_bytes(self) -> int:
        stat = os.statvfs(os.path.dirname(os.path.abspath(self.path)) or ".")
        return stat.f_bavail * stat.f_frsize
