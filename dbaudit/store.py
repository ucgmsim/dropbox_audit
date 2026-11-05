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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DIR_CACHE_LIMIT = 200_000
TIME_CACHE_LIMIT = 50_000

# Built once a pass completes, never during it: a random-order index over 100M
# rows would slow the crawl for no benefit while it runs.
ANALYSIS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS files_dir ON files(dir_id)",
    "CREATE INDEX IF NOT EXISTS files_hash ON files(content_hash)",
    "CREATE INDEX IF NOT EXISTS files_size ON files(size)",
    "CREATE INDEX IF NOT EXISTS files_smod ON files(server_modified)",
    "CREATE INDEX IF NOT EXISTS files_ext ON files(ext)",
    "CREATE INDEX IF NOT EXISTS dirs_parent ON dirs(parent_id)",
    "CREATE INDEX IF NOT EXISTS dirs_pathlower ON dirs(path_lower)",
]


@dataclass
class Shard:
    """One row of the shards table: a subtree listed under one cursor.

    Parameters
    ----------
    id : int
        The shard's id.
    path : str
        The Dropbox folder the shard lists.
    depth : int
        Depth below the crawl root; the root shard is 0.
    mode : str
        ``"recursive"`` for one listing of the whole subtree, ``"split"`` for a
        one-level listing whose subfolders become shards of their own.
    state : str
        ``"pending"``, ``"running"``, ``"done"`` or ``"error"``.
    cursor : str or None
        The cursor of the last committed page; None before the first page.
    pages : int
        Pages committed so far.
    entries : int
        Entries committed so far.
    """

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
    """What one committed page added.

    Parameters
    ----------
    files : int
        Files written.
    dirs : int
        Directory rows newly inserted, ancestors created from child paths included.
    bytes : int
        Total size of the files, in bytes.
    """

    files: int = 0
    dirs: int = 0
    bytes: int = 0


def _epoch(stamp: str | None) -> int | None:
    """Parse Dropbox's ``2020-01-02T03:04:05Z`` stamps without strptime overhead.

    Parameters
    ----------
    stamp : str or None
        A stamp such as ``"2020-01-02T03:04:05Z"``; None or empty is accepted.

    Returns
    -------
    int or None
        Seconds since the Unix epoch, UTC; None if `stamp` is empty or malformed.
    """
    if not stamp:
        return None
    try:
        return calendar.timegm(
            (
                int(stamp[0:4]),
                int(stamp[5:7]),
                int(stamp[8:10]),
                int(stamp[11:13]),
                int(stamp[14:16]),
                int(stamp[17:19]),
                0,
                0,
                0,
            )
        )
    except (ValueError, IndexError):
        return None


def _depth_of(path: str) -> int:
    """Return the number of folders above a path.

    Parameters
    ----------
    path : str
        A folder path, with or without a trailing slash.

    Returns
    -------
    int
        The count of slashes in `path` once a trailing slash is dropped; the
        top-level folder ``/x`` is 1.
    """
    return path.rstrip("/").count("/")


MAX_EXT_LEN = 12


def _extension(name: str) -> str:
    """Lowercased extension, or '' -- stored so the type profile is a plain GROUP BY.

    Bounded in length so that dotted filenames (``run.2024-01-01.backup.tar``) do not
    turn arbitrary text into a pseudo-extension.

    Parameters
    ----------
    name : str
        A file name.

    Returns
    -------
    str
        The lowercased extension without its dot, or ``""`` when there is none or it
        is longer than ``MAX_EXT_LEN``.
    """
    dot = name.rfind(".")
    if dot <= 0 or dot == len(name) - 1 or len(name) - dot - 1 > MAX_EXT_LEN:
        return ""
    return name[dot + 1 :].lower()


class Store:
    """The audit's SQLite database, with one connection per thread.

    Parameters
    ----------
    path : str or os.PathLike
        The database file. It is created on the first connection.
    """

    def __init__(self, path: str | os.PathLike[str]):  # noqa: D107
        self.path = str(path)
        self._local = threading.local()
        self._meta_lock = threading.Lock()

    # ---- connections ---------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """One connection per thread. WAL lets readers run while a worker writes.

        Returns
        -------
        sqlite3.Connection
            This thread's connection, opened on first use. It is in autocommit mode
            (``isolation_level`` None), so callers open transactions themselves.
        """
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
        """Close this thread's connection, if it has one."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _caches(
        self,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int | None]]:
        """Return the calling thread's lookup caches, opening the connection if needed.

        Returns
        -------
        tuple of dict
            The directory cache (lowercased path to dir id), the principal cache (Dropbox
            account id to principal id) and the time cache (stamp to Unix seconds, or
            None for a malformed stamp).
        """
        self.connect()
        return (
            self._local.dir_cache,
            self._local.principal_cache,
            self._local.time_cache,
        )

    def _drop_caches(self) -> None:
        """Clear the caches after a rollback: cached ids may reference vanished rows."""
        self._local.dir_cache = {}
        self._local.principal_cache = {}

    # ---- schema and metadata -------------------------------------------

    def init_schema(self) -> None:
        """Create the schema from ``schema.sql``, leaving tables that exist as they are."""
        conn = self.connect()
        conn.executescript(SCHEMA_PATH.read_text())

    def set_meta(self, key: str, value: object) -> None:
        """Set a metadata value, replacing any earlier one.

        Parameters
        ----------
        key : str
            The metadata key.
        value : object
            The value; it is stored as ``str(value)``.
        """
        with self._meta_lock:
            self.connect().execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def is_initialised(self) -> bool:
        """Return whether `init` has created the schema; probing must not raise.

        Returns
        -------
        bool
            Whether the database file exists and holds the ``meta`` table.
        """
        if not os.path.exists(self.path):
            return False
        row = (
            self.connect()
            .execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
            )
            .fetchone()
        )
        return row is not None

    @overload
    def get_meta(self, key: str, default: None = None) -> str | None: ...

    @overload
    def get_meta(self, key: str, default: str) -> str: ...

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """Read a metadata value.

        Parameters
        ----------
        key : str
            The metadata key.
        default : str, optional
            Returned when the key is absent or the schema does not exist yet; None by
            default.

        Returns
        -------
        str or None
            The stored text, or `default`.
        """
        try:
            row = (
                self.connect()
                .execute("SELECT value FROM meta WHERE key=?", (key,))
                .fetchone()
            )
        except sqlite3.OperationalError:
            return default  # schema not created yet
        return default if row is None else row[0]

    def query(self, sql: str, params: Sequence[object] = ()) -> list[tuple]:
        """Run a read query on this thread's connection.

        Parameters
        ----------
        sql : str
            The SQL text, with ``?`` placeholders.
        params : sequence of object, optional
            The values for the placeholders.

        Returns
        -------
        list of tuple
            Every row.
        """
        return self.connect().execute(sql, params).fetchall()

    def log_event(self, kind: str, detail: str = "") -> None:
        """Record an event in ``api_events`` with the current time.

        Parameters
        ----------
        kind : str
            The event's kind, such as ``"rate_limited"``.
        detail : str, optional
            More about it; cut to 500 characters.
        """
        self.connect().execute(
            "INSERT INTO api_events(ts, kind, detail) VALUES(?, ?, ?)",
            (time.time(), kind, detail[:500]),
        )

    # ---- shards ---------------------------------------------------------

    def add_shard(self, path: str, depth: int, mode: str = "recursive") -> int:
        """Add a pending shard for a folder, unless one exists for that path.

        Parameters
        ----------
        path : str
            The Dropbox folder.
        depth : int
            Depth below the crawl root.
        mode : str, optional
            ``"recursive"`` (the default) or ``"split"``.

        Returns
        -------
        int
            The id of the shard for `path`, new or existing.
        """
        conn = self.connect()
        conn.execute(
            "INSERT OR IGNORE INTO shards(path, depth, mode, state, created_at) "
            "VALUES(?, ?, ?, 'pending', ?)",
            (path, depth, mode, time.time()),
        )
        return conn.execute("SELECT id FROM shards WHERE path=?", (path,)).fetchone()[0]

    def get_shard(self, shard_id: int) -> Shard | None:
        """Read one shard.

        Parameters
        ----------
        shard_id : int
            The shard's id.

        Returns
        -------
        Shard or None
            The shard, or None if there is none with that id.
        """
        row = (
            self.connect()
            .execute(
                "SELECT id, path, depth, mode, state, cursor, pages, entries FROM shards WHERE id=?",
                (shard_id,),
            )
            .fetchone()
        )
        return Shard(*row) if row else None

    def claim_shard(self, owner: str) -> Shard | None:
        """Atomically take the shallowest pending shard, so progress stays breadth-first.

        Parameters
        ----------
        owner : str
            The name of the claiming worker, stored on the shard.

        Returns
        -------
        Shard or None
            The shard, now running with its attempts incremented, or None if none is
            pending.
        """
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
        """Mark a shard done.

        Parameters
        ----------
        shard_id : int
            The shard's id.
        note : str, optional
            A note to keep on the shard; None leaves any earlier note.
        """
        self.connect().execute(
            "UPDATE shards SET state='done', finished_at=?, owner=NULL, note=COALESCE(?, note) "
            "WHERE id=?",
            (time.time(), note, shard_id),
        )

    def fail_shard(self, shard_id: int, error: str) -> None:
        """Mark a shard as ended in error.

        Parameters
        ----------
        shard_id : int
            The shard's id.
        error : str
            Why it failed; cut to 1000 characters.
        """
        self.connect().execute(
            "UPDATE shards SET state='error', finished_at=?, owner=NULL, error=? WHERE id=?",
            (time.time(), error[:1000], shard_id),
        )

    def release_shard(self, shard_id: int) -> None:
        """Hand an unfinished shard back to the queue, keeping its cursor.

        Parameters
        ----------
        shard_id : int
            The shard's id. A shard that is not running is left as it is.
        """
        self.connect().execute(
            "UPDATE shards SET state='pending', owner=NULL WHERE id=? AND state='running'",
            (shard_id,),
        )

    def set_shard_mode(self, shard_id: int, mode: str) -> None:
        """Persist the recursive/split choice *before* the first page.

        A cursor belongs to the kind of listing that produced it, so a resumed shard
        must continue in the same mode. Persisting the decision is what stops a
        restart from resuming a non-recursive cursor as a recursive one.

        Parameters
        ----------
        shard_id : int
            The shard's id.
        mode : str
            ``"recursive"`` or ``"split"``.
        """
        self.connect().execute("UPDATE shards SET mode=? WHERE id=?", (mode, shard_id))

    def clear_cursor(self, shard_id: int) -> None:
        """After a cursor reset: forget the cursor and drop the rows it produced.

        `files.dbx_id` is UNIQUE, so a re-list would update rather than duplicate --
        but rows for entries deleted since the reset would linger, so clear them.

        Directory rows are deliberately left alone. A directory may have been created
        as an ancestor of another shard's files, and deleting it here would orphan
        their dir_id. Re-listing upserts directories anyway -- and inserts none of
        these, so `n_dirs` keeps counting them.

        Parameters
        ----------
        shard_id : int
            The shard's id.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM files WHERE shard_id=?", (shard_id,))
            conn.execute(
                "UPDATE shards SET cursor=NULL, pages=0, entries=0, n_files=0, "
                "bytes=0 WHERE id=?",
                (shard_id,),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            self._drop_caches()

    def reset_stale_shards(self) -> int:
        """Reclaim shards owned by a process that died. Cursors are preserved.

        Returns
        -------
        int
            The number of shards put back to pending.
        """
        cur = self.connect().execute(
            "UPDATE shards SET state='pending', owner=NULL WHERE state='running'"
        )
        return cur.rowcount

    def retry_failed_shards(self) -> int:
        """Queue every shard that ended in error again, from scratch.

        Its cursor is forgotten and its rows dropped, as after a cursor reset.
        Resuming would repeat the request that kept failing; starting over lets the
        crawler split a subtree that was too much for one listing.

        Returns
        -------
        int
            The number of shards queued again.
        """
        # A shard that never committed a page has no rows to drop and no cursor, so it
        # is spared clear_cursor's scan of `files`, which has no index on shard_id.
        listed = [
            row[0]
            for row in self.connect()
            .execute("SELECT id FROM shards WHERE state='error' AND pages > 0")
            .fetchall()
        ]
        for shard_id in listed:
            self.clear_cursor(shard_id)
        return (
            self.connect()
            .execute(
                "UPDATE shards SET state='pending', owner=NULL, error=NULL, finished_at=NULL "
                "WHERE state='error'"
            )
            .rowcount
        )

    def pending_count(self) -> int:
        """Count the shards waiting to be claimed.

        Returns
        -------
        int
            The number of pending shards.
        """
        return (
            self.connect()
            .execute("SELECT COUNT(*) FROM shards WHERE state='pending'")
            .fetchone()[0]
        )

    def running_count(self) -> int:
        """Count the shards a worker is listing.

        Returns
        -------
        int
            The number of running shards.
        """
        return (
            self.connect()
            .execute("SELECT COUNT(*) FROM shards WHERE state='running'")
            .fetchone()[0]
        )

    def shard_count(self) -> int:
        """Count all shards, whatever their state.

        Returns
        -------
        int
            The number of shards.
        """
        return self.connect().execute("SELECT COUNT(*) FROM shards").fetchone()[0]

    # ---- the hot path ---------------------------------------------------

    def _principal_id(
        self, conn: sqlite3.Connection, cache: dict[str, int], account_id: str | None
    ) -> int | None:
        """Return the id of the principal for a Dropbox account, creating the row if needed.

        Parameters
        ----------
        conn : sqlite3.Connection
            The connection inside the page's transaction.
        cache : dict of str to int
            The principal cache, updated here.
        account_id : str or None
            The Dropbox account id; None or empty means no principal.

        Returns
        -------
        int or None
            The principal's id, or None when there is no account id.
        """
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

    def ensure_dir(
        self,
        conn: sqlite3.Connection,
        path_display: str,
        cache: dict[str, int],
        shard_id: int | None = None,
        created: list[int] | None = None,
    ) -> int:
        """Return the id of ``path_display``, creating it and any missing ancestors.

        Called with the parent path of every file, so it must be cheap; the cache
        makes it roughly free within a page, since recursive listings are local.

        ``created`` is a one-element list used as a counter of rows actually inserted.
        Counting insertions rather than folder entries is what keeps SUM(shards.n_dirs)
        equal to COUNT(*) FROM dirs: a recursive listing repeats its own root, and
        ancestors are created from child paths without ever appearing as an entry.

        Parameters
        ----------
        conn : sqlite3.Connection
            The connection inside the page's transaction.
        path_display : str
            The folder's path as Dropbox displays it; a trailing slash is ignored.
        cache : dict of str to int
            The directory cache, keyed by lowercased path and updated here.
        shard_id : int, optional
            The shard recorded on directories created here; None records none.
        created : list of int, optional
            One-element counter, incremented for each row inserted; None counts nothing.

        Returns
        -------
        int
            The folder's dir id.
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
            "                           shard_id) VALUES(?, ?, ?, ?, ?, ?)",
            (parent_id, name, path_display, key, _depth_of(path_display), shard_id),
        )
        if created is not None and conn.total_changes > before:
            created[0] += 1
        dir_id = conn.execute(
            "SELECT id FROM dirs WHERE path_lower=?", (key,)
        ).fetchone()[0]
        if len(cache) >= DIR_CACHE_LIMIT:
            cache.clear()
        cache[key] = dir_id
        return dir_id

    def commit_page(
        self,
        shard_id: int,
        entries: list[dict],
        cursor: str | None,
        has_more: bool,
        child_shards: Iterable[tuple[str, int, str]] = (),
        shard_path: str = "",
    ) -> PageStats:
        """Write one page and its cursor in a single transaction.

        Anything raised in here rolls the whole page back, cursor included, so the
        page is simply re-fetched on restart.

        Parameters
        ----------
        shard_id : int
            The shard the page belongs to.
        entries : list of dict
            The page's entries as Dropbox returned them (``.tag`` of ``file``, ``folder``
            or ``deleted``).
        cursor : str or None
            The cursor that follows this page, stored with it.
        has_more : bool
            Whether Dropbox has more pages; if not, the shard is marked done.
        child_shards : iterable of tuple, optional
            ``(path, depth, mode)`` of shards to queue in the same transaction.
        shard_path : str, optional
            The shard's folder, under which an entry with no path is placed.

        Returns
        -------
        PageStats
            What the page added.
        """
        conn = self.connect()
        dir_cache, principal_cache, time_cache = self._caches()
        stats = PageStats()
        created_dirs = [0]

        try:
            conn.execute("BEGIN IMMEDIATE")

            file_rows = []
            for entry in entries:
                tag = entry.get(".tag")
                path_display = entry.get("path_display")
                if not path_display:
                    # Rare, but seen with some shared-folder entries. Place it under
                    # the shard root rather than dropping it, and leave a trace.
                    name = entry.get("name") or "<unnamed>"
                    path_display = (
                        f"{shard_path.rstrip('/')}/{name}" if shard_path else f"/{name}"
                    )
                    conn.execute(
                        "INSERT INTO api_events(ts, kind, detail) VALUES(?, 'no_path_display', ?)",
                        (time.time(), path_display[:500]),
                    )

                if tag == "file":
                    parent = path_display.rsplit("/", 1)[0] or "/"
                    dir_id = self.ensure_dir(
                        conn, parent, dir_cache, shard_id, created_dirs
                    )
                    size = int(entry["size"])
                    content_hash = entry.get("content_hash")
                    sharing = entry.get("sharing_info") or {}
                    file_name = entry.get("name") or path_display.rsplit("/", 1)[-1]
                    file_rows.append(
                        (
                            entry.get("id"),
                            dir_id,
                            file_name,
                            _extension(file_name),
                            size,
                            self._hash_bytes(conn, content_hash, path_display),
                            entry.get("rev"),
                            self._cached_epoch(
                                entry.get("client_modified"), time_cache
                            ),
                            self._cached_epoch(
                                entry.get("server_modified"), time_cache
                            ),
                            self._principal_id(
                                conn, principal_cache, sharing.get("modified_by")
                            ),
                            1 if entry.get("is_downloadable", True) else 0,
                            shard_id,
                        )
                    )
                    stats.files += 1
                    stats.bytes += size

                elif tag == "folder":
                    dir_id = self.ensure_dir(
                        conn, path_display, dir_cache, shard_id, created_dirs
                    )
                    sharing = entry.get("sharing_info") or {}
                    shared_folder_id = entry.get("shared_folder_id")
                    conn.execute(
                        "UPDATE dirs SET dbx_id=?, shared_folder_id=?, parent_shared_folder_id=?, "
                        "  is_mount=? WHERE id=?",
                        (
                            entry.get("id"),
                            shared_folder_id,
                            entry.get("parent_shared_folder_id")
                            or sharing.get("parent_shared_folder_id"),
                            1 if shared_folder_id else 0,
                            dir_id,
                        ),
                    )

                elif tag == "deleted":
                    # A crawl asks for no deleted entries, so one arriving names
                    # something that went while the crawl ran -- which a snapshot taken
                    # over hours cannot be exact about either way. Keep the rows and
                    # leave a trace.
                    conn.execute(
                        "INSERT INTO api_events(ts, kind, detail) VALUES(?, 'deleted_entry', ?)",
                        (time.time(), path_display[:500]),
                    )

            stats.dirs = created_dirs[0]

            if file_rows:
                conn.executemany(
                    "INSERT INTO files(dbx_id, dir_id, name, ext, size, content_hash, rev, "
                    "  client_modified, server_modified, modified_by, is_downloadable, "
                    "  shard_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(dbx_id) DO UPDATE SET "
                    "  dir_id=excluded.dir_id, name=excluded.name, ext=excluded.ext, "
                    "  size=excluded.size, "
                    "  content_hash=excluded.content_hash, rev=excluded.rev, "
                    "  client_modified=excluded.client_modified, "
                    "  server_modified=excluded.server_modified, "
                    "  modified_by=excluded.modified_by, shard_id=excluded.shard_id",
                    file_rows,
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
                    cursor,
                    len(entries),
                    stats.files,
                    stats.dirs,
                    stats.bytes,
                    time.time(),
                    1 if has_more else 0,
                    1 if has_more else 0,
                    time.time(),
                    shard_id,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            self._drop_caches()
            raise

        return stats

    @staticmethod
    def _hash_bytes(
        conn: sqlite3.Connection, content_hash: str | None, path_display: str
    ) -> bytes | None:
        """Never let one malformed hash cost us a whole shard.

        Dropbox always returns 64 hex chars, but a file recorded without its hash is
        a far better outcome than a subtree that fails to crawl.

        Parameters
        ----------
        conn : sqlite3.Connection
            The connection inside the page's transaction, used to log a bad hash.
        content_hash : str or None
            Dropbox's content hash as hex text.
        path_display : str
            The file's path, for the log entry.

        Returns
        -------
        bytes or None
            The hash as bytes, or None if it is absent or not valid hex.
        """
        if not content_hash:
            return None
        try:
            return bytes.fromhex(content_hash)
        except (ValueError, TypeError):
            conn.execute(
                "INSERT INTO api_events(ts, kind, detail) VALUES(?, 'bad_content_hash', ?)",
                (time.time(), path_display[:500]),
            )
            return None

    def _cached_epoch(
        self, stamp: str | None, cache: dict[str, int | None]
    ) -> int | None:
        """Parse a Dropbox stamp, remembering the result.

        Parameters
        ----------
        stamp : str or None
            A stamp such as ``"2020-01-02T03:04:05Z"``.
        cache : dict of str to int or None
            The time cache, updated here and cleared when it is full.

        Returns
        -------
        int or None
            Seconds since the Unix epoch, UTC; None for an absent or malformed stamp.
        """
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
        """Count what the store holds.

        Returns
        -------
        dict
            ``files``, ``dirs``, ``bytes``, ``shards_done`` and ``shards_total``.
        """
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
        """Cheap counters straight off `shards`, for frequent progress logging.

        Returns
        -------
        dict
            ``files``, ``dirs``, ``bytes``, ``pages``, ``shards_done``, ``shards_total``
            and ``shards_error``, summed over the shards.
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(n_files),0), COALESCE(SUM(n_dirs),0), "
            "       COALESCE(SUM(bytes),0), COALESCE(SUM(pages),0), "
            "       COALESCE(SUM(state='done'),0), COUNT(*), "
            "       COALESCE(SUM(state='error'),0) FROM shards"
        ).fetchone()
        return dict(
            zip(
                (
                    "files",
                    "dirs",
                    "bytes",
                    "pages",
                    "shards_done",
                    "shards_total",
                    "shards_error",
                ),
                row,
            )
        )

    def build_indexes(self) -> None:
        """Create the analysis indexes and run ANALYZE; call it once a pass completes."""
        conn = self.connect()
        for statement in ANALYSIS_INDEXES:
            conn.execute(statement)
        conn.execute("ANALYZE")

    def free_space_bytes(self) -> int:
        """Measure the free space where the database lives.

        Returns
        -------
        int
            Bytes available to an unprivileged process on that file system.
        """
        stat = os.statvfs(os.path.dirname(os.path.abspath(self.path)) or ".")
        return stat.f_bavail * stat.f_frsize
