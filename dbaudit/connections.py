"""Each thread's SQLite connection, tracked so that its owner can close them all.

Both stores (`dbaudit.store.Store` and `dbaudit.archive.store.ArchiveStore`) give each
thread a connection of its own. A worker closes its own when it finishes (`close`). The
owner closes whatever is still open, on any thread, once its workers are done
(`close_all`).
"""

from __future__ import annotations

import sqlite3
import threading


class ThreadConnections:
    """One SQLite connection per thread, each kept until it is closed.

    `close_all` closes connections that other threads opened, so the stores open them
    with ``check_same_thread=False``. ``sqlite3.threadsafety`` is 3 (serialized), which
    makes that safe.
    """

    def __init__(self) -> None:  # noqa: D107
        self._local = threading.local()
        self._lock = threading.Lock()
        self._open: set[sqlite3.Connection] = set()
        # `close_all` moves this on, so each thread finds out that its connection was
        # closed under it without taking the lock on every `get`.
        self._generation = 0

    def get(self) -> sqlite3.Connection | None:
        """Return the calling thread's connection, unless it has none open.

        Returns
        -------
        sqlite3.Connection or None
            The connection this thread last added, or None if it has added none, or
            `close` or `close_all` has closed it since.
        """
        if getattr(self._local, "generation", None) != self._generation:
            return None
        return self._local.conn

    def add(self, conn: sqlite3.Connection) -> None:
        """Make ``conn`` the calling thread's connection, until it is closed.

        Parameters
        ----------
        conn : sqlite3.Connection
            A connection the calling thread has just opened, with
            ``check_same_thread=False``.
        """
        with self._lock:
            self._open.add(conn)
            self._local.conn = conn
            self._local.generation = self._generation

    def close(self) -> None:
        """Close the calling thread's connection, if it has one open."""
        conn = self.get()
        if conn is None:
            return
        with self._lock:
            self._open.discard(conn)
        self._local.conn = None
        conn.close()

    def close_all(self) -> None:
        """Close every connection still open, whichever thread opened it.

        Call it only once no other thread is using them. A thread's next connection is a
        new one.
        """
        with self._lock:
            conns, self._open = self._open, set()
            self._generation += 1
        for conn in conns:
            conn.close()
