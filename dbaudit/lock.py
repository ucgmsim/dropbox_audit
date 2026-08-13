"""Single-instance lock.

Two crawlers sharing one database would each reclaim the other's in-flight shards
as "stale", so the crawler takes an exclusive flock for the life of a run.
"""

from __future__ import annotations

import fcntl
import os
import socket


class LockHeld(RuntimeError):
    """Another process is already crawling this database."""


class InstanceLock:
    def __init__(self, path):
        self.path = str(path)
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.seek(0)
            holder = self._fh.read().strip() or "unknown process"
            self._fh.close()
            self._fh = None
            raise LockHeld(f"{self.path} is held by {holder}") from None
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{os.getpid()}@{socket.gethostname()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False
