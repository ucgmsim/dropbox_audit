"""Single-instance lock.

Two crawlers sharing one database would each reclaim the other's in-flight shards
as "stale", so the crawler takes an exclusive flock for the life of a run.
"""

from __future__ import annotations

import fcntl
import os
import socket


class LockHeldError(RuntimeError):
    """Another process is already crawling this database."""


class InstanceLock:
    """An exclusive, advisory lock on a file, held while the ``with`` block runs.

    The file then holds ``pid@hostname`` of the holder, so a refused process can say
    who has it.

    Parameters
    ----------
    path : str or os.PathLike
        The lock file. Created if it does not exist.
    """

    def __init__(self, path: str | os.PathLike[str]):  # noqa: D107
        self.path = str(path)
        self._fh = None

    def __enter__(self):
        """Take the lock without waiting and record the holder in the file.

        Returns
        -------
        InstanceLock
            This lock.

        Raises
        ------
        LockHeldError
            Another process holds the lock; the message names it.
        """
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.seek(0)
            holder = self._fh.read().strip() or "unknown process"
            self._fh.close()
            self._fh = None
            raise LockHeldError(f"{self.path} is held by {holder}") from None
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{os.getpid()}@{socket.gethostname()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        """Release the lock.

        Parameters
        ----------
        *exc : object
            The exception type, value and traceback, if the block raised; unused.

        Returns
        -------
        bool
            Always False, so an exception from the block propagates.
        """
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False
