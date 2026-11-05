"""Stores a test keeps open until it ends.

`close_after_test` hands back the store it is given and remembers it. The autouse
fixture in conftest.py closes every remembered store when the test ends, so a helper
can return an open store and its callers need not close it. Stores the code under test
opens are never handed to `close_after_test`: those it must close itself.
"""

from dbaudit.archive.store import ArchiveStore
from dbaudit.store import Store

_opened: list[Store | ArchiveStore] = []


def close_after_test[S: Store | ArchiveStore](store: S) -> S:
    """Return ``store``, to be closed when the running test ends."""
    _opened.append(store)
    return store


def close_opened() -> None:
    """Close every store handed to `close_after_test` since the last call."""
    while _opened:
        _opened.pop().close_all()
