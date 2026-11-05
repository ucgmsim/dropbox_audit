"""Independent verification of the crawl, using rclone as a second opinion.

The crawl and rclone reach Dropbox by different routes -- one recursive listing
per subtree versus one call per directory -- so agreement between them is real
evidence rather than a tautology. This is what lets you keep trusting a crawl that
ran unattended for hours.

rclone is deliberately run at ``--checkers 4``. At 32 it earned a 300-second
account-wide rate-limit penalty during design, which would stall the crawler too.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from .store import Store

VERIFY_CHECKERS = 4


@dataclass
class VerifyResult:
    """How the audit database and an rclone listing of one subtree differ.

    Parameters
    ----------
    only_in_db : list of str
        Paths relative to the subtree that the database has and rclone does not.
    only_in_rclone : list of str
        Paths that rclone lists and the database lacks.
    size_mismatches : list of tuple
        ``(path, database size, rclone size)`` for paths in both whose sizes differ,
        in bytes.
    db_files : int
        Files the database holds under the subtree.
    rclone_files : int
        Files rclone listed.
    db_bytes : int
        Total size of the database's files, in bytes.
    rclone_bytes : int
        Total size of rclone's files, in bytes.
    """

    only_in_db: list[str] = field(default_factory=list)
    only_in_rclone: list[str] = field(default_factory=list)
    size_mismatches: list[tuple] = field(default_factory=list)
    db_files: int = 0
    rclone_files: int = 0
    db_bytes: int = 0
    rclone_bytes: int = 0

    @property
    def differences(self) -> int:
        """Count of paths that differ in any way: missing on either side or of different size.

        Returns
        -------
        int
            Zero when the two agree.
        """
        return (
            len(self.only_in_db) + len(self.only_in_rclone) + len(self.size_mismatches)
        )

    @property
    def ok(self) -> bool:
        """Whether the database and rclone agree exactly.

        Returns
        -------
        bool
            True if there are no differences.
        """
        return self.differences == 0


def _run_rclone(remote: str, path: str) -> str:
    """List a subtree's files with rclone.

    Parameters
    ----------
    remote : str
        The rclone remote's name.
    path : str
        The subtree, starting with ``/``.

    Returns
    -------
    str
        The ``lsjson`` output of rclone.

    Raises
    ------
    RuntimeError
        rclone exited with an error.
    """
    proc = subprocess.run(
        [
            "rclone",
            "lsjson",
            "-R",
            "--files-only",
            "--checkers",
            str(VERIFY_CHECKERS),
            f"{remote}:{path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def verify_subtree(
    store: Store,
    path: str,
    rclone_fn: Callable[[str], str] | None = None,
    remote: str | None = None,
) -> VerifyResult:
    """Compare the audit database with an independent rclone listing of one subtree.

    Parameters
    ----------
    store : Store
        The audit database.
    path : str
        The subtree to compare, as a Dropbox path.
    rclone_fn : callable, optional
        Takes the path and returns rclone's ``lsjson`` output; the default runs rclone.
        Tests pass a stand-in.
    remote : str, optional
        The rclone remote's name. Defaults to the remote the database was crawled from.

    Returns
    -------
    VerifyResult
        What differs between the two.
    """
    path = "/" + path.strip("/")
    remote = remote or store.get_meta("remote", "dropbox")
    rclone_fn = rclone_fn or (lambda p: _run_rclone(remote, p))

    prefix = path.lower() + "/"
    rows = store.query(
        "SELECT dirs.path_display || '/' || files.name, files.size "
        "FROM files JOIN dirs ON dirs.id = files.dir_id "
        "WHERE lower(dirs.path_display) = ? OR lower(dirs.path_display) LIKE ?",
        (path.lower(), prefix + "%"),
    )
    db = {full[len(path) + 1 :]: size for full, size in rows}

    listing = json.loads(rclone_fn(path) or "[]")
    remote_files = {entry["Path"]: entry["Size"] for entry in listing}

    db_keys, remote_keys = set(db), set(remote_files)
    return VerifyResult(
        only_in_db=sorted(db_keys - remote_keys),
        only_in_rclone=sorted(remote_keys - db_keys),
        size_mismatches=sorted(
            (name, db[name], remote_files[name])
            for name in db_keys & remote_keys
            if db[name] != remote_files[name]
        ),
        db_files=len(db),
        rclone_files=len(remote_files),
        db_bytes=sum(db.values()),
        rclone_bytes=sum(remote_files.values()),
    )
