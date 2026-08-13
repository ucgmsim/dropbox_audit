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
from dataclasses import dataclass, field

VERIFY_CHECKERS = 4


@dataclass
class VerifyResult:
    only_in_db: list[str] = field(default_factory=list)
    only_in_rclone: list[str] = field(default_factory=list)
    size_mismatches: list[tuple] = field(default_factory=list)
    db_files: int = 0
    rclone_files: int = 0
    db_bytes: int = 0
    rclone_bytes: int = 0

    @property
    def differences(self) -> int:
        return len(self.only_in_db) + len(self.only_in_rclone) + len(self.size_mismatches)

    @property
    def ok(self) -> bool:
        return self.differences == 0


def _run_rclone(remote: str, path: str) -> str:
    proc = subprocess.run(
        ["rclone", "lsjson", "-R", "--files-only", "--checkers", str(VERIFY_CHECKERS),
         f"{remote}:{path}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def verify_subtree(store, path: str, rclone_fn=None, remote: str | None = None) -> VerifyResult:
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
    db = {full[len(path) + 1:]: size for full, size in rows}

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
