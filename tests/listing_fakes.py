"""Tiny audit listings and archive indexes for the kinds tests, with stand-in names only."""

import calendar
import hashlib
from pathlib import Path

from dbaudit.archive.store import ArchiveStore
from dbaudit.store import Store
from tests.stores import close_after_test

LISTED = calendar.timegm((2026, 10, 1, 0, 0, 0))


def content_hash(content: str) -> str:
    """The content hash `file` gives a file holding `content`."""
    return hashlib.sha256(content.encode()).hexdigest()


def file(
    path: str,
    size: int,
    content: str | None = None,
    modified: str = "2026-01-01T00:00:00Z",
) -> dict:
    """A file as Dropbox lists it. Files with the same `content` are copies."""
    return {
        ".tag": "file",
        "id": f"id:{path}",
        "name": path.rsplit("/", 1)[1],
        "path_display": path,
        "size": size,
        "rev": "r",
        "content_hash": content_hash(content or path),
        "client_modified": modified,
        "server_modified": modified,
    }


def folder(path: str) -> dict:
    return {
        ".tag": "folder",
        "id": f"id:{path}",
        "name": path.rsplit("/", 1)[1],
        "path_display": path,
    }


def listing(tmp_path: Path, *entries: dict, finished: bool = True) -> Store:
    """An audit database of `entries` under the root /R, its crawl finished at LISTED."""
    store = close_after_test(Store(tmp_path / "audit.db"))
    store.init_schema()
    store.set_meta("root", "/R")
    if finished:
        store.set_meta("last_pass_completed_at", LISTED)
    shard = store.add_shard("/R", 1)
    store.claim_shard("test")
    store.commit_page(shard, [folder("/R"), *entries], cursor=None, has_more=False)
    return store


def dir_of(store: Store, path: str) -> int:
    return store.query("SELECT id FROM dirs WHERE path_lower = ?", (path.lower(),))[0][
        0
    ]


def archive(
    name: str,
    folder: str,
    parts: list[tuple[str, int, str]],
    members: list[tuple],
    state: str = "complete",
) -> dict:
    """An archive as the index holds it. `parts` holds (name, size, content) and `members`
    (dir, name, size, mtime), with a fifth item for a type other than a regular file."""
    return {
        "name": name,
        "folder": folder,
        "parts": parts,
        "members": members,
        "state": state,
    }


def archive_index(tmp_path: Path, *archives: dict, name: str = "archives.db") -> str:
    """An archive index database holding `archives`, flushed and closed."""
    path = str(tmp_path / name)
    store = ArchiveStore(path)
    store.init_schema()
    conn = store.connect()
    for number, held in enumerate(archives):
        archive_id = conn.execute(
            "INSERT INTO archives (name, folder, total_size, n_parts, set_hash, state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                held["name"],
                held["folder"],
                sum(size for _, size, _ in held["parts"]),
                len(held["parts"]),
                f"set-{number}",
                held["state"],
            ),
        ).lastrowid
        offset = 0
        for idx, (part, size, content) in enumerate(held["parts"]):
            conn.execute(
                "INSERT INTO parts (archive_id, idx, name, size, offset, "
                "content_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (archive_id, idx, part, size, offset, content_hash(content)),
            )
            offset += size
        for i, (where, member, size, mtime, *kind) in enumerate(held["members"]):
            conn.execute(
                "INSERT INTO members (archive_id, hdr_offset, data_offset, size, "
                "type, mtime, dir, name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    archive_id,
                    512 * i,
                    512 * i + 512,
                    size,
                    kind[0] if kind else "0",
                    mtime,
                    where,
                    member,
                ),
            )
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.close()
    return path
