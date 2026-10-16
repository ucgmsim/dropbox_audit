"""Every tar archive an audit database knows of, as `archive register` would see it.

A candidate is a folder and a base name: a file named `x.tar`, or a set of parts named
`x.tar.aa`, `x.tar.part00`, `x.tar.00` and so on. Each is checked the way `register`
checks it -- `ArchiveSet.from_entries` over every file in the folder whose name starts
with the base name -- so a set `register` would refuse is reported, not attempted. The
same content (`set_hash`) at another path is a copy: it is walked once, and the audit
database already says where every copy lives. Read-only.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .parts import ArchiveSet, ArchiveSetError

PART = re.compile(r"^(?P<base>.+\.tar)\.(?:part[-_]?)?(?P<suffix>[a-z]{2,}|\d{2,})$")
# A compressed tar cannot be walked header by header, and `gz`, `xz` or `zst` would
# otherwise parse as a part suffix.
COMPRESSED = re.compile(r"\.(tgz|tar\.(gz|bz2|xz|zst|zstd|lz4|lzma|lz|z|br))$", re.I)


@dataclass(frozen=True)
class Candidate:
    folder: str
    name: str
    size: int
    n_parts: int
    set_hash: str           # "" for a refused set
    status: str             # ok | copy | refused
    detail: str = ""


@dataclass(frozen=True)
class Listing:
    candidates: list        # Candidate, largest first
    compressed: tuple       # (files, bytes) skipped by their compressed names
    other: tuple            # (files, bytes) named like `x.tar.<something>` that are neither


def find_candidates(audit_db) -> Listing:
    """Every candidate in the audit database at ``audit_db``, largest first."""
    # As a URI, so that a ?, # or % in the path is escaped rather than read as syntax.
    audit = sqlite3.connect(Path(audit_db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return _find(audit)
    finally:
        audit.close()


def _find(audit) -> Listing:
    found, compressed, other = {}, [0, 0], [0, 0]
    for dir_id, folder, name, size in audit.execute(
            "SELECT f.dir_id, d.path_display, f.name, f.size FROM files f "
            "JOIN dirs d ON d.id = f.dir_id "
            "WHERE f.name LIKE '%.tar' OR f.name LIKE '%.tar.%' OR f.name LIKE '%.tgz'"):
        part = PART.match(name)
        if COMPRESSED.search(name):
            compressed[0] += 1
            compressed[1] += size
        elif name.endswith(".tar"):
            found[(dir_id, name)] = folder
        elif part:
            found[(dir_id, part.group("base"))] = folder
        else:
            other[0] += 1
            other[1] += size

    rows = []
    for (dir_id, base), folder in found.items():
        entries = [
            {"name": name, "size": size, "path_display": f"{folder}/{name}",
             "id": dbx_id or "", "rev": rev or "",
             "content_hash": digest.hex() if digest else ""}
            for name, size, dbx_id, rev, digest in audit.execute(
                "SELECT name, size, dbx_id, rev, content_hash FROM files "
                "WHERE dir_id = ? AND substr(name, 1, ?) = ?", (dir_id, len(base), base))]
        total = sum(e["size"] for e in entries)
        # One part per folder, each folder named after its part: `register` reads one
        # folder, so the first part alone would pass as a whole archive and be walked to
        # the end of that part only to be called truncated.
        holder = PART.match(folder.rsplit("/", 1)[-1])
        if holder and holder.group("base") == base:
            rows.append(Candidate(folder, base, total, len(entries), "", "refused",
                                  "its parts are in separate folders"))
            continue
        try:
            archive_set = ArchiveSet.from_entries(entries, base)
        except ArchiveSetError as exc:
            rows.append(Candidate(folder, base, total, len(entries), "", "refused", str(exc)))
            continue
        rows.append(Candidate(folder, base, archive_set.total_size, len(archive_set.parts),
                              archive_set.set_hash(), "ok"))

    rows.sort(key=lambda c: (-c.size, c.folder, c.name))
    first, listed = {}, []
    for candidate in rows:
        if candidate.status == "ok" and candidate.set_hash in first:
            candidate = Candidate(candidate.folder, candidate.name, candidate.size,
                                  candidate.n_parts, candidate.set_hash, "copy",
                                  f"copy of {first[candidate.set_hash]}")
        elif candidate.status == "ok":
            first[candidate.set_hash] = f"{candidate.folder}/{candidate.name}"
        listed.append(candidate)
    return Listing(listed, tuple(compressed), tuple(other))
