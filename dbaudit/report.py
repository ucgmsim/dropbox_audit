"""Reports aimed at one question: where could this storage be reduced?

Each report answers a different reduction strategy -- delete duplicates, archive cold
data, clean up a directory full of junk, or purge version history -- so they are
separate functions returning plain rows rather than one giant summary.
"""

from __future__ import annotations

import csv
import os
from array import array
from dataclasses import dataclass


@dataclass
class DirRow:
    path: str
    files: int
    bytes: int


@dataclass
class DuplicateRow:
    content_hash: str
    copies: int
    size: int
    reclaimable: int
    example: str = ""


@dataclass
class YearRow:
    year: int
    files: int
    bytes: int


@dataclass
class ExtRow:
    ext: str
    files: int
    bytes: int


@dataclass
class HotspotRow:
    path: str
    files: int
    bytes: int
    mean_size: float


def reconcile(store, reported_used_bytes: int) -> tuple[int, int, int]:
    """Live bytes we measured vs what Dropbox reports as used.

    The residual is version history plus deleted-but-retained data: it cannot be
    listed (``DeletedMetadata`` carries no size, and per-file ``list_revisions``
    would take months), but it can be inferred, and it may be the largest single
    reduction opportunity.
    """
    live = store.query("SELECT COALESCE(SUM(size), 0) FROM files")[0][0]
    return live, reported_used_bytes, reported_used_bytes - live


def duplicate_report(store, limit: int = 50) -> list[DuplicateRow]:
    rows = store.query(
        "SELECT content_hash, COUNT(*), MAX(size), SUM(size) - MAX(size) AS reclaimable "
        "FROM files WHERE content_hash IS NOT NULL "
        "GROUP BY content_hash HAVING COUNT(*) > 1 "
        "ORDER BY reclaimable DESC LIMIT ?",
        (limit,),
    )
    out = []
    for content_hash, copies, size, reclaimable in rows:
        example = store.query(
            "SELECT dirs.path_display || '/' || files.name FROM files "
            "JOIN dirs ON dirs.id = files.dir_id WHERE files.content_hash = ? LIMIT 1",
            (content_hash,),
        )
        out.append(
            DuplicateRow(
                content_hash=content_hash.hex() if isinstance(content_hash, bytes) else str(content_hash),
                copies=copies, size=size, reclaimable=reclaimable,
                example=example[0][0] if example else "",
            )
        )
    return out


def total_reclaimable_duplicates(store) -> int:
    return store.query(
        "SELECT COALESCE(SUM(reclaimable), 0) FROM ("
        "  SELECT SUM(size) - MAX(size) AS reclaimable FROM files "
        "  WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING COUNT(*) > 1)"
    )[0][0]


def cold_bytes(store) -> list[YearRow]:
    rows = store.query(
        "SELECT CAST(strftime('%Y', server_modified, 'unixepoch') AS INTEGER) AS y, "
        "       COUNT(*), COALESCE(SUM(size), 0) "
        "FROM files WHERE server_modified IS NOT NULL GROUP BY y ORDER BY y"
    )
    return [YearRow(*row) for row in rows]


def bytes_older_than(store, years: float, now: float | None = None) -> tuple[int, int]:
    import time

    cutoff = (now if now is not None else time.time()) - years * 365.25 * 86400
    row = store.query(
        "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM files WHERE server_modified < ?",
        (cutoff,),
    )[0]
    return row[0], row[1]


def top_dirs(store, limit: int = 50) -> list[DirRow]:
    """Recursive per-directory rollup, computed bottom-up in one linear pass.

    The obvious SQL -- joining every directory to every descendant path -- is
    quadratic and unusable at this scale. Walking the parent pointers in decreasing
    depth order visits each directory once.
    """
    conn = store.connect()
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM dirs").fetchone()[0]
    if max_id == 0:
        return []

    files_of = array("q", [0]) * (max_id + 1)
    bytes_of = array("q", [0]) * (max_id + 1)
    for dir_id, count, total in conn.execute(
        "SELECT dir_id, COUNT(*), COALESCE(SUM(size), 0) FROM files GROUP BY dir_id"
    ):
        if dir_id is not None and dir_id <= max_id:
            files_of[dir_id] = count
            bytes_of[dir_id] = total

    rows = conn.execute(
        "SELECT id, parent_id, path_display FROM dirs ORDER BY depth DESC"
    ).fetchall()
    for dir_id, parent_id, _ in rows:
        if parent_id is not None and 0 < parent_id <= max_id:
            files_of[parent_id] += files_of[dir_id]
            bytes_of[parent_id] += bytes_of[dir_id]

    out = [DirRow(path, files_of[dir_id], bytes_of[dir_id]) for dir_id, _, path in rows]
    out.sort(key=lambda r: -r.bytes)
    return out[:limit]


def small_file_hotspots(store, min_files: int = 100_000, max_mean_size: int = 65536,
                        limit: int = 50) -> list[HotspotRow]:
    """Directories holding many tiny files.

    These dominate crawl time and are usually the best cleanup candidates: lots of
    objects, little data, often machine-generated.
    """
    rows = store.query(
        "SELECT dirs.path_display, COUNT(*) AS n, COALESCE(SUM(files.size), 0), AVG(files.size) "
        "FROM files JOIN dirs ON dirs.id = files.dir_id "
        "GROUP BY files.dir_id HAVING n >= ? AND AVG(files.size) <= ? "
        "ORDER BY n DESC LIMIT ?",
        (min_files, max_mean_size, limit),
    )
    return [HotspotRow(*row) for row in rows]


def extension_profile(store, limit: int = 40) -> list[ExtRow]:
    rows = store.query(
        "SELECT COALESCE(NULLIF(ext, ''), '(none)'), COUNT(*), COALESCE(SUM(size), 0) AS b "
        "FROM files GROUP BY 1 ORDER BY b DESC LIMIT ?",
        (limit,),
    )
    return [ExtRow(*row) for row in rows]


def top_level_summary(store) -> list[DirRow]:
    """One row per immediate child of the crawl root -- usually per person or project."""
    root = (store.get_meta("root") or "").rstrip("/")
    depth = root.count("/") + 1
    rolled = {row.path: row for row in top_dirs(store, limit=10**9)}
    out = [
        rolled[path]
        for (path,) in store.query(
            "SELECT path_display FROM dirs WHERE depth = ? AND path_lower LIKE ?",
            (depth, root.lower() + "/%"),
        )
        if path in rolled
    ]
    out.sort(key=lambda r: -r.bytes)
    return out


def export_csv(store, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    written = []
    tables = {
        "top_dirs": (["path", "files", "bytes"],
                     [(r.path, r.files, r.bytes) for r in top_dirs(store, limit=5000)]),
        "duplicates": (["content_hash", "copies", "size", "reclaimable", "example"],
                       [(r.content_hash, r.copies, r.size, r.reclaimable, r.example)
                        for r in duplicate_report(store, limit=5000)]),
        "by_year": (["year", "files", "bytes"],
                    [(r.year, r.files, r.bytes) for r in cold_bytes(store)]),
        "by_extension": (["ext", "files", "bytes"],
                         [(r.ext, r.files, r.bytes) for r in extension_profile(store, limit=1000)]),
        "small_file_hotspots": (["path", "files", "bytes", "mean_size"],
                                [(r.path, r.files, r.bytes, round(r.mean_size, 1))
                                 for r in small_file_hotspots(store, limit=1000)]),
    }
    for name, (header, rows) in tables.items():
        path = os.path.join(outdir, f"{name}.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(path)
    return written
