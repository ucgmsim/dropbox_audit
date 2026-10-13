"""The browsable tree page: one self-contained HTML file showing where the space goes.

Four steps, one job each. `measure` turns the database into figures for every folder,
kept in arrays indexed by dir id so that two million folders cost about 160 MB.
`layout` decides which folders and files get a row of their own and which fold away,
and `tables` picks the folders worth a closer look. `render` writes the HTML.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field

YEAR = 365.25 * 86400
COLD_YEARS = 2
TINY_FILES = 100_000
TINY_MEAN = 1 << 20
DUP_TABLE_SHARE = 0.5
COLD_TABLE_SHARE = 0.9
DUP_BADGE = 0.10
COLD_BADGE = 0.50
TABLE_ROWS = 25
TYPE_ROWS = 20
BAR_WIDTH = 28


# ---- measure: figures for every folder --------------------------------------


@dataclass
class Figures:
    """Per-folder figures indexed by dir id. Totals cover each folder's whole subtree."""

    parent: array
    depth: array
    order: array          # dir ids, deepest first
    kids: array           # direct subfolders
    own_files: array      # files directly inside
    files: array
    folders: array
    size: array
    newest: array         # the latest client_modified below, 0 if none
    cold: array           # bytes last modified before the cold cut-off
    held: array           # bytes whose content also exists outside the folder
    on_page: bytearray    # 1 for the page root and everything below it
    page_root: int
    copies: dict = field(default_factory=dict)  # files.rowid -> copies of its content
    reclaimable: int = 0  # page bytes deletable while one copy of everything stays


def find_dir(store, path: str):
    """The (id, path_display) of the folder at `path`, matched without case, or None."""
    rows = store.query("SELECT id, path_display FROM dirs WHERE path_lower = ?",
                       (path.rstrip("/").lower(),))
    return rows[0] if rows else None


def crawl_time(store) -> tuple[float, bool]:
    """When the listing was taken, and whether a pass over it finished."""
    finished = store.get_meta("last_pass_completed_at")
    if finished:
        return float(finished), True
    return float(store.get_meta("created_at") or time.time()), False


def measure(store, listed: float, min_size: int, page_root: int) -> Figures:
    """Figures for every folder, as of `listed`, the time the listing was taken."""
    conn = store.connect()
    # Bound the snapshot before reading it: folders created while a crawl runs get
    # higher ids, and a folder's parent is always created before the folder.
    top = conn.execute("SELECT COALESCE(MAX(id), 0) FROM dirs").fetchone()[0] + 1

    def zeros():
        return array("q", [0]) * top

    f = Figures(parent=zeros(), depth=zeros(), order=array("q"), kids=zeros(),
                own_files=zeros(), files=zeros(), folders=zeros(), size=zeros(),
                newest=zeros(), cold=zeros(), held=zeros(), on_page=bytearray(top),
                page_root=page_root)
    for dir_id, parent_id, depth in conn.execute(
            "SELECT id, parent_id, depth FROM dirs WHERE id < ? ORDER BY depth DESC", (top,)):
        f.parent[dir_id] = parent_id or 0
        f.depth[dir_id] = depth
        f.order.append(dir_id)

    # A file's date is whatever uploaded it sent, so a wrong clock can date it after the
    # listing; such a date is never a folder's newest.
    for dir_id, count, total, latest, old in conn.execute(
            "SELECT dir_id, COUNT(*), COALESCE(SUM(size), 0), "
            "       COALESCE(MAX(CASE WHEN client_modified <= ? THEN client_modified END), 0), "
            "       COALESCE(SUM(CASE WHEN client_modified < ? THEN size END), 0) "
            "FROM files GROUP BY dir_id", (listed, listed - COLD_YEARS * YEAR)):
        if 0 < dir_id < top:
            f.own_files[dir_id] = f.files[dir_id] = count
            f.size[dir_id], f.newest[dir_id], f.cold[dir_id] = total, latest, old

    for dir_id in f.order:  # deepest first, so each folder is complete before its parent
        up = f.parent[dir_id]
        if up:
            f.kids[up] += 1
            f.files[up] += f.files[dir_id]
            f.folders[up] += f.folders[dir_id] + 1
            f.size[up] += f.size[dir_id]
            f.cold[up] += f.cold[dir_id]
            f.newest[up] = max(f.newest[up], f.newest[dir_id])

    f.on_page[page_root] = 1
    for dir_id in reversed(f.order):  # shallowest first
        if f.on_page[f.parent[dir_id]]:
            f.on_page[dir_id] = 1

    return f
