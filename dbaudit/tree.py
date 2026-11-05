"""The browsable tree page: one self-contained HTML file showing where the space goes.

Four steps, one job each. `measure` turns the database into figures for every folder,
kept in arrays indexed by dir id so that two million folders cost about 160 MB.
`layout` decides which folders and files get a row of their own and which fold away,
and `tables` picks the folders worth a closer look. `render` writes the HTML.
"""

from __future__ import annotations

import heapq
import html
import sqlite3
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TextIO

from . import kinds, report
from .kinds import Tally
from .store import Store

YEAR = 365.25 * 86400
COLD_YEARS = 2
TINY_FILES = 100_000
TINY_MEAN = 1 << 20
DUP_TABLE_SHARE = 0.5
COLD_TABLE_SHARE = 0.9
DUP_BADGE = 0.10
COLD_BADGE = 0.50
KIND_BADGE = 0.50
TABLE_ROWS = 25
TYPE_ROWS = 20
BAR_WIDTH = 28


# ---- measure: figures for every folder --------------------------------------


@dataclass
class Figures:
    """Per-folder figures indexed by dir id. Totals cover each folder's whole subtree.

    Parameters
    ----------
    parent : array of int
        Each folder's parent id; 0 for a top folder.
    depth : array of int
        Each folder's depth below the account root.
    order : array of int
        Dir ids, deepest first.
    kids : array of int
        Direct subfolders.
    own_files : array of int
        Files directly inside.
    files : array of int
        Files in the whole subtree.
    folders : array of int
        Folders in the whole subtree, not counting the folder itself.
    size : array of int
        Bytes in the whole subtree.
    newest : array of int
        The latest ``client_modified`` below, in seconds since the Unix epoch; 0 if none.
    cold : array of int
        Bytes last modified before the cold cut-off.
    held : array of int
        Bytes whose content also exists outside the folder.
    on_page : bytearray
        1 for the page root and everything below it, else 0.
    page_root : int
        The dir id of the page's root folder.
    copies : array of int, optional
        By ``files.rowid``: the copies of that file's content, or 0 for a file below the
        minimum size.
    reclaimable : int, optional
        Bytes on the page that deleting duplicates frees while one copy of everything
        stays.
    """

    parent: array
    depth: array
    order: array  # dir ids, deepest first
    kids: array  # direct subfolders
    own_files: array  # files directly inside
    files: array
    folders: array
    size: array
    newest: array  # the latest client_modified below, 0 if none
    cold: array  # bytes last modified before the cold cut-off
    held: array  # bytes whose content also exists outside the folder
    on_page: bytearray  # 1 for the page root and everything below it
    page_root: int
    copies: array = field(
        default_factory=lambda: array("I")
    )  # by files.rowid: copies, or 0
    reclaimable: int = 0  # page bytes deletable while one copy of everything stays


def find_dir(store: Store, path: str) -> tuple[int, str] | None:
    """The (id, path_display) of the folder at `path`, matched without case, or None.

    Parameters
    ----------
    store : Store
        The audit database.
    path : str
        A folder path; a trailing slash and case are ignored.

    Returns
    -------
    tuple of (int, str) or None
        The folder's dir id and displayed path, or None if it is not listed.
    """
    rows = store.query(
        "SELECT id, path_display FROM dirs WHERE path_lower = ?",
        (path.rstrip("/").lower(),),
    )
    return rows[0] if rows else None


def crawl_time(store: Store) -> tuple[float, bool]:
    """When the listing was taken, and whether a pass over it finished.

    Parameters
    ----------
    store : Store
        The audit database.

    Returns
    -------
    tuple of (float, bool)
        The time in seconds since the Unix epoch: when the last pass finished, else when
        the listing began, else now; and whether a pass finished.
    """
    finished = store.get_meta("last_pass_completed_at")
    if finished:
        return float(finished), True
    return float(store.get_meta("created_at") or time.time()), False


# Every content stored more than once, copy by copy, in content order. Empty files
# are left out: they all share one hash and hold no bytes.
DUPLICATES = (
    "SELECT content_hash, dir_id, size, rowid FROM files "
    "WHERE size > 0 AND content_hash IN ("
    "  SELECT content_hash FROM files WHERE size > 0 AND content_hash IS NOT NULL "
    "  GROUP BY content_hash HAVING COUNT(*) > 1) "
    "ORDER BY content_hash"
)


def duplicate_groups(
    conn: sqlite3.Connection, top: int
) -> Iterator[list[tuple[int, int, int]]]:
    """Yield each duplicated content's copies as (dir_id, size, rowid), one content at a
    time, so memory stays flat however many duplicates there are.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection to the audit database.
    top : int
        One more than the largest dir id to count; copies in other folders are ignored.

    Yields
    ------
    list of tuple
        The copies of one content, as ``(dir_id, size, rowid)``, when it has more than
        one.
    """
    group, last = [], None
    for digest, dir_id, size, rowid in conn.execute(DUPLICATES):
        if digest != last:
            if len(group) > 1:
                yield group
            group, last = [], digest
        if 0 < dir_id < top:
            group.append((dir_id, size, rowid))
    if len(group) > 1:
        yield group


def common_folder(dirs: list[int], parent: array[int], depth: array[int]) -> int:
    """The deepest folder holding every one of `dirs`: their lowest common ancestor.

    Parameters
    ----------
    dirs : list of int
        Dir ids; at least one.
    parent : array of int
        Each folder's parent id.
    depth : array of int
        Each folder's depth.

    Returns
    -------
    int
        The dir id of the lowest common ancestor.
    """
    meet = dirs[0]
    for other in dirs[1:]:
        while depth[meet] > depth[other]:
            meet = parent[meet]
        while depth[other] > depth[meet]:
            other = parent[other]
        while meet != other:
            meet, other = parent[meet], parent[other]
    return meet


def measure(store: Store, listed: float, min_size: int, page_root: int) -> Figures:
    """Figures for every folder, as of `listed`, the time the listing was taken.

    Parameters
    ----------
    store : Store
        The audit database.
    listed : float
        When the listing was taken, in seconds since the Unix epoch.
    min_size : int
        Files of at least this many bytes get their number of copies counted.
    page_root : int
        The dir id of the page's root folder.

    Returns
    -------
    Figures
        The figures for every folder.
    """
    conn = store.connect()
    # Bound the snapshot before reading it: folders created while a crawl runs get
    # higher ids, and a folder's parent is always created before the folder.
    top = conn.execute("SELECT COALESCE(MAX(id), 0) FROM dirs").fetchone()[0] + 1
    rowids = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM files").fetchone()[0] + 1

    def zeros():
        """An array of zeros with a slot for every dir id.

        Returns
        -------
        array of int
            ``top`` zeros, of 64-bit integers.
        """
        return array("q", [0]) * top

    f = Figures(
        parent=zeros(),
        depth=zeros(),
        order=array("q"),
        kids=zeros(),
        own_files=zeros(),
        files=zeros(),
        folders=zeros(),
        size=zeros(),
        newest=zeros(),
        cold=zeros(),
        held=zeros(),
        on_page=bytearray(top),
        page_root=page_root,
        copies=array("I", [0]) * rowids,
    )
    for dir_id, parent_id, depth in conn.execute(
        "SELECT id, parent_id, depth FROM dirs WHERE id < ? ORDER BY depth DESC", (top,)
    ):
        f.parent[dir_id] = parent_id or 0
        f.depth[dir_id] = depth
        f.order.append(dir_id)

    # A file's date is whatever uploaded it sent, so a wrong clock can date it after the
    # listing; such a date is never a folder's newest.
    for dir_id, count, total, latest, old in conn.execute(
        "SELECT dir_id, COUNT(*), COALESCE(SUM(size), 0), "
        "       COALESCE(MAX(CASE WHEN client_modified <= ? THEN client_modified END), 0), "
        "       COALESCE(SUM(CASE WHEN client_modified < ? THEN size END), 0) "
        "FROM files GROUP BY dir_id",
        (listed, listed - COLD_YEARS * YEAR),
    ):
        if 0 < dir_id < top:
            f.own_files[dir_id] = f.files[dir_id] = count
            f.size[dir_id], f.newest[dir_id], f.cold[dir_id] = total, latest, old

    for (
        dir_id
    ) in f.order:  # deepest first, so each folder is complete before its parent
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

    for group in duplicate_groups(conn, top):
        for _, size, rowid in group:
            if size >= min_size and rowid < rowids:
                f.copies[rowid] = len(group)
        dirs = [d for d, _, _ in group]
        if min(dirs) != max(dirs):
            # A copy's bytes are held elsewhere from each folder on its path below the
            # deepest folder holding every copy; that folder and those above hold them all.
            meet = common_folder(dirs, f.parent, f.depth)
            for d, size, _ in group:
                while d != meet:
                    f.held[d] += size
                    d = f.parent[d]
        on = [size for d, size, _ in group if f.on_page[d]]
        if on:
            f.reclaimable += sum(on) - (max(on) if len(on) == len(group) else 0)
    return f


# ---- layout: which entries get a row ----------------------------------------


def is_tiny(figures: Figures, d: int) -> bool:
    """Whether a folder is made of many tiny files.

    Parameters
    ----------
    figures : Figures
        The folders' figures.
    d : int
        The folder's dir id.

    Returns
    -------
    bool
        True if it holds at least ``TINY_FILES`` files, averaging ``TINY_MEAN`` bytes or
        less.
    """
    return (
        figures.files[d] >= TINY_FILES
        and figures.size[d] <= figures.files[d] * TINY_MEAN
    )


@dataclass
class Row:
    """One row of the page's tree: a folder, a file, or a fold of the rest.

    Parameters
    ----------
    kind : str
        ``"dir"``, ``"file"`` or ``"fold"``.
    name : str
        The folder or file name; empty for a fold.
    path : str
        Below the crawl root; a fold row carries its folder's.
    size : int
        Bytes.
    files : int, optional
        Files in the subtree.
    folders : int, optional
        Folders in the subtree.
    date : int, optional
        A folder's newest file, or a file's own date, in seconds since the Unix epoch.
    dir_id : int, optional
        A folder's dir id.
    copies : int, optional
        A file's content: copies in the listing, this one included.
    held : int, optional
        Bytes whose content is also stored outside the folder.
    cold : int, optional
        Bytes last modified before the cold cut-off.
    tiny : bool, optional
        Whether the folder is made of tiny files.
    entries : int, optional
        A fold row: the folder's direct children folded into it.
    group : str, optional
        A folder's largest kind of data (kinds.GROUPS); empty if none.
    group_bytes : float, optional
        The bytes of that group.
    children : list of Row, optional
        The rows below this one.
    """

    kind: str  # "dir", "file" or "fold"
    name: str
    path: str  # below the crawl root; a fold row carries its folder's
    size: int
    files: int = 0
    folders: int = 0
    date: int = 0  # a folder's newest file, or a file's own date
    dir_id: int = 0
    copies: int = 1  # a file's content: copies in the listing, this one included
    held: int = 0
    cold: int = 0
    tiny: bool = False
    entries: int = 0  # a fold row: the folder's direct children folded into it
    group: str = ""  # a folder's largest kind of data (kinds.GROUPS), and its bytes
    group_bytes: float = 0.0
    children: list = field(default_factory=list)


@dataclass
class Page:
    """The rows the page shows.

    Parameters
    ----------
    root : Row
        The page's root folder, with its rows below it.
    threshold : int
        Entries smaller than this many bytes fold away.
    rows : int
        The number of rows, folds included.
    shown : set of int
        Dir ids that have a row of their own.
    """

    root: Row
    threshold: int  # entries below this fold
    rows: int
    shown: set  # dir ids with a row of their own


def below(path: str, crawl_root: str) -> str:
    """`path` relative to the crawl root, as `data-path` carries it.

    Parameters
    ----------
    path : str
        A displayed path in the listing.
    crawl_root : str
        The folder the crawl began at.

    Returns
    -------
    str
        `path` without the crawl root, and without a leading slash; empty for the root itself.
    """
    root = crawl_root.rstrip("/")
    if path.lower() == root.lower():
        return ""
    if root and path.lower().startswith(root.lower() + "/"):
        return path[len(root) + 1 :]
    return path.lstrip("/")


def chunked(
    conn: sqlite3.Connection, query: str, ids: Iterable[int]
) -> Iterator[tuple]:
    """The rows `query` finds for `ids`, a chunk at a time; it ends in `IN ({})`.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection to the audit database.
    query : str
        SQL with ``{}`` where the placeholders go.
    ids : iterable of int
        The values to look for, taken 500 at a time.

    Yields
    ------
    tuple
        Each row the query finds.
    """
    ids = list(ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        yield from conn.execute(query.format(",".join("?" * len(chunk))), chunk)


def dir_names(
    conn: sqlite3.Connection, ids: Iterable[int]
) -> dict[int, tuple[str, str]]:
    """{dir id: (name, path_display)}.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection to the audit database.
    ids : iterable of int
        Dir ids.

    Returns
    -------
    dict of int to tuple of (str, str)
        Each folder's name and displayed path.
    """
    return {
        d: (name, path)
        for d, name, path in chunked(
            conn, "SELECT id, name, path_display FROM dirs WHERE id IN ({})", ids
        )
    }


def past(threshold: int, size: int) -> int:
    """The first threshold, doubling from `threshold`, that is above `size`.

    Parameters
    ----------
    threshold : int
        The size to start from, in bytes.
    size : int
        The size to get above, in bytes.

    Returns
    -------
    int
        A bytes threshold greater than `size`.
    """
    while threshold <= size:
        threshold = max(threshold * 2, 1)
    return threshold


def folder_row(f: Figures, d: int, name: str, path: str) -> Row:
    """The row for a folder.

    Parameters
    ----------
    f : Figures
        The folders' figures.
    d : int
        The folder's dir id.
    name : str
        Its name.
    path : str
        Its path below the crawl root.

    Returns
    -------
    Row
        A ``"dir"`` row with the folder's figures.
    """
    return Row(
        "dir",
        name,
        path,
        f.size[d],
        files=f.files[d],
        folders=f.folders[d],
        date=f.newest[d],
        dir_id=d,
        held=f.held[d],
        cold=f.cold[d],
        tiny=is_tiny(f, d),
    )


def layout(store: Store, figures: Figures, threshold: int, max_rows: int) -> Page:
    """Decide which folders and files get a row of their own.

    Entries below the threshold fold away; the threshold doubles until the page fits
    `max_rows`.

    Parameters
    ----------
    store : Store
        The audit database.
    figures : Figures
        From `measure`.
    threshold : int
        The size, in bytes, from which an entry gets a row.
    max_rows : int
        The most rows the page may have.

    Returns
    -------
    Page
        The rows, as a tree under the page root.
    """
    f, root = figures, figures.page_root
    conn = store.connect()
    top = len(f.size)
    # Each folder and file at or above the threshold gets a row, so while more than
    # max_rows of either qualify the page cannot fit: start past the (max_rows + 1)-th
    # largest. Memory then follows --max-rows, whatever --min-size is.
    folders = heapq.nlargest(
        max_rows + 1,
        (
            d
            for d in range(1, top)
            if f.on_page[d] and d != root and f.size[d] >= threshold
        ),
        key=lambda d: f.size[d],
    )
    if len(folders) > max_rows:
        threshold = past(threshold, f.size[folders[-1]])
    files = heapq.nlargest(
        max_rows + 1,
        (
            row
            for row in conn.execute(
                "SELECT rowid, dir_id, size FROM files WHERE size >= ?", (threshold,)
            )
            if 0 < row[1] < top and f.on_page[row[1]]
        ),
        key=lambda row: row[2],
    )
    if len(files) > max_rows:
        threshold = past(threshold, files[-1][2])

    def rows_at(t: int) -> int:
        """Count the rows the page would have at a threshold.

        Parameters
        ----------
        t : int
            A threshold, in bytes.

        Returns
        -------
        int
            The rows: the root, each shown folder and file, and a fold for each folder with
            anything left out.
        """
        shown = [d for d in folders if f.size[d] >= t]
        subfolders = Counter(f.parent[d] for d in shown)
        own = Counter(row[1] for row in files if row[2] >= t)
        folds = sum(
            1
            for d in [root, *shown]
            if f.kids[d] > subfolders[d] or f.own_files[d] > own[d]
        )
        return 1 + len(shown) + sum(own.values()) + folds

    rows = rows_at(threshold)
    while rows > max_rows and threshold <= f.size[root]:
        threshold = max(threshold * 2, 1)
        rows = rows_at(threshold)

    shown = {d for d in folders if f.size[d] >= threshold}
    names = dir_names(conn, shown | {root})
    details = {
        rowid: (name, modified)
        for rowid, name, modified in chunked(
            conn,
            "SELECT rowid, name, client_modified FROM files WHERE rowid IN ({})",
            [row[0] for row in files if row[2] >= threshold],
        )
    }
    crawl_root = store.get_meta("root") or ""
    inside = defaultdict(list)
    for d in shown:
        name, path = names[d]
        inside[f.parent[d]].append(folder_row(f, d, name, below(path, crawl_root)))
    for rowid, dir_id, size in files:
        if size >= threshold:
            name, modified = details[rowid]
            folder = below(names[dir_id][1], crawl_root)
            copies = f.copies[rowid] if rowid < len(f.copies) else 0
            inside[dir_id].append(
                Row(
                    "file",
                    name,
                    f"{folder}/{name}" if folder else name,
                    size,
                    files=1,
                    date=modified or 0,
                    copies=copies or 1,
                )
            )

    def assemble(row: Row) -> None:
        """Attach a folder's rows to it, and add a fold for what is left out.

        Parameters
        ----------
        row : Row
            A shown folder; its ``children`` are set here.
        """
        children = sorted(inside.get(row.dir_id, ()), key=lambda r: (-r.size, r.name))
        for child in children:
            if child.kind == "dir":
                assemble(child)
        subfolders = [c for c in children if c.kind == "dir"]
        entries = (
            f.kids[row.dir_id]
            - len(subfolders)
            + f.own_files[row.dir_id]
            - (len(children) - len(subfolders))
        )
        if entries:
            children.append(
                Row(
                    "fold",
                    "",
                    row.path,
                    row.size - sum(c.size for c in children),
                    files=row.files - sum(c.files for c in children),
                    folders=row.folders - sum(c.folders + 1 for c in subfolders),
                    entries=entries,
                )
            )
        row.children = children

    name, path = names[root]
    page_row = folder_row(f, root, name, below(path, crawl_root))
    assemble(page_row)
    return Page(page_row, threshold, rows, shown | {root})


def measure_kinds(
    figures: Figures, page: Page, rows: Iterable[kinds.WalkRow], listed: float
) -> Tally:
    """Give each folder row on the page its largest kind of data, and return the tally of
    `rows`, a kinds.walk read once.

    Only the folders with a row keep totals, so memory follows --max-rows: each byte counts
    toward the nearest of them at or above its folder, and the totals roll up from there.

    Parameters
    ----------
    figures : Figures
        From `measure`.
    page : Page
        From `layout`; its folder rows get their ``group`` set.
    rows : iterable of tuple
        What `kinds.walk` gives, read once.
    listed : float
        When the listing was taken, in seconds since the Unix epoch.

    Returns
    -------
    Tally
        The bytes by kind, month and age.
    """
    shown = [d for d in figures.order if d in page.shown]  # deepest first
    at = {d: i for i, d in enumerate(shown)}
    near = array("i", [-1]) * len(
        figures.size
    )  # the nearest folder row, as an index in shown
    for d in reversed(
        figures.order
    ):  # shallowest first, so a parent is settled before its folders
        near[d] = at[d] if d in at else near[figures.parent[d]]
    width = len(kinds.GROUPS)
    totals = array("d", [0.0]) * (width * len(shown))
    slot = {kind: kinds.GROUPS.index(group) for kind, (_, group) in kinds.KINDS.items()}

    def counted():
        """Pass the walk's rows on, adding each to its nearest folder row.

        Yields
        ------
        tuple
            Each row unchanged.
        """
        for row in rows:
            if 0 < row[0] < len(near) and near[row[0]] >= 0:
                totals[near[row[0]] * width + slot[row[1]]] += row[2]
            yield row

    found = kinds.tally(counted(), listed)
    for d in shown:  # deepest first, so each folder is complete before the row above it
        up = near[figures.parent[d]]
        if up >= 0:
            for g in range(width):
                totals[up * width + g] += totals[at[d] * width + g]
    stack = [page.root]
    while stack:
        row = stack.pop()
        stack.extend(row.children)
        if row.kind == "dir" and row.dir_id in at:
            sizes = totals[at[row.dir_id] * width : (at[row.dir_id] + 1) * width]
            largest = max(range(width), key=sizes.__getitem__)
            if sizes[largest] > 0:
                row.group, row.group_bytes = kinds.GROUPS[largest], sizes[largest]
    return found


# ---- tables: the folders worth a closer look --------------------------------


def ancestors(d: int, parent: array[int]) -> Iterator[int]:
    """The folders above a folder, nearest first.

    Parameters
    ----------
    d : int
        A dir id.
    parent : array of int
        Each folder's parent id.

    Yields
    ------
    int
        The dir ids of its parent, grandparent and so on, up to a top folder.
    """
    d = parent[d]
    while d:
        yield d
        d = parent[d]


@dataclass
class Entry:
    """A folder in one of the tables.

    Parameters
    ----------
    dir_id : int
        The folder's dir id.
    path : str
        Its path below the crawl root.
    target : int
        The row its link opens: itself, or its nearest shown ancestor.
    size : int
        Bytes.
    files : int
        Files in the subtree.
    held : int
        Bytes whose content is also stored outside the folder.
    cold : int
        Bytes last modified before the cold cut-off.
    newest : int
        The latest file date below, in seconds since the Unix epoch.
    place : str, optional
        Where most of its held-elsewhere bytes have their copies; empty if unknown.
    place_share : float, optional
        The share of the held bytes that lie in `place`.
    """

    dir_id: int
    path: str
    target: int  # the row its link opens: itself, or its nearest shown ancestor
    size: int
    files: int
    held: int
    cold: int
    newest: int
    place: str = ""  # where most of its held-elsewhere bytes have their copies
    place_share: float = 0.0


def choose(
    ranked: Iterable[int], parent: array[int], limit: int = TABLE_ROWS
) -> list[int]:
    """Take folders in ranking order, skipping any inside or above one already taken.

    Parameters
    ----------
    ranked : iterable of int
        Dir ids, best first.
    parent : array of int
        Each folder's parent id.
    limit : int, optional
        The most folders to take.

    Returns
    -------
    list of int
        The chosen dir ids, in ranking order.
    """
    taken, above, out = set(), set(), []
    for d in ranked:
        chain = list(ancestors(d, parent))
        if d in above or any(a in taken for a in chain):
            continue
        out.append(d)
        taken.add(d)
        above.update(chain)
        if len(out) == limit:
            break
    return out


def place(d: int, near: set, parent: array[int], depth: array[int], floor: int) -> int:
    """Where a copy in folder `d` lies, seen from a folder whose ancestry is `near`.

    The copy's ancestor one level below where the two paths part, never shallower than
    `floor`; a copy shallower than that is its own place.

    Parameters
    ----------
    d : int
        The dir id of the folder holding the copy.
    near : set of int
        The folder and its ancestors.
    parent : array of int
        Each folder's parent id.
    depth : array of int
        Each folder's depth.
    floor : int
        The shallowest depth a place may have.

    Returns
    -------
    int
        The dir id of the place.
    """
    chain = [d]
    while chain[-1] and chain[-1] not in near:
        chain.append(parent[chain[-1]])
    level = max(depth[chain[-1]] + 1, floor)
    for step in chain:
        if depth[step] <= level:
            return step
    return d


def counterparts(
    conn: sqlite3.Connection, figures: Figures, folders: list[int], floor: int
) -> dict:
    """{folder: (place, bytes)}: the place holding copies of the most of each folder's
    held-elsewhere bytes. A second pass over the duplicates, for these folders only.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection to the audit database.
    figures : Figures
        From `measure`.
    folders : list of int
        Dir ids to find places for.
    floor : int
        The shallowest depth a place may have.

    Returns
    -------
    dict of int to tuple of (int, int)
        For each folder with held bytes, the dir id of the place and the bytes held there.
    """
    if not folders:
        return {}
    f, wanted = figures, set(folders)
    owner = array("q", [0]) * len(f.size)
    for d in reversed(f.order):
        owner[d] = d if d in wanted else owner[f.parent[d]]
    near = {d: {d, *ancestors(d, f.parent)} for d in folders}
    credit = {d: defaultdict(int) for d in folders}
    for group in duplicate_groups(conn, len(f.size)):
        for mine in {owner[d] for d, _, _ in group} - {0}:
            inside = sum(size for d, size, _ in group if owner[d] == mine)
            for spot in {
                place(d, near[mine], f.parent, f.depth, floor)
                for d, _, _ in group
                if owner[d] != mine
            }:
                credit[mine][spot] += inside
    out = {}
    for d in folders:
        if credit[d]:
            spot = max(credit[d], key=lambda p: (credit[d][p], -p))
            out[d] = (spot, credit[d][spot])
    return out


def tables(store: Store, figures: Figures, page: Page) -> dict[str, list[Entry]]:
    """Pick the folders worth a closer look.

    Parameters
    ----------
    store : Store
        The audit database.
    figures : Figures
        From `measure`.
    page : Page
        From `layout`.

    Returns
    -------
    dict of str to list of Entry
        The ``"dup"``, ``"cold"`` and ``"tiny"`` tables.
    """
    f, root = figures, figures.page_root
    below_root = [d for d in range(1, len(f.size)) if f.on_page[d] and d != root]
    big = [d for d in below_root if f.size[d] >= page.threshold]
    dup = choose(
        sorted(
            (d for d in big if f.held[d] and f.held[d] >= DUP_TABLE_SHARE * f.size[d]),
            key=lambda d: (-f.held[d], d),
        ),
        f.parent,
    )
    cold = choose(
        sorted(
            (d for d in big if f.cold[d] and f.cold[d] >= COLD_TABLE_SHARE * f.size[d]),
            key=lambda d: (-f.cold[d], d),
        ),
        f.parent,
    )
    tiny = choose(
        sorted(
            (d for d in below_root if is_tiny(f, d)), key=lambda d: (-f.files[d], d)
        ),
        f.parent,
    )

    conn = store.connect()
    crawl_root = store.get_meta("root") or ""
    spots = counterparts(conn, f, dup, crawl_root.rstrip("/").count("/") + 2)
    names = dir_names(conn, {*dup, *cold, *tiny, *(spot for spot, _ in spots.values())})

    def entry(d: int) -> Entry:
        """The table entry for a folder.

        Parameters
        ----------
        d : int
            A dir id.

        Returns
        -------
        Entry
            Its figures, and where its held bytes have their copies.
        """
        target = d
        while target not in page.shown:
            target = f.parent[target]
        e = Entry(
            d,
            below(names[d][1], crawl_root),
            target,
            f.size[d],
            f.files[d],
            f.held[d],
            f.cold[d],
            f.newest[d],
        )
        if d in spots:
            spot, credit = spots[d]
            e.place, e.place_share = (
                below(names[spot][1], crawl_root),
                credit / f.held[d],
            )
        return e

    return {
        "dup": [entry(d) for d in dup],
        "cold": [entry(d) for d in cold],
        "tiny": [entry(d) for d in tiny],
    }


# ---- build: everything the page shows ---------------------------------------


@dataclass
class Summary:
    """What the page's header and tables show besides the tree.

    Parameters
    ----------
    title : str
        The page root's path.
    db_name : str
        The database's name.
    listed : float
        When the listing was taken, in seconds since the Unix epoch.
    finished : bool
        Whether a pass over the listing finished.
    generated : float
        When the page was made, in seconds since the Unix epoch.
    reclaimable : int
        Bytes that deleting duplicates frees on the page.
    pending : int
        Pieces of the listing still waiting.
    running : int
        Pieces being listed.
    failed : int
        Pieces that failed.
    min_size : int
        The size, in bytes, from which an entry gets a row.
    types : list of ExtRow
        Bytes by file type.
    years : list of YearRow
        Bytes by year last modified.
    kinds : Tally, optional
        Bytes by kind; None is shown as empty.
    looked_inside : bool, optional
        Whether an archive index was given, to look inside tarballs.
    """

    title: str  # the page root's path
    db_name: str
    listed: float  # when the listing was taken
    finished: bool  # a pass over the listing finished
    generated: float
    reclaimable: int
    pending: int
    running: int
    failed: int
    min_size: int
    types: list[report.ExtRow]
    years: list[report.YearRow]
    kinds: Tally | None = None
    looked_inside: bool = False  # an archive index was given, to look inside tarballs


def build(
    store: Store,
    page_path: str,
    min_size: int,
    max_rows: int,
    db_name: str,
    rules: kinds.Rules | None = None,
    index: sqlite3.Connection | None = None,
):
    """(page, summary, tables) for the folder at `page_path`. `rules` and `index` (an
    archive index, see kinds.archive_index) say what kinds of data the files are.

    Parameters
    ----------
    store : Store
        The audit database.
    page_path : str
        The folder to show.
    min_size : int
        The size, in bytes, from which an entry gets a row.
    max_rows : int
        The most rows the tree may have.
    db_name : str
        The database's name, shown in the header.
    rules : kinds.Rules, optional
        The classifier; None uses the built-in rules.
    index : sqlite3.Connection, optional
        The archive index; None judges tarballs by their names.

    Returns
    -------
    tuple
        The `Page`, the `Summary` and the tables (see `tables`).
    """
    conn = store.connect()
    # One read transaction for every query below: read separately, a crawl still writing
    # could show measure one listing and layout another.
    conn.execute("BEGIN")
    try:
        listed, finished = crawl_time(store)
        crawl_root = store.get_meta("root") or ""
        whole = page_path.rstrip("/").lower() == crawl_root.rstrip("/").lower()
        under = None if whole else page_path
        found = find_dir(store, page_path)
        if found is None:  # nothing listed yet
            page, chosen, reclaimable = (
                Page(Row("dir", page_path, "", 0), min_size, 1, {0}),
                {"dup": [], "cold": [], "tiny": []},
                0,
            )
            counted = kinds.Tally(listed)
        else:
            figures = measure(store, listed, min_size, found[0])
            page = layout(store, figures, min_size, max_rows)
            counted = measure_kinds(
                figures,
                page,
                kinds.walk(store, rules or kinds.Rules(), index, under),
                listed,
            )
            chosen = tables(store, figures, page)
            reclaimable = figures.reclaimable
        summary = Summary(
            title=found[1] if found else page_path,
            db_name=db_name,
            listed=listed,
            finished=finished,
            generated=time.time(),
            reclaimable=reclaimable,
            pending=store.pending_count(),
            running=store.running_count(),
            failed=store.progress()["shards_error"],
            min_size=min_size,
            types=report.extension_profile(store, limit=TYPE_ROWS, under=under),
            years=report.cold_bytes(store, under=under),
            kinds=counted,
            looked_inside=index is not None,
        )
        return page, summary, chosen
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")


# ---- render: the HTML -------------------------------------------------------


MID, LAST = "├─ ", "└─ "
BLOCKS = " ▏▎▍▌▋▊▉"


def esc(text: str) -> str:
    """Escape text for HTML.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        `text` with ``&``, ``<``, ``>`` and quotes escaped.
    """
    return html.escape(text, quote=True)


def compact(n: float) -> str:
    """A size in at most four characters, the way the reference page shows them.

    Parameters
    ----------
    n : float
        A size in bytes.

    Returns
    -------
    str
        For example ``"512B"``, ``"1.5K"`` or ``"230G"``.
    """
    for unit in "BKMGTP":
        if n < 1024 or unit == "P":
            break
        n /= 1024
    if unit == "B":
        return f"{int(n)}B"
    return f"{n:.1f}{unit}" if n < 100 else f"{n:.0f}{unit}"


def bar(share: float) -> str:
    """A text bar for a share.

    Parameters
    ----------
    share : float
        A share of the whole, clamped to 0 to 1.

    Returns
    -------
    str
        Block characters up to ``BAR_WIDTH`` wide, in eighths.
    """
    eighths = round(max(0.0, min(share, 1.0)) * BAR_WIDTH * 8)
    full, part = divmod(eighths, 8)
    return "█" * full + (BLOCKS[part] if part else "")


def day(t: float) -> str:
    """Format a date.

    Parameters
    ----------
    t : float
        Seconds since the Unix epoch; 0 means no date.

    Returns
    -------
    str
        ``"YYYY-MM-DD"`` in UTC, or empty for 0.
    """
    return time.strftime("%Y-%m-%d", time.gmtime(t)) if t else ""


def plural(n: int, word: str) -> str:
    """A count with its noun.

    Parameters
    ----------
    n : int
        The count.
    word : str
        The noun, singular.

    Returns
    -------
    str
        For example ``"1 file"`` or ``"1,200 files"``.
    """
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _counts(row: Row) -> str:
    """The files and folders of a row, as text.

    Parameters
    ----------
    row : Row
        A folder or fold row.

    Returns
    -------
    str
        For example ``"3 files, 1 folder"``; folders are left out when there are none.
    """
    text = plural(row.files, "file")
    return text + (", " + plural(row.folders, "folder") if row.folders else "")


def _cells(size: int, total: int, tree: str) -> str:
    """The size, share, bar and tree cells of a row.

    Parameters
    ----------
    size : int
        The row's bytes.
    total : int
        The page's bytes, which the share is of.
    tree : str
        The tree-drawing characters before the name.

    Returns
    -------
    str
        HTML.
    """
    share = size / total if total else 0.0
    return (
        f'<span class="size">{compact(size)}</span><span class="pct">{share:.1%}</span>'
        f'<span class="bar">{bar(share)}</span><span class="tree">{tree}</span>'
        '<span class="caret"></span>'
    )


def _badges(row: Row) -> str:
    """The badges of a folder row: kind, duplicate, cold and tiny.

    Parameters
    ----------
    row : Row
        A folder row.

    Returns
    -------
    str
        HTML; empty if no badge applies.
    """
    out = ""
    if row.group_bytes >= KIND_BADGE * row.size > 0:
        out += (
            f'<span class="badge kind">{kinds.SHORT[row.group]} '
            f"{row.group_bytes / row.size:.0%}</span>"
        )
    if row.size and row.held / row.size >= DUP_BADGE:
        out += f'<span class="badge dup">dup {row.held / row.size:.0%}</span>'
    if row.size and row.cold / row.size >= COLD_BADGE:
        out += f'<span class="badge cold">cold {row.cold / row.size:.0%}</span>'
    if row.tiny:
        out += '<span class="badge tiny">tiny</span>'
    return out


def _folder(out: TextIO, row: Row, total: int, lines: str, twig: str) -> None:
    """Write a folder's row and, inside it, everything below it.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    row : Row
        The folder.
    total : int
        The page's bytes, which the shares are of.
    lines : str
        The tree-drawing characters carried down from the folders above.
    twig : str
        This folder's own branch, ``MID`` or ``LAST``; empty for the page root.
    """
    top = twig == ""
    meta = _counts(row) + (f" &middot; newest {day(row.date)}" if row.date else "")
    out.write(
        f'<details class="dir" id="d{row.dir_id}" data-path="{esc(row.path)}"'
        f'{" open" if top else ""}><summary class="row">{_cells(row.size, total, lines + twig)}'
        f'<span class="name">{esc(row.name)}</span><span class="meta">{meta}</span>'
        f"{_badges(row)}</summary>\n"
    )
    inner = lines if top else lines + ("   " if twig == LAST else "│  ")
    for i, child in enumerate(row.children):
        branch = LAST if i == len(row.children) - 1 else MID
        cells = _cells(child.size, total, inner + branch)
        if child.kind == "dir":
            _folder(out, child, total, inner, branch)
        elif child.kind == "file":
            copies = (
                f'<span class="badge copies">×{child.copies}</span>'
                if child.copies > 1
                else ""
            )
            out.write(
                f'<div class="row file" data-path="{esc(child.path)}">{cells}'
                f'<span class="name">{esc(child.name)}</span>'
                f'<span class="meta">{day(child.date)}</span>{copies}</div>\n'
            )
        else:
            word = "entry" if child.entries == 1 else "entries"
            out.write(
                f'<div class="row fold" data-path="{esc(child.path)}">{cells}'
                f'<span class="name">… {child.entries:,} more {word}</span>'
                f'<span class="meta">{_counts(child)}</span></div>\n'
            )
    out.write("</details>\n")


def _header(out: TextIO, page: Page, summary: Summary) -> None:
    """Write the header: title, banners, tiles and tools.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    page : Page
        The page's rows.
    summary : Summary
        The page's figures.
    """
    root = page.root
    made = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(summary.generated))
    when = "listed" if summary.finished else "listing begun"
    out.write(
        f'<header>\n<h1>{esc(summary.title)}</h1>\n<div class="sub">{esc(summary.db_name)} '
        f"&middot; {when} {day(summary.listed)} &middot; page made {made}</div>\n"
    )
    if summary.pending or summary.running or summary.failed:
        out.write(
            f'<div class="banner">The listing is incomplete: {summary.pending:,} pending, '
            f"{summary.running:,} running and {summary.failed:,} failed pieces. "
            "Every figure is a lower bound.</div>\n"
        )
    if page.threshold > summary.min_size:
        out.write(
            f'<div class="banner">The page would have too many rows at {compact(summary.min_size)}, '
            f"so entries under {compact(page.threshold)} are folded.</div>\n"
        )
    cold = root.cold / root.size if root.size else 0.0
    tiles = [
        (compact(root.size), "total"),
        (f"{root.files:,}", "files"),
        (f"{root.folders:,}", "folders"),
        (compact(summary.reclaimable), "reclaimable duplicates"),
        (f"{compact(root.cold)} ({cold:.0%})", "cold"),
    ]
    out.write(
        '<div class="stats">'
        + "".join(
            f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
            for value, label in tiles
        )
        + '</div>\n<div class="tools"><input type="search" id="q" placeholder="filter by path…" '
        'autocomplete="off"><button data-all="open">expand all</button>'
        '<button data-all="close">collapse all</button><span class="meta" id="count"></span>'
        "</div>\n</header>\n"
    )


def _table(
    out: TextIO,
    headings: list[str],
    rows: list[list[str]],
    left: tuple[int, ...] = (0,),
) -> None:
    """Write a table, or a note that it is empty.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    headings : list of str
        The column headings, as HTML.
    rows : list of list of str
        The cells, as HTML.
    left : tuple of int, optional
        The columns aligned left.
    """
    if not rows:
        out.write('<p class="note">None.</p>\n')
        return

    def cell(tag: str, i: int, text: str) -> str:
        """One table cell.

        Parameters
        ----------
        tag : str
            ``"th"`` or ``"td"``.
        i : int
            The column.
        text : str
            The cell's HTML.

        Returns
        -------
        str
            The element.
        """
        return (
            f'<{tag} class="left">{text}</{tag}>'
            if i in left
            else f"<{tag}>{text}</{tag}>"
        )

    out.write(
        "<table>\n<tr>"
        + "".join(cell("th", i, h) for i, h in enumerate(headings))
        + "</tr>\n"
    )
    out.writelines(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>\n"
        for row in rows
    )
    out.write("</table>\n")


def _link(e: Entry) -> str:
    """A link to a folder's row.

    Parameters
    ----------
    e : Entry
        The table entry.

    Returns
    -------
    str
        HTML; it says "(folded)" when the row is an ancestor's.
    """
    folded = " (folded)" if e.target != e.dir_id else ""
    return f'<a href="#d{e.target}">{esc(e.path or "/")}</a>{folded}'


def _tables(out: TextIO, page: Page, summary: Summary, chosen: dict) -> None:
    """Write the tables below the tree.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    page : Page
        The page's rows.
    summary : Summary
        The page's figures.
    chosen : dict
        The folder tables, from `tables`.
    """
    total = page.root.size

    def share(n: float) -> str:
        """A share of the page, as text.

        Parameters
        ----------
        n : float
            Bytes.

        Returns
        -------
        str
            A percentage of the page root's bytes to one decimal place.
        """
        return f"{n / total:.1%}" if total else "0.0%"

    out.write(
        '<h2>Biggest duplicated folders</h2>\n<p class="note">Folders with at least half '
        "their bytes also stored elsewhere, the most held elsewhere first. Two folders that "
        "name each other are copies of each other: delete at most one. These figures "
        "overlap; the reclaimable figure at the top is what deleting duplicates can "
        "actually free.</p>\n"
    )
    _table(
        out,
        ["folder", "size", "held elsewhere", "share", "held mostly in"],
        [
            [
                _link(e),
                compact(e.size),
                compact(e.held),
                f"{e.held / e.size:.0%}",
                f"{esc(e.place or '/')} ({e.place_share:.0%})" if e.place_share else "",
            ]
            for e in chosen["dup"]
        ],
        left=(0, 4),
    )
    out.write(
        f'<h2>Coldest big folders</h2>\n<p class="note">Folders with at least '
        f"{COLD_TABLE_SHARE:.0%} of their bytes last modified more than {COLD_YEARS} years "
        "before the listing, the most cold bytes first.</p>\n"
    )
    _table(
        out,
        ["folder", "size", "cold", "newest"],
        [
            [_link(e), compact(e.size), f"{e.cold / e.size:.0%}", day(e.newest)]
            for e in chosen["cold"]
        ],
    )
    out.write(
        f'<h2>Folders made of tiny files</h2>\n<p class="note">Folders of '
        f"{TINY_FILES:,} files or more averaging {TINY_MEAN >> 20} MiB or less, the most "
        "files first.</p>\n"
    )
    _table(
        out,
        ["folder", "files", "size", "mean file"],
        [
            [_link(e), f"{e.files:,}", compact(e.size), compact(e.size / e.files)]
            for e in chosen["tiny"]
        ],
    )
    out.write("<h2>Where the bytes are, by file type</h2>\n")
    _table(
        out,
        ["type", "size", "share", "files"],
        [
            [
                "(no extension)" if r.ext == "(none)" else esc("." + r.ext),
                compact(r.bytes),
                share(r.bytes),
                f"{r.files:,}",
            ]
            for r in summary.types
        ],
    )
    counted = summary.kinds or kinds.Tally(summary.listed)
    note = kinds.tarball_note(
        counted, summary.looked_inside, lambda n: compact(round(n))
    )
    out.write(
        f'<h2>Where the bytes are, by kind</h2>\n<p class="note">{esc(note)}</p>\n'
    )
    _table(
        out,
        ["kind", "size", "share", "loose", "in tarballs"],
        [
            [
                esc(group),
                compact(round(loose + tarred)),
                share(loose + tarred),
                compact(round(loose)),
                compact(round(tarred)),
            ]
            for group, (loose, tarred) in counted.groups().items()
        ],
    )
    out.write("<h2>Bytes by year last modified</h2>\n")
    _table(
        out,
        ["year", "size", "share", "files"],
        [
            [str(r.year), compact(r.bytes), share(r.bytes), f"{r.files:,}"]
            for r in summary.years
        ],
    )
    by_year = counted.group_years()
    groups = list(counted.groups())
    out.write(
        '<h2>Bytes by kind and year</h2>\n<p class="note">Each byte is dated by its '
        "oldest copy, or inside a tarball by its member's own time.</p>\n"
    )
    _table(
        out,
        ["year", *(esc(kinds.SHORT[group]) for group in groups)],
        [
            [
                str(year),
                *(
                    compact(round(by_year.get(group, {}).get(year, 0)))
                    for group in groups
                ),
            ]
            for year in sorted(
                {year for group in groups for year in by_year.get(group, {})}
            )
        ],
    )


def _reading(out: TextIO, page: Page) -> None:
    """Write the notes on how to read the page.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    page : Page
        The page's rows.
    """
    out.write(f"""<h2>Reading this</h2>
<p class="note">Folder sizes are rolled up from the files listed below them: Dropbox reports
no folder sizes. Percentages and bars are shares of this page's {compact(page.root.size)}.
Entries under {compact(page.threshold)} are folded into one "&hellip; more entries" row in
their folder.</p>
<p class="note">Dates are each file's own last-modified date, Dropbox's
<code>client_modified</code>. Dropbox's <code>server_modified</code> can be years newer than
the content.</p>
<p class="note"><b>dup N%</b>, shown from {DUP_BADGE:.0%}: the share of a folder's bytes whose
content is also stored outside it, anywhere in the listing, so deleting that folder alone would
lose none of them. <b>cold N%</b>, shown from {COLD_BADGE:.0%}: the share last modified more
than {COLD_YEARS} years before the listing. <b>tiny</b>: {TINY_FILES:,} files or more,
averaging {TINY_MEAN >> 20} MiB or less. <b>&times;N</b>: the file's content is stored N times
in the listing, this copy included.</p>
<p class="note"><b>vm N%</b> and the other kind badges, shown from {KIND_BADGE:.0%}: the kind
of data that makes up that share of a folder's bytes (sim wf for simulated waveforms, rec wf
for recorded ones, src for rupture sources, im for IM and hazard results). Kinds come from file
and folder names only, so they are a first cut. A tarball's bytes count toward the folder that
holds it: shared out over its members when the archive index lists them, otherwise judged by
the tarball's name.</p>
<p class="note"><b>Reclaimable duplicates</b> is what deleting duplicates could free on this
page while one copy of everything stays somewhere in the listing. In the folder tables, a folder
inside or above one already listed is skipped, so no two listed folders overlap.</p>
""")


def render(out: TextIO, page: Page, summary: Summary, chosen: dict) -> None:
    """Write the whole page to the text file `out`.

    Parameters
    ----------
    out : TextIO
        The text file or stream written to.
    page : Page
        The page's rows.
    summary : Summary
        The page's figures.
    chosen : dict
        The folder tables, from `tables`.
    """
    out.write(HEAD.replace("{title}", esc(summary.title)))
    _header(out, page, summary)
    out.write('<main>\n<div id="tree">\n')
    _folder(out, page.root, page.root.size, "", "")
    out.write("</div>\n")
    _tables(out, page, summary, chosen)
    _reading(out, page)
    out.write(f"</main>\n<script>\n{SCRIPT}</script>\n</body>\n</html>\n")


HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {
  --bg: #12141a; --fg: #d7dae0; --dim: #6f7683; --line: #262a33; --dir: #6fb3ff;
  --file: #cfd4dc; --fold: #6f7683; --bar-dir: #3d6ea8; --bar-file: #4c5563;
  --accent: #79d17d; --warn: #e5c07b; --dup: #d9a35b; --cold: #7fb7d9; --tiny: #c38fd6;
  --kind: #5cc8b8;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #fbfbfa; --fg: #24262b; --dim: #767b86; --line: #e4e5e9; --dir: #1f6bc0;
    --file: #3b3f47; --fold: #8a8f99; --bar-dir: #a8c8ee; --bar-file: #d3d6dc;
    --accent: #237a37; --warn: #8a6100; --dup: #9a5b12; --cold: #1f6f9a; --tiny: #7d3d99;
    --kind: #1f7a6f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 13px/1.55 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
header { position: sticky; top: 0; z-index: 2; background: var(--bg);
         padding: 18px 24px 12px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 4px; font-size: 15px; }
.sub { color: var(--dim); font-size: 12px; }
.banner { margin-top: 8px; color: var(--warn); }
.stats { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; }
.stat b { display: block; font-size: 15px; color: var(--accent); }
.stat span { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
input, button { font: inherit; color: var(--fg); background: transparent;
                border: 1px solid var(--line); border-radius: 5px; padding: 4px 9px; }
input { min-width: 240px; }
button { cursor: pointer; }
main { padding: 12px 0 60px; overflow-x: auto; }
#tree { min-width: 980px; }
.row { display: flex; align-items: baseline; gap: 10px; padding: 0 24px;
       white-space: pre; list-style: none; }
summary.row { cursor: pointer; }
summary.row::-webkit-details-marker { display: none; }
.row:hover { background: color-mix(in srgb, var(--fg) 7%, transparent); }
.size, .pct { flex: none; width: 6ch; text-align: right; }
.pct, .meta { color: var(--dim); }
.bar { flex: none; width: 28ch; overflow: hidden; color: var(--bar-file); letter-spacing: -.5px; }
.dir > summary > .bar { color: var(--bar-dir); }
.tree { flex: none; color: var(--line); }
.caret { flex: none; width: 1ch; color: var(--dim); }
details[open] > summary .caret::before { content: "\\2500"; }
details:not([open]) > summary .caret::before { content: "+"; }
.name { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.dir > summary > .name { color: var(--dir); font-weight: 600; }
.file > .name { color: var(--file); }
.fold > .name { color: var(--fold); font-style: italic; }
.meta { flex: none; font-size: 11px; }
.badge { flex: none; font-size: 11px; padding: 0 5px; border: 1px solid; border-radius: 3px; }
.badge.dup, .badge.copies { color: var(--dup); }
.badge.cold { color: var(--cold); }
.badge.tiny { color: var(--tiny); }
.badge.kind { color: var(--kind); }
.hidden { display: none !important; }
.flash > summary, .row.flash { background: color-mix(in srgb, var(--accent) 25%, transparent); }
h2 { font-size: 13px; margin: 34px 24px 4px; padding-top: 18px; border-top: 1px solid var(--line); }
.note { color: var(--dim); margin: 4px 24px; max-width: 92ch; }
table { border-collapse: collapse; margin: 8px 24px 0; }
th, td { text-align: right; padding: 2px 14px 2px 0; white-space: nowrap; }
th { color: var(--dim); font-weight: 500; }
.left { text-align: left; }
a { color: var(--dir); }
</style>
</head>
<body>
"""

SCRIPT = """const tree = document.getElementById('tree');
const q = document.getElementById('q');
const count = document.getElementById('count');
const nodes = [...tree.querySelectorAll('[data-path]')];

document.querySelectorAll('[data-all]').forEach(button => button.onclick = () => {
  const open = button.dataset.all === 'open';
  tree.querySelectorAll('details').forEach(d => d.open = open);
});

let timer;
q.oninput = () => { clearTimeout(timer); timer = setTimeout(filter, 150); };

function filter() {
  const term = q.value.trim().toLowerCase();
  let hits = 0;
  for (const n of nodes) {
    const hit = !term || n.dataset.path.toLowerCase().includes(term);
    n.classList.toggle('hidden', !hit);
    if (hit && term) hits++;
  }
  if (term) {
    // Reveal and open every match's folders, so each match can be reached.
    for (const n of nodes) {
      if (n.classList.contains('hidden')) continue;
      for (let p = n.parentElement; p && p !== tree; p = p.parentElement) {
        if (p.tagName === 'DETAILS') { p.classList.remove('hidden'); p.open = true; }
      }
    }
  }
  count.textContent = term ? hits + (hits === 1 ? ' match' : ' matches') : '';
}

function reveal(id) {
  const target = document.getElementById(id);
  if (!target) return;
  for (let p = target.parentElement; p && p !== tree; p = p.parentElement) {
    if (p.tagName === 'DETAILS') p.open = true;
  }
  target.scrollIntoView({block: 'center'});
  target.classList.add('flash');
  setTimeout(() => target.classList.remove('flash'), 1600);
}

document.querySelectorAll('a[href^="#d"]').forEach(a => a.onclick = event => {
  event.preventDefault();
  history.replaceState(null, '', a.hash);
  reveal(a.hash.slice(1));
});
if (location.hash) reveal(location.hash.slice(1));
"""
