"""The browsable tree page: one self-contained HTML file showing where the space goes.

Four steps, one job each. `measure` turns the database into figures for every folder,
kept in arrays indexed by dir id so that two million folders cost about 160 MB.
`layout` decides which folders and files get a row of their own and which fold away,
and `tables` picks the folders worth a closer look. `render` writes the HTML.
"""

from __future__ import annotations

import time
from array import array
from collections import Counter, defaultdict
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


# Every content stored more than once, copy by copy, in content order. Empty files
# are left out: they all share one hash and hold no bytes.
DUPLICATES = (
    "SELECT content_hash, dir_id, size, rowid FROM files "
    "WHERE size > 0 AND content_hash IN ("
    "  SELECT content_hash FROM files WHERE size > 0 AND content_hash IS NOT NULL "
    "  GROUP BY content_hash HAVING COUNT(*) > 1) "
    "ORDER BY content_hash"
)


def duplicate_groups(conn, top: int):
    """Yield each duplicated content's copies as (dir_id, size, rowid), one content at a
    time, so memory stays flat however many duplicates there are."""
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


def common_folder(dirs: list[int], parent, depth) -> int:
    """The deepest folder holding every one of `dirs`: their lowest common ancestor."""
    meet = dirs[0]
    for other in dirs[1:]:
        while depth[meet] > depth[other]:
            meet = parent[meet]
        while depth[other] > depth[meet]:
            other = parent[other]
        while meet != other:
            meet, other = parent[meet], parent[other]
    return meet


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

    for group in duplicate_groups(conn, top):
        for _, size, rowid in group:
            if size >= min_size:
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
    return (figures.files[d] >= TINY_FILES
            and figures.size[d] <= figures.files[d] * TINY_MEAN)


@dataclass
class Row:
    kind: str                 # "dir", "file" or "fold"
    name: str
    path: str                 # below the crawl root; a fold row carries its folder's
    size: int
    files: int = 0
    folders: int = 0
    date: int = 0             # a folder's newest file, or a file's own date
    dir_id: int = 0
    copies: int = 1           # a file's content: copies in the listing, this one included
    held: int = 0
    cold: int = 0
    tiny: bool = False
    entries: int = 0          # a fold row: the folder's direct children folded into it
    children: list = field(default_factory=list)


@dataclass
class Page:
    root: Row
    threshold: int            # entries below this fold
    rows: int
    shown: set                # dir ids with a row of their own


def below(path: str, crawl_root: str) -> str:
    """`path` relative to the crawl root, as `data-path` carries it."""
    root = crawl_root.rstrip("/")
    if path.lower() == root.lower():
        return ""
    if root and path.lower().startswith(root.lower() + "/"):
        return path[len(root) + 1:]
    return path.lstrip("/")


def dir_names(conn, ids) -> dict[int, tuple[str, str]]:
    """{dir id: (name, path_display)}, looked up a chunk at a time."""
    ids, out = list(ids), {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        out.update((d, (name, path)) for d, name, path in conn.execute(
            f"SELECT id, name, path_display FROM dirs WHERE id IN ({','.join('?' * len(chunk))})",
            chunk))
    return out


def folder_row(f: Figures, d: int, name: str, path: str) -> Row:
    return Row("dir", name, path, f.size[d], files=f.files[d], folders=f.folders[d],
               date=f.newest[d], dir_id=d, held=f.held[d], cold=f.cold[d],
               tiny=is_tiny(f, d))


def layout(store, figures: Figures, threshold: int, max_rows: int) -> Page:
    f, root = figures, figures.page_root
    conn = store.connect()
    top = len(f.size)
    folders = [d for d in range(1, top) if f.on_page[d] and d != root and f.size[d] >= threshold]
    files = [row for row in conn.execute(
                 "SELECT rowid, dir_id, name, size, client_modified FROM files WHERE size >= ?",
                 (threshold,))
             if 0 < row[1] < top and f.on_page[row[1]]]

    def rows_at(t):
        shown = [d for d in folders if f.size[d] >= t]
        subfolders = Counter(f.parent[d] for d in shown)
        own = Counter(row[1] for row in files if row[3] >= t)
        folds = sum(1 for d in [root, *shown]
                    if f.kids[d] > subfolders[d] or f.own_files[d] > own[d])
        return 1 + len(shown) + sum(own.values()) + folds

    rows = rows_at(threshold)
    while rows > max_rows and threshold <= f.size[root]:
        threshold = max(threshold * 2, 1)
        rows = rows_at(threshold)

    shown = {d for d in folders if f.size[d] >= threshold}
    names = dir_names(conn, shown | {root})
    crawl_root = store.get_meta("root") or ""
    inside = defaultdict(list)
    for d in shown:
        name, path = names[d]
        inside[f.parent[d]].append(folder_row(f, d, name, below(path, crawl_root)))
    for rowid, dir_id, name, size, modified in files:
        if size >= threshold:
            folder = below(names[dir_id][1], crawl_root)
            inside[dir_id].append(Row("file", name, f"{folder}/{name}" if folder else name,
                                      size, files=1, date=modified or 0,
                                      copies=f.copies.get(rowid, 1)))

    def assemble(row):
        children = sorted(inside.get(row.dir_id, ()), key=lambda r: (-r.size, r.name))
        for child in children:
            if child.kind == "dir":
                assemble(child)
        subfolders = [c for c in children if c.kind == "dir"]
        entries = (f.kids[row.dir_id] - len(subfolders)
                   + f.own_files[row.dir_id] - (len(children) - len(subfolders)))
        if entries:
            children.append(Row("fold", "", row.path, row.size - sum(c.size for c in children),
                                files=row.files - sum(c.files for c in children),
                                folders=row.folders - sum(c.folders + 1 for c in subfolders),
                                entries=entries))
        row.children = children

    name, path = names[root]
    page_row = folder_row(f, root, name, below(path, crawl_root))
    assemble(page_row)
    return Page(page_row, threshold, rows, shown | {root})


# ---- tables: the folders worth a closer look --------------------------------


def ancestors(d: int, parent):
    d = parent[d]
    while d:
        yield d
        d = parent[d]


@dataclass
class Entry:
    dir_id: int
    path: str
    target: int               # the row its link opens: itself, or its nearest shown ancestor
    size: int
    files: int
    held: int
    cold: int
    newest: int
    place: str = ""           # where most of its held-elsewhere bytes have their copies
    place_share: float = 0.0


def choose(ranked, parent, limit: int = TABLE_ROWS) -> list[int]:
    """Take folders in ranking order, skipping any inside or above one already taken."""
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


def place(d: int, near: set, parent, depth, floor: int) -> int:
    """Where a copy in folder `d` lies, seen from a folder whose ancestry is `near`.

    The copy's ancestor one level below where the two paths part, never shallower than
    `floor`; a copy shallower than that is its own place.
    """
    chain = [d]
    while chain[-1] and chain[-1] not in near:
        chain.append(parent[chain[-1]])
    level = max(depth[chain[-1]] + 1, floor)
    for step in chain:
        if depth[step] <= level:
            return step
    return d


def counterparts(conn, figures: Figures, folders: list[int], floor: int) -> dict:
    """{folder: (place, bytes)}: the place holding copies of the most of each folder's
    held-elsewhere bytes. A second pass over the duplicates, for these folders only."""
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
            for spot in {place(d, near[mine], f.parent, f.depth, floor)
                         for d, _, _ in group if owner[d] != mine}:
                credit[mine][spot] += inside
    out = {}
    for d in folders:
        if credit[d]:
            spot = max(credit[d], key=lambda p: (credit[d][p], -p))
            out[d] = (spot, credit[d][spot])
    return out


def tables(store, figures: Figures, page: Page) -> dict[str, list[Entry]]:
    f, root = figures, figures.page_root
    below_root = [d for d in range(1, len(f.size)) if f.on_page[d] and d != root]
    big = [d for d in below_root if f.size[d] >= page.threshold]
    dup = choose(sorted((d for d in big if f.held[d] and f.held[d] >= DUP_TABLE_SHARE * f.size[d]),
                        key=lambda d: (-f.held[d], d)), f.parent)
    cold = choose(sorted((d for d in big if f.cold[d] and f.cold[d] >= COLD_TABLE_SHARE * f.size[d]),
                         key=lambda d: (-f.cold[d], d)), f.parent)
    tiny = choose(sorted((d for d in below_root if is_tiny(f, d)),
                         key=lambda d: (-f.files[d], d)), f.parent)

    conn = store.connect()
    crawl_root = store.get_meta("root") or ""
    spots = counterparts(conn, f, dup, crawl_root.rstrip("/").count("/") + 2)
    names = dir_names(conn, {*dup, *cold, *tiny, *(spot for spot, _ in spots.values())})

    def entry(d):
        target = d
        while target not in page.shown:
            target = f.parent[target]
        e = Entry(d, below(names[d][1], crawl_root), target, f.size[d], f.files[d],
                  f.held[d], f.cold[d], f.newest[d])
        if d in spots:
            spot, credit = spots[d]
            e.place, e.place_share = below(names[spot][1], crawl_root), credit / f.held[d]
        return e

    return {"dup": [entry(d) for d in dup], "cold": [entry(d) for d in cold],
            "tiny": [entry(d) for d in tiny]}
