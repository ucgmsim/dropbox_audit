"""The browsable tree page: one self-contained HTML file showing where the space goes.

Four steps, one job each. `measure` turns the database into figures for every folder,
kept in arrays indexed by dir id so that two million folders cost about 160 MB.
`layout` decides which folders and files get a row of their own and which fold away,
and `tables` picks the folders worth a closer look. `render` writes the HTML.
"""

from __future__ import annotations

import heapq
import html
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import report

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
    copies: array = field(default_factory=lambda: array("I"))  # by files.rowid: copies, or 0
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
    rowids = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM files").fetchone()[0] + 1

    def zeros():
        return array("q", [0]) * top

    f = Figures(parent=zeros(), depth=zeros(), order=array("q"), kids=zeros(),
                own_files=zeros(), files=zeros(), folders=zeros(), size=zeros(),
                newest=zeros(), cold=zeros(), held=zeros(), on_page=bytearray(top),
                page_root=page_root, copies=array("I", [0]) * rowids)
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


def chunked(conn, query: str, ids):
    """The rows `query` finds for `ids`, a chunk at a time; it ends in `IN ({})`."""
    ids = list(ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        yield from conn.execute(query.format(",".join("?" * len(chunk))), chunk)


def dir_names(conn, ids) -> dict[int, tuple[str, str]]:
    """{dir id: (name, path_display)}."""
    return {d: (name, path) for d, name, path in
            chunked(conn, "SELECT id, name, path_display FROM dirs WHERE id IN ({})", ids)}


def past(threshold: int, size: int) -> int:
    """The first threshold, doubling from `threshold`, that is above `size`."""
    while threshold <= size:
        threshold = max(threshold * 2, 1)
    return threshold


def folder_row(f: Figures, d: int, name: str, path: str) -> Row:
    return Row("dir", name, path, f.size[d], files=f.files[d], folders=f.folders[d],
               date=f.newest[d], dir_id=d, held=f.held[d], cold=f.cold[d],
               tiny=is_tiny(f, d))


def layout(store, figures: Figures, threshold: int, max_rows: int) -> Page:
    f, root = figures, figures.page_root
    conn = store.connect()
    top = len(f.size)
    # Each folder and file at or above the threshold gets a row, so while more than
    # max_rows of either qualify the page cannot fit: start past the (max_rows + 1)-th
    # largest. Memory then follows --max-rows, whatever --min-size is.
    folders = heapq.nlargest(max_rows + 1, (d for d in range(1, top) if f.on_page[d]
                                            and d != root and f.size[d] >= threshold),
                             key=lambda d: f.size[d])
    if len(folders) > max_rows:
        threshold = past(threshold, f.size[folders[-1]])
    files = heapq.nlargest(max_rows + 1, (row for row in conn.execute(
                               "SELECT rowid, dir_id, size FROM files WHERE size >= ?", (threshold,))
                           if 0 < row[1] < top and f.on_page[row[1]]), key=lambda row: row[2])
    if len(files) > max_rows:
        threshold = past(threshold, files[-1][2])

    def rows_at(t):
        shown = [d for d in folders if f.size[d] >= t]
        subfolders = Counter(f.parent[d] for d in shown)
        own = Counter(row[1] for row in files if row[2] >= t)
        folds = sum(1 for d in [root, *shown]
                    if f.kids[d] > subfolders[d] or f.own_files[d] > own[d])
        return 1 + len(shown) + sum(own.values()) + folds

    rows = rows_at(threshold)
    while rows > max_rows and threshold <= f.size[root]:
        threshold = max(threshold * 2, 1)
        rows = rows_at(threshold)

    shown = {d for d in folders if f.size[d] >= threshold}
    names = dir_names(conn, shown | {root})
    details = {rowid: (name, modified) for rowid, name, modified in chunked(
        conn, "SELECT rowid, name, client_modified FROM files WHERE rowid IN ({})",
        [row[0] for row in files if row[2] >= threshold])}
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
            inside[dir_id].append(Row("file", name, f"{folder}/{name}" if folder else name,
                                      size, files=1, date=modified or 0, copies=copies or 1))

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


# ---- build: everything the page shows ---------------------------------------


@dataclass
class Summary:
    title: str                # the page root's path
    db_name: str
    listed: float             # when the listing was taken
    finished: bool            # a pass over the listing finished
    generated: float
    reclaimable: int
    pending: int
    running: int
    failed: int
    min_size: int
    types: list               # report.ExtRow
    years: list               # report.YearRow


def build(store, page_path: str, min_size: int, max_rows: int, db_name: str):
    """(page, summary, tables) for the folder at `page_path`."""
    conn = store.connect()
    # One read transaction for every query below: read separately, a crawl still writing
    # could show measure one listing and layout another.
    conn.execute("BEGIN")
    try:
        listed, finished = crawl_time(store)
        crawl_root = store.get_meta("root") or ""
        found = find_dir(store, page_path)
        if found is None:  # nothing listed yet
            page, chosen, reclaimable = (Page(Row("dir", page_path, "", 0), min_size, 1, {0}),
                                         {"dup": [], "cold": [], "tiny": []}, 0)
        else:
            figures = measure(store, listed, min_size, found[0])
            page = layout(store, figures, min_size, max_rows)
            chosen = tables(store, figures, page)
            reclaimable = figures.reclaimable
        whole = page_path.rstrip("/").lower() == crawl_root.rstrip("/").lower()
        under = None if whole else page_path
        summary = Summary(
            title=found[1] if found else page_path, db_name=db_name, listed=listed,
            finished=finished, generated=time.time(), reclaimable=reclaimable,
            pending=store.pending_count(), running=store.running_count(),
            failed=store.progress()["shards_error"], min_size=min_size,
            types=report.extension_profile(store, limit=TYPE_ROWS, under=under),
            years=report.cold_bytes(store, under=under))
        return page, summary, chosen
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")


# ---- render: the HTML -------------------------------------------------------


MID, LAST = "├─ ", "└─ "
BLOCKS = " ▏▎▍▌▋▊▉"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def compact(n: float) -> str:
    """A size in at most four characters, the way the reference page shows them."""
    for unit in "BKMGTP":
        if n < 1024 or unit == "P":
            break
        n /= 1024
    if unit == "B":
        return f"{int(n)}B"
    return f"{n:.1f}{unit}" if n < 100 else f"{n:.0f}{unit}"


def bar(share: float) -> str:
    eighths = round(max(0.0, min(share, 1.0)) * BAR_WIDTH * 8)
    full, part = divmod(eighths, 8)
    return "█" * full + (BLOCKS[part] if part else "")


def day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t)) if t else ""


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _counts(row: Row) -> str:
    text = plural(row.files, "file")
    return text + (", " + plural(row.folders, "folder") if row.folders else "")


def _cells(size: int, total: int, tree: str) -> str:
    share = size / total if total else 0.0
    return (f'<span class="size">{compact(size)}</span><span class="pct">{share:.1%}</span>'
            f'<span class="bar">{bar(share)}</span><span class="tree">{tree}</span>'
            '<span class="caret"></span>')


def _badges(row: Row) -> str:
    out = ""
    if row.size and row.held / row.size >= DUP_BADGE:
        out += f'<span class="badge dup">dup {row.held / row.size:.0%}</span>'
    if row.size and row.cold / row.size >= COLD_BADGE:
        out += f'<span class="badge cold">cold {row.cold / row.size:.0%}</span>'
    if row.tiny:
        out += '<span class="badge tiny">tiny</span>'
    return out


def _folder(out, row: Row, total: int, lines: str, twig: str) -> None:
    """Write a folder's row and, inside it, everything below it."""
    top = twig == ""
    meta = _counts(row) + (f" &middot; newest {day(row.date)}" if row.date else "")
    out.write(f'<details class="dir" id="d{row.dir_id}" data-path="{esc(row.path)}"'
              f'{" open" if top else ""}><summary class="row">{_cells(row.size, total, lines + twig)}'
              f'<span class="name">{esc(row.name)}</span><span class="meta">{meta}</span>'
              f'{_badges(row)}</summary>\n')
    inner = lines if top else lines + ("   " if twig == LAST else "│  ")
    for i, child in enumerate(row.children):
        branch = LAST if i == len(row.children) - 1 else MID
        cells = _cells(child.size, total, inner + branch)
        if child.kind == "dir":
            _folder(out, child, total, inner, branch)
        elif child.kind == "file":
            copies = f'<span class="badge copies">×{child.copies}</span>' if child.copies > 1 else ""
            out.write(f'<div class="row file" data-path="{esc(child.path)}">{cells}'
                      f'<span class="name">{esc(child.name)}</span>'
                      f'<span class="meta">{day(child.date)}</span>{copies}</div>\n')
        else:
            word = "entry" if child.entries == 1 else "entries"
            out.write(f'<div class="row fold" data-path="{esc(child.path)}">{cells}'
                      f'<span class="name">… {child.entries:,} more {word}</span>'
                      f'<span class="meta">{_counts(child)}</span></div>\n')
    out.write("</details>\n")


def _header(out, page: Page, summary: Summary) -> None:
    root = page.root
    made = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(summary.generated))
    when = "listed" if summary.finished else "listing begun"
    out.write(f'<header>\n<h1>{esc(summary.title)}</h1>\n<div class="sub">{esc(summary.db_name)} '
              f'&middot; {when} {day(summary.listed)} &middot; page made {made}</div>\n')
    if summary.pending or summary.running or summary.failed:
        out.write(f'<div class="banner">The listing is incomplete: {summary.pending:,} pending, '
                  f'{summary.running:,} running and {summary.failed:,} failed pieces. '
                  'Every figure is a lower bound.</div>\n')
    if page.threshold > summary.min_size:
        out.write(f'<div class="banner">The page would have too many rows at {compact(summary.min_size)}, '
                  f'so entries under {compact(page.threshold)} are folded.</div>\n')
    cold = root.cold / root.size if root.size else 0.0
    tiles = [(compact(root.size), "total"), (f"{root.files:,}", "files"),
             (f"{root.folders:,}", "folders"), (compact(summary.reclaimable), "reclaimable duplicates"),
             (f"{compact(root.cold)} ({cold:.0%})", "cold")]
    out.write('<div class="stats">' + "".join(
        f'<div class="stat"><b>{value}</b><span>{label}</span></div>' for value, label in tiles)
        + '</div>\n<div class="tools"><input type="search" id="q" placeholder="filter by path…" '
          'autocomplete="off"><button data-all="open">expand all</button>'
          '<button data-all="close">collapse all</button><span class="meta" id="count"></span>'
          '</div>\n</header>\n')


def _table(out, headings: list[str], rows: list[list[str]], left=(0,)) -> None:
    if not rows:
        out.write('<p class="note">None.</p>\n')
        return

    def cell(tag, i, text):
        return f'<{tag} class="left">{text}</{tag}>' if i in left else f"<{tag}>{text}</{tag}>"

    out.write("<table>\n<tr>" + "".join(cell("th", i, h) for i, h in enumerate(headings))
              + "</tr>\n")
    for row in rows:
        out.write("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>\n")
    out.write("</table>\n")


def _link(e: Entry) -> str:
    folded = " (folded)" if e.target != e.dir_id else ""
    return f'<a href="#d{e.target}">{esc(e.path or "/")}</a>{folded}'


def _tables(out, page: Page, summary: Summary, chosen: dict) -> None:
    total = page.root.size

    def share(n):
        return f"{n / total:.1%}" if total else "0.0%"

    out.write('<h2>Biggest duplicated folders</h2>\n<p class="note">Folders with at least half '
              'their bytes also stored elsewhere, the most held elsewhere first. Two folders that '
              'name each other are copies of each other: delete at most one. These figures '
              'overlap; the reclaimable figure at the top is what deleting duplicates can '
              'actually free.</p>\n')
    _table(out, ["folder", "size", "held elsewhere", "share", "held mostly in"],
           [[_link(e), compact(e.size), compact(e.held), f"{e.held / e.size:.0%}",
             f"{esc(e.place or '/')} ({e.place_share:.0%})" if e.place_share else ""]
            for e in chosen["dup"]], left=(0, 4))
    out.write(f'<h2>Coldest big folders</h2>\n<p class="note">Folders with at least '
              f'{COLD_TABLE_SHARE:.0%} of their bytes last modified more than {COLD_YEARS} years '
              'before the listing, the most cold bytes first.</p>\n')
    _table(out, ["folder", "size", "cold", "newest"],
           [[_link(e), compact(e.size), f"{e.cold / e.size:.0%}", day(e.newest)]
            for e in chosen["cold"]])
    out.write(f'<h2>Folders made of tiny files</h2>\n<p class="note">Folders of '
              f'{TINY_FILES:,} files or more averaging {TINY_MEAN >> 20} MiB or less, the most '
              'files first.</p>\n')
    _table(out, ["folder", "files", "size", "mean file"],
           [[_link(e), f"{e.files:,}", compact(e.size), compact(e.size / e.files)]
            for e in chosen["tiny"]])
    out.write("<h2>Where the bytes are, by file type</h2>\n")
    _table(out, ["type", "size", "share", "files"],
           [["(no extension)" if r.ext == "(none)" else esc("." + r.ext), compact(r.bytes),
             share(r.bytes), f"{r.files:,}"] for r in summary.types])
    out.write("<h2>Bytes by year last modified</h2>\n")
    _table(out, ["year", "size", "share", "files"],
           [[str(r.year), compact(r.bytes), share(r.bytes), f"{r.files:,}"]
            for r in summary.years])


def _reading(out, page: Page) -> None:
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
<p class="note"><b>Reclaimable duplicates</b> is what deleting duplicates could free on this
page while one copy of everything stays somewhere in the listing. In the folder tables, a folder
inside or above one already listed is skipped, so no two listed folders overlap.</p>
""")


def render(out, page: Page, summary: Summary, chosen: dict) -> None:
    """Write the whole page to the text file `out`."""
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
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #fbfbfa; --fg: #24262b; --dim: #767b86; --line: #e4e5e9; --dir: #1f6bc0;
    --file: #3b3f47; --fold: #8a8f99; --bar-dir: #a8c8ee; --bar-file: #d3d6dc;
    --accent: #237a37; --warn: #8a6100; --dup: #9a5b12; --cold: #1f6f9a; --tiny: #7d3d99;
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
