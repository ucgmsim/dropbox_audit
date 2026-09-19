"""Rollups over one archive's manifest: what is actually inside a tar, without
reading the whole thing by hand.

Nothing here can corrupt the index it summarises: it is read-only, and it never
reaches Dropbox (T9-10). What it *can* do is mislead -- these numbers are what a
human reads to decide whether several TiB of simulation output is worth keeping,
re-publishing or deleting, so a wrong-looking-right answer here is worse than a
crash. Three design choices follow directly from that:

* Every section in `summary()`/`write_csv()` is rolled up in Python from a single
  *streamed* pass over `members` (`ArchiveStore.stream`, not `query_members`'s
  `fetchall`), inside one `read_transaction()`, rather than one query per section.
  `report`/`export` can run against a walk in progress (Task 11 does exactly
  that), and computing `n_members`, `by_extension`, `owners` and the rest from
  separately timed autocommit statements could let two sections of the same
  report describe two different moments of a manifest that is still being
  written. The explicit transaction is what makes "one snapshot" a guarantee
  rather than an accident of also happening to read everything at once -- see
  fix round 1 below, where reading everything at once turned out to have its own
  cost (521 MB / 0.79 s materialising 900,000 members, measured, to build a
  handful of small aggregate dicts). Only `largest` is a second query rather
  than part of the one pass, because it wants a different order (`size DESC`)
  that an index (`members_size`, built by `ArchiveStore.build_indexes`) answers
  directly -- sorting the whole manifest in Python to keep the top 25 is the one
  thing here with no defence once the index already exists for exactly that.
* The extension rule is imported from `dbaudit.store`, not restated. Two copies
  of "lowercased text after the last dot, bounded to 12 characters" would drift,
  and a file classified one way in the audit's own report and another in an
  archive's would be exactly the kind of disagreement that makes a reader trust
  neither.
* `cmd_archive_report`/`cmd_archive_export` (dbaudit/cli.py) are the two commands
  built on this module; they share one "this index is a lower bound" warning
  (`_print_index_completeness_warning`) since that is about how the figures are
  presented, not what they are -- fix round 1 extended it from `report` alone to
  both, on the reasoning that `export`'s whole product is files meant to be
  handed to someone else, so it needs the same caveat at least as much as a
  printed report does.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from dbaudit.store import _extension

#: Members whose `dir` is empty sit at the archive's own root (the live FaultSZ03
#: archive is all of these). Grouping them under "" would print as a blank column or
#: a blank directory name, which reads as missing data rather than "the root itself";
#: this label says what it means instead.
ROOT_LABEL = "(root)"

#: Matches `dbaudit.report.extension_profile`'s convention for a file with no
#: (or too-long a) extension, so the two reports agree on how to spell "none".
NO_EXTENSION = "(none)"


def _utc(ts: int) -> datetime:
    """The one correct conversion (T9-3): `fromtimestamp(ts, tz=timezone.utc)`, never
    `utcfromtimestamp` (deprecated since 3.12; `pytest.ini` turns that into a hard
    failure) and never `fromtimestamp(ts)` without a zone (silently the *process's*
    local zone -- correct on a UTC machine, silently wrong on any other).
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(ts: int) -> str:
    """Full timestamp, for a per-member record (`manifest.csv`) where the hour can
    matter -- e.g. telling apart files written moments apart during one run.
    """
    return _utc(ts).isoformat()


def _iso_date(ts: int) -> str:
    """Date only, for a whole-archive span where a date is enough (T9-3) -- the
    question `mtime_span` answers is "roughly when is this data from", not which
    second a particular file was written.
    """
    return _utc(ts).date().isoformat()


def _dir_prefix(dir_: str, depth: int) -> str:
    """The first `depth` components of `dir`, or `ROOT_LABEL` for a root member.

    A `dir` shorter than `depth` (e.g. `"run"` at depth 2) returns as many
    components as it has, rather than padding or falling back to the root label --
    it is a real, if shallow, member of that directory, not one with no directory.
    """
    if not dir_:
        return ROOT_LABEL
    return "/".join(dir_.split("/")[:depth])


def _bump(table: dict, key, size: int) -> None:
    entry = table.setdefault(key, {"count": 0, "bytes": 0})
    entry["count"] += 1
    entry["bytes"] += size


def _by_bytes_desc(table: dict) -> list[tuple]:
    """`(key, {count, bytes})` pairs -- a plain dict would not survive `dict(...)` in
    the caller and then back out again with a defined order, and the brief's own test
    relies on descending-by-bytes being the order a caller sees without re-sorting.
    """
    return sorted(table.items(), key=lambda kv: (-kv[1]["bytes"], kv[0]))


def _by_count_desc(table: dict) -> list[tuple]:
    return sorted(table.items(), key=lambda kv: (-kv[1]["count"], kv[0]))


def summary(store, archive_id, top: int = 25) -> dict:
    """Everything a human needs to decide what an archive is, in one dict.

    - `n_members`, `member_bytes` -- recomputed from the member rows read here
      (not `stats()`'s own copy), so they agree by construction with every other
      section below, including when a walk is still adding rows underneath this
      call.
    - `state`, `n_parts` -- so a caller can head the report with what it is
      looking at before showing a single figure.
    - `by_top_dir` -- first path component of `dir`; the same rule as
      `by_depth[1]`, and literally backed by the same table, so the two can never
      disagree with each other.
    - `by_depth` -- `{1: [...], 2: [...], 3: [...]}`, each grouped by the first
      N components of `dir` (T9-5): a run tree's faults show up at depth 2 and its
      realisations at depth 3 without anyone reading the manifest.
    - `by_extension` -- via `dbaudit.store._extension` (T9-2), never restated.
    - `largest` -- up to `top` member rows (`dir`, `name`, `size` only -- the only
      fields anything reads off it), ordered by size descending, ties broken by
      `hdr_offset` ascending. Answered by `ORDER BY size DESC, hdr_offset ASC LIMIT
      ?` against `members_size` (`ArchiveStore.build_indexes`), not by sorting the
      whole manifest in Python: measured (fix round 1) at ~0.0001 s against 300,000
      rows, including a worst-case 100,000-way tie at the top, versus ~0.19 s to
      materialise and sort the same rows in Python. The `hdr_offset` tie-break
      matters, not just for determinism: it is exactly the order a stable Python
      sort of `query_members`'s `hdr_offset`-ordered rows used to produce, so this
      query reproduces the pre-existing tie order rather than an arbitrary new one.
    - `mtime_span` -- `(earliest, latest)` as UTC dates, or `None` for an archive
      with no members (nothing to guess a span from).
    - `owners` -- `(uname, gname)` pairs with counts and bytes.
    - `types` -- counts by the raw tar type flag (`Member.from_tarinfo`'s own
      single-character encoding: `"0"` regular, `"5"` directory, `"2"` symlink,
      `"1"` hard link, ...), so directories, symlinks and hard links are visible
      rather than folded into "files".

    Every section above tolerates zero members (T9-9): the loop below simply does
    not run, `largest` and every rollup come back empty, and `mtime_span` is
    `None` rather than crashing on `max()` of nothing.

    The whole function runs inside one `store.read_transaction()`: the archive-row
    lookup, the streamed per-member pass, and the `largest` query all see the one
    snapshot that transaction opens, rather than whatever a walker happens to have
    committed between three separate autocommit statements.
    """
    top = max(int(top), 0)

    by_depth = {1: {}, 2: {}, 3: {}}
    by_extension: dict = {}
    owners: dict = {}
    types: dict = {}
    n_members = 0
    member_bytes = 0
    earliest = latest = None

    with store.read_transaction():
        archive_row = store.query(
            "SELECT state, n_parts FROM archives WHERE id=?", (archive_id,))[0]

        # Only the columns this loop actually reads -- not `SELECT *`, which would
        # also carry `linkname`/`hdr_offset`/`data_offset`/`archive_id` through
        # every one of possibly millions of rows for nothing. Streamed via `stream`
        # (a generator over the cursor), not `query_members`'s `fetchall`, so peak
        # memory here is the size of the aggregate dicts below, not the manifest.
        for member in store.stream(
            "SELECT dir, name, size, mtime, uname, gname, type FROM members "
            "WHERE archive_id=?", (archive_id,),
        ):
            size = member["size"]
            n_members += 1
            member_bytes += size
            dir_ = member["dir"] or ""
            for depth in (1, 2, 3):
                _bump(by_depth[depth], _dir_prefix(dir_, depth), size)

            ext = _extension(member["name"]) or NO_EXTENSION
            _bump(by_extension, ext, size)

            _bump(owners, (member["uname"] or "", member["gname"] or ""), size)
            _bump(types, member["type"], size)

            mtime = member["mtime"]
            if mtime is not None:
                earliest = mtime if earliest is None else min(earliest, mtime)
                latest = mtime if latest is None else max(latest, mtime)

        largest = store.query(
            "SELECT dir, name, size FROM members WHERE archive_id=? "
            "ORDER BY size DESC, hdr_offset ASC LIMIT ?", (archive_id, top))

    return {
        "state": archive_row["state"],
        "n_parts": archive_row["n_parts"],
        "n_members": n_members,
        "member_bytes": member_bytes,
        "by_top_dir": _by_bytes_desc(by_depth[1]),
        "by_depth": {depth: _by_bytes_desc(table) for depth, table in by_depth.items()},
        "by_extension": _by_bytes_desc(by_extension),
        "largest": largest,
        "mtime_span": (_iso_date(earliest), _iso_date(latest)) if earliest is not None else None,
        "owners": _by_count_desc(owners),
        "types": _by_count_desc(types),
    }


def write_csv(store, archive_id, out_dir) -> list[str]:
    """Write `manifest.csv`, `by_dir.csv` and `by_extension.csv` into `out_dir`.

    `out_dir` is created (`parents=True, exist_ok=True`) rather than assumed to
    exist (T9-6): the brief's own test passes a directory that does not exist yet.
    Every file is opened with `newline=""`, as the `csv` module requires, and
    every row goes through `csv.writer` rather than manual string-joining -- a tar
    member name can and does contain a comma or a quote, and only the `csv` module
    is trusted to round-trip that correctly.

    `by_dir.csv` groups by the *whole* `dir` string (one row per distinct
    directory, however deep), unlike `summary()`'s `by_top_dir`/`by_depth`, which
    roll up to the first N components -- this file is the detailed companion to
    those coarser figures, not a third copy of the same rollup.

    `manifest.csv` (the larger of the three outputs by far) is written from one
    `store.stream()` pass inside one `store.read_transaction()`, not
    `query_members`'s `fetchall` -- see `summary()`'s docstring for the 521 MB /
    0.79 s a full materialisation cost on a 900,000-member archive. `by_dir`/
    `by_extension` are accumulated into two small dicts *while* that same pass
    writes `manifest.csv`'s rows, rather than read a second time: one streamed
    scan of `members`, not two.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    by_dir: dict = {}
    by_extension: dict = {}

    manifest_path = out_dir / "manifest.csv"
    with store.read_transaction():
        with open(manifest_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "size", "mtime", "mode", "uname", "gname", "type",
                              "hdr_offset", "data_offset"])
            for member in store.stream(
                "SELECT dir, name, size, mtime, mode, uname, gname, type, hdr_offset, "
                "data_offset FROM members WHERE archive_id=? ORDER BY hdr_offset",
                (archive_id,),
            ):
                path = f"{member['dir']}/{member['name']}" if member["dir"] else member["name"]
                mtime = _iso(member["mtime"]) if member["mtime"] is not None else ""
                mode = f"{(member['mode'] or 0):04o}"
                writer.writerow([path, member["size"], mtime, mode, member["uname"] or "",
                                  member["gname"] or "", member["type"], member["hdr_offset"],
                                  member["data_offset"]])
                _bump(by_dir, member["dir"] or ROOT_LABEL, member["size"])
                _bump(by_extension, _extension(member["name"]) or NO_EXTENSION, member["size"])
    written.append(str(manifest_path))

    dir_path = out_dir / "by_dir.csv"
    with open(dir_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dir", "count", "bytes"])
        writer.writerows([key, agg["count"], agg["bytes"]] for key, agg in _by_bytes_desc(by_dir))
    written.append(str(dir_path))

    ext_path = out_dir / "by_extension.csv"
    with open(ext_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["extension", "count", "bytes"])
        writer.writerows([key, agg["count"], agg["bytes"]]
                          for key, agg in _by_bytes_desc(by_extension))
    written.append(str(ext_path))

    return written
