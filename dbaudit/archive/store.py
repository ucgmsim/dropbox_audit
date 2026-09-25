"""SQLite store for the archive content index.

The one invariant everything else depends on: **a batch's member rows and the
segment cursor that resumes it are written in a single transaction.** Kill the
process at any instant and SQLite rolls back the partial batch, so a restart
re-walks exactly that batch and nothing is lost or duplicated.

Segments add a second axis on top of the audit's own shard model
(``dbaudit/store.py``): a split archive is walked one chain per part, so each
segment can be claimed, reset or retired independently of its neighbours while
Task 7 reconciles where each chain actually starts and ends.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from pathlib import Path

from dbaudit.archive.parts import ArchiveSet, Part

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Segment states whose span is fully accounted for -- the walk that owned it ran
#: to one of these outcomes, so every byte of the span counts as covered regardless
#: of exactly where the cursor or first_header sit. 'error' is deliberately absent:
#: a segment that failed has not necessarily covered anything.
FINISHED_SEGMENT_STATES = frozenset({"crossed", "complete", "truncated", "corrupt", "beyond"})


def _as_int(value):
    """Raise rather than coerce: a bad size must stop the batch, not silently NULL it.

    This is what makes `commit_batch`'s atomicity meaningful to test -- see its
    docstring for why the conversion happens mid-transaction rather than before.
    """
    if not isinstance(value, int):
        raise ValueError(f"member size is not an int: {value!r}")
    return value


class ArchiveStore:
    def __init__(self, path):
        self.path = str(path)
        self._local = threading.local()

    # ---- connections and schema -----------------------------------------

    def connect(self) -> sqlite3.Connection:
        """One connection per thread. Task 7 shares one ArchiveStore across a thread
        pool -- one chain per part -- so each worker gets its own autocommit
        connection while the main thread reads `segments`/`stats` between rounds.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def init_schema(self) -> None:
        self.connect().executescript(SCHEMA_PATH.read_text())

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        """Mirrors `Store.query` (dbaudit/store.py:166) -- the seam Task 9's report
        module is built on, rather than reaching past this store into SQL the way
        Task 7's review flagged `cmd_archive_status` for doing. Rows come back as
        `sqlite3.Row` (this connection's own `row_factory`), so callers can index a
        result by column name as well as by position.

        For a result set worth holding in a Python list -- one row, or a handful --
        not for a scan over every member of a multi-TiB archive; see `stream` for that.
        """
        return self.connect().execute(sql, params).fetchall()

    def stream(self, sql: str, params=()):
        """Like `query`, but yields rows one at a time instead of calling `fetchall`.

        Added for Task 9's fix round 1: `report.summary`/`write_csv` used to read
        every member of an archive into one Python list before rolling it up, which a
        review measured at 521 MB / 0.79 s for a 900,000-row manifest -- the whole
        point of that pass is a handful of small aggregate dicts, not a second copy of
        the table in memory. A generator can't grow a `.fetchall()` call back onto
        this by accident the way returning the raw cursor could, so callers that need
        genuine streaming get it by construction, not by convention.

        Must be consumed inside the same connection's `read_transaction()` as any
        other read it needs to agree with -- on its own, an autocommit statement here
        is no more of a snapshot than a bare `query()` call is.
        """
        yield from self.connect().execute(sql, params)

    @contextlib.contextmanager
    def read_transaction(self):
        """A read-only `BEGIN`/`COMMIT` around a caller's sequence of `query`/`stream`
        calls, so they provably see one snapshot instead of however many happen to
        line up from separate autocommit statements landing between a concurrent
        walker's commits. This connection is WAL (`connect()`), so a reader here never
        blocks -- or is blocked by -- `commit_batch` writing; it costs nothing beyond
        an ordinary read when nothing else happens to be running at the same moment.

        Keep this even where a single query would look sufficient on its own: that
        sufficiency is what makes it tempting to delete as redundant, and deleting it
        is exactly what would turn "one snapshot, guaranteed" back into "one snapshot,
        by accident of how the code happens to be written today."
        """
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def log_event(self, archive_id, kind: str, detail: str = "") -> None:
        """One row per notable happening -- retries, outcomes, run starts -- so a
        post-mortem on a stalled walk has something to read besides state columns.
        """
        self.connect().execute(
            "INSERT INTO events (ts, archive_id, kind, detail) VALUES (?, ?, ?, ?)",
            (time.time(), archive_id, kind, (detail or "")[:500]))

    # ---- archives ---------------------------------------------------------

    def register(self, name, kind, folder, source, parts) -> int:
        """Insert or update an archive keyed by the content of its parts, not by
        name or folder -- folders get rearranged (v01p0_incomplete moved), but the
        part contents are what actually identify an archive.

        The identity hash is `ArchiveSet.set_hash()` (Task 2), never re-derived
        here: two copies of an identity rule is exactly the duplication that
        drifts. Re-registering the same content updates the existing row (parts
        are replaced wholesale, so a moved part's new path_display takes effect).
        Re-registering the same name and folder with *different* content creates a
        new row and flags every other archive of that name and folder as `stale`:
        the folder now holds a different a.tar than the one that was indexed, so
        the old index of it can no longer be trusted. A same-named archive in a
        different folder is untouched -- FaultSZ03_Source.tar exists in many fault
        folders and they are genuinely different archives.

        Landing on the `ON CONFLICT` branch means the *same* set_hash was seen
        again -- the live parts hash back to this archive's identity, so if this
        row was `stale` it no longer is. That is the other half of the stale
        story: `commit_batch`/`start_walk`/`finish` refuse to overwrite a `stale`
        state (see their docstrings), but only `register` can be the one to lift
        it, and only when the content actually verifies again. Both the `state`
        and the `detail` reset are conditional on the row currently being
        `stale`, so a merely-in-progress or already-`complete` archive being
        re-registered (a routine re-listing of its parts) is not reset to
        `registered` -- only a `stale` one is.

        `unsupported` is lifted the same way. It is terminal for `index` and `cat`,
        and re-registering is the one way back should it ever be reached wrongly; a
        real pax archive loses nothing, since the next walk meets its header again.
        """
        parts = list(parts)
        set_hash = ArchiveSet(parts).set_hash()
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO archives (name, kind, source, folder, total_size, n_parts,
                                         set_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(set_hash) DO UPDATE SET
                       name=excluded.name, folder=excluded.folder, source=excluded.source,
                       state=CASE WHEN state IN ('stale', 'unsupported') THEN 'registered'
                                  ELSE state END,
                       detail=CASE WHEN state IN ('stale', 'unsupported') THEN NULL
                                   ELSE detail END""",
                (name, kind, source, folder, sum(p.size for p in parts), len(parts),
                 set_hash, time.time()))
            archive_id = conn.execute(
                "SELECT id FROM archives WHERE set_hash=?", (set_hash,)).fetchone()[0]
            conn.execute("DELETE FROM parts WHERE archive_id=?", (archive_id,))
            conn.executemany(
                """INSERT INTO parts (archive_id, idx, name, path_display, dbx_id, rev,
                                      size, offset, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(archive_id, p.idx, p.name, p.path_display, p.dbx_id, p.rev, p.size,
                  p.offset, p.content_hash) for p in parts])
            conn.execute(
                """UPDATE archives SET state='stale', detail=?
                   WHERE name=? AND folder IS ? AND set_hash!=? AND id!=?""",
                (f"superseded by archive {archive_id}", name, folder, set_hash, archive_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return archive_id

    def get(self, name):
        """The archive to use for `name`.

        Multiple rows can share a name: parts move between folders (updated in
        place, same row), and a replaced part creates a new row while flagging the
        old one `stale` (see `register`). This returns the newest non-stale row,
        or else the newest row of any state, so a stale index can still be
        inspected rather than vanishing outright.

        Known limitation: two *live* (non-stale) archives of the same name in
        different folders resolve to whichever is newest, and choosing between
        them properly needs a selector that Task 8+ does not have yet.
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM archives WHERE name=? AND state!='stale' ORDER BY id DESC LIMIT 1",
            (name,)).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM archives WHERE name=? ORDER BY id DESC LIMIT 1",
                (name,)).fetchone()
        return row

    def parts_of(self, archive_id) -> list[Part]:
        rows = self.connect().execute(
            "SELECT idx, name, size, offset, path_display, dbx_id, rev, content_hash "
            "FROM parts WHERE archive_id=? ORDER BY idx", (archive_id,)).fetchall()
        return [Part(idx=r["idx"], name=r["name"], size=r["size"], offset=r["offset"],
                     path_display=r["path_display"] or "", dbx_id=r["dbx_id"] or "",
                     rev=r["rev"] or "", content_hash=r["content_hash"] or "")
                for r in rows]

    def start_walk(self, archive_id) -> int:
        """Resume a walk: reclaim what a dead run left behind, and return the
        confirmed frontier to resume from.

        Called from Task 7's main thread while holding the instance lock, which is
        what makes "a `walking` segment belongs to a process that is no longer
        running" a safe assumption -- nothing else could be walking it right now.
        `error` segments are reclaimed too, on the same logic that an archive-level
        `error` state gets retried: a chain that failed on a previous run deserves
        another attempt on this one. Without this, a killed run's `walking`
        segments are never picked up again by `claim_segment`, and the archive can
        never finish.

        A `stale` archive stays `stale`: the `CASE` guards `state` the same way
        `commit_batch` and `finish` do, so a fingerprint check that fired while a
        walk was in flight is not silently erased by the walk's own bookkeeping.
        Only `register` may lift it, and only by re-verifying the content (see its
        docstring). `started_at`/`error`/`finished_at` still get cleared normally
        -- only the visible `state` text is protected.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            conn.execute(
                """UPDATE archives SET
                       state=CASE WHEN state='stale' THEN 'stale' ELSE 'walking' END,
                       started_at=COALESCE(started_at, ?),
                       error=NULL, finished_at=NULL
                   WHERE id=?""",
                (now, archive_id))
            conn.execute(
                """UPDATE segments SET state='pending', owner=NULL, error=NULL
                   WHERE archive_id=? AND state IN ('walking', 'error')""",
                (archive_id,))
            covered = self._covered(conn, archive_id)
            self.log_event(archive_id, "run_start", str(covered))
            cursor_offset = conn.execute(
                "SELECT cursor_offset FROM archives WHERE id=?", (archive_id,)).fetchone()[0]
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return cursor_offset

    def finish(self, archive_id, result) -> None:
        """Record a completed walk's final outcome and confirmed frontier.

        `n_members`/`member_bytes` are recomputed from `members` rather than
        carried as running counters -- same reasoning as `stats` -- so a batch
        replayed during the walk cannot have inflated them. `cursor_offset`, the
        confirmed frontier, is set outright to `result.end_offset` here rather
        than the `MAX` that `mark_joined`/`reset_segment` use: a finished walk's
        own verdict on where the chain ended is authoritative.

        `state` and `detail` are the two protected columns: a `stale` archive
        stays `stale`, and keeps its stale reason, even though a walk just
        concluded -- both guarded by the same `CASE WHEN state='stale' THEN
        <keep the old value> ELSE <the walk's own> END` pattern (fix-round
        amendment: a flag the next write silently erases is not a flag, and the
        same argument applies to the reason behind it). Every other column
        (`end_offset`, `n_members`, `member_bytes`, `finished_at`,
        `cursor_offset`) is still written unconditionally -- a guard that skipped
        the whole statement would lose those too. `result.detail`, the walk's own
        outcome, is not lost when the guard fires: it is already on the segment
        that produced it (`segments.detail`), and the `log_event` call below
        records it a second time, so it stays recoverable from two places. The
        stale reason would have been recoverable from none once overwritten here,
        and it is the more urgent fact anyway -- it says the index may not
        describe the bytes that are there now.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            n_members, member_bytes = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM members WHERE archive_id=?",
                (archive_id,)).fetchone()
            conn.execute(
                """UPDATE archives SET
                       state=CASE WHEN state='stale' THEN 'stale' ELSE ? END,
                       end_offset=?,
                       detail=CASE WHEN state='stale' THEN detail ELSE ? END,
                       n_members=?,
                       member_bytes=?, cursor_offset=?, finished_at=?
                   WHERE id=?""",
                (result.state, result.end_offset, result.detail, n_members, member_bytes,
                 result.end_offset, time.time(), archive_id))
            self.log_event(archive_id, "finish", result.detail)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def mark_unsupported(self, archive_id, detail: str) -> None:
        """The archive is in a format the walker does not index (pax), for good.

        Terminal, unlike `error`, which the next `index` retries: re-walking cannot
        change an archive's format, so `index` and `cat` refuse this state outright. A
        `stale` archive keeps its flag and its reason, as `finish` keeps them -- the
        parts changed, which says more, and re-registering walks it again anyway. The
        refusal still survives in the event below and on the segment that met it.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE archives SET
                       state=CASE WHEN state='stale' THEN 'stale' ELSE 'unsupported' END,
                       detail=CASE WHEN state='stale' THEN detail ELSE ? END,
                       finished_at=?
                   WHERE id=?""",
                (detail, time.time(), archive_id))
            self.log_event(archive_id, "unsupported", detail)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def fail(self, archive_id, error: str) -> None:
        """Record a walk-level failure, as opposed to `fail_segment`'s one chain.

        `state` is guarded the same way `commit_batch`/`start_walk`/`finish` guard
        it: a `stale` archive must stay `stale`, not become `error`. The reachable
        path this closes is stale -> fail -> 'error' -> a later `index` run treats
        `error` as retryable and re-walks it, mixing bytes from changed parts into
        an old index -- precisely what the flag exists to prevent. `error` (the
        message) and `finished_at` still write unconditionally: knowing why a run
        failed is useful whatever the resulting state.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE archives SET
                       state=CASE WHEN state='stale' THEN 'stale' ELSE 'error' END,
                       error=?, finished_at=?
                   WHERE id=?""",
                (error, time.time(), archive_id))
            self.log_event(archive_id, "fail", error)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def mark_stale(self, archive_id, detail: str) -> None:
        """Flag an index as no longer trustworthy, without touching its rows.

        For a fingerprint check (a live part's content_hash) that disagrees with
        what this archive was registered from -- the parts changed underneath a
        walk that had already started.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE archives SET state='stale', detail=? WHERE id=?",
                         (detail, archive_id))
            self.log_event(archive_id, "stale", detail)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def stats(self, archive_id) -> dict:
        """A snapshot for `status`.

        `n_members`/`member_bytes` are recomputed from `members`, never carried as
        counters, so a replayed batch cannot inflate them. `covered` and
        `confirmed` are the segment-progress and confirmed-frontier readings (see
        `_covered` and `mark_joined`); `run_started_at`/`run_start_covered` come
        from the most recent `run_start` event, so a caller can show the rate of
        the run in progress rather than of the walk's whole lifetime.
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT state, total_size, requests, bytes_fetched, cursor_offset, "
            "       started_at, updated_at FROM archives WHERE id=?",
            (archive_id,)).fetchone()
        n_members, member_bytes = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM members WHERE archive_id=?",
            (archive_id,)).fetchone()
        event = conn.execute(
            "SELECT ts, detail FROM events WHERE archive_id=? AND kind='run_start' "
            "ORDER BY ts DESC LIMIT 1", (archive_id,)).fetchone()
        return {
            "state": row["state"],
            "total_size": row["total_size"],
            "n_members": n_members,
            "member_bytes": member_bytes,
            "requests": row["requests"],
            "bytes_fetched": row["bytes_fetched"],
            "covered": self._covered(conn, archive_id),
            "confirmed": row["cursor_offset"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "run_started_at": event["ts"] if event else None,
            "run_start_covered": int(event["detail"]) if event else None,
        }

    def _covered(self, conn, archive_id) -> int:
        """Bytes of the archive whose place in the chain is settled.

        Summed per segment rather than read off one column, because a segment's
        own progress changes moment to moment while `archives.cursor_offset` (the
        *confirmed* frontier) only moves when a segment becomes `joined`. Every
        term is clamped at 0, in case a segment's cursor sits before its own scan
        start (it should not, but this reports rather than crashes if it ever did).
        """
        total_size = conn.execute(
            "SELECT total_size FROM archives WHERE id=?", (archive_id,)).fetchone()[0]
        segments = conn.execute(
            "SELECT scan_from, stop_at, first_header, cursor_offset, state "
            "FROM segments WHERE archive_id=?", (archive_id,)).fetchall()
        covered = 0
        for seg in segments:
            span_end = seg["stop_at"] if seg["stop_at"] is not None else total_size
            if seg["state"] in FINISHED_SEGMENT_STATES:
                covered += max(span_end - seg["scan_from"], 0)
            elif seg["cursor_offset"] is not None:
                covered += max(min(seg["cursor_offset"], span_end) - seg["scan_from"], 0)
            elif seg["first_header"] is not None:
                covered += max(min(seg["first_header"], span_end) - seg["scan_from"], 0)
        return covered

    def build_indexes(self, archive_id) -> None:
        """Composite indexes leading with archive_id, for the per-archive queries
        Task 8+ runs (`find_members`, browsing by directory or by size).

        Built once a walk is done, not while it runs: the indexes cover the whole
        table rather than just this archive, but there is no benefit to paying
        their maintenance cost on every insert before any walk has finished.
        """
        conn = self.connect()
        conn.execute("CREATE INDEX IF NOT EXISTS members_dir ON members(archive_id, dir)")
        conn.execute("CREATE INDEX IF NOT EXISTS members_size ON members(archive_id, size DESC)")

    # ---- members ------------------------------------------------------------

    def commit_batch(self, archive_id, segment_id, members, next_offset,
                     requests=0, bytes_fetched=0) -> None:
        """Members and the resume cursor in one transaction: the whole resumability
        story. The cursor belongs to the segment, because each chain resumes
        independently; the archive row carries only the totals `stats` reports.

        The segment's cursor is updated *before* the members are inserted, and the
        members are converted to row tuples by a generator that `executemany`
        consumes lazily. That ordering is deliberate: it makes a bad size
        (`_as_int` raising `ValueError`) fire only after any earlier rows in this
        batch are already written to the members table, so the `ROLLBACK` below is
        actually exercised by the failure -- not merely raising before any SQL ran,
        which would pass the same assertions without proving atomicity at all.

        `segments.members` is recomputed over the segment's own span *after* the
        insert -- `COUNT(*) WHERE hdr_offset >= scan_from AND (stop_at IS NULL OR
        hdr_offset < stop_at)` -- rather than incremented by `len(members)`. An
        incrementing counter is exactly the kind of thing this store elsewhere
        rejects (`stats`, `finish`): the design guarantees replays ("a restart
        re-walks exactly that batch"), and `INSERT OR REPLACE` already makes the
        *rows* replay-safe, so the count next to them has to be too, or resuming a
        killed walk quietly inflates it on every single restart, not just on some
        rare race. This is a plain range COUNT against the members primary key
        (archive_id, hdr_offset), negligible next to the 1.6s a batch's own
        Dropbox request costs.

        `state` is guarded the same way `start_walk` and `finish` guard it: a
        `stale` archive must not be silently un-flagged by the next worker batch
        landing after `mark_stale` fired mid-walk.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE segments SET cursor_offset=? WHERE id=?",
                         (next_offset, segment_id))

            def rows():
                for m in members:
                    yield (archive_id, m.hdr_offset, m.data_offset, _as_int(m.size), m.type,
                           m.mode, m.mtime, m.uname, m.gname, m.dir, m.name, m.linkname)

            conn.executemany(
                """INSERT OR REPLACE INTO members (archive_id, hdr_offset, data_offset,
                       size, type, mode, mtime, uname, gname, dir, name, linkname)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows())
            conn.execute(
                """UPDATE segments SET members = (
                       SELECT COUNT(*) FROM members
                       WHERE members.archive_id = segments.archive_id
                         AND members.hdr_offset >= segments.scan_from
                         AND (segments.stop_at IS NULL OR members.hdr_offset < segments.stop_at)
                   ) WHERE id=?""",
                (segment_id,))
            conn.execute(
                """UPDATE archives SET
                       state=CASE WHEN state='stale' THEN 'stale' ELSE 'walking' END,
                       requests=requests+?, bytes_fetched=bytes_fetched+?,
                       started_at=COALESCE(started_at, ?), updated_at=?
                   WHERE id=?""",
                (requests, bytes_fetched, time.time(), time.time(), archive_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def member_at(self, archive_id, hdr_offset):
        """The member whose header sits at ``hdr_offset``, or None -- how `cat` knows
        what must follow the member it extracts."""
        return self.connect().execute(
            "SELECT * FROM members WHERE archive_id=? AND hdr_offset=?",
            (archive_id, hdr_offset)).fetchone()

    def repeated_paths(self, archive_id, limit: int) -> list[sqlite3.Row]:
        """Every row whose path is recorded more than once, in header order, at most
        ``limit`` of them.

        A tar can hold a path twice (`tar -r` appends a newer copy), but a bad read that
        is valid tar from elsewhere in the archive usually makes one: it records that
        other member's header, path and all, a second time. Not always -- a member of a
        tarball stored inside the archive, or a long-named member's header without its
        long-name block, has a path recorded nowhere else.
        """
        return self.connect().execute(
            """SELECT m.* FROM members AS m
               JOIN (SELECT dir, name FROM members WHERE archive_id=?
                     GROUP BY dir, name HAVING COUNT(*) > 1) AS r
                 ON m.dir = r.dir AND m.name = r.name
               WHERE m.archive_id=?
               ORDER BY m.hdr_offset LIMIT ?""",
            (archive_id, archive_id, limit)).fetchall()

    def charge(self, archive_id, requests: int, bytes_fetched: int) -> None:
        """Add reads made outside any segment -- an audit's -- to the archive's cost."""
        self.connect().execute(
            "UPDATE archives SET requests=requests+?, bytes_fetched=bytes_fetched+? "
            "WHERE id=?", (requests, bytes_fetched, archive_id))

    def find_members(self, archive_id, path) -> list[sqlite3.Row]:
        """Every member whose (dir, name) matches `path`, oldest header first.

        Split exactly as `Member.from_tarinfo` builds the columns (`rstrip("/")`,
        then `rpartition("/")`), so a lookup path and a stored path always compare
        the same way. `cat` (Task 8) relies on getting every match back: a tar may
        hold the same path twice, and guessing between them is a wrong answer.
        """
        stripped = path.rstrip("/")
        head, _, tail = stripped.rpartition("/")
        return self.connect().execute(
            "SELECT * FROM members WHERE archive_id=? AND dir=? AND name=? "
            "ORDER BY hdr_offset",
            (archive_id, head, tail or stripped)).fetchall()

    def query_members(self, archive_id) -> list[sqlite3.Row]:
        return self.connect().execute(
            "SELECT * FROM members WHERE archive_id=? ORDER BY hdr_offset",
            (archive_id,)).fetchall()

    def drop_members_between(self, archive_id, start, end) -> None:
        """Throw away rows in ``[start, end)``: they were never on the confirmed
        chain, whether because a segment is being reset or retired. ``end=None``
        means no upper bound, for a segment with no `stop_at` (the last part).
        """
        conn = self.connect()
        if end is None:
            conn.execute(
                "DELETE FROM members WHERE archive_id=? AND hdr_offset>=?",
                (archive_id, start))
        else:
            conn.execute(
                "DELETE FROM members WHERE archive_id=? AND hdr_offset>=? AND hdr_offset<?",
                (archive_id, start, end))

    # ---- segments -----------------------------------------------------------

    def seed_segments(self, archive_id, parts) -> None:
        """One segment per part. Segment 0 is authoritative: a tar begins at
        offset 0, so it needs no scan and no confirmation from a predecessor.

        `INSERT OR IGNORE` on the (archive_id, idx) unique pair, wrapped in its
        own transaction (each row would otherwise be its own autocommit
        transaction), so re-registering an archive whose parts merely moved
        folders never disturbs a walk already in progress.
        """
        parts = list(parts)
        rows = []
        for index, part in enumerate(parts):
            stop_at = parts[index + 1].offset if index + 1 < len(parts) else None
            rows.append((archive_id, index, part.offset, stop_at,
                         0 if index == 0 else None,
                         1 if index == 0 else 0))
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                """INSERT OR IGNORE INTO segments
                       (archive_id, idx, scan_from, stop_at, first_header, joined)
                   VALUES (?, ?, ?, ?, ?, ?)""", rows)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def segments(self, archive_id) -> list[sqlite3.Row]:
        return self.connect().execute(
            "SELECT * FROM segments WHERE archive_id=? ORDER BY idx", (archive_id,)).fetchall()

    def claim_segment(self, archive_id, owner):
        """Atomically take the lowest-index pending segment, or None.

        Mirrors `Store.claim_shard` (dbaudit/store.py:193): one
        `UPDATE ... RETURNING` is exclusive under SQLite's own locking, so two
        callers racing on the same archive can never claim the same segment, and
        the caller sees the claimed row's *post*-claim values (state='walking',
        its own owner) rather than a stale pre-claim read.
        """
        return self.connect().execute(
            "UPDATE segments SET state='walking', owner=? "
            "WHERE id = (SELECT id FROM segments WHERE archive_id=? AND state='pending' "
            "            ORDER BY idx LIMIT 1) "
            "RETURNING *",
            (owner, archive_id)).fetchone()

    def set_segment_start(self, segment_id, offset) -> None:
        """Record where a scan found this segment's first header, before any batch
        is committed.

        Deliberately not `joined`: that is earned only once the segment before
        this one's chain walks into exactly this offset (`mark_joined`). Until
        then this offset is only a guess -- which is also why `_covered` treats a
        bare `first_header` differently from a `cursor_offset`.
        """
        self.connect().execute(
            "UPDATE segments SET first_header=? WHERE id=?", (offset, segment_id))

    def mark_joined(self, segment_id) -> None:
        """Confirm a segment's start, and raise the archive's frontier to match.

        A segment is believed only once the chain before it walks into exactly the
        offset it started from. Once that happens, everything up to its
        `first_header` is on the confirmed chain, so `archives.cursor_offset` can
        advance to at least that point -- but never retreat, hence the `MAX`.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT archive_id, first_header FROM segments WHERE id=?",
                (segment_id,)).fetchone()
            conn.execute("UPDATE segments SET joined=1 WHERE id=?", (segment_id,))
            if row["first_header"] is not None:
                conn.execute(
                    "UPDATE archives SET cursor_offset=MAX(cursor_offset, ?) WHERE id=?",
                    (row["first_header"], row["archive_id"]))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def reset_segment(self, segment_id, first_header) -> None:
        """Drop a segment's rows and re-arm it from a corrected start, atomically.

        Split across two calls, a crash in between leaves rows from a chain nobody
        will re-walk, silently poisoning the manifest. `joined=1`: a start handed
        down by a confirmed predecessor is itself confirmed, so this also raises
        the archive's frontier the same way `mark_joined` does.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT archive_id, scan_from, stop_at FROM segments WHERE id=?",
                (segment_id,)).fetchone()
            self.drop_members_between(row["archive_id"], row["scan_from"], row["stop_at"])
            conn.execute(
                """UPDATE segments SET first_header=?, cursor_offset=NULL, exit_offset=NULL,
                       detail=NULL, error=NULL, owner=NULL, members=0, state='pending',
                       joined=1
                   WHERE id=?""",
                (first_header, segment_id))
            conn.execute(
                "UPDATE archives SET cursor_offset=MAX(cursor_offset, ?) WHERE id=?",
                (first_header, row["archive_id"]))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def retire_segment(self, segment_id) -> None:
        """A segment lying entirely past where the archive's chain ended.

        This is the normal case, not an exotic one: a tar's terminator is usually
        not in the last part, so the segment covering it typically has nothing of
        its own to contribute and its rows (found before the chain's true end was
        known) were never really on the chain.

        `error` is cleared for that same reason. A trailing segment's scan fails
        ("no header between here and the end of the archive") while its fate is
        still unknown, and that failure stops being one the moment the join proves
        the segment was never on the chain. Leaving it on the row makes `status`
        report a fault beside the most ordinary state a segment has. `detail` is
        kept: it still explains how the row got here.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT archive_id, scan_from, stop_at FROM segments WHERE id=?",
                (segment_id,)).fetchone()
            self.drop_members_between(row["archive_id"], row["scan_from"], row["stop_at"])
            conn.execute(
                "UPDATE segments SET state='beyond', members=0, joined=0, error=NULL "
                "WHERE id=?",
                (segment_id,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def release_segment(self, segment_id, requests=0, bytes_fetched=0) -> None:
        """Hand a segment back to the queue without losing its progress, or its cost.

        Called when a walk stops early (Ctrl-C, `--max-batches`), so the next run
        resumes from the committed cursor instead of rescanning for the start
        again. Mirrors `Store.release_shard` (dbaudit/store.py:220).

        Takes a cost for the same reason `finish_segment` and `fail_segment` do:
        reads made after the last committed batch are otherwise charged to
        nothing. Here that is the batch a `--max-batches` stop refuses -- it was
        read before it was refused -- and stopping early is this walker's normal
        workflow rather than an exception, so the undercount would be routine.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            archive_id = conn.execute(
                "SELECT archive_id FROM segments WHERE id=?", (segment_id,)).fetchone()[0]
            conn.execute(
                "UPDATE segments SET state='pending', owner=NULL "
                "WHERE id=? AND state='walking'",
                (segment_id,))
            conn.execute(
                "UPDATE archives SET requests=requests+?, bytes_fetched=bytes_fetched+? "
                "WHERE id=?",
                (requests, bytes_fetched, archive_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def finish_segment(self, segment_id, result, requests=0, bytes_fetched=0) -> None:
        """Record one chain's outcome, and the reader cost of producing it.

        Requests/bytes made after the last batch -- the terminator probe, or a
        cold-start scan that found no header at all -- are otherwise never
        counted, and bytes_fetched / member_bytes is the headline efficiency
        number for the whole project.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            archive_id = conn.execute(
                "SELECT archive_id FROM segments WHERE id=?", (segment_id,)).fetchone()[0]
            conn.execute(
                "UPDATE segments SET state=?, exit_offset=?, detail=?, owner=NULL WHERE id=?",
                (result.state, result.end_offset, result.detail, segment_id))
            conn.execute(
                "UPDATE archives SET requests=requests+?, bytes_fetched=bytes_fetched+? "
                "WHERE id=?",
                (requests, bytes_fetched, archive_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def fail_segment(self, segment_id, error: str, requests=0, bytes_fetched=0) -> None:
        """Record one chain's failure, and the reader cost incurred before it failed."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            archive_id = conn.execute(
                "SELECT archive_id FROM segments WHERE id=?", (segment_id,)).fetchone()[0]
            conn.execute(
                "UPDATE segments SET state='error', error=?, owner=NULL WHERE id=?",
                (error, segment_id))
            conn.execute(
                "UPDATE archives SET requests=requests+?, bytes_fetched=bytes_fetched+? "
                "WHERE id=?",
                (requests, bytes_fetched, archive_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
