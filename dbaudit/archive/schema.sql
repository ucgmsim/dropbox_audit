-- dbaudit/archive/schema.sql
--
-- The index of what is inside archives. Separate from the audit database: it must
-- survive a fresh crawl, and an archive is identified by the content of its parts
-- rather than by where they sit, because folders get rearranged.

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS archives (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'tar',
    source        TEXT NOT NULL DEFAULT 'dropbox',   -- 'dropbox' | 'local'
    folder        TEXT,
    total_size    INTEGER NOT NULL,
    n_parts       INTEGER NOT NULL,
    set_hash      TEXT NOT NULL UNIQUE,
    state         TEXT NOT NULL DEFAULT 'registered',
    cursor_offset INTEGER NOT NULL DEFAULT 0,
    n_members     INTEGER NOT NULL DEFAULT 0,
    member_bytes  INTEGER NOT NULL DEFAULT 0,
    end_offset    INTEGER,
    requests      INTEGER NOT NULL DEFAULT 0,
    bytes_fetched INTEGER NOT NULL DEFAULT 0,
    detail        TEXT,
    error         TEXT,
    created_at    REAL,
    started_at    REAL,
    updated_at    REAL,
    finished_at   REAL
);

CREATE TABLE IF NOT EXISTS parts (
    archive_id   INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    name         TEXT NOT NULL,
    path_display TEXT,
    dbx_id       TEXT,
    rev          TEXT,
    size         INTEGER NOT NULL,
    offset       INTEGER NOT NULL,
    content_hash TEXT,
    PRIMARY KEY (archive_id, idx)
);

-- One segment per part. Segment 0 starts at offset 0 and is authoritative; every
-- other segment has to find its first header by scanning, and its members only count
-- once the segment before it walks into exactly that offset.
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY,
    archive_id    INTEGER NOT NULL,
    idx           INTEGER NOT NULL,
    scan_from     INTEGER NOT NULL,      -- this part's global start offset
    stop_at       INTEGER,               -- the next part's start; NULL for the last
    first_header  INTEGER,               -- found by scanning, or handed down by a repair
    cursor_offset INTEGER,               -- resume point inside this segment
    exit_offset   INTEGER,               -- where this segment's walk stopped: the first
                                          -- header at or past stop_at when it crossed,
                                          -- otherwise where the chain ended
    state         TEXT NOT NULL DEFAULT 'pending',
        -- pending|walking|crossed|complete|truncated|corrupt|error|beyond
    joined        INTEGER NOT NULL DEFAULT 0,       -- start confirmed by the predecessor
    members       INTEGER NOT NULL DEFAULT 0,
    owner         TEXT,
    detail        TEXT,                  -- the outcome's explanation (WalkResult.detail)
    error         TEXT,                  -- the walker's own failure, if any
    UNIQUE (archive_id, idx)
);

CREATE TABLE IF NOT EXISTS members (
    archive_id  INTEGER NOT NULL,
    hdr_offset  INTEGER NOT NULL,
    data_offset INTEGER NOT NULL,
    size        INTEGER NOT NULL,
    type        TEXT NOT NULL,
    mode        INTEGER,
    mtime       INTEGER,
    uname       TEXT,
    gname       TEXT,
    dir         TEXT NOT NULL,
    name        TEXT NOT NULL,
    linkname    TEXT,
    PRIMARY KEY (archive_id, hdr_offset)
);

CREATE TABLE IF NOT EXISTS events (
    ts         REAL,
    archive_id INTEGER,
    kind       TEXT,
    detail     TEXT
);
