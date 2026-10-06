-- dbaudit schema.
--
-- Directories are normalised: `dirs` carries the materialised path (a few million
-- rows) and `files` carries only dir_id + name. Storing a full path on every file
-- row would cost ~150 bytes x 100M rows before anything useful was stored, and it
-- would make "which directories consume the space" -- the question this audit
-- exists to answer -- a 100M-row string operation instead of a cheap rollup.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    started_at  REAL,
    finished_at REAL,
    host        TEXT,
    notes       TEXT
);

-- A shard is a subtree listed under its own cursor. `cursor` is the resume point:
-- it is written in the same transaction as the rows of the page it follows, which
-- is what makes an interrupted crawl resume exactly.
CREATE TABLE IF NOT EXISTS shards (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    depth        INTEGER NOT NULL,
    mode         TEXT NOT NULL DEFAULT 'recursive',   -- 'recursive' | 'split'
    state        TEXT NOT NULL DEFAULT 'pending',     -- pending|running|done|error
    cursor       TEXT,
    pages        INTEGER NOT NULL DEFAULT 0,
    entries      INTEGER NOT NULL DEFAULT 0,
    n_files      INTEGER NOT NULL DEFAULT 0,
    n_dirs       INTEGER NOT NULL DEFAULT 0,
    bytes        INTEGER NOT NULL DEFAULT 0,
    attempts     INTEGER NOT NULL DEFAULT 0,
    owner        TEXT,
    heartbeat_at REAL,
    note         TEXT,
    error        TEXT,
    created_at   REAL,
    started_at   REAL,
    finished_at  REAL
);

CREATE INDEX IF NOT EXISTS shards_state ON shards(state, depth, id);

CREATE TABLE IF NOT EXISTS principals (
    id             INTEGER PRIMARY KEY,
    dbx_account_id TEXT NOT NULL UNIQUE
);

-- path_lower is always derived locally as path_display.lower(), never taken from
-- the API response. Ancestor rows are created from child paths, where no API-supplied
-- path_lower exists, so deriving it consistently is what keeps the key unique.
CREATE TABLE IF NOT EXISTS dirs (
    id                      INTEGER PRIMARY KEY,
    parent_id               INTEGER,
    name                    TEXT NOT NULL,
    path_display            TEXT NOT NULL,
    path_lower              TEXT NOT NULL UNIQUE,
    depth                   INTEGER NOT NULL,
    dbx_id                  TEXT,
    shared_folder_id        TEXT,
    parent_shared_folder_id TEXT,
    is_mount                INTEGER NOT NULL DEFAULT 0,
    shard_id                INTEGER
);

-- dbx_id is UNIQUE so that re-listing a subtree after a cursor reset updates rows
-- instead of duplicating them. It is the only index maintained on `files` during
-- the crawl; analysis indexes are built afterwards.
CREATE TABLE IF NOT EXISTS files (
    dbx_id          TEXT UNIQUE,
    dir_id          INTEGER NOT NULL,
    name            TEXT NOT NULL,
    ext             TEXT,
    size            INTEGER NOT NULL,
    content_hash    BLOB,
    rev             TEXT,
    client_modified INTEGER,
    server_modified INTEGER,
    modified_by     INTEGER,
    is_downloadable INTEGER,
    shard_id        INTEGER
);

CREATE TABLE IF NOT EXISTS api_events (
    ts     REAL,
    kind   TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS api_events_ts ON api_events(ts);
