# dbaudit — Dropbox storage audit

Enumerates every file and folder under a Dropbox path into a queryable SQLite
database, with reports aimed at deciding what storage can be reclaimed.

**It is strictly read-only.** It never calls a Dropbox endpoint that modifies data.

## Why not rclone?

rclone is the obvious tool, and it is used here — for credentials and for
verification — but not for the crawl itself. `rclone backend features dropbox:`
reports **`ListR: false`**: the Dropbox backend has no recursive listing, so
`--fast-list` does nothing and rclone must make **one API call per directory**. On a
directory-dense tree that is millions of calls, which is where "this will take
months" comes from.

The Dropbox API itself supports `files/list_folder(recursive=true)`, which returns
2000 entries per call for an entire subtree. Measured against the live account on
2026-08-13:

| Engine | Throughput | Rate limiting |
|---|---|---|
| rclone walk, `--checkers 32` | stalled | **`retry_after: 300`** (5-minute ban) |
| API recursive, 1 worker | 578 entries/s | none |
| API recursive, 4 workers | 2,329 entries/s | none |
| **API recursive, 8 workers** | **4,621 entries/s** | **none over 2 minutes** |
| API recursive, 16 workers | 4,867 entries/s | 4 × 429 |

At ~4,600 entries/s a 100M-entry tree finishes in about six hours.

## Install

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Requires Python 3.11+, `requests`, and a working rclone remote. Credentials are read
from `rclone config dump` and refreshed by rclone, so there is no second OAuth app to
register and no client secret stored here.

## Running the audit

```bash
.venv/bin/python -m dbaudit init --db audit.db --root /TeamSpace
.venv/bin/python -m dbaudit run  --db audit.db --workers 8
```

`init` preflights credentials, the root path, and free disk space. `run` crawls until
finished and builds the analysis indexes.

**To resume, just run `run` again.** Interruption is safe at any instant: each page of
2000 entries is committed in the same SQLite transaction as its cursor, so a kill
`-9`, a reboot, or a dropped network loses at most the page in flight and never
duplicates a row. `Ctrl-C` is cleaner still — it finishes the page in flight, commits,
and exits 0.

Unattended:

```bash
nohup .venv/bin/python -m dbaudit run --db audit.db --log audit.log &
```

Check on it from another shell at any time — `status` reads the database, so it is
safe while `run` is going:

```bash
.venv/bin/python -m dbaudit status --db audit.db
```

## Tuning

| Flag | Default | Notes |
|---|---|---|
| `--workers` | 8 | Measured sweet spot. 16 starts drawing 429s. |
| `--rps` | 5.0 | Global request budget, shared by all workers. |
| `--queue-target` | 2 x workers | Shards kept pending. Raising it usually goes *slower*. |
| `--split-depth` | 6 | Hard bound on how deep splitting may go. |
| `--max-shards` | 2,000,000 | Safety cap on shard creation. |
| `--progress-interval` | 30 | Seconds between progress log lines. |

**Resist the urge to raise `--queue-target`.** Late in a crawl you will often see
`status` reporting only one or two shards running while eight workers are
configured — one big subtree is still being listed and the rest are idle. The
obvious fix is to split more aggressively so there is always work queued. It was
tried twice on `/TeamSpace/Public` and both variants were dramatically worse than
leaving it alone:

| Setting | Shards | Result (5 min, same subtree) |
|---|---|---|
| default (`2 x workers`) | 72–115 | **~319,000 files** |
| split top 2 levels unconditionally | 17,908 | 18k files in the first 30s, vs 53k |
| keep 200 shards queued | 17,878 | throughput collapsed to **4 files/s** |

The scarce resource is *requests*, not workers. A recursive listing of a large
subtree returns 2000 entries per call; a shard covering one small directory returns
a handful. Splitting trades one high-yield call for many low-yield ones — which is
precisely rclone's per-directory cost model, the thing this tool exists to avoid. An
idle worker costs one concurrent request; an over-split tree costs the whole budget.

So stragglers are accepted deliberately. Raise `--queue-target` only if your tree is
shallow and wide enough that shards stay large.

**If 429s start climbing** (`status` shows the last hour), lower `--workers` first,
then `--rps`. The limiter already backs off on its own — a 429 parks *every* worker
for the full `Retry-After` and drops concurrency by one — but sustained 429s mean the
account is over budget, often because something else is using the same Dropbox API
quota. The audit competes with any other rclone job running as this account.

## Verifying the result

```bash
.venv/bin/python -m dbaudit verify --db audit.db /TeamSpace/some/subtree
```

This re-walks the subtree with rclone (at `--checkers 4`, deliberately below the level
that earns a ban) and diffs it against the database. The two reach Dropbox by
different routes — one recursive listing per subtree versus one call per directory —
so agreement is real evidence. Exit code is non-zero if anything differs.

Verified during development on `/TeamSpace/arr65`: 1,522 files and 88,117,687,729
bytes, matching rclone exactly, with zero duplicate rows.

## Reading the results

```bash
.venv/bin/python -m dbaudit report --db audit.db          # to the terminal
.venv/bin/python -m dbaudit export --db audit.db --out ./exports   # CSV
```

The report covers:

- **Reconciliation** — measured live bytes vs what Dropbox reports as used. The
  difference is version history, deleted-but-retained data, and anything outside the
  crawl root. Version history cannot be listed (`DeletedMetadata` carries no size and
  per-file `list_revisions` would take months), so it is inferred here.
- **Reclaimable duplicates** — grouped by `content_hash`, which Dropbox provides free
  with every file: `Σ size − max(size)` per group.
- **Cold data** — bytes by year last modified, plus 2/5/10-year buckets.
- **Largest directories** — recursive rollups.
- **Small-file hotspots** — directories with many tiny files. These dominate crawl
  time and are usually the best cleanup candidates.
- **File-type profile** — bytes by extension.

The database is plain SQLite, so anything not covered is one query away. DuckDB can
read it directly. Full file paths are one join, because directories are normalised:

```sql
SELECT d.path_display || '/' || f.name AS path, f.size
FROM files f JOIN dirs d ON d.id = f.dir_id
ORDER BY f.size DESC LIMIT 20;
```

## Repeat audits

```bash
.venv/bin/python -m dbaudit run --db audit.db --incremental
```

`init` records a whole-tree cursor via `files/list_folder/get_latest_cursor` (0.43s
even for 250 TB, since it lists nothing). An incremental pass continues from it and
receives only what changed, so it costs one API call per 2000 changes rather than one
per shard. Because the cursor is taken *before* the full crawl starts, the first
incremental pass also closes the gap for files that changed while the crawl was
running.

It refuses to run until a full pass has completed. If Dropbox ever invalidates the
cursor, the pass fails with a message telling you to run a fresh full pass.

## Sizing

Measured on `/TeamSpace/Public`: 68 MB for 240,010 files and 65,355 directories, i.e.
**~283 bytes per file** all-in (including the crawl-time uniqueness index). Extrapolating,
100M files is roughly **28 GB**, or **~40 GB** once the analysis indexes are built.

That figure is sensitive to how directory-dense the tree is — this one runs about one
directory per four files, and directory rows are the expensive ones because they carry
the materialised path. `init` refuses to start if free space is below `--min-free-gb`
(default 30), so raise it for a large root.

Directories are normalised out of the file rows — `dirs` holds the materialised path,
`files` holds `dir_id + name` — which both keeps the database compact and makes
per-directory rollups cheap instead of 100M string operations.

## Known limits

- **Team folders nobody has joined are invisible.** rclone's app token has no
  team-admin scope (`team/get_info` returns `USER_AUTH_NOT_ALLOWED`), so the crawl
  sees the member's root namespace: the crawl root and every team folder this account
  has joined. Auditing beyond that needs a Dropbox Business app with team scopes.
- **Version history is not itemised**, only inferred as a residual.
- **One crawler per database.** A lockfile enforces it; a second `run` exits 3.

## Development

```bash
.venv/bin/python -m pytest -q          # 107 tests, no network
DBAUDIT_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -q   # hits Dropbox
```

The API sits behind a `Lister` protocol, so the crawler is tested against a fake that
replays canned pages, including 429s with `Retry-After`, cursor resets, mid-crawl
interruption, and the fact that a recursive listing returns the listed folder as its
own first entry.
