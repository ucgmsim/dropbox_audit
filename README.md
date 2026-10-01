# dbaudit — Dropbox storage audit

dbaudit crawls a Dropbox folder tree into a SQLite database and reports where the
space goes: duplicates, cold data, the largest directories and small-file hotspots.
It can also list what is inside tar archives stored on Dropbox, including
multi-terabyte archives split into parts, without downloading them.

**It is strictly read-only:** it never calls a Dropbox endpoint that modifies data.

## Install

Requires Python 3.11+ and a working [rclone](https://rclone.org) remote for the
account. dbaudit borrows rclone's credentials (`rclone config dump`) and lets rclone
refresh them, so there is no OAuth app to register and no secret stored here. The
remote is assumed to be called `dropbox`; `init` and the `archive` commands take
`--remote` to use another.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Auditing a folder tree

```bash
python -m dbaudit init   --db audit.db --root /TeamSpace
python -m dbaudit run    --db audit.db
python -m dbaudit status --db audit.db     # safe to run while `run` is going
python -m dbaudit report --db audit.db     # or `export --db audit.db --out DIR` for CSVs
```

- `init` checks credentials, the root path and free disk space.
- **`run` can be stopped at any moment and resumed by running it again.** Each page
  of 2,000 entries is committed together with its cursor, so a crash loses at most
  the page in flight and never duplicates a row. Ctrl-C finishes that page and exits
  cleanly. A lockfile keeps it to one crawler per database, and `--log FILE` keeps a
  log for unattended runs.
- `run --incremental` re-audits only what has changed since a completed pass.
- `verify --db audit.db [subtree]` re-walks a subtree with rclone and diffs it
  against the database. The exit code is non-zero if anything differs.

The report covers:

- **Reconciliation:** crawled bytes against Dropbox's own usage figure. The gap is
  version history, deleted-but-retained data and anything outside the root.
- **Reclaimable duplicates,** grouped by Dropbox's `content_hash`.
- **Cold data:** bytes by year last modified.
- **Largest directories, small-file hotspots and bytes by file type.**

Anything else is one SQL query away. Directories are stored once, so full paths
take one join:

```sql
SELECT d.path_display || '/' || f.name AS path, f.size
FROM files f JOIN dirs d ON d.id = f.dir_id
ORDER BY f.size DESC LIMIT 20;
```

## Why not just use rclone?

rclone's Dropbox backend has no recursive listing, so it makes one API call per
directory; on a directory-dense tree that is millions of calls. The Dropbox API's
`files/list_folder(recursive=true)` returns 2,000 entries per call for a whole
subtree. Measured on a live account:

| Engine | Throughput | Rate limiting |
|---|---|---|
| rclone, `--checkers 32` | stalled | 5-minute ban |
| API recursive, 1 worker | 578 entries/s | none |
| **API recursive, 8 workers (default)** | **4,621 entries/s** | **none** |
| API recursive, 16 workers | 4,867 entries/s | 429s |

At that rate a 100-million-entry tree takes about six hours. rclone is still used
for credentials and for `verify`.

### Tuning

- **If 429s (rate-limit responses) climb,** as `status` shows for the last hour,
  lower `--workers`, then `--rps` (default 5 requests/s). Each 429 pauses every
  worker for the full `Retry-After` and drops concurrency by one. Sustained 429s
  usually mean another job is using the same account's API quota.
- **Leave `--queue-target` alone.** Late in a crawl, a few large subtrees can leave
  most workers idle. Splitting harder looks like the fix, but measured far slower:
  a recursive call on a big subtree returns 2,000 entries, while one on a small
  split-off directory returns a handful. Requests are the scarce resource, not
  workers.
- **Disk:** about 283 bytes per file, so 100 million files need roughly 40 GB with
  the analysis indexes. `init` refuses to start below `--min-free-gb` (default 30).

## Indexing tar archives in place

The `archive` commands record every member of a tar stored on Dropbox (path, size,
owner, mtime and offsets) by reading only its 512-byte headers, over HTTP range
requests. An archive split into parts (`big.tar.part00`, `big.tar.aa`, ...) is read
as one continuous file.

```bash
python -m dbaudit archive register --db archives.db --name big.tar --folder /TeamSpace/runs
python -m dbaudit archive index    --db archives.db --archive big.tar
python -m dbaudit archive status   --db archives.db
python -m dbaudit archive report   --db archives.db --archive big.tar   # or `export --out DIR`
python -m dbaudit archive cat      --db archives.db --archive big.tar \
    --member path/inside/archive --out member.bin
```

- **Few requests.** Reads go through a window that doubles from 64 KiB up to 16 MiB
  while headers keep landing inside it, so a run of small members shares one
  request and a large member costs one small read. A 5 TiB archive in 18 parts was
  indexed in about six hours.
- **Resumable.** Each batch of members is committed with the cursor that resumes
  it, so `index` can be interrupted and re-run at any time.
- **Checked reads.** Dropbox occasionally answers a range request with the wrong
  bytes, so nothing is believed on one read: members are committed, and a verdict
  such as "complete" or "corrupt" stands, only once two readings agree. `cat`
  checks every read against the member's header and the blocks around it.
- `cat` pulls one member straight from its recorded offset. `register --local-dir`
  indexes a local copy instead of Dropbox.

## Known limits

- **Team folders the account has not joined are invisible.** rclone's token has no
  team-admin scope, so the crawl only sees what the account itself can see.
- **Version history is inferred, not itemised.** Listing revisions file by file
  would take months, so it shows up only as the reconciliation gap.
- **GNU and ustar tars only.** pax archives are refused rather than indexed, because
  pax can keep a member's size outside the checksummed header.

## Development

```bash
pip install pytest
python -m pytest                                     # offline; live tests are skipped
DBAUDIT_LIVE=1 python -m pytest tests/test_live.py   # crawls DBAUDIT_LIVE_ROOT for real
```

The Dropbox API sits behind small interfaces, so the crawler and the tar walker are
tested against fakes. These replay canned pages and byte ranges, including 429s with
`Retry-After`, cursor resets, interruption mid-crawl and reads that return the
wrong bytes.
