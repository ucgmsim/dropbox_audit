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

It's a [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists what it needs, `uv.lock` pins every
version, and `.python-version` the Python.

```bash
uv sync     # makes .venv with that Python and the locked packages
```

Run the commands below with `uv run`, as in `uv run python -m dbaudit status --db audit.db`, or after
`. .venv/bin/activate`. Every command lists its options with `--help`, e.g. `python -m dbaudit tree --help`.

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
- **A subtree too big for Dropbox to list in one go is split on the spot.** Dropbox
  answers a recursive listing of a big enough subtree with an HTTP 500 after about
  three minutes, so the crawler lists that folder one level down instead and tries
  each child on its own. Once 20 of a folder's children have been listed and at most
  one in ten held folders, the rest are listed one level too: one request each, where
  a recursive listing of a few files can take several. Running `run` again retries
  anything that still failed, from scratch.
- **Each database is one snapshot.** To audit again, crawl into a new database;
  keeping the old one makes before-and-after comparisons a join between the two.
- `verify --db audit.db [subtree]` re-walks a subtree with rclone and diffs it
  against the database. The exit code is non-zero if anything differs.

The report covers:

- **Reconciliation:** crawled bytes against Dropbox's own usage figure. The gap is
  version history, deleted-but-retained data and anything outside the root.
- **Reclaimable duplicates,** grouped by Dropbox's `content_hash`.
- **Cold data:** bytes by year last modified, going by each file's own date
  (`client_modified`). Dropbox's `server_modified` can be years newer than the
  content.
- **Largest directories, small-file hotspots and bytes by file type.**
- **By kind:** each kind of data's size, loose and in tarballs, and how much of it is
  older than 2, 5 and 10 years, as [`kinds`](#kinds-of-data) works it out. Give it
  `--archive-db` to look inside tarballs and `--rules` for your Dropbox's folder words.

### A browsable page

```bash
python -m dbaudit tree --db audit.db --out tree.html
```

`tree` writes one self-contained HTML page, in the style of a `du` tree: every folder
with its size, its share and a bar, biggest first. Folders and files under 1 GiB fold into
one row per folder. Badges mark folders whose bytes are mostly stored elsewhere too
(`dup`), last modified two or more years before the listing (`cold`), or made of 100,000
or more small files (`tiny`), and a kind badge (`vm 85%`) names the kind of data that
makes up at least half a folder's bytes. Tables after the tree list the biggest duplicated
folders, the coldest big folders, folders made of tiny files, and bytes by file type, by
kind, by year, and by kind and year.

- `--subtree PATH` covers one folder; duplicates are still found across the whole listing.
- `--min-size 100MiB` shows more of the tree. `--max-rows N` (default 100,000) doubles
  the threshold until the page fits.
- `--archive-db` and `--rules` work as they do for [`kinds`](#kinds-of-data): bytes inside
  a tarball count toward the folder that holds it, by the kinds of its members. Working
  out kinds adds about three minutes on 15 M files.
- The page holds real folder names, so share it as carefully as the database.

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

- **The request rate finds its own level.** It starts at `--rps` (default 5
  requests/s). After every five minutes in which requests went out at close to that
  rate, without a 429, it rises by 0.5/s, up to `--max-rps` (default 20). A 429
  (rate-limit response) pauses every worker for the full `Retry-After`, and once per
  overload halves the rate and drops one worker: 429s that come back during the pause,
  or within a second of the first, are one overload. The log records each change. If 429s
  keep coming, as `status` shows for the last hour, another job is probably using the
  same account's API quota: lower `--max-rps`, which also caps where the rate starts.
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
python -m dbaudit archive status   --db archives.db --archive big.tar
python -m dbaudit archive report   --db archives.db --archive big.tar   # or `export --out DIR`
python -m dbaudit archive cat      --db archives.db --archive big.tar \
    --member path/inside/archive --out member.bin
```

- **Few requests.** Reads go through a window that doubles from 64 KiB up to 16 MiB
  while headers keep landing inside it, so a run of small members shares one
  request and a large member costs one small read. A 5 TiB archive in 18 parts was
  indexed in about six hours. Each request also fetches up to 64 KiB either side and
  throws it away (see "Checked reads"). On archives of large members that nearly
  doubles `bytes fetched`, and the ratio `index` and `status` report, but it adds well
  under 1% to the time: a request costs about 1.6 s, a MiB about 0.04 s.
- **Resumable.** Each batch of members is committed with the cursor that resumes
  it, so `index` can be interrupted and re-run at any time.
- **Checked reads.** Dropbox sometimes answers a range request with the wrong bytes,
  and some requests it gets wrong every time they are asked. So a verdict such as
  "complete" or "corrupt" stands only once two readings agree, and every reading made
  to confirm another asks differently: its request starts or ends tens of kilobytes
  away from the first. The members around a verdict are read twice, and so is every
  path the index holds twice. `cat` checks every read against the member's header and
  the blocks around it.
- `cat` pulls one member straight from its recorded offset. `register --local-dir`
  indexes a local copy instead of Dropbox.

### Every archive in a listing

`archive run` walks every tar archive an audit database lists into one archive
database, many at a time:

```bash
python -m dbaudit archive run    --db archives.db --audit-db audit.db --dry-run
python -m dbaudit archive run    --db archives.db --audit-db audit.db
python -m dbaudit archive status --db archives.db   # counts by state, and walks under way
```

- **Largest first.** Each archive walks on its own thread. The next one starts whenever
  the chains walking number fewer than the streams allowed, plus two. One archive walks
  at most `--workers` chains, one for each part with work left.
- **The streams adapt.** A run starts at `--streams` (12) and adds one after every five
  minutes without a 429, up to `--max-streams` (24). A 429 pauses every walk and drops a
  stream. The log records each change, and every 10 minutes a progress line gives the
  TiB done, the streams, the request rate, and the time left at the last hour's pace, by
  count and by size.
- **Each archive is listed at its turn,** in the folder where the audit saw it. Content
  that changed since is walked as new, and an archive registered elsewhere before is
  walked here, from where its walk stopped. One that has gone, or lost a part, is
  recorded, and a copy of it elsewhere is walked instead. An archive that moved after
  the audit waits for the next audit.
- **Skipped:** archives that already have a verdict, copies of another (the same
  content elsewhere, walked only if that one has gone), compressed names such as
  `.tar.gz`, and sets of parts with gaps.
  An archive whose first bytes are gzip, bzip2, xz or zstd is recorded `unsupported`.
- **Stopping and failures.** SIGINT or SIGTERM stops every walk at its next batch, and
  the run exits 0; run it again to resume. An archive whose walk ends in `error` is
  tried once more at the end of the queue. The third unexpected failure stops the run,
  with status 1.

`archive import --db TARGET --from SOURCE` copies every archive of another archive
database into TARGET, finished or mid-walk, and replaces one already there with the
same content. `archive forget --db DB --archive NAME --folder FOLDER` drops an archive
and everything recorded about it, so that the next run walks it from scratch.

## Kinds of data

`kinds` says what kinds of data a listing holds (simulated and recorded waveforms,
velocity models, xyts, rupture sources, IM and hazard results), whether each byte sits in
a loose file or inside a tarball, and how old it is:

```bash
python -m dbaudit kinds --db audit.db --archive-db archives.db --rules site.toml
python -m dbaudit kinds --db audit.db --subtree /TeamSpace/proj00001 --out kinds-csv
```

It prints each group's size, share, loose bytes and bytes in tarballs, with the group's
kinds below it when it has more than one; the bytes in each group older than 6 months and
1, 2 and 5 years; and the bytes per group per year. It reads both databases, writes
neither, and never contacts Dropbox. A walk over 15 M files and 7 M tar members takes
about two minutes.

- `--archive-db` is the archive index that `archive run` writes. Without it, every
  tarball is judged by its name.
- `--rules` names a TOML file of the folder words that only mean something on your
  Dropbox. Every key is optional, and without the file the built-in rules apply alone:

  ```toml
  recorded = ["observed_db"]                # words in a folder path that mark recorded waveforms
  im_folders = ["im_results", "hazard"]     # folders that hold IM and hazard results
  lf_converted = ["lf_netcdf"]              # words that mark LF converted to netCDF
  vm_folders = ["velmodels"]                # more folder names that hold velocity models
  keep = ["/teamspace/scratch/proj00001/"]  # path prefixes whose velocity models are kept
  ```

- `--subtree` covers one folder instead of the crawl root.
- `--out DIR` also writes four CSV files, in whole bytes: `kinds.csv` (each kind, loose
  and in tarballs), `kinds_by_month.csv` (each kind by month), `kinds_older_than.csv`
  (each kind older than 0.5, 1, 2, 3, 5 and 10 years) and `kinds_listing.csv` (the root,
  when it was listed, whether the crawl finished, which databases and rules file were
  read, and the bytes with no date, which the age and month figures leave out).

How the figures are made:

- **Kinds come from names and folders only.** The file-name rules follow the outputs of
  EMOD3D-based simulations. They run in order, so a file named like an xyts or a velocity
  model wins over its folder. Every byte gets a kind, but it is a first cut.
- **Every copy counts,** wherever it sits.
- **Tarballs.** A tarball, or a part of one, whose content hash the archive index knows is
  shared out over the archive's listed members by size, when that listing can be trusted:
  the walk finished (`complete`, or `truncated` because the tar itself ends early), or it
  lists at least half the archive's bytes. Any other tarball is judged by its name and
  folder, and an LF tarball among them is split 30% seis, 68% xyts and 1% other, as the
  listed ones were. Bytes in a tarball count toward the folder that holds it.
- **Dates.** A loose file is dated by its `client_modified`, moved back to the oldest copy
  of the same content. (Dropbox's own `server_modified` is reset when a folder moves.) A
  member of a tarball is dated by its own time in the tar, and a tarball judged by its
  name by its `client_modified`. Ages count back from when the listing was finished.
- **Kept velocity models,** those under a `keep` prefix, are a group of their own.

## Known limits

- **Team folders the account has not joined are invisible.** rclone's token has no
  team-admin scope, so the crawl only sees what the account itself can see.
- **Version history is inferred, not itemised.** Listing revisions file by file
  would take months, so it shows up only as the reconciliation gap.
- **GNU and ustar tars only.** pax archives are refused rather than indexed, because
  pax can keep a member's size outside the checksummed header.
- **A part smaller than one request is always asked the same way.** A request never
  crosses into another part, so one that already spans a whole part has nowhere to
  move, and a fault tied to how it is asked would be confirmed. In practice only a
  small last part is like this.

## Development

```bash
uv run pytest                                        # offline; live tests are skipped
DBAUDIT_LIVE=1 uv run pytest tests/test_live.py      # crawls DBAUDIT_LIVE_ROOT for real
DBAUDIT_FUZZ=300 uv run pytest tests/test_confirming_reads_fuzz.py   # 300 random tars
uvx lefthook install                                 # once per clone: CI's checks run before every push
uvx lefthook run pre-push --all-files --no-tty       # CI's checks now, without pushing
uv run pytest --cov=dbaudit                          # coverage; CI requires 95%
```

CI runs the team's shared checks (ucgmsim/meta-ci-action) on every push to main.

The Dropbox API sits behind small interfaces, so the crawler and the tar walker are
tested against fakes. These replay canned pages and byte ranges, including 429s with
`Retry-After`, cursor resets, interruption mid-crawl and reads that return the
wrong bytes, once or every time a request is asked a certain way. Each test draws
its request shapes from a generator seeded by its name, so a failure repeats;
`DBAUDIT_SEED=N` reseeds the run, and `--unshaped` asks as dbaudit did before shapes.
