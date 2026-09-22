#!/usr/bin/env python3
"""Are members that share a name and a size actually the same bytes?

The index records headers, not content, so it can say two files are the same size and
nothing more. Stochastic fields -- a velocity model perturbation, say -- have identical
sizes by construction and different bytes by design, so size alone must not be read as
duplication. This samples a few windows from each candidate and compares them.

Sampling proves difference, never identity: windows that match leave the files *likely*
duplicates, and the report says so. Windows that differ settle it outright.
"""
from __future__ import annotations

import argparse, collections, hashlib, os, sqlite3, sys, threading
from concurrent.futures import ThreadPoolExecutor

for _root in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getcwd()):
    if os.path.isdir(os.path.join(_root, "dbaudit")):
        sys.path.insert(0, _root)
        break

from dbaudit.archive.parts import ArchiveSet, Part
from dbaudit.archive.reader import ConcatFile, DropboxRangeReader, LocalRangeReader
from dbaudit.auth import TokenProvider
from dbaudit.limiter import AdaptiveLimiter

WINDOW = 64 << 10


def windows(data_offset: int, size: int, n: int) -> list[tuple[int, int]]:
    """Up to `n` (offset, length) samples inside one member, first and last included.

    Every sample stays inside the member: a member no bigger than one window is read
    whole, never a full window that runs on into whatever the archive holds next -- that
    made two byte-identical small files look different.
    """
    if size <= WINDOW:
        return [(data_offset, size)]
    if n <= 1:
        return [(data_offset, WINDOW)]
    span = size - WINDOW
    return [(data_offset + (span * i // (n - 1) // 8) * 8, WINDOW) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="archives2.db")
    ap.add_argument("--archive", default="v01p0_incomplete.tar")
    ap.add_argument("--remote", default="dropbox")
    ap.add_argument("--local-dir")
    ap.add_argument("--like", default="v01p0/Data/VMs/%",
                    help="only compare members whose dir matches this SQL LIKE")
    ap.add_argument("--group-depth", type=int, default=4,
                    help="how many path components make a group's scope")
    ap.add_argument("--samples", type=int, default=4, help="windows per member")
    ap.add_argument("--members", type=int, default=3,
                    help="members compared per group; 0 samples every one and clusters them")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM archives WHERE name=?", (args.archive,)).fetchone()
    parts = [Part(idx=p["idx"], name=p["name"], size=p["size"], offset=p["offset"],
                  path_display=p["path_display"] or "", dbx_id=p["dbx_id"] or "",
                  rev=p["rev"] or "", content_hash=p["content_hash"] or "")
             for p in con.execute("SELECT * FROM parts WHERE archive_id=? ORDER BY idx",
                                  (row["id"],))]
    archive_set = ArchiveSet(parts)

    groups = collections.defaultdict(list)
    for m in con.execute("SELECT data_offset,size,dir,name FROM members "
                         "WHERE type='0' AND dir LIKE ? ORDER BY dir", (args.like,)):
        scope = "/".join(m["dir"].split("/")[:args.group_depth])
        groups[(scope, m["name"], m["size"])].append(m)
    candidates = {k: v for k, v in groups.items() if len(v) > 1}
    total = sum(k[2] * (len(v) - 1) for k, v in candidates.items())
    print(f"{len(candidates)} groups of same-name same-size members; "
          f"{total / 2**40:.2f} TiB sits in copies beyond the first\n")

    limiter = AdaptiveLimiter(rps=5.0, max_concurrency=args.workers)
    tokens = None if args.local_dir else TokenProvider(remote=args.remote)
    readers = []
    # One reader per thread, built once. Indexing a shared list by job number let two
    # threads share a `requests.Session`, which the rest of this codebase avoids
    # (Ruling T7-3), and grew the list past `--workers` under a race -- every extra
    # reader paying for its own TLS handshake. That is what made the first full run
    # take four times its estimate.
    tls = threading.local()

    def mine():
        c = getattr(tls, "concat", None)
        if c is None:
            r = (LocalRangeReader(args.local_dir, archive_set) if args.local_dir
                 else DropboxRangeReader(archive_set, tokens, limiter))
            readers.append(r)
            c = tls.concat = ConcatFile(archive_set, r, window_min=1, window_max=1)
        return c

    def fetch(job):
        at, length = job
        c = mine()
        c.seek(at)
        return hashlib.sha256(c.read(length)).hexdigest()[:16]

    verdicts = collections.Counter()
    reclaimable = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (scope, name, size), members in sorted(
                candidates.items(), key=lambda kv: -kv[0][2] * (len(kv[1]) - 1)):
            picked = members if args.members == 0 else members[:args.members]
            jobs, owner = [], []
            for m in picked:
                for sample in windows(m["data_offset"], size, args.samples):
                    owner.append(m)
                    jobs.append(sample)
            digests = list(pool.map(fetch, jobs))
            per = collections.defaultdict(list)
            for m, d in zip(owner, digests):
                per[m["data_offset"]].append(d)
            clusters = collections.Counter(tuple(v) for v in per.values())
            same = len(clusters) == 1
            verdicts["same" if same else "different"] += 1
            # Every member past the first in each cluster is a copy of that cluster's
            # first. With --members 0 this is the whole group, so the figure is real
            # rather than extrapolated.
            redundant = sum(n - 1 for n in clusters.values())
            if args.members == 0:
                reclaimable += size * redundant
            elif same:
                reclaimable += size * (len(members) - 1)
            print(f"  {name:22} {size / 2**30:7.1f} GiB x {len(members):>3} in {scope}")
            how = ("read whole" if size <= WINDOW
                   else f"sampled at {len(windows(0, size, args.samples))} windows")
            print(f"    {len(picked)} {how} -> "
                  f"{len(clusters)} distinct content signature(s)"
                  f"{'  [all the same]' if same else ''}")
            if len(clusters) < len(picked):
                shape = ", ".join(f"{n}x" for n in sorted(clusters.values(), reverse=True)
                                  if n > 1)
                print(f"      repeats: {shape} -> {redundant} of {len(picked)} are copies, "
                      f"{size * redundant / 2**30:,.1f} GiB")

    print(f"\n=== {verdicts['same']} groups identical throughout, "
          f"{verdicts['different']} holding more than one distinct content ===")
    scope = ("copies beyond the first within each content cluster" if args.members == 0
             else "copies beyond the first in the wholly-identical groups")
    print(f"  {scope}: {reclaimable / 2**40:.2f} TiB "
          f"(matching at every sampled window; sampling never proves identity)")
    req = sum(r.requests for r in readers)
    got = sum(r.bytes_fetched for r in readers)
    print(f"  cost: {req} requests, {got / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
