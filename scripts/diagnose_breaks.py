#!/usr/bin/env python3
"""Size the damage at a tar archive's breaks. Reads bytes; writes nothing.

Two questions, cheapest first:

1. Are the bytes at the break the same ones the walk saw? A transient bad read would
   make the whole corruption finding an artefact, and two 1 KiB reads settle it.

2. Where does the header chain resume? A continuous forward scan answers that exactly,
   and it costs exactly as much as the answer is far away: it stops at the first valid
   header. Probing would be cheaper only if the damage turned out to be enormous, and
   `--cap-gib` is what bounds that case.

Every window fetched is also checked for `ustar` magic *off* the 512-byte grid. That
distinction matters: content shifted by a non-multiple of 512 is intact but invisible
to an aligned scan, and "destroyed" and "shifted" are very different answers for
whoever owns the data.
"""
from __future__ import annotations

import argparse, os, sqlite3, struct, sys, tarfile, threading, time
from concurrent.futures import ThreadPoolExecutor

for _root in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getcwd()):
    if os.path.isdir(os.path.join(_root, "dbaudit")):
        sys.path.insert(0, _root)
        break

from dbaudit.archive.parts import ArchiveSet, Part
from dbaudit.archive.reader import (WINDOW_MAX, WINDOW_MIN, ConcatFile,
                                    DropboxRangeReader, LocalRangeReader)
from dbaudit.archive.tarwalk import BLOCK, UnsettledRead, find_chain_start, walk
from dbaudit.auth import TokenProvider
from dbaudit.limiter import AdaptiveLimiter

CHUNK = 16 << 20          # reader.WINDOW_MAX, and what the cost model was measured at


def header(block: bytes):
    """A checksum-valid tar header, or None. Mirrors find_chain_start's test."""
    if len(block) < BLOCK or block[257:262] != b"ustar":
        return None
    try:
        return tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
    except tarfile.HeaderError:
        return None


def analyse(buf: bytes, base: int) -> dict:
    """Everything one window can say, without fetching another byte."""
    aligned, unaligned, magic = [], [], 0
    i = buf.find(b"ustar")
    while i != -1:
        magic += 1
        start = i - 257
        if start >= 0 and start + BLOCK <= len(buf):
            info = header(buf[start:start + BLOCK])
            if info is not None:
                (aligned if (base + start) % BLOCK == 0 else unaligned).append(
                    (base + start, info.name, info.size))
        i = buf.find(b"ustar", i + 1)
    vals = struct.unpack(f"<{len(buf) // 4}f", buf[:len(buf) // 4 * 4])
    plausible = sum(1 for v in vals if v == v and (v == 0.0 or 1e-30 < abs(v) < 1e30))
    return {"aligned": aligned, "unaligned": unaligned, "magic": magic,
            "zeros": buf.count(0), "floats": plausible, "vals": len(vals)}


def phases(buf: bytes) -> str:
    """Which byte phase, if any, reads as plausible float32 ground motion.

    A tar member's data starts on a 512-byte boundary, so a 4-byte float grid inside a
    healthy member sits at phase 0. Any other phase means the bytes came from somewhere
    that is not 4-byte aligned with here -- content from elsewhere, not this stream a
    few blocks late. Known-good members are read first to establish what phase 0 is.
    """
    out = []
    for p in range(4):
        b = buf[p:]
        n = len(b) // 4
        if n < 16:
            continue
        v = struct.unpack(f"<{n}f", b[:n * 4])
        good = [x for x in v if x == x and (x == 0.0 or 1e-8 < abs(x) < 1e4)]
        out.append(f"p{p}={len(good) * 100 // n}%")
    return " ".join(out)


def scan(concats, x: int, cap: int, total: int, chunk: int, pool, label: str) -> dict:
    """Read forward from `x` until an aligned header turns up or `cap` bytes are read.

    Chunks are fetched `len(concats)` at a time and analysed in order, so the answer is
    the same as a serial scan's; the cost of parallelism is up to one batch of overshoot.
    """
    offset = -(-x // BLOCK) * BLOCK
    read = 0
    agg = {"zeros": 0, "floats": 0, "vals": 0, "magic": 0, "unaligned": []}
    t0 = time.time()
    while read < cap and offset + read < total:
        batch = []
        for k in range(len(concats)):
            at = offset + read + k * chunk
            n = min(chunk, cap - read - k * chunk, total - at)
            if n < BLOCK:
                break
            batch.append((k, at, n))
        if not batch:
            break

        def fetch(job):
            k, at, n = job
            concats[k].seek(at)
            return at, concats[k].read(n)

        for (at, buf), (k, _, _) in zip(pool.map(fetch, batch), batch):
            if not buf:
                read = cap
                break
            r = analyse(buf, at)
            for key in ("zeros", "floats", "vals", "magic"):
                agg[key] += r[key]
            agg["unaligned"] += r["unaligned"]
            read = at + len(buf) - offset
            if r["aligned"]:
                agg.update(found=r["aligned"][0], read=read, seconds=time.time() - t0)
                return agg
        print(f"    {label}: scanned {read / 2**30:7.3f} GiB   zeros={agg['zeros'] / max(agg['vals'] * 4, 1):6.1%}  "
              f"float-like={agg['floats'] / max(agg['vals'], 1):6.1%}  ustar hits={agg['magic']}  "
              f"off-grid headers={len(agg['unaligned'])}  ({time.time() - t0:.0f}s)", flush=True)
    agg.update(found=None, read=read, seconds=time.time() - t0)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="archives.db")
    ap.add_argument("--archive", default="v01p0_incomplete.tar")
    ap.add_argument("--remote", default="dropbox")
    ap.add_argument("--members-db", help="where to look members up (default: --db). The "
                    "live DB keeps only the proven chain; the pre-join backup has them all.")
    ap.add_argument("--local-dir", help="read parts from this directory instead of Dropbox")
    ap.add_argument("--break", dest="breaks", action="append", metavar="DEAD:ALIVE",
                    help="a chain's last offset and the next offset a chain was seen alive")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--walk-gaps", action="store_true",
                    help="walk each hole end to end and check it joins the next chain")
    ap.add_argument("--verify-sample", type=int, default=0,
                    help="re-read this many recorded member headers and check the names")
    ap.add_argument("--skip-scan", action="store_true", help="skip the forward scans")
    ap.add_argument("--cap-gib", type=float, default=16.0,
                    help="most that will be read past any one break")
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
    total = archive_set.total_size
    print(f"{row['name']}: {len(parts)} parts, {total:,} bytes, state={row['state']}", flush=True)

    limiter = AdaptiveLimiter(rps=5.0, max_concurrency=args.workers)
    tokens = None if args.local_dir else TokenProvider(remote=args.remote)
    # One reader per thread: a requests.Session is not thread-safe (Ruling T7-3).
    readers = [LocalRangeReader(args.local_dir, archive_set) if args.local_dir
               else DropboxRangeReader(archive_set, tokens, limiter)
               for _ in range(args.workers)]
    concats = [ConcatFile(archive_set, r, window_min=1, window_max=1) for r in readers]
    cost = lambda: (sum(r.requests for r in readers), sum(r.bytes_fetched for r in readers))
    t0 = time.time()

    if args.breaks:
        BREAKS = [(f"break {i + 1}", int(b.split(":")[0]), int(b.split(":")[1]))
                  for i, b in enumerate(args.breaks)]
    else:
        BREAKS = [("break 1", 307_056_607_232, 322_134_851_072),
                  ("break 2", 715_934_138_368, 966_566_893_056)]

    print("\n=== 1. are the break bytes stable? (re-read what the walk saw) ===", flush=True)
    for tag, x, _ in BREAKS:
        concats[0].seek(x)
        got = concats[0].read(1024).hex()
        was = (row["detail"] or "").split(": ")[-1] if f"at {x}:" in (row["detail"] or "") else None
        verdict = ("identical to the walk's probe" if was and got.startswith(was[:128])
                   else "DIFFERS from the walk's probe!" if was else "(no stored probe to compare)")
        info = header(bytes.fromhex(got)[:BLOCK])
        print(f"  {tag} @ {x:,}: {got[:80]}...\n"
              f"           {verdict};  parses as a header? "
              f"{'YES -> ' + info.name if info else 'no'}\n"
              f"           float phase: {phases(bytes.fromhex(got))}", flush=True)

    print("\n=== 1b. control: what phase does a KNOWN-GOOD member read at? ===", flush=True)
    mcon = sqlite3.connect(f"file:{args.members_db or args.db}?mode=ro", uri=True)
    mcon.row_factory = sqlite3.Row
    picks, seen = [], set()
    for order in ("hdr_offset ASC", "hdr_offset DESC", "size DESC"):
        r = mcon.execute("SELECT data_offset,size,name FROM members WHERE type='0' "
                         f"AND size > 200000 ORDER BY {order} LIMIT 1").fetchone()
        if r and r["data_offset"] not in seen:
            seen.add(r["data_offset"]); picks.append(r)
    for c in picks:
        at = c["data_offset"] + 65536          # 65536 % 4 == 0, so the phase is preserved
        concats[0].seek(at)
        print(f"  {c['name'][-52:]:>52} @ {at:,}: {phases(concats[0].read(1024))}", flush=True)
    print("  (a healthy member should be plausible at phase 0)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for tag, x, alive in BREAKS:
            gap = alive - x
            cap = min(int(args.cap_gib * 2**30), gap)
            print(f"\n=== 2. {tag}: scanning forward from {x:,} "
                  f"(gap to the next live chain is {gap / 2**30:.2f} GiB; "
                  f"reading at most {cap / 2**30:.2f} GiB) ===", flush=True)
            if args.skip_scan:
                print("  (skipped)"); continue
            r = scan(concats, x, cap, total, args.chunk, pool, tag)
            if r["found"]:
                off, name, size = r["found"]
                print(f"  => CHAIN RESUMES at {off:,}, {(off - x) / 2**20:,.1f} MiB past the break")
                print(f"     first member there: {name} ({size:,} B)")
                print(f"     damaged span is therefore [{x:,} .. {off:,}) = "
                      f"{off - x:,} B ({(off - x) / 2**30:.3f} GiB)")
                chain, at = [], off
                for _ in range(6):        # confirm it is a chain, not one lucky block
                    concats[0].seek(at)
                    info = header(concats[0].read(BLOCK))
                    if info is None:
                        break
                    chain.append((at, info.name, info.size))
                    at += BLOCK + (-(-info.size // BLOCK)) * BLOCK
                print(f"     chains on for {len(chain)} members:")
                for a, n, s in chain:
                    print(f"       {a:>16,}  {s:>13,}  {n[-64:]}")
            else:
                print(f"  => NO aligned header in the {r['read'] / 2**30:.2f} GiB after the break")
            if r["unaligned"]:
                print(f"  !! {len(r['unaligned'])} OFF-GRID headers -- content shifted, not destroyed:")
                for a, n, s in r["unaligned"][:4]:
                    print(f"       {a:,} (off-grid by {a % BLOCK} B): {n[-58:]} ({s:,} B)")
            req, got = cost()
            print(f"  [cost so far: {req} requests, {got / 2**30:.2f} GiB, "
                  f"{time.time() - t0:.0f} s, {limiter.rate_limit_events} rate-limit events]",
                  flush=True)

    if args.verify_sample:
        print(f"\n=== 3. re-reading {args.verify_sample} recorded member headers ===", flush=True)
        picks = mcon.execute(
            "SELECT hdr_offset,name,size FROM members ORDER BY RANDOM() LIMIT ?",
            (args.verify_sample,)).fetchall()
        bad, checked = [], 0

        # One ConcatFile per thread. Sharing them is exactly the interleaved-seek race
        # that would fake the failure this whole run is trying to characterise.
        tls = threading.local()

        def mine():
            c = getattr(tls, "concat", None)
            if c is None:
                r = (LocalRangeReader(args.local_dir, archive_set) if args.local_dir
                     else DropboxRangeReader(archive_set, tokens, limiter))
                readers.append(r)
                c = tls.concat = ConcatFile(archive_set, r, window_min=1, window_max=1)
            return c

        def check(job):
            k, m = job
            c = mine()
            c.seek(m["hdr_offset"])
            info = header(c.read(BLOCK))
            return m, (info.name.rsplit("/", 1)[-1] if info else None)

        with ThreadPoolExecutor(max_workers=args.workers) as pool2:
            for m, got_name in pool2.map(check, list(enumerate(picks))):
                checked += 1
                if got_name != m["name"]:
                    bad.append((m["hdr_offset"], m["name"], got_name))
        print(f"  {checked - len(bad)}/{checked} headers matched the manifest exactly")
        for off, want, gotn in bad[:10]:
            print(f"    MISMATCH at {off:,}: manifest says {want}, archive has {gotn}")
        if not bad:
            print("  -> no read returned wrong bytes in this sample")

    if args.walk_gaps:
        for tag, x, alive in BREAKS:
            print(f"\n=== 4. {tag}: walking the hole [{x:,} .. {alive:,}) "
                  f"= {(alive - x) / 2**30:.2f} GiB ===", flush=True)
            members, at, res, heals = [], x, None, []
            for attempt in range(6):
                # A fresh reader each attempt: ConcatFile's cache is exactly what made
                # the original verdict non-independent.
                fresh_reader = (LocalRangeReader(args.local_dir, archive_set)
                                if args.local_dir
                                else DropboxRangeReader(archive_set, tokens, limiter))
                fresh = ConcatFile(archive_set, fresh_reader, window_min=WINDOW_MIN,
                                   window_max=WINDOW_MAX)
                readers.append(fresh_reader)
                try:
                    def rewind(offset):
                        members[:] = [m for m in members if m.hdr_offset < offset]

                    res = walk(fresh, at, lambda ms, nxt: members.extend(ms),
                               rewind=rewind, stop_at=alive)
                except UnsettledRead as exc:
                    # The walker re-reads for itself now, and gives up only when reads
                    # will not agree -- a finding about the server, worth reporting.
                    print(f"    reads would not settle: {exc}", flush=True)
                    break
                print(f"    attempt {attempt + 1}: {res.state} at {res.end_offset:,} "
                      f"after {len(members):,} members  ({time.time() - t0:.0f}s)", flush=True)
                if res.state != "corrupt":
                    break
                probe_reader = (LocalRangeReader(args.local_dir, archive_set)
                                if args.local_dir
                                else DropboxRangeReader(archive_set, tokens, limiter))
                readers.append(probe_reader)
                pc = ConcatFile(archive_set, probe_reader, window_min=1, window_max=1)
                pc.seek(res.end_offset)
                info = header(pc.read(BLOCK))
                if info is None:
                    print(f"    a fresh read at {res.end_offset:,} is ALSO not a header "
                          f"-- this one looks real")
                    break
                print(f"    but a FRESH read at {res.end_offset:,} parses fine: "
                      f"{info.name[-56:]} -- the walk's read was bad; resuming")
                heals.append(res.end_offset)
                at = res.end_offset
            ok = res.state == "crossed" and res.end_offset == alive
            print(f"  => {res.state} at {res.end_offset:,}; recovered {len(members):,} members"
                  f"{' over ' + str(len(heals)) + ' bad read(s)' if heals else ''}")
            print(f"  => joins the next chain exactly? "
                  f"{'YES -- this hole is intact' if ok else 'NO (expected ' + format(alive, ',') + ')'}")
            if members:
                rels = sorted({m.dir.split("/")[3] for m in members if m.dir.count("/") > 3})
                print(f"  => {len(rels)} realisations in the hole: {', '.join(rels[:14])}"
                      f"{' ...' if len(rels) > 14 else ''}")
                out = f"gap-members-{x}.tsv"
                with open(out, "w") as fh:
                    for m in members:
                        fh.write(f"{m.hdr_offset}\t{m.size}\t{m.type}\t{m.dir}/{m.name}\n")
                print(f"  => written to {out}")
            req, got = cost()
            print(f"  [cost so far: {req} requests, {got / 2**30:.2f} GiB, "
                  f"{time.time() - t0:.0f} s]", flush=True)

    req, got = cost()
    print(f"\n=== cost ===\n  {req} requests, {got / 2**30:.2f} GiB, {time.time() - t0:.0f} s, "
          f"{limiter.rate_limit_events} rate-limit events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
