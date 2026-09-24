#!/usr/bin/env python3
"""Read chosen members' headers again and check them against an index. Writes nothing.

An index row records what one read of the archive said. This asks the server again --
a request of its own per member -- and compares every field the row holds. It is the
check for rows that only one walk ever saw:

    python scripts/verify_members.py --db archives2.db --archive v01p0_incomplete.tar \\
        --extended --unless-matched-in archives-prejoin.db

A member whose fresh read disagrees is read once more, so a bad read by this script is
told apart from a wrong row in the index. For a GNU long name or link target it also
checks the fresh read against the checksummed header after those blocks.
`--dry-run` lists what would be read, and reads nothing.
"""
from __future__ import annotations

import argparse, os, sqlite3, sys

for _root in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getcwd()):
    if os.path.isdir(os.path.join(_root, "dbaudit")):
        sys.path.insert(0, _root)
        break

from dbaudit.archive.parts import ArchiveSet, Part
from dbaudit.archive.reader import ConcatFile, DropboxRangeReader, LocalRangeReader
from dbaudit.archive.tarwalk import BLOCK, read_member
from dbaudit.auth import TokenProvider
from dbaudit.limiter import AdaptiveLimiter

FIELDS = ("hdr_offset", "data_offset", "size", "type", "mode", "mtime", "uname", "gname",
          "dir", "name", "linkname")
WINDOW = 4 << 10          # a member's whole header sequence, with room to spare


def _connect(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _row(item):
    if isinstance(item, sqlite3.Row):
        return tuple(item[f] for f in FIELDS)
    return tuple(getattr(item, f) for f in FIELDS)


def _path(item):
    """The member's path, from a stored row or a freshly read Member."""
    field = item.__getitem__ if isinstance(item, sqlite3.Row) else item.__getattribute__
    return f"{field('dir')}/{field('name')}" if field("dir") else field("name")


def _long_fields_agree(concat, member):
    """The fresh read's long name and link target begin as the header after them says."""
    if member.data_offset - member.hdr_offset <= BLOCK or member.type == "S":
        return True
    concat.seek(member.data_offset - BLOCK)
    header = concat.read(BLOCK)
    name_field = header[0:100].split(b"\0", 1)[0].rstrip(b"/")
    link_field = header[157:257].split(b"\0", 1)[0]
    path = f"{member.dir}/{member.name}" if member.dir else member.name
    return (path.encode("utf-8", "surrogateescape").startswith(name_field)
            and member.linkname.encode("utf-8", "surrogateescape").startswith(link_field))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--remote", default="dropbox")
    ap.add_argument("--local-dir", help="read parts from this directory instead of Dropbox")
    ap.add_argument("--extended", action="store_true",
                    help="only members with a GNU long name or link target")
    ap.add_argument("--unless-matched-in", metavar="DB",
                    help="skip members this other index recorded identically")
    ap.add_argument("--offset", type=int, action="append", default=[],
                    help="only the member whose header is at this offset (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="list the members; read nothing")
    args = ap.parse_args()

    con = _connect(args.db)
    archive = con.execute("SELECT * FROM archives WHERE name=?", (args.archive,)).fetchone()
    if archive is None:
        print(f"error: no archive named {args.archive!r} in {args.db}", file=sys.stderr)
        return 2
    rows = con.execute("SELECT * FROM members WHERE archive_id=? ORDER BY hdr_offset",
                       (archive["id"],)).fetchall()
    if args.extended:
        rows = [r for r in rows if r["data_offset"] - r["hdr_offset"] > BLOCK]
    if args.offset:
        rows = [r for r in rows if r["hdr_offset"] in set(args.offset)]
    if args.unless_matched_in:
        other = _connect(args.unless_matched_in)
        other_id = other.execute("SELECT id FROM archives WHERE name=?",
                                 (args.archive,)).fetchone()["id"]
        seen = {_row(r) for r in other.execute(
            "SELECT * FROM members WHERE archive_id=?", (other_id,))}
        rows = [r for r in rows if _row(r) not in seen]
    print(f"{args.archive}: {len(rows)} member(s) to read again", flush=True)
    if args.dry_run or not rows:
        for r in rows:
            print(f"  {r['hdr_offset']:>16,}  {_path(r)}  -> {r['linkname']}"[:160])
        return 0

    parts = [Part(idx=p["idx"], name=p["name"], size=p["size"], offset=p["offset"],
                  path_display=p["path_display"] or "", dbx_id=p["dbx_id"] or "",
                  rev=p["rev"] or "", content_hash=p["content_hash"] or "")
             for p in con.execute("SELECT * FROM parts WHERE archive_id=? ORDER BY idx",
                                  (archive["id"],))]
    archive_set = ArchiveSet(parts)
    reader = (LocalRangeReader(args.local_dir, archive_set) if args.local_dir else
              DropboxRangeReader(archive_set, TokenProvider(remote=args.remote),
                                 AdaptiveLimiter(rps=5.0, max_concurrency=1)))
    concat = ConcatFile(archive_set, reader, window_min=WINDOW, window_max=WINDOW)

    def fresh(offset):
        concat.drop_cache()                  # every read is a request of its own
        return read_member(concat, offset)

    bad = 0
    for r in rows:
        first = fresh(r["hdr_offset"])
        if first is not None and _row(first) == _row(r):
            verdict = "ok" if _long_fields_agree(concat, first) else \
                "ok, but the long blocks do not begin as the header after them says"
        else:
            again = fresh(r["hdr_offset"])
            if again is not None and _row(again) == _row(r):
                verdict = "ok on a second read (this script's first read was bad)"
            elif again is not None and first is not None and _row(again) == _row(first):
                bad += 1
                verdict = (f"INDEX ROW WRONG: two fresh reads say {_path(again)!r} "
                           f"-> {again.linkname!r}, size {again.size}")
            else:
                bad += 1
                verdict = "UNSETTLED: the fresh reads disagree with each other"
        print(f"  {r['hdr_offset']:>16,}  {_path(r)[-60:]}: {verdict}", flush=True)
    print(f"{len(rows) - bad} of {len(rows)} agree with the index; "
          f"{reader.requests} request(s), {reader.bytes_fetched:,} bytes", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
