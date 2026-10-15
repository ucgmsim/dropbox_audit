"""Helpers for the archive tests: build real tars, split them, count reads, and a
folder-by-folder fake of Dropbox for `archive run`."""

import hashlib
import io
import random
import tarfile
from pathlib import Path

from dbaudit.api import Page, PathNotFound
from dbaudit.archive.parts import ArchiveSet
from dbaudit.archive.reader import LocalRangeReader
from dbaudit.limiter import AdaptiveLimiter
from dbaudit.store import Store


def suffix(index):
    return chr(ord("a") + index // 26) + chr(ord("a") + index % 26)


def build_tar(members, format=tarfile.GNU_FORMAT):
    """members: (name, payload) or (name, payload, attrs) tuples. Returns tar bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=format) as tf:
        for spec in members:
            name, payload = spec[0], spec[1]
            attrs = dict(spec[2]) if len(spec) > 2 else {}
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = attrs.pop("mtime", 1_700_000_000)
            info.mode = attrs.pop("mode", 0o644)
            info.uname = attrs.pop("uname", "user")
            info.gname = attrs.pop("gname", "proj00001")
            info.type = attrs.pop("type", tarfile.REGTYPE)
            for key, value in attrs.items():
                setattr(info, key, value)
            tf.addfile(info, io.BytesIO(payload) if info.size else None)
    return buf.getvalue()


def write_parts(directory, data, part_size, base="a.tar"):
    """Split `data` into part files on disk and return an ArchiveSet over them."""
    directory = Path(directory)
    entries = []
    for index, start in enumerate(range(0, max(len(data), 1), part_size)):
        chunk = data[start:start + part_size]
        name = f"{base}.{suffix(index)}"
        (directory / name).write_bytes(chunk)
        entries.append({"name": name, "size": len(chunk), "path_display": f"/d/{name}",
                        "id": f"id:{name}", "rev": "r", "content_hash": f"{index:064x}"})
    return ArchiveSet.from_entries(entries, base)


class World:
    """Archives as Dropbox would list them, with their parts on local disk."""

    def __init__(self, tmp_path):
        self.tmp_path, self.folders, self.local, self.gone = tmp_path, {}, {}, set()
        self.faults = {}            # folder -> exceptions its next listings raise, in turn
        self.page_size = None       # entries per listing page; None lists a folder in one
        self.made = 0

    def add(self, folder, name, data, part_size=None):
        self.made += 1
        directory = self.tmp_path / "local" / str(self.made)
        directory.mkdir(parents=True)
        write_parts(directory, data, part_size or max(len(data), 1), base=name)
        files = sorted(directory.iterdir())
        if len(files) == 1:                         # a whole archive is named plainly
            files[0].rename(directory / name)
        self.local[folder] = directory
        entries = self.folders.setdefault(folder, [])
        for path in sorted(directory.iterdir()):
            body = path.read_bytes()
            entries.append({".tag": "file", "name": path.name, "size": len(body),
                            "path_display": f"{folder}/{path.name}",
                            "id": f"id:{folder}/{path.name}", "rev": "r",
                            "content_hash": hashlib.sha256(body).hexdigest(),
                            "client_modified": "2026-01-01T00:00:00Z",
                            "server_modified": "2026-01-01T00:00:00Z"})
        return directory

    def audit_db(self):
        store = Store(self.tmp_path / "audit.db")
        store.init_schema()
        store.set_meta("root", "/R")
        shard = store.add_shard("/R", 1)
        store.claim_shard("test")
        entries = [{".tag": "folder", "id": "id:/R", "name": "R", "path_display": "/R"}]
        for files in self.folders.values():
            entries += files
        store.commit_page(shard, entries, cursor=None, has_more=False)
        store.close()
        return str(self.tmp_path / "audit.db")

    def install(self, monkeypatch):
        world = self

        class Lister:
            def list_folder(self, path, recursive=False):
                if world.faults.get(path):
                    raise world.faults[path].pop(0)
                if path in world.gone:
                    raise PathNotFound(f"path/not_found: {path}")
                return self.continue_(f"{path}|0")

            def continue_(self, cursor):
                path, start = cursor.rsplit("|", 1)
                entries = world.folders.get(path, [])
                end = len(entries) if world.page_size is None else int(start) + world.page_size
                return Page(entries=list(entries[int(start):end]), cursor=f"{path}|{end}",
                            has_more=end < len(entries))

        monkeypatch.setattr("dbaudit.cli.build_lister",
                            lambda remote, tokens=None: (object(), Lister()))
        monkeypatch.setattr("dbaudit.cli._reader_for", self.reader)
        # Every listing takes its turn at the run's limiter; at the real rate, the few
        # hundred in a test would take it half a minute.
        monkeypatch.setattr("dbaudit.cli._run_limiter", lambda args: AdaptiveLimiter(
            rps=1000.0, max_concurrency=args.max_streams, start_concurrency=args.streams))

    def reader(self, row, archive_set, tokens=None, limiter=None):
        """`_reader_for`, reading the parts from the archive's folder on local disk."""
        return LocalRangeReader(self.local[row["folder"]], archive_set)


def members(n, seed=0):
    rng = random.Random(seed)
    return [(f"run/m{i:03d}.bin", rng.randbytes(rng.choice([10, 700, 5000, 70_000])))
            for i in range(n)]
