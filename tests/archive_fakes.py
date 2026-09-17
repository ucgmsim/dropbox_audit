"""Helpers for the archive tests: build real tars, split them, count reads."""

import io
import tarfile
from pathlib import Path

from dbaudit.archive.parts import ArchiveSet


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
