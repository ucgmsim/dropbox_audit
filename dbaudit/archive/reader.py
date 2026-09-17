"""Range readers, and the file-like view the tar walker reads through.

A tar header is 512 bytes, but a request costs a round trip, so reads go through a
window: a fetch pulls a little more than asked, and the following headers are served
from it. The window doubles while the walk keeps landing inside it and resets after a
jump past its end -- so a run of small members costs one request for many headers,
while a run of large ones costs the minimum per header.

The bounds are measured, not guessed. Against the live account a request costs
1.60 s + 0.044 s/MiB, so time is charged per request and size is nearly free below a
few MiB. That argues for a generous cap: in a dense pocket one 16 MiB fetch serves
hundreds of headers. It also argues for a small floor, because the floor is paid on
every jump and 99.3% of that archive's bytes sit inside members larger than 2 MiB --
at a 1 MiB floor the walk drags ~210 GiB, at 64 KiB it drags ~13 GiB for the same
wall time.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

WINDOW_MIN = 1 << 16
WINDOW_MAX = 16 << 20


class ReaderError(RuntimeError):
    """A range read could not be satisfied as asked."""


class ShortRead(ReaderError):
    """Fewer bytes came back than were requested."""


class RangeReader(Protocol):
    """The seam that keeps the walker testable without a network."""

    requests: int
    bytes_fetched: int

    def read_range(self, part_idx: int, offset: int, length: int) -> bytes: ...


class LocalRangeReader:
    """Parts as files in a directory: used by the tests, and for local archives."""

    def __init__(self, directory, archive):
        self.directory = Path(directory)
        self.archive = archive
        self.requests = 0
        self.bytes_fetched = 0

    def read_range(self, part_idx: int, offset: int, length: int) -> bytes:
        part = self.archive.parts[part_idx]
        with open(self.directory / part.name, "rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ShortRead(f"{part.name}: asked {length} at {offset}, got {len(data)}")
        self.requests += 1
        self.bytes_fetched += len(data)
        return data


class ConcatFile:
    """A seekable, read-only file over an ArchiveSet. Enough of the interface for tarfile."""

    def __init__(self, archive, reader, window_min: int = WINDOW_MIN,
                 window_max: int = WINDOW_MAX):
        self.archive = archive
        self.reader = reader
        self._window_min = window_min
        self._window_max = window_max
        self._window = window_min
        self._cache = b""
        self._cache_start = -1          # nothing cached: the first fetch uses window_min
        self._pos = 0

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self.archive.total_size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise ValueError("ConcatFile does not support unbounded reads")
        end = min(self._pos + size, self.archive.total_size)
        if end <= self._pos:
            return b""
        if not (self._cache_start <= self._pos
                and end <= self._cache_start + len(self._cache)):
            self._fill(self._pos, end - self._pos)
        start = self._pos - self._cache_start
        data = self._cache[start:start + (end - self._pos)]
        self._pos += len(data)
        return data

    def _fill(self, pos: int, need: int) -> None:
        cache_end = self._cache_start + len(self._cache)
        gap = pos - cache_end
        if self._cache_start >= 0 and 0 <= gap <= self._window:
            self._window = min(self._window * 2, self._window_max)
        else:
            self._window = self._window_min
        length = min(max(need, self._window), self.archive.total_size - pos)
        chunks = [self.reader.read_range(idx, offset, count)
                  for idx, offset, count in self.archive.slices(pos, length)]
        self._cache = b"".join(chunks)
        self._cache_start = pos
