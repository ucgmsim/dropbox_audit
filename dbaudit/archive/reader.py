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
import json
import time
from pathlib import Path
from typing import Protocol

WINDOW_MIN = 1 << 16
WINDOW_MAX = 16 << 20

CONTENT_BASE = "https://content.dropboxapi.com/2"
DOWNLOAD_URL = f"{CONTENT_BASE}/files/download"
#: Every URL this module may contact. The audit is read-only, and a test asserts this.
ALLOWED_URLS = frozenset({DOWNLOAD_URL})

DEFAULT_TIMEOUT = 180


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
        #: Fetches made so far. The walker tells from this, less the fetches it made only to
        #: read one place again, when a fetch of its own served a header -- part of how it
        #: knows which headers to hold back and read again to confirm a verdict.
        self.fills = 0

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

    def cached(self) -> tuple[int, int]:
        """The stretch of the archive the last fetch holds: (start, end), empty if none."""
        return self._cache_start, self._cache_start + len(self._cache)

    def drop_cache(self, keep_window: bool = False) -> None:
        """Forget the fetched window, so the next read asks the server again.

        A read that came back wrong is indistinguishable from one that came back right
        until something reads those bytes a second time -- and a second read served out
        of this cache is the same read. The walker drops it before reading again the
        fetches behind any verdict it reaches, so that verdict is read twice.

        ``keep_window`` keeps the size the window had grown to, for re-reading the same
        stretch of archive: in a dense run of small members the window is at 16 MiB, and
        starting again from the floor would cost several fetches to grow back.
        """
        self._cache = b""
        self._cache_start = -1
        if not keep_window:
            self._window = self._window_min

    def _fill(self, pos: int, need: int) -> None:
        cache_end = self._cache_start + len(self._cache)
        gap = pos - cache_end
        if self._cache_start >= 0 and 0 <= gap <= self._window:
            self._window = min(self._window * 2, self._window_max)
        elif self._cache_start >= 0:
            self._window = self._window_min
        # With nothing cached the window is what drop_cache left: the floor, or kept.
        length = min(max(need, self._window), self.archive.total_size - pos)
        chunks = [self.reader.read_range(idx, offset, count)
                  for idx, offset, count in self.archive.slices(pos, length)]
        self._cache = b"".join(chunks)
        self._cache_start = pos
        self.fills += 1


class DropboxRangeReader:
    """Reads byte ranges from Dropbox with ``files/download``.

    Parts are addressed by file id, so a part that is moved or renamed mid-walk still
    resolves -- which is not hypothetical: all 18 parts of v01p0_incomplete moved
    between folders in the five weeks before this was written.

    ``TokenProvider.headers()`` is deliberately not reused: it sends
    ``Content-Type: application/json``, which Dropbox's content endpoints reject.
    """

    def __init__(self, archive, tokens, limiter, session=None, retries: int = 5,
                 timeout: int = DEFAULT_TIMEOUT, sleep=None):
        self.archive = archive
        self.tokens = tokens
        self.limiter = limiter
        self.retries = retries
        self.timeout = timeout
        self.sleep = sleep or time.sleep
        self.requests = 0
        self.bytes_fetched = 0
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def _headers(self, target: str, offset: int, length: int) -> dict:
        return {
            "Authorization": f"Bearer {self.tokens.access_token()}",
            "Dropbox-API-Path-Root": json.dumps(
                {".tag": "root", "root": self.tokens.root_namespace_id()}),
            "Dropbox-API-Arg": json.dumps({"path": target}),
            "Range": f"bytes={offset}-{offset + length - 1}",
        }

    def read_range(self, part_idx: int, offset: int, length: int) -> bytes:
        part = self.archive.parts[part_idx]
        target = part.dbx_id or part.path_display
        last = None
        for attempt in range(self.retries):
            try:
                # The limiter is account-wide: a 429 anywhere parks every reader, and
                # this is where a parked reader waits.
                with self.limiter.slot():
                    response = self.session.post(
                        DOWNLOAD_URL, headers=self._headers(target, offset, length),
                        timeout=self.timeout)
            except Exception as exc:                      # network, DNS, read timeout
                last = ReaderError(f"{part.name}: {type(exc).__name__}: {exc}")
                self.sleep(min(2 ** attempt, 30))
                continue

            status = response.status_code
            if status == 206:
                return self._validate(part, response, offset, length)
            if status == 429:
                self.limiter.on_rate_limited(_retry_after(response))
                last = ReaderError(f"{part.name}: rate limited")
                continue
            if status == 401:
                self.tokens.invalidate()
                last = ReaderError(f"{part.name}: token expired")
                continue
            if status >= 500:
                last = ReaderError(f"{part.name}: {status} from Dropbox")
                self.sleep(min(2 ** attempt, 30))
                continue
            if status == 200:
                raise ReaderError(
                    f"{part.name}: expected 206 for a range request, got 200 -- the "
                    "server ignored the Range header and sent the whole file")
            raise ReaderError(f"{part.name}: {status} from Dropbox: "
                              f"{getattr(response, 'text', '')[:200]}")
        raise last or ReaderError(f"{part.name}: no attempts made")

    def _validate(self, part, response, offset, length) -> bytes:
        wanted = f"bytes {offset}-{offset + length - 1}/"
        got = response.headers.get("Content-Range", "")
        if not got.startswith(wanted):
            raise ReaderError(f"{part.name}: server answered a different range: "
                              f"asked {wanted!r}, got {got!r}")
        data = response.content
        if len(data) != length:
            raise ShortRead(f"{part.name}: asked {length} bytes at {offset}, "
                            f"got {len(data)}")
        # The limiter takes a worker off on every 429 and puts one back only here.
        # Without this, one 429 costs a stream for the rest of a multi-hour walk and
        # a shared account ratchets the pool down to a single reader.
        self.limiter.on_success()
        self.requests += 1
        self.bytes_fetched += len(data)
        return data


def _retry_after(response) -> float:
    body = {}
    try:
        body = response.json()
    except Exception:
        pass
    error = body.get("error") if isinstance(body, dict) else None
    value = error.get("retry_after") if isinstance(error, dict) else None
    if value is None:
        value = response.headers.get("Retry-After", 60)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 60.0
