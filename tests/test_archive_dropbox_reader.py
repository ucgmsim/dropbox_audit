import json

import pytest

from dbaudit.archive.parts import ArchiveSet
from dbaudit.archive.reader import (ALLOWED_URLS, DOWNLOAD_URL, SHAPES, ConcatFile,
                                    DropboxRangeReader, ReaderError, ReadStopped)
from dbaudit.archive.tarwalk import walk
from dbaudit.limiter import AdaptiveLimiter
from tests.archive_fakes import build_tar


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None, body=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self._body = body if body is not None else {}

    def json(self):
        return self._body

    @property
    def text(self):
        return json.dumps(self._body)


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTokens:
    def __init__(self):
        self.invalidated = 0

    def access_token(self):
        return "tok"

    def root_namespace_id(self):
        return "ns1"

    def invalidate(self):
        self.invalidated += 1


class CountingLimiter(AdaptiveLimiter):
    """Records the acknowledgements the reader sends, which are otherwise invisible."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.successes = 0

    def on_success(self):
        self.successes += 1
        super().on_success()


def make(*responses, retries=5, limiter=None):
    archive = ArchiveSet.from_entries(
        [{"name": "a.tar.aa", "size": 4096, "path_display": "/d/a.tar.aa",
          "id": "id:aa", "rev": "r", "content_hash": "ab" * 32}], "a.tar")
    session = FakeSession(*responses)
    reader = DropboxRangeReader(archive, FakeTokens(),
                                limiter or AdaptiveLimiter(rps=1000.0),
                                session=session, retries=retries, sleep=lambda _s: None)
    return reader, session


def ok(offset, length, total=4096, payload=None):
    return FakeResponse(206, payload or b"\x00" * length,
                        {"Content-Range": f"bytes {offset}-{offset + length - 1}/{total}"})


def test_a_range_read_returns_the_bytes_and_asks_the_right_question():
    reader, session = make(ok(0, 512, payload=b"h" * 512))
    assert reader.read_range(0, 0, 512) == b"h" * 512
    url, headers = session.calls[0]
    assert url == DOWNLOAD_URL and url in ALLOWED_URLS
    assert headers["Range"] == "bytes=0-511"
    assert json.loads(headers["Dropbox-API-Arg"]) == {"path": "id:aa"}   # id, not path
    assert "Content-Type" not in headers          # content endpoints reject application/json
    assert reader.requests == 1 and reader.bytes_fetched == 512


def test_a_whole_file_answer_is_a_hard_failure():
    """A 200 means the server ignored the range; using those bytes would corrupt the index."""
    reader, _ = make(FakeResponse(200, b"x" * 4096))
    with pytest.raises(ReaderError, match="206"):
        reader.read_range(0, 0, 512)


def test_a_different_range_than_asked_is_a_hard_failure():
    reader, _ = make(FakeResponse(206, b"x" * 512, {"Content-Range": "bytes 512-1023/4096"}))
    with pytest.raises(ReaderError, match="different range"):
        reader.read_range(0, 0, 512)


def test_a_short_body_is_a_hard_failure():
    reader, _ = make(FakeResponse(206, b"x" * 100,
                                  {"Content-Range": "bytes 0-511/4096"}))
    with pytest.raises(ReaderError):
        reader.read_range(0, 0, 512)


def test_rate_limiting_parks_every_reader_then_retries():
    limited = FakeResponse(429, body={"error": {"retry_after": 7}},
                           headers={"Retry-After": "7"})
    reader, session = make(limited, ok(0, 512))
    assert len(reader.read_range(0, 0, 512)) == 512
    assert reader.limiter.rate_limit_events == 1
    assert reader.limiter.paused_until > 0
    assert len(session.calls) == 2


def test_an_expired_token_is_refreshed_once_then_retried():
    reader, session = make(FakeResponse(401, body={"error": "expired"}), ok(0, 512))
    assert len(reader.read_range(0, 0, 512)) == 512
    assert reader.tokens.invalidated == 1
    assert len(session.calls) == 2


def test_a_validated_read_tells_the_limiter_it_succeeded():
    """AdaptiveLimiter steps concurrency down on every 429 and back up only in
    on_success. Without this call a multi-hour walk on a shared account ratchets to a
    single stream and never recovers. Only a *validated* 206 counts -- a body that
    failed validation is not evidence the account is healthy.
    """
    limiter = CountingLimiter(rps=1000.0)
    reader, _ = make(ok(0, 512), limiter=limiter)
    reader.read_range(0, 0, 512)
    assert limiter.successes == 1

    limiter = CountingLimiter(rps=1000.0)
    reader, _ = make(FakeResponse(206, b"x" * 512, {"Content-Range": "bytes 512-1023/4096"}),
                     limiter=limiter)
    with pytest.raises(ReaderError):
        reader.read_range(0, 0, 512)
    assert limiter.successes == 0


def test_transient_failures_are_retried_and_then_give_up():
    reader, session = make(FakeResponse(503), FakeResponse(503), FakeResponse(503),
                           retries=3)
    with pytest.raises(ReaderError):
        reader.read_range(0, 0, 512)
    assert len(session.calls) == 3


def test_a_long_run_of_429s_does_not_use_up_a_reads_tries():
    """Every reader waits out each pause together, as folder listings do: a read that
    meets a storm of 429s is asked again until Dropbox answers, not failed after five."""
    limited = FakeResponse(429, body={"error": {"retry_after": 0}})
    reader, session = make(*[limited] * 6, ok(0, 512), retries=5)
    assert len(reader.read_range(0, 0, 512)) == 512
    assert len(session.calls) == 7


def test_a_read_gives_way_to_its_callers_stop_while_dropbox_pushes_back():
    """With 429s no longer counted, a stop is what ends such a read early -- as a stop,
    not a failure, so the walk hands its segment back."""
    stop = []
    limited = FakeResponse(429, body={"error": {"retry_after": 0}})
    reader, _ = make()

    class Storm(FakeSession):
        def post(self, url, headers=None, timeout=None):
            if len(self.calls) == 2:
                stop.append(True)               # a signal lands mid-storm
            return super().post(url, headers, timeout)

    reader.session = Storm(*[limited] * 3, ok(0, 512))
    reader.should_stop = lambda: bool(stop)
    with pytest.raises(ReadStopped):
        reader.read_range(0, 0, 512)
    assert len(reader.session.calls) == 3


#: Dropbox stores a file in blocks of this size.
BLOCK = 4 << 20


class DropboxLikeSession:
    """Serves one part's bytes as Dropbox does, including its one known wrong answer: a
    range that starts on the last byte of a 4 MiB block comes back as that block from its
    start, under a 206 with the asked-for Content-Range and length. Measured live on
    2026-10-15, and the same every time it was asked.
    """

    def __init__(self, data):
        self.data = data
        self.ranges = []

    def post(self, url, headers=None, timeout=None):
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append((first, last))
        start = first - first % BLOCK if first % BLOCK == BLOCK - 1 else first
        return FakeResponse(206, self.data[start:start + last - first + 1],
                            {"Content-Range": f"bytes {first}-{last}/{len(self.data)}"})


def dropbox_like(data):
    archive = ArchiveSet.from_entries(
        [{"name": "a.tar.aa", "size": len(data), "path_display": "/d/a.tar.aa",
          "id": "id:aa", "rev": "r", "content_hash": "ab" * 32}], "a.tar")
    session = DropboxLikeSession(data)
    reader = DropboxRangeReader(archive, FakeTokens(), AdaptiveLimiter(rps=1000.0),
                                session=session, sleep=lambda _s: None)
    return archive, reader, session


def test_a_range_starting_on_the_last_byte_of_a_block_still_reads_true():
    data = (bytes(range(251)) * (2 * BLOCK // 251 + 64))[:2 * BLOCK + 4096]
    _, reader, session = dropbox_like(data)
    for offset in (BLOCK - 1, 2 * BLOCK - 1):    # every block's last byte, not just the first's
        assert reader.read_range(0, offset, 1024) == data[offset:offset + 1024]
    # Anywhere else, the range asked for is exactly the one wanted.
    for offset in (0, BLOCK - 2, BLOCK, 2 * BLOCK - 2):
        session.ranges.clear()
        assert reader.read_range(0, offset, 1024) == data[offset:offset + 1024]
        assert session.ranges == [(offset, offset + 1023)]


def test_a_walk_reaches_a_header_that_starts_a_block():
    """What broke the October inventory. tarfile reads the byte before each header it
    seeks to, so a header that starts a 4 MiB block was fetched from the block before's
    last byte, came back as that block's start, and the walk called the archive corrupt
    there -- twice, since the confirming read asked the same way."""
    tar = build_tar([("first.bin", b"\x5a" * (BLOCK - 512)), ("second.bin", b"x" * 1000)])
    archive, reader, _ = dropbox_like(tar)
    members = []
    # Shape 0 asks from the byte before the header, as October's walks did: the guard's
    # to get right, not the confirming pass's.
    result = walk(ConcatFile(archive, reader, shape=0), 0,
                  lambda batch, _next: members.extend(batch))
    assert result.state == "complete"
    assert [(m.name, m.hdr_offset) for m in members] == [("first.bin", 0),
                                                         ("second.bin", BLOCK)]


def test_a_lead_that_lands_on_a_blocks_last_byte_is_still_guarded():
    """A lead can put a request's start on a block's last byte; the guard must still move
    it. Shape 30,599 has a lead of one byte, so a read at a block's first byte asks from
    the last byte of the block before."""
    data = (bytes(range(251)) * (2 * BLOCK // 251 + 64))[:2 * BLOCK + 4096]
    archive, reader, session = dropbox_like(data)
    for pos, lead in ((BLOCK, 1), (BLOCK + 1000, 1001)):
        handle = ConcatFile(archive, reader, window_min=1024, window_max=1024,
                            shape=lead * 30_599 % SHAPES)
        session.ranges.clear()
        handle.seek(pos)
        assert handle.read(512) == data[pos:pos + 512]
        assert session.ranges == [(BLOCK - 2, pos + 1023 + lead)]


def test_the_block_edge_fault_is_caught_even_without_its_guard(monkeypatch):
    """Dropbox's own fault with the guard that avoids it switched off. Shape 0 asks from
    the byte before the header that starts the second block, as the October walks did, and
    gets that block's start. The confirming pass asks in the next shape and is answered
    truly, so the walk ends complete -- where October's called the archive corrupt."""
    monkeypatch.setattr(DropboxRangeReader, "read_range", DropboxRangeReader._read_range)
    tar = build_tar([("first.bin", b"\x5a" * (BLOCK - 512)), ("second.bin", b"x" * 1000)])
    archive, reader, session = dropbox_like(tar)
    members = {}
    result = walk(ConcatFile(archive, reader, shape=0), 0,
                  lambda batch, _next: members.update((m.hdr_offset, m.name) for m in batch))
    assert result.state == "complete", result.detail
    assert members == {0: "first.bin", BLOCK: "second.bin"}
    assert f"re-read at {BLOCK}" in result.detail
    assert any(first == BLOCK - 1 for first, _ in session.ranges), "the fault was not asked"
