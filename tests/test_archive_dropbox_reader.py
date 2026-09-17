import json

import pytest

from dbaudit.archive.parts import ArchiveSet
from dbaudit.archive.reader import (ALLOWED_URLS, DOWNLOAD_URL, DropboxRangeReader,
                                    ReaderError)
from dbaudit.limiter import AdaptiveLimiter


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
