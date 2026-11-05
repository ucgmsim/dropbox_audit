import pytest
import requests

from dbaudit.api import (
    DEFAULT_RETRY_AFTER,
    ApiError,
    AuthExpiredError,
    CursorResetError,
    HttpLister,
    PathNotFoundError,
    RateLimitedError,
    TransientError,
)


class FakeResp:
    def __init__(
        self,
        status: int,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.content = b""
        self.text = str(self._body)

    def json(self) -> dict:
        return self._body


class FakeSession:
    def __init__(self, resps: list[FakeResp]):
        self.resps = list(resps)
        self.calls: list[tuple[str, str | None]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        data: str | None = None,
    ) -> FakeResp:
        self.calls.append((url, data))
        return self.resps.pop(0)


class FakeTokens:
    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer x", "Content-Type": "application/json"}


def lister(resps: list[FakeResp]) -> HttpLister:
    return HttpLister(FakeTokens(), session=FakeSession(resps))


def test_list_folder_returns_page():
    lst = lister(
        [
            FakeResp(
                200, {"entries": [{".tag": "file"}], "cursor": "c1", "has_more": True}
            )
        ]
    )
    page = lst.list_folder("/TeamSpace", recursive=True)
    assert len(page.entries) == 1
    assert page.cursor == "c1"
    assert page.has_more is True


def test_list_folder_sends_expected_body():
    session = FakeSession(
        [FakeResp(200, {"entries": [], "cursor": "c", "has_more": False})]
    )
    lst = HttpLister(FakeTokens(), session=session)
    lst.list_folder("/TeamSpace", recursive=True)
    import json

    url, data = session.calls[0]
    assert data is not None
    body = json.loads(data)
    assert url.endswith("/files/list_folder")
    assert body["path"] == "/TeamSpace"
    assert body["recursive"] is True
    assert body["limit"] == 2000
    assert body["include_mounted_folders"] is True
    assert body["include_media_info"] is False


def test_continue_uses_the_continue_endpoint():
    session = FakeSession(
        [FakeResp(200, {"entries": [], "cursor": "c2", "has_more": False})]
    )
    lst = HttpLister(FakeTokens(), session=session)
    lst.continue_("cursor-1")
    url, data = session.calls[0]
    assert url.endswith("/files/list_folder/continue")
    assert data is not None
    assert "cursor-1" in data


def test_429_body_retry_after_is_used():
    lst = lister(
        [
            FakeResp(
                429,
                {
                    "error": {
                        "reason": {".tag": "too_many_requests"},
                        "retry_after": 300,
                    }
                },
            )
        ]
    )
    with pytest.raises(RateLimitedError) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after == 300


def test_429_header_retry_after_fallback():
    lst = lister([FakeResp(429, {}, {"Retry-After": "12"})])
    with pytest.raises(RateLimitedError) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after == 12


def test_429_without_any_hint_uses_a_safe_default():
    lst = lister([FakeResp(429, {})])
    with pytest.raises(RateLimitedError) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after >= 30


def test_cursor_reset_classified():
    lst = lister([FakeResp(409, {"error": {".tag": "reset"}})])
    with pytest.raises(CursorResetError):
        lst.continue_("stale-cursor")


def test_path_not_found_classified():
    lst = lister(
        [FakeResp(409, {"error": {".tag": "path", "path": {".tag": "not_found"}}})]
    )
    with pytest.raises(PathNotFoundError):
        lst.list_folder("/gone", recursive=True)


def test_other_409_is_api_error():
    lst = lister([FakeResp(409, {"error": {".tag": "other"}})])
    with pytest.raises(ApiError):
        lst.list_folder("/x", recursive=True)


def test_401_is_auth_expired():
    lst = lister([FakeResp(401, {"error_summary": "expired_access_token/"})])
    with pytest.raises(AuthExpiredError):
        lst.list_folder("/x", recursive=True)


def test_5xx_is_transient():
    lst = lister([FakeResp(503, {})])
    with pytest.raises(TransientError):
        lst.list_folder("/x", recursive=True)


def test_a_listing_asks_for_no_deleted_entries():
    """The store records what exists and never applies a delete, so a crawl must not
    ask for entries that are already gone."""
    import json

    session = FakeSession(
        [FakeResp(200, {"entries": [], "cursor": "c", "has_more": False})]
    )
    lst = HttpLister(FakeTokens(), session=session)
    lst.list_folder("/x", recursive=True)
    body = session.calls[0][1]
    assert body is not None
    assert json.loads(body)["include_deleted"] is False


def test_missing_has_more_defaults_to_false():
    lst = lister([FakeResp(200, {"entries": [], "cursor": "c"})])
    assert lst.list_folder("/x", recursive=True).has_more is False


class GarbledResp(FakeResp):
    """A reply whose body is not JSON, such as an HTML error page from a proxy."""

    def __init__(self, status: int, text: str):
        super().__init__(status)
        self.text = text

    def json(self) -> dict:
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class DroppedSession:
    """A session whose connection drops before any reply arrives."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        data: str | None = None,
    ) -> FakeResp:
        raise ConnectionResetError("peer reset the connection")


def test_a_retry_after_that_is_not_a_number_falls_back_to_the_default():
    lst = lister([FakeResp(429, {}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})])
    with pytest.raises(RateLimitedError) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after == DEFAULT_RETRY_AFTER


def test_an_unlisted_status_is_an_api_error_naming_the_status():
    lst = lister([FakeResp(400, {"error": "bad request"})])
    with pytest.raises(ApiError, match="400 from Dropbox") as excinfo:
        lst.list_folder("/x", recursive=True)
    assert not isinstance(excinfo.value, TransientError)


def test_a_dropped_connection_is_a_transient_error_that_names_the_cause():
    lst = HttpLister(FakeTokens(), session=DroppedSession())
    with pytest.raises(TransientError, match="ConnectionResetError") as excinfo:
        lst.list_folder("/x", recursive=True)
    assert isinstance(excinfo.value.__cause__, ConnectionResetError)


def test_a_garbled_error_body_still_reports_the_http_status():
    lst = lister([GarbledResp(502, "<html>Bad Gateway</html>")])
    with pytest.raises(TransientError, match="502 from Dropbox") as excinfo:
        lst.list_folder("/x", recursive=True)
    assert "Bad Gateway" in str(excinfo.value)


def test_without_a_session_a_lister_makes_a_requests_session():
    assert isinstance(HttpLister(FakeTokens()).session, requests.Session)
