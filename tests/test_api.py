import pytest

from dbaudit.api import (
    ApiError,
    AuthExpired,
    CursorReset,
    HttpLister,
    PathNotFound,
    RateLimited,
    TransientError,
)


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, resps):
        self.resps = list(resps)
        self.calls = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append((url, data))
        return self.resps.pop(0)


class FakeTokens:
    def headers(self):
        return {"Authorization": "Bearer x", "Content-Type": "application/json"}


def lister(resps):
    return HttpLister(FakeTokens(), session=FakeSession(resps))


def test_list_folder_returns_page():
    lst = lister([FakeResp(200, {"entries": [{".tag": "file"}], "cursor": "c1", "has_more": True})])
    page = lst.list_folder("/TeamSpace", recursive=True)
    assert len(page.entries) == 1
    assert page.cursor == "c1"
    assert page.has_more is True


def test_list_folder_sends_expected_body():
    lst = lister([FakeResp(200, {"entries": [], "cursor": "c", "has_more": False})])
    lst.list_folder("/TeamSpace", recursive=True)
    import json

    url, data = lst.session.calls[0]
    body = json.loads(data)
    assert url.endswith("/files/list_folder")
    assert body["path"] == "/TeamSpace"
    assert body["recursive"] is True
    assert body["limit"] == 2000
    assert body["include_mounted_folders"] is True
    assert body["include_media_info"] is False


def test_continue_uses_the_continue_endpoint():
    lst = lister([FakeResp(200, {"entries": [], "cursor": "c2", "has_more": False})])
    lst.continue_("cursor-1")
    url, data = lst.session.calls[0]
    assert url.endswith("/files/list_folder/continue")
    assert "cursor-1" in data


def test_429_body_retry_after_is_used():
    lst = lister(
        [FakeResp(429, {"error": {"reason": {".tag": "too_many_requests"}, "retry_after": 300}})]
    )
    with pytest.raises(RateLimited) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after == 300


def test_429_header_retry_after_fallback():
    lst = lister([FakeResp(429, {}, {"Retry-After": "12"})])
    with pytest.raises(RateLimited) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after == 12


def test_429_without_any_hint_uses_a_safe_default():
    lst = lister([FakeResp(429, {})])
    with pytest.raises(RateLimited) as excinfo:
        lst.list_folder("/x", recursive=True)
    assert excinfo.value.retry_after >= 30


def test_cursor_reset_classified():
    lst = lister([FakeResp(409, {"error": {".tag": "reset"}})])
    with pytest.raises(CursorReset):
        lst.continue_("stale-cursor")


def test_path_not_found_classified():
    lst = lister([FakeResp(409, {"error": {".tag": "path", "path": {".tag": "not_found"}}})])
    with pytest.raises(PathNotFound):
        lst.list_folder("/gone", recursive=True)


def test_other_409_is_api_error():
    lst = lister([FakeResp(409, {"error": {".tag": "other"}})])
    with pytest.raises(ApiError):
        lst.list_folder("/x", recursive=True)


def test_401_is_auth_expired():
    lst = lister([FakeResp(401, {"error_summary": "expired_access_token/"})])
    with pytest.raises(AuthExpired):
        lst.list_folder("/x", recursive=True)


def test_5xx_is_transient():
    lst = lister([FakeResp(503, {})])
    with pytest.raises(TransientError):
        lst.list_folder("/x", recursive=True)


def test_a_listing_asks_for_no_deleted_entries():
    """The store records what exists and never applies a delete, so a crawl must not
    ask for entries that are already gone."""
    import json

    lst = HttpLister(FakeTokens(), session=FakeSession(
        [FakeResp(200, {"entries": [], "cursor": "c", "has_more": False})]))
    lst.list_folder("/x", recursive=True)
    assert json.loads(lst.session.calls[0][1])["include_deleted"] is False


def test_missing_has_more_defaults_to_false():
    lst = lister([FakeResp(200, {"entries": [], "cursor": "c"})])
    assert lst.list_folder("/x", recursive=True).has_more is False
