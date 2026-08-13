import json

import pytest

from dbaudit.auth import AuthError, TokenProvider

FAKE_DUMP = json.dumps(
    {
        "dropbox": {
            "type": "dropbox",
            "token": json.dumps(
                {
                    "access_token": "tok-abc",
                    "refresh_token": "r",
                    "token_type": "bearer",
                    "expiry": "2099-01-01T00:00:00Z",
                }
            ),
        }
    }
)


def test_headers_include_bearer_and_path_root():
    tp = TokenProvider(
        remote="dropbox",
        _dump_fn=lambda: FAKE_DUMP,
        _refresh_fn=lambda: None,
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1234567890"}},
    )
    h = tp.headers()
    assert h["Authorization"] == "Bearer tok-abc"
    assert json.loads(h["Dropbox-API-Path-Root"]) == {".tag": "root", "root": "1234567890"}
    assert h["Content-Type"] == "application/json"


def test_expired_token_triggers_rclone_refresh():
    calls = []
    expired = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "old", "expiry": "2000-01-01T00:00:00Z"})}}
    )
    fresh = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "new", "expiry": "2099-01-01T00:00:00Z"})}}
    )
    dumps = iter([expired, fresh])
    tp = TokenProvider(
        _dump_fn=lambda: next(dumps),
        _refresh_fn=lambda: calls.append("refreshed"),
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1"}},
    )
    assert tp.headers()["Authorization"] == "Bearer new"
    assert calls == ["refreshed"]


def test_fresh_token_is_cached_and_not_refetched():
    dumps = [FAKE_DUMP]
    tp = TokenProvider(
        _dump_fn=lambda: dumps.pop(),
        _refresh_fn=lambda: None,
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1"}},
    )
    tp.headers()
    tp.headers()  # would raise IndexError if it re-read the config


def test_missing_remote_raises():
    tp = TokenProvider(
        remote="nope",
        _dump_fn=lambda: FAKE_DUMP,
        _refresh_fn=lambda: None,
        _account_fn=lambda tok: {},
    )
    with pytest.raises(AuthError, match="nope"):
        tp.headers()


def test_unparseable_config_raises_actionable_error():
    tp = TokenProvider(
        _dump_fn=lambda: "not json at all",
        _refresh_fn=lambda: None,
        _account_fn=lambda tok: {},
    )
    with pytest.raises(AuthError, match="rclone config"):
        tp.headers()


def test_root_namespace_id_is_fetched_once():
    calls = []

    def account(tok):
        calls.append(tok)
        return {"root_info": {"root_namespace_id": "42"}}

    tp = TokenProvider(
        _dump_fn=lambda: FAKE_DUMP, _refresh_fn=lambda: None, _account_fn=account
    )
    assert tp.root_namespace_id() == "42"
    assert tp.root_namespace_id() == "42"
    assert len(calls) == 1


def test_expiry_within_refresh_margin_is_treated_as_stale():
    """A token expiring in 60s must be refreshed before a long page fetch starts."""
    import datetime as dt

    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    stale = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "old", "expiry": soon})}}
    )
    dumps = iter([stale, FAKE_DUMP])
    refreshed = []
    tp = TokenProvider(
        refresh_margin=600,
        _dump_fn=lambda: next(dumps),
        _refresh_fn=lambda: refreshed.append(1),
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1"}},
    )
    assert tp.headers()["Authorization"] == "Bearer tok-abc"
    assert refreshed == [1]
