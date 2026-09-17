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


def test_refresh_is_not_attempted_on_every_call():
    """Load-bearing: rclone refuses to refresh a token that has not nearly expired.

    Without a cooldown, every API request inside the refresh margin would spawn
    rclone subprocesses -- thousands of them, hours into a long crawl.
    """
    import datetime as dt

    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=300)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    stale = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "still-valid", "expiry": soon})}}
    )
    refreshes, dumps = [], []
    clock = [0.0]

    def dump():
        dumps.append(1)
        return stale

    tp = TokenProvider(
        refresh_margin=600, refresh_cooldown=60,
        _dump_fn=dump, _refresh_fn=lambda: refreshes.append(1),
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1"}},
        _clock=lambda: clock[0],
    )
    for _ in range(50):
        assert tp.access_token() == "still-valid"
    assert len(refreshes) == 1, f"refreshed {len(refreshes)} times for 50 calls"
    assert len(dumps) <= 2

    clock[0] += 61  # past the cooldown: one more attempt is allowed
    tp.access_token()
    assert len(refreshes) == 2


def test_expired_token_still_refreshes_despite_cooldown():
    """A genuinely expired token must never be served, cooldown or not."""
    expired = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "old", "expiry": "2000-01-01T00:00:00Z"})}}
    )
    fresh = json.dumps(
        {"dropbox": {"type": "dropbox", "token": json.dumps(
            {"access_token": "new", "expiry": "2099-01-01T00:00:00Z"})}}
    )
    dumps = iter([expired, fresh, fresh, fresh])
    tp = TokenProvider(
        refresh_cooldown=1e9,
        _dump_fn=lambda: next(dumps), _refresh_fn=lambda: None,
        _account_fn=lambda tok: {"root_info": {"root_namespace_id": "1"}},
    )
    assert tp.access_token() == "new"


def test_invalidate_forces_a_reread_of_the_token():
    """After a 401 the cached token is worthless; the next call must go back to rclone."""
    calls = []

    def dump():
        calls.append(1)
        return json.dumps({"dropbox": {"token": json.dumps(
            {"access_token": f"t{len(calls)}", "expiry": "2099-01-01T00:00:00Z"})}})

    provider = TokenProvider(_dump_fn=dump, _refresh_fn=lambda: None)
    assert provider.access_token() == "t1"
    assert provider.access_token() == "t1"       # cached
    provider.invalidate()
    assert provider.access_token() == "t2"
