"""Dropbox credentials, borrowed from an existing rclone remote.

rclone already holds a refresh token for the account and knows how to exchange it,
so we read the access token straight out of ``rclone config dump`` and let rclone
do the refreshing. That means no second OAuth app to register and no client secret
stored here.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import threading
import time

API_BASE = "https://api.dropboxapi.com/2"
_FRACTIONAL_SECONDS = re.compile(r"\.(\d+)")


class AuthError(RuntimeError):
    """Credentials could not be obtained. The message says what to do about it."""


def _parse_rfc3339(text: str) -> dt.datetime:
    """Parse rclone's expiry stamps, which carry nanosecond precision.

    ``datetime.fromisoformat`` accepts at most microseconds, so trim the tail.
    """
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = _FRACTIONAL_SECONDS.sub(lambda m: "." + m.group(1)[:6], s, count=1)
    parsed = dt.datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _rclone_config_dump() -> str:
    try:
        proc = subprocess.run(
            ["rclone", "config", "dump"], capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError as exc:
        raise AuthError("rclone is not on PATH; dbaudit reads its credentials from rclone") from exc
    if proc.returncode != 0:
        raise AuthError(f"`rclone config dump` failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def _rclone_force_refresh(remote: str) -> None:
    """Make rclone perform a token refresh and write it back to its config."""
    subprocess.run(
        ["rclone", "about", f"{remote}:", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _fetch_account(access_token: str) -> dict:
    import requests

    resp = requests.post(
        f"{API_BASE}/users/get_current_account",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=60,
    )
    if resp.status_code != 200:
        raise AuthError(f"users/get_current_account failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()


class TokenProvider:
    """Supplies request headers for the Dropbox API, refreshing as needed.

    Thread-safe: every crawler worker calls :meth:`headers` on the hot path.
    """

    def __init__(
        self,
        remote: str = "dropbox",
        refresh_margin: int = 600,
        refresh_cooldown: float = 60.0,
        *,
        _dump_fn=None,
        _refresh_fn=None,
        _account_fn=None,
        _clock=None,
    ):
        self.remote = remote
        self.refresh_margin = refresh_margin
        self.refresh_cooldown = refresh_cooldown
        self._clock = _clock or time.monotonic
        self._last_refresh_attempt = float("-inf")
        self._dump_fn = _dump_fn or _rclone_config_dump
        self._refresh_fn = _refresh_fn or (lambda: _rclone_force_refresh(remote))
        self._account_fn = _account_fn or _fetch_account
        self._lock = threading.Lock()
        self._token: dict | None = None
        self._account: dict | None = None

    # ---- internals -----------------------------------------------------

    def _read_token(self) -> dict:
        raw = self._dump_fn()
        try:
            config = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AuthError(
                "could not parse `rclone config dump` output; "
                "if the rclone config is encrypted, set RCLONE_CONFIG_PASS"
            ) from exc
        if self.remote not in config:
            raise AuthError(
                f"rclone remote {self.remote!r} not found; "
                f"available: {', '.join(sorted(config)) or 'none'}"
            )
        entry = config[self.remote]
        token_field = entry.get("token")
        if not token_field:
            raise AuthError(f"rclone remote {self.remote!r} has no OAuth token")
        token = json.loads(token_field) if isinstance(token_field, str) else token_field
        if not token.get("access_token"):
            raise AuthError(f"rclone remote {self.remote!r} has no access_token")
        return token

    def _seconds_until_expiry(self, token: dict) -> float:
        expiry = token.get("expiry")
        if not expiry:
            return float("inf")  # non-expiring legacy token
        try:
            return (_parse_rfc3339(expiry) - dt.datetime.now(dt.timezone.utc)).total_seconds()
        except ValueError:
            return float("-inf")  # unparseable: treat as expired so we refresh

    def _current_token(self) -> dict:
        now = self._clock()
        if self._token is not None:
            ttl = self._seconds_until_expiry(self._token)
            if ttl > self.refresh_margin:
                return self._token
            if ttl > 0 and now - self._last_refresh_attempt < self.refresh_cooldown:
                # Inside our margin but still valid, and we asked rclone recently.
                #
                # This cooldown is load-bearing. rclone applies its *own* staleness
                # rule (about ten seconds), so for the whole window between our
                # margin and actual expiry it hands back the same token unchanged.
                # Without the cooldown every API request would spawn three rclone
                # subprocesses -- thousands of them, several hours into a run.
                return self._token

        token = self._read_token()
        if self._seconds_until_expiry(token) <= self.refresh_margin:
            self._last_refresh_attempt = now
            self._refresh_fn()
            token = self._read_token()
            if self._seconds_until_expiry(token) <= 0:
                raise AuthError(
                    f"rclone remote {self.remote!r} has an expired token that rclone "
                    f"would not refresh; try `rclone about {self.remote}:` by hand"
                )
        self._token = token
        return token

    # ---- public API ----------------------------------------------------

    def access_token(self) -> str:
        with self._lock:
            return self._current_token()["access_token"]

    def account(self) -> dict:
        with self._lock:
            if self._account is None:
                self._account = self._account_fn(self._current_token()["access_token"])
            return self._account

    def root_namespace_id(self) -> str:
        root_info = self.account().get("root_info") or {}
        namespace_id = root_info.get("root_namespace_id")
        if not namespace_id:
            raise AuthError("account has no root_namespace_id; is this a Dropbox team account?")
        return str(namespace_id)

    def headers(self) -> dict[str, str]:
        """Auth plus the path-root header that anchors paths at the team space root.

        Without ``Dropbox-API-Path-Root`` the API resolves paths against the member's
        home folder, where ``/TeamSpace`` does not exist. This mirrors what rclone does.
        """
        token = self.access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Dropbox-API-Path-Root": json.dumps(
                {".tag": "root", "root": self.root_namespace_id()}
            ),
        }
