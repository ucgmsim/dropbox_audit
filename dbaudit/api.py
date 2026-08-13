"""Thin, read-only client for Dropbox's ``files/list_folder`` endpoints.

Only two calls matter for an audit:

``files/list_folder(recursive=true)``
    Returns up to 2000 entries per request for an *entire subtree*, which is what
    makes a 250 TB audit take hours instead of months. rclone cannot do this --
    its Dropbox backend reports ``ListR: false`` and so pays one API call per
    directory.

``files/list_folder/continue``
    Resumes from a cursor. The cursor is what makes the crawl restartable, and it
    stays valid after the crawl finishes, so a later pass returns only changes.

Everything here classifies failures into the small set of outcomes the crawler
knows how to act on; no retrying or sleeping happens at this layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

API_BASE = "https://api.dropboxapi.com/2"
PAGE_LIMIT = 2000
DEFAULT_RETRY_AFTER = 60.0
REQUEST_TIMEOUT = 180


@dataclass(frozen=True)
class Page:
    entries: list[dict] = field(default_factory=list)
    cursor: str | None = None
    has_more: bool = False


class ApiError(RuntimeError):
    """An API failure the crawler cannot do anything clever about."""


class RateLimited(ApiError):
    """429. Honour ``retry_after`` exactly; observed values reach 300 seconds."""

    def __init__(self, message: str, retry_after: float = DEFAULT_RETRY_AFTER):
        super().__init__(message)
        self.retry_after = float(retry_after)


class CursorReset(ApiError):
    """The cursor is no longer usable; the subtree must be re-listed from scratch."""


class PathNotFound(ApiError):
    """The path vanished between seeding and listing. Not an error worth failing on."""


class TransientError(ApiError):
    """5xx or a network fault. Retry with backoff."""


class AuthExpired(ApiError):
    """The access token expired mid-crawl; refresh and retry."""


class Lister(Protocol):
    """The seam that keeps the crawler testable without a network."""

    def list_folder(self, path: str, recursive: bool) -> Page: ...

    def continue_(self, cursor: str) -> Page: ...

    def get_latest_cursor(self, path: str, recursive: bool, include_deleted: bool) -> str: ...


def _classify(status: int, body: dict, headers: dict) -> ApiError:
    if status == 429:
        retry_after = None
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            retry_after = error.get("retry_after")
        if retry_after is None:
            retry_after = headers.get("Retry-After")
        try:
            retry_after = float(retry_after)
        except (TypeError, ValueError):
            retry_after = DEFAULT_RETRY_AFTER
        return RateLimited(f"rate limited, retry after {retry_after}s", retry_after)

    if status == 401:
        return AuthExpired(f"401 from Dropbox: {str(body)[:200]}")

    if status == 409:
        error = body.get("error") if isinstance(body, dict) else {}
        tag = error.get(".tag") if isinstance(error, dict) else None
        if tag == "reset":
            return CursorReset("cursor invalidated by Dropbox; re-listing subtree")
        if tag == "path":
            path_tag = (error.get("path") or {}).get(".tag")
            if path_tag in ("not_found", "not_folder", "restricted_content"):
                return PathNotFound(f"path unusable: {path_tag}")
        return ApiError(f"409 from Dropbox: {str(body)[:300]}")

    if status >= 500:
        return TransientError(f"{status} from Dropbox: {str(body)[:200]}")

    return ApiError(f"{status} from Dropbox: {str(body)[:300]}")


class HttpLister:
    """Real :class:`Lister`. Reuses one HTTP session so TLS handshakes are amortised."""

    def __init__(self, token_provider, session=None, include_deleted: bool = False):
        self.tokens = token_provider
        self.include_deleted = include_deleted
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def _post(self, url: str, payload: dict) -> Page:
        try:
            resp = self.session.post(
                url,
                headers=self.tokens.headers(),
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:  # connection reset, DNS, read timeout
            raise TransientError(f"request failed: {type(exc).__name__}: {exc}") from exc

        if resp.status_code == 200:
            body = resp.json()
            return Page(
                entries=body.get("entries", []),
                cursor=body.get("cursor"),
                has_more=bool(body.get("has_more", False)),
            )

        try:
            body = resp.json()
        except Exception:
            body = {"raw": getattr(resp, "text", "")[:300]}
        raise _classify(resp.status_code, body, getattr(resp, "headers", {}) or {})

    def list_folder(self, path: str, recursive: bool) -> Page:
        return self._post(
            f"{API_BASE}/files/list_folder",
            {
                "path": path,
                "recursive": recursive,
                "limit": PAGE_LIMIT,
                "include_deleted": self.include_deleted,
                "include_media_info": False,
                "include_mounted_folders": True,
                "include_non_downloadable_files": True,
            },
        )

    def continue_(self, cursor: str) -> Page:
        return self._post(f"{API_BASE}/files/list_folder/continue", {"cursor": cursor})

    def get_latest_cursor(self, path: str, recursive: bool = True,
                          include_deleted: bool = True) -> str:
        """A cursor for "the tree as it is right now", without listing anything.

        Returns in well under a second even for a 250 TB tree, which is what makes a
        repeat audit cost one call per 2000 *changes* rather than one call per shard.
        """
        page = self._post(
            f"{API_BASE}/files/list_folder/get_latest_cursor",
            {
                "path": path,
                "recursive": recursive,
                "limit": PAGE_LIMIT,
                "include_deleted": include_deleted,
                "include_media_info": False,
                "include_mounted_folders": True,
                "include_non_downloadable_files": True,
            },
        )
        return page.cursor
