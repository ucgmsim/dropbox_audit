"""A synthetic Dropbox that behaves like the real one for listing purposes.

Lets the crawler be tested against rate limits, cursor resets and mid-page death
without touching the network.

Cursors are encoded, not stored in a dict, so that a *new* FakeLister can resume a
cursor issued by an earlier one -- exactly as a restarted process resumes a cursor
issued to the process that died.
"""

from dbaudit.api import Page, PathNotFound


class FakeLister:
    """``tree`` maps a directory path to child names; a trailing '/' marks a directory."""

    def __init__(self, tree, page_size=2, faults=None, continue_faults=None):
        self.tree = tree
        self.page_size = page_size
        self.faults = dict(faults or {})            # path -> exception, for list_folder
        self.continue_faults = dict(continue_faults or {})  # nth call -> exception
        self.calls = []
        self._continue_calls = 0

    def _walk(self, root, recursive):
        out, queue = [], [root]
        if recursive:
            # The real API returns the folder itself as the first entry of a
            # recursive listing (verified against Dropbox 2026-08-13). A
            # non-recursive listing does not.
            out.append({
                ".tag": "folder", "id": f"id:{root}", "name": root.rsplit("/", 1)[-1],
                "path_display": root, "path_lower": root.lower(),
            })
        while queue:
            current = queue.pop(0)
            for child in self.tree.get(current, []):
                path = f"{current}/{child.rstrip('/')}"
                if child.endswith("/"):
                    out.append({
                        ".tag": "folder", "id": f"id:{path}", "name": child.rstrip("/"),
                        "path_display": path, "path_lower": path.lower(),
                    })
                    if recursive:
                        queue.append(path)
                else:
                    out.append({
                        ".tag": "file", "id": f"id:{path}", "name": child, "size": 1,
                        "path_display": path, "path_lower": path.lower(),
                        "content_hash": "cd" * 32, "rev": "r",
                        "client_modified": "2020-01-01T00:00:00Z",
                        "server_modified": "2020-01-01T00:00:00Z",
                    })
        return out

    def _page(self, root, recursive, offset):
        items = self._walk(root, recursive)
        chunk = items[offset:offset + self.page_size]
        nxt = min(offset + self.page_size, len(items))
        has_more = nxt < len(items)
        # The real API returns a cursor on every page, including the last one --
        # that final cursor is what an incremental pass resumes from.
        return Page(entries=chunk, cursor=f"{root}|{int(recursive)}|{nxt}", has_more=has_more)

    def list_folder(self, path, recursive):
        self.calls.append(("list", path, recursive))
        if path in self.faults:
            raise self.faults.pop(path)
        if path not in self.tree:
            raise PathNotFound(f"no such folder: {path}")
        return self._page(path, recursive, 0)

    def continue_(self, cursor):
        self.calls.append(("cont", cursor, None))
        self._continue_calls += 1
        if self._continue_calls in self.continue_faults:
            raise self.continue_faults.pop(self._continue_calls)
        root, recursive, offset = cursor.rsplit("|", 2)
        return self._page(root, bool(int(recursive)), int(offset))


TREE = {
    "/R": ["a/", "b/", "top.txt"],
    "/R/a": ["a1.txt", "a2.txt", "deep/"],
    "/R/a/deep": ["d1.txt"],
    "/R/b": ["b1.txt"],
}

ALL_FILES = {"/R/top.txt", "/R/a/a1.txt", "/R/a/a2.txt", "/R/a/deep/d1.txt", "/R/b/b1.txt"}
