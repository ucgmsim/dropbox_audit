"""Which tar archives an audit database holds, as `archive run` will walk them."""

import hashlib

from dbaudit.archive.candidates import find_candidates
from dbaudit.store import Store


def file(path, size, content=None):
    """A file as Dropbox lists it. Files with the same `content` are copies."""
    return {".tag": "file", "id": f"id:{path}", "name": path.rsplit("/", 1)[1],
            "path_display": path, "size": size, "rev": "r",
            "content_hash": hashlib.sha256((content or path).encode()).hexdigest(),
            "client_modified": "2026-01-01T00:00:00Z",
            "server_modified": "2026-01-01T00:00:00Z"}


def folder(path):
    return {".tag": "folder", "id": f"id:{path}", "name": path.rsplit("/", 1)[1],
            "path_display": path}


def listing(tmp_path, *entries):
    """An audit database holding ``entries`` under the root /R."""
    store = Store(tmp_path / "audit.db")
    store.init_schema()
    store.set_meta("root", "/R")
    shard = store.add_shard("/R", 1)
    store.claim_shard("test")
    store.commit_page(shard, [folder("/R"), *entries], cursor=None, has_more=False)
    store.close()
    return tmp_path / "audit.db"


def test_a_plain_tar_and_a_set_of_parts_are_each_one_candidate_largest_first(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/a/one.tar", 10),
                                    file("/R/b/two.tar.aa", 30), file("/R/b/two.tar.ab", 20)))
    assert [(c.folder, c.name, c.size, c.n_parts, c.status) for c in found.candidates] == [
        ("/R/b", "two.tar", 50, 2, "ok"), ("/R/a", "one.tar", 10, 1, "ok")]
    assert all(len(c.set_hash) == 64 for c in found.candidates)


def test_the_same_content_elsewhere_is_a_copy_walked_once(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/a/x.tar", 9, content="same"),
                                    file("/R/b/x.tar", 9, content="same")))
    assert [(c.folder, c.status, c.detail) for c in found.candidates] == [
        ("/R/a", "ok", ""), ("/R/b", "copy", "copy of /R/a/x.tar")]


def test_compressed_names_are_counted_not_listed(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/a/y.tar.gz", 7), file("/R/a/z.tgz", 5)))
    assert found.candidates == []
    assert found.compressed == (2, 12)


def test_a_set_register_would_refuse_is_listed_as_refused(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/a/g.tar.aa", 4),
                                    file("/R/a/g.tar.ac", 4)))
    [refused] = found.candidates
    assert (refused.name, refused.status, refused.set_hash) == ("g.tar", "refused", "")
    assert "missing part 'ab'" in refused.detail


def test_parts_each_in_a_folder_of_their_own_are_refused(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/h.tar.aa/h.tar.aa", 4)))
    [refused] = found.candidates
    assert (refused.status, refused.detail) == ("refused", "its parts are in separate folders")


def test_names_that_are_neither_a_tar_nor_a_part_are_counted(tmp_path):
    found = find_candidates(listing(tmp_path, file("/R/a/old.tar.1", 3)))
    assert found.candidates == []
    assert found.other == (1, 3)
