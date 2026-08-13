import pytest

from dbaudit.store import Store

FILE = {
    ".tag": "file", "id": "id:1", "name": "a.bin", "size": 10,
    "path_display": "/TeamSpace/x/a.bin", "path_lower": "/teamspace/x/a.bin",
    "content_hash": "ab" * 32, "rev": "r1",
    "client_modified": "2020-01-02T03:04:05Z", "server_modified": "2020-01-02T03:04:06Z",
    "sharing_info": {"modified_by": "dbid:AAA"},
}
FOLDER = {
    ".tag": "folder", "id": "id:d", "name": "x",
    "path_display": "/TeamSpace/x", "path_lower": "/teamspace/x",
}


def new_store(tmp_path):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/TeamSpace")
    return store


def test_commit_page_writes_rows_and_cursor(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace/x", depth=2, mode="recursive")
    store.claim_shard("w1")
    stats = store.commit_page(sid, [FOLDER, FILE], cursor="c1", has_more=True)
    assert stats.files == 1 and stats.dirs == 1 and stats.bytes == 10
    assert store.get_shard(sid).cursor == "c1"
    assert store.stats()["files"] == 1


def test_page_commit_is_atomic(tmp_path):
    """A failure mid-page must leave neither rows nor an advanced cursor."""
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace/x", depth=2, mode="recursive")
    store.claim_shard("w1")
    bad = dict(FILE, id="id:2", size="not-an-int")
    with pytest.raises(ValueError):
        store.commit_page(sid, [FILE, bad], cursor="c1", has_more=True)
    assert store.stats()["files"] == 0
    assert store.stats()["dirs"] == 0
    assert store.get_shard(sid).cursor is None


def test_replaying_a_page_does_not_duplicate(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace/x", depth=2, mode="recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FOLDER, FILE], cursor="c1", has_more=True)
    after_first = store.stats()
    store.commit_page(sid, [FOLDER, FILE], cursor="c1", has_more=True)
    after_second = store.stats()
    assert after_first["files"] == after_second["files"] == 1
    # /TeamSpace and /TeamSpace/x -- ancestors count, but replaying adds nothing.
    assert after_first["dirs"] == after_second["dirs"] == 2


def test_dirs_are_normalised_with_ancestors(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace", depth=1, mode="recursive")
    store.claim_shard("w1")
    deep = dict(FILE, path_display="/TeamSpace/a/b/c/f.txt", name="f.txt", id="id:2")
    store.commit_page(sid, [deep], cursor="c", has_more=False)
    rows = store.query("SELECT path_display FROM dirs ORDER BY depth")
    assert [r[0] for r in rows] == [
        "/TeamSpace", "/TeamSpace/a", "/TeamSpace/a/b", "/TeamSpace/a/b/c",
    ]
    joined = store.query(
        "SELECT dirs.path_display || '/' || files.name FROM files "
        "JOIN dirs ON dirs.id = files.dir_id"
    )
    assert joined[0][0] == "/TeamSpace/a/b/c/f.txt"


def test_folder_entry_upgrades_dir_created_from_child_path(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace", depth=1, mode="recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE], cursor="c1", has_more=True)     # implies /TeamSpace/x
    store.commit_page(sid, [FOLDER], cursor="c2", has_more=False)  # the real folder entry
    assert store.query("SELECT dbx_id FROM dirs WHERE path_lower='/teamspace/x'")[0][0] == "id:d"
    assert store.query("SELECT COUNT(*) FROM dirs WHERE path_lower='/teamspace/x'")[0][0] == 1


def test_mount_points_are_flagged(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace", depth=1, mode="recursive")
    store.claim_shard("w1")
    mount = dict(FOLDER, shared_folder_id="3161601617")
    store.commit_page(sid, [mount], cursor=None, has_more=False)
    assert store.query("SELECT is_mount FROM dirs WHERE path_lower='/teamspace/x'")[0][0] == 1


def test_claim_shard_is_exclusive(tmp_path):
    store = new_store(tmp_path)
    store.add_shard("/a", 1, "recursive")
    assert store.claim_shard("w1") is not None
    assert store.claim_shard("w2") is None


def test_claim_shard_prefers_shallow_shards(tmp_path):
    store = new_store(tmp_path)
    store.add_shard("/a/b/c", 3, "recursive")
    store.add_shard("/a", 1, "recursive")
    assert store.claim_shard("w1").path == "/a"


def test_reset_stale_shards_preserves_cursor(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE], cursor="c9", has_more=True)
    assert store.reset_stale_shards() == 1
    shard = store.get_shard(sid)
    assert shard.state == "pending" and shard.cursor == "c9"


def test_last_page_marks_shard_done(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE], cursor=None, has_more=False)
    assert store.get_shard(sid).state == "done"


def test_clear_cursor_removes_rows_from_that_shard_only(tmp_path):
    store = new_store(tmp_path)
    keep = store.add_shard("/keep", 1, "recursive")
    drop = store.add_shard("/drop", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(keep, [dict(FILE, id="id:keep")], cursor="c", has_more=True)
    store.commit_page(drop, [dict(FILE, id="id:drop", path_display="/TeamSpace/y/b.bin")],
                      cursor="c", has_more=True)
    store.clear_cursor(drop)
    assert store.stats()["files"] == 1
    assert store.get_shard(drop).cursor is None
    assert store.get_shard(keep).cursor == "c"


def test_content_hash_stored_as_blob(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE], cursor=None, has_more=False)
    value = store.query("SELECT content_hash FROM files")[0][0]
    assert isinstance(value, bytes) and len(value) == 32


def test_timestamps_stored_as_epoch_integers(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE], cursor=None, has_more=False)
    cmod, smod = store.query("SELECT client_modified, server_modified FROM files")[0]
    assert cmod == 1577934245 and smod == 1577934246


def test_modified_by_is_normalised_into_principals(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FILE, dict(FILE, id="id:9", name="b.bin")], cursor=None, has_more=False)
    assert store.query("SELECT COUNT(*) FROM principals")[0][0] == 1
    assert store.query(
        "SELECT dbx_account_id FROM principals JOIN files ON files.modified_by = principals.id "
        "LIMIT 1"
    )[0][0] == "dbid:AAA"


def test_entry_without_path_display_is_kept_not_dropped(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace/x", depth=2, mode="recursive")
    store.claim_shard("w1")
    orphan = {k: v for k, v in FILE.items() if k != "path_display"}
    store.commit_page(sid, [orphan], cursor=None, has_more=False, shard_path="/TeamSpace/x")
    assert store.stats()["files"] == 1
    assert store.query("SELECT COUNT(*) FROM api_events WHERE kind='no_path_display'")[0][0] == 1


def test_child_shards_are_enqueued_in_the_same_transaction(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/TeamSpace", depth=1, mode="split")
    store.claim_shard("w1")
    store.commit_page(sid, [FOLDER], cursor=None, has_more=False,
                      child_shards=[("/TeamSpace/x", 2, "recursive")])
    assert store.query("SELECT path FROM shards WHERE state='pending'")[0][0] == "/TeamSpace/x"


def test_progress_counters_track_committed_pages(tmp_path):
    store = new_store(tmp_path)
    sid = store.add_shard("/a", 1, "recursive")
    store.claim_shard("w1")
    store.commit_page(sid, [FOLDER, FILE], cursor=None, has_more=False)
    prog = store.progress()
    assert prog["files"] == 1 and prog["dirs"] == 1 and prog["bytes"] == 10 and prog["pages"] == 1


def test_build_indexes_is_idempotent(tmp_path):
    store = new_store(tmp_path)
    store.build_indexes()
    store.build_indexes()
