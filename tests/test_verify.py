import json

from dbaudit.store import Store
from dbaudit.verify import verify_subtree

FILE = {
    ".tag": "file", "id": "id:1", "name": "a.bin", "size": 10,
    "path_display": "/R/a.bin", "path_lower": "/r/a.bin",
    "content_hash": "ab" * 32, "rev": "r",
    "client_modified": "2020-01-01T00:00:00Z", "server_modified": "2020-01-01T00:00:00Z",
}
NESTED = dict(FILE, id="id:2", name="c.bin", size=7, path_display="/R/sub/c.bin")


def seed(tmp_path, entries):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    sid = store.add_shard("/R", 0, "recursive")
    store.claim_shard("w")
    store.commit_page(sid, entries, cursor=None, has_more=False)
    return store


def test_clean_match(tmp_path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(store, "/R",
                            rclone_fn=lambda p: json.dumps([{"Path": "a.bin", "Size": 10}]))
    assert result.ok and result.differences == 0
    assert result.db_files == result.rclone_files == 1
    assert result.db_bytes == result.rclone_bytes == 10


def test_missing_file_detected(tmp_path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: json.dumps(
        [{"Path": "a.bin", "Size": 10}, {"Path": "b.bin", "Size": 5}]))
    assert result.only_in_rclone == ["b.bin"]
    assert not result.ok


def test_extra_file_detected(tmp_path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: json.dumps([]))
    assert result.only_in_db == ["a.bin"]


def test_size_mismatch_detected(tmp_path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(store, "/R",
                            rclone_fn=lambda p: json.dumps([{"Path": "a.bin", "Size": 99}]))
    assert result.size_mismatches == [("a.bin", 10, 99)]


def test_nested_paths_are_compared_relative_to_the_subtree(tmp_path):
    store = seed(tmp_path, [FILE, NESTED])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: json.dumps(
        [{"Path": "a.bin", "Size": 10}, {"Path": "sub/c.bin", "Size": 7}]))
    assert result.ok, (result.only_in_db, result.only_in_rclone)


def test_verifying_a_subdirectory_only(tmp_path):
    store = seed(tmp_path, [FILE, NESTED])
    result = verify_subtree(store, "/R/sub",
                            rclone_fn=lambda p: json.dumps([{"Path": "c.bin", "Size": 7}]))
    assert result.ok and result.db_files == 1


def test_empty_rclone_output_is_handled(tmp_path):
    store = seed(tmp_path, [])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: "")
    assert result.ok and result.db_files == 0
