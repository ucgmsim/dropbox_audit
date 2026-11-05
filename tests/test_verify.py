import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from dbaudit.store import Store
from dbaudit.verify import _run_rclone, verify_subtree

FILE = {
    ".tag": "file",
    "id": "id:1",
    "name": "a.bin",
    "size": 10,
    "path_display": "/R/a.bin",
    "path_lower": "/r/a.bin",
    "content_hash": "ab" * 32,
    "rev": "r",
    "client_modified": "2020-01-01T00:00:00Z",
    "server_modified": "2020-01-01T00:00:00Z",
}
NESTED = dict(FILE, id="id:2", name="c.bin", size=7, path_display="/R/sub/c.bin")


def seed(tmp_path: Path, entries: list[dict]):
    store = Store(tmp_path / "a.db")
    store.init_schema()
    store.set_meta("root", "/R")
    sid = store.add_shard("/R", 0, "recursive")
    store.claim_shard("w")
    store.commit_page(sid, entries, cursor=None, has_more=False)
    return store


def test_clean_match(tmp_path: Path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(
        store, "/R", rclone_fn=lambda p: json.dumps([{"Path": "a.bin", "Size": 10}])
    )
    assert result.ok and result.differences == 0
    assert result.db_files == result.rclone_files == 1
    assert result.db_bytes == result.rclone_bytes == 10


def test_missing_file_detected(tmp_path: Path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(
        store,
        "/R",
        rclone_fn=lambda p: json.dumps(
            [{"Path": "a.bin", "Size": 10}, {"Path": "b.bin", "Size": 5}]
        ),
    )
    assert result.only_in_rclone == ["b.bin"]
    assert not result.ok


def test_extra_file_detected(tmp_path: Path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: json.dumps([]))
    assert result.only_in_db == ["a.bin"]


def test_size_mismatch_detected(tmp_path: Path):
    store = seed(tmp_path, [FILE])
    result = verify_subtree(
        store, "/R", rclone_fn=lambda p: json.dumps([{"Path": "a.bin", "Size": 99}])
    )
    assert result.size_mismatches == [("a.bin", 10, 99)]


def test_nested_paths_are_compared_relative_to_the_subtree(tmp_path: Path):
    store = seed(tmp_path, [FILE, NESTED])
    result = verify_subtree(
        store,
        "/R",
        rclone_fn=lambda p: json.dumps(
            [{"Path": "a.bin", "Size": 10}, {"Path": "sub/c.bin", "Size": 7}]
        ),
    )
    assert result.ok, (result.only_in_db, result.only_in_rclone)


def test_verifying_a_subdirectory_only(tmp_path: Path):
    store = seed(tmp_path, [FILE, NESTED])
    result = verify_subtree(
        store, "/R/sub", rclone_fn=lambda p: json.dumps([{"Path": "c.bin", "Size": 7}])
    )
    assert result.ok and result.db_files == 1


def test_empty_rclone_output_is_handled(tmp_path: Path):
    store = seed(tmp_path, [])
    result = verify_subtree(store, "/R", rclone_fn=lambda p: "")
    assert result.ok and result.db_files == 0


def _fake_run(
    commands: list[list[str]], returncode: int, stdout: str = "", stderr: str = ""
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A stand-in for ``subprocess.run`` that records its command and starts nothing."""

    def run(args: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(list(args))
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    return run


def test_rclone_is_asked_to_list_the_subtree_and_its_output_is_returned(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(commands, 0, stdout="[]"))
    assert _run_rclone("dropbox", "/R") == "[]"
    assert commands[0][:4] == ["rclone", "lsjson", "-R", "--files-only"]
    assert commands[0][-1] == "dropbox:/R"


def test_a_failing_rclone_listing_raises_with_rclones_complaint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        subprocess, "run", _fake_run([], 3, stderr=" directory not found \n")
    )
    with pytest.raises(
        RuntimeError, match="rclone lsjson failed: directory not found$"
    ):
        _run_rclone("dropbox", "/R")


def test_verifying_without_a_listing_function_uses_rclone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    commands: list[list[str]] = []
    listing = json.dumps([{"Path": "a.bin", "Size": 10}])
    monkeypatch.setattr(subprocess, "run", _fake_run(commands, 0, stdout=listing))
    result = verify_subtree(seed(tmp_path, [FILE]), "/R", remote="dropbox")
    assert result.ok
    assert commands[0][-1] == "dropbox:/R"
