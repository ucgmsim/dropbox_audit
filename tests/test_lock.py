import pytest

from dbaudit.lock import InstanceLock, LockHeldError


def test_second_instance_is_refused(tmp_path):
    with (
        InstanceLock(tmp_path / "l"),
        pytest.raises(LockHeldError),
        InstanceLock(tmp_path / "l"),
    ):
        pass


def test_lock_released_on_exit(tmp_path):
    with InstanceLock(tmp_path / "l"):
        pass
    with InstanceLock(tmp_path / "l"):
        pass


def test_lock_names_the_holder(tmp_path):
    import os

    with (
        InstanceLock(tmp_path / "l"),
        pytest.raises(LockHeldError, match=str(os.getpid())),
        InstanceLock(tmp_path / "l"),
    ):
        pass


def test_lock_released_when_body_raises(tmp_path):
    with pytest.raises(ValueError), InstanceLock(tmp_path / "l"):
        raise ValueError("boom")
    with InstanceLock(tmp_path / "l"):
        pass
