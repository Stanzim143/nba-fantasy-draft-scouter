"""Single-instance lock: exclusive, stale-aware, never silently deleting."""
import json
import os

import pytest

from src.ops.runlock import LockBusy, RunLock, pid_alive


def test_second_acquire_is_refused_while_owner_alive(tmp_path):
    a = RunLock(tmp_path / "lock.json").acquire()
    with pytest.raises(LockBusy):
        RunLock(tmp_path / "lock.json").acquire()
    a.release()
    assert not (tmp_path / "lock.json").exists()
    RunLock(tmp_path / "lock.json").acquire().release()


def test_dead_owner_lock_is_moved_aside_not_deleted(tmp_path):
    p = tmp_path / "lock.json"
    RunLock(p, pid=999999, alive=lambda pid: True).acquire()      # a "foreign" lock
    lock = RunLock(p, alive=lambda pid: False)
    lock.acquire()
    assert lock.recovered_stale is not None and lock.recovered_stale.exists()
    assert json.loads(lock.recovered_stale.read_text())["pid"] == 999999
    lock.release()


def test_old_lock_is_stale_even_if_pid_alive(tmp_path):
    p = tmp_path / "lock.json"
    RunLock(p, pid=4242, clock=lambda: 1000.0).acquire()
    lock = RunLock(p, stale_after=60, alive=lambda pid: True, clock=lambda: 5000.0)
    lock.acquire()
    assert lock.recovered_stale is not None


def test_release_never_removes_someone_elses_lock(tmp_path):
    p = tmp_path / "lock.json"
    mine = RunLock(p).acquire()
    p.write_text(json.dumps({"pid": 1, "host": None, "acquired_at": 0}))
    mine.release()
    assert p.exists()


def test_pid_alive_for_self_and_bogus():
    assert pid_alive(os.getpid()) and not pid_alive(0)
