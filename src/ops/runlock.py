"""Single-instance lock for the daily refresh: a JSON file created atomically (``O_EXCL``), stale-lock aware.

A lock is *stale* when its owner process is gone, or when it is older than ``stale_after`` seconds (a hung run or a
recycled pid). A stale lock is never silently deleted: it is renamed to ``<lock>.stale-<timestamp>`` (evidence
preserved, listed by ``schedule status``) and the caller is told. A live, fresh lock makes acquisition fail with
:class:`LockBusy` so a second run exits instead of racing the first.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class LockBusy(RuntimeError):
    def __init__(self, message: str, owner: dict | None = None):
        super().__init__(message)
        self.owner = owner or {}


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists. Never signals it (``os.kill(pid, 0)`` would terminate it on Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                   # access denied means it exists
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                              # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class RunLock:
    path: Path
    stale_after: float = 3 * 3600
    pid: int | None = None
    alive: Callable[[int], bool] = pid_alive
    clock: Callable[[], float] = time.time
    held: bool = False
    recovered_stale: Path | None = None

    def _owner_text(self) -> str:
        return json.dumps({"pid": self.pid or os.getpid(), "host": socket.gethostname(), "acquired_at": self.clock(),
                           "acquired_iso": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()})

    def _read_owner(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _try_create(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(self._owner_text())
        return True

    def acquire(self) -> "RunLock":
        if self._try_create():
            self.held = True
            return self
        owner = self._read_owner()
        age = None
        reason = None
        if owner is None:
            reason = "unreadable lock file"
        else:
            age = self.clock() - float(owner.get("acquired_at", 0))
            same_host = owner.get("host") in (None, socket.gethostname())
            if same_host and not self.alive(int(owner.get("pid", 0))):
                reason = f"owner pid {owner.get('pid')} is not running"
            elif age > self.stale_after:
                reason = f"held for {age / 3600:.1f} h (limit {self.stale_after / 3600:.1f} h)"
        if reason is None:
            raise LockBusy(f"another daily refresh is running (pid {owner.get('pid')}, "
                           f"{(age or 0) / 60:.0f} min ago); lock {self.path}", owner)
        moved = self.path.with_name(f"{self.path.name}.stale-{datetime.fromtimestamp(self.clock(), timezone.utc):%Y%m%dT%H%M%SZ}")
        try:
            os.replace(self.path, moved)                          # preserved, not deleted
        except OSError as exc:
            raise LockBusy(f"stale lock ({reason}) could not be moved aside: {exc}", owner) from exc
        self.recovered_stale = moved
        if not self._try_create():                                # lost a race with another instance
            raise LockBusy("another daily refresh took the lock while recovering a stale one")
        self.held = True
        return self

    def release(self) -> None:
        if not self.held:
            return
        owner = self._read_owner()
        if owner is not None and int(owner.get("pid", -1)) == (self.pid or os.getpid()):   # never remove someone else's
            self.path.unlink(missing_ok=True)
        self.held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
