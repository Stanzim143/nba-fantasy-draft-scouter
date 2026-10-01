"""Integration tests for the ``./dev`` tooling script.

Every test drives the real ``dev`` script (a copy of it) inside a throwaway git repository
under ``tmp_path``. The suite command is replaced through ``DEV_TEST_CMD`` so nothing here
recurses into the project's own test suite. The whole module skips cleanly when bash or git
is unavailable.

These tests run inside ``./dev integrate`` (twice, under the integration lock), so they are
deliberately consolidated: each test covers a family of behaviours with as few process
spawns as possible, and the primary repository is copied from a template built once.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

DEV_SOURCE = Path(__file__).resolve().parents[1] / "dev"
TIMEOUT = 180


def _find_bash() -> str | None:
    candidates: list[str] = []
    if sys.platform == "win32":
        for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            root = os.environ.get(var)
            if root:
                candidates.append(str(Path(root) / "Git" / "bin" / "bash.exe"))
                candidates.append(str(Path(root) / "Programs" / "Git" / "bin" / "bash.exe"))
    found = shutil.which("bash")
    if found and "system32" not in found.lower():  # C:\Windows\System32\bash.exe is WSL, not Git Bash
        candidates.append(found)
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


BASH = _find_bash()
if BASH is None or shutil.which("git") is None or not DEV_SOURCE.exists():
    pytest.skip("bash, git or the dev script is unavailable", allow_module_level=True)


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DEV_", "NBA_", "GIT_"))}
    git_config = {
        "core.autocrlf": "false",
        "commit.gpgsign": "false",
        "gc.auto": "0",
        "advice.detachedHead": "false",
    }
    env.update(
        GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.com",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT=str(len(git_config)),
    )
    for i, (key, value) in enumerate(git_config.items()):
        env[f"GIT_CONFIG_KEY_{i}"] = key
        env[f"GIT_CONFIG_VALUE_{i}"] = value
    return env


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=TIMEOUT)


@pytest.fixture(scope="module")
def template_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A committed repo (dev script + a shared file), built once and copied per test."""
    root = tmp_path_factory.mktemp("template")
    repo = root / "primary repo"  # path contains a space, like the real checkout
    empty_templates = root / "empty-git-templates"
    repo.mkdir()
    empty_templates.mkdir()
    env = _clean_env()
    res = _run(["git", "init", "-q", "-b", "main", f"--template={empty_templates.as_posix()}"], repo, env)
    assert res.returncode == 0, res.stderr
    (repo / "dev").write_bytes(DEV_SOURCE.read_bytes().replace(b"\r\n", b"\n"))
    (repo / "dev").chmod(0o755)  # match the +x index mode below, or POSIX git reports the checkout as dirty
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    (repo / "shared.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")
    for step in (["git", "add", "dev", "README.md", "shared.txt"],
                 ["git", "update-index", "--chmod=+x", "dev"],
                 ["git", "commit", "-q", "-m", "initial"]):
        res = _run(step, repo, env)
        assert res.returncode == 0, res.stderr
    return repo


class Sandbox:
    """A copy of the template as the primary checkout, plus a worktree root beside it."""

    def __init__(self, root: Path, template: Path):
        self.root = root
        self.primary = root / "primary repo"
        self.wt_root = root / "worktrees"
        shutil.copytree(template, self.primary)
        self.env = _clean_env()
        self.env.update(
            NBA_WORKTREE_ROOT=str(self.wt_root),  # native (backslash) form on Windows on purpose
            NBA_VENV=str(root / "no-such-venv"),
            DEV_TEST_CMD="true",
            DEV_LOCK_SLEEP="0.1",
            DEV_LOCK_TRIES="600",
            DEV_RUFF="none",
        )

    # -- helpers ---------------------------------------------------------
    def git(self, cwd: Path, *args: str) -> str:
        res = _run(["git", *args], cwd, self.env)
        assert res.returncode == 0, f"git {' '.join(args)} failed in {cwd}: {res.stderr}"
        return res.stdout.strip()

    def _env(self, extra: dict[str, str] | None) -> dict[str, str]:
        return {**self.env, **(extra or {})}

    def dev(self, cwd: Path, *args: str, env: dict[str, str] | None = None, script: str = "dev"):
        return _run([BASH, script, *args], cwd, self._env(env))

    def popen_dev(self, cwd: Path, *args: str, env: dict[str, str] | None = None):
        return subprocess.Popen([BASH, "dev", *args], cwd=cwd, env=self._env(env), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    def create(self, task: str) -> Path:
        res = self.dev(self.primary, "worktree", "create", task)
        assert res.returncode == 0, res.stderr
        path = self.wt_root / task
        assert path.is_dir()
        return path

    def commit_file(self, cwd: Path, name: str, content: str) -> str:
        (cwd / name).write_text(content, encoding="utf-8")
        self.git(cwd, "add", name)
        self.git(cwd, "commit", "-q", "-m", f"add {name}")
        return self.git(cwd, "rev-parse", "HEAD")

    def head(self, cwd: Path, ref: str = "HEAD") -> str:
        return self.git(cwd, "rev-parse", ref)

    @property
    def lock(self) -> Path:
        return self.primary / ".git" / "integration.lock"

    def git_dir(self, cwd: Path) -> Path:
        return self.primary / ".git" if cwd == self.primary else self.primary / ".git" / "worktrees" / cwd.name

    def assert_no_rebase_state(self, cwd: Path) -> None:
        for name in ("rebase-merge", "rebase-apply"):
            assert not (self.git_dir(cwd) / name).exists(), f"leftover {name} in {cwd}"

    def assert_clean(self, cwd: Path) -> None:
        assert self.git(cwd, "status", "--porcelain") == ""

    def assert_integrate_refused(self, cwd: Path, message: str) -> None:
        before = self.head(self.primary, "main")
        res = self.dev(cwd, "integrate")
        assert res.returncode != 0, res.stdout
        assert message in res.stderr, res.stderr
        assert self.head(self.primary, "main") == before, "a refused integrate must not move main"
        assert not self.lock.exists(), "lock leaked after a refused integrate"


@pytest.fixture
def sb(tmp_path: Path, template_repo: Path) -> Sandbox:
    return Sandbox(tmp_path, template_repo)


# --- basics ------------------------------------------------------------------

def test_usage_and_error_paths_outside_a_repo(sb: Sandbox, tmp_path: Path):
    res = sb.dev(sb.primary, "help")
    assert res.returncode == 0 and "worktree create" in res.stdout and "integrate" in res.stdout
    assert "lint" in res.stdout and "ci" in res.stdout
    res = sb.dev(sb.primary)
    assert res.returncode == 1 and "usage" in res.stderr
    res = sb.dev(sb.primary, "frobnicate")
    assert res.returncode != 0 and "unknown command: frobnicate" in res.stderr

    plain = tmp_path / "plain"  # not a git repository
    plain.mkdir()
    shutil.copy(sb.primary / "dev", plain / "dev")
    for args in (("worktree", "list"), ("integrate",), ("worktree", "create", "x")):
        res = sb.dev(plain, *args)
        assert res.returncode != 0 and "not inside a git repository" in res.stderr, args


def test_dev_test_passes_arguments_through(sb: Sandbox):
    res = sb.dev(sb.primary, "test", "a b", "c", env={"DEV_TEST_CMD": 'printf "%s|" "$@" > args.txt'})
    assert res.returncode == 0, res.stderr
    assert (sb.primary / "args.txt").read_text() == "a b|c|"
    res = sb.dev(sb.primary, "test", env={"DEV_TEST_CMD": "exit 7"})
    assert res.returncode == 7  # the suite's exit status is `dev test`'s exit status


# --- worktree create / list / remove -----------------------------------------------

def test_worktree_create_list_remove_and_duplicates(sb: Sandbox):
    res = sb.dev(sb.primary, "worktree", "create", "feature-1")
    assert res.returncode == 0, res.stderr
    path = sb.wt_root / "feature-1"
    assert path.is_dir()
    assert res.stdout.strip().startswith("cd ") and "feature-1" in res.stdout  # eval-able cd line only
    assert sb.git(sb.primary, "rev-parse", "--verify", "task/feature-1") == sb.head(sb.primary)
    assert sb.git(path, "symbolic-ref", "--short", "HEAD") == "task/feature-1"

    listing = sb.dev(sb.primary, "worktree", "list")
    assert listing.returncode == 0 and "task/feature-1" in listing.stdout and "[main]" in listing.stdout

    # a new task branches from the integration head even when created from inside another task
    sb.commit_file(path, "x.txt", "x\n")  # unintegrated commit on task/feature-1
    res = sb.dev(path, "worktree", "create", "second")
    assert res.returncode == 0, res.stderr
    assert sb.head(sb.wt_root / "second") == sb.head(sb.primary)  # main, not task/feature-1

    # duplicate branch, duplicate path
    res = sb.dev(sb.primary, "worktree", "create", "feature-1")
    assert res.returncode != 0 and "already exists" in res.stderr
    (sb.wt_root / "taken").mkdir()
    res = sb.dev(sb.primary, "worktree", "create", "taken")
    assert res.returncode != 0 and "already exists" in res.stderr
    assert sb.git(sb.primary, "branch", "--list", "task/taken") == ""  # nothing half-created

    # remove: the untouched worktree goes, the one with an unintegrated commit stays
    res = sb.dev(sb.primary, "worktree", "remove", "second")
    assert res.returncode == 0, res.stderr
    assert not (sb.wt_root / "second").exists()
    assert sb.git(sb.primary, "branch", "--list", "task/second") == ""
    res = sb.dev(sb.primary, "worktree", "remove", "feature-1")
    assert res.returncode != 0 and "not fully integrated" in res.stderr
    assert path.is_dir() and sb.git(sb.primary, "branch", "--list", "task/feature-1")


def test_create_validation(sb: Sandbox):
    bad_names = ["bad name", "a/b", "../x", ".", "..", ".hidden", "-x", "a..b", "x.lock", "semi;colon", "$(id)"]
    for name in bad_names:
        res = sb.dev(sb.primary, "worktree", "create", name)
        assert res.returncode != 0, (name, res.stdout)
        assert res.stderr.startswith("dev:"), (name, res.stderr)
    assert sb.git(sb.primary, "branch", "--list", "task/*") == ""
    assert not sb.wt_root.exists() or not any(sb.wt_root.iterdir())

    res = sb.dev(sb.primary, "worktree", "create")
    assert res.returncode != 0 and "usage" in res.stderr

    sb.git(sb.primary, "checkout", "-q", "--detach")
    res = sb.dev(sb.primary, "worktree", "create", "t")
    assert res.returncode != 0 and "detached HEAD" in res.stderr


def test_remove_refusals(sb: Sandbox):
    res = sb.dev(sb.primary, "worktree", "remove", "ghost")
    assert res.returncode != 0 and "no worktree" in res.stderr
    res = sb.dev(sb.primary, "worktree", "remove", "../primary repo")
    assert res.returncode != 0 and res.stderr.startswith("dev:")
    res = sb.dev(sb.primary, "worktree", "remove")
    assert res.returncode != 0 and "usage" in res.stderr

    path = sb.create("keep")
    res = sb.dev(path, "worktree", "remove", "keep")  # your shell is inside it
    assert res.returncode != 0 and "cd elsewhere" in res.stderr

    (path / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")  # untracked
    res = sb.dev(sb.primary, "worktree", "remove", "keep")
    assert res.returncode != 0 and "uncommitted changes" in res.stderr
    (path / "scratch.txt").unlink()
    (path / "README.md").write_text("modified\n", encoding="utf-8")  # tracked modification
    res = sb.dev(sb.primary, "worktree", "remove", "keep")
    assert res.returncode != 0 and "uncommitted changes" in res.stderr
    assert path.is_dir() and (path / "README.md").read_text() == "modified\n"


# --- integrate ---------------------------------------------------------------------

def test_integrate_fast_forwards_a_clean_task_and_it_can_then_be_removed(sb: Sandbox):
    path = sb.create("ff")
    commit = sb.commit_file(path, "feature.txt", "feature\n")
    res = sb.dev(path, "integrate")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "integrated task/ff into main" in res.stdout
    assert sb.head(sb.primary, "main") == commit  # fast-forward: same commit id, no merge commit
    assert (sb.primary / "feature.txt").read_text() == "feature\n"
    sb.assert_clean(sb.primary)
    assert not sb.lock.exists()
    res = sb.dev(sb.primary, "worktree", "remove", "ff")
    assert res.returncode == 0, res.stderr


def test_integrate_rebases_onto_a_moved_main(sb: Sandbox):
    path = sb.create("rebase")
    task_commit = sb.commit_file(path, "mine.txt", "mine\n")
    other = sb.commit_file(sb.primary, "theirs.txt", "theirs\n")  # main moves on independently
    res = sb.dev(path, "integrate")
    assert res.returncode == 0, res.stdout + res.stderr
    main = sb.head(sb.primary, "main")
    assert main != task_commit  # the task commit was rewritten on top of the new base
    assert sb.head(sb.primary, "main~1") == other
    assert sb.git(sb.primary, "rev-list", "--merges", "main") == ""  # linear history
    assert (sb.primary / "mine.txt").exists() and (sb.primary / "theirs.txt").exists()
    assert sb.head(path) == main
    sb.assert_clean(sb.primary)
    assert not sb.lock.exists()


def test_integrate_aborts_cleanly_on_a_rebase_conflict(sb: Sandbox):
    path = sb.create("conflict")
    task_commit = sb.commit_file(path, "shared.txt", "line one\nTASK EDIT\nline three\n")
    main_commit = sb.commit_file(sb.primary, "shared.txt", "line one\nMAIN EDIT\nline three\n")
    res = sb.dev(path, "integrate")
    assert res.returncode != 0
    assert "rebase conflict" in res.stderr and "Aborted the rebase cleanly" in res.stderr
    sb.assert_no_rebase_state(path)
    sb.assert_no_rebase_state(sb.primary)
    sb.assert_clean(path)
    sb.assert_clean(sb.primary)
    assert sb.head(path) == task_commit                # task branch untouched
    assert sb.head(sb.primary, "main") == main_commit  # integration branch untouched
    assert sb.git(path, "symbolic-ref", "--short", "HEAD") == "task/conflict"  # not left detached
    assert not sb.lock.exists()


def test_integrate_refusals_never_touch_anyone_elses_work(sb: Sandbox):
    # run from the primary checkout / the integration branch
    sb.assert_integrate_refused(sb.primary, "not the primary checkout")

    path = sb.create("refuse")
    sb.commit_file(path, "a.txt", "a\n")

    # dirty task tree (checked before the lock is even taken)
    (path / "stray.txt").write_text("oops\n", encoding="utf-8")
    sb.assert_integrate_refused(path, "uncommitted changes")
    assert (path / "stray.txt").exists()  # never cleaned for the user
    (path / "stray.txt").unlink()

    # dirty primary: untracked file, then a tracked modification (checked while holding the lock)
    (sb.primary / "someone-elses.txt").write_text("wip\n", encoding="utf-8")
    sb.assert_integrate_refused(path, "integration worktree")
    (sb.primary / "someone-elses.txt").unlink()
    (sb.primary / "README.md").write_text("half-edited\n", encoding="utf-8")
    sb.assert_integrate_refused(path, "dirty")
    assert (sb.primary / "README.md").read_text() == "half-edited\n"  # left exactly as found
    sb.git(sb.primary, "checkout", "--", "README.md")

    # a rebase in progress in the primary checkout
    (sb.primary / ".git" / "rebase-merge").mkdir()
    sb.assert_integrate_refused(path, "rebase in progress")
    (sb.primary / ".git" / "rebase-merge").rmdir()

    # detached HEAD in the task worktree
    sb.git(path, "checkout", "-q", "--detach")
    sb.assert_integrate_refused(path, "detached HEAD")
    sb.git(path, "checkout", "-q", "task/refuse")

    # and after all those refusals the very same task still integrates
    assert sb.dev(path, "integrate").returncode == 0
    assert (sb.primary / "a.txt").exists()


# --- the lock ------------------------------------------------------------------------

def test_lock_is_released_after_success_and_after_every_kind_of_failure(sb: Sandbox):
    """Regression: the lock once leaked because the EXIT trap read a function-local variable."""
    # while held, the lock names its owner (branch, pid, UTC time); released after success
    ok = sb.create("lock-ok")
    sb.commit_file(ok, "ok.txt", "ok\n")
    seen = sb.root / "seen.txt"
    owner_file = (sb.lock / "owner").as_posix()
    res = sb.dev(ok, "integrate", env={"DEV_TEST_CMD": f'cat "{owner_file}" >> "{seen.as_posix()}"'})
    assert res.returncode == 0, res.stderr
    assert "task/lock-ok pid=" in seen.read_text() and "Z" in seen.read_text()
    assert not sb.lock.exists(), "leaked after success"

    # verification of the task branch fails
    bad = sb.create("lock-tests")
    sb.commit_file(bad, "bad.txt", "bad\n")
    res = sb.dev(bad, "integrate", env={"DEV_TEST_CMD": "false"})
    assert res.returncode != 0 and "tests failed on updated task branch" in res.stderr
    assert not sb.lock.exists(), "leaked after test failure"
    assert not (sb.primary / "bad.txt").exists()  # nothing integrated

    # rebase conflict
    conflict = sb.create("lock-conflict")
    sb.commit_file(conflict, "shared.txt", "line one\nTASK\nline three\n")
    sb.commit_file(sb.primary, "shared.txt", "line one\nMAIN\nline three\n")
    assert sb.dev(conflict, "integrate").returncode != 0
    assert not sb.lock.exists(), "leaked after rebase conflict"

    # the combined-state check fails (it only ever runs inside the primary checkout)
    combined = sb.create("lock-combined")
    sb.commit_file(combined, "combined.txt", "c\n")
    res = sb.dev(combined, "integrate", env={"DEV_TEST_CMD": 'test "$(basename "$PWD")" != "primary repo"'})
    assert res.returncode != 0 and "tests failed on combined state" in res.stderr
    assert not sb.lock.exists(), "leaked after combined-state failure"


@pytest.mark.parametrize("sig", ["TERM", "INT", "HUP"])
def test_lock_is_released_when_integrate_is_signalled_mid_verification(sb: Sandbox, sig: str):
    """A signal while the tests run (the long phase) must stop the tests, release the lock, integrate nothing.

    The signal is sent from a second bash to the pid recorded in the lock's owner file (an MSYS pid, which
    ``Popen.terminate`` on Windows could not deliver as a catchable signal)."""
    path = sb.create(f"sig-{sig.lower()}")
    sb.commit_file(path, "sig.txt", "s\n")
    before = sb.head(sb.primary, "main")
    started = sb.root / f"started-{sig}.txt"
    proc = sb.popen_dev(path, "integrate", env={"DEV_TEST_CMD": f'echo up > "{started.as_posix()}"; sleep 60'})
    watchdog = threading.Timer(120, proc.kill)
    watchdog.start()
    try:
        deadline = time.time() + 60
        while not started.exists() and time.time() < deadline:
            time.sleep(0.2)
        assert started.exists(), "the verification step never started"
        pid = re.search(r"pid=(\d+)", (sb.lock / "owner").read_text()).group(1)
        assert proc.poll() is None
        kill = _run([BASH, "-c", f"kill -{sig} {pid}"], sb.root, sb.env)
        assert kill.returncode == 0, kill.stderr
        proc.communicate(timeout=30)   # returns promptly: it does not sit out the 60 s test
    finally:
        watchdog.cancel()
    assert proc.returncode != 0
    assert not sb.lock.exists(), f"lock leaked after SIG{sig}"
    assert sb.head(sb.primary, "main") == before, "an interrupted integrate must not move main"
    sb.assert_no_rebase_state(path)


def test_a_signalled_integrate_never_removes_someone_elses_lock(sb: Sandbox):
    """The EXIT trap only removes a lock whose owner file is this run's own tag."""
    path = sb.create("sig-foreign")
    sb.commit_file(path, "f.txt", "f\n")
    lock = sb.lock.as_posix()
    # the "test" swaps our lock for a foreign one (as if ours had been removed and another integrate took
    # it), then signals its own integrate (the pid is read from our owner file before the swap)
    swap = (f"pid=$(sed 's/.*pid=//; s/ .*//' '{lock}/owner'); rm -rf '{lock}'; mkdir '{lock}'; "
            f"echo 'task/other pid=1 now' > '{lock}/owner'; kill -TERM $pid; sleep 60")
    res = sb.dev(path, "integrate", env={"DEV_TEST_CMD": swap})
    assert res.returncode != 0
    assert sb.lock.is_dir() and (sb.lock / "owner").read_text().startswith("task/other"), "foreign lock was removed"


def test_stale_or_foreign_lock_is_reported_and_never_deleted(sb: Sandbox):
    path = sb.create("stale")
    commit = sb.commit_file(path, "s.txt", "s\n")
    fast = {"DEV_LOCK_TRIES": "3", "DEV_LOCK_SLEEP": "0.05"}

    sb.lock.mkdir()
    (sb.lock / "owner").write_text("task/ghost pid=2147483 2020-01-01T00:00:00Z\n", encoding="utf-8")
    res = sb.dev(path, "integrate", env=fast)
    assert res.returncode != 0
    assert "could not get" in res.stderr and "task/ghost" in res.stderr
    assert "stale" in res.stderr and "never removed automatically" in res.stderr
    assert sb.lock.is_dir(), "a foreign lock must never be deleted"
    assert (sb.lock / "owner").read_text().startswith("task/ghost")  # nor overwritten
    assert sb.head(sb.primary) != commit  # nothing integrated
    sb.assert_no_rebase_state(path)

    (sb.lock / "owner").unlink()  # a lock with no owner file is just as untouchable
    res = sb.dev(path, "integrate", env=fast)
    assert res.returncode != 0 and "could not get" in res.stderr
    assert sb.lock.is_dir()

    shutil.rmtree(sb.lock)  # the operator removes it by hand after verifying; integrate then works
    assert sb.dev(path, "integrate").returncode == 0
    assert sb.head(sb.primary) == commit


def test_a_waiting_integrate_proceeds_when_the_holder_releases(sb: Sandbox):
    path = sb.create("waiter")
    commit = sb.commit_file(path, "w.txt", "w\n")
    sb.lock.mkdir()
    (sb.lock / "owner").write_text("task/holder pid=1 now\n", encoding="utf-8")
    proc = sb.popen_dev(path, "integrate")
    watchdog = threading.Timer(120, proc.kill)  # never hang the suite if the message never comes
    watchdog.start()
    try:
        # deterministic: block until integrate itself says it is waiting, then release the lock
        first_line = proc.stderr.readline()
        assert "waiting for integration lock" in first_line, first_line
        assert "task/holder" in first_line
        assert proc.poll() is None, "integrate must keep waiting for a held lock"
        time.sleep(0.5)
        assert proc.poll() is None
        shutil.rmtree(sb.lock)  # the holder finishes
        out, err = proc.communicate(timeout=TIMEOUT)
    finally:
        watchdog.cancel()
    assert proc.returncode == 0, out + err
    assert sb.head(sb.primary) == commit
    assert not sb.lock.exists()


def test_concurrent_integrates_serialize_without_corruption(sb: Sandbox):
    a, b = sb.create("conc-a"), sb.create("conc-b")
    sb.commit_file(a, "a.txt", "a\n")
    sb.commit_file(b, "b.txt", "b\n")
    base = sb.head(sb.primary)
    log = sb.root / "verify.log"
    cmd = 'echo "start $$" >> "$VERIFY_LOG"; sleep 0.4; echo "end $$" >> "$VERIFY_LOG"'
    env = {"DEV_TEST_CMD": cmd, "VERIFY_LOG": log.as_posix()}
    pa, pb = sb.popen_dev(a, "integrate", env=env), sb.popen_dev(b, "integrate", env=env)
    out_a, err_a = pa.communicate(timeout=TIMEOUT)
    out_b, err_b = pb.communicate(timeout=TIMEOUT)
    assert pa.returncode == 0, out_a + err_a
    assert pb.returncode == 0, out_b + err_b

    # every verification run is bracketed start/end by the same process: no two ever overlapped
    lines = [line.split() for line in log.read_text().splitlines()]
    assert len(lines) == 8  # 2 integrates x (task branch + combined state) x (start, end)
    for i in range(0, len(lines), 2):
        assert lines[i][0] == "start" and lines[i + 1][0] == "end" and lines[i][1] == lines[i + 1][1], lines

    assert sb.git(sb.primary, "rev-list", "--count", f"{base}..main") == "2"
    assert sb.git(sb.primary, "rev-list", "--merges", "main") == ""
    assert (sb.primary / "a.txt").exists() and (sb.primary / "b.txt").exists()
    sb.assert_clean(sb.primary)
    sb.assert_no_rebase_state(a)
    sb.assert_no_rebase_state(b)
    assert not sb.lock.exists()
    sb.git(sb.primary, "fsck", "--no-progress")  # object store intact (asserts exit status 0)
    for task in ("conc-a", "conc-b"):
        assert sb.dev(sb.primary, "worktree", "remove", task).returncode == 0


# --- lint / ci -------------------------------------------------------------------------

def _fake_ruff(root: Path, exit_code: int = 0) -> tuple[Path, Path]:
    log = root / "ruff-args.txt"
    script = root / f"fake-ruff-{exit_code}"
    script.write_bytes(
        f'#!/usr/bin/env bash\necho "$PWD|$*" >> "{log.as_posix()}"\nexit {exit_code}\n'.encode()
    )
    script.chmod(0o755)
    return script, log


def test_lint_uses_ruff_from_the_repo_root_and_propagates_failure(sb: Sandbox):
    (sb.primary / "sub").mkdir()
    ok, log = _fake_ruff(sb.root)
    res = sb.dev(sb.primary / "sub", "lint", env={"DEV_RUFF": ok.as_posix()},
                 script=(sb.primary / "dev").as_posix())
    assert res.returncode == 0, res.stderr
    cwd, args = log.read_text().strip().split("|")
    assert args == "check ." and Path(cwd).name == "primary repo"  # ran from the toplevel

    failing, _ = _fake_ruff(sb.root, exit_code=3)
    res = sb.dev(sb.primary, "lint", env={"DEV_RUFF": failing.as_posix()})
    assert res.returncode == 3


def test_lint_falls_back_to_a_syntax_check_when_ruff_is_unavailable(sb: Sandbox):
    env = {"DEV_RUFF": "none", "NBA_VENV": sys.prefix}
    res = sb.dev(sb.primary, "lint", env=env)
    assert res.returncode == 0, res.stderr
    assert "ruff not found" in res.stderr and "nothing is installed automatically" in res.stderr

    sb.commit_file(sb.primary, "good.py", "x = 1\n")
    res = sb.dev(sb.primary, "lint", env=env)
    assert res.returncode == 0 and "syntax check: 1 file(s), 0 error(s)" in res.stdout

    sb.commit_file(sb.primary, "broken.py", "def f(:\n    pass\n")
    res = sb.dev(sb.primary, "lint", env=env)
    assert res.returncode == 1 and "broken.py:1: syntax error" in res.stderr

    res = sb.dev(sb.primary, "lint", env={**env, "DEV_REQUIRE_LINT": "1"})  # strict mode never falls back
    assert res.returncode != 0 and "DEV_REQUIRE_LINT=1" in res.stderr


def test_ci_runs_tests_and_lint_and_reports_both(sb: Sandbox):
    ok, log = _fake_ruff(sb.root)
    env = {"DEV_RUFF": ok.as_posix()}
    res = sb.dev(sb.primary, "ci", env=env)
    assert res.returncode == 0, res.stderr
    assert ">> ci ok" in res.stdout

    log.unlink()
    res = sb.dev(sb.primary, "ci", env={**env, "DEV_TEST_CMD": "false"})
    assert res.returncode != 0 and "ci failed" in res.stderr
    assert log.exists(), "lint must still run when tests fail, so one run reports everything"

    failing, _ = _fake_ruff(sb.root, exit_code=1)
    res = sb.dev(sb.primary, "ci", env={"DEV_RUFF": failing.as_posix()})
    assert res.returncode != 0 and "ci failed" in res.stderr


# --- the shipped script itself -------------------------------------------------------------

def test_shipped_script_is_lf_only_executable_in_git_and_valid_bash():
    data = DEV_SOURCE.read_bytes()
    assert b"\r" not in data, "dev must have LF endings or bash will choke (see .gitattributes)"
    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -euo pipefail" in data
    res = subprocess.run(["git", "ls-files", "-s", "dev"], cwd=DEV_SOURCE.parent, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():  # skipped outside a git checkout (e.g. sdist)
        assert res.stdout.startswith("100755"), "dev lost its executable bit in git"
    res = subprocess.run([BASH, "-n", DEV_SOURCE.as_posix()], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_setup_passes_paths_safely_and_warns_loudly_when_pinned_install_fails(sb: Sandbox, tmp_path: Path):
    """cmd_setup must not splice the checkout path into python code (a quote in the path broke it), must hand
    pip a native path, and must never silently fall back to unpinned installs."""
    repo = tmp_path / "it's a repo"  # a single quote used to break the python -c string
    shutil.copytree(sb.primary, repo)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["alpha>=1"]\n[project.optional-dependencies]\ndev = ["beta"]\n',
        encoding="utf-8")
    (repo / "constraints.txt").write_text("alpha==1.0\n", encoding="utf-8")
    fake = repo / ".venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    log = tmp_path / "pip.log"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  echo "$*" >> "$PIP_LOG"\n'
        '  case "$*" in *" -c "*) exit 1;; esac\n'
        '  exit 0\n'
        'fi\n'
        'exec "$REAL_PY" "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {"PIP_LOG": str(log), "REAL_PY": sys.executable}
    res = sb.dev(repo, "setup", env=env)
    assert res.returncode == 0, res.stderr
    assert (repo / ".venv" / "requirements.txt").read_text(encoding="utf-8").split() == ["alpha>=1", "beta"]
    assert "WARNING" in res.stderr and "UNPINNED" in res.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert any("constraints.txt" in c for c in calls) and any("constraints.txt" not in c and "-r" in c for c in calls)
