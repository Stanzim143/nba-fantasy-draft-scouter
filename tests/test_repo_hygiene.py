"""Repo hygiene gate (DATA.md): no secrets or bulk data in tracked files, and the ignore rules cover them.

Runs in CI through the normal suite. Skips when git or a git checkout is unavailable."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
# A real ESPN SWID is a braced GUID; a real espn_s2 is a long opaque token. Empty / placeholder values
# (".env.example", docs) never match.
SECRET = re.compile(r"(?i)(espn_s2|swid)[\"']?\s*[=:]\s*[\"']?(\{[0-9A-F]{8}-[0-9A-F-]{27}\}|[A-Za-z0-9%+/]{60,})")


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


@pytest.fixture(scope="module")
def tracked() -> list[Path]:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        out = _git("ls-files", "-z")
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("not a git checkout")
    return [ROOT / f.decode() for f in out.split(b"\0") if f]


def test_no_tracked_file_is_over_1mb(tracked):
    big = [f"{p.relative_to(ROOT)} ({p.stat().st_size} bytes)" for p in tracked
           if p.is_file() and p.stat().st_size > MAX_BYTES]
    assert not big, f"tracked files over 1 MB (data must never be committed): {big}"


def test_no_tracked_file_contains_espn_cookie_values(tracked):
    hits = []
    for p in tracked:
        if not p.is_file() or p.stat().st_size > MAX_BYTES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET.search(text):
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"possible ESPN cookie value (espn_s2 / SWID) in tracked files: {hits}"


def test_secret_pattern_catches_real_shapes_and_ignores_placeholders():
    assert SECRET.search("espn_s2=" + "A" * 80)
    assert SECRET.search("SWID={12345678-ABCD-ABCD-ABCD-1234567890AB}")
    assert not SECRET.search("ESPN_S2=\nESPN_SWID=\n")
    assert not SECRET.search("espn_s2=...")


def test_gitignore_blocks_secrets_and_data_but_not_the_env_example():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    probes = [".env", ".env.local", "data/processed/x.parquet", ".ci-data/x", "a/b.parquet", "a.csv",
              "a.pdf", "a.duckdb", "a.zip", "nightly.json"]
    for name in probes:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", name], cwd=ROOT)
        assert r.returncode == 0, f"{name} is not git-ignored"
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", ".env.example"], cwd=ROOT)
    assert r.returncode == 1, ".env.example must stay trackable"
