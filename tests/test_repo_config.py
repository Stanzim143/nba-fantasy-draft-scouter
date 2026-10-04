"""Repo configuration hygiene: pytest/ruff settings, CI workflow, dependabot, tracked-file rules."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


# --- pytest / ruff config ----------------------------------------------------------------

def test_pytest_markers_are_registered_and_network_is_excluded_by_default():
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    registered = {m.split(":")[0].strip() for m in ini["markers"]}
    assert {"slow", "network", "real_data"} <= registered
    assert ini["testpaths"] == ["tests"] and "." in ini["pythonpath"]
    assert "not network" in ini["addopts"]


def test_default_selection_skips_network_tests_and_opt_in_selects_them(tmp_path: Path):
    """Run pytest in a scratch project using the real ini options: proves the semantics, not the text."""
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    markers = "\n".join(f"    {m}" for m in ini["markers"])
    (tmp_path / "pytest.ini").write_text(
        f"[pytest]\naddopts = {ini['addopts']} --strict-markers\nmarkers =\n{markers}\n", encoding="utf-8")
    (tmp_path / "test_marks.py").write_text(
        "import pytest\n"
        "def test_plain(): pass\n"
        "@pytest.mark.network\ndef test_net(): pass\n"
        "@pytest.mark.slow\ndef test_slow(): pass\n"
        "@pytest.mark.real_data\ndef test_real(): pass\n", encoding="utf-8")

    def collected(*args: str) -> set[str]:
        res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
                             cwd=tmp_path, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
        return set(re.findall(r"::(test_\w+)", res.stdout))

    assert collected() == {"test_plain", "test_slow", "test_real"}
    assert collected("-m", "network") == {"test_net"}
    # a command-line -m REPLACES the default, so to skip slow tests and still skip network ones, say both
    assert collected("-m", "not slow") == {"test_plain", "test_net", "test_real"}
    assert collected("-m", "not slow and not network") == {"test_plain", "test_real"}


def test_ruff_config_is_sane():
    ruff = _pyproject()["tool"]["ruff"]
    assert ruff["line-length"] == 110
    assert ruff["target-version"] == "py312"
    lint = ruff["lint"]
    assert {"E", "F"} <= set(lint["select"])
    assert all(isinstance(r, str) for r in lint["select"] + lint["ignore"])
    for patterns in lint["per-file-ignores"].values():
        assert patterns and all(re.fullmatch(r"[A-Z]+\d+", code) for code in patterns)


def test_core_project_dependencies_are_declared():
    deps = _pyproject()["project"]["dependencies"]
    assert {"pandas", "pyarrow", "pyyaml", "nba_api"} <= set(deps)
    assert "pytest" in _pyproject()["project"]["optional-dependencies"]["dev"]


# --- CI workflow -------------------------------------------------------------------------

def _workflow() -> dict:
    wf = _yaml(".github/workflows/ci.yml")
    if True in wf and "on" not in wf:  # YAML 1.1 parses the bare key `on` as boolean True
        wf["on"] = wf.pop(True)
    return wf


def _steps_text(job: dict) -> str:
    return "\n".join(str(s.get("run", "")) + str(s.get("uses", "")) for s in job["steps"])


def test_ci_triggers_permissions_and_concurrency():
    wf = _workflow()
    assert "pull_request" in wf["on"] and "main" in wf["on"]["push"]["branches"]
    assert wf["permissions"] == {"contents": "read"}  # minimal token
    conc = wf["concurrency"]
    # only PR runs are superseded; a push to main is never cancelled
    assert "pull_request" in str(conc["cancel-in-progress"]) and "github.ref" in conc["group"]
    assert all(isinstance(job.get("timeout-minutes"), int) for job in wf["jobs"].values())
    for job in wf["jobs"].values():
        assert "permissions" not in job or job["permissions"] in ({"contents": "read"}, "read-all")


def test_ci_test_job_runs_the_full_suite_on_linux_and_windows_with_pip_cache():
    job = _workflow()["jobs"]["test"]
    matrix = job["strategy"]["matrix"]
    assert set(matrix["os"]) == {"ubuntu-latest", "windows-latest"}
    assert matrix["python-version"] == ["3.12"]
    assert job["strategy"]["fail-fast"] is False
    setup = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/setup-python@"))
    assert setup["with"]["cache"] == "pip"
    text = _steps_text(job)
    assert "pytest" in text
    # the whole suite (leakage tests included) runs: no path/keyword/marker filtering, and nothing opts into network
    run_pytest = next(s["run"] for s in job["steps"] if "pytest" in str(s.get("run", "")))
    assert "-k" not in run_pytest.split() and "tests/" not in run_pytest
    assert "network" not in run_pytest and "--deselect" not in run_pytest
    assert "NBA_DATA_DIR" in job["env"], "tests must not touch a shared data directory"


def test_ci_installs_dependencies_from_pyproject_only():
    text = _steps_text(_workflow()["jobs"]["test"])
    assert "pyproject.toml" in text and "pip install -r" in text
    assert "pip install ." not in text and "pip install -e" not in text  # flat layout: not installable


def test_ci_lint_job_requires_ruff():
    job = _workflow()["jobs"]["lint"]
    text = _steps_text(job)
    assert "requirements-lint.txt" in text and "./dev lint" in text
    lint_step = next(s for s in job["steps"] if "./dev lint" in str(s.get("run", "")))
    assert lint_step["env"]["DEV_REQUIRE_LINT"] == "1"
    pin = (ROOT / ".github" / "requirements-lint.txt").read_text(encoding="utf-8")
    assert re.search(r"^ruff==\d+\.\d+\.\d+$", pin, re.M), "ruff must be pinned exactly in CI"


def test_ci_actions_are_pinned_to_a_full_commit_sha_with_a_version_comment():
    for job in _workflow()["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses = [ln for ln in text.splitlines() if ln.strip().startswith(("- uses:", "uses:"))]
    assert uses and all(re.search(r"@[0-9a-f]{40} # v\d+", ln) for ln in uses), uses


def test_ci_yaml_files_are_valid_yaml():
    for path in list((ROOT / ".github").rglob("*.yml")) + list((ROOT / ".github").rglob("*.yaml")) + [
            ROOT / "config" / "league.yaml"]:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_dependabot_covers_pip_and_actions_weekly():
    cfg = _yaml(".github/dependabot.yml")
    assert cfg["version"] == 2
    ecosystems = {u["package-ecosystem"] for u in cfg["updates"]}
    assert {"pip", "github-actions"} <= ecosystems
    assert all(u["schedule"]["interval"] == "weekly" for u in cfg["updates"])
    for u in cfg["updates"]:
        assert (ROOT / u["directory"].lstrip("/")).is_dir(), u


def test_pull_request_template_is_present_and_covers_the_rules():
    text = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for phrase in ("./dev ci", "leakage", "secrets", "synthetic"):
        assert phrase in text


# --- editor / git config -----------------------------------------------------------------

def test_editorconfig_and_gitattributes():
    editorconfig = (ROOT / ".editorconfig").read_text(encoding="utf-8")
    assert re.search(r"^root\s*=\s*true", editorconfig, re.M)
    assert "end_of_line = lf" in editorconfig and "charset = utf-8" in editorconfig
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^dev\s+text\s+eol=lf", attrs, re.M)


def test_gitignore_protects_secrets_and_data():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for entry in (".env", "__pycache__/", ".pytest_cache/"):
        assert entry in ignored


def _tracked_files() -> list[str]:
    res = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if res.returncode != 0:
        pytest.skip("not a git checkout")
    return [f for f in res.stdout.decode().split("\0") if f]


def test_no_secrets_or_data_files_are_tracked():
    tracked = _tracked_files()
    forbidden = [f for f in tracked
                 if f == ".env" or (f.startswith(".env.") and f != ".env.example")
                 or f.endswith((".parquet", ".duckdb", ".pkl", ".sqlite", ".db", ".csv.gz"))
                 or f.startswith(("data/raw/", "data/processed/"))]
    assert not forbidden, f"tracked files that must never be committed: {forbidden}"
    for name in (".env.example",):
        text = (ROOT / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                if key.strip() in {"ESPN_S2", "ESPN_SWID"}:
                    assert value.strip() == "", f"{name} must not contain a real {key.strip()}"


def test_no_tracked_text_file_has_crlf_line_endings_in_scripts():
    for name in _tracked_files():
        if name == "dev" or name.endswith(".sh"):
            assert b"\r\n" not in (ROOT / name).read_bytes(), f"{name} must be LF"


def test_test_module_basenames_are_unique():
    """There are no __init__.py files under tests/, so duplicate basenames break collection."""
    names = [p.name for p in (ROOT / "tests").rglob("test_*.py")]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})


# --- public-repo privacy hygiene ---------------------------------------------------------------

# The maintainer's league id and name are kept only as SHA-256 digests so the literals never enter the repo.
_ID_DIGEST = "5255dcb454d04716e9f7681d6ccbdff5c14a98adf13d58402b6cb485ffc95384"
_NAME_DIGEST = "c2ff57add0c79c88fc22f54dd7fc655b8b21cca8d2114c184b34068f8d9da96b"


def _digest_hits(text):
    hits = []
    if any(hashlib.sha256(t.encode()).hexdigest() == _ID_DIGEST for t in re.findall(r"(?<!\d)\d{10}(?!\d)", text)):
        hits.append("the maintainer's league id")
    words = re.sub(r"\s+", " ", text.lower()).split(" ")
    if any(hashlib.sha256(" ".join(words[i:i + 4]).encode()).hexdigest() == _NAME_DIGEST for i in range(len(words))):
        hits.append("the maintainer's league name")
    return hits


_PRIVATE_PATTERNS = {
    "a personal Windows profile path": re.compile(r"[A-Za-z]:\\Users\\(?!user\b|you\b|x\b|<)\w+", re.I),
}


def test_no_tracked_text_file_leaks_private_identifiers():
    hits = []
    for name in _tracked_files():
        if name == "tests/test_repo_config.py":
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        hits += [f"{name}: {label}" for label, pat in _PRIVATE_PATTERNS.items() if pat.search(text)]
        hits += [f"{name}: {label}" for label in _digest_hits(text)]
    assert not hits, f"private identifiers tracked in a shareable repo: {hits}"


def test_public_release_files_exist():
    for name in ("LICENSE", "DATA.md", "README.md", ".env.example", "constraints.txt", "docs/deploy.md",
                 ".devcontainer/devcontainer.json"):
        assert (ROOT / name).is_file(), f"{name} is missing"
