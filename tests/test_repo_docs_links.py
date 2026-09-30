"""Docs hygiene: relative Markdown links and repo-path mentions must resolve.

Scans README, CONTRIBUTING, CLAUDE, PLANNING, ``docs/**/*.md`` and the PR template for

* inline Markdown links ``[text](relative/path.md#anchor)``: the file must exist and, for Markdown
  targets, so must the heading the anchor points at;
* backticked repo paths such as ``src/backtest/`` or ``docs/architecture.md`` in the prose docs.

Some paths are documented before the track that owns them has merged its code. Those, and only
those, are listed in ``PLANNED_PATHS`` with the reason; every entry must still be mentioned by a doc
(so the list cannot rot) and it is fine for an entry to exist on disk already.

One pattern is tolerated instead of listed: a link to a numbered ADR (``docs/adr/NNNN-*.md``) that does
not exist yet. ADRs 0002 to 0005 are written by different tracks and merge independently, so they
legitimately cross-reference each other before the sibling lands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# path (relative to the repo root, forward slashes) -> why it may be missing right now
PLANNED_PATHS = {
    "src/ingest/": "ingest track: src/ingest/nba_stats.py is being written (git does not track empty dirs)",
    "src/features/": "feature layers are not started; directory appears with the first feature builder",
    "src/models/": "model track: projectors are being written",
    "src/value/board.py": "model+value track: `python -m src.value.board` is being written",
    "src/backtest/": "backtest track: `python -m src.backtest` is being written",
    "src/app/": "live app is a later roadmap phase",
    "docs/research/": "source-research track: research notes land with the data-sources ADR",
    "tests/ingest/": "ingest track (named in ADR 0001): tests land with the ingest code",
    "tests/backtest/": "backtest track (named in ADR 0001): tests land with the harness",
}
_PLANNED_NORMALIZED = {p.rstrip("/") for p in PLANNED_PATHS}


def _is_planned(path: str) -> bool:
    return path.rstrip("/") in _PLANNED_NORMALIZED

DOC_FILES = sorted(
    {ROOT / n for n in ("README.md", "CONTRIBUTING.md", "CLAUDE.md", "PLANNING.md")}
    | set((ROOT / "docs").rglob("*.md"))
    | {ROOT / ".github" / "pull_request_template.md"}
)
PROSE_DOCS = [p for p in DOC_FILES if p.name != "PLANNING.md" and "research" not in p.parts]

FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"(?<!\\)!?\[(?:[^\]\n]|\\\])*\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
SIBLING_ADR = re.compile(r"^docs/adr/\d{4}-[\w.-]+\.md$")
REPO_PATH = re.compile(r"^(?:src|docs|config|tests|\.github)/[A-Za-z0-9_./-]+$")


def _strip_fences(text: str) -> str:
    return FENCE.sub("", text)


def _slug(heading: str) -> str:
    """GitHub's heading-anchor algorithm (close enough for our headings)."""
    heading = re.sub(r"`", "", heading.strip().lower())
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def _anchors(md_path: Path) -> set[str]:
    text = _strip_fences(md_path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    out: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = _slug(m.group(1))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def _links(md_path: Path) -> list[str]:
    text = INLINE_CODE.sub("", _strip_fences(md_path.read_text(encoding="utf-8")))
    return [m.group(1).strip("<>") for m in LINK.finditer(text)]


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def test_the_docs_under_test_exist():
    names = {_rel(p) for p in DOC_FILES}
    for required in ("README.md", "CONTRIBUTING.md", "CLAUDE.md", "PLANNING.md",
                     "docs/architecture.md", "docs/adr/README.md", "docs/adr/0001-data-contract.md"):
        assert required in names, f"{required} is missing"


@pytest.mark.parametrize("doc", DOC_FILES, ids=_rel)
def test_relative_markdown_links_resolve(doc: Path):
    problems = []
    for target in _links(doc):
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):  # http:, https:, mailto:, ...
            continue
        path_part, _, fragment = target.partition("#")
        dest = doc if not path_part else (doc.parent / path_part).resolve()
        rel = None
        try:
            rel = dest.relative_to(ROOT).as_posix()
        except ValueError:
            problems.append(f"{target}: points outside the repository")
            continue
        if not dest.exists():
            if _is_planned(rel) or SIBLING_ADR.match(rel):
                continue
            problems.append(f"{target}: {rel} does not exist")
            continue
        if fragment and dest.suffix == ".md" and fragment.lower() not in _anchors(dest):
            problems.append(f"{target}: no heading '#{fragment}' in {rel}")
    assert not problems, f"{_rel(doc)} has broken links:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("doc", PROSE_DOCS, ids=_rel)
def test_backticked_repo_paths_exist(doc: Path):
    text = _strip_fences(doc.read_text(encoding="utf-8"))
    problems = []
    for span in CODE_SPAN.findall(text):
        if not REPO_PATH.match(span) or any(ch in span for ch in "*<>{}"):
            continue
        if (ROOT / span).exists() or _is_planned(span):
            continue
        problems.append(span)
    assert not problems, f"{_rel(doc)} mentions paths that do not exist: {sorted(set(problems))}"


def test_planned_paths_allow_list_cannot_rot():
    mentioned = set()
    for doc in PROSE_DOCS + [ROOT / "PLANNING.md"]:
        text = doc.read_text(encoding="utf-8")
        mentioned |= {p for p in PLANNED_PATHS if p.rstrip("/") in text}
    stale = sorted(set(PLANNED_PATHS) - mentioned)
    assert not stale, f"PLANNED_PATHS entries no doc mentions any more (remove them): {stale}"
    for path, reason in PLANNED_PATHS.items():
        assert reason.strip(), f"{path} needs a reason"


def test_anchor_checker_understands_github_slugs(tmp_path: Path):
    md = tmp_path / "x.md"
    md.write_text("# Title\n\n## Data sources and legal notes\n\n## `./dev` env\n\n## Same\n\n## Same\n\n"
                  "```\n# not a heading\n```\n", encoding="utf-8")
    assert _anchors(md) == {"title", "data-sources-and-legal-notes", "dev-env", "same", "same-1"}


def test_link_extractor_ignores_code_and_handles_titles(tmp_path: Path):
    md = tmp_path / "x.md"
    md.write_text('[a](one.md) `[b](two.md)` ![i](img.png "title")\n```\n[c](three.md)\n```\n[d](<four.md#h>)\n',
                  encoding="utf-8")
    assert _links(md) == ["one.md", "img.png", "four.md#h"]


def test_readme_links_the_core_docs_and_names_the_real_entry_points():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = set(_links(ROOT / "README.md"))
    for required in ("docs/architecture.md", "docs/adr/README.md", "PLANNING.md", "CONTRIBUTING.md", "CLAUDE.md",
                     "config/league.yaml"):
        assert required in links, f"README.md must link {required}"
    for command in ("./dev bootstrap", "python -m src.value.board", "python -m src.backtest",
                    "./dev test"):
        assert command in readme, f"README quickstart must show `{command}`"
    for heading in ("## Quickstart (about 5 minutes, no data needed)", "## Use your real data", "## Common commands",
                    "## Where to read next", "## License"):
        assert heading in readme, f"README is missing the section {heading!r}"
    assert "no data" in readme.lower() and "DATA.md" in readme, "README must say the repo ships no data"


def test_dev_commands_documented_in_claude_md_match_the_dev_script():
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `\./dev (\w+)", claude, re.M))
    dev = (ROOT / "dev").read_text(encoding="utf-8")
    dispatch = dev[dev.rindex('case "${1:-}" in'):]
    implemented = set(re.findall(r"^  (\w+)\)", dispatch, re.M)) - {"help"}
    assert documented == implemented, f"CLAUDE.md table {sorted(documented)} vs dev script {sorted(implemented)}"


def test_claude_md_keeps_its_safety_rules_verbatim():
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for rule in (
        "Implementation agents MUST work in a dedicated task worktree.",
        "Do not modify tracked files in the primary integration checkout.",
        "If a worktree cannot be created, stop and report the reason rather than\nsilently falling back to the integration checkout.",
        "Never discard, reset, clean, overwrite, or revert changes you did not create.",
        "Avoid `git add -A` or equivalent broad staging in a dirty checkout.",
        "Only one worker may update the integration branch at a time.",
        "never resolve a conflict by discarding another task's change;",
        "never weaken tests or contracts to make branches combine;",
        "If integration exposes an actual semantic conflict, stop rather than guess.",
        "Never commit `.env`, cookies, or licensed/raw data (see `.gitignore`).",
    ):
        assert rule in claude, f"CLAUDE.md lost or reworded a rule: {rule!r}"
