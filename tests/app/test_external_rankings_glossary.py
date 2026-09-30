"""Glossary-coverage extension for the external rankings comparison page (ADR 0028), following the
same discipline as tests/app/test_glossary.py: every column the comparison/unmatched tables can show
must have a real glossary entry, and the app page's own st.dataframe calls must pass column_config."""
from __future__ import annotations

from pathlib import Path

from src.app import glossary as G
from src.value.rankings_compare import COMPARISON_COLUMNS, UNMATCHED_COLUMNS

PAGE = Path(__file__).resolve().parents[2] / "src" / "app" / "pages" / "external_rankings.py"


def test_every_comparison_column_has_a_glossary_entry():
    missing = set(COMPARISON_COLUMNS) - set(G.COLUMN_GLOSSARY)
    assert not missing, f"comparison columns with no glossary entry: {sorted(missing)}"


def test_every_unmatched_report_column_has_a_glossary_entry():
    missing = set(UNMATCHED_COLUMNS) - set(G.COLUMN_GLOSSARY)
    assert not missing, f"unmatched-report columns with no glossary entry: {sorted(missing)}"


def test_external_rankings_group_covers_exactly_its_own_new_columns():
    group = dict(next(cols for name, cols in G.GROUPS if name.startswith("External rankings")))
    expected = (set(COMPARISON_COLUMNS) | set(UNMATCHED_COLUMNS)) - {"player_id", "name"}
    assert set(group) == expected


def test_page_dataframe_calls_pass_column_config():
    source = PAGE.read_text(encoding="utf-8")
    calls = 0
    idx = 0
    while True:
        idx = source.find("st.dataframe(", idx)
        if idx == -1:
            break
        calls += 1
        open_paren = idx + len("st.dataframe(") - 1
        depth, i = 0, open_paren
        while True:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        stmt = source[idx:i + 1]
        assert "column_config=" in stmt, f"st.dataframe call with no column_config:\n{stmt}"
        idx = i + 1
    assert calls >= 3, f"expected at least 3 st.dataframe call sites, found {calls}"
