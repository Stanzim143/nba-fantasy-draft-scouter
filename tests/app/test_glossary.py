"""Coverage for the app's single source of truth for column tooltips (``src.app.glossary``) and
its wiring into ``draft_board.py``: every column ``build_board`` and its overlays can produce, and
every column actually shown in a table in the app, must have a glossary entry. This mirrors the
"every board column must be documented" discipline already established in
``docs/categories.md``'s own section 20."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.app import glossary as G
from src.app import inseason_view
from src.app.state import (
    BOARD_COLUMNS,
    LM_TABLE_COLUMNS,
    RANKINGS_DISAGREEMENT_TABLE_COLUMNS,
    RETURN_TABLE_COLUMNS,
    display_columns,
)
from src.app.txn_view import DISPLAY_COLUMNS as TXN_DISPLAY_COLUMNS
from src.app.coach_view import TABLE_COLUMNS as COACH_TABLE_COLUMNS
from src.value.board import BASE_COLUMNS, CARRIED
from src.value.breakouts import WATCH_COLUMNS
from src.value.contract_flags import FLAG_COLUMNS as CONTRACT_COLUMNS
from src.value.contract_flags import VALIDATION_COLUMN as CONTRACT_VALIDATION_COLUMN
from src.value.load_flag import LM_COLUMNS
from src.value.load_flag import VALIDATION_COLUMN as LM_VALIDATION_COLUMN
from src.value.need import NEED_TABLE_COLUMNS, POSITION_NEED_COLUMNS
from src.value.return_flag import RETURN_COLUMNS
from src.value.return_flag import VALIDATION_COLUMN as RETURN_VALIDATION_COLUMN
from src.value.risk import RISK_COLUMNS

PAGE = Path(__file__).resolve().parents[2] / "src" / "app" / "draft_board.py"


# --------------------------------------------------------------------------------------------
# 1. Every column a real board can carry has a glossary entry.
#
# Derived straight from each module's own exported column-name constant (the same constants the
# board-building/overlay code uses to select which columns to attach/keep), never re-typed by
# hand, so this fails automatically the moment a new column is added anywhere in that pipeline.
# --------------------------------------------------------------------------------------------

def _every_possible_board_column() -> set[str]:
    cols = set(BASE_COLUMNS) | set(CARRIED) | {"adp", "adp_gap"}
    cols |= set(RISK_COLUMNS)
    cols |= set(RETURN_COLUMNS) | {RETURN_VALIDATION_COLUMN}
    cols |= set(LM_COLUMNS) | {LM_VALIDATION_COLUMN}
    cols |= set(CONTRACT_COLUMNS) | {CONTRACT_VALIDATION_COLUMN}
    return cols


def test_every_board_and_overlay_column_has_a_glossary_entry():
    missing = _every_possible_board_column() - set(G.COLUMN_GLOSSARY)
    assert not missing, f"columns with no glossary entry: {sorted(missing)}"


def test_glossary_has_no_orphan_entries_for_removed_board_columns():
    """Not required by the task, but keeps the registry honest: every board/overlay column also
    documented here should still exist in the code (guards against a stale entry surviving a
    rename)."""
    board_only_groups = {"Identity and projection", "Value and VORP / ADP", "Risk overlay (ADR 0016)",
                         "Returned-healthy overlay (ADR 0023)", "Load-management overlay (ADR 0024)",
                         "Contract overlay (ADR 0019)"}
    known = _every_possible_board_column()
    for group_name, cols in G.GROUPS:
        if group_name not in board_only_groups:
            continue
        for col in cols:
            assert col in known, f"{col!r} in group {group_name!r} is not produced by build_board or any overlay"


@pytest.mark.parametrize("built_board_columns", [
    list(BOARD_COLUMNS), list(RETURN_TABLE_COLUMNS), list(LM_TABLE_COLUMNS),
    list(WATCH_COLUMNS), list(TXN_DISPLAY_COLUMNS), list(COACH_TABLE_COLUMNS),
    list(NEED_TABLE_COLUMNS), list(POSITION_NEED_COLUMNS),
    list(inseason_view.TRADE_DISPLAY_COLUMNS), list(RANKINGS_DISAGREEMENT_TABLE_COLUMNS),
])
def test_every_app_display_column_list_is_fully_covered(built_board_columns):
    missing = [c for c in built_board_columns if c not in G.COLUMN_GLOSSARY]
    assert not missing, f"shown but undocumented columns: {missing}"


def test_display_columns_output_is_fully_covered_for_boards_with_every_overlay():
    board = pd.DataFrame({c: [] for c in [
        "rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp",
        "vorp", "vorp_per_game", "fppg_p10", "fppg_p50", "fppg_p90", "tier",
        "adp", "adp_gap", "projection_class", "p_play", "risk_level", "risk_flags",
        "contract_flag", "return_flag", "return_tail", "lm_flag",
    ]})
    cols = display_columns(board)
    missing = [c for c in cols if c not in G.COLUMN_GLOSSARY]
    assert not missing, f"display_columns() produced undocumented columns: {missing}"
    # drafted_by is inserted by the Full board tab itself, not display_columns(); still must be documented.
    assert "drafted_by" in G.COLUMN_GLOSSARY


def test_extra_ad_hoc_app_columns_are_documented():
    """Columns shown by tabs that build their own explicit list rather than reusing a shared
    constant (Debutants & risk, Coaches' affected-players table, the Drafted tab)."""
    ad_hoc = {
        "rank", "name", "position", "age", "proj_fppg", "proj_gp", "proj_total_fp", "vorp",
        "p_play", "adp", "risk_flags", "risk_level", "risk_gp", "contract_years_left",
        "contract_status", "contract_basis", "team", "coach", "previous_coach",
        "first_time_head_coach", "pick_no", "player_id", "drafted_by",
    }
    missing = ad_hoc - set(G.COLUMN_GLOSSARY)
    assert not missing, f"columns with no glossary entry: {sorted(missing)}"


# --------------------------------------------------------------------------------------------
# 2. Registry sanity: every entry is well-formed (used by both the tooltip and the Glossary tab).
# --------------------------------------------------------------------------------------------

def test_every_entry_has_a_short_tooltip_a_long_explanation_and_a_source():
    for col, entry in G.COLUMN_GLOSSARY.items():
        assert entry.short and entry.short.strip(), f"{col} has no short tooltip"
        assert entry.long and entry.long.strip(), f"{col} has no long explanation"
        assert entry.source and entry.source.strip(), f"{col} has no source citation"
        # Hover tooltips should stay short: 1-2 sentences, not a paragraph.
        assert len(entry.short) <= 320, f"{col}'s short tooltip is too long for a hover ({len(entry.short)} chars)"


def test_tooltip_and_entry_helpers_agree_with_the_dict():
    for col, entry in G.COLUMN_GLOSSARY.items():
        assert G.tooltip(col) == entry.short
        assert G.entry(col) is entry
    assert G.tooltip("not_a_real_column") is None
    assert G.entry("not_a_real_column") is None


def test_groups_cover_every_registry_entry_exactly_once():
    seen = []
    for _group_name, cols in G.GROUPS:
        seen.extend(cols.keys())
    assert sorted(seen) == sorted(G.COLUMN_GLOSSARY)
    assert len(seen) == len(set(seen)), "a column appears in more than one group"


def test_return_tail_glossary_text_does_not_repeat_the_known_doc_gap_as_settled_fact():
    """categories.md section 17 item 4 flags the in-app checkbox help text as omitting the
    >=15-remaining-games condition. The glossary must state the actual (documented) code
    condition and flag the gap, not silently repeat the incomplete claim as if it were complete."""
    entry = G.COLUMN_GLOSSARY["return_tail"]
    assert "15" in entry.long
    assert "categories.md section 17" in entry.long or "17-code-versus-documentation" in entry.source


# --------------------------------------------------------------------------------------------
# 3. Wiring into the app: every st.dataframe/st.data_editor call site passes column_config with a
# help= tooltip for every column it shows. Checked directly against the source's column lists
# (import draft_board's own helper) rather than by rendering, since most tabs need a full board
# and external files (ledger, coach table) the test suite does not build.
# --------------------------------------------------------------------------------------------

def test_column_config_helper_returns_a_tooltip_for_every_glossary_column(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")  # confirms streamlit.testing is on the path

    from src.app.draft_board import _column_config

    cfg = _column_config(list(G.COLUMN_GLOSSARY))
    assert set(cfg) == set(G.COLUMN_GLOSSARY)
    for col, value in cfg.items():
        assert value.get("help") == G.tooltip(col)


def test_column_config_helper_skips_columns_with_no_entry_rather_than_crashing():
    from src.app.draft_board import _column_config

    cfg = _column_config(["rank", "not_a_real_column"])
    assert "rank" in cfg and "not_a_real_column" not in cfg


def test_every_st_dataframe_call_site_passes_column_config():
    """Source-level check: every ``st.dataframe(`` call in draft_board.py must be passed
    ``column_config=`` in the same statement (not added conditionally later), so no table is
    silently left without tooltips."""
    source = PAGE.read_text(encoding="utf-8")
    calls = 0
    idx = 0
    while True:
        idx = source.find("st.dataframe(", idx)
        if idx == -1:
            break
        calls += 1
        # Walk forward from the opening paren to its matching close, tracking depth, so a call
        # that wraps onto several lines is captured whole regardless of surrounding blank lines.
        open_paren = idx + len("st.dataframe(") - 1
        depth = 0
        i = open_paren
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
    assert calls == 17, f"expected 17 st.dataframe call sites in draft_board.py, found {calls}"


def test_trade_analyzer_tab_dataframe_call_sites_pass_column_config():
    """Same source-level discipline as ``test_every_st_dataframe_call_site_passes_column_config``,
    scoped to the Trade analyzer tab's own functions in ``src/app/pages/inseason.py`` (ADR 0027):
    every ``st.dataframe(`` call inside ``_trade_side_summary``/``_trade_tab`` must carry
    ``column_config=`` in the same statement. Scoped rather than whole-file because the page's
    pre-existing Schedule/Rest-of-season tabs predate this convention (ADR 0027's known gap)."""
    page = Path(__file__).resolve().parents[2] / "src" / "app" / "pages" / "inseason.py"
    source = page.read_text(encoding="utf-8")
    start = source.index("def _trade_side_summary")
    end = source.index("def _waivers_tab")
    assert start < end
    section = source[start:end]
    calls = 0
    idx = 0
    while True:
        idx = section.find("st.dataframe(", idx)
        if idx == -1:
            break
        calls += 1
        open_paren = idx + len("st.dataframe(") - 1
        depth, i = 0, open_paren
        while True:
            if section[i] == "(":
                depth += 1
            elif section[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        stmt = section[idx:i + 1]
        assert "column_config=" in stmt, f"st.dataframe call with no column_config:\n{stmt}"
        idx = i + 1
    assert calls == 2, "expected exactly the trade-summary and trade-lines dataframes"


def test_trade_display_columns_are_all_in_the_trade_glossary_group():
    """The Trade analyzer group in glossary.py must be exactly the columns the app can show for it
    (no leftover/renamed entry, no display column left uncovered by that specific group)."""
    trade_group = dict(next(cols for name, cols in G.GROUPS if name.startswith("Trade analyzer")))
    assert set(trade_group) == set(inseason_view.TRADE_DISPLAY_COLUMNS)


def test_best_available_tab_dataframe_help_reaches_the_rendered_column_config(monkeypatch):
    """End-to-end confirmation for one representative tab (AppTest, the pattern used in
    ``tests/app/test_board_cache_refresh.py``): the column_config actually reaches the rendered
    proto, not just the Python call, and every shown column carries a help string from the
    glossary."""
    import json

    at_mod = pytest.importorskip("streamlit.testing.v1")

    from src.app import loader

    def fake_load_board(*_args, **_kwargs):
        return pd.DataFrame({
            "rank": [1], "player_id": [1], "name": ["Player1"], "position": ["PG"], "tier": [1],
            "proj_fppg": [50.0], "proj_gp": [70.0], "proj_total_fp": [3500.0], "vorp": [900.0],
            "adp": [1.0], "fppg_p10": [30.0], "fppg_p50": [40.0], "fppg_p90": [50.0],
        })

    monkeypatch.setattr(loader, "load_board", fake_load_board)

    at = at_mod.AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    load_button = [b for b in at.sidebar.button if b.label.startswith("Load")][0]
    load_button.click()
    at.run()
    assert not at.exception, at.exception

    df_el = at.tabs[0].dataframe[0]
    shown_columns = list(df_el.value.columns)
    config = json.loads(df_el.proto.columns) if df_el.proto.columns else {}
    for col in shown_columns:
        help_text = (config.get(col) or {}).get("help")
        assert help_text, f"column {col!r} shown on Best available has no rendered help tooltip"
        assert help_text == G.tooltip(col)
