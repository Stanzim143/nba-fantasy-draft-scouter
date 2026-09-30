"""The advisory external-rankings disagreement table in the app's plain functions (ADR 0029):
joining the session-scoped comparison frame (``src.value.rankings_compare.build_comparison``) onto
the board, drafted-player exclusion, and graceful behavior with no comparison loaded. No Streamlit —
mirrors the discipline of ``tests/app/test_return_flag_app.py`` / the need-board tests."""
from __future__ import annotations

import pandas as pd

from src.app.state import RANKINGS_DISAGREEMENT_TABLE_COLUMNS, rankings_disagreement_table
from src.value.rankings_compare import build_comparison


def _board():
    return pd.DataFrame({
        "rank": [1, 2, 3], "player_id": [10, 11, 12], "name": ["A", "B", "C"],
        "position": ["PG", "C", "SF"], "tier": [1, 1, 2],
        "proj_fppg": [50.0, 45.0, 41.0], "proj_gp": [70.0, 60.0, 50.0],
        "proj_total_fp": [3500.0, 2700.0, 2050.0], "vorp": [900.0, 700.0, 400.0], "adp": [1.0, 5.0, 9.0],
        "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0,
    })


def _resolved_row(source_id, name, player_id, ext_rank):
    from src.ingest.external_rankings import RESOLVED_COLUMNS

    row = {
        "source": "yahoo", "source_id": source_id, "source_name_raw": name, "name_parsed": name,
        "team": "DEN", "positions": "C", "status_tag": None, "ext_rank": ext_rank,
        "adp": None, "ecr_vs_adp": None, "player_id": player_id, "match_method": "exact",
        "confidence": 1.0, "matched": True,
    }
    return pd.DataFrame([row], columns=RESOLVED_COLUMNS).astype({"player_id": "Int64"})


def _comparison_with_one_sharp_disagreement():
    board = _board().rename(columns={"rank": "rank"})[["player_id", "name", "rank"]]
    # player 10 (our rank 1): yahoo ranks him 100th -> |delta| = 99, well above the default threshold.
    yahoo = pd.concat([
        _resolved_row("1", "A", 10, 100),
        _resolved_row("2", "B", 11, 2),    # delta = -0... actually our_rank(2)-2=0, not flagged
        _resolved_row("3", "C", 12, 3),    # our_rank(3)-3=0, not flagged
    ], ignore_index=True)
    return build_comparison(board, {"yahoo": yahoo})


def test_empty_comparison_gives_an_empty_table_not_an_error():
    t = rankings_disagreement_table(_board(), pd.DataFrame(), drafted_ids=[])
    assert t.empty
    assert list(t.columns) == RANKINGS_DISAGREEMENT_TABLE_COLUMNS
    t2 = rankings_disagreement_table(_board(), None, drafted_ids=[])
    assert t2.empty


def test_no_flagged_players_gives_an_empty_table():
    board = _board()[["player_id", "name", "rank"]]
    yahoo = pd.concat([_resolved_row("1", "A", 10, 1)], ignore_index=True)
    cmp = build_comparison(board, {"yahoo": yahoo})
    t = rankings_disagreement_table(_board(), cmp, drafted_ids=[])
    assert t.empty


def test_flagged_player_is_joined_onto_the_board_with_expected_columns():
    cmp = _comparison_with_one_sharp_disagreement()
    t = rankings_disagreement_table(_board(), cmp, drafted_ids=[])
    assert t["name"].tolist() == ["A"]
    for c in ("yahoo_rank", "rank_delta_yahoo", "max_abs_rank_delta",
              "rankings_disagreement_direction", "rankings_disagreement_text",
              "proj_fppg", "proj_gp", "proj_total_fp", "vorp"):
        assert c in t.columns
    assert t.iloc[0]["yahoo_rank"] == 100
    # our_rank(1) - yahoo_rank(100) = -99: we rank him much earlier than Yahoo does -> we_favor.
    assert t.iloc[0]["rankings_disagreement_direction"] == "we_favor"


def test_drafted_players_are_excluded():
    cmp = _comparison_with_one_sharp_disagreement()
    t = rankings_disagreement_table(_board(), cmp, drafted_ids=[10])
    assert t.empty


def test_sorted_by_biggest_disagreement_first():
    board = _board()[["player_id", "name", "rank"]]
    yahoo = pd.concat([
        _resolved_row("1", "A", 10, 90),   # |delta| = 89
        _resolved_row("2", "C", 12, 150),  # our_rank 3 - 150 = |147|
    ], ignore_index=True)
    cmp = build_comparison(board, {"yahoo": yahoo})
    t = rankings_disagreement_table(_board(), cmp, drafted_ids=[])
    assert t["name"].tolist() == ["C", "A"]
