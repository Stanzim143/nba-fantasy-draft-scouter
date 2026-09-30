"""Positional/roster-need scoring (ADR 0026): pure functions, no Streamlit."""
from __future__ import annotations

import pandas as pd
import pytest

from src.value.need import (
    DEFAULT_NEED_BONUS,
    compute_need_board,
    is_categories_league,
    player_need_scores,
    position_capacity,
    position_fill_counts,
    position_need_scores,
    position_need_table,
)

# A small league config mirroring config/league.yaml's shape (points format, one starter per
# specific position, one G, one F, 3 UTIL, 3 bench) so capacity math is easy to hand-check.
LEAGUE_CFG = {
    "league": {
        "format": "h2h_points",
        "roster": {
            "size": 13,
            "starters": {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3},
            "bench": 3,
            "ir": 1,
        },
    }
}

CATEGORIES_CFG = {"league": {"format": "roto_categories", "roster": LEAGUE_CFG["league"]["roster"]}}


def _board(rows: list[dict]) -> pd.DataFrame:
    base = {"player_id": None, "name": "", "position": "PG", "vorp": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


# --------------------------------------------------------------------------------------------
# is_categories_league
# --------------------------------------------------------------------------------------------

def test_points_league_is_not_a_categories_league():
    assert is_categories_league(LEAGUE_CFG) is False


def test_roto_categories_config_is_detected():
    assert is_categories_league(CATEGORIES_CFG) is True
    assert is_categories_league({"league": {"format": "categories"}}) is True


def test_missing_format_defaults_to_not_categories():
    assert is_categories_league({"league": {}}) is False
    assert is_categories_league({}) is False


# --------------------------------------------------------------------------------------------
# position_capacity: hand-checkable against LEAGUE_CFG
# --------------------------------------------------------------------------------------------

def test_position_capacity_spreads_flex_util_and_bench():
    cap = position_capacity(LEAGUE_CFG)
    # PG: 1 (specific) + 1/2 (G flex) + 3/5 (UTIL) + 3/5 (bench) = 2.7
    assert cap["PG"] == pytest.approx(1 + 0.5 + 0.6 + 0.6)
    assert cap["SG"] == pytest.approx(cap["PG"])          # symmetric with PG under this config
    assert cap["SF"] == pytest.approx(cap["PF"])          # symmetric F flex
    # C gets no flex share, only UTIL and bench
    assert cap["C"] == pytest.approx(1 + 0.6 + 0.6)
    assert sum(cap.values()) > 0


def test_position_capacity_is_zero_when_a_position_has_no_slot_at_all():
    cfg = {"league": {"format": "h2h_points", "roster": {"starters": {"PG": 1}, "bench": 0}}}
    cap = position_capacity(cfg)
    assert cap["C"] == 0.0
    assert cap["PG"] == 1.0


# --------------------------------------------------------------------------------------------
# position_fill_counts / position_need_scores
# --------------------------------------------------------------------------------------------

def test_position_fill_counts_counts_each_eligible_position():
    board = _board([
        {"player_id": 1, "position": "PG"},
        {"player_id": 2, "position": "G"},       # eligible PG and SG
        {"player_id": 3, "position": "C"},
    ])
    counts = position_fill_counts(board, {1, 2, 3})
    assert counts["PG"] == 2
    assert counts["SG"] == 1
    assert counts["C"] == 1
    assert counts["SF"] == 0


def test_position_fill_counts_empty_roster_is_all_zero():
    board = _board([{"player_id": 1, "position": "PG"}])
    counts = position_fill_counts(board, set())
    assert counts == {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0}


def test_need_score_open_position_between_zero_and_one():
    """Position still open: some capacity, nothing filled -> need_score == 1.0 (nothing covered)."""
    scores = position_need_scores({"PG": 2.0}, {"PG": 0})
    assert scores["PG"] == 1.0


def test_need_score_partially_filled_position_is_between_zero_and_one():
    scores = position_need_scores({"PG": 2.0}, {"PG": 1})
    assert 0.0 < scores["PG"] < 1.0
    assert scores["PG"] == pytest.approx(0.5)


def test_need_score_fully_filled_position_is_zero_not_negative():
    """Position complete / overfilled: clamp to 0, never negative (deprioritize, don't crash or
    invert)."""
    scores = position_need_scores({"PG": 1.0}, {"PG": 1})
    assert scores["PG"] == 0.0
    overfilled = position_need_scores({"PG": 1.0}, {"PG": 5})
    assert overfilled["PG"] == 0.0


def test_need_score_zero_capacity_position_is_zero_not_a_crash():
    scores = position_need_scores({"C": 0.0}, {"C": 0})
    assert scores["C"] == 0.0


# --------------------------------------------------------------------------------------------
# player_need_scores
# --------------------------------------------------------------------------------------------

def test_player_need_score_is_best_of_eligible_positions():
    position_need = {"PG": 0.2, "SG": 0.9, "SF": 0.0, "PF": 0.0, "C": 0.0}
    scores = player_need_scores(pd.Series(["G", "PG", "C"]), position_need)
    assert scores.iloc[0] == pytest.approx(0.9)   # G -> max(PG=0.2, SG=0.9)
    assert scores.iloc[1] == pytest.approx(0.2)   # PG only
    assert scores.iloc[2] == pytest.approx(0.0)   # C


def test_player_need_score_unknown_position_is_zero_never_invented():
    position_need = {"PG": 1.0, "SG": 1.0, "SF": 1.0, "PF": 1.0, "C": 1.0}
    scores = player_need_scores(pd.Series([None, "??"]), position_need)
    assert (scores == 0.0).all()


# --------------------------------------------------------------------------------------------
# compute_need_board: the integration-level pure function
# --------------------------------------------------------------------------------------------

def test_empty_roster_falls_back_to_a_constant_shift_preserving_vorp_order():
    """Requirement: an empty roster should sensibly fall back to plain best-available-by-VORP."""
    board = _board([
        {"player_id": 1, "position": "PG", "vorp": 100.0},
        {"player_id": 2, "position": "C", "vorp": 90.0},
        {"player_id": 3, "position": "SF", "vorp": 80.0},
    ])
    result = compute_need_board(board, set(), LEAGUE_CFG)
    assert result.fallback is True
    # every position is wide open, so every player's need_score is 1.0 and the bonus is a constant
    assert (result.frame["need_score"] == 1.0).all()
    # ranking by need_adj_vorp must match ranking by vorp
    order_by_vorp = board.sort_values("vorp", ascending=False)["player_id"].tolist()
    order_by_need = result.frame.assign(player_id=board["player_id"]).sort_values(
        "need_adj_vorp", ascending=False)["player_id"].tolist()
    assert order_by_vorp == order_by_need


def test_nonempty_roster_boosts_open_positions_over_filled_ones():
    """Position still open vs. position fully filled: an open-position player should be able to
    overtake a slightly-higher-VORP player at an already-filled position."""
    board = _board([
        {"player_id": 1, "position": "C", "vorp": 100.0},   # my only drafted player: fills C
        {"player_id": 2, "position": "C", "vorp": 95.0},     # C is now relatively covered
        {"player_id": 3, "position": "PG", "vorp": 90.0},    # PG wide open
    ])
    result = compute_need_board(board, {1}, LEAGUE_CFG)
    assert result.fallback is False
    scores = result.frame["need_score"]
    # C has one drafted player against >1 capacity, so still has some need but less than PG's.
    assert scores.loc[board["player_id"] == 3].iloc[0] > scores.loc[board["player_id"] == 2].iloc[0]
    adj = result.frame["need_adj_vorp"]
    # The PG's need bonus is large enough to close a 5-point VORP gap against the more-covered C.
    assert adj.loc[board["player_id"] == 3].iloc[0] > adj.loc[board["player_id"] == 2].iloc[0]


def test_need_adj_vorp_never_changes_raw_vorp_column():
    board = _board([{"player_id": 1, "position": "PG", "vorp": 42.0}])
    result = compute_need_board(board, set(), LEAGUE_CFG)
    assert board["vorp"].iloc[0] == 42.0   # untouched
    assert result.frame["need_adj_vorp"].iloc[0] == pytest.approx(42.0 + DEFAULT_NEED_BONUS * 1.0)


def test_categories_flag_is_surfaced_but_not_computed():
    board = _board([{"player_id": 1, "position": "PG", "vorp": 1.0}])
    result = compute_need_board(board, set(), CATEGORIES_CFG)
    assert result.categories_applicable is True
    result2 = compute_need_board(board, set(), LEAGUE_CFG)
    assert result2.categories_applicable is False


def test_compute_need_board_handles_empty_board_without_crashing():
    board = pd.DataFrame(columns=["player_id", "name", "position", "vorp"])
    result = compute_need_board(board, set(), LEAGUE_CFG)
    assert result.frame.empty
    assert result.fallback is True


def test_position_need_table_shape_and_columns():
    board = _board([{"player_id": 1, "position": "PG", "vorp": 1.0}])
    result = compute_need_board(board, {1}, LEAGUE_CFG)
    table = position_need_table(result)
    assert list(table.columns) == ["need_position", "need_capacity", "need_filled", "need_score"]
    assert len(table) == 5   # one row per ESPN specific position
