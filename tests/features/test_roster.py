"""Unit tests for src.features.roster: build_roster_panel, RosterFeatures.

Uses tests/models/model_testkit.py's synthetic league for a valid team schedule/skeleton and
tests/features/roster_testkit.py to place players in EXACT known minutes/position patterns, so the
pace/role_share/pos_crowding math has a known right answer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import make_league  # noqa: E402

from roster_testkit import add_team_season_roster  # noqa: E402

from src.contracts import History, season_start
from src.features.roster import (
    CROWD_MPG_THRESHOLD, MIN_FIT_ROWS, MPG_SHIFT_CAP, RosterFeatures, build_roster_panel,
)

TARGET = "2018-19"       # one season after PATTERN_SEASON, so PATTERN_SEASON is history
PATTERN_SEASON = "2017-18"


@pytest.fixture(scope="module")
def tables():
    return make_league()


# --------------------------------------------------------------------------- build_roster_panel

def test_pace_role_share_hand_computed(tables):
    """Construct a tiny two-player team with exact, hand-computable box lines and check pace/
    role_share come out exactly as expected."""
    t, team = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[
            # (player_id, name, position, minutes_per_game, n_games, box kwargs)
            dict(player_id=9_910_001, name="Star", position="G", minutes=36.0, n_games=10,
                fga=20, oreb=1, tov=3, fta=8),
            dict(player_id=9_910_002, name="Role Player", position="F", minutes=24.0, n_games=10,
                fga=10, oreb=2, tov=1, fta=2),
        ],
    )
    h = History.until(t, TARGET)
    panel = build_roster_panel(h)
    s = season_start(PATTERN_SEASON)

    # Hand-computed possessions per game for this two-player team:
    # POSS = (fga_1+fga_2) - (oreb_1+oreb_2) + (tov_1+tov_2) + 0.44*(fta_1+fta_2)
    expected_poss = (20 + 10) - (1 + 2) + (3 + 1) + 0.44 * (8 + 2)
    row1 = panel[(panel.player_id == 9_910_001) & (panel.s == s)].iloc[0]
    row2 = panel[(panel.player_id == 9_910_002) & (panel.s == s)].iloc[0]
    assert row1["pace"] == pytest.approx(expected_poss)
    assert row2["pace"] == pytest.approx(expected_poss)

    # team_min_per_game = 36 + 24 = 60; role_share = player_mpg / (team_min_per_game / 5)
    team_min_per_game = 36.0 + 24.0
    assert row1["role_share"] == pytest.approx(36.0 / (team_min_per_game / 5.0))
    assert row2["role_share"] == pytest.approx(24.0 / (team_min_per_game / 5.0))


def test_pos_crowding_distinguishes_crowded_from_uncrowded(tables):
    """Three guards at >= CROWD_MPG_THRESHOLD mpg (crowded position) vs. one lone center
    (uncrowded position) on the same team-season."""
    t, team = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[
            dict(player_id=9_910_010, name="Guard A", position="G", minutes=30.0, n_games=10),
            dict(player_id=9_910_011, name="Guard B", position="G", minutes=28.0, n_games=10),
            dict(player_id=9_910_012, name="Guard C", position="G", minutes=16.0, n_games=10),
            dict(player_id=9_910_013, name="Lone Center", position="C", minutes=32.0, n_games=10),
            # Below the crowding threshold: should not count towards any pos_crowding total.
            dict(player_id=9_910_014, name="Bench Guard", position="G", minutes=8.0, n_games=10),
        ],
    )
    h = History.until(t, TARGET)
    panel = build_roster_panel(h)
    s = season_start(PATTERN_SEASON)

    def crowding(pid):
        return panel[(panel.player_id == pid) & (panel.s == s)].iloc[0]["pos_crowding"]

    # Each guard >= threshold sees 2 OTHER qualifying guards (3 qualifying guards total, minus self).
    assert crowding(9_910_010) == 2
    assert crowding(9_910_011) == 2
    assert crowding(9_910_012) == 2
    # The lone center has no other qualifying centers.
    assert crowding(9_910_013) == 0
    # The bench guard (below threshold) still counts the 3 qualifying guards as "others".
    assert crowding(9_910_014) == 3


def test_panel_columns_and_shape_on_real_shaped_synthetic_history(tables):
    h = History.until(tables, TARGET)
    panel = build_roster_panel(h)
    assert not panel.empty
    for c in ("player_id", "s", "pace", "role_share", "pos_crowding"):
        assert c in panel.columns
    assert (panel["pace"] > 0).all()
    assert (panel["role_share"] >= 0).all()
    assert (panel["pos_crowding"] >= 0).all()


def test_build_roster_panel_on_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    panel = build_roster_panel(h)
    assert panel.empty


def test_panel_has_no_row_for_player_with_zero_history(tables):
    h = History.until(tables, TARGET)
    panel = build_roster_panel(h)
    assert not (panel.player_id == 9_999_998).any()


# --------------------------------------------------------------------------- RosterFeatures

def test_roster_features_build_deterministic(tables):
    t, _ = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_920_001, name="Repeat", position="F", minutes=28.0, n_games=15)],
    )
    h = History.until(t, TARGET)
    panel = build_roster_panel(h)
    pids = panel["player_id"].to_numpy()
    target_s = panel["s"].to_numpy() + 1
    mpg_est = np.full(len(pids), 20.0)
    actual_mpg = mpg_est + 1.0
    weight = np.full(len(pids), 70.0)

    a = RosterFeatures.fit(h, pids, target_s, mpg_est, actual_mpg, weight, n_lags=3, decay=0.6)
    b = RosterFeatures.fit(h, pids, target_s, mpg_est, actual_mpg, weight, n_lags=3, decay=0.6)
    out_a = a.build(np.array([9_920_001]), np.array([season_start(PATTERN_SEASON) + 1]))
    out_b = b.build(np.array([9_920_001]), np.array([season_start(PATTERN_SEASON) + 1]))
    np.testing.assert_array_equal(out_a, out_b)


def test_roster_features_fit_falls_back_to_zero_with_too_few_rows(tables):
    h = History.until(tables, TARGET)
    panel = build_roster_panel(h)
    assert len(panel) >= 5
    # Fewer than MIN_FIT_ROWS usable training rows -> zero-adjustment fallback.
    n = min(MIN_FIT_ROWS - 1, len(panel))
    sub = panel.iloc[:n]
    pids = sub["player_id"].to_numpy()
    target_s = sub["s"].to_numpy() + 1
    mpg_est = np.full(n, 20.0)
    actual_mpg = mpg_est + 5.0
    weight = np.full(n, 70.0)
    feats = RosterFeatures.fit(h, pids, target_s, mpg_est, actual_mpg, weight, n_lags=3, decay=0.6)
    assert np.all(feats.beta == 0.0)
    out = feats.build(pids, target_s)
    assert np.all(out == 0.0)


def test_roster_features_fit_on_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    feats = RosterFeatures.fit(h, np.array([1, 2]), np.array([2018, 2018]),
                               np.array([20.0, 20.0]), np.array([22.0, 18.0]), np.array([70.0, 70.0]),
                               n_lags=3, decay=0.6)
    assert np.all(feats.beta == 0.0)
    out = feats.build(np.array([1, 2]), np.array([2018, 2018]))
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()


def test_roster_features_build_no_context_returns_zero_not_nan(tables):
    h = History.until(tables, TARGET)
    panel = build_roster_panel(h)
    pids = panel["player_id"].to_numpy()
    target_s = panel["s"].to_numpy() + 1
    mpg_est = np.full(len(pids), 20.0)
    actual_mpg = mpg_est + 3.0
    weight = np.full(len(pids), 70.0)
    feats = RosterFeatures.fit(h, pids, target_s, mpg_est, actual_mpg, weight, n_lags=3, decay=0.6)

    # A brand-new player_id never seen in the panel has no lagged context at all.
    out = feats.build(np.array([9_999_999_999]), np.array([season_start(TARGET)]))
    assert out.shape == (1,)
    assert not np.isnan(out).any()
    assert out[0] == 0.0


def test_roster_features_build_clips_to_mpg_shift_cap(tables):
    h = History.until(tables, TARGET)
    panel = build_roster_panel(h)
    pids = panel["player_id"].to_numpy()
    target_s = panel["s"].to_numpy() + 1
    # Force an extreme beta directly (bypassing fit) to check .build()'s clip, deterministically.
    extreme_beta = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    feats = RosterFeatures(panel, n_lags=3, decay=0.6, beta=extreme_beta)
    out = feats.build(pids, target_s)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    assert np.all(np.abs(finite) <= MPG_SHIFT_CAP + 1e-9)
    # And it actually hits the cap for at least one row (not just staying under it).
    assert np.any(np.isclose(np.abs(finite), MPG_SHIFT_CAP))


def test_roster_features_zero_beta_is_noop():
    empty_panel = pd.DataFrame(columns=["player_id", "s", "pace", "role_share", "pos_crowding"])
    feats = RosterFeatures(empty_panel, n_lags=3, decay=0.6, beta=np.zeros(4))
    out = feats.build(np.array([1, 2, 3]), np.array([2018, 2018, 2018]))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_crowd_mpg_threshold_is_the_documented_value():
    assert CROWD_MPG_THRESHOLD == pytest.approx(15.0)


def test_min_fit_rows_is_the_documented_value():
    assert MIN_FIT_ROWS == 60


def test_mpg_shift_cap_is_the_documented_value():
    assert MPG_SHIFT_CAP == pytest.approx(4.0)
