"""Unit tests for src.features.injury: streak stats, the injury panel, InjuryFeatures.

Uses tests/models/model_testkit.py's synthetic league for a valid team schedule/skeleton and
tests/features/injury_testkit.py to place a player in EXACT known played/missed patterns, so the
streak-length/frequency math has a known right answer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import make_league  # noqa: E402

from injury_testkit import add_player_season_pattern  # noqa: E402

from src.contracts import History, season_start
from src.features.injury import (
    InjuryFeatures, LONG_ABSENCE_THRESHOLD, _streak_stats, build_injury_panel, fit_age_absence_curve,
)

TARGET = "2018-19"       # one season after PATTERN_SEASON, so PATTERN_SEASON is history
PATTERN_SEASON = "2017-18"


@pytest.fixture(scope="module")
def tables():
    return make_league()


# --------------------------------------------------------------------------- _streak_stats (pure function)

def test_streak_stats_no_misses():
    assert _streak_stats(np.zeros(10, dtype=bool)) == (0, 0)


def test_streak_stats_all_missed():
    n = 12
    longest, count = _streak_stats(np.ones(n, dtype=bool))
    assert (longest, count) == (n, 1)


def test_streak_stats_one_long_streak():
    missed = np.zeros(20, dtype=bool)
    missed[3:10] = True  # 7 in a row
    assert _streak_stats(missed) == (7, 1)


def test_streak_stats_many_short_streaks():
    missed = np.zeros(20, dtype=bool)
    missed[[1, 3, 5, 7, 9]] = True  # five isolated absences
    assert _streak_stats(missed) == (1, 5)


def test_streak_stats_same_total_different_shape():
    """The whole point of this feature: two players missing the same number of games can have
    very different streak shapes, and the stats must tell them apart."""
    one_big = np.zeros(20, dtype=bool)
    one_big[2:12] = True  # 10 missed, one streak
    scattered = np.zeros(20, dtype=bool)
    scattered[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = True  # 10 missed, ten streaks
    assert one_big.sum() == scattered.sum() == 10
    l1, n1 = _streak_stats(one_big)
    l2, n2 = _streak_stats(scattered)
    assert l1 > l2 and n1 < n2


def test_streak_stats_streak_at_the_edges():
    missed = np.zeros(10, dtype=bool)
    missed[:3] = True
    missed[-2:] = True
    assert _streak_stats(missed) == (3, 2)


# --------------------------------------------------------------------------- build_injury_panel

def _default_team_and_len(tables, season):
    tg = tables["team_games"]
    team = tg[tg.season == season].groupby("team_id").size().index[0]
    n = int((tg.season == season).to_numpy().__and__((tg.team_id == team).to_numpy()).sum())
    return int(team), n


def test_one_long_streak_vs_scattered_absences(tables):
    """Construct two players with the SAME games-missed total but different streak shapes
    (patterns must cover the full team schedule -- build_injury_panel uses the whole season's
    team_games as the denominator, not just the games a test happens to touch) and check the
    panel tells them apart."""
    team, n = _default_team_and_len(tables, PATTERN_SEASON)
    assert n >= 40, "synthetic schedule too short for this pattern"
    pattern1 = [True] * 30 + [False] * 10 + [True] * (n - 40)
    # 10 isolated single-game absences spread through the first 30 games, rest played.
    scattered_idx = {3 * i + 2 for i in range(10)}
    pattern2 = [(i not in scattered_idx) for i in range(30)] + [True] * (n - 30)
    t = add_player_season_pattern(tables, 9_900_001, "Long Absence", PATTERN_SEASON,
                                  played=pattern1, team_of_season=team)
    t = add_player_season_pattern(t, 9_900_002, "Scattered Absences", PATTERN_SEASON,
                                  played=pattern2, team_of_season=team)
    h = History.until(t, TARGET)
    panel = build_injury_panel(h)
    s = season_start(PATTERN_SEASON)
    row1 = panel[(panel.player_id == 9_900_001) & (panel.s == s)].iloc[0]
    row2 = panel[(panel.player_id == 9_900_002) & (panel.s == s)].iloc[0]
    assert row1.n_streaks == 1
    assert row1.longest_streak_frac == pytest.approx(10 / n)
    assert row2.n_streaks == 10
    assert row2.longest_streak_frac == pytest.approx(1 / n)
    assert row1.missed_frac == pytest.approx(row2.missed_frac)  # same total, different shape
    assert row1.missed_frac == pytest.approx(10 / n)


def test_panel_empty_for_player_with_zero_history(tables):
    """A player who has never appeared in game_logs before the target season has no panel row."""
    h = History.until(tables, TARGET)
    panel = build_injury_panel(h)
    assert not (panel.player_id == 9_999_999).any()


def test_player_who_missed_an_entire_season_gets_no_row_for_it(tables):
    """Consistent with src.models.panel.build_panel: a fully-missed season contributes zero
    exposure, not a fabricated one-streak-covers-everything row."""
    n_games = len(tables["team_games"][(tables["team_games"].season == PATTERN_SEASON)
                                       & (tables["team_games"].team_id ==
                                          tables["team_games"][tables["team_games"].season == PATTERN_SEASON]
                                          .groupby("team_id").size().index[0])])
    t = add_player_season_pattern(tables, 9_900_003, "Redshirt", PATTERN_SEASON, played=[False] * n_games)
    h = History.until(t, TARGET)
    panel = build_injury_panel(h)
    s = season_start(PATTERN_SEASON)
    assert not ((panel.player_id == 9_900_003) & (panel.s == s)).any()


def test_panel_columns_and_dtypes_on_real_shaped_synthetic_history(tables):
    h = History.until(tables, TARGET)
    panel = build_injury_panel(h)
    assert not panel.empty
    for c in ("player_id", "s", "season_len", "missed_frac", "longest_streak_frac", "n_streaks", "gp"):
        assert c in panel.columns
    assert (panel["missed_frac"] >= 0).all() and (panel["missed_frac"] <= 1).all()
    assert (panel["longest_streak_frac"] <= panel["missed_frac"] + 1e-9).all()
    assert (panel["n_streaks"] >= 0).all()
    assert (panel["gp"] + (panel["missed_frac"] * panel["season_len"]).round() <= panel["season_len"] + 1).all()


def test_build_injury_panel_on_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    panel = build_injury_panel(h)
    assert panel.empty


# --------------------------------------------------------------------------- fit_age_absence_curve

def test_age_absence_curve_recovers_rising_risk_with_age():
    """Synthetic panel where absence rate rises linearly with age; the fit should recover a
    positive age slope (not exact, but the right sign and a materially better fit than flat)."""
    rng = np.random.default_rng(0)
    ages = rng.uniform(20, 38, 400)
    missed = np.clip(0.05 + 0.01 * (ages - 20) + rng.normal(0, 0.02, 400), 0, 1)
    panel = pd.DataFrame({"age": ages, "missed_frac": missed, "season_len": np.full(400, 70.0)})
    a, b, c = fit_age_absence_curve(panel, min_rows=60)
    predicted_young = a + b * 22 + c * 22 ** 2
    predicted_old = a + b * 35 + c * 35 ** 2
    assert predicted_old > predicted_young


def test_age_absence_curve_falls_back_when_too_little_data():
    panel = pd.DataFrame({"age": [25.0, 26.0], "missed_frac": [0.1, 0.2], "season_len": [70.0, 70.0]})
    a, b, c = fit_age_absence_curve(panel, min_rows=60)
    assert b == 0.0 and c == 0.0
    assert a == pytest.approx(0.15)


def test_age_absence_curve_on_empty_panel():
    a, b, c = fit_age_absence_curve(pd.DataFrame(columns=["age", "missed_frac", "season_len"]))
    assert (a, b, c) == (0.15, 0.0, 0.0)


# --------------------------------------------------------------------------- InjuryFeatures

def test_injury_features_flags_long_absence_last_season(tables):
    t = add_player_season_pattern(tables, 9_900_004, "Just Hurt", PATTERN_SEASON,
                                  played=[True] * 30 + [False] * 15 + [True] * 15)
    t = add_player_season_pattern(t, 9_900_005, "Healthy", PATTERN_SEASON, played=[True] * 60)
    h = History.until(t, TARGET)
    feats = InjuryFeatures.fit(h, ages=np.array([]), panel_pids=np.array([]), panel_s=np.array([]),
                               n_lags=4, decay=0.6)
    s = season_start(PATTERN_SEASON) + 1  # target row looks back at PATTERN_SEASON as lag 1
    out = feats.build(np.array([9_900_004, 9_900_005]), np.array([s, s]), np.array([28.0, 28.0]))
    streak_bar, long_absence = out[:, 0], out[:, 2]
    assert long_absence[0] == 1.0
    assert long_absence[1] == 0.0
    assert streak_bar[0] > streak_bar[1]


def test_injury_features_build_handles_rookie_with_no_history():
    """A brand-new player_id absent from the injury panel gets NaN extra features, not a crash."""
    empty_panel = pd.DataFrame(columns=["player_id", "s", "season_len", "missed_frac",
                                        "longest_streak_frac", "n_streaks", "gp", "age"])
    feats = InjuryFeatures(empty_panel, n_lags=4, decay=0.6, age_curve=(0.15, 0.0, 0.0))
    out = feats.build(np.array([1, 2]), np.array([2018, 2018]), np.array([20.0, 21.0]))
    assert out.shape == (2, 4)
    assert np.isnan(out).all()


def test_injury_features_deterministic(tables):
    t = add_player_season_pattern(tables, 9_900_006, "Repeat", PATTERN_SEASON,
                                  played=[True] * 25 + [False] * 8 + [True] * 27)
    h = History.until(t, TARGET)
    a = InjuryFeatures.fit(h, np.array([]), np.array([]), np.array([]), n_lags=4, decay=0.6)
    b = InjuryFeatures.fit(h, np.array([]), np.array([]), np.array([]), n_lags=4, decay=0.6)
    pids = np.array([9_900_006])
    s = np.array([season_start(PATTERN_SEASON) + 1])
    np.testing.assert_array_equal(a.build(pids, s, np.array([26.0])), b.build(pids, s, np.array([26.0])))


def test_long_absence_threshold_is_the_documented_value():
    assert LONG_ABSENCE_THRESHOLD == pytest.approx(0.15)
