import math

import numpy as np
import pandas as pd
import pytest

from bt_helpers import SCORING, make_logs
from src.backtest import actuals as act
from src.backtest.benchmarks import stats_to_projection
from src.backtest.errors import BacktestError
from src.contracts import STAT_COLUMN_MAP
from src.value.frame import fantasy_points_frame

SC = {"PTS": 1, "REB": 1, "AST": 2}      # simple hand-checkable scoring


def _proj(rows, season="2020-21"):
    """Projections from [(player_id, gp, pts, reb, ast)] under the simple scoring."""
    stats = pd.DataFrame({
        "player_id": [r[0] for r in rows], "player_name": [f"P{r[0]}" for r in rows],
        "proj_gp": [float(r[1]) for r in rows], "proj_mpg": 30.0,
        **{f"proj_{c}": 0.0 for c in STAT_COLUMN_MAP.values()},
    })
    stats["proj_pts"] = [float(r[2]) for r in rows]
    stats["proj_reb"] = [float(r[3]) for r in rows]
    stats["proj_ast"] = [float(r[4]) for r in rows]
    return stats_to_projection(stats, season=season, model="m", scoring=SC)


def test_season_actuals_hand_computed():
    logs = make_logs([(1, 10, 20, 5, 3), (2, 4, 10, 2, 0)])     # p1: 20+5+6=31 FP/game
    a = act.season_actuals(logs, "2020-21", SC)
    r1 = a[a.player_id == 1].iloc[0]
    assert r1.gp == 10 and r1.total_fp == pytest.approx(310) and r1.fppg == pytest.approx(31)
    assert r1.mpg == pytest.approx(30) and r1.minutes == pytest.approx(300)
    r2 = a[a.player_id == 2].iloc[0]
    assert r2.gp == 4 and r2.total_fp == pytest.approx(48)
    assert list(a.columns) == act.ACTUAL_COLUMNS
    assert a["player_id"].is_monotonic_increasing


def test_actuals_use_league_scoring(tables):
    gl = tables["game_logs"]
    s = "2016-17"
    a = act.season_actuals(gl, s, SCORING)
    direct = gl[gl.season == s].assign(fp=lambda d: fantasy_points_frame(d, SCORING)).groupby("player_id")["fp"].sum()
    np.testing.assert_allclose(a.set_index("player_id")["total_fp"].sort_index(), direct.sort_index())


def test_traded_player_is_one_row_with_last_team():
    a = make_logs([(1, 3, 10, 0, 0)], team_id=1)
    b = make_logs([(1, 3, 10, 0, 0)], team_id=2, start="2020-12-01")
    out = act.season_actuals(pd.concat([a, b], ignore_index=True), "2020-21", SC)
    assert len(out) == 1 and out.iloc[0].gp == 6 and out.iloc[0].n_teams == 2 and out.iloc[0].team_id == 2


def test_missing_season_is_a_clear_error():
    with pytest.raises(BacktestError, match="no game_logs rows for season 2030-31"):
        act.season_actuals(make_logs([(1, 2, 10, 0, 0)]), "2030-31", SC)
    with pytest.raises(ValueError):
        act.season_actuals(make_logs([(1, 2, 10, 0, 0)]), "garbage", SC)


def _frame(logs, proj_rows):
    a = act.season_actuals(logs, "2020-21", SC)
    g = act.game_fp(logs, "2020-21", SC)
    return act.build_eval_frame(_proj(proj_rows), a, g, "2020-21")


def test_zero_games_projected_player_counts_as_zero_not_dropped():
    logs = make_logs([(1, 10, 20, 5, 3)])
    f = _frame(logs, [(1, 10, 20, 5, 3), (2, 60, 30, 8, 5)])      # p2 projected for 60 GP, never played
    r2 = f[f.player_id == 2].iloc[0]
    assert r2.projected and not r2.played
    assert r2.actual_gp == 0 and r2.actual_total_fp == 0
    assert math.isnan(r2.actual_fppg)                              # per-game rate undefined
    assert r2.proj_total_fp == pytest.approx(60 * (30 + 8 + 10))   # model still owns 2880 projected FP
    assert len(f) == 2


def test_unprojected_player_who_played_is_a_coverage_miss():
    logs = make_logs([(1, 10, 20, 5, 3), (3, 8, 40, 10, 6)])
    f = _frame(logs, [(1, 10, 20, 5, 3)])
    r3 = f[f.player_id == 3].iloc[0]
    assert r3.played and not r3.projected
    assert math.isnan(r3.proj_total_fp) and math.isnan(r3.proj_gp)
    assert r3.actual_total_fp == pytest.approx(8 * (40 + 10 + 12))
    assert r3.player_name == "P3"                                  # name taken from the actuals


def test_eval_frame_sorted_deterministic_and_complete_columns():
    logs = make_logs([(5, 5, 10, 1, 1), (2, 5, 12, 1, 1), (9, 5, 14, 1, 1)])
    f = _frame(logs, [(9, 5, 14, 1, 1), (2, 5, 12, 1, 1), (7, 5, 1, 1, 1)])
    assert f["player_id"].tolist() == [2, 5, 7, 9]
    for c in ["season", "projected", "played", "proj_total_fp", "actual_total_fp", "actual_gp", "actual_mpg",
              "n_teams", "band_games", "n_below_p10"]:
        assert c in f.columns
    assert f[["band_games", "n_below_p10", "n_above_p90", "p50_games", "n_below_p50"]].dtypes.eq("int64").all()


def test_band_counts_are_game_level():
    logs = make_logs([(1, 10, 20, 5, 3)])
    # vary FP per game: replace pts for a few games
    logs.loc[:1, ["pts", "fgm", "fga"]] = [[2, 1, 1], [2, 1, 1]]          # two bad games (FP 13)
    logs.loc[2, ["pts", "fgm", "fga"]] = [60, 30, 30]                       # one huge game (FP 71)
    proj = _proj([(1, 10, 20, 5, 3)])
    proj["fppg_p10"], proj["fppg_p50"], proj["fppg_p90"] = 20.0, 31.0, 45.0
    g = act.game_fp(logs, "2020-21", SC)
    bc = act.band_counts(g, proj).iloc[0]
    assert bc.band_games == 10 and bc.n_below_p10 == 2 and bc.n_above_p90 == 1
    assert bc.p50_games == 10 and bc.n_below_p50 == 2


def test_band_counts_ignores_missing_bands():
    logs = make_logs([(1, 10, 20, 5, 3)])
    proj = _proj([(1, 10, 20, 5, 3)])                                     # bands are NaN
    bc = act.band_counts(act.game_fp(logs, "2020-21", SC), proj).iloc[0]
    assert bc.band_games == 0 and bc.n_below_p10 == 0 and bc.p50_games == 0


def test_rank_only_eval_frame_keeps_only_the_score():
    logs = make_logs([(1, 10, 20, 5, 3), (2, 10, 10, 5, 3)])
    a = act.season_actuals(logs, "2020-21", SC)
    g = act.game_fp(logs, "2020-21", SC)
    proj = pd.DataFrame({"season": "2020-21", "player_id": [1, 2], "player_name": ["P1", "P2"], "model": "adp",
                         "proj_total_fp": [-1.0, -2.0]})
    f = act.build_eval_frame(proj, a, g, "2020-21", rank_only=True)
    assert f["proj_total_fp"].tolist() == [-1.0, -2.0]
    assert f["proj_gp"].isna().all() and f["proj_fppg"].isna().all()
    assert (f["band_games"] == 0).all()
