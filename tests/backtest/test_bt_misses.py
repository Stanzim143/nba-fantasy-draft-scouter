import numpy as np
import pandas as pd
import pytest

from bt_helpers import frame_from_arrays, make_logs
from src.backtest.benchmarks import NaiveLastSeason
from src.backtest.harness import BacktestResult, walk_forward
from src.backtest.misses import analyze_misses

S = "2020-21"


def _tables():
    prev = make_logs([(p, 5, 20, 5, 3) for p in (1, 2, 4, 5, 6, 7)], season="2019-20", team_id=1)
    cur = make_logs([(2, 5, 20, 5, 3), (4, 5, 20, 5, 3), (5, 5, 20, 5, 3), (6, 5, 20, 5, 3), (7, 5, 20, 5, 3),
                     (3, 5, 20, 5, 3)], season=S, team_id=1)
    cur.loc[cur.player_id == 5, "team_id"] = 2                 # p5 changed teams
    bio = pd.DataFrame({"season": S, "player_id": [1, 2, 3, 4, 5, 7], "age_at_season_start": [27.0, 26, 21, 27, 27, 27],
                        "team_id": 1})
    players = pd.DataFrame({"player_id": [6], "birthdate": [pd.Timestamp("1985-06-01")]})     # age via birthdate
    return {"game_logs": pd.concat([prev, cur], ignore_index=True), "player_season_bio": bio, "players": players}


def _result():
    # columns: p1..p7 ; proj/actual totals chosen so the biggest misses are all distinguishable
    f = frame_from_arrays(
        pred_total=[3000, 500, np.nan, 1500, 1500, 1500, 1500],
        actual_total=[0, 1500, 1600, 900, 1000, 1400, 1420],
        played=[False, True, True, True, True, True, True],
        pred_gp=[60, 20, 60, 60, 60, 60, 60],
        actual_gp=[0, 70, 62, 60, 60, 60, 60])
    f["season"] = S
    f["proj_fppg"] = f["proj_total_fp"] / f["proj_gp"]
    f["proj_mpg"] = [30, 30, np.nan, 30, 30, 30, 30]
    f["actual_mpg"] = [np.nan, 30, 30, 18, 30, 30, 30]
    f["team_id"] = [np.nan, 1, 1, 1, 2, 1, 1]
    f["n_teams"] = [np.nan, 1, 1, 1, 1, 1, 1]
    return BacktestResult("m", [S], pd.DataFrame(index=[S]), f, {"ks": [12], "min_gp": 10}, {})


def test_each_heuristic_tag_and_primary_priority():
    ma = analyze_misses(_result(), _tables(), top_n=10, gp_gap=20, mpg_gap=5)
    t = ma.table.set_index("player_id")
    assert t.loc[1, "primary"] == "availability"           # projected 3000, played 0 games
    assert t.loc[1, "error"] == -3000 and t.loc[1, "actual_total_fp"] == 0
    assert t.loc[2, "primary"] == "availability"           # under-projected: played 70 of a projected 20
    assert t.loc[3, "primary"] == "rookie/debut" and np.isnan(t.loc[3, "proj_total_fp"])   # coverage miss
    assert t.loc[3, "error"] == 1600
    assert t.loc[4, "primary"] == "role/minutes" and t.loc[4, "mpg_delta"] == pytest.approx(-12)
    assert t.loc[5, "primary"] == "team change"
    assert t.loc[6, "primary"] == "age" and "age" in t.loc[6, "tags"]      # 35 via birthdate fallback
    assert t.loc[7, "primary"] == "unexplained"
    # ordered by absolute error, descending
    assert ma.table["error"].abs().is_monotonic_decreasing


def test_availability_needs_the_games_component_to_dominate():
    r = _result()
    f = r.players
    # p4: projected 60 GP, played 30 (gap 30 >= 20) but per-game rate collapse dominates the miss
    f.loc[f.player_id == 4, ["actual_gp", "actual_total_fp"]] = [30.0, -600.0]
    f.loc[f.player_id == 4, "actual_fppg"] = -20.0
    ma = analyze_misses(r, _tables(), top_n=10, gp_gap=20)
    row = ma.table.set_index("player_id").loc[4]
    assert row["avail_share"] < 0.5 and "availability" not in row["tags"]


def test_top_n_limit_and_summary_shares():
    ma = analyze_misses(_result(), _tables(), top_n=3)
    assert len(ma.table) == 3
    assert ma.table["player_id"].tolist() == [1, 3, 2]          # |errors| 3000, 1600, 1000
    assert ma.tag_summary["share_abs_error"].sum() == pytest.approx(1.0)
    assert ma.tag_summary["count"].sum() == 3


def test_markdown_table():
    ma = analyze_misses(_result(), _tables(), top_n=4)
    md = ma.to_markdown()
    assert "**2020-21**" in md and "| player |" in md and "availability" in md and "rookie/debut" in md
    assert "n/a" in md                                          # the unprojected player's projection


def test_empty_result_is_handled():
    r = _result()
    r.players = r.players.iloc[0:0]
    ma = analyze_misses(r, _tables())
    assert ma.table.empty and ma.to_markdown() == "_No misses to report._"


def test_end_to_end_on_synthetic_league(tables):
    res = walk_forward(tables, NaiveLastSeason(), ["2017-18", "2018-19"])
    ma = analyze_misses(res, tables, top_n=8, gp_gap=10)
    assert len(ma.table) == 16 and set(ma.table["season"]) == {"2017-18", "2018-19"}
    assert set(ma.table["primary"]) <= {"rookie/debut", "availability", "team change", "role/minutes", "age",
                                        "unexplained"}
    assert (ma.table.groupby("season")["error"].apply(lambda s: s.abs().is_monotonic_decreasing)).all()
    # the biggest synthetic misses are injuries / debuts, which the tagger must be able to see
    assert {"availability", "rookie/debut"} & set(ma.table["primary"])
