"""Minutes / usage trends, absences and who inherits minutes, on hand-built game logs."""
import numpy as np
import pandas as pd
import pytest
from ise_testkit import SEASON, date_after_games, make_tables

from src.inseason import signals as G
from src.inseason.ros import season_to_date
from src.value.league import load_league

TEAM = 1610612737
N = 20


def _logs(spec, n=N, team=TEAM):
    """spec: player_id -> (mpg by game function or list)."""
    rows, tg = [], []
    for g in range(n):
        gid = f"g{g:02d}"
        day = pd.Timestamp("2023-11-01") + pd.Timedelta(days=g)
        tg.append({"season": SEASON, "game_id": gid, "game_date": day, "team_id": team, "team_abbr": "ATL",
                   "is_home": True, "pts_for": 100, "pts_against": 99})
        for pid, mins in spec.items():
            m = mins[g] if not callable(mins) else mins(g)
            if m and m > 0:
                rows.append({"season": SEASON, "game_id": gid, "game_date": day, "player_id": pid,
                             "player_name": f"P{pid}", "team_id": team, "team_abbr": "ATL", "min": float(m),
                             "fgm": 4, "fga": 9, "fg3m": 1, "fg3a": 3, "ftm": 2, "fta": 3, "oreb": 1, "dreb": 3, "reb": 4,
                             "ast": 3, "stl": 1, "blk": 0, "tov": 2, "pf": 2, "pts": 11})
    gl = pd.DataFrame(rows)
    gl["fp"] = gl["min"] * 1.0
    return gl, pd.DataFrame(tg)


def _spec():
    return {
        1: lambda g: 30 if g < 14 else 0,                      # star, out from game 14 on (6 games)
        2: lambda g: 15 if g < 15 else 22,                     # bench guard, minutes jump in the last 5
        3: lambda g: 25,                                       # steady wing
        4: lambda g: 28,                                       # steady centre
        5: lambda g: 6,                                        # deep bench
        6: lambda g: 20 if g >= 15 else 0,                     # newcomer: too few base games
    }


POS = pd.Series({1: "G", 2: "G", 3: "F", 4: "C", 5: "G", 6: "F"})


def test_trend_signals_flag_the_rising_player_only():
    gl, _ = _logs(_spec())
    t = G.trend_signals(gl).set_index("player_id")
    assert bool(t.at[2, "rising_minutes"]) and t.at[2, "min_delta"] == pytest.approx(22 - 15)
    assert not t.loc[3, "rising_minutes"] and not t.loc[4, "rising_minutes"] and not t.loc[3, "rising_usage"]
    assert 6 not in t.index                                     # fewer than 8 baseline games: no signal, not a guess
    assert G.trend_signals(gl.iloc[0:0]).empty


def test_usage_trend_uses_the_per36_volume_proxy():
    gl, _ = _logs(_spec())
    gl.loc[(gl["player_id"] == 3) & (gl["game_date"] >= gl["game_date"].max() - pd.Timedelta(days=4)), ["fga", "fta"]] = [20, 8]
    t = G.trend_signals(gl).set_index("player_id")
    assert bool(t.at[3, "rising_usage"]) and t.at[3, "usg_delta"] > G.USG_DELTA and not bool(t.at[3, "rising_minutes"])


def test_absent_players_finds_the_rotation_player_who_missed_games():
    gl, tg = _logs(_spec())
    a = G.absent_players(gl, tg).set_index("player_id")
    assert list(a.index) == [1] and a.at[1, "streak"] == 6 and not bool(a.at[1, "long_term"])
    assert G.absent_players(gl, tg, min_mpg=40).empty
    gl2, tg2 = _logs({1: lambda g: 30 if g < 9 else 0, 3: lambda g: 25}, n=20)
    assert bool(G.absent_players(gl2, tg2).set_index("player_id").at[1, "long_term"])   # 11 straight misses


def test_espn_out_status_adds_a_player_who_has_no_missed_game_yet():
    gl, tg = _logs(_spec())
    a = G.absent_players(gl, tg, injuries=pd.Series({4: "OUT", 3: "DAY_TO_DAY", 5: "OUT"}))
    row = a.set_index("player_id")
    assert row.at[4, "source"] == "ESPN OUT" and row.at[4, "streak"] == 0
    assert 3 not in row.index and 5 not in row.index          # day-to-day is not out; a 6 mpg player is not rotation


def test_allocate_minutes_conserves_caps_and_prefers_headroom_and_position():
    mates = pd.DataFrame({"mpg": [34.0, 20.0, 12.0, 20.0], "group": ["G", "G", "G", "C"]}, index=[10, 11, 12, 13])
    g = G.allocate_minutes(30.0, mates, "G")
    assert g.sum() == pytest.approx(30.0)
    assert (mates["mpg"] + g <= G.MINUTES_CAP + 1e-9).all()
    assert g[12] > g[11] > g[10]                               # more headroom, more minutes
    assert g[11] > g[13]                                        # same position group gets the boost
    tiny = G.allocate_minutes(100.0, mates, "G")                # more than the roster can absorb: capped, not invented
    assert tiny.sum() == pytest.approx((G.MINUTES_CAP - mates["mpg"]).sum())
    assert G.allocate_minutes(0.0, mates, "G").sum() == 0


def test_beneficiaries_rank_the_bench_guard_first_and_blend_with_without_data():
    gl, tg = _logs(_spec())
    absent = G.absent_players(gl, tg)
    fpm = gl.groupby("player_id")["fp"].sum() / gl.groupby("player_id")["min"].sum()
    b = G.beneficiaries(gl, tg, absent, POS, fp_per_min=fpm)
    assert set(b["out_player_id"]) == {1} and b.iloc[0]["player_id"] in (5, 6, 2)
    assert (b["gain_mpg"] > 0.25).all() and b["gain_fppg"].notna().all()
    assert b["method"].str.contains("with/without").any()       # the star missed 6 games, so observed data blends in
    assert G.beneficiaries(gl, tg, absent.iloc[0:0], POS).empty


def test_signals_ignore_everything_after_as_of():
    tables = make_tables()
    cfg = load_league()
    as_of = date_after_games(tables, 20)
    gl, tg = season_to_date(tables, SEASON, as_of, cfg["scoring"])
    base = (G.trend_signals(gl), G.absent_players(gl, tg))
    t = {k: v.copy(deep=True) for k, v in tables.items()}
    fut = (t["game_logs"]["season"] == SEASON) & (t["game_logs"]["game_date"] > as_of)
    t["game_logs"].loc[fut, ["min", "fga", "tov"]] = 99
    t["game_logs"] = t["game_logs"].drop(index=t["game_logs"][fut].index[::2])
    tg_fut = (t["team_games"]["season"] == SEASON) & (t["team_games"]["game_date"] > as_of)
    t["team_games"] = t["team_games"].drop(index=t["team_games"][tg_fut].index[::2])
    gl2, tg2 = season_to_date(t, SEASON, as_of, cfg["scoring"])
    pd.testing.assert_frame_equal(base[0], G.trend_signals(gl2))
    pd.testing.assert_frame_equal(base[1], G.absent_players(gl2, tg2))


def test_event_study_on_the_synthetic_league_runs_and_summarises():
    tables = make_tables()
    ev = G.evaluate_rule(tables, [SEASON], min_star_mpg=20.0, min_out=3)
    if ev.empty:
        pytest.skip("synthetic league had no qualifying absences")
    s = G.summarize_rule(ev)
    assert s["pairs"] == len(ev) and np.isfinite(s["mean_actual_all"])
    assert G.summarize_rule(pd.DataFrame()) == {}
