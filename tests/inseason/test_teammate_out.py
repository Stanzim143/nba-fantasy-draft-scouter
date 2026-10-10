"""Teammate-out adjustment and its evaluation panels: behaviour on hand-built logs, and leakage."""
import numpy as np
import pandas as pd
import pytest
from ise_testkit import SEASON, date_after_games, make_tables

from src.contracts import HISTORY_TABLES, History
from src.inseason import teammate_out as T
from src.inseason import teammate_out_eval as E
from src.inseason.schedule import from_team_games
from src.models.registry import get_projector
from src.value.frame import fantasy_points_frame
from src.value.league import load_league
from src.value.positions import position_group

TEAM = 1610612737


def _logs(spec, n=20):
    rows, tg = [], []
    for g in range(n):
        gid, day = f"g{g:02d}", pd.Timestamp("2023-11-01") + pd.Timedelta(days=g)
        tg.append({"season": SEASON, "game_id": gid, "game_date": day, "team_id": TEAM, "team_abbr": "ATL",
                   "is_home": True, "pts_for": 100, "pts_against": 99})
        for pid, f in spec.items():
            m = f(g)
            if m > 0:
                rows.append({"season": SEASON, "game_id": gid, "game_date": day, "player_id": pid, "team_id": TEAM,
                             "min": float(m), "fga": 9, "fta": 3, "tov": 2, "fp": float(m)})
    return pd.DataFrame(rows), pd.DataFrame(tg)


SPEC = {1: lambda g: 30 if g % 2 == 0 else 0,    # star, plays every other game (share present 0.5)
        2: lambda g: 15, 3: lambda g: 25, 4: lambda g: 28, 5: lambda g: 6}
POS = pd.Series({1: "G", 2: "G", 3: "F", 4: "C", 5: "G"})


def _ros(gl):
    per = gl.groupby("player_id").agg(ros_mpg=("min", "mean"), ros_fppg=("fp", "mean")).reset_index()
    per["ros_fppg"] *= 1.2                                     # 1.2 FP per minute
    return per


def test_status_absences_keeps_only_rotation_players_listed_out():
    gl, _ = _logs(SPEC)
    a = T.status_absences(gl, pd.Series({4: "OUT", 3: "QUESTIONABLE", 5: "OUT", 99: "OUT"}))
    assert list(a["player_id"]) == [4] and a["source"].iloc[0] == "status OUT"
    assert T.status_absences(gl, pd.Series(dtype=object)).empty


def test_share_present_counts_games_since_joining():
    gl, tg = _logs(SPEC)
    assert T.share_present(gl, tg, 1, TEAM) == pytest.approx(0.5)
    assert T.share_present(gl, tg, 3, TEAM) == pytest.approx(1.0)
    assert T.share_present(gl, tg, 42, TEAM) == 0.0


def test_adjustment_converts_minutes_to_fp_and_scales_by_share_present():
    gl, tg = _logs(SPEC)
    ros = _ros(gl)
    absent = T.status_absences(gl, pd.Series({4: "OUT"}))
    adj = T.teammate_out_adjustment(gl, tg, absent, POS, ros).set_index("player_id")
    assert 4 not in adj.index and set(adj.index) <= {1, 2, 3, 5}
    assert (adj["adj_fppg"] > 0).all()
    assert np.allclose(adj["adj_fppg"], adj["gain_mpg"] * 1.2)
    assert adj["gain_mpg"].sum() <= 28 + 1e-9
    half = T.teammate_out_adjustment(gl, tg, absent, POS, ros, beta=0.5).set_index("player_id")
    assert np.allclose(half["adj_fppg"], adj["adj_fppg"] * 0.5)
    # the half-time star is out: his minutes are only half missing from the season averages
    ab1 = T.status_absences(gl, pd.Series({1: "OUT"}))
    one = T.teammate_out_adjustment(gl, tg, ab1, POS, ros)
    assert one["gain_mpg"].sum() <= 0.5 * 30 + 1e-9
    assert T.teammate_out_adjustment(gl, tg, ab1.iloc[0:0], POS, ros).empty


def test_gain_is_capped_at_36_minutes():
    gl, tg = _logs({1: lambda g: 34, 2: lambda g: 35, 3: lambda g: 33})
    adj = T.teammate_out_adjustment(gl, tg, T.status_absences(gl, pd.Series({1: "OUT"})), POS, _ros(gl))
    r = adj.set_index("player_id")
    assert r.at[2, "gain_mpg"] <= 1 + 1e-9 and r.at[3, "gain_mpg"] <= 3 + 1e-9


# ---------------------------------------------------------------- leakage

@pytest.fixture(scope="module")
def world():
    tables = make_tables()
    cfg = load_league()
    prior = get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))
    return tables, cfg, prior


def _season(tables, cfg):
    gl = tables["game_logs"][tables["game_logs"]["season"] == SEASON].copy()
    gl["fp"] = fantasy_points_frame(gl, cfg["scoring"])
    gl["game_date"] = pd.to_datetime(gl["game_date"])
    tg = tables["team_games"][tables["team_games"]["season"] == SEASON].copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    return {"game_logs": gl, "team_games": tg, "players": tables["players"],
            "player_season_bio": tables["player_season_bio"]}


def _rows(tables, cfg, prior, day, statuses):
    s = _season(tables, cfg)
    pos = tables["players"].drop_duplicates("player_id").set_index("player_id")["position"]
    return E.forecast_rows(s, SEASON, day, prior=prior, cfg=cfg, schedule=from_team_games(tables["team_games"], SEASON),
                           season_games=40.0, positions=pos, statuses=statuses)


def _wreck(tables, day):
    """Stats of every game on or after ``day`` multiplied; later games partly dropped; ``day``'s rows kept present."""
    t = {k: v.copy(deep=True) for k, v in tables.items()}
    gl = t["game_logs"]
    d = pd.to_datetime(gl["game_date"])
    hit = (gl["season"] == SEASON) & (d >= day)
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m", "min"):
        gl.loc[hit, c] = gl.loc[hit, c] * 5 + 2
    later = gl[(gl["season"] == SEASON) & (d > day)].index[::2]
    t["game_logs"] = gl.drop(index=later)
    tg = t["team_games"]
    t["team_games"] = tg.drop(index=tg[(tg["season"] == SEASON) & (pd.to_datetime(tg["game_date"]) > day)].index[::3])
    return t


def test_forecast_features_never_read_the_target_day_or_later(world):
    tables, cfg, prior = world
    day = date_after_games(tables, 25)
    gl = tables["game_logs"]
    some = gl[(gl["season"] == SEASON) & (pd.to_datetime(gl["game_date"]) < day)].groupby("player_id")["min"].mean()
    statuses = pd.Series("OUT", index=some.sort_values().index[-6:])     # a few heavy-minute players listed OUT
    a = _rows(tables, cfg, prior, day, statuses)
    b = _rows(_wreck(tables, day), cfg, prior, day, statuses)
    assert len(a) and (a["x_G"] > 0).any() and (a["x_R"] > 0).any()
    feats = ["season", "date", "team_id", "player_id", "ros_fppg", "ros_mpg", "x_G", "x_R", "n_out_G", "n_out_R"]
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert not np.allclose(a["fp"].to_numpy(), b["fp"].to_numpy())        # only the target moved


def test_study_baselines_use_earlier_games_only(world):
    tables, cfg, _ = world
    day = date_after_games(tables, 25)
    groups = pd.Series({p: position_group(x) for p, x in
                        tables["players"].drop_duplicates("player_id").set_index("player_id")["position"].items()})
    s0 = _season(tables, cfg)
    s1 = _season(_wreck(tables, day), cfg)
    a = E.study_panel(s0["game_logs"], s0["team_games"], groups)
    b = E.study_panel(s1["game_logs"], s1["team_games"], groups)
    cols = ["team_id", "player_id", "game_id", "base_min", "base_fp", "miss_min", "miss_same", "miss_usg_share", "b2b"]
    key = ["team_id", "player_id", "game_id"]
    on_day_a = a[a["game_date"] <= day].sort_values(key).reset_index(drop=True)
    on_day_b = b[b["game_date"] <= day].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(on_day_a[cols], on_day_b[cols])
