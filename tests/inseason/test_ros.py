"""Rest-of-season projection: shrinkage behaviour, games remaining, value, and (above all) leakage."""
import numpy as np
import pandas as pd
import pytest
from ise_testkit import SEASON, date_after_games, make_tables

from src.contracts import HISTORY_TABLES, History
from src.inseason.ros import MODES, RosParams, build_ros
from src.inseason.schedule import from_team_games
from src.models.registry import get_projector
from src.value.league import load_league

CFG = load_league()


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def prior(tables):
    return get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))


@pytest.fixture(scope="module")
def as_of(tables):
    return date_after_games(tables, 20)


def _ros(tables, prior, as_of, **kw):
    return build_ros(tables, SEASON, as_of, prior=prior, cfg=CFG, season_games=40, **kw)


def test_before_the_season_it_is_the_preseason_projection(tables, prior):
    ros = _ros(tables, prior, pd.Timestamp("2023-09-01"), params=RosParams())
    assert (ros["gp"] == 0).all() and (ros["w_sample"] == 0).all()
    p = prior.set_index("player_id")
    r = ros.set_index("player_id").loc[p.index]
    np.testing.assert_allclose(r["ros_fppg"], p["proj_fppg"], rtol=1e-9)
    np.testing.assert_allclose(r["avail"], np.clip(p["proj_gp"] / 40, 0, 1), rtol=1e-9)
    np.testing.assert_allclose(r["ros_games"], r["avail"] * 40, rtol=1e-9)     # all 40 games remain


def test_prior_mode_ignores_the_season_to_date(tables, prior, as_of):
    early = _ros(tables, prior, pd.Timestamp("2023-09-01"), params=RosParams(mode="prior"))
    late = _ros(tables, prior, as_of, params=RosParams(mode="prior"))
    a, b = early.set_index("player_id"), late.set_index("player_id")
    common = a.index.intersection(b.index)
    np.testing.assert_allclose(a.loc[common, "ros_fppg"], b.loc[common, "ros_fppg"])


def test_blend_lies_between_prior_and_sample_for_every_player(tables, prior, as_of):
    ros = _ros(tables, prior, as_of, params=RosParams()).set_index("player_id")
    played = ros[(ros["gp"] >= 3) & ros["has_prior"]]
    assert len(played) > 30
    lo = np.minimum(played["prior_mpg"], played["mpg"]) - 1e-9
    hi = np.maximum(played["prior_mpg"], played["mpg"]) + 1e-9
    assert ((played["ros_mpg"] >= lo) & (played["ros_mpg"] <= hi)).all()
    fp = _ros(tables, prior, as_of, params=RosParams(mode="blend_fppg")).set_index("player_id").loc[played.index]
    lo = np.minimum(fp["prior_fppg"], fp["fppg_to_date"]) - 1e-9
    hi = np.maximum(fp["prior_fppg"], fp["fppg_to_date"]) + 1e-9
    assert ((fp["ros_fppg"] >= lo) & (fp["ros_fppg"] <= hi)).all()


def test_sample_weight_grows_with_games_and_pseudo_counts_move_the_answer(tables, prior):
    a = _ros(tables, prior, date_after_games(tables, 10), params=RosParams()).set_index("player_id")
    b = _ros(tables, prior, date_after_games(tables, 30), params=RosParams()).set_index("player_id")
    common = a.index[(a["gp"] > 0)].intersection(b.index)
    assert (b.loc[common, "w_sample"] >= a.loc[common, "w_sample"] - 1e-12).all()
    heavy_prior = _ros(tables, prior, date_after_games(tables, 30),
                       params=RosParams(minutes_pseudo=1e9, games_pseudo=1e9, avail_pseudo=1e9)).set_index("player_id")
    pure = _ros(tables, prior, date_after_games(tables, 30), params=RosParams(mode="prior")).set_index("player_id")
    np.testing.assert_allclose(heavy_prior["ros_fppg"], pure["ros_fppg"], rtol=1e-5)
    np.testing.assert_allclose(heavy_prior["avail"], pure["avail"], rtol=1e-5)
    light = _ros(tables, prior, date_after_games(tables, 30),
                 params=RosParams(minutes_pseudo=1e-6, games_pseudo=1e-6, avail_pseudo=1e-6)).set_index("player_id")
    sample = _ros(tables, prior, date_after_games(tables, 30), params=RosParams(mode="sample")).set_index("player_id")
    ok = light[light["gp"] >= 5].index
    np.testing.assert_allclose(light.loc[ok, "ros_fppg"], sample.loc[ok, "ros_fppg"], rtol=1e-3)


def test_games_remaining_from_a_schedule_match_the_calendar(tables, prior, as_of):
    sch = from_team_games(tables["team_games"], SEASON)
    with_sched = _ros(tables, prior, as_of, params=RosParams(), schedule=sch)
    without = _ros(tables, prior, as_of, params=RosParams())
    tg = tables["team_games"]
    left = tg[(tg["season"] == SEASON) & (tg["game_date"] > as_of)].groupby("team_id").size()
    r = with_sched.dropna(subset=["team_id"]).drop_duplicates("team_id")
    r = pd.Series(r["team_games_left"].to_numpy(), index=r["team_id"].astype("int64"))
    common = left.index.intersection(r.index)
    assert len(common) >= 8 and (r[common] == left[common]).all()
    assert with_sched["team_games_left"].between(0, 40).all() and without["team_games_left"].between(0, 40).all()


def test_teams_with_no_games_left_have_zero_not_a_phantom_median(tables, prior):
    sch = from_team_games(tables["team_games"], SEASON)
    end = pd.Timestamp(tables["team_games"]["game_date"].max()) + pd.Timedelta(days=1)
    ros = _ros(tables, prior, end, params=RosParams(), schedule=sch)
    known = ros.dropna(subset=["team_id"])
    assert len(known) and (known["team_games_left"] == 0).all()
    assert (known["ros_games"] == 0).all()


def test_value_columns_and_ordering(tables, prior, as_of):
    ros = _ros(tables, prior, as_of, params=RosParams())
    assert ros["ros_rank"].tolist() == list(range(1, len(ros) + 1))
    assert ros["ros_vorp"].is_monotonic_decreasing
    assert ros.attrs["replacement"]["total"] > 0
    assert {"ros_total_fp", "ros_fppg", "ros_games", "avail", "prior_fppg", "w_sample", "has_prior"} <= set(ros.columns)
    np.testing.assert_allclose(ros["ros_total_fp"], ros["ros_fppg"] * ros["ros_games"], rtol=1e-9)
    assert not _ros(tables, prior, as_of, params=RosParams(), with_value=False).columns.isin(["ros_vorp"]).any()


def test_players_without_a_projection_get_a_weak_prior(tables, prior, as_of):
    thin = prior[prior["player_id"] != prior["player_id"].iloc[0]]
    missing = prior["player_id"].iloc[0]
    ros = _ros(tables, thin, as_of, params=RosParams()).set_index("player_id")
    if missing in ros.index:                              # he played this season, so he is still listed
        row = ros.loc[missing]
        assert not row["has_prior"] and np.isfinite(row["ros_fppg"])
    assert ros["has_prior"].sum() == len(thin)


def test_errors(tables, prior, as_of):
    with pytest.raises(ValueError, match="mode"):
        _ros(tables, prior, as_of, params=RosParams(mode="bogus"))
    with pytest.raises(ValueError, match="no preseason projection"):
        _ros(tables, prior.assign(season="2000-01"), as_of, params=RosParams())
    assert set(MODES) == {"blend", "blend_fppg", "prior", "sample"}


# ---------------------------------------------------------------- leakage: nothing after as_of may matter

def _poisoned(tables, as_of):
    """Copy of the tables with every game dated after ``as_of`` (this season) wrecked or removed."""
    t = {k: v.copy(deep=True) for k, v in tables.items()}
    gl, tg = t["game_logs"], t["team_games"]
    future = (gl["season"] == SEASON) & (gl["game_date"] > as_of)
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m", "min"):
        gl.loc[future, c] = gl.loc[future, c] * 7 + 3
    half = gl[future].index[::2]
    t["game_logs"] = gl.drop(index=half)
    fut_tg = (tg["season"] == SEASON) & (tg["game_date"] > as_of)
    t["team_games"] = tg.drop(index=tg[fut_tg].index[::3])
    return t


@pytest.mark.parametrize("mode", MODES)
def test_nothing_dated_after_as_of_changes_the_output(tables, prior, as_of, mode):
    base = _ros(tables, prior, as_of, params=RosParams(mode=mode), schedule=from_team_games(tables["team_games"], SEASON))
    poisoned = _ros(_poisoned(tables, as_of), prior, as_of, params=RosParams(mode=mode),
                    schedule=from_team_games(tables["team_games"], SEASON))
    pd.testing.assert_frame_equal(base, poisoned)


def test_the_whole_target_season_cannot_reach_the_preseason_prior(tables, as_of):
    """Without a precomputed prior, the projector is built from History.until: wrecking every target-season
    row must not change the prior, only (through the sample) the blend, so use the preseason as_of."""
    before = pd.Timestamp("2023-09-01")
    a = build_ros(tables, SEASON, before, cfg=CFG, params=RosParams())
    t = {k: v.copy(deep=True) for k, v in tables.items()}
    m = t["game_logs"]["season"] == SEASON
    t["game_logs"].loc[m, "pts"] = 999
    b = build_ros(t, SEASON, before, cfg=CFG, params=RosParams())
    pd.testing.assert_frame_equal(a, b)


def test_as_of_is_inclusive_of_that_days_games(tables, prior):
    d = date_after_games(tables, 15)
    a = _ros(tables, prior, d - pd.Timedelta(days=1), params=RosParams()).set_index("player_id")
    b = _ros(tables, prior, d, params=RosParams()).set_index("player_id")
    assert b["gp"].sum() > a["gp"].sum()
