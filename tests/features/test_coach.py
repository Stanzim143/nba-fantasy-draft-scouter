"""Coach features: style from box scores, the coach table, the shift a coaching change implies, and the leakage guard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coach_testkit import T1, T2, bio_for, coach_table, logs, season_games, two_team_history  # noqa: E402

from src.features import coach as C  # noqa: E402

def test_team_style_measures_concentration_and_depth():
    heavy = season_games("2019-20", T1, 25, [38, 36, 34, 32, 30, 12, 8, 6, 4, 2])
    deep = season_games("2019-20", T2, 25, [28, 27, 26, 25, 24, 22, 20, 18, 12, 8])
    gl = logs(heavy + deep)
    ts = C.team_style(gl, bio_for(gl)).set_index("team_id")
    assert ts.loc[T1, "star"] == 38 and ts.loc[T2, "star"] == 28
    assert ts.loc[T1, "top5"] == 34 and ts.loc[T2, "top5"] == 26
    assert ts.loc[T1, "depth10"] == 6 and ts.loc[T2, "depth10"] == 9
    assert abs(ts.loc[T1, "three"] - 0.4) < 1e-9
    assert abs(ts.loc[T1, "pace"] - (10 + 0.44 * 2 - 1 + 2) * 10) < 1e-6      # ten players' totals per game
    assert abs(ts["star_x"].sum()) < 1e-9                                       # league-relative: sums to zero within a season


def test_youth_and_veteran_minute_shares():
    gl = logs(season_games("2019-20", T1, 25, [30, 30, 30, 30, 30]))
    bio = bio_for(gl)
    bio["age_at_season_start"] = [21.0, 22.0, 27.0, 32.0, 35.0]
    ts = C.team_style(gl, bio)
    assert abs(ts.loc[0, "young"] - 0.4) < 1e-9 and abs(ts.loc[0, "old"] - 0.4) < 1e-9


def test_short_team_seasons_and_thin_boxscores_are_ignored():
    gl = logs(season_games("2019-20", T1, 5, [30, 30, 30, 30, 30]) + season_games("2019-20", T2, 25, [30, 20, 10]))
    assert C.team_style(gl, bio_for(gl)).empty


def test_opening_coaches_and_the_single_flag():
    t = coach_table([("2019-20", T1, [("a", ""), ("b", "interim")]), ("2020-21", T1, [("c", "")])])
    oc = C.opening_coaches(t).set_index("season")
    assert oc.loc["2019-20", "coach_key"] == "a" and not oc.loc["2019-20", "single"] and oc.loc["2020-21", "single"]
    assert list(C.opening_coaches(t, through_start=2019)["season"]) == ["2019-20"]


def test_a_coaching_change_shifts_style_toward_the_new_coachs_history():
    h, coaches = two_team_history()
    ctx = C.CoachContext.build(h, coaches)
    assert ctx.is_new(T2, 2021) and ctx.is_new(T1, 2021) and not ctx.is_new(T2, 2020)
    sh = ctx.shift(T2, 2021)                                    # the 'star' coach arrives at the deep team
    assert sh["star"] > 0 and sh["top5"] > 0 and sh["depth10"] < 0
    damp = 3 / (3 + C.PRIOR_SEASONS_K)                          # three earlier seasons behind him
    assert abs(sh["star"] - damp * (ctx.prior[("star", 2021)]["star"] - ctx.prev_style[(T2, 2021)]["star"])) < 1e-9
    assert all(v == 0.0 for v in ctx.shift(T1, 2021).values())  # a first-time coach has no history: no style shift, only the flag
    assert ctx.debut_coach(T1, 2021) and not ctx.debut_coach(T2, 2021)


def test_an_unchanged_coach_means_no_shift_and_a_mid_season_change_is_not_a_prior():
    h, coaches = two_team_history()
    ctx = C.CoachContext.build(h, coaches)
    assert all(v == 0.0 for v in ctx.shift(T1, 2020).values())
    mid = coach_table([("2018-19", T1, [("star", ""), ("x", "interim")])])
    both = pd.concat([coaches[coaches["season"] != "2018-19"], mid])
    ctx2 = C.CoachContext.build(h, both)
    assert ctx2.prior[("star", 2021)]["n"] == 2                 # 2018-19 had a mid-season change, so it does not count toward his prior


def test_the_target_season_and_later_never_leak_into_a_prior():
    h, coaches = two_team_history()
    ctx = C.CoachContext.build(h, coaches)
    future = coach_table([("2022-23", T1, [("star", "")]), ("2023-24", T2, [("newbie", "")])])
    ctx2 = C.CoachContext.build(h, pd.concat([coaches, future], ignore_index=True))
    assert ctx.prior == ctx2.prior and ctx.shift(T2, 2021) == ctx2.shift(T2, 2021)
    assert not any(s > 2021 for (_, s) in ctx2.coach_of)


def feats(h, coaches, **kw):
    return C.CoachFeatures.build_context(h, coaches, None, **kw)


def test_features_reach_high_minute_players_more_and_zero_when_the_coach_is_unchanged():
    h, coaches = two_team_history()
    f = feats(h, coaches)
    star_pid, bench_pid = T2 * 100 + 0, T2 * 100 + 9              # last season: 28 mpg (T2's top player) and 8 mpg
    X = f.features(np.array([star_pid, bench_pid]), np.array([2021, 2021]))
    assert X[0, -1] == 1.0 and X[1, -1] == 1.0                    # both are on a team with a coaching change
    assert X[0, 0] > 0 and X[1, 0] == 0.0                         # only the high-minute player gets the top-5 shift
    assert X[1, 1] > 0                                            # the bench player gets the 'low' interaction
    same = f.features(np.array([T1 * 100]), np.array([2020]))
    assert (same == 0).all()
    assert (f.features(np.array([99999]), np.array([2021])) == 0).all()      # no team: zeros, never NaN


def test_live_roster_overrides_the_team_only_in_the_target_season():
    h, coaches = two_team_history()
    p = T1 * 100                                                    # was on T1 (a first-time coach arrives there)
    base = feats(h, coaches).features(np.array([p]), np.array([2021]))
    moved = feats(h, coaches, live_team={p: T2}).features(np.array([p]), np.array([2021]))
    assert base[0, -1] == 1.0 and moved[0, 2] != base[0, 2]        # now judged by T2's coaching change (star coach arriving)
    past = feats(h, coaches, live_team={p: T2}).features(np.array([p]), np.array([2020]))
    assert (past == 0).all()


def test_fit_recovers_a_planted_effect_and_build_never_applies_the_intercept():
    h, coaches = two_team_history()
    f = feats(h, coaches)
    rng = np.random.default_rng(0)
    pids = np.tile(np.array([T2 * 100 + i for i in range(10)]), 20)
    s = np.full(len(pids), 2021)
    X = f.features(pids, s)
    resid = -1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.05, len(pids))          # global bias -1, plus 2.0 x the top-5 interaction
    est = np.full(len(pids), 20.0)
    f.fit(pids, s, est, est + resid, np.full(len(pids), 50.0))
    adj = f.build(pids, s)
    hi, lo = pids == T2 * 100, pids == T2 * 100 + 9          # a single team-season is collinear, so test the identified quantity:
    diff = adj[hi].mean() - adj[lo].mean()                   # the difference between two players' adjustments
    assert abs(diff - 2.0 * (X[hi, 0].mean() - X[lo, 0].mean())) < 0.05
    assert np.abs(adj).max() <= C.MPG_SHIFT_CAP
    assert np.allclose(adj, np.clip(X @ f.beta[1:], -C.MPG_SHIFT_CAP, C.MPG_SHIFT_CAP))    # the intercept is fitted, never applied


def test_too_few_rows_is_a_no_op():
    h, coaches = two_team_history()
    f = feats(h, coaches)
    f.fit(np.array([201]), np.array([2021]), np.array([20.0]), np.array([25.0]), np.array([50.0]))
    assert (f.beta == 0).all() and (f.build(np.array([201]), np.array([2021])) == 0).all()
