"""Offseason feature layer: leakage slicing, per-event tables, cohort z-scores, design matrix, the fit gate."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features import offseason as fo
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]


def line(pid, ev, ctx, gid, d, minutes, *, team=1, pts=10, reb=4, ast=2, stl=1, blk=0, tov=1, fga=8, fta=2, fgm=4, ftm=1, fg3m=1, **kw):
    """One offseason log row with a consistent box line (pts = 2*fgm + fg3m + ftm needs fgm=4,fg3m=1,ftm=1 -> 10)."""
    y = int(ev[:4])
    return dict(season=f"{y - 1}-{y % 100:02d}", event_season=ev, context=ctx, game_id=gid, game_date=pd.Timestamp(d),
                player_id=pid, player_name=f"P{pid}", team_id=team, team_abbr="TT", matchup="TT vs. ZZ", min=minutes,
                fgm=fgm, fga=fga, fg3m=fg3m, fg3a=3, ftm=ftm, fta=fta, oreb=1, dreb=reb - 1, reb=reb, ast=ast, stl=stl,
                blk=blk, tov=tov, pf=2, pts=pts, plus_minus=0.0)


def logs_of(rows):
    return pd.DataFrame(rows)


def team_games_of(rows):
    d = pd.DataFrame(rows)
    return (d.groupby(["event_season", "context", "game_id", "game_date", "team_id"]).size().rename("n").reset_index()
            .drop(columns="n"))


# --------------------------------------------------------------------------- leakage

def test_slice_keeps_only_events_that_precede_the_target_season():
    rows = [line(1, "2018-19", "summer_league", "a", "2018-07-10", 20),      # tag 2017-18: known before 2018-19 starts
            line(1, "2019-20", "summer_league", "b", "2019-07-10", 20),      # tag 2018-19: NOT known for 2018-19
            line(1, "2019-20", "preseason", "c", "2019-10-05", 20)]
    df = logs_of(rows)
    kept = fo.slice_offseason(df, "2018-19")
    assert set(kept["event_season"]) == {"2018-19"}
    assert set(fo.slice_offseason(df, "2019-20")["event_season"]) == {"2018-19", "2019-20"}
    fo.assert_offseason_no_future(kept, "2018-19")
    with pytest.raises(AssertionError, match="leakage"):
        fo.assert_offseason_no_future(df, "2018-19")
    assert fo.slice_offseason(df.iloc[0:0], "2018-19").empty


# --------------------------------------------------------------------------- per-event table

def _cohort_logs():
    """A cohort of 6 players (45 minutes each, above the cohort threshold) with distinct fp36; player 9 plays only
    10 minutes, so he is scored against the cohort without being part of it."""
    rows = []
    for i, pid in enumerate(range(1, 7)):
        rows.append(line(pid, "2018-19", "summer_league", f"g{pid}", "2018-07-10", 45, pts=8 + 2 * i, fgm=3 + i, fg3m=0,
                         ftm=2 - min(i, 2), fga=6 + i, fta=1 + i % 3, tov=1 + i % 2, ast=1 + i, reb=3 + (i * 2) % 5,
                         stl=i % 3, blk=(i + 1) % 2))
    rows.append(line(9, "2018-19", "summer_league", "g9", "2018-07-10", 10, pts=10))
    return logs_of(rows)


def test_event_table_matches_hand_computed_fp36():
    logs = _cohort_logs()
    t = fo.event_player_table(logs, None, SCORING).set_index("player_id")
    r = logs[logs["player_id"] == 3].iloc[0]
    fp = float(fantasy_points_frame(pd.DataFrame([r]), SCORING).iloc[0])
    assert t.at[3, "sl_fp36"] == pytest.approx(fp / 45 * 36)
    assert t.at[3, "sl_minutes"] == 45 and t.at[3, "sl_gp"] == 1 and t.at[3, "sl_mpg"] == 45
    assert t["pre_minutes"].isna().all()                          # no preseason in this table at all


def test_cohort_z_is_the_minutes_weighted_standardisation_of_the_cohort():
    logs = _cohort_logs()
    t = fo.event_player_table(logs, None, SCORING)
    cohort = t[t["sl_minutes"] >= fo.MIN_COHORT_MINUTES]
    w = cohort["sl_minutes"].to_numpy(float)
    mu = np.average(cohort["sl_fp36"], weights=w)
    sd = np.sqrt(np.average((cohort["sl_fp36"] - mu) ** 2, weights=w))
    expected = ((t["sl_fp36"] - mu) / sd).clip(-fo.Z_CLIP, fo.Z_CLIP)
    np.testing.assert_allclose(t["sl_z"].to_numpy(float), expected.to_numpy(float))
    small = t[t["player_id"] == 9]                                   # 10 minutes: scored against the cohort, not part of it
    assert small["sl_z"].notna().all()
    assert cohort["sl_z"].abs().max() <= fo.Z_CLIP


def test_a_tiny_cohort_gives_no_z_scores():
    t = fo.event_player_table(logs_of([line(1, "2018-19", "summer_league", "g", "2018-07-10", 30)]), None, SCORING)
    assert t["sl_z"].isna().all() and t["sl_fp36"].notna().all()


def test_games_played_share_uses_the_teams_own_schedule():
    rows = [line(1, "2018-19", "summer_league", g, d, 20, team=1) for g, d in (("a", "2018-07-10"), ("b", "2018-07-12"))]
    rows += [line(2, "2018-19", "summer_league", "a", "2018-07-10", 20, team=1)]
    rows += [line(3, "2018-19", "summer_league", "c", "2018-07-11", 20, team=2)]   # team 2 plays one game
    df = logs_of(rows)
    t = fo.event_player_table(df, team_games_of(rows), SCORING).set_index("player_id")
    assert t.at[1, "sl_gp_share"] == 1.0 and t.at[2, "sl_gp_share"] == 0.5 and t.at[3, "sl_gp_share"] == 1.0


def test_contexts_are_kept_apart_and_can_be_selected():
    rows = [line(1, "2018-19", "summer_league", "a", "2018-07-10", 20), line(1, "2018-19", "preseason", "b", "2018-10-05", 12)]
    df = logs_of(rows)
    both = fo.event_player_table(df, None, SCORING).iloc[0]
    assert both["sl_minutes"] == 20 and both["pre_minutes"] == 12
    only_sl = fo.event_player_table(df, None, SCORING, contexts=(fo.SUMMER_LEAGUE,)).iloc[0]
    assert only_sl["sl_minutes"] == 20 and np.isnan(only_sl["pre_minutes"])
    only_pre = fo.event_player_table(df, None, SCORING, contexts=(fo.PRESEASON,)).iloc[0]
    assert np.isnan(only_pre["sl_minutes"]) and only_pre["pre_minutes"] == 12


def test_as_of_cuts_preseason_games_but_never_summer_league():
    rows = [line(1, "2018-19", "summer_league", "s", "2018-07-20", 20),
            line(1, "2018-19", "preseason", "p1", "2018-10-01", 10), line(1, "2018-19", "preseason", "p2", "2018-10-12", 14)]
    t = fo.event_player_table(logs_of(rows), None, SCORING, as_of=date(2018, 10, 5)).iloc[0]
    assert t["sl_minutes"] == 20 and t["pre_minutes"] == 10 and t["pre_gp"] == 1
    t0 = fo.event_player_table(logs_of(rows), None, SCORING, as_of=date(2018, 9, 1)).iloc[0]
    assert t0["sl_minutes"] == 20 and np.isnan(t0["pre_minutes"])


def test_preseason_fraction_keeps_the_first_share_of_game_dates():
    rows = [line(1, "2018-19", "preseason", f"p{k}", f"2018-10-{2 + k:02d}", 10 + k) for k in range(4)]
    rows.append(line(1, "2018-19", "summer_league", "s", "2018-07-10", 20))
    df = logs_of(rows)
    half = fo.event_player_table(df, None, SCORING, preseason_fraction=0.5).iloc[0]
    assert half["pre_gp"] == 2 and half["sl_minutes"] == 20            # the first 2 of 4 dates; Summer League untouched
    assert fo.event_player_table(df, None, SCORING, preseason_fraction=1.0).iloc[0]["pre_gp"] == 4
    assert fo.event_player_table(df, None, SCORING, preseason_fraction=0.01).iloc[0]["pre_gp"] == 1   # at least one date


def test_component_z_scores_are_present_and_bounded():
    t = fo.event_player_table(_cohort_logs(), None, SCORING)
    for c in fo.COMPONENTS:
        z = t[f"sl_{c}_z"]
        assert z.notna().any() and z.abs().max() <= fo.Z_CLIP + 1e-9


# --------------------------------------------------------------------------- design matrix

def test_youth_weight_is_linear_between_20_and_26_and_zero_for_unknown_age():
    y = fo.youth_weight(np.array([19.0, 20.0, 23.0, 26.0, 30.0, np.nan]))
    np.testing.assert_allclose(y, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0])


def _events(minutes_sl, z_sl, mpg_sl, minutes_pre=np.nan, z_pre=np.nan, mpg_pre=np.nan):
    return pd.DataFrame({"sl_minutes": [minutes_sl], "sl_z": [z_sl], "sl_mpg": [mpg_sl],
                         "pre_minutes": [minutes_pre], "pre_z": [z_pre], "pre_mpg": [mpg_pre]})


def test_design_width_matches_the_feature_names():
    ev = _events(80, 1.0, 26.0, 60, 0.5, 18.0)
    for comps in (False, True):
        cols = fo.event_columns(comps)
        full = pd.concat([ev, pd.DataFrame({c: [0.3] for c in cols if c not in ev.columns})], axis=1)
        X = fo.design(full, np.array([21.0]), np.array([1.0]), 120.0, comps)
        assert X.shape == (1, len(fo.feature_names(comps)))


def test_design_gives_zeros_for_a_missing_block_and_shrinks_by_minutes():
    ev = _events(80, 2.0, 30.0)                                        # summer league only
    X = fo.design(ev, np.array([20.0]), np.array([1.0]), 120.0)
    names = fo.feature_names()
    row = dict(zip(names, X[0]))
    assert row["has_sl"] == 1.0 and row["has_pre"] == 0.0
    assert row["pre_z"] == row["pre_mpg"] == row["pre_z_youth"] == 0.0
    assert row["sl_z"] == pytest.approx(2.0 * 80 / (80 + 120))         # shrunk by minutes / (minutes + K)
    assert row["sl_z_youth"] == pytest.approx(row["sl_z"] * 1.0)       # age 20 -> full youth weight
    assert row["sl_z_rookie"] == pytest.approx(row["sl_z"])
    assert row["sl_mpg"] == pytest.approx((30 - 25) / 10)
    vet = dict(zip(names, fo.design(ev, np.array([30.0]), np.array([0.0]), 120.0)[0]))
    assert vet["sl_z_youth"] == 0.0 and vet["sl_z_rookie"] == 0.0 and vet["sl_z"] == row["sl_z"]
    more_k = dict(zip(names, fo.design(ev, np.array([20.0]), np.array([1.0]), 480.0)[0]))
    assert abs(more_k["sl_z"]) < abs(row["sl_z"])                      # a larger K trusts a short line less


# --------------------------------------------------------------------------- fit and gate

def _training(n_per_season=120, seasons=(2015, 2016, 2017, 2018), slope=1.5, noise=1.0, seed=0, planted=True):
    rng = np.random.default_rng(seed)
    n = n_per_season * len(seasons)
    season = np.repeat(seasons, n_per_season)
    z = rng.normal(size=n)
    minutes = rng.uniform(40, 200, n)
    age = rng.uniform(19, 30, n)
    rookie = (age < 21).astype(float)
    ev = pd.DataFrame({"sl_minutes": np.nan, "sl_z": np.nan, "sl_mpg": np.nan,
                       "pre_minutes": minutes, "pre_z": z, "pre_mpg": rng.uniform(8, 30, n)})
    shrunk = z * minutes / (minutes + 120.0)
    resid = (slope * shrunk if planted else 0.0) + rng.normal(0, noise, n)
    return ev, age, rookie, resid, np.full(n, 40.0), season


def test_fit_finds_a_planted_effect_and_enables_the_adjustment():
    ev, age, rookie, resid, w, s = _training()
    fit = fo.fit_adjustment(ev, age, rookie, resid, w, s)
    assert fit.enabled and fit.diagnostics["cv_gain_vs_zero"] > fo.MIN_CV_GAIN
    adj = fit.adjustment(ev, age, rookie)
    assert np.corrcoef(adj, resid)[0, 1] > 0.5
    assert fit.diagnostics["coefficients"]["pre_z"] > 0


def test_fit_switches_itself_off_when_there_is_no_signal():
    ev, age, rookie, resid, w, s = _training(planted=False)
    fit = fo.fit_adjustment(ev, age, rookie, resid, w, s)
    assert not fit.enabled and "below" in fit.diagnostics["reason"]
    assert (fit.adjustment(ev, age, rookie) == 0).all()


def test_fit_refuses_too_few_rows_or_seasons():
    ev, age, rookie, resid, w, s = _training(n_per_season=20, seasons=(2015, 2016))
    assert not fo.fit_adjustment(ev, age, rookie, resid, w, s).enabled
    ev, age, rookie, resid, w, s = _training(n_per_season=300, seasons=(2015,))
    fit = fo.fit_adjustment(ev, age, rookie, resid, w, s)
    assert not fit.enabled and "too few" in fit.diagnostics["reason"]


def test_rows_without_any_event_get_exactly_zero_adjustment():
    ev, age, rookie, resid, w, s = _training()
    fit = fo.fit_adjustment(ev, age, rookie, resid, w, s)
    none = pd.DataFrame({c: [np.nan] for c in ev.columns})
    assert fit.adjustment(none, np.array([21.0]), np.array([1.0]))[0] == 0.0


def test_fit_with_components_uses_the_wider_design():
    ev, age, rookie, resid, w, s = _training()
    for prefix in ("sl", "pre"):
        for c in fo.COMPONENTS:
            ev[f"{prefix}_{c}_z"] = np.random.default_rng(1).normal(size=len(ev))
    fit = fo.fit_adjustment(ev, age, rookie, resid, w, s, components=True)
    assert fit.components is True
    if fit.enabled:
        assert len(fit.coefs) == 1 + len(fo.feature_names(True))


def test_fit_is_deterministic():
    a = fo.fit_adjustment(*_training())
    b = fo.fit_adjustment(*_training())
    np.testing.assert_array_equal(a.coefs, b.coefs)


# --------------------------------------------------------------------------- applying an adjustment

def test_factor_scales_projection_to_projection_plus_adjustment_and_is_clipped():
    p = np.array([20.0, 20.0, 20.0, 0.5, 20.0])
    adj = np.array([4.0, -4.0, 400.0, 3.0, 0.0])
    f = fo.factor_from_adjustment(p, adj)
    assert f[0] == pytest.approx(1.2) and f[1] == pytest.approx(0.8)
    assert f[2] == fo.FACTOR_HI                                  # a wild adjustment is clipped
    assert f[3] == 1.0                                           # a near-zero projection cannot be scaled
    assert f[4] == 1.0
    assert (fo.factor_from_adjustment(p, -1000 * np.ones(5)) >= fo.FACTOR_LO - 1e-12).all()
