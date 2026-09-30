"""Unit and statistical tests for the baseline's building blocks."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, add_player_seasons, history_for, make_league

from src.contracts import History, season_start
from src.models.age import FLAT, AgeCurve, consecutive_pairs, fit_age_curve
from src.models.availability import AvailabilityModel, availability_features
from src.models.baseline import BaselineProjector
from src.models.panel import AgeLookup, build_panel, season_lengths, target_season_games
from src.models.rookies import UNDRAFTED_PICK, fit_rookie_prior, pick_effective, rookie_design
from src.models.shrink import fit_kappa, shrink, wls
from src.synthetic import _age_delta, make_synthetic_tables
from src.value.league import load_league

SCORING = load_league()["scoring"]


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def fitted(tables):
    return BaselineProjector().fit(history_for(tables))


# --------------------------------------------------------------------------- shrinkage maths

def test_shrink_limits():
    prior, est = 10.0, 30.0
    assert shrink(est, prior, 0.0, 100.0) == prior                       # no data -> prior exactly
    assert shrink(est, prior, 1e9, 100.0) == pytest.approx(est, abs=1e-4)  # huge sample -> own rate
    assert shrink(est, prior, 100.0, 100.0) == pytest.approx(20.0)        # n == kappa -> halfway
    n = np.array([1, 10, 100, 1000, 10000.0])
    out = shrink(est, prior, n, 100.0)
    assert (np.diff(out) > 0).all() and (out > prior).all() and (out < est).all()


def test_fit_kappa_recovers_the_bayes_optimal_constant():
    """talent ~ N(prior, tau^2), own estimate = talent + N(0, s^2/n): optimal kappa = s^2 / tau^2."""
    rng = np.random.default_rng(0)
    n = rng.uniform(50, 3000, 20000)
    tau, s = 0.05, 1.5
    prior = np.full_like(n, 0.30)
    talent = prior + rng.normal(0, tau, n.size)
    est = talent + rng.normal(0, 1, n.size) * s / np.sqrt(n)
    y = talent + rng.normal(0, 0.01, n.size)                 # next-season outcome ~ talent
    kappa, loss, fitted = fit_kappa(est, prior, n, y, np.ones_like(n), default=1.0)
    optimum = s**2 / tau**2                                  # 900
    assert fitted
    assert 0.75 * optimum < kappa < 1.3 * optimum
    for k in (kappa / 3, kappa * 3):                         # and it is the loss minimum
        assert loss <= np.mean((y - shrink(est, prior, n, k)) ** 2) + 1e-12


def test_fit_kappa_falls_back_with_too_little_data():
    kappa, loss, fitted = fit_kappa([1.0] * 5, [0.5] * 5, [10.0] * 5, [1.0] * 5, [1.0] * 5, default=77.0)
    assert (kappa, fitted) == (77.0, False)


def test_wls_recovers_coefficients_and_ignores_zero_weight_rows():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = 2.0 + 3.0 * X[:, 1] + rng.normal(0, 0.1, 500)
    w = np.ones(500)
    y[:10], w[:10] = 1e6, 0.0
    b = wls(X, y, w)
    assert b == pytest.approx([2.0, 3.0], abs=0.05)


# --------------------------------------------------------------------------- model-level shrinkage

def test_tiny_sample_projects_near_the_league_prior_and_huge_sample_near_own_rate(tables):
    fga_per_min_own = 21.0 / 30.0
    seasons = {f"{y}-{(y + 1) % 100:02d}": (55, dict(minutes=30.0, fga=21, pts_parts=(7, 1, 3), fta=4)) for y in (2015, 2016, 2017)}
    t = add_player_seasons(tables, 9_990_001, "Big Sample", seasons, position="F")
    # 2 games, 5 minutes each, absurd rates (1.2 fga/min and 1 steal/min).
    t = add_player_seasons(t, 9_990_002, "Tiny Sample", {"2017-18": (2, dict(minutes=5.0, fga=6, stl=5, pts_parts=(2, 1, 0), fta=0))}, position="F")
    h = history_for(t)
    fit = BaselineProjector().fit(h)
    out = fit.predict().set_index("player_id")
    league_rate = fit.panel["fga"].sum() / fit.panel["min"].sum()
    big, tiny = out.loc[9_990_001], out.loc[9_990_002]
    big_rate, tiny_rate = big["proj_fga"] / big["proj_mpg"], tiny["proj_fga"] / tiny["proj_mpg"]
    # Huge sample: within 12% of his own 0.70 fga/min; tiny sample: nowhere near his raw 1.2.
    assert big_rate == pytest.approx(fga_per_min_own, rel=0.12)
    assert abs(tiny_rate - league_rate) < 0.25 * abs(1.2 - league_rate)
    # Steals: raw 1.0/min from a 10-minute sample must be regressed to almost the league rate.
    assert tiny["proj_stl"] / tiny["proj_mpg"] < 0.08
    # Tiny sample of minutes is also pulled to the league mean instead of taking 5 mpg at face value.
    assert tiny["proj_mpg"] > 10.0


def test_noisy_stats_are_regressed_harder_than_stable_ones(fitted):
    k = {n: f.kappa for n, f in fitted.rates.items()}
    assert k["stl"] > k["fga"] and k["blk"] > k["fga"]      # steals/blocks: mostly noise per minute
    assert k["fg3_pct"] > k["fg_pct"]                         # 3P% needs more regression than FG%
    assert all(f.fitted for f in fitted.rates.values())


def test_recent_seasons_count_more_than_old_ones(tables):
    """Same three seasons of minutes in opposite order: the riser projects more minutes than the faller."""
    up = {"2015-16": (60, dict(minutes=14.0)), "2016-17": (60, dict(minutes=22.0)), "2017-18": (60, dict(minutes=30.0))}
    down = {"2015-16": (60, dict(minutes=30.0)), "2016-17": (60, dict(minutes=22.0)), "2017-18": (60, dict(minutes=14.0))}
    t = add_player_seasons(tables, 9_990_011, "Riser", up, birthdate="1994-05-01")
    t = add_player_seasons(t, 9_990_012, "Faller", down, birthdate="1994-05-01")
    out = BaselineProjector().project(history_for(t)).set_index("player_id")
    assert out.loc[9_990_011, "proj_mpg"] > out.loc[9_990_012, "proj_mpg"] + 3.0


def test_young_players_project_above_identical_older_players(tables):
    line = dict(minutes=28.0, fga=14, pts_parts=(5, 1, 3), reb=5, ast=3)
    seasons = {f"{y}-{(y + 1) % 100:02d}": (60, line) for y in (2015, 2016, 2017)}
    t = add_player_seasons(tables, 9_990_021, "Young", seasons, birthdate="1996-06-01")   # 22 at target
    t = add_player_seasons(t, 9_990_022, "Old", seasons, birthdate="1985-06-01")           # 33 at target
    out = BaselineProjector().project(history_for(t)).set_index("player_id")
    assert out.loc[9_990_021, "proj_fppg"] > out.loc[9_990_022, "proj_fppg"]


# --------------------------------------------------------------------------- age curves

def test_age_curve_algebra_matches_the_integral():
    c = AgeCurve(coefs=(0.1, -0.02), center=27.0, lo=20.0, hi=36.0, n_pairs=100)
    # delta(a) = 0.1 - 0.02*(a-27). Integral from 25 to 28 = 0.3 - 0.01*((1)^2 - (-2)^2) = 0.33.
    assert c.shift(25.0, 28.0) == pytest.approx(0.33)
    assert c.shift(28.0, 25.0) == pytest.approx(-0.33)
    assert c.shift(30.0, 30.0) == 0.0
    # Held flat beyond the observed range: a year past `hi` adds exactly delta(hi).
    assert c.shift(36.0, 37.0) == pytest.approx(c.delta(36.0))
    assert c.shift(np.array([np.nan, 25.0]), np.array([28.0, 28.0]))[0] == 0.0


def test_flat_curve_when_there_are_too_few_pairs():
    pairs = pd.DataFrame({"age": [22.0, 23.0], "delta": [0.1, 0.0], "weight": [1.0, 1.0]})
    assert fit_age_curve(pairs, min_pairs=40) is FLAT
    assert FLAT.shift(20.0, 30.0) == 0.0


def test_age_curve_recovers_the_synthetic_generating_shape():
    """The synthetic league grows skill +0.12/yr under 23, +0.05 to 25, flat to 29, -0.08 after.

    From the panel alone (delta of fantasy points per minute between consecutive seasons) the
    fitted curve must have the same sign pattern, high correlation with the generating step
    function, and roughly the right decline-to-growth ratio (true ratio -0.08/0.12 = -0.67).
    """
    t = make_synthetic_tables(2010, 2019, n_teams=30, seed=1)
    panel = build_panel(History.until(t, "2019-20"), SCORING).df
    curve = fit_age_curve(consecutive_pairs(panel, "fp_sum", "min"))
    ages = np.array([20, 22, 24, 26, 28, 30, 33, 36.0])
    est = curve.delta(ages)
    truth = _age_delta(ages + 1)          # generator applies the delta at the age reached next season
    assert curve.n_pairs > 1000
    assert est[0] > 0 > est[-2]           # improving at 20, declining at 33
    assert (np.diff(est) < 0).all()       # monotone decline across the range
    assert np.corrcoef(est, truth)[0, 1] > 0.9
    assert -1.2 < est[-2] / est[0] < -0.4


def test_fitted_age_curves_are_stored_per_stat(fitted):
    assert fitted.minutes.curve.n_pairs > 0
    assert all(f.curve.n_pairs > 0 for f in fitted.rates.values())
    assert fitted.volatility_fit.curve.is_flat          # volatility is not age-adjusted


# --------------------------------------------------------------------------- panel helpers

def test_age_lookup_prefers_birthdate_and_falls_back_to_bio(tables):
    h = history_for(tables)
    look = AgeLookup(h.players, h.player_season_bio)
    b = h.player_season_bio.iloc[:200]
    ages = look.at(b["player_id"], b["season"].map(season_start))
    np.testing.assert_allclose(ages, b["age_at_season_start"], atol=0.02)
    nobirth = h.players.assign(birthdate=pd.NaT)
    look2 = AgeLookup(nobirth, h.player_season_bio)
    b2 = b[b["season"] == h.player_season_bio["season"].max()]
    np.testing.assert_allclose(look2.at(b2["player_id"], b2["season"].map(season_start)), b2["age_at_season_start"], atol=1e-9)
    assert np.isnan(look2.at([123], [2015])[0])


def test_target_season_games_ignores_a_shortened_season(tables):
    h = history_for(tables)
    lens = season_lengths(h)
    assert target_season_games(lens) == 60
    short = pd.Series({2017: 72.0, 2018: 82.0, 2019: 66.0})
    assert target_season_games(short) == 82           # a lockout year does not cap next season


# --------------------------------------------------------------------------- availability

def test_availability_is_monotone_in_recent_health(fitted):
    av = fitted.availability
    assert av.fitted
    healthy = np.tile([1.0, 0.95, 0.95, 0.9], (1, 1))
    fragile = np.tile([0.4, 0.95, 0.95, 0.9], (1, 1))
    both = np.vstack([healthy, fragile])
    mu = av.predict_mean(both, np.array([27.0, 27.0]), np.array([28.0, 28.0]))
    assert mu[0] > mu[1]
    assert ((mu > 0) & (mu < 1)).all()


def test_availability_falls_with_age_for_the_same_record(fitted):
    lags = np.tile([0.9, 0.9, 0.9, 0.9], (3, 1))
    mu = fitted.availability.predict_mean(lags, np.array([22.0, 29.0, 37.0]), np.full(3, 25.0))
    assert mu[0] > mu[2]


def test_games_played_distribution_is_bounded_and_ordered(fitted):
    av = fitted.availability
    mu = np.array([0.3, 0.7, 0.95])
    sd, lo, hi = av.distribution(mu, 82.0)
    assert (lo < mu * 82).all() and (hi > mu * 82).all()
    assert (lo < hi).all() and (hi <= 82.0).all() and (lo >= 0).all() and (sd > 0).all()
    # Beta shape: near the cap the distribution is left-skewed (median above mean).
    from scipy.stats import beta as b
    nu = 1 / av.phi - 1
    assert b.median(0.95 * nu, 0.05 * nu) > 0.95


def test_availability_falls_back_when_history_is_too_thin():
    m = AvailabilityModel.fit(np.full((10, 4), 0.8), np.full(10, 0.8), np.full(10, 27.0), np.full(10, 20.0),
                              decay=0.6, C=1.0, min_rows=60)
    assert not m.fitted
    mu = m.predict_mean(np.array([[1.0, np.nan, np.nan, np.nan]]), np.array([27.0]), np.array([20.0]))
    assert 0.02 < mu[0] < 0.99


def test_availability_features_handle_missing_seasons():
    f = np.array([[np.nan, 0.8, 0.9, np.nan], [0.5, np.nan, np.nan, np.nan]])
    X = availability_features(f, 0.6, np.array([25.0, np.nan]), np.array([20.0, 20.0]))
    assert np.isfinite(X).all()
    assert X[0, 3] == 1.0 and X[1, 3] == 0.0        # "absent last season" flag
    assert X[0, 1] == pytest.approx(X[0, 0])        # f_last falls back to the weighted mean when absent


# --------------------------------------------------------------------------- volatility

def test_volatility_prior_and_ratio_shrinkage(fitted):
    vp = fitted.vol_prior
    assert vp.b > 0 and vp.a > 0                            # spread grows with level
    assert vp.sd_prior(np.array([5.0, 25.0, 50.0])).tolist() == sorted(vp.sd_prior(np.array([5.0, 25.0, 50.0])))
    q = fitted.quantile_shape
    assert q[0] < q[1] < q[2] and q[0] < 0 < q[2]
    assert fitted.volatility_fit.kappa > 0


def test_floor_and_ceiling_are_calibrated_out_of_sample(tables):
    """About 10% of a player's actual games fall below fppg_p10 and 10% above fppg_p90 (10% +/- 4 pts)."""
    proj = BaselineProjector().project(history_for(tables)).set_index("player_id")
    gl = tables["game_logs"]
    from src.value.frame import fantasy_points_frame

    g = gl[gl["season"] == TARGET].copy()
    g["fp"] = fantasy_points_frame(g, SCORING)
    g = g.join(proj[["fppg_p10", "fppg_p90"]], on="player_id", how="inner")
    below, above = (g["fp"] < g["fppg_p10"]).mean(), (g["fp"] > g["fppg_p90"]).mean()
    assert 0.06 < below < 0.14
    assert 0.06 < above < 0.14


def test_more_volatile_player_gets_a_wider_band(tables):
    """Same mean line, different game-to-game swings (alternating big/small games)."""
    calm = dict(minutes=30.0)
    seasons_calm = {f"{y}-{(y + 1) % 100:02d}": (60, calm) for y in (2015, 2016, 2017)}
    t = add_player_seasons(tables, 9_990_031, "Calm", seasons_calm)
    # Wild: identical average via two alternating games, but with extreme spread.
    t2 = add_player_seasons(t, 9_990_032, "Wild", seasons_calm)
    gl = t2["game_logs"]
    wild = gl["player_id"] == 9_990_032
    idx = gl.index[wild]
    swing = np.where(np.arange(len(idx)) % 2 == 0, 12, -12)
    gl.loc[idx, "pts"] = gl.loc[idx, "pts"] + swing
    gl.loc[idx, "ftm"] = gl.loc[idx, "ftm"] + swing
    gl.loc[idx, "fta"] = gl.loc[idx, "fta"] + np.where(swing > 0, 12, 0)
    gl.loc[idx, "fta"] = np.maximum(gl.loc[idx, "fta"], gl.loc[idx, "ftm"])
    t2["game_logs"] = gl[gl["ftm"] >= 0]
    out = BaselineProjector().project(history_for(t2)).set_index("player_id")
    width = out["fppg_p90"] - out["fppg_p10"]
    assert width.loc[9_990_032] > width.loc[9_990_031]


# --------------------------------------------------------------------------- rookies

def test_pick_effective_handles_missing_values():
    p = pick_effective([5, np.nan, np.nan, np.nan, 0, 0], [1, 2, np.nan, 1, 0, 0])
    assert p[4] == UNDRAFTED_PICK and p[5] == UNDRAFTED_PICK      # real data uses 0 for "unknown"
    assert p[0] == 5
    assert p[1] == pytest.approx(45.5)             # second-round midpoint
    assert p[2] == UNDRAFTED_PICK
    assert p[3] == pytest.approx(15.5)


def test_rookie_prior_slope_is_negative_when_late_picks_produce_less():
    t = make_synthetic_tables(2010, 2019, n_teams=24, games_per_team=30, seed=11)
    pl = t["players"]
    first = t["game_logs"].groupby("player_id")["pts"].mean()
    r = pl[(pl["draft_year"] == pl["from_year"]) & (pl["draft_year"] <= 2017)]
    ranks = first.reindex(r["player_id"]).rank(ascending=False, method="first").to_numpy()
    pl.loc[r.index, "draft_number"] = np.clip(ranks, 1, 60).astype("int64")
    fit = BaselineProjector().fit(History.until(t, "2018-19"))
    assert fit.rookie_prior.mode == "draft_slot"
    assert fit.rookie_prior.n_rookies >= 20
    assert fit.rookie_prior.slot_slope["mpg"] < 0
    assert fit.rookie_prior.slot_slope["fga"] < 0


def _synthetic_rookie_panel(n_top=12, n_rest=80, seed=3):
    """Rookie rows whose games-played fraction *saturates* for the top picks (0.70) but rises steeply
    with log(pick) elsewhere, so a log-linear fit extrapolates far above 0.70 for pick 1."""
    rng = np.random.default_rng(seed)
    picks = np.concatenate([rng.integers(1, 11, n_top), rng.integers(11, 61, n_rest)]).astype(float)
    f = np.where(picks <= 10, 0.70, np.clip(0.70 - 0.20 * (np.log(picks) - np.log(10)), 0.05, 0.70))
    f = np.clip(f + rng.normal(0, 0.03, len(f)), 0.02, 0.99)
    n = len(picks)
    panel = pd.DataFrame({
        "player_id": np.arange(n) + 1, "s": 2020, "age": 20.0, "pos_group": "G",
        "mpg": 20.0, "gp": 50.0, "f": f, "fga": 5.0 * 20.0, "min_": 20.0 * 50.0,
    })
    players = pd.DataFrame({
        "player_id": np.arange(n) + 1, "draft_year": 2020, "draft_number": picks,
        "draft_round": np.where(picks <= 30, 1, 2),
    })
    return panel, players


def test_rookie_availability_is_capped_at_the_top_picks_own_plateau():
    panel, players = _synthetic_rookie_panel()
    prior = fit_rookie_prior(panel, players, {}, {"mpg": {"G": 20.0}, "f": {"G": 0.5}}, min_rows=20)
    assert prior.mode == "draft_slot" and prior.f_cap is not None
    assert prior.f_cap == pytest.approx(0.70, abs=0.05)
    grp, age = np.array(["G"] * 3), np.full(3, 20.0)
    top = prior.predict("f", np.array([1.0, 2.0, 3.0]), grp, age)
    assert (top <= prior.f_cap + 1e-12).all()
    # Without the cap the same regression extrapolates well past the plateau for the very top slot.
    uncapped = rookie_design(np.array([1.0]), np.array(["G"]), np.array([20.0]), prior.centers) @ prior.coefs["f"]
    assert uncapped[0] > prior.f_cap + 0.05


def test_rookie_cap_leaves_later_picks_and_other_targets_alone():
    panel, players = _synthetic_rookie_panel()
    prior = fit_rookie_prior(panel, players, {}, {"mpg": {"G": 20.0}, "f": {"G": 0.5}}, min_rows=20)
    grp, age = np.array(["G"]), np.array([20.0])
    late = prior.predict("f", np.array([45.0]), grp, age)
    assert late[0] == pytest.approx((rookie_design(np.array([45.0]), np.array(["G"]), age, prior.centers) @ prior.coefs["f"])[0])
    assert late[0] < prior.f_cap
    mpg = prior.predict("mpg", np.array([1.0]), grp, age)  # only "f" is capped
    assert mpg[0] == pytest.approx((rookie_design(np.array([1.0]), np.array(["G"]), age, prior.centers) @ prior.coefs["mpg"])[0])


def test_rookie_cap_is_skipped_when_too_few_top_picks_exist():
    panel, players = _synthetic_rookie_panel(n_top=2, n_rest=80)
    prior = fit_rookie_prior(panel, players, {}, {"mpg": {"G": 20.0}, "f": {"G": 0.5}}, min_rows=20)
    assert prior.mode == "draft_slot" and prior.f_cap is None
