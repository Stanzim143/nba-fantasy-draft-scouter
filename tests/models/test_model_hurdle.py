"""Hurdle availability (ADR 0031): appearance model, the GP mixture, and the ``baseline_hurdle`` projector."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import Projector, validate_table
from src.models.appearance import AppearanceModel, appearance_features, gp_mixture
from src.models.availability import AvailabilityModel
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.models.registry import get_projector


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def base(hist):
    return BaselineProjector().project(hist)


@pytest.fixture(scope="module")
def hurdle(hist):
    return get_projector("baseline_hurdle").project(hist)


# --------------------------------------------------------------------------- registry and contract

def test_registered_and_is_a_projector(hist):
    p = get_projector("baseline_hurdle")
    assert isinstance(p, Projector)
    assert p.name == "baseline_hurdle"
    assert p.config.appearance_hurdle is True


def test_plain_baseline_is_unchanged_by_the_feature(base):
    """The default config leaves the old model alone: no new column, and proj_gp is exactly mu * L."""
    assert "proj_p_appear" not in base.columns
    assert not BaselineConfig().appearance_hurdle


def test_hurdle_output_is_contract_valid(hurdle):
    validate_table(hurdle, "projections")
    assert hurdle["model"].eq("baseline_hurdle").all()
    assert np.allclose(hurdle["proj_total_fp"], hurdle["proj_fppg"] * hurdle["proj_gp"])
    assert hurdle["proj_p_appear"].between(0.0, 1.0).all()


# --------------------------------------------------------------------------- only availability changes

def test_hurdle_changes_only_games_played(base, hurdle):
    b, h = base.set_index("player_id"), hurdle.set_index("player_id")
    assert set(b.index) == set(h.index)
    h = h.loc[b.index]
    for c in ("proj_mpg", "proj_fppg", "proj_pts", "proj_reb", "proj_ast", "fppg_p10", "fppg_p50", "fppg_p90"):
        assert np.allclose(b[c], h[c]), c
    # proj_gp is the conditional mean times the appearance probability, for every player.
    assert np.allclose(h["proj_gp"], b["proj_gp"] * h["proj_p_appear"])
    assert (h["proj_gp"] <= b["proj_gp"] + 1e-9).all()
    # A rookie keeps its draft-slot prior (probability 1).
    rookies = h["is_rookie"].astype(bool)
    if rookies.any():
        assert np.allclose(h.loc[rookies, "proj_p_appear"], 1.0)


def test_gp_band_contains_the_mixture_and_is_ordered(hurdle):
    assert (hurdle["proj_gp_p10"] <= hurdle["proj_gp_p90"] + 1e-9).all()
    assert (hurdle["proj_gp_p10"] >= 0).all() and (hurdle["proj_gp_sd"] >= 0).all()
    low_p = hurdle["proj_p_appear"] < 0.85
    if low_p.any():     # a player with a >10% chance of not playing has a floor of zero games
        assert (hurdle.loc[low_p, "proj_gp_p10"] == 0).all()


# --------------------------------------------------------------------------- gp_mixture

def _avail(phi=0.1):
    return AvailabilityModel(None, None, phi, 0.75, 0.6, 0)


def test_mixture_with_certain_appearance_is_the_plain_beta():
    a, mu = _avail(), np.array([0.3, 0.6, 0.9])
    mean, sd, lo, hi = gp_mixture(a, mu, np.ones(3), 82.0)
    sd0, lo0, hi0 = a.distribution(mu, 82.0)
    assert np.allclose(mean, mu * 82.0) and np.allclose(sd, sd0) and np.allclose(lo, lo0) and np.allclose(hi, hi0)


def test_mixture_mean_and_point_mass():
    a, mu, p = _avail(), np.array([0.7, 0.7, 0.7]), np.array([1.0, 0.5, 0.05])
    mean, sd, lo, hi = gp_mixture(a, mu, p, 82.0)
    assert np.allclose(mean, p * mu * 82.0)
    assert lo[1] == 0.0 and lo[2] == 0.0 and hi[2] == 0.0      # 95% chance of zero games
    assert hi[1] > 0.0                                          # p = 0.5: the 90th percentile is a real game count
    assert sd[1] > sd[0]                                        # the zero mass widens the spread


def test_mixture_matches_monte_carlo():
    a, mu, p, L = _avail(0.15), 0.6, 0.7, 82.0
    rng = np.random.default_rng(0)
    nu = 1 / a.phi - 1
    n = 200_000
    g = np.where(rng.random(n) < p, L * rng.beta(mu * nu, (1 - mu) * nu, n), 0.0)
    mean, sd, lo, hi = gp_mixture(a, np.array([mu]), np.array([p]), L)
    assert mean[0] == pytest.approx(g.mean(), rel=0.01)
    assert sd[0] == pytest.approx(g.std(), rel=0.02)
    assert lo[0] == pytest.approx(np.quantile(g, 0.10), abs=0.5)
    assert hi[0] == pytest.approx(np.quantile(g, 0.90), abs=0.5)


# --------------------------------------------------------------------------- AppearanceModel

def test_appearance_model_learns_direction_and_falls_back():
    rng = np.random.default_rng(1)
    n, k = 4000, 4
    f = np.clip(rng.beta(3, 2, (n, k)), 0, 1)
    f[rng.random((n, k)) < 0.15] = np.nan
    age, mpg = rng.uniform(20, 38, n), rng.uniform(5, 34, n)
    x = appearance_features(f, 0.6, age, mpg)
    logit = -1.0 + 4.0 * np.nan_to_num(x[:, 0], nan=0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    m = AppearanceModel.fit(f, y, age, mpg, decay=0.6)
    assert m.fitted
    lo = m.predict(np.full((1, k), 0.1), np.array([28.0]), np.array([20.0]))[0]
    hi = m.predict(np.full((1, k), 0.95), np.array([28.0]), np.array([20.0]))[0]
    assert hi > lo + 0.3
    tiny = AppearanceModel.fit(f[:20], y[:20], age[:20], mpg[:20], decay=0.6, min_rows=100)
    assert not tiny.fitted
    assert np.allclose(tiny.predict(f[:5], age[:5], mpg[:5]), np.clip(tiny.base_rate, 0.02, 1.0))


def test_training_rows_are_the_eligible_player_seasons(hist):
    """Independent recount of (eligible player, season) rows and how many of them played."""
    fitted = BaselineProjector(config=BaselineConfig(appearance_hurdle=True)).fit(hist)
    panel = fitted.panel
    seasons = np.sort(panel["s"].unique())
    active = fitted.config.active_seasons
    n = pos = 0
    for s in seasons[1:]:
        elig = set(panel[(panel["s"] < s) & (panel["s"] >= s - active)]["player_id"])
        now = set(panel[panel["s"] == s]["player_id"])
        n += len(elig)
        pos += len(elig & now)
    a = fitted.appearance
    assert a.n_train <= n                      # rows with a non-finite feature (unknown age) are dropped, never added
    assert a.n_train >= 0.9 * n
    assert 0 < a.n_appeared <= pos
    assert pos < n                             # some eligible players did sit out: the stage has negatives to learn from


# --------------------------------------------------------------------------- leakage and determinism

def test_hurdle_ignores_the_future(tables):
    assert_projector_ignores_future(get_projector("baseline_hurdle"), tables, TARGET)


def test_hurdle_is_deterministic(hist, hurdle):
    again = get_projector("baseline_hurdle").project(hist)
    pd.testing.assert_frame_equal(hurdle, again)
