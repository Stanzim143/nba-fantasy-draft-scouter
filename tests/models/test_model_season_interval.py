"""Season-total intervals (ADR 0033): the uncertainty fit, the simulation, and the projector columns."""
import numpy as np
import pytest
from model_testkit import TARGET, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future
from src.models.availability import AvailabilityModel
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.models.registry import get_projector
from src.models.season_interval import TAU_FLOOR, SeasonFppgUncertainty, simulate_total_quantiles


def _avail(phi=0.1):
    return AvailabilityModel(None, None, phi, 0.75, 0.6, 0)


# --------------------------------------------------------------------------- SeasonFppgUncertainty

def test_fit_recovers_tau_and_discounts_game_noise():
    rng = np.random.default_rng(0)
    n = 20_000
    pred = rng.uniform(15, 45, n)
    gp = rng.integers(20, 82, n).astype(float)
    sd_game = 0.4 * pred
    tau_true = 0.12
    actual = pred * (1 + tau_true * rng.standard_normal(n)) + sd_game / np.sqrt(gp) * rng.standard_normal(n)
    u = SeasonFppgUncertainty.fit(pred, actual, gp, sd_game, np.full(n, 2))
    assert u.tau[2] == pytest.approx(tau_true, abs=0.01)     # game noise is subtracted, not double counted
    assert u.tau[0] == u.tau[3] == u.tau[2]                  # buckets without rows borrow the pooled estimate


def test_fit_is_floored_and_ignores_thin_and_low_rows():
    n = 500
    pred = np.full(n, 30.0)
    u = SeasonFppgUncertainty.fit(pred, pred.copy(), np.full(n, 60.0), np.full(n, 10.0), np.ones(n, int))
    assert u.tau[1] == TAU_FLOOR                              # perfect projections still leave a minimum band
    # rows below the minimum games or with a tiny projection do not enter the estimate
    v = SeasonFppgUncertainty.fit(np.r_[pred, 1.0, 30.0], np.r_[pred, 50.0, 90.0], np.r_[np.full(n, 60.0), 60.0, 3.0],
                                  np.full(n + 2, 10.0), np.ones(n + 2, int))
    assert v.tau[1] == TAU_FLOOR


# --------------------------------------------------------------------------- simulation

def _sim(pids, p, mu=0.7, fppg=30.0, tau=0.1, sd=12.0, **kw):
    k = len(pids)
    return simulate_total_quantiles(pids, np.full(k, fppg), np.full(k, mu), np.asarray(p, float), 82.0, _avail(),
                                    np.full(k, tau), np.full(k, sd), **kw)


def test_quantiles_are_ordered_and_a_likely_absentee_has_a_zero_floor():
    q = _sim(np.array([1, 2, 3]), [1.0, 0.5, 0.05])
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    assert q[0, 0] > 0                       # certain to play: a positive floor
    assert q[1, 0] == 0 and q[1, 2] > 0      # p = 0.5: floor zero, ceiling a real season
    assert q[2, 2] == 0                      # 95% chance of no games: even the p90 is zero


def test_simulation_is_independent_of_order_and_of_other_players():
    a = _sim(np.array([10, 20, 30]), [0.9, 0.9, 0.9])
    b = _sim(np.array([30, 10]), [0.9, 0.9])
    assert np.allclose(a[2], b[0]) and np.allclose(a[0], b[1])
    assert np.array_equal(a, _sim(np.array([10, 20, 30]), [0.9, 0.9, 0.9]))


def test_more_talent_uncertainty_widens_the_band():
    lo = _sim(np.array([1]), [1.0], tau=0.02)
    hi = _sim(np.array([1]), [1.0], tau=0.25)
    assert (hi[0, 2] - hi[0, 0]) > (lo[0, 2] - lo[0, 0])


def test_simulated_median_tracks_the_point_projection_for_a_healthy_player():
    q = _sim(np.array([5]), [1.0], mu=0.85, fppg=35.0, tau=0.05, sd=10.0, n_draws=4000)
    assert q[0, 1] == pytest.approx(35.0 * 0.85 * 82.0, rel=0.06)


# --------------------------------------------------------------------------- projector integration

@pytest.fixture(scope="module")
def hist():
    return history_for(make_league())


@pytest.fixture(scope="module")
def out(hist):
    return get_projector("baseline_hurdle").project(hist)


def test_projector_emits_ordered_total_bands(out):
    for c in ("proj_total_fp_p10", "proj_total_fp_p50", "proj_total_fp_p90"):
        assert out[c].notna().all() and (out[c] >= 0).all()
    assert (out["proj_total_fp_p10"] <= out["proj_total_fp_p50"] + 1e-9).all()
    assert (out["proj_total_fp_p50"] <= out["proj_total_fp_p90"] + 1e-9).all()


def test_intervals_are_additive_only(hist, out):
    """Turning intervals off changes nothing but the three columns; a plain baseline never has them."""
    no_int = BaselineProjector(config=BaselineConfig(appearance_hurdle=True), name="baseline_hurdle").project(hist)
    assert not any(c.startswith("proj_total_fp_p") for c in no_int.columns)
    shared = [c for c in no_int.columns]
    assert np.allclose(no_int[shared].select_dtypes("number").to_numpy(),
                       out[shared].select_dtypes("number").to_numpy(), equal_nan=True)
    assert not any(c.startswith("proj_total_fp_p") for c in BaselineProjector().project(hist).columns)


def test_hurdle_with_intervals_ignores_the_future():
    assert_projector_ignores_future(get_projector("baseline_hurdle"), make_league(), TARGET)
