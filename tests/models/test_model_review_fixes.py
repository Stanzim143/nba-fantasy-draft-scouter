"""Review fixes: out-of-fold interval calibration, missing-feature imputation, memo keys, contract rescaling, roster/transactions intercept."""
import numpy as np
import pytest
from model_testkit import TARGET, history_for, make_league
from test_model_contract import MIN_ROWS, _plant_contract_year_effect

from src.contracts import season_start
from src.models.appearance import AppearanceModel
from src.models.availability import AvailabilityModel
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.models.contract_baseline import BaselineContractProjector
from src.models.offseason_baseline import BaselineOffseasonProjector


@pytest.fixture(scope="module")
def tables():
    return make_league()


# --------------------------------------------------------------------------- out-of-fold calibration

def test_oof_projections_cover_only_past_training_seasons(tables):
    h = history_for(tables)
    proj = BaselineProjector(config=BaselineConfig(oof_folds=2))
    fitted = proj.fit(h)
    oof = proj.oof_projections(h, fitted.panel)
    assert oof is not None
    last = fitted.panel["s"].max()
    assert set(oof["s"]) == {last - 1, last}
    assert (oof["s"] < season_start(TARGET)).all()
    assert oof["pred"].notna().all() and (oof["sd"] > 0).all()


def test_oof_calibration_is_the_default_and_differs_from_in_sample(tables):
    h = history_for(tables)
    oof = BaselineProjector().fit(h)
    ins = BaselineProjector(config=BaselineConfig(oof_calibration=False)).fit(h)
    assert BaselineConfig().oof_calibration
    assert oof.quantile_shape != ins.quantile_shape
    q = oof.quantile_shape
    assert q[0] < q[1] < q[2] and q[0] < 0 < q[2]
    # only the calibration moves; the point projection is untouched
    a = BaselineProjector().project(h).set_index("player_id")
    b = BaselineProjector(config=BaselineConfig(oof_calibration=False)).project(h).set_index("player_id")
    np.testing.assert_allclose(a["proj_fppg"], b["proj_fppg"])
    np.testing.assert_allclose(a["proj_gp"], b["proj_gp"])


def test_too_little_history_falls_back_to_in_sample(tables):
    h = history_for(tables)
    proj = BaselineProjector(config=BaselineConfig(oof_min_seasons=99))
    fitted = proj.fit(h)
    assert proj.oof_projections(h, fitted.panel) is None
    ins = BaselineProjector(config=BaselineConfig(oof_calibration=False)).fit(h)
    assert fitted.quantile_shape == ins.quantile_shape


def test_season_uncertainty_fits_from_oof_rows(tables):
    h = history_for(tables)
    f = BaselineProjector(config=BaselineConfig(season_intervals=True)).fit(h)
    assert f.season_uncertainty is not None and all(t >= 0.02 for t in f.season_uncertainty.tau.values())


# --------------------------------------------------------------------------- missing features

def test_missing_feature_is_imputed_with_the_training_column_mean():
    rng = np.random.default_rng(0)
    n = 400
    f_lags = np.clip(rng.normal(0.8, 0.15, size=(n, 4)), 0.05, 1.0)
    age, mpg = rng.uniform(20, 36, n), rng.uniform(8, 34, n)
    extra = rng.normal(2.0, 1.0, size=(n, 1))
    target = np.clip(0.8 + 0.05 * extra[:, 0] + rng.normal(0, 0.1, n), 0.1, 1.0)
    m = AvailabilityModel.fit(f_lags, target, age, mpg, decay=0.6, C=1.0, min_rows=60, extra=extra)
    col_mean = m.scaler.mean_[-1]
    with_nan = m.predict_mean(f_lags[:5], age[:5], mpg[:5], extra=np.full((5, 1), np.nan))
    with_mean = m.predict_mean(f_lags[:5], age[:5], mpg[:5], extra=np.full((5, 1), col_mean))
    np.testing.assert_allclose(with_nan, with_mean)
    assert abs(col_mean - 2.0) < 0.3                       # a feature-scale mean, not a probability


def test_appearance_imputes_with_the_training_column_mean():
    rng = np.random.default_rng(1)
    n = 500
    f_lags = np.clip(rng.normal(0.7, 0.2, size=(n, 4)), 0.05, 1.0)
    age, mpg = rng.uniform(20, 36, n), rng.uniform(8, 34, n)
    extra = rng.normal(5.0, 1.0, size=(n, 1))
    y = (rng.random(n) < 0.8).astype(float)
    m = AppearanceModel.fit(f_lags, y, age, mpg, decay=0.6, C=1.0, min_rows=100, extra=extra)
    assert m.fitted
    a = m.predict(f_lags[:5], age[:5], mpg[:5], extra=np.full((5, 1), np.nan))
    b = m.predict(f_lags[:5], age[:5], mpg[:5], extra=np.full((5, 1), m.scaler.mean_[-1]))
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------- walk-forward memo key

def test_walk_forward_memo_separates_base_configs(tables):
    h = history_for(tables)
    cache: dict = {}
    for cfg in (BaselineConfig(), BaselineConfig(default_decay=0.9, decay_grid=(0.9,))):
        for cls, kw in ((BaselineOffseasonProjector, {}), (BaselineContractProjector, {"min_fit_rows": MIN_ROWS})):
            p = cls(base=BaselineProjector(config=cfg), walk_forward_cache=cache, **kw)
            p._base_projection(p._history_tables(h), "2017-18")
    seasons_keys = [k for k in cache if k[0] == "2017-18"]
    # two projector classes x two configs; the config must be part of the key (same config, different class share nothing extra)
    assert len({k[2] for k in seasons_keys}) == 2
    assert len(cache) >= 2


# --------------------------------------------------------------------------- contract rescaling

def test_contract_layer_rescales_the_season_total_band(tables):
    t = _plant_contract_year_effect(tables, bump=8)
    base = BaselineProjector(config=BaselineConfig(season_intervals=True))
    p = BaselineContractProjector(base=base, min_fit_rows=MIN_ROWS, walk_forward_cache={})
    out = p.project(history_for(t))
    raw = base.project(history_for(t)).set_index("player_id")
    assert p.last_fit.enabled
    out = out.set_index("player_id")
    f = out["contract_factor"]
    assert (f != 1.0).any()
    for c in ("proj_total_fp_p10", "proj_total_fp_p50", "proj_total_fp_p90", "fppg_p10"):
        np.testing.assert_allclose(out[c], raw.loc[out.index, c] * f, rtol=1e-9)
