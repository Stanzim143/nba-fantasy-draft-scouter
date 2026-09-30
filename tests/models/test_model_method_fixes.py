"""ADR 0030: honest shrinkage tuning (predicted MPG) and the leak-free position-prior switch."""
import numpy as np
import pytest
from model_testkit import history_for, make_league

from src.models import rates
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.models.rates import SPECS, fit_spec, frame_from_panel_rows
from src.models.registry import get_projector


@pytest.fixture(scope="module")
def hist():
    return history_for(make_league())


@pytest.fixture(scope="module")
def parts(hist):
    cfg = BaselineConfig()
    panel = BaselineProjector(config=cfg).fit(hist).panel   # includes the volatility ratio columns
    return cfg, panel, frame_from_panel_rows(panel, cfg)


def _captured_mpg(monkeypatch, cfg, panel, train, mpg_pred):
    seen = []
    orig = rates.SpecFit.prior_for

    def spy(self, group, mpg):
        seen.append(None if mpg is None else np.array(mpg, float))
        return orig(self, group, mpg)

    monkeypatch.setattr(rates.SpecFit, "prior_for", spy)
    fit_spec(SPECS["reb"], panel, train, cfg, mpg_pred)
    return seen[0]


def test_tuning_uses_predicted_mpg_by_default(monkeypatch, parts):
    cfg, panel, train = parts
    rows = train.has_history()
    pred = np.full(len(panel), 17.5)
    got = _captured_mpg(monkeypatch, cfg, panel, train, pred)
    assert np.allclose(got, 17.5) and len(got) == rows.sum()


def test_switch_restores_actual_mpg_tuning(monkeypatch, parts):
    _, panel, train = parts
    cfg = BaselineConfig(tune_with_predicted_mpg=False)
    rows = train.has_history()
    got = _captured_mpg(monkeypatch, cfg, panel, train, np.full(len(panel), 17.5))
    assert np.allclose(got, panel["mpg"].to_numpy(float)[rows])


def test_missing_prediction_falls_back_to_actual(monkeypatch, parts):
    cfg, panel, train = parts
    got = _captured_mpg(monkeypatch, cfg, panel, train, None)
    assert np.allclose(got, panel["mpg"].to_numpy(float)[train.has_history()])


def test_position_priors_off_is_leak_free_and_valid(hist):
    f = BaselineProjector(config=BaselineConfig(use_position_priors=False)).fit(hist)
    assert (f.panel["pos_group"] == "U").all()
    assert (f._positions(np.array([1, 2, 3])) == "U").all()
    out = get_projector("baseline_nopos").project(hist)
    assert (out["model"] == "baseline_nopos").all() and out["proj_total_fp"].notna().all()


def test_defaults_unchanged_flags(hist):
    cfg = BaselineConfig()
    assert cfg.tune_with_predicted_mpg and cfg.use_position_priors
