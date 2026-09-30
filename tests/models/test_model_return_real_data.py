"""Opt-in: run the return-health projectors on real ingested data when it exists (skipped otherwise). ADR 0022."""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.features.return_health import N_COLS, ReturnFeatures, build_return_panel
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _last(tables):
    return season_str(sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})[-1])


@pytest.mark.parametrize("name", ["baseline_return", "baseline_injury_return"])
def test_projects_the_last_real_season(name, real):
    h = History.until(real, _last(real))
    h.assert_no_future()
    out = get_projector(name).project(h)
    validate_table(out, "projections")
    assert (out["model"] == name).all() and out["proj_gp"].max() <= 82 + 1e-9
    assert np.isfinite(out[["proj_fppg", "fppg_p10", "fppg_p90", "proj_gp"]]).all().all()


def test_leak_check_passes_on_real_data_for_the_return_projector(real):
    assert_projector_ignores_future(get_projector("baseline_return"), real, _last(real), seed=1)


def test_panel_and_features_build_on_real_history_without_nan(real):
    h = History.until(real, _last(real))
    panel = build_return_panel(h)
    assert len(panel) > 1000 and panel["lead_frac"].between(0, 1).all()
    f = ReturnFeatures.fit(h, n_lags=3, decay=0.6)
    pids = panel["player_id"].unique()[:200].astype("int64")
    x = f.build(pids, np.full(len(pids), season_start(_last(real)) + 1, "int64"), np.full(len(pids), 26.0))
    assert x.shape == (len(pids), N_COLS) and np.isfinite(x).all()
