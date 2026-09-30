"""Opt-in: run BaselineContractProjector on real ingested data when it exists (skipped otherwise)."""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.features.contract import contract_clock
from src.models.contract_baseline import BaselineContractProjector
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")]


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _last_target_season(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return season_str(starts[-1])


def test_baseline_contract_projects_a_real_season(real):
    h = History.until(real, _last_target_season(real))
    p = BaselineContractProjector()
    out = p.project(h)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_contract").all()
    assert np.isfinite(out[["proj_fppg", "fppg_p10", "fppg_p90", "proj_gp"]]).all().all()
    assert p.last_fit is not None and "reason" in p.last_fit.diagnostics


def test_registry_runs_on_real_data(real):
    out = get_projector("baseline_contract").project(History.until(real, _last_target_season(real)))
    assert len(out) > 100


def test_real_rookie_scale_clock_by_draft_class(real):
    """No model needed: in 2025-26 the 2022..2025 first-round classes are in scale years 4..1 exactly."""
    c = contract_clock(History.until(real, "2025-26").players, "2025-26")
    for draft_year, scale in ((2022, 4.0), (2023, 3.0), (2024, 2.0), (2025, 1.0)):
        sel = (c.years_since_draft == 2025 - draft_year + 1) & (c.draft_round == 1)
        assert sel.sum() > 20
        assert set(np.unique(c.scale_year[sel])) == {scale}
    assert c.is_contract_year.sum() == ((c.years_since_draft == 4) & (c.draft_round == 1)).sum()


def test_leak_check_passes_on_real_data(real):
    assert_projector_ignores_future(BaselineContractProjector(), real, _last_target_season(real), seed=1)
