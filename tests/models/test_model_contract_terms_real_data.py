"""Opt-in: run BaselineContractTermsProjector on real ingested data when it exists (skipped otherwise)."""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.features.contract_terms import coverage_by_season
from src.ingest.wiki_contracts import player_contracts_path, read_player_contracts
from src.models.contract_terms_baseline import BaselineContractTermsProjector
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
if not player_contracts_path().exists():
    _MISSING.append("player_contracts")
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")]


@pytest.fixture(scope="module")
def real():
    t = dict(load_tables(HISTORY_TABLES))
    t["player_contracts"] = read_player_contracts()
    return t


def _last_target_season(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return season_str(starts[-1])


def test_projects_a_real_season_and_labels_unknown_as_unknown(real):
    h = History.until(real, _last_target_season(real))
    p = BaselineContractTermsProjector()
    out = p.project(h)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_contract_terms").all()
    assert np.isfinite(out[["proj_fppg", "fppg_p10", "fppg_p90", "proj_gp"]]).all().all()
    assert p.last_fit is not None and "reason" in p.last_fit.diagnostics
    st = out["terms_status"]
    assert {"known", "unknown"} <= set(st) and 0.02 < (st == "known").mean() < 0.6
    assert out.loc[st == "unknown", "terms_years_remaining"].isna().all()
    assert out.loc[out["terms_contract_year"], "terms_years_remaining"].eq(1).all()


def test_registry_runs_on_real_data(real):
    out = get_projector("baseline_contract_terms").project(History.until(real, _last_target_season(real)))
    assert len(out) > 100


def test_real_coverage_is_partial_and_never_uses_the_target_season(real):
    seasons = ["2018-19", "2022-23", "2025-26"]
    cov = coverage_by_season(real["player_contracts"], real["game_logs"], seasons).set_index("season")
    assert (cov["veterans"] > 200).all()
    assert ((cov["any_event_pct"] > 0.2) & (cov["known_pct"] < 0.8)).all()      # partial, biased coverage: never "everyone"
    assert (cov["known"] + cov["no_length"] + cov["lapsed"] + cov["unknown"] == cov["veterans"]).all()


def test_leak_check_passes_on_real_data(real):
    assert_projector_ignores_future(BaselineContractTermsProjector(), real, _last_target_season(real), seed=1)
