"""Opt-in: run BaselineTransactionsProjector on real ingested data when it exists (skipped
otherwise -- both ``team_transactions.parquet`` and ``src.ingest.wiki_transactions`` itself may not
exist yet, since that module is built by a concurrent track; see ADR 0011).

Mirrors tests/models/test_model_roster_real_data.py's pattern for the new projector.
"""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.models.registry import get_projector
from src.models.transactions_baseline import BaselineTransactionsProjector
from src.store import load_tables

_MISSING_HISTORY = [t for t in HISTORY_TABLES if not table_path(t).exists()]

try:
    from src.ingest.wiki_transactions import read_team_transactions
    _team_transactions = read_team_transactions()
    _TRANSACTIONS_AVAILABLE = True
except (ImportError, FileNotFoundError):
    _team_transactions = None
    _TRANSACTIONS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    bool(_MISSING_HISTORY) or not _TRANSACTIONS_AVAILABLE,
    reason=(f"real data not ingested yet (missing history tables {_MISSING_HISTORY}) or "
            "team_transactions.parquet / src.ingest.wiki_transactions not built/ingested yet"))


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _last_target_season(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return season_str(starts[-1])


def test_baseline_transactions_projects_real_seasons(real):
    season = _last_target_season(real)
    h = History.until(real, season)
    h.assert_no_future()
    out = BaselineTransactionsProjector().project(h)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_transactions").all()
    assert out["proj_gp"].max() <= 82 + 1e-9
    assert np.isfinite(out[["proj_fppg", "fppg_p10", "fppg_p90", "proj_gp"]]).all().all()


def test_registry_baseline_transactions_runs_on_real_data(real):
    season = _last_target_season(real)
    h = History.until(real, season)
    out = get_projector("baseline_transactions").project(h)
    assert len(out) > 100


def test_leak_check_passes_on_real_data_for_transactions_projector(real):
    """The exact machinery behind `python -m src.backtest --ablate baseline,baseline_transactions
    --leak-check`, on one real season."""
    season = _last_target_season(real)
    assert_projector_ignores_future(BaselineTransactionsProjector(), real, season, seed=1)


def test_transactions_panel_builds_on_real_history_without_crashing(real):
    from src.features.transactions import build_transactions_panel

    season = _last_target_season(real)
    h = History.until(real, season)
    panel = build_transactions_panel(h, _team_transactions)
    assert len(panel) > 1000  # thousands of real player-seasons of history
    assert set(panel["changed_team"].unique()) <= {0.0, 1.0}
    assert (panel["team_departures_lost"] >= 0).all()
