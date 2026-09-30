"""Opt-in: run BaselineInjuryProjector on real ingested data when it exists (skipped otherwise).

Mirrors tests/models/test_model_real_data.py's pattern for the new projector.
"""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.models.injury_baseline import BaselineInjuryProjector
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _last_target_season(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return season_str(starts[-1])


def test_baseline_injury_projects_real_seasons(real):
    season = _last_target_season(real)
    h = History.until(real, season)
    h.assert_no_future()
    out = BaselineInjuryProjector().project(h)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_injury").all()
    assert out["proj_gp"].max() <= 82 + 1e-9
    assert np.isfinite(out[["proj_fppg", "fppg_p10", "fppg_p90", "proj_gp"]]).all().all()


def test_registry_baseline_injury_runs_on_real_data(real):
    season = _last_target_season(real)
    h = History.until(real, season)
    out = get_projector("baseline_injury").project(h)
    assert len(out) > 100


def test_leak_check_passes_on_real_data_for_injury_projector(real):
    """The exact machinery behind `python -m src.backtest --ablate baseline,baseline_injury
    --leak-check`, on one real season -- the streak panel is rebuilt from real game_logs/
    team_games every fit, on real pandas nullable dtypes, so this is where a real-data-only
    leakage bug (as previously found twice for other code, see tests/backtest/test_bt_real_data.py)
    would show up for this projector specifically."""
    season = _last_target_season(real)
    assert_projector_ignores_future(BaselineInjuryProjector(), real, season, seed=1)


def test_injury_panel_builds_on_real_history_without_crashing(real):
    from src.features.injury import build_injury_panel

    season = _last_target_season(real)
    h = History.until(real, season)
    panel = build_injury_panel(h)
    assert len(panel) > 1000  # thousands of real player-seasons of history
    assert (panel["missed_frac"] >= 0).all() and (panel["missed_frac"] <= 1).all()
    assert (panel["longest_streak_frac"] <= panel["missed_frac"] + 1e-9).all()
