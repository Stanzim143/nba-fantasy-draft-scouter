"""Opt-in: the offseason projector and the 2026-27 board on real ingested data (skipped when absent)."""
import numpy as np
import pytest

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import HISTORY_TABLES, History, data_dir, table_path, validate_table
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
_HAVE = not _MISSING and (data_dir() / "processed" / "offseason_logs.parquet").exists()
pytestmark = pytest.mark.skipif(not _HAVE, reason=f"real data not ingested (missing {_MISSING or 'offseason tables'})")


@pytest.fixture(scope="module")
def real():
    from src.ingest.nba_offseason import read_offseason_logs, read_offseason_team_games

    tables = dict(load_tables(HISTORY_TABLES))
    tables["offseason_logs"], tables["offseason_team_games"] = read_offseason_logs(), read_offseason_team_games()
    return tables


@pytest.fixture(scope="module")
def last_completed(real):
    proj = get_projector("baseline_offseason")
    out = proj.project(History.until(real, "2025-26"))
    return proj, out


def test_the_layer_fits_and_switches_on_for_the_last_completed_season(last_completed):
    proj, out = last_completed
    d = proj.last_fit.diagnostics
    assert proj.last_fit.enabled and d["cv_gain_vs_zero"] > 0.02 and d["train_rows"] > 2000 and d["train_seasons"] >= 8
    validate_table(out, "projections")
    assert out["offseason_enabled"].all() and (out["offseason_adj"] != 0).sum() > 100
    assert out["offseason_factor"].between(0.6, 1.6).all()


def test_the_layer_moves_players_who_have_a_line_and_leaves_the_rest(last_completed):
    _, out = last_completed
    has = out[["sl_gp", "pre_gp"]].fillna(0).sum(axis=1) > 0
    assert (out.loc[~has, "offseason_adj"] == 0).all()
    assert out.loc[has, "offseason_adj"].abs().mean() > 0.3


@pytest.mark.slow
def test_future_invariance_holds_on_real_data(real):
    assert_projector_ignores_future(get_projector("baseline_offseason"), real, "2025-26")


def test_the_live_board_projects_the_whole_rookie_class_with_capped_availability(real):
    out = get_projector("baseline").project(History.until(real, "2026-27"))
    rookies = out[out["is_rookie"]]
    assert len(rookies) >= 45
    assert rookies["proj_gp"].max() < 70                              # the top-pick availability plateau (was ~80.8)
    assert rookies["age"].between(17, 26).all()


def test_the_live_offseason_projection_uses_summer_league_only_until_preseason_games_exist(real):
    proj = get_projector("baseline_offseason")
    out = proj.project(History.until(real, "2026-27"))
    assert out["sl_gp"].notna().sum() > 100
    assert out["pre_gp"].notna().sum() == 0 or (real["offseason_logs"].query("event_season == '2026-27' and context == 'preseason'").shape[0] > 0)
    validate_table(out, "projections")
    assert np.isfinite(out["proj_total_fp"]).all()
