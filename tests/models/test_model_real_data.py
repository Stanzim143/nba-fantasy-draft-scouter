"""Opt-in: run the models on real ingested data when it exists (skipped otherwise).

Enabled automatically when every history table is present under ``data_dir()/processed``
(``NBA_DATA_DIR`` selects another root). Nothing else in the suite depends on it.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from src.contracts import HISTORY_TABLES, History, season_start, season_str, table_path, validate_table
from src.models.baseline import BaselineProjector
from src.models.naive import NaiveLastSeason
from src.store import load_tables
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _last_two_target_seasons(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return [season_str(s) for s in starts[-2:]]


def test_baseline_projects_real_seasons_and_is_not_worse_than_naive(real):
    scoring = load_league()["scoring"]
    gl = real["game_logs"].copy()
    gl["fp"] = fantasy_points_frame(gl, scoring)
    actual = gl.groupby(["season", "player_id"])["fp"].sum()
    for season in _last_two_target_seasons(real):
        h = History.until(real, season)
        h.assert_no_future()
        base = BaselineProjector().project(h)
        validate_table(base, "projections")
        assert base["proj_gp"].max() <= 82 + 1e-9
        naive = NaiveLastSeason().project(h)
        a = actual.xs(season, level="season")
        common = base.set_index("player_id").index.intersection(naive.set_index("player_id").index).intersection(a.index)
        rb = spearmanr(base.set_index("player_id").loc[common, "proj_total_fp"], a.loc[common]).statistic
        rn = spearmanr(naive.set_index("player_id").loc[common, "proj_total_fp"], a.loc[common]).statistic
        print(f"{season}: baseline rho={rb:.3f} naive rho={rn:.3f} (n={len(common)})")
        assert rb > rn - 0.01, "baseline should not be materially worse than last season as-is"
        assert np.isfinite(base[["proj_fppg", "fppg_p10", "fppg_p90"]]).all().all()
