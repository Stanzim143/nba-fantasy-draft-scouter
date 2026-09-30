"""Opt-in: the debutant projector on real ingested data (skipped when absent)."""
import pytest

from src.contracts import HISTORY_TABLES, History, data_dir, table_path
from src.models.registry import get_projector
from src.store import load_tables

_NEED = ["offseason_logs", "player_profiles", "roster_snapshots"]
_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()] + [t for t in _NEED if not (data_dir() / "processed" / f"{t}.parquet").exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested (missing {_MISSING})")

NAMED = ("Sorber", "Toohey", "Diop", "Biberovic")


@pytest.fixture(scope="module")
def board_rows():
    tables = dict(load_tables(HISTORY_TABLES))
    o = get_projector("baseline_debut").project(History.until(tables, "2026-27"))
    return o


def test_the_rostered_debutants_are_projected_and_flagged_low_confidence(board_rows):
    d = board_rows[board_rows["projection_class"].isin(["stash", "undrafted"])]
    stash = d[d["projection_class"] == "stash"]
    names = " ".join(stash["player_name"].tolist())
    for n in NAMED:
        assert n in names
    assert (d["confidence"] == "low").all() and d["proj_total_fp"].gt(0).all()


def test_veterans_and_rookies_are_still_the_baselines(board_rows):
    base = get_projector("baseline").project(History.until(dict(load_tables(HISTORY_TABLES)), "2026-27")).set_index("player_id")
    old = board_rows[board_rows["projection_class"].isin(["veteran", "rookie"])].set_index("player_id")
    assert set(old.index) == set(base.index)
    assert (old["proj_total_fp"] - base.loc[old.index, "proj_total_fp"]).abs().max() < 1e-6
