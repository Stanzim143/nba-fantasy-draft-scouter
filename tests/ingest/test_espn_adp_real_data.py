"""Opt-in: run the real ADP file + player_id_map + AdpBenchmark against real ingested data.

Skipped cleanly when the ADP file has not been produced yet (``python -m src.ingest.espn_adp`` has
not been run) -- this is not a substitute for the fake-HTTP pipeline tests in ``test_espn_adp.py``,
it is the final check that the real, on-disk artifact actually works end to end with the unmodified
backtest consumer.
"""
import pytest

from src.backtest.benchmarks import AdpBenchmark, load_adp
from src.backtest.leakage import build_history
from src.contracts import HISTORY_TABLES, table_path
from src.ingest.espn_adp import adp_path
from src.store import load_tables

_ADP_MISSING = not adp_path().exists()
_ID_MAP_MISSING = not table_path("player_id_map").exists()
_HISTORY_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = pytest.mark.skipif(
    _ADP_MISSING or _ID_MAP_MISSING or bool(_HISTORY_MISSING),
    reason=(f"real ADP data not ingested yet (adp.parquet missing={_ADP_MISSING}, "
            f"player_id_map missing={_ID_MAP_MISSING}, history missing={_HISTORY_MISSING})"),
)


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


@pytest.fixture(scope="module")
def id_map():
    from src.store import read_table
    return read_table("player_id_map")


def test_real_adp_file_loads_through_unmodified_load_adp(id_map):
    data = load_adp(adp_path(), id_map, source="espn")
    assert data.n_rows > 0
    assert data.n_unmapped >= 0
    # the vast majority of ESPN's ADP-carrying players should be resolvable onto NBA player_id
    assert data.unmapped_share < 0.25


def test_real_adp_covers_multiple_seasons(id_map):
    data = load_adp(adp_path(), id_map, source="espn")
    seasons = set(data.frame["season"])
    assert len(seasons) >= 5


def test_real_adp_excludes_the_wiped_2025_26_season(id_map):
    data = load_adp(adp_path(), id_map, source="espn")
    # 2025-26 is documented as wiped (ADR 0007): ESPN's own values must never reach the benchmark file
    # as-is (every one would read exactly 140.0). If FantasyPros gap-filled the season instead, its own
    # ADP scale is unrelated to ESPN's 140.0 sentinel (FantasyPros covers more drafted players, so
    # values legitimately run higher) -- the real signature of an un-filled wipe is every value being
    # identical, not any particular ceiling.
    gap = data.frame[data.frame["season"] == "2025-26"]
    if len(gap):
        assert gap["adp"].nunique() > 1, "2025-26 rows are a constant value: looks like the ESPN wipe slipped through"


def test_adp_benchmark_runs_on_a_real_season(real, id_map):
    data = load_adp(adp_path(), id_map, source="espn")
    covered = sorted(set(data.frame["season"]) & set(real["game_logs"]["season"].unique()))
    assert covered, "no overlap between real ADP seasons and real game-log seasons"
    season = covered[-1]
    bench = AdpBenchmark(data)
    history = build_history(real, season)
    proj = bench.project(history)
    assert bench.rank_only is True
    assert len(proj) > 0
    assert proj["player_id"].is_unique
