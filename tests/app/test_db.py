"""DuckDB query layer: fixture-based (always runs) plus a real-data-gated smoke test."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.app.db import available_seasons, available_tables, connect, query
from src.contracts import HISTORY_TABLES, table_path
from src.store import write_table
from src.synthetic import make_synthetic_tables

SMALL = dict(first_start=2015, last_start=2017, n_teams=6, games_per_team=10, seed=3)


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    tables = make_synthetic_tables(**SMALL)
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base=tmp_path)
    return tmp_path


def test_connect_registers_only_tables_with_files_on_disk(fixture_dir):
    con = connect(base=fixture_dir)
    tables = available_tables(con)
    assert set(HISTORY_TABLES) <= set(tables)
    # player_id_map and projections were never written to this fixture dir.
    assert "player_id_map" not in tables
    assert "projections" not in tables


def test_connect_with_no_data_registers_nothing(tmp_path):
    con = connect(base=tmp_path / "empty")
    assert available_tables(con) == []
    assert available_seasons(con) == []


def test_available_seasons_matches_the_source_frame(fixture_dir):
    con = connect(base=fixture_dir)
    tables = make_synthetic_tables(**SMALL)
    expected = sorted(tables["game_logs"]["season"].unique().tolist())
    assert available_seasons(con) == expected


def test_available_seasons_on_table_without_season_column(fixture_dir):
    con = connect(base=fixture_dir)
    # player_season_bio has a season column; player_id_map (not written here) is absent entirely.
    assert available_seasons(con, "player_id_map") == []


def test_query_reads_real_rows(fixture_dir):
    con = connect(base=fixture_dir)
    tables = make_synthetic_tables(**SMALL)
    df = query(con, "SELECT COUNT(*) AS n FROM game_logs")
    assert df["n"].iloc[0] == len(tables["game_logs"])


def test_query_supports_parameters(fixture_dir):
    con = connect(base=fixture_dir)
    tables = make_synthetic_tables(**SMALL)
    season = sorted(tables["game_logs"]["season"].unique())[0]
    df = query(con, "SELECT * FROM game_logs WHERE season = ?", [season])
    assert (df["season"] == season).all()
    assert len(df) == (tables["game_logs"]["season"] == season).sum()


def test_view_survives_a_path_with_special_characters(tmp_path):
    # A single quote in a directory name would break naive string interpolation into the DDL;
    # make sure the escaping in connect() actually works rather than just not crashing by luck.
    odd_dir = tmp_path / "o'dell's data"
    odd_dir.mkdir()
    tables = make_synthetic_tables(**SMALL)
    write_table(tables["players"], "players", base=odd_dir)
    con = connect(base=odd_dir)
    assert "players" in available_tables(con)
    df = query(con, "SELECT COUNT(*) AS n FROM players")
    assert df["n"].iloc[0] == len(tables["players"])


# --------------------------------------------------------------------------- real-data-gated

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark_real = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested (missing {_MISSING})")


@pytestmark_real
def test_connect_on_real_data_dir_exposes_history_tables():
    con = connect()
    tables = available_tables(con)
    assert set(HISTORY_TABLES) <= set(tables)
    seasons = available_seasons(con)
    assert len(seasons) > 0
    df = query(con, "SELECT * FROM players LIMIT 5")
    assert len(df) == 5
