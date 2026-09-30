"""Board loading: synthetic-data happy path, error paths that must raise BoardUnavailable with a
plain-English message (never a raw traceback the app would have to catch by class-sniffing), and a
real-data-gated smoke test."""
from __future__ import annotations

import pytest

from src.app.loader import BoardUnavailable, load_board
from src.contracts import HISTORY_TABLES, table_path


def test_synthetic_board_has_the_expected_columns():
    board = load_board("2021-22", "baseline", synthetic=True)
    for col in ("rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp",
               "vorp", "tier", "fppg_p10", "fppg_p50", "fppg_p90"):
        assert col in board.columns
    assert len(board) > 0
    assert board["rank"].iloc[0] == 1


def test_synthetic_board_with_naive_model():
    board = load_board("2021-22", "naive_last_season", synthetic=True)
    assert len(board) > 0


def test_teams_override_changes_replacement_level():
    small = load_board("2021-22", "baseline", synthetic=True, teams=4)
    big = load_board("2021-22", "baseline", synthetic=True, teams=16)
    assert small.attrs["replacement"]["rank"] != big.attrs["replacement"]["rank"]


def test_bad_season_format_raises_board_unavailable_with_readable_message():
    with pytest.raises(BoardUnavailable, match="not a valid season"):
        load_board("2021", "baseline", synthetic=True)


def test_unknown_model_raises_board_unavailable_with_readable_message():
    with pytest.raises(BoardUnavailable, match="unknown model"):
        load_board("2021-22", "not_a_real_model", synthetic=True)


def test_bad_positional_mode_raises_board_unavailable():
    with pytest.raises(BoardUnavailable, match="positional mode"):
        load_board("2021-22", "baseline", synthetic=True, positional="bogus")


def test_real_data_not_found_raises_board_unavailable_not_a_crash(tmp_path):
    # A data dir that exists but has none of the parquet files: this is the "real data not
    # ingested yet" case the app must show a message for instead of crashing.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(BoardUnavailable, match="real data isn't available"):
        load_board("2021-22", "baseline", synthetic=False, data_dir=empty_dir)


def test_season_with_no_prior_history_raises_board_unavailable(tmp_path):
    # Real data exists on disk, but every row is dated on/after the requested season, so
    # History.until finds nothing to project from. Must be a clean message, not a crash.
    from src.store import write_table
    from src.synthetic import make_synthetic_tables

    tables = make_synthetic_tables(first_start=2015, last_start=2017, n_teams=6, games_per_team=10, seed=2)
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base=tmp_path)
    with pytest.raises(BoardUnavailable, match="no historical data"):
        load_board("2010-11", "baseline", synthetic=False, data_dir=tmp_path)


# --------------------------------------------------------------------------- real-data-gated

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark_real = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested (missing {_MISSING})")


@pytestmark_real
def test_loads_a_real_board_for_the_most_recent_completed_season():
    from src.contracts import BACKTEST_SEASONS, season_start, season_str

    target = season_str(season_start(BACKTEST_SEASONS[-1]) + 1)
    board = load_board(target, "baseline")
    assert len(board) > 0
    assert board["rank"].iloc[0] == 1


# --------------------------------------------------------------------------- ADP and the watchlist

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))


@pytest.fixture(scope="module")
def real_like_store(tmp_path_factory):
    """A tmp data dir shaped like the real one: history tables, offseason tables, an id map and a raw ADP table."""
    from model_testkit import make_league
    from offseason_testkit import make_offseason

    from src.ingest import nba_offseason as no
    from src.store import write_table

    base = tmp_path_factory.mktemp("store")
    tables = make_league()
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    logs, tg = make_offseason(tables, "signal", seed=3)
    no.write_offseason_table(logs, "offseason_logs", base)
    no.write_offseason_table(tg, "offseason_team_games", base)
    gl = tables["game_logs"]
    pids = gl[gl["season"] == "2018-19"]["player_id"].unique()[:3]
    id_map = pd.DataFrame({"player_id": pids, "source": "espn", "source_id": ["9001", "9002", "9003"],
                           "source_name": ["a", "b", "c"], "match_method": "exact", "confidence": 1.0})
    write_table(id_map, "player_id_map", base)
    return base, pids


def _write_adp(base, seasons=("2019-20",)):
    rows = [{"season": s, "source": "espn", "source_id": sid, "adp": a, "name": "x", "adp_source": "espn"}
            for s in seasons for sid, a in (("9001", 3.0), ("9002", 40.0), ("9003", 120.0))]
    (base / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(base / "processed" / "adp.parquet")


def test_real_board_carries_adp_when_an_adp_file_exists(real_like_store):
    base, pids = real_like_store
    _write_adp(base)
    board = load_board("2019-20", "baseline", data_dir=base)
    assert {"adp", "adp_gap"} <= set(board.columns) and board.attrs["adp_note"] is None
    listed = board[board["adp"].notna()]
    assert set(listed["player_id"]) <= set(pids) and len(listed) >= 1


def test_board_without_an_adp_file_still_builds_and_says_why(real_like_store):
    base, _ = real_like_store
    (base / "processed" / "adp.parquet").unlink(missing_ok=True)
    board = load_board("2019-20", "baseline", data_dir=base)
    assert "adp" not in board.columns and "no ADP file" in board.attrs["adp_note"]


def test_an_unusable_adp_file_never_blocks_the_board(real_like_store):
    base, _ = real_like_store
    pd.DataFrame({"unrelated": [1]}).to_parquet(base / "processed" / "adp.parquet")
    board = load_board("2019-20", "baseline", data_dir=base)
    assert "adp" not in board.columns and "could not be used" in board.attrs["adp_note"]


def test_adp_for_another_season_is_reported_not_used(real_like_store):
    base, _ = real_like_store
    _write_adp(base, seasons=("2015-16",))
    board = load_board("2019-20", "baseline", data_dir=base)
    assert "adp" not in board.columns and "no rows for 2019-20" in board.attrs["adp_note"]


def test_synthetic_boards_skip_adp_entirely():
    board = load_board("2021-22", "baseline", synthetic=True)
    assert "adp" not in board.columns and "synthetic" in board.attrs["adp_note"]


def test_the_offseason_model_is_available_to_the_app(real_like_store):
    base, _ = real_like_store
    board = load_board("2019-20", "baseline_offseason", data_dir=base)
    assert len(board) > 100 and board["rank"].iloc[0] == 1


def test_load_watchlist_returns_the_full_unfiltered_frame(real_like_store):
    from src.app.loader import load_watchlist
    from src.value.breakouts import select_watchlist

    base, _ = real_like_store
    result = load_watchlist("2019-20", data_dir=base)
    assert result.model == "baseline_offseason" and len(result.watchlist) > 100
    assert {"young", "under_radar", "useful_prob"} <= set(result.watchlist.columns)
    narrowed = select_watchlist(result.watchlist, young_only=True, under_radar_only=True)
    assert len(narrowed) < len(result.watchlist)


def test_load_watchlist_errors_are_user_facing(tmp_path):
    from src.app.loader import load_watchlist

    with pytest.raises(BoardUnavailable, match="not a valid season"):
        load_watchlist("2019")
    with pytest.raises(BoardUnavailable, match="synthetic"):
        load_watchlist("2019-20", synthetic=True)
    with pytest.raises(BoardUnavailable, match="unknown model"):
        load_watchlist("2019-20", model="nope")
    with pytest.raises(BoardUnavailable, match="isn't available"):
        load_watchlist("2019-20", data_dir=tmp_path / "empty")
