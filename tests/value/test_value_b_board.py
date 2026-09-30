"""Draft board assembly and CLI."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_testkit import make_league  # noqa: E402

from src.contracts import HISTORY_TABLES, ContractError, History  # noqa: E402
from src.models.baseline import BaselineProjector  # noqa: E402
from src.store import write_table  # noqa: E402
from src.value import board as board_mod  # noqa: E402
from src.value.board import build_board  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tables():
    return make_league(n_teams=14, games_per_team=40, seed=8)


@pytest.fixture(scope="module")
def hist(tables):
    return History.until(tables, "2018-19")


@pytest.fixture(scope="module")
def proj(hist):
    return BaselineProjector().project(hist)


@pytest.fixture(scope="module")
def board(proj, hist):
    return build_board(proj, hist.players, season_games=40)


def test_board_is_sorted_complete_and_ranked(board, proj):
    assert list(board["rank"]) == list(range(1, len(proj) + 1))
    assert set(board["player_id"]) == set(proj["player_id"]) and board["player_id"].is_unique
    assert board["vorp"].is_monotonic_decreasing
    assert len(board) == len(proj)


def test_board_has_the_documented_columns(board):
    expected = ["rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp", "vorp",
                "vorp_per_game", "fppg_p10", "fppg_p50", "fppg_p90", "tier"]
    assert list(board.columns[:len(expected)]) == expected
    assert "adp" not in board.columns and "adp_gap" not in board.columns
    assert board[expected].drop(columns=["position"]).notna().all().all()
    assert board["position"].notna().all()


def test_tiers_are_monotone_down_the_board_and_last_tier_is_replacement(board):
    assert (board["tier"].diff().dropna() >= 0).all()
    assert board["tier"].iloc[0] == 1
    below = board[board["vorp"] <= 0]
    above = board[board["vorp"] > 0]
    assert below["tier"].nunique() == 1 and (below["tier"] > above["tier"].max()).all()


def test_vorp_matches_replacement_level(board):
    rep = board.attrs["replacement"]
    assert rep["teams"] == 14 and rep["starter_slots"] == 10 and rep["bench"] == 3
    if not board.attrs["positional_used"]:
        np.testing.assert_allclose(board["vorp"], board["proj_total_fp"] - rep["total"])
    np.testing.assert_allclose(board["vorp_per_game"], board["proj_fppg"] - rep["per_game"])
    assert (board["vorp"] > 0).sum() == pytest.approx(rep["rank"], abs=1)      # about R players clear replacement


def test_more_teams_lowers_replacement_and_grows_the_positive_pool(proj, hist):
    b10 = build_board(proj, hist.players, teams=10, season_games=40)
    b13 = build_board(proj, hist.players, teams=13, season_games=40)
    assert b13.attrs["replacement"]["total"] < b10.attrs["replacement"]["total"]
    assert (b13["vorp"] > 0).sum() > (b10["vorp"] > 0).sum()
    assert b10.attrs["replacement"]["teams"] == 10 and b13.attrs["replacement"]["teams"] == 13


def test_adp_columns_and_gap_direction(proj, hist, board):
    top, mid = board.iloc[0]["player_id"], board.iloc[9]["player_id"]
    adp = pd.DataFrame({"player_id": [top, mid, 123456789], "adp": [4.0, 5.0, 1.0]})
    b = build_board(proj, hist.players, adp=adp, season_games=40)
    r = b.set_index("player_id")
    assert r.loc[top, "adp_gap"] == 4.0 - 1          # model rank 1, market pick 4: market lets him fall -> positive
    assert r.loc[mid, "adp_gap"] == 5.0 - 10         # model rank 10, market pick 5: overdrafted -> negative
    assert r["adp"].notna().sum() == 2 and r["adp_gap"].notna().sum() == 2
    assert 123456789 not in r.index
    assert list(b.columns[-2:]) == ["adp", "adp_gap"]
    with pytest.raises(ValueError, match="adp"):
        build_board(proj, hist.players, adp=pd.DataFrame({"pid": [1]}))


def test_positional_modes_are_passed_through(proj, hist):
    on = build_board(proj, hist.players, positional="on", season_games=40)
    off = build_board(proj, hist.players, positional="off", season_games=40)
    assert on.attrs["positional_used"] and not off.attrs["positional_used"]
    assert on.attrs["positional"]["levels"].keys() == {"PG", "SG", "SF", "PF", "C"}


def test_board_without_positions_still_builds(proj):
    b = build_board(proj, players=None, season_games=40)
    assert b["position"].isna().all() and b.attrs["positional"] is None
    assert list(b["rank"]) == list(range(1, len(b) + 1))


def test_board_rejects_mixed_models_or_invalid_frames(proj, hist):
    two = pd.concat([proj, proj.assign(model="other")], ignore_index=True)
    with pytest.raises(ValueError, match="single model"):
        build_board(two, hist.players)
    with pytest.raises(ContractError):
        build_board(proj.drop(columns="proj_fppg"), hist.players)


def test_tier_kwargs_are_forwarded(proj, hist):
    one = build_board(proj, hist.players, max_tiers=1, season_games=40)
    assert one["tier"].max() == 2          # one tier above replacement + the replacement tier


def test_board_is_deterministic(proj, hist):
    a, b = build_board(proj, hist.players, season_games=40), build_board(proj, hist.players, season_games=40)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_board_from_the_naive_model_works_too(hist):
    from src.models.naive import NaiveLastSeason

    b = build_board(NaiveLastSeason().project(hist), hist.players, season_games=40)
    assert b["vorp"].is_monotonic_decreasing


# --------------------------------------------------------------------------- CLI

def test_cli_synthetic_writes_a_board_csv(tmp_path, capsys):
    out = tmp_path / "board.csv"
    code = board_mod.main(["--season", "2019-20", "--synthetic", "--out", str(out), "--top", "3"])
    assert code == 0
    df = pd.read_csv(out)
    assert list(df["rank"]) == list(range(1, len(df) + 1))
    assert {"player_id", "name", "vorp", "tier", "fppg_p10", "fppg_p90"} <= set(df.columns)
    text = capsys.readouterr().out
    assert "baseline board for 2019-20" in text and "replacement rank" in text


def test_cli_reads_tables_from_the_store_and_accepts_teams_model_adp(tmp_path, tables):
    base = tmp_path / "data"
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    adp_path = tmp_path / "adp.csv"
    first = tables["players"]["player_id"].iloc[0]
    pd.DataFrame({"player_id": [first], "adp": [7.0]}).to_csv(adp_path, index=False)
    out = tmp_path / "b13.csv"
    code = board_mod.main(["--season", "2018-19", "--model", "naive_last_season", "--teams", "13",
                           "--adp", str(adp_path), "--data-dir", str(base), "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert {"adp", "adp_gap"} <= set(df.columns)
    assert len(df) > 100


def test_cli_accepts_the_raw_ingested_adp_table_and_maps_it(tmp_path, tables, proj, capsys):
    """--adp can point straight at the ingested adp.parquet shape (season/source/source_id/adp)
    plus player_id_map in the same data dir, without a manual player_id join step first --
    this is the shape src.ingest.espn_adp actually writes."""
    base = tmp_path / "data"
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    first, second = proj["player_id"].iloc[:2]
    id_map = pd.DataFrame({
        "player_id": [first, second],
        "source": ["espn", "espn"],
        "source_id": ["1001", "1002"],
        "source_name": ["Player One", "Player Two"],
        "match_method": ["exact", "exact"],
        "confidence": [1.0, 1.0],
    })
    write_table(id_map, "player_id_map", base)

    adp_path = tmp_path / "raw_adp.parquet"
    pd.DataFrame({
        # A different season's rows, all deliberately unmapped, sit alongside the target season
        # in the same file -- exactly the shape of the real multi-season adp.parquet. The reported
        # mapping-coverage stats (below) must describe only 2018-19, not the whole file.
        "season": ["2018-19", "2018-19", "2018-19", "2099-00", "2099-00"],
        "source": ["espn", "espn", "espn", "espn", "espn"],
        "source_id": ["1001", "1002", "9999", "8888", "7777"],  # 9999/8888/7777 have no id_map row
        "adp": [3.0, 9.0, 2.0, 1.0, 2.0],
        "name": ["Player One", "Player Two", "Ghost Player", "Other Season A", "Other Season B"],
    }).to_parquet(adp_path)

    out = tmp_path / "raw_adp_board.csv"
    code = board_mod.main(["--season", "2018-19", "--model", "naive_last_season",
                           "--adp", str(adp_path), "--data-dir", str(base), "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert {"adp", "adp_gap"} <= set(df.columns)
    row = df.set_index("player_id")
    assert row.loc[first, "adp"] == 3.0
    assert row.loc[second, "adp"] == 9.0
    assert row["adp"].notna().sum() == 2  # the unmapped ghost row never reaches the board

    text = capsys.readouterr().out
    # Scoped to 2018-19's own 3 rows (1 unmapped), NOT the file's 5 rows / 4 unmapped total --
    # a regression test for a real bug where these stats leaked counts from other seasons.
    assert "adp: 2 players for 2018-19 (1 unmapped" in text
    assert "out of 3 raw rows" in text


def test_cli_raw_adp_table_with_no_player_id_map_is_a_clean_error(tmp_path, tables):
    base = tmp_path / "data"
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    adp_path = tmp_path / "raw_adp.parquet"
    pd.DataFrame({"season": ["2018-19"], "source": ["espn"], "source_id": ["1"], "adp": [1.0]}).to_parquet(adp_path)
    out = tmp_path / "x.csv"
    code = board_mod.main(["--season", "2018-19", "--model", "naive_last_season",
                           "--adp", str(adp_path), "--data-dir", str(base), "--out", str(out)])
    assert code == 2
    assert not out.exists()


def test_cli_reports_missing_data_cleanly(tmp_path, capsys):
    code = board_mod.main(["--season", "2026-27", "--data-dir", str(tmp_path / "nothing"), "--out", str(tmp_path / "x.csv")])
    assert code == 2
    assert "not found" in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


def test_cli_runs_as_a_module(tmp_path):
    out = tmp_path / "m.csv"
    res = subprocess.run([sys.executable, "-m", "src.value.board", "--season", "2019-20", "--synthetic", "--out", str(out)],
                         cwd=REPO, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert out.exists() and len(pd.read_csv(out)) > 100


def test_cli_unknown_model_is_an_error(tmp_path):
    with pytest.raises(KeyError, match="available"):
        board_mod.main(["--season", "2019-20", "--synthetic", "--model", "nope", "--out", str(tmp_path / "x.csv")])
