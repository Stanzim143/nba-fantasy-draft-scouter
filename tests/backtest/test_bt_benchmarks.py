import numpy as np
import pandas as pd
import pytest

from bt_helpers import SCORING
from src.backtest import benchmarks as B
from src.backtest.errors import BacktestError
from src.backtest.leakage import build_history
from src.backtest.actuals import season_actuals
from src.contracts import History, validate_table


# ------------------------------------------------------------------ naive last season

def test_naive_is_contract_valid_and_ranks_by_last_seasons_fantasy_points(tables):
    h = build_history(tables, "2017-18")
    proj = B.NaiveLastSeason().project(h)
    validate_table(proj, "projections")
    assert (proj["season"] == "2017-18").all() and (proj["model"] == "naive_last_season").all()
    last = season_actuals(tables["game_logs"], "2016-17", SCORING).set_index("player_id")
    assert set(proj["player_id"]) == set(last.index)                     # exactly last season's players
    np.testing.assert_allclose(proj.set_index("player_id")["proj_total_fp"].sort_index(),
                               last["total_fp"].sort_index())
    np.testing.assert_allclose(proj.set_index("player_id")["proj_gp"].sort_index(), last["gp"].sort_index())


def test_naive_bands_only_for_players_with_enough_games(tables):
    proj = B.NaiveLastSeason().project(build_history(tables, "2017-18"))
    few = proj["proj_gp"] < B.MIN_GAMES_FOR_BAND
    assert proj.loc[few, ["fppg_p10", "fppg_p50", "fppg_p90"]].isna().all().all()
    ok = proj[~few]
    assert ok[["fppg_p10", "fppg_p50", "fppg_p90"]].notna().all().all()
    assert (ok["fppg_p10"] <= ok["fppg_p50"]).all() and (ok["fppg_p50"] <= ok["fppg_p90"]).all()


def test_naive_ignores_the_target_season_and_beyond(tables):
    a = B.NaiveLastSeason().project(build_history(tables, "2017-18"))
    tables["game_logs"].loc[tables["game_logs"]["season"] >= "2017-18", "pts"] = 0
    tables["game_logs"].loc[tables["game_logs"]["season"] >= "2017-18", "fgm"] = 0
    b = B.NaiveLastSeason().project(build_history(tables, "2017-18"))
    pd.testing.assert_frame_equal(a, b)


def test_naive_with_empty_history_returns_a_valid_empty_frame(tables):
    h = History.until(tables, "2015-16")
    assert h.game_logs.empty
    out = B.NaiveLastSeason().project(h)
    assert out.empty
    validate_table(out, "projections")


def test_naive_respects_custom_scoring(tables):
    pts_only = B.NaiveLastSeason({"PTS": 1.0}).project(build_history(tables, "2017-18"))
    last = tables["game_logs"].query("season == '2016-17'").groupby("player_id")["pts"].sum()
    np.testing.assert_allclose(pts_only.set_index("player_id")["proj_total_fp"].sort_index(), last.sort_index())


def test_stats_to_projection_scores_with_the_given_scoring():
    st = pd.DataFrame({"player_id": [1], "player_name": ["A"], "proj_gp": [10.0], "proj_mpg": [30.0],
                       **{f"proj_{c}": [0.0] for c in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                                                        "reb", "ast", "stl", "blk", "tov", "pf", "pts"]}})
    st["proj_pts"], st["proj_reb"], st["proj_ast"] = 20.0, 5.0, 3.0
    out = B.stats_to_projection(st.drop(columns=["proj_fg3a", "proj_fta"], errors="ignore").assign(
        proj_fg3a=0.0, proj_fta=0.0), season="2020-21", model="m", scoring={"PTS": 1, "REB": 1, "AST": 2})
    assert out.loc[0, "proj_fppg"] == 31.0 and out.loc[0, "proj_total_fp"] == 310.0


# ------------------------------------------------------------------ ADP hook

def _id_map(rows):
    return pd.DataFrame(rows, columns=["player_id", "source", "source_id", "source_name", "match_method", "confidence"])


ID_MAP = _id_map([
    (11, "espn", "e1", "Ann A", "exact", 1.0),
    (12, "espn", "e2", "Bob B", "fuzzy", 0.6),
    (13, "espn", "e3", "Cy C", "exact", 1.0),
    (13, "spotrac", "s3", "Cy C", "exact", 1.0),
])


def _write(tmp_path, df, name="adp.csv"):
    p = tmp_path / name
    (df.to_parquet(p, index=False) if name.endswith(".parquet") else df.to_csv(p, index=False))
    return p


ADP = pd.DataFrame({"season": ["2017-18"] * 5 + ["2018-19"], "source": "espn",
                    "source_id": ["e1", "e2", "e3", "zz", "e3", "e1"], "adp": [3.0, 1.5, 2.0, 9.0, 1.0, 4.0]})


def test_missing_adp_file_fails_with_the_expected_schema(tmp_path):
    with pytest.raises(BacktestError, match=r"not found.*season.*source.*source_id.*adp"):
        B.load_adp(tmp_path / "nope.csv", ID_MAP)


def test_adp_schema_and_map_problems_are_clear_errors(tmp_path):
    p = _write(tmp_path, ADP.drop(columns=["adp"]))
    with pytest.raises(BacktestError, match="missing columns"):
        B.load_adp(p, ID_MAP)
    p = _write(tmp_path, ADP.assign(adp=[1, 2, None, 4, 5, 6.0]), "n.csv")
    with pytest.raises(BacktestError, match="null adp"):
        B.load_adp(p, ID_MAP)
    p = _write(tmp_path, ADP, "ok.csv")
    with pytest.raises(BacktestError, match="no rows for source 'bbref'"):
        B.load_adp(p, ID_MAP, source="bbref")
    with pytest.raises(BacktestError, match="player_id_map invalid"):
        B.load_adp(p, ID_MAP.drop(columns=["confidence"]))
    junk = tmp_path / "junk.parquet"
    junk.write_text("not parquet")
    with pytest.raises(BacktestError, match="could not read"):
        B.load_adp(junk, ID_MAP)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_load_adp_maps_ids_and_reports_unmapped(tmp_path, fmt):
    data = B.load_adp(_write(tmp_path, ADP, f"adp.{fmt}"), ID_MAP)
    f = data.frame.sort_values(["season", "player_id"]).reset_index(drop=True)
    assert f["player_id"].tolist() == [11, 12, 13, 11] and f["player_name"].tolist()[0] == "Ann A"
    assert data.n_rows == 6 and data.n_unmapped == 1 and data.unmapped_share == pytest.approx(1 / 6)
    # e3 appears twice in 2017-18: the earliest pick (lowest ADP = 1.0) is kept
    assert f.loc[(f.season == "2017-18") & (f.player_id == 13), "adp"].iloc[0] == 1.0
    assert data.n_duplicates == 1


def test_load_adp_min_confidence_drops_weak_matches(tmp_path):
    data = B.load_adp(_write(tmp_path, ADP), ID_MAP, min_confidence=0.9)
    assert 12 not in set(data.frame["player_id"]) and data.n_low_confidence == 1


def test_adp_benchmark_serves_only_the_target_season(tmp_path, tables):
    data = B.load_adp(_write(tmp_path, ADP), ID_MAP)
    b = B.AdpBenchmark(data)
    assert b.rank_only and b.name == "adp"
    out = b.project(build_history(tables, "2017-18"))
    assert set(out["season"]) == {"2017-18"}
    assert out.sort_values("proj_total_fp", ascending=False)["player_id"].tolist() == [13, 12, 11]   # by ADP asc
    assert b.project(build_history(tables, "2016-17")).empty          # no ADP rows for that season
