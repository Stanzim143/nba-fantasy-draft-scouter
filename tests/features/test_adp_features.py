"""ADP features for minutes and availability (ADR 0032): loading, leakage slicing, the feature columns, the minutes fit."""
import numpy as np
import pandas as pd
import pytest

from src.features.adp import (LOG_ADP_CENTER, MIN_COVERED_ROWS, AdpFeatures, assert_adp_no_future, load_store_adp,
                              slice_adp)


def _adp():
    return pd.DataFrame({
        "season": ["2016-17", "2016-17", "2017-18", "2017-18", "2018-19"],
        "player_id": [1, 2, 1, 3, 1],
        "adp": [5.0, 60.0, 8.0, 20.0, 99.0],
    })


# --------------------------------------------------------------------------- leakage

def test_slice_keeps_the_target_season_and_drops_later_ones():
    a = slice_adp(_adp(), "2017-18")
    assert sorted(a["season"].unique()) == ["2016-17", "2017-18"]
    assert len(slice_adp(_adp(), "2016-17")) == 2


def test_assert_no_future_raises_on_later_seasons():
    assert_adp_no_future(slice_adp(_adp(), "2017-18"), "2017-18")
    with pytest.raises(AssertionError, match="leakage"):
        assert_adp_no_future(_adp(), "2017-18")


def test_from_table_never_reads_later_seasons():
    feats = AdpFeatures.from_table(_adp(), "2016-17")
    assert feats.covered == frozenset({2016})
    assert feats.build([1], [2018]).tolist() == [[0.0, 0.0, 0.0]]     # 2018 has no ADP coverage in this history


# --------------------------------------------------------------------------- columns

def test_build_columns_listed_unlisted_and_uncovered():
    feats = AdpFeatures.from_table(_adp(), "2018-19")
    out = feats.build([1, 2, 99, 1], [2017, 2017, 2017, 2014])
    assert out[0].tolist() == pytest.approx([1.0, 1.0, np.log(8.0) - LOG_ADP_CENTER])
    assert out[1].tolist() == [1.0, 0.0, 0.0]          # covered season, player has no ADP: unlisted, not missing
    assert out[2].tolist() == [1.0, 0.0, 0.0]
    assert out[3].tolist() == [0.0, 0.0, 0.0]          # a season before any ADP: all zeros, the row trains the base model only


def test_lowest_adp_wins_a_duplicate_and_empty_table_is_all_zero():
    dup = pd.concat([_adp(), pd.DataFrame({"season": ["2016-17"], "player_id": [1], "adp": [3.0]})], ignore_index=True)
    feats = AdpFeatures.from_table(dup.sort_values("adp", ascending=False), "2018-19")
    assert feats.build([1], [2016])[0, 2] == pytest.approx(np.log(3.0) - LOG_ADP_CENTER)
    empty = AdpFeatures.from_table(_adp().iloc[0:0], "2018-19")
    assert (empty.build([1, 2], [2016, 2017]) == 0).all() and (empty.build_minutes([1], [2016]) == 0).all()


# --------------------------------------------------------------------------- minutes fit

def test_fit_minutes_recovers_the_relation_and_needs_enough_rows():
    rng = np.random.default_rng(0)
    n = 1500
    pids = np.arange(n)
    adp = pd.DataFrame({"season": "2017-18", "player_id": pids[: n // 2], "adp": rng.uniform(2, 150, n // 2)})
    feats = AdpFeatures.from_table(adp, "2017-18")
    ts = np.full(n, 2017)
    _, listed, z = feats._parts(pids, ts)
    resid = 1.0 * listed - 1.5 * z + 0.3 * rng.standard_normal(n)
    feats.fit_minutes(pids, ts, np.full(n, 20.0), 20.0 + resid, np.full(n, 50.0))
    adj = feats.build_minutes(pids, ts)
    assert np.corrcoef(adj, resid)[0, 1] > 0.8
    assert adj[listed > 0][np.argsort(z[listed > 0])][0] > adj[listed > 0][np.argsort(z[listed > 0])][-1]   # better ADP -> more minutes
    few = AdpFeatures.from_table(adp, "2017-18")
    few.fit_minutes(pids[: MIN_COVERED_ROWS - 1], ts[: MIN_COVERED_ROWS - 1], np.full(MIN_COVERED_ROWS - 1, 20.0),
                    np.full(MIN_COVERED_ROWS - 1, 25.0), np.full(MIN_COVERED_ROWS - 1, 50.0))
    assert (few.minutes_coef == 0).all()


def test_minutes_adjustment_is_zero_outside_adp_coverage():
    feats = AdpFeatures.from_table(_adp(), "2018-19")
    feats.minutes_coef = np.array([0.5, 1.0, -2.0])
    assert feats.build_minutes([1], [2014])[0] == 0.0
    assert feats.build_minutes([1], [2017])[0] == pytest.approx(0.5 + 1.0 - 2.0 * (np.log(8.0) - LOG_ADP_CENTER))


# --------------------------------------------------------------------------- store loader

def _write_store(base, adp_rows, id_rows):
    (base / "processed").mkdir(parents=True)
    pd.DataFrame(adp_rows).to_parquet(base / "processed" / "adp.parquet")
    pd.DataFrame(id_rows).to_parquet(base / "processed" / "player_id_map.parquet")


def test_load_store_adp_maps_ids_keeps_lowest_adp_and_skips_unknowns(tmp_path):
    ids = [dict(player_id=11, source="espn", source_id="a", source_name="A", match_method="exact", confidence=1.0),
           dict(player_id=12, source="espn", source_id="b", source_name="B", match_method="fuzzy", confidence=0.4),
           dict(player_id=13, source="other", source_id="a", source_name="A", match_method="exact", confidence=1.0)]
    adp = [dict(season="2024-25", source="espn", source_id="a", adp=30.0),
           dict(season="2024-25", source="espn", source_id="a", adp=12.0),
           dict(season="2024-25", source="espn", source_id="b", adp=70.0),
           dict(season="2024-25", source="espn", source_id="zzz", adp=90.0),
           dict(season="2024-25", source="yahoo", source_id="a", adp=1.0)]
    _write_store(tmp_path, adp, ids)
    got = load_store_adp(tmp_path).sort_values("player_id").reset_index(drop=True)
    assert got[["player_id", "adp"]].values.tolist() == [[11, 12.0], [12, 70.0]]
    assert load_store_adp(tmp_path, min_confidence=0.5)["player_id"].tolist() == [11]


def test_load_store_adp_reports_a_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="espn_adp"):
        load_store_adp(tmp_path)
