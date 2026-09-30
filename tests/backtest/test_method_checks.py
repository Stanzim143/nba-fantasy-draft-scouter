"""ADR 0030 methodology checks: ADP + model arms, tiers, risk groups, leave-one-season-out."""
import numpy as np
import pandas as pd
import pytest

from bt_helpers import SMALL
from src.backtest import method_checks as mc
from src.backtest.benchmarks import NaiveLastSeason
from src.backtest.harness import walk_forward
from src.synthetic import make_synthetic_tables

SEASONS = ["2016-17", "2017-18", "2018-19"]


@pytest.fixture(scope="module")
def tables():
    return make_synthetic_tables(**SMALL)


@pytest.fixture(scope="module")
def result(tables):
    return walk_forward(tables, NaiveLastSeason(), SEASONS, synthetic=True)


def _adp(result, noise_seed=0):
    rng = np.random.default_rng(noise_seed)
    rows = []
    for s in result.seasons:
        f = result.season_frame(s)
        f = f[f["played"]]
        score = f["actual_total_fp"].to_numpy() * rng.uniform(0.7, 1.3, len(f))
        rank = pd.Series(-score).rank(method="first").to_numpy()
        rows.append(pd.DataFrame({"season": s, "player_id": f["player_id"].to_numpy(), "adp": rank}))
    return pd.concat(rows, ignore_index=True)


def test_arms_drop_first_season_and_share_universe(result):
    arms = mc.adp_model_arms(result, _adp(result), ks=(5, 10))
    assert arms.seasons == SEASONS[1:]
    for s in arms.seasons:
        proj = {a: set(arms.frames[a][s].loc[arms.frames[a][s]["projected"], "player_id"]) for a in mc.ARMS}
        assert proj["adp"] == proj["model"] == proj["adp_model"] or proj["model"] <= proj["adp"]
    assert set(arms.metrics) == set(mc.ARMS)


def test_arms_are_out_of_sample(result):
    """Changing a season's realised outcomes must not change that season's ADP / ADP+model predictions."""
    adp = _adp(result)
    a = mc.adp_model_arms(result, adp, ks=(5,))
    last = SEASONS[-1]
    r2 = walk_forward(make_synthetic_altered(last), NaiveLastSeason(), SEASONS, synthetic=True)
    b = mc.adp_model_arms(r2, adp, ks=(5,))
    for arm in ("adp", "adp_model"):
        pa = a.frames[arm][last].set_index("player_id")["proj_total_fp"]
        pb = b.frames[arm][last].set_index("player_id")["proj_total_fp"]
        common = pa.index.intersection(pb.index)
        assert len(common)
        assert np.allclose(pa[common].fillna(0), pb[common].fillna(0))


def make_synthetic_altered(last_season):
    t = make_synthetic_tables(**SMALL)
    gl = t["game_logs"].copy()
    m = gl["season"] == last_season
    for c in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"):
        gl.loc[m, c] = gl.loc[m, c] * 2
    t["game_logs"] = gl
    return t


def test_tier_table_and_verdicts(result):
    arms = mc.adp_model_arms(result, _adp(result), ks=(5, 10))
    tiers = mc.tier_table(arms, ks=(5, 10))
    assert set(tiers["top-N"]) == {5, 10} and set(tiers["metric"]) == {"hit", "capture"}
    lines = mc.tier_verdict(tiers)
    assert len(lines) == 2 and all("ADP" in x for x in lines)
    err = mc.adp_tier_error(arms)
    assert list(err.columns)[1:] == list(mc.ARMS)


def test_needs_two_seasons_with_adp(result):
    with pytest.raises(ValueError):
        mc.adp_model_arms(result, _adp(result).iloc[:0])


def test_risk_groups_partition_and_prior_only(result, tables):
    tab = mc.risk_group_table(result, tables["game_logs"], tables["players"])
    projected = sum(int(result.season_frame(s)["projected"].sum()) for s in result.seasons)
    assert tab["n"].sum() == projected
    assert set(tab["group"]) <= set(mc.RISK_GROUPS)
    assert ((tab["appear"] >= 0) & (tab["appear"] <= 1)).all()


def test_rookies_have_no_prior_games(result, tables):
    slen = mc._season_len(tables["game_logs"])
    s = SEASONS[-1]
    f = result.season_frame(s)
    g = mc.assign_risk_groups(f, s, tables["game_logs"], tables["players"], slen)
    prior_ids = set(tables["game_logs"].loc[tables["game_logs"]["season"] < s, "player_id"])
    assert (g[~f["player_id"].isin(prior_ids)] == mc.RISK_GROUPS[0]).all()
    assert (g[f["player_id"].isin(prior_ids)] != mc.RISK_GROUPS[0]).all()


def test_loso_range_brackets_mean():
    sm = pd.DataFrame({"m": [1.0, 2.0, 3.0, 10.0]}, index=list("abcd"))
    t = mc.loso_table(sm, ["m"]).iloc[0]
    assert t["all seasons"] == pytest.approx(4.0)
    assert t["min when dropping"] == pytest.approx(2.0) and t["(season)"] == "d"
    assert t["max when dropping"] == pytest.approx(5.0) and t["(season) "] == "a"
    assert mc.loso_table(sm.iloc[:2], ["m"]).empty
