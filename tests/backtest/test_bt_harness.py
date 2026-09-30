import numpy as np
import pandas as pd
import pytest

from bt_helpers import NoiseProjector, OracleProjector, SCORING
from src.backtest import harness as H
from src.backtest.benchmarks import AdpBenchmark, NaiveLastSeason
from src.backtest.errors import BacktestError, ProjectionValidationError
from src.backtest.harness import BacktestResult, expand_seasons, validate_projections, walk_forward
from src.backtest.leakage import data_hash

SEASONS = ["2016-17", "2017-18", "2018-19"]


# ------------------------------------------------------------------ season handling

def test_expand_seasons():
    assert expand_seasons("2016-17:2018-19") == SEASONS
    assert expand_seasons("2016-17, 2018-19") == ["2016-17", "2018-19"]
    assert expand_seasons("2020-21") == ["2020-21"]
    assert expand_seasons("2016-17:2016-17") == ["2016-17"]
    for bad in ("2018-19:2016-17", "", "2016-18", "abc"):
        with pytest.raises(ValueError):
            expand_seasons(bad)


def test_season_window_validation(tables):
    p = NaiveLastSeason()
    with pytest.raises(ValueError, match="no seasons"):
        walk_forward(tables, p, [])
    with pytest.raises(ValueError, match="duplicate"):
        walk_forward(tables, p, ["2016-17", "2016-17"])
    with pytest.raises(ValueError, match="chronological"):
        walk_forward(tables, p, ["2017-18", "2016-17"])
    with pytest.raises(ValueError, match="malformed"):
        walk_forward(tables, p, ["2016/17"])
    with pytest.raises(BacktestError, match="no game_logs for requested seasons"):
        walk_forward(tables, p, ["2030-31"])
    with pytest.raises(BacktestError, match="no earlier history"):
        walk_forward(tables, p, ["2015-16"])           # first data season: nothing to train on
    with pytest.raises(BacktestError, match="tables missing"):
        walk_forward({"game_logs": tables["game_logs"]}, p, ["2016-17"])


def test_result_structure_and_metadata(tables):
    res = walk_forward(tables, NaiveLastSeason(), SEASONS, synthetic=True)
    assert isinstance(res, BacktestResult)
    assert res.projector == "naive_last_season" and res.seasons == SEASONS
    assert list(res.season_metrics.index) == SEASONS
    assert set(res.players["season"]) == set(SEASONS)
    for c in ("spearman_total_fp", "top12_hit", "top50_hit", "top100_hit", "ndcg_100", "mae_fppg", "mae_gp",
              "rmse_total_fp", "bias_total_fp", "vorp_weighted_mae", "n_coverage_miss", "share_in_p10_p90"):
        assert c in res.season_metrics.columns
    md, cfg = res.metadata, res.config
    assert md["data_hash"] == data_hash({t: tables[t] for t in H.HISTORY_TABLES})
    assert md["synthetic"] is True and md["harness_version"] == H.HARNESS_VERSION
    assert "pandas" in md["versions"] and md["created_utc"].endswith("+00:00")
    assert set(md["seconds_per_season"]) == set(SEASONS)
    assert cfg["scoring"] == SCORING and cfg["ks"] == [12, 50, 100] and cfg["min_gp"] == 10
    assert "N=" in cfg["replacement"]
    assert not res.players.empty and res.players["player_id"].notna().all()


def test_custom_ks_min_gp_and_replacement(tables):
    res = walk_forward(tables, NaiveLastSeason(), ["2017-18"], ks=(5, 20), min_gp=3, replacement=250.0)
    assert {"top5_hit", "top20_hit", "ndcg_5"} <= set(res.season_metrics.columns)
    assert "top50_hit" not in res.season_metrics.columns
    assert res.season_metrics.loc["2017-18", "replacement_level"] == 250.0
    assert "fixed 250" in res.config["replacement"]
    res2 = walk_forward(tables, NaiveLastSeason(), ["2017-18"], replacement=lambda v: float(np.percentile(v, 50)))
    assert res2.season_metrics.loc["2017-18", "replacement_level"] > 0


# ------------------------------------------------------------------ oracle / noise / naive

def test_oracle_scores_perfectly_on_every_metric(tables):
    res = walk_forward(tables, OracleProjector(tables), SEASONS)
    m = res.season_metrics
    assert (m["spearman_total_fp"] > 0.9999).all() and (m["spearman_fppg"] > 0.9999).all()
    for k in (12, 50, 100):
        # float-rounding near-ties can swap one of the top 12, so allow a little slack
        assert (m[f"top{k}_hit"] >= 0.9).all() and (m[f"top{k}_capture"] > 0.9999).all()
    assert (m["ndcg_100"] > 0.9999).all()
    assert (m["mae_total_fp"] < 1e-6).all() and (m["mae_gp"] < 1e-9).all() and (m["bias_total_fp"].abs() < 1e-6).all()
    assert (m["n_coverage_miss"] == 0).all() and (m["n_projected_no_games"] == 0).all()


def test_noise_projector_degrades_monotonically(tables):
    sigmas = [0.0, 0.1, 0.3, 0.7, 1.5]
    rows = []
    for s in sigmas:
        r = walk_forward(tables, NoiseProjector(tables, s, seed=3), SEASONS).summary()
        rows.append((r["spearman_total_fp"], r["top50_hit"], r["ndcg_50"], r["mae_total_fp"], r["rmse_total_fp"]))
    sp, top, nd, mae, rmse = map(np.array, zip(*rows))
    assert sp[0] > 0.9999
    assert np.all(np.diff(sp) < 0), sp
    assert np.all(np.diff(nd) < 0), nd
    assert np.all(np.diff(top) <= 0) and top[-1] < top[0]
    assert np.all(np.diff(mae) > 0) and np.all(np.diff(rmse) > 0)


def test_naive_carries_real_signal_but_is_far_from_perfect(tables):
    r = walk_forward(tables, NaiveLastSeason(), SEASONS).summary()
    assert 0.3 < r["spearman_total_fp"] < 0.95
    assert r["top100_hit"] > 0.4
    assert r["n_coverage_miss"] > 0            # rookies are a real coverage miss for a last-season model


# ------------------------------------------------------------------ zero-games / coverage rules

def test_projected_but_zero_games_players_are_scored_not_dropped(tables):
    res = walk_forward(tables, NaiveLastSeason(), ["2018-19"])
    f = res.players
    dnp = f[f["projected"] & ~f["played"]]
    assert len(dnp) > 0                                         # retired / injured-all-year players exist
    assert (dnp["actual_total_fp"] == 0).all() and (dnp["actual_gp"] == 0).all()
    assert dnp["actual_fppg"].isna().all()
    proj = f[f["projected"]]
    manual_mae = (proj["proj_total_fp"] - proj["actual_total_fp"]).abs().mean()
    assert res.season_metrics.loc["2018-19", "mae_total_fp"] == pytest.approx(manual_mae)
    # dropping them would flatter the model; assert the flattering number really is lower
    played_only = proj[proj["played"]]
    flattered = (played_only["proj_total_fp"] - played_only["actual_total_fp"]).abs().mean()
    assert flattered < manual_mae
    assert res.season_metrics.loc["2018-19", "n_projected_no_games"] == len(dnp)


def test_unprojected_players_are_reported_as_coverage_misses(tables):
    res = walk_forward(tables, NaiveLastSeason(), ["2018-19"])
    f = res.players
    miss = f[f["played"] & ~f["projected"]]
    m = res.season_metrics.loc["2018-19"]
    assert m["n_coverage_miss"] == len(miss) > 0
    assert m["coverage_miss_fp_share"] == pytest.approx(miss["actual_total_fp"].sum() / f.loc[f["played"], "actual_total_fp"].sum())
    assert miss["proj_total_fp"].isna().all()
    # coverage misses cap achievable top-K hit rate
    assert m["top100_hit"] <= 1 - m["top100_unprojected"] / 100 + 1e-12


# ------------------------------------------------------------------ contract validation

class Fixed:
    """Projector that returns a doctored copy of the naive projection."""

    def __init__(self, mutate, name="fixed"):
        self.name, self._mutate, self._inner = name, mutate, NaiveLastSeason()

    def project(self, history):
        out = self._mutate(self._inner.project(history).copy())
        return out


def _set(col, val):
    def f(df):
        df[col] = val
        return df
    return f


def _dup(df):
    return pd.concat([df, df.iloc[:2]], ignore_index=True)


def _other_model_dup(df):
    extra = df.iloc[:1].copy()
    extra["model"] = "some_other_model"
    return pd.concat([df, extra], ignore_index=True)


def _break_total(df):
    df["proj_total_fp"] = df["proj_total_fp"] * 1.5
    return df


def _wrong_scoring(df):
    df["proj_pts"] = df["proj_pts"] + 5.0          # stat line no longer matches proj_fppg
    return df


def _nan(df):
    df.loc[df.index[0], "proj_fppg"] = np.nan
    return df


def _inf(df):
    df.loc[df.index[0], "proj_pts"] = np.inf
    return df


def _crossed(df):
    df["fppg_p10"], df["fppg_p90"] = 50.0, 10.0
    df["fppg_p50"] = 30.0
    return df


@pytest.mark.parametrize("mutate,fragment", [
    (_set("season", "2001-02"), "expected only"),
    (lambda df: df.drop(columns=["proj_gp"]), "contract"),
    (_dup, "duplicate"),
    (_other_model_dup, "duplicate player_id"),
    (_break_total, "proj_total_fp != proj_fppg"),
    (_wrong_scoring, "league scoring"),
    (_nan, "not nullable"),
    (_inf, "non-finite"),
    (_set("proj_gp", 250.0), "proj_gp outside"),
    (_crossed, "floor exceeds ceiling"),
    (lambda df: df.iloc[0:0], "no players"),
    (lambda df: "not a frame", "expected a DataFrame"),
])
def test_bad_projections_fail_loudly_with_projector_and_season(tables, mutate, fragment):
    with pytest.raises(ProjectionValidationError) as ei:
        walk_forward(tables, Fixed(mutate, name="badmodel"), ["2017-18"])
    msg = str(ei.value)
    assert "badmodel" in msg and "2017-18" in msg and fragment in msg


def test_wrong_scoring_is_caught_against_harness_scoring(tables):
    other = dict(SCORING, BLK=0.0)                 # a projector that used a different league
    with pytest.raises(ProjectionValidationError, match="league scoring"):
        walk_forward(tables, NaiveLastSeason(other), ["2017-18"])
    walk_forward(tables, NaiveLastSeason(other), ["2017-18"], scoring=other)         # consistent -> fine
    walk_forward(tables, NaiveLastSeason(other), ["2017-18"], check_scoring=False)    # explicit opt-out


def test_projector_exception_is_wrapped_with_context(tables):
    class Crash:
        name = "crashy"

        def project(self, history):
            raise KeyError("oops")

    with pytest.raises(BacktestError, match="crashy.*KeyError.*2016-17"):
        walk_forward(tables, Crash(), ["2016-17"])


def test_validate_projections_accepts_a_good_frame(tables):
    from src.backtest.leakage import build_history
    proj = NaiveLastSeason().project(build_history(tables, "2017-18"))
    assert validate_projections(proj, season="2017-18", projector_name="x") is proj


# ------------------------------------------------------------------ determinism / order invariance

def test_walk_forward_is_deterministic(tables):
    a = walk_forward(tables, NaiveLastSeason(), SEASONS)
    b = walk_forward(tables, NaiveLastSeason(), SEASONS)
    pd.testing.assert_frame_equal(a.season_metrics, b.season_metrics)
    pd.testing.assert_frame_equal(a.players, b.players)
    assert a.metadata["data_hash"] == b.metadata["data_hash"]
    pd.testing.assert_frame_equal(a.summary_ci(n_boot=50, seed=1), b.summary_ci(n_boot=50, seed=1))


def test_result_does_not_depend_on_projection_row_order(tables):
    base = walk_forward(tables, NaiveLastSeason(), SEASONS)
    shuffled = walk_forward(tables, Fixed(lambda d: d.sample(frac=1.0, random_state=5).reset_index(drop=True)), SEASONS)
    pd.testing.assert_frame_equal(base.season_metrics.drop(columns=[]), shuffled.season_metrics)
    assert base.players["player_id"].tolist() == shuffled.players["player_id"].tolist()


def test_result_does_not_depend_on_input_row_order(tables):
    base = walk_forward(tables, NaiveLastSeason(), SEASONS)
    reordered = {k: v.sample(frac=1.0, random_state=9).reset_index(drop=True) for k, v in tables.items()}
    other = walk_forward(reordered, NaiveLastSeason(), SEASONS)
    pd.testing.assert_frame_equal(base.season_metrics, other.season_metrics)


# ------------------------------------------------------------------ extras / scoring / rank-only

def test_extras_reach_the_projector_sliced(tables):
    seen = {}

    class UsesExtras:
        name = "extras"

        def __init__(self):
            self.inner = NaiveLastSeason()

        def project(self, history):
            seen[history.target_season] = history.extras["injuries"]["season"].tolist()
            return self.inner.project(history)

    tables["injuries"] = pd.DataFrame({"season": ["2015-16", "2016-17", "2017-18", "2018-19"],
                                       "player_id": 1, "days": [1, 2, 3, 4]})
    walk_forward(tables, UsesExtras(), ["2016-17", "2017-18"])
    assert seen == {"2016-17": ["2015-16"], "2017-18": ["2015-16", "2016-17"]}


def test_custom_scoring_changes_the_actuals(tables):
    pts_only = {"PTS": 1.0}
    res = walk_forward(tables, NaiveLastSeason(pts_only), ["2018-19"], scoring=pts_only)
    f = res.players
    g = tables["game_logs"]
    truth = g[g.season == "2018-19"].groupby("player_id")["pts"].sum()
    played = f[f["played"]].set_index("player_id")["actual_total_fp"]
    np.testing.assert_allclose(played.sort_index(), truth.sort_index())


def test_rank_only_projector_gets_rank_metrics_only(tables):
    naive = walk_forward(tables, NaiveLastSeason(), SEASONS)
    f = naive.players[naive.players["projected"]]
    adp = pd.DataFrame({"season": f["season"], "player_id": f["player_id"], "player_name": f["player_name"],
                        "adp": -f["proj_total_fp"]})                          # ADP rank == naive ranking
    res = walk_forward(tables, AdpBenchmark(adp), SEASONS)
    assert res.config["rank_only"]
    np.testing.assert_allclose(res.season_metrics["spearman_total_fp"], naive.season_metrics["spearman_total_fp"])
    np.testing.assert_allclose(res.season_metrics["top50_hit"], naive.season_metrics["top50_hit"])
    assert res.season_metrics["mae_total_fp"].isna().all() and res.season_metrics["mae_fppg"].isna().all()
    assert res.season_metrics["share_in_p10_p90"].isna().all()


# ------------------------------------------------------------------ cache

class Counting(NaiveLastSeason):
    def __init__(self, fingerprint="v1"):
        super().__init__()
        self.calls = 0
        self.fingerprint = fingerprint

    def project(self, history):
        self.calls += 1
        return super().project(history)


def test_cache_requires_a_fingerprint(tables, tmp_path):
    class NoFp:
        name = "nofp"

        def project(self, history):
            return NaiveLastSeason().project(history)

    with pytest.raises(ValueError, match="fingerprint"):
        walk_forward(tables, NoFp(), ["2017-18"], cache_dir=tmp_path)


def test_cache_hits_and_invalidation(tables, tmp_path):
    p = Counting("v1")
    first = walk_forward(tables, p, SEASONS, cache_dir=tmp_path)
    assert p.calls == 3
    again = walk_forward(tables, p, SEASONS, cache_dir=tmp_path)
    assert p.calls == 3                                            # everything served from cache
    pd.testing.assert_frame_equal(first.season_metrics, again.season_metrics)
    # new model code (fingerprint) -> recompute
    q = Counting("v2")
    walk_forward(tables, q, SEASONS, cache_dir=tmp_path)
    assert q.calls == 3
    # changed *history* data -> recompute only seasons whose history changed
    tables["game_logs"].loc[tables["game_logs"]["season"] == "2016-17", "pts"] += 0   # no-op: still cached
    r = Counting("v1")
    walk_forward(tables, r, SEASONS, cache_dir=tmp_path)
    assert r.calls == 0
    gl = tables["game_logs"]
    idx = gl.index[gl["season"] == "2017-18"][0]
    tables["game_logs"].loc[idx, "ast"] += 1
    tables["game_logs"].loc[idx, "pts"] += 0
    s = Counting("v1")
    walk_forward(tables, s, SEASONS, cache_dir=tmp_path)
    assert s.calls == 1                                            # only 2018-19's history contains 2017-18


def test_cached_projections_are_still_validated(tables, tmp_path):
    p = Counting("v1")
    walk_forward(tables, p, ["2017-18"], cache_dir=tmp_path)
    for f in tmp_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        df["proj_total_fp"] = df["proj_total_fp"] * 2
        df.to_parquet(f, index=False)
    with pytest.raises(ProjectionValidationError):
        walk_forward(tables, Counting("v1"), ["2017-18"], cache_dir=tmp_path)


# ------------------------------------------------------------------ persistence / CI

def test_save_load_roundtrip(tables, tmp_path):
    res = walk_forward(tables, NaiveLastSeason(), SEASONS, synthetic=True)
    res.save(tmp_path / "run")
    back = BacktestResult.load(tmp_path / "run")
    assert back.projector == res.projector and back.seasons == res.seasons
    pd.testing.assert_frame_equal(back.players, res.players)
    np.testing.assert_allclose(back.season_metrics["spearman_total_fp"], res.season_metrics["spearman_total_fp"])
    assert back.config["ks"] == res.config["ks"] and back.metadata["data_hash"] == res.metadata["data_hash"]


def test_summary_ci_brackets_point_estimates(tables):
    res = walk_forward(tables, NaiveLastSeason(), SEASONS)
    ci = res.summary_ci(["spearman_total_fp", "top50_hit", "mae_total_fp", "vorp_weighted_mae"], n_boot=100, seed=2)
    summ = res.summary()
    for name, row in ci.iterrows():
        assert row.lo <= row.estimate <= row.hi, name
        assert row.estimate == pytest.approx(summ[name])
    assert (ci["hi"] - ci["lo"] > 0).all()
