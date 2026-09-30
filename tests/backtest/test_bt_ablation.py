import pandas as pd
import pytest

from bt_helpers import NoiseProjector
from src.backtest import ablation as A
from src.backtest.benchmarks import NaiveLastSeason
from src.backtest.harness import walk_forward

SEASONS = ["2016-17", "2017-18", "2018-19"]
KW = dict(n_boot=120, seed=4)


@pytest.fixture(scope="module")
def abl(_tables_master):
    tables = _tables_master          # read-only use: shared across this module
    variants = [("baseline", NoiseProjector(tables, 1.2, seed=1)),
                ("+injury", NoiseProjector(tables, 0.5, seed=2)),       # genuinely better
                ("+roster", NoiseProjector(tables, 0.5, seed=2)),       # identical to +injury -> zero lift
                ("+contract", NoiseProjector(tables, 1.5, seed=3))]     # genuinely worse
    return A.run_ablation(tables, variants, SEASONS, sig_metrics=("spearman_total_fp", "top50_hit", "mae_total_fp"),
                          **KW)


def test_table_shape_rows_and_columns(abl):
    t = abl.table
    assert list(t.index) == ["baseline", "+injury", "+roster", "+contract"]
    for c in ("n_seasons", "spearman_total_fp", "top50_hit", "mae_total_fp", "ndcg_100", "n_coverage_miss", "verdict"):
        assert c in t.columns
    for m in ("spearman_total_fp", "top50_hit", "mae_total_fp"):
        for suffix in ("", "_lo", "_hi", "_p", "_wins"):
            assert f"lift_{m}{suffix}" in t.columns
    assert t.loc["baseline", "verdict"] == "baseline"
    assert t.loc["baseline", ["lift_spearman_total_fp"]].isna().all()
    assert (t["n_seasons"] == 3).all()


def test_significance_logic_on_constructed_cases(abl):
    imp, same, worse = (abl.lifts["+injury"], abl.lifts["+roster"], abl.lifts["+contract"])
    # real lift: CI wholly positive, tiny p, wins every season, all metrics agree in direction
    for m in ("spearman_total_fp", "top50_hit", "mae_total_fp"):
        pl = imp[m]
        assert pl.estimate > 0 and pl.wins >= 2 and pl.n_seasons == 3
    assert imp["spearman_total_fp"].verdict == "improves" and imp["spearman_total_fp"].lo > 0
    assert imp["spearman_total_fp"].p_le_zero < 0.05
    assert imp["mae_total_fp"].verdict == "improves"            # error metric: lower MAE counted as positive lift
    # identical predictions: exactly zero lift, CI collapses on 0, no significance, no wins
    z = same["spearman_total_fp"]
    assert z.estimate == 0 and z.lo == 0 == z.hi and z.wins == 0
    assert z.verdict == "no significant change" and z.p_le_zero == 1.0
    # degradation: negative lift and flagged as hurting
    w = worse["spearman_total_fp"]
    assert w.estimate < 0 and w.hi < 0 and w.verdict == "hurts"
    assert worse["mae_total_fp"].estimate < 0
    assert abl.table.loc["+injury", "verdict"] == "improves"
    assert abl.table.loc["+roster", "verdict"] == "no significant change"
    assert abl.table.loc["+contract", "verdict"] == "hurts"


def test_lift_columns_match_lift_objects(abl):
    row = abl.table.loc["+injury"]
    pl = abl.lifts["+injury"]["spearman_total_fp"]
    assert row["lift_spearman_total_fp"] == pl.estimate and row["lift_spearman_total_fp_wins"] == pl.wins
    assert row["lift_spearman_total_fp_lo"] < row["lift_spearman_total_fp"] < row["lift_spearman_total_fp_hi"]


def test_lifts_are_computed_against_the_previous_variant_not_the_baseline(abl):
    r = abl.results
    direct = A.paired_lift(r["+roster"], r["+contract"], "spearman_total_fp", **KW)
    assert direct.estimate == abl.lifts["+contract"]["spearman_total_fp"].estimate


def test_markdown_contains_every_variant_and_verdict(abl):
    md = abl.to_markdown()
    for token in ("baseline", "+injury", "+roster", "+contract", "improves", "hurts", "no significant change",
                  "Lift over the previous variant", "seasons won"):
        assert token in md


def test_deterministic_given_seed(tables):
    def mk():
        return [("a", NoiseProjector(tables, 1.0, seed=1)), ("b", NoiseProjector(tables, 0.4, seed=1))]
    t1 = A.run_ablation(tables, mk(), SEASONS, **KW).table
    t2 = A.run_ablation(tables, mk(), SEASONS, **KW).table
    pd.testing.assert_frame_equal(t1, t2)


def test_bad_inputs(tables):
    a = walk_forward(tables, NaiveLastSeason(), SEASONS)
    b = walk_forward(tables, NaiveLastSeason(), SEASONS[:2])
    with pytest.raises(ValueError, match="different seasons"):
        A.paired_lift(a, b, "spearman_total_fp")
    with pytest.raises(ValueError, match="no better/worse"):
        A.paired_lift(a, a, "bias_total_fp")
    with pytest.raises(KeyError):
        A.paired_lift(a, a, "nonsense")
    with pytest.raises(ValueError, match="unique"):
        A.ablation_from_results([("x", a), ("x", a)])
    with pytest.raises(ValueError):
        A.ablation_from_results([])


def test_single_variant_and_plain_projector_list(tables):
    res = A.run_ablation(tables, [NaiveLastSeason()], SEASONS, **KW)
    assert list(res.table.index) == ["naive_last_season"] and res.table["verdict"].iloc[0] == "baseline"
    two = A.run_ablation(tables, [NoiseProjector(tables, 1.0), NoiseProjector(tables, 0.3)], SEASONS, **KW)
    assert list(two.table.index) == ["noise_1", "noise_0.3"]      # labels default to projector.name


def test_paired_comparison_uses_only_players_projected_by_both(tables):
    full = walk_forward(tables, NoiseProjector(tables, 0.3), SEASONS)

    class Half(NoiseProjector):
        def project(self, history):
            p = super().project(history)
            return p[p["player_id"] % 2 == 0].reset_index(drop=True)

    half = walk_forward(tables, Half(tables, 0.3), SEASONS)
    pl = A.paired_lift(half, full, "spearman_total_fp", **KW)
    n_even = int(sum((full.season_frame(s)["player_id"] % 2 == 0).sum() for s in SEASONS))
    assert pl.n_players == n_even
    assert abs(pl.estimate) < 1e-9                                  # same predictions on common players
    # but the un-paired table exposes the coverage difference
    tbl = A.ablation_from_results([("half", half), ("full", full)], **KW).table
    assert tbl.loc["half", "n_coverage_miss"] > tbl.loc["full", "n_coverage_miss"]


def test_verdict_function():
    assert A.verdict(0.01, 0.05) == "improves"
    assert A.verdict(-0.05, -0.01) == "hurts"
    assert A.verdict(-0.01, 0.02) == "no significant change"
    assert A.verdict(float("nan"), 0.1) == "n/a"
