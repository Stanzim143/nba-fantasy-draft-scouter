import math

import numpy as np
import pytest

from bt_helpers import frame_from_arrays
from src.backtest import metrics as M

nan = float("nan")


# ------------------------------------------------------------------ spearman

def test_spearman_perfect_and_reversed():
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_matches_scipy_with_ties_and_random_data():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 60))
        a = rng.integers(0, 6, n).astype(float)          # heavy ties
        b = a + rng.integers(-2, 3, n)
        assert M.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-12)
    x, y = rng.normal(size=200), rng.normal(size=200)
    assert M.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_hand_computed():
    # ranks x = 1,2,3,4,5 ; ranks y = 2,1,4,3,5 -> d^2 = 1+1+1+1+0 = 4 -> 1 - 6*4/(5*24) = 0.8
    assert M.spearman([1, 2, 3, 4, 5], [20, 10, 40, 30, 50]) == pytest.approx(0.8)


def test_spearman_edge_cases():
    assert math.isnan(M.spearman([], []))
    assert math.isnan(M.spearman([1.0], [2.0]))
    assert math.isnan(M.spearman([1, 1, 1], [1, 2, 3]))            # constant side
    assert math.isnan(M.spearman([nan, nan], [1, 2]))
    assert M.spearman([1, 2, nan, 4], [1, 2, 3, 4]) == pytest.approx(1.0)   # NaN pair dropped
    with pytest.raises(ValueError):
        M.spearman([1, 2], [1, 2, 3])


# ------------------------------------------------------------------ errors

def test_error_metrics_hand_computed():
    pred, actual = [1, 2, 3], [2, 2, 5]           # errors -1, 0, -2
    assert M.mae(pred, actual) == pytest.approx(1.0)
    assert M.rmse(pred, actual) == pytest.approx(math.sqrt(5 / 3))
    assert M.bias(pred, actual) == pytest.approx(-1.0)   # negative = under-projection


def test_error_metrics_nan_and_empty():
    assert M.mae([1, nan, 3], [2, 5, 5]) == pytest.approx(1.5)
    for f in (M.mae, M.rmse, M.bias):
        assert math.isnan(f([], []))
        assert math.isnan(f([nan], [1.0]))


# ------------------------------------------------------------------ top-K

def test_topk_overlap_hand_computed():
    pred = [9, 8, 7, 1, 2]
    actual = [5, 6, 1, 9, 8]
    assert M.topk_overlap(pred, actual, 2) == 0.0          # {0,1} vs {3,4}
    assert M.topk_overlap(pred, actual, 3) == pytest.approx(1 / 3)  # {0,1,2} vs {3,4,1}
    assert M.topk_overlap(actual, actual, 4) == 1.0


def test_topk_overlap_unprojected_cannot_be_selected():
    # player 0 is the best actual player but has no projection (NaN pred).
    pred = [nan, 8, 7]
    actual = [10, 5, 1]
    assert M.topk_overlap(pred, actual, 2) == pytest.approx(0.5)     # picks {1,2}; truth {0,1}


def test_topk_overlap_k_larger_than_universe_and_empty():
    assert M.topk_overlap([3, 2, 1], [3, 2, 1], 100) == 1.0
    assert math.isnan(M.topk_overlap([], [], 5))
    assert math.isnan(M.topk_overlap([1.0], [nan], 5))
    with pytest.raises(ValueError):
        M.topk_overlap([1], [1], 0)


def test_topk_ties_are_deterministic_by_position():
    pred = [5, 5, 5, 5]
    actual = [4, 3, 2, 1]
    assert M.topk_overlap(pred, actual, 2) == 1.0     # stable: takes positions 0,1 == true top-2
    assert M.topk_overlap(pred, actual, 2) == M.topk_overlap(pred, actual, 2)


def test_topk_value_capture():
    actual = [10, 5, 1, 0]
    assert M.topk_value_capture([4, 3, 2, 1], actual, 2) == pytest.approx(1.0)
    assert M.topk_value_capture([1, 3, 4, 2], actual, 2) == pytest.approx((5 + 1) / 15)
    assert math.isnan(M.topk_value_capture([1, 2], [0, 0], 1))         # ideal is 0
    # projected-but-absent (NaN actual) is a dead pick, not an error
    assert M.topk_value_capture([9, 1], [nan, 5], 1) == 0.0


# ------------------------------------------------------------------ NDCG

def test_ndcg_hand_computed():
    actual = [3, 2, 1]
    pred = [2, 3, 1]                                  # order: item1, item0, item2
    dcg = 2 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert M.ndcg(pred, actual) == pytest.approx(dcg / ideal)
    assert M.ndcg(actual, actual) == pytest.approx(1.0)


def test_ndcg_truncation_and_top_heaviness():
    actual = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    swap_top = [9, 10, 8, 7, 6, 5, 4, 3, 2, 1]
    swap_bottom = [10, 9, 8, 7, 6, 5, 4, 3, 1, 2]
    assert M.ndcg(swap_top, actual) < M.ndcg(swap_bottom, actual) < 1.0
    # error below the cutoff is invisible to NDCG@3
    assert M.ndcg(swap_bottom, actual, k=3) == pytest.approx(1.0)


def test_ndcg_ties_use_expected_dcg_and_are_order_invariant():
    actual = np.array([5.0, 3.0, 1.0])
    all_tied = np.array([1.0, 1.0, 1.0])
    disc = 1 / np.log2(np.arange(3) + 2)
    expected = actual.sum() * disc.mean() / (np.sort(actual)[::-1] @ disc)
    assert M.ndcg(all_tied, actual) == pytest.approx(expected)
    perm = np.array([2, 0, 1])
    assert M.ndcg(all_tied[perm], actual[perm]) == pytest.approx(expected)


def test_ndcg_nan_pred_ranked_last_and_degenerate():
    actual = [5, 3, 1]
    assert M.ndcg([nan, 2, 1], actual) < M.ndcg([3, 2, 1], actual)
    assert math.isnan(M.ndcg([1, 2], [0, 0]))
    assert math.isnan(M.ndcg([], []))
    assert M.ndcg([1, 2, 3], [nan, 2, 3]) <= 1.0   # NaN actual has zero gain
    with pytest.raises(ValueError):
        M.ndcg([1], [1], k=0)


# ------------------------------------------------------------------ replacement / VORP

def test_replacement_level():
    assert M.replacement_level(range(1, 11), 3) == 7           # 4th best of 10..1 is 7
    assert M.replacement_level([5, 4], 10) == 0.0               # universe smaller than the roster pool
    assert M.replacement_level([nan, 5, 4, 3], 1) == 4


def test_default_replacement_uses_league_roster_size():
    vals = np.arange(1, 201, dtype=float)                        # 200 players
    from src.value.league import load_league
    lg = load_league()["league"]
    n = M.default_n_rostered()
    assert n == lg["teams"] * lg["roster"]["size"]               # derive from config, don't hardcode: it can change
    assert M.default_replacement(vals) == 200 - n                # (N+1)-th best of 1..200


def test_resolve_replacement_forms():
    vals = np.array([10.0, 8.0, 6.0, 4.0])
    assert M.resolve_replacement(5.5, vals) == 5.5
    assert M.resolve_replacement(lambda v: float(np.median(v)), vals) == 7.0
    assert M.resolve_replacement(None, vals) == 0.0              # tiny universe -> zero floor


def test_vorp_weighted_error_hand_computed():
    pred, actual = [100, 50], [80, 60]
    # R=55 -> weights max(100,80)-55=45 and max(50,60)-55=5
    assert M.vorp_weighted_error(pred, actual, 55, "mae") == pytest.approx((45 * 20 + 5 * 10) / 50)
    assert M.vorp_weighted_error(pred, actual, 55, "bias") == pytest.approx((45 * 20 - 5 * 10) / 50)
    assert math.isnan(M.vorp_weighted_error(pred, actual, 1000, "mae"))   # nobody above replacement
    assert math.isnan(M.vorp_weighted_error([], [], 0, "mae"))
    with pytest.raises(ValueError):
        M.vorp_weighted_error(pred, actual, 55, "rmse")


def test_vorp_weighting_ignores_replacement_filler_errors():
    # Same huge error on a filler player must not move the metric.
    base = M.vorp_weighted_error([100, 10], [90, 10], 50, "mae")
    with_filler_error = M.vorp_weighted_error([100, 10], [90, 40], 50, "mae")
    assert base == with_filler_error == pytest.approx(10.0)


# ------------------------------------------------------------------ bootstrap

def test_bootstrap_ci_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    p = a + rng.normal(scale=0.5, size=300)
    b1 = M.bootstrap_ci(p, a, M.spearman, n_boot=200, seed=7)
    b2 = M.bootstrap_ci(p, a, M.spearman, n_boot=200, seed=7)
    assert b1 == b2
    assert b1.lo < b1.estimate < b1.hi
    assert b1.estimate == pytest.approx(M.spearman(p, a))
    assert M.bootstrap_ci(p, a, M.spearman, n_boot=200, seed=8) != b1


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(2)

    def width(n):
        a = rng.normal(size=n)
        p = a + rng.normal(scale=0.7, size=n)
        b = M.bootstrap_ci(p, a, M.spearman, n_boot=300, seed=0)
        return b.hi - b.lo

    assert width(400) < width(40)


def test_bootstrap_edge_cases():
    b = M.bootstrap_ci([], [], M.mae, n_boot=20)
    assert math.isnan(b.estimate) and math.isnan(b.lo)
    b = M.bootstrap_ci([1.0], [1.0], M.spearman, n_boot=20)          # metric undefined on n=1
    assert math.isnan(b.lo)
    with pytest.raises(ValueError):
        M.bootstrap_ci([1], [1], M.mae, n_boot=0)
    with pytest.raises(ValueError):
        M.bootstrap_ci([1], [1], M.mae, level=1.5)


def test_paired_bootstrap_detects_real_lift_and_null_lift():
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    good = a + rng.normal(scale=0.3, size=400)
    bad = a + rng.normal(scale=1.5, size=400)
    lift = M.paired_bootstrap_diff([(bad, good, a)], M.spearman, n_boot=300, seed=1)
    assert lift.estimate > 0 and lift.lo > 0 and lift.p_le_zero < 0.01
    null = M.paired_bootstrap_diff([(good, good.copy(), a)], M.spearman, n_boot=300, seed=1)
    assert null.estimate == 0 and null.lo == 0 == null.hi and null.p_le_zero == 1.0
    worse = M.paired_bootstrap_diff([(good, bad, a)], M.spearman, n_boot=300, seed=1)
    assert worse.estimate < 0 and worse.hi < 0 and worse.p_le_zero > 0.99


def test_paired_bootstrap_direction_flips_for_error_metrics():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    close = a + rng.normal(scale=0.2, size=300)
    far = a + rng.normal(scale=1.0, size=300)
    lift = M.paired_bootstrap_diff([(far, close, a)], M.mae, higher_is_better=False, n_boot=200, seed=0)
    assert lift.estimate > 0 and lift.lo > 0        # lower MAE == improvement == positive lift


def test_bootstrap_groups_are_stratified_and_skip_empty_groups():
    rng = np.random.default_rng(5)
    g = [(rng.normal(size=50), rng.normal(size=50)), (np.array([]), np.array([]))]
    b = M.bootstrap_groups(g, lambda x: M.mae(x[0], x[1]), n_boot=50, seed=0)
    assert b.n_groups == 2 and not math.isnan(b.estimate)


# ------------------------------------------------------------------ frame-level

def test_build_specs_registry_and_lookup():
    specs = M.build_specs((12, 50), replacement_value=100.0)
    for name in ["spearman_total_fp", "spearman_fppg", "top12_hit", "top50_capture", "ndcg_12", "mae_gp",
                 "rmse_total_fp", "bias_fppg", "vorp_weighted_mae", "vorp_weighted_bias"]:
        assert name in specs
    assert M.metric_spec("top7_hit").higher_is_better is True        # arbitrary K parses
    assert M.metric_spec("mae_total_fp").higher_is_better is False
    assert M.metric_spec("bias_total_fp").higher_is_better is None
    with pytest.raises(KeyError):
        M.metric_spec("no_such_metric")


def test_compute_season_metrics_on_constructed_frame():
    # 6 players: p1..p6.  p6 played but unprojected; p5 projected but did not play.
    pred = [600, 500, 400, 300, 200, nan]
    actual = [600, 500, 300, 400, 0, 550]
    played = [True, True, True, True, False, True]
    f = frame_from_arrays(pred, actual, played=played)
    m = M.compute_season_metrics(f, ks=(2, 3), replacement=100.0, min_gp=1)
    assert m["n_projected"] == 5 and m["n_played"] == 5
    assert m["n_projected_no_games"] == 1 and m["n_coverage_miss"] == 1
    assert m["coverage_miss_fp_share"] == pytest.approx(550 / 2350)
    # actual top-2 = {p1 (600), p6 (550)}; predicted top-2 = {p1, p2} -> overlap 1/2
    assert m["top2_hit"] == pytest.approx(0.5)
    assert m["top2_unprojected"] == 1                     # p6 is in the actual top-2 but unprojected
    # actual top-3 = {p1, p6, p2}; predicted {p1,p2,p3} -> 2/3
    assert m["top3_hit"] == pytest.approx(2 / 3)
    # Spearman over the 5 projected players, INCLUDING the zero-game one at actual 0
    assert m["spearman_total_fp"] == pytest.approx(M.spearman(pred[:5], actual[:5]))
    assert m["mae_total_fp"] == pytest.approx((0 + 0 + 100 + 100 + 200) / 5)
    assert m["replacement_level"] == 100.0


def test_min_gp_filters_fppg_metrics_only():
    f = frame_from_arrays([100, 100, 100], [100, 100, 100], actual_gp=[5, 30, 60], pred_gp=[50, 50, 50])
    m10 = M.compute_season_metrics(f, ks=(2,), replacement=0.0, min_gp=10)
    m1 = M.compute_season_metrics(f, ks=(2,), replacement=0.0, min_gp=1)
    # FPPG scored on 2 players (gp>=10) vs 3; total FP metric always on all 3
    spec = M.metric_spec("mae_fppg")
    p10, _ = M.select_arrays(f, spec, 10)
    p1, _ = M.select_arrays(f, spec, 1)
    assert len(p10) == 2 and len(p1) == 3
    assert m10["mae_total_fp"] == m1["mae_total_fp"]


def test_rank_only_nans_the_error_metrics():
    f = frame_from_arrays([3, 2, 1], [30, 20, 10])
    f[["proj_gp", "proj_fppg"]] = np.nan
    m = M.compute_season_metrics(f, ks=(2,), replacement=0.0, rank_only=True)
    assert m["spearman_total_fp"] == pytest.approx(1.0) and m["top2_hit"] == 1.0
    assert math.isnan(m["mae_total_fp"]) and math.isnan(m["vorp_weighted_mae"])


def test_calibration_shares():
    f = frame_from_arrays([1, 1], [1, 1])
    f["band_games"] = [50, 50]
    f["n_below_p10"] = [5, 5]
    f["n_above_p90"] = [10, 0]
    f["p50_games"] = [50, 50]
    f["n_below_p50"] = [25, 25]
    c = M.calibration_shares(f)
    assert c["share_below_p10"] == pytest.approx(0.10)
    assert c["share_above_p90"] == pytest.approx(0.10)
    assert c["share_in_p10_p90"] == pytest.approx(0.80)
    assert c["share_below_p50"] == pytest.approx(0.5)
    empty = M.calibration_shares(frame_from_arrays([1], [1]))
    assert math.isnan(empty["share_in_p10_p90"])


def test_metric_orientation_helper():
    assert M.metric_higher_is_better("spearman_fppg") is True
    assert M.metric_higher_is_better("top50_hit") is True
    assert M.metric_higher_is_better("mae_gp") is False
    assert M.metric_higher_is_better("bias_total_fp") is None
    assert M.metric_higher_is_better("top100_unprojected") is None
