"""Gap-based tiering: cliffs, monotonicity, size limits, invariances."""
import numpy as np
import pytest

from src.value.tiers import assign_tiers


def _clustered(sizes=(6, 8, 10), centers=(900.0, 600.0, 300.0), spread=8.0, seed=0):
    rng = np.random.default_rng(seed)
    parts = [c - np.sort(rng.uniform(0, spread * s / 6, s)) for s, c in zip(sizes, centers)]
    return np.concatenate(parts)


def test_obvious_cliffs_start_new_tiers():
    v = _clustered()
    t = assign_tiers(v, floor=None, min_tier_size=3)
    assert set(t[:6]) == {1} and set(t[6:14]) == {2} and set(t[14:]) == {3}


def test_players_at_or_below_the_floor_share_the_last_tier():
    v = np.r_[_clustered(), [0.0, -5.0, -40.0, -100.0]]
    t = assign_tiers(v, floor=0.0, min_tier_size=3)
    assert set(t[-4:]) == {t.max()}
    assert t.max() == t[:-4].max() + 1
    assert (t[:-4] < t.max()).all()


def test_tiers_are_monotone_in_value_for_random_data():
    for seed in range(25):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 400))
        v = rng.gamma(2.0, 300.0, n) - 250.0
        t = assign_tiers(v)
        order = np.argsort(-v, kind="stable")
        assert (np.diff(t[order]) >= 0).all(), f"seed {seed}"
        assert t.min() == 1 and t.max() <= 13


def test_input_order_does_not_matter():
    rng = np.random.default_rng(4)
    v = rng.gamma(2.0, 300.0, 200) - 200.0
    perm = rng.permutation(len(v))
    t, tp = assign_tiers(v), assign_tiers(v[perm])
    np.testing.assert_array_equal(t[perm], tp)


def test_equal_values_share_a_tier():
    v = np.r_[np.full(5, 500.0), np.full(5, 500.0), _clustered()[:12]]
    t = assign_tiers(v)
    assert len(set(t[:10])) == 1


def test_min_tier_size_and_max_tiers_are_respected():
    rng = np.random.default_rng(1)
    v = np.sort(rng.gamma(1.5, 400.0, 300))[::-1] + 1.0
    for min_size in (3, 6, 10):
        for max_tiers in (3, 6, 12):
            t = assign_tiers(v, floor=0.0, min_tier_size=min_size, max_tiers=max_tiers)
            counts = np.bincount(t)[1:]
            assert t.max() <= max_tiers
            assert (counts[:-1] >= min_size).all() or len(counts) == 1
            assert (np.diff(t) >= 0).all()


def test_max_tiers_of_one_gives_a_single_tier_above_floor():
    t = assign_tiers(_clustered(), floor=0.0, max_tiers=1)
    assert set(t) == {1}


def test_smooth_decline_has_no_spurious_cliffs():
    v = np.linspace(1000, 1, 300)
    t = assign_tiers(v, floor=None)
    assert t.max() == 1


def test_a_lone_outlier_at_the_top_is_below_min_size_so_it_does_not_form_a_tier():
    v = np.r_[2000.0, np.linspace(900, 100, 60)]
    t = assign_tiers(v, floor=None, min_tier_size=3)
    assert t[0] == t[1]                       # one man cannot be a tier of his own with min size 3
    t1 = assign_tiers(v, floor=None, min_tier_size=1)
    assert t1[0] == 1 and t1[1] == 2          # ... but with min size 1 he can


def test_degenerate_inputs():
    assert assign_tiers([]).tolist() == []
    assert assign_tiers([5.0]).tolist() == [1]
    assert assign_tiers([-1.0, -2.0]).tolist() == [1, 1]          # everybody replaceable: one tier
    assert assign_tiers([3.0, 3.0, 3.0]).tolist() == [1, 1, 1]
    assert assign_tiers([10.0, 0.0, -3.0]).tolist() == [1, 2, 2]
    with pytest.raises(TypeError):
        assign_tiers(5.0)


def test_max_tiers_caps_above_replacement_tiers_and_the_replaceable_tier_is_extra():
    """``max_tiers`` limits the tiers above replacement; the replaceable tier is added on top (at most max_tiers + 1)."""
    v = np.r_[_clustered(sizes=(6, 6, 6, 6, 6, 6), centers=(1500.0, 1200.0, 900.0, 600.0, 300.0, 100.0), spread=4.0),
              [0.0, -5.0, -40.0, -100.0]]
    uncapped = assign_tiers(v, floor=0.0, min_tier_size=3, max_tiers=12)
    assert uncapped[:-4].max() == 6                           # six clusters, six above-replacement tiers
    capped = assign_tiers(v, floor=0.0, min_tier_size=3, max_tiers=3)
    assert capped[:-4].max() == 3                             # max_tiers above replacement ...
    assert set(capped[-4:]) == {4} and capped.max() == 4      # ... plus one replaceable tier: max_tiers + 1
    default = assign_tiers(np.r_[_clustered(sizes=(4,) * 14, centers=tuple(2000.0 - 140.0 * i for i in range(14)),
                                            spread=2.0), [-1.0, -2.0]], floor=0.0, min_tier_size=3)
    assert default[:-2].max() == 12 and default.max() == 13   # default max_tiers = 12 -> up to 13 tiers on a board
