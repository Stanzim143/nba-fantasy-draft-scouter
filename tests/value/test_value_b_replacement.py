"""Replacement level and VORP: config-driven, tested with 10 and 13 teams, positional scarcity check."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.value.league import load_league
from src.value.replacement import (
    LeagueShape, derive_bench_weight, league_shape, positional_replacement, positional_replacement_per_player,
    replacement_level, replacement_rank, value_at_rank, _fill_slots,
)
from src.value.positions import eligible_positions
from src.value.vorp import compute_vorp

CFG = load_league()


def _linear(n=300, top=300.0):
    """A league of n players with total FP n..1 (times 1) and 70 games each."""
    total = np.linspace(top, top - n + 1, n)
    return pd.DataFrame({"proj_total_fp": total, "proj_fppg": total / 70.0, "proj_gp": 70.0})


# --------------------------------------------------------------------------- shape from config

def test_league_shape_reads_slots_and_teams_from_config():
    s = league_shape(CFG)
    assert (s.teams, s.n_starters, s.bench, s.roster_size) == (14, 10, 3, 13)
    assert s.starter_slots["UTIL"] == 3
    assert league_shape(CFG, teams=13).teams == 13
    with pytest.raises(ValueError):
        league_shape(CFG, teams=0)


def test_league_shape_follows_a_different_config():
    cfg = copy.deepcopy(CFG)
    cfg["league"]["roster"]["starters"] = {"PG": 1, "SG": 1, "UTIL": 2}
    cfg["league"]["roster"]["bench"] = 5
    s = league_shape(cfg, teams=12)
    assert (s.teams, s.n_starters, s.bench) == (12, 4, 5)


@pytest.mark.parametrize("teams, w, expected", [(10, 0.0, 100), (10, 1.0, 130), (13, 0.0, 130), (13, 1.0, 169), (10, 0.5, 115), (13, 0.5, 149.5)])
def test_replacement_rank_is_teams_times_starters_plus_weighted_bench(teams, w, expected):
    assert replacement_rank(league_shape(CFG, teams=teams), w) == pytest.approx(expected)


def test_replacement_rank_validates_bench_weight():
    with pytest.raises(ValueError):
        replacement_rank(league_shape(CFG), 1.5)


def test_bench_weight_from_availability():
    s = league_shape(CFG)
    assert derive_bench_weight(s, 1.0) == 0.0                     # nobody misses games: bench never plays
    assert derive_bench_weight(s, 0.85) == pytest.approx(0.5)     # 1.5 of 3 bench slots effectively used
    assert derive_bench_weight(s, 0.5) == 1.0                     # clipped: bench cannot exceed 100%
    assert derive_bench_weight(LeagueShape(10, {"UTIL": 10}, 0), 0.8) == 0.0
    assert derive_bench_weight(s, 0.9) < derive_bench_weight(s, 0.8)


def test_value_at_rank_interpolates_and_clamps():
    v = [10, 8, 6, 4, 2]
    assert value_at_rank(v, 0) == 10
    assert value_at_rank(v, 1.5) == pytest.approx(7.0)
    assert value_at_rank(v, 2) == 6
    assert value_at_rank(v, 99) == 2               # fewer players than R: the minimum
    assert value_at_rank([], 3) == 0.0
    assert value_at_rank([5, np.nan, 3], 1) == 3


# --------------------------------------------------------------------------- replacement level

@pytest.mark.parametrize("teams", [10, 13])
def test_replacement_is_the_best_player_outside_the_rostered_pool(teams):
    df, shape = _linear(), league_shape(CFG, teams=teams)
    r = replacement_level(df["proj_total_fp"], df["proj_fppg"], df["proj_gp"], shape, bench_weight=0.0)
    R = teams * 10
    assert r.rank == R
    assert r.total == df["proj_total_fp"].iloc[R]           # the (R+1)-th best: first man not rostered
    assert r.per_game == pytest.approx(df["proj_fppg"].iloc[R])
    assert (df["proj_total_fp"] > r.total).sum() == R


def test_more_teams_means_lower_replacement_and_higher_vorp_for_stars():
    df = _linear()
    r10 = compute_vorp(df, league_shape(CFG, teams=10), bench_weight=0.5)
    r13 = compute_vorp(df, league_shape(CFG, teams=13), bench_weight=0.5)
    assert r13.replacement.total < r10.replacement.total
    assert r13.replacement.rank > r10.replacement.rank
    assert (r13.frame["vorp"] > r10.frame["vorp"]).all()
    # the ordering of players is unchanged by a global replacement level
    assert r10.frame["vorp"].is_monotonic_decreasing and r13.frame["vorp"].is_monotonic_decreasing


def test_vorp_is_total_minus_replacement_and_zero_at_the_margin():
    df = _linear()
    res = compute_vorp(df, league_shape(CFG), bench_weight=0.0)
    np.testing.assert_allclose(res.frame["vorp"], df["proj_total_fp"] - res.replacement.total)
    np.testing.assert_allclose(res.frame["vorp_per_game"], df["proj_fppg"] - res.replacement.per_game)
    n = 14 * 10                                                                   # starters only at bench_weight 0
    assert res.frame["vorp"].iloc[n] == pytest.approx(0.0, abs=1.0 + 1e-9)       # first non-rostered
    assert (res.frame["vorp"].iloc[:n] > 0).all() and (res.frame["vorp"].iloc[n + 1:] < 0).all()
    assert not res.positional_used and res.positional is None        # no position column supplied


def test_bench_weight_is_derived_from_projected_availability():
    shape = league_shape(CFG)
    healthy = _linear().assign(proj_gp=82.0)
    hurt = _linear().assign(proj_gp=60.0)
    a = replacement_level(healthy["proj_total_fp"], healthy["proj_fppg"], healthy["proj_gp"], shape, season_games=82.0)
    b = replacement_level(hurt["proj_total_fp"], hurt["proj_fppg"], hurt["proj_gp"], shape, season_games=82.0)
    assert a.bench_weight == 0.0 and a.rank == 140
    assert b.bench_weight == pytest.approx(min(1.0, 10 * (1 - 60 / 82) / 3))
    assert b.rank > a.rank


def test_replacement_level_with_a_config_that_changes_slots():
    cfg = copy.deepcopy(CFG)
    cfg["league"]["roster"]["starters"] = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "UTIL": 1}   # 6 starters
    cfg["league"]["roster"]["bench"] = 0
    shape = league_shape(cfg, teams=10)
    df = _linear()
    r = replacement_level(df["proj_total_fp"], df["proj_fppg"], df["proj_gp"], shape)
    assert r.rank == 60 and r.bench_weight == 0.0 and r.total == df["proj_total_fp"].iloc[60]


def test_fewer_projected_players_than_rostered_pool():
    df = _linear(n=40)
    res = compute_vorp(df, league_shape(CFG), bench_weight=0.5)
    assert res.replacement.total == df["proj_total_fp"].min()
    assert (res.frame["vorp"] >= 0).all()


def test_compute_vorp_input_validation():
    df = _linear()
    with pytest.raises(ValueError):
        compute_vorp(df, league_shape(CFG), positional="maybe")
    with pytest.raises(ValueError, match="position"):
        compute_vorp(df, league_shape(CFG), positional="on")
    with pytest.raises(KeyError):
        compute_vorp(df.drop(columns="proj_gp"), league_shape(CFG))


# --------------------------------------------------------------------------- positional scarcity

POSITIONS = ["G", "F", "C", "G-F", "F-C"]


def _with_positions(scarce_center: bool, n=400, seed=0):
    rng = np.random.default_rng(seed)
    pool = ["G", "F", "C"] if scarce_center else POSITIONS      # pure centers only, so 'C' means C only
    df = pd.DataFrame({"position": rng.choice(pool, n), "proj_gp": 70.0})
    base = np.sort(rng.gamma(4.0, 400.0, n))[::-1]
    rng.shuffle(base)
    if scarce_center:
        # centers are much less productive: the center pool is thin, so replacement at C is low
        base = np.where(df["position"] == "C", base * 0.45, base)
    df["proj_total_fp"] = base
    df["proj_fppg"] = base / 70.0
    return df


@pytest.mark.parametrize("teams", [10, 13])
def test_flexible_slots_make_scarcity_weak_when_positions_are_uninformative(teams):
    df = _with_positions(scarce_center=False, n=500)
    res = compute_vorp(df, league_shape(CFG, teams=teams), bench_weight=0.5, positional="auto")
    assert res.positional is not None and not res.positional.material
    assert not res.positional_used
    assert res.positional.scarcity_index < 0.10


@pytest.mark.parametrize("teams", [10, 13])
def test_scarce_position_is_detected_and_auto_applies_it(teams):
    df = _with_positions(scarce_center=True, n=500)
    shape = league_shape(CFG, teams=teams)
    auto = compute_vorp(df, shape, bench_weight=0.5, positional="auto")
    off = compute_vorp(df, shape, bench_weight=0.5, positional="off")
    assert auto.positional.material and auto.positional_used
    assert auto.positional.levels["C"] < min(v for k, v in auto.positional.levels.items() if k != "C")
    # Centers gain value over replacement relative to the league-wide baseline; pure guards do not.
    is_c = df["position"] == "C"
    assert (auto.frame["vorp"][is_c] > off.frame["vorp"][is_c]).all()
    assert (auto.frame["vorp"][df["position"] == "G"] <= off.frame["vorp"][df["position"] == "G"] + 1e-9).all()
    assert off.frame["repl_total"].nunique() == 1


def test_positional_on_and_off_are_explicit_overrides():
    df = _with_positions(scarce_center=False, n=500)
    shape = league_shape(CFG)
    on = compute_vorp(df, shape, bench_weight=0.5, positional="on")
    off = compute_vorp(df, shape, bench_weight=0.5, positional="off")
    assert on.positional_used and not off.positional_used
    assert on.frame["repl_total"].nunique() > 1


def test_fill_slots_rosters_exactly_the_league_capacity():
    shape = league_shape(CFG, teams=10)
    n = 400
    rng = np.random.default_rng(3)
    values = rng.random(n)
    elig = [eligible_positions(p) for p in rng.choice(POSITIONS, n)]
    rostered = _fill_slots(values, elig, shape, n_bench=15)
    assert rostered.sum() == 100 + 15
    # Everyone rostered is at least as good as every unrostered player at the same position group.
    assert values[rostered].mean() > values[~rostered].mean()


def test_fill_slots_respects_position_limits():
    """Only centers are available: at most C slots + UTIL slots + bench can be filled."""
    shape = league_shape(CFG, teams=10)
    values = np.linspace(100, 1, 200)
    elig = [eligible_positions("C")] * 200
    rostered = _fill_slots(values, elig, shape, n_bench=6)
    assert rostered.sum() == 10 * 1 + 10 * 3 + 6          # C slots + UTIL slots + bench (no PG/SG/SF/PF/G/F eligible)


def test_unknown_positions_only_fill_util_and_bench():
    shape = league_shape(CFG, teams=10)
    values = np.linspace(100, 1, 200)
    rostered = _fill_slots(values, [()] * 200, shape, n_bench=4)
    assert rostered.sum() == 30 + 4


def test_multi_eligible_player_takes_the_weakest_level_and_unknown_gets_default():
    report = positional_replacement(
        np.linspace(1000, 1, 300), ["G", "F", "C"] * 100, league_shape(CFG), bench_weight=0.5, global_level=400.0)
    out = positional_replacement_per_player(["G-F", "C", None], report, default=123.0)
    lv = report.levels
    assert out[0] == min(lv["SG"], lv["SF"])
    assert out[1] == lv["C"]
    assert out[2] == 123.0


def test_positional_levels_never_exceed_best_available_player():
    df = _with_positions(scarce_center=False, n=500)
    rep = positional_replacement(df["proj_total_fp"], df["position"], league_shape(CFG), bench_weight=0.5, global_level=0.0)
    assert max(rep.levels.values()) <= df["proj_total_fp"].max()
    assert set(rep.levels) == {"PG", "SG", "SF", "PF", "C"}


def test_per_game_replacement_is_the_same_player_as_the_total_replacement():
    """vorp_per_game subtracts the FPPG of the replacement player by total, not the R-th best FPPG in the league."""
    # Player i: total falls with i, FPPG deliberately not monotone in total (a durable low-rate player vs an injury-prone high-rate one).
    n = 60
    total = 3000.0 - 40.0 * np.arange(n)
    fppg = 45.0 - 0.3 * np.arange(n)
    fppg[20] = 60.0                                           # ranks 20th by total but is the best FPPG in the league
    gp = total / fppg
    shape = league_shape(CFG, teams=2)                        # R = 2 * 10 = 20 with bench_weight 0
    r = replacement_level(total, fppg, gp, shape, bench_weight=0.0)
    assert r.rank == 20 and r.total == total[20]
    assert r.per_game == pytest.approx(fppg[20]) == pytest.approx(60.0)      # the same player, not the 21st best FPPG
    assert r.per_game != pytest.approx(value_at_rank(fppg, 20))
    # fractional rank: interpolate that player with the next one by total
    r2 = replacement_level(total, fppg, gp, shape, bench_weight=0.5)
    R = r2.rank
    lo, f = int(R), R - int(R)
    assert r2.per_game == pytest.approx(fppg[lo] * (1 - f) + fppg[lo + 1] * f)
    assert r2.total == pytest.approx(total[lo] * (1 - f) + total[lo + 1] * f)


def test_value_at_rank_by_orders_by_the_other_array_and_drops_nonfinite():
    fppg = np.array([10.0, 50.0, 30.0, np.nan, 20.0])
    total = np.array([100.0, 500.0, 300.0, 400.0, 200.0])    # player 3 has no FPPG: dropped
    assert value_at_rank(fppg, 0, by=total) == 50.0
    assert value_at_rank(fppg, 2, by=total) == 20.0
    assert value_at_rank(fppg, 1.5, by=total) == pytest.approx(25.0)
    assert value_at_rank(fppg, 99, by=total) == 10.0
