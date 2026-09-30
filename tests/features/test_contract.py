"""Unit tests for src.features.contract: the rookie-scale clock, its design matrix, the fitted adjustment
and its honesty gate. The clock is hand-checked against the CBA rules stated in the module docstring."""
import numpy as np
import pandas as pd
import pytest

from src.features.contract import (
    FEATURE_NAMES, MIN_CV_GAIN, ContractFit, build_training_set, clock_design, contract_clock, factor_from_adjustment,
    fit_adjustment, FACTOR_HI, FACTOR_LO,
)


def _players(rows):
    df = pd.DataFrame(rows, columns=["player_id", "draft_year", "draft_round", "draft_number"])
    for c in df.columns:
        df[c] = df[c].astype("Int64")
    return df


PLAYERS = _players([
    (1, 2023, 1, 1),      # 2026-27: year 4 of the scale, top pick -> contract year, extension eligible
    (2, 2024, 1, 22),     # year 3: option year
    (3, 2025, 1, 14),     # year 2
    (4, 2026, 1, 5),      # rookie year
    (5, 2022, 1, 25),     # year 5: post-scale (RFA / re-signed)
    (6, 2018, 1, 3),      # veteran: no clock
    (7, 2024, 2, 40),     # second-rounder, year 3
    (8, 2023, 1, 29),     # late first-rounder in his contract year
    (9, 2027, 1, 2),      # drafted AFTER the target season: never a clock
    (10, None, None, None),   # undrafted: no clock
    (11, 2025, 0, 0),     # round/pick 0 = "not available" in real data, year known -> round unknown
])


@pytest.fixture(scope="module")
def clock():
    return contract_clock(PLAYERS, "2026-27")


def _at(clock, pid, field):
    return getattr(clock, field)[list(clock.player_id).index(pid)]


def test_years_since_draft_and_scale_year(clock):
    assert [_at(clock, p, "years_since_draft") for p in (1, 2, 3, 4, 5, 6)] == [4, 3, 2, 1, 5, 9]
    assert [_at(clock, p, "scale_year") for p in (1, 2, 3, 4)] == [4, 3, 2, 1]
    assert np.isnan(_at(clock, 5, "scale_year")) and np.isnan(_at(clock, 6, "scale_year"))


def test_flags_follow_the_cba_stages(clock):
    assert _at(clock, 1, "is_contract_year") and _at(clock, 1, "is_extension_eligible")
    assert _at(clock, 8, "is_contract_year")
    assert _at(clock, 2, "is_option_year") and not _at(clock, 2, "is_contract_year")
    assert _at(clock, 5, "is_post_scale_year")
    for pid in (3, 4, 6, 7, 9, 10):
        assert not _at(clock, pid, "is_contract_year") and not _at(clock, pid, "is_post_scale_year")


def test_future_draft_year_and_undrafted_get_no_clock(clock):
    for pid in (9, 10):
        assert np.isnan(_at(clock, pid, "years_since_draft"))
        assert _at(clock, pid, "draft_round") == 0
    assert clock_design(clock)[[list(clock.player_id).index(p) for p in (9, 10)]].sum() == 0


def test_second_rounders_have_no_scale(clock):
    assert _at(clock, 7, "draft_round") == 2 and np.isnan(_at(clock, 7, "scale_year"))


def test_round_falls_back_to_pick_number_when_round_missing():
    p = _players([(1, 2026, None, 12), (2, 2026, None, 44), (3, 2026, None, None)])
    c = contract_clock(p, "2026-27")
    assert c.draft_round[0] == 1 and c.draft_round[1] == 2
    # a known draft year with neither round nor pick: no first-round clock is invented for him
    assert c.draft_round[2] == 0 and c.scale_year[2] != c.scale_year[2]


def test_design_columns_and_one_hot_structure(clock):
    X = clock_design(clock)
    assert X.shape == (len(PLAYERS), len(FEATURE_NAMES))
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    row = {p: X[i] for i, p in enumerate(clock.player_id)}
    assert row[1][col["r1_y4"]] == 1 and row[1][col["r1_y4_top"]] == 1 and row[1][col["r1_y4_late"]] == 0
    assert row[8][col["r1_y4"]] == 1 and row[8][col["r1_y4_late"]] == 1 and row[8][col["r1_y4_top"]] == 0
    assert row[2][col["r1_y3"]] == 1 and row[5][col["r1_y5"]] == 1 and row[4][col["r1_y1"]] == 1
    assert row[7][col["r2_y3"]] == 1
    assert row[6].sum() == 0 and row[9].sum() == 0 and row[10].sum() == 0


def test_clock_depends_only_on_draft_facts_and_season():
    """Changing anything except draft_year/round/number must not move the clock (no leakage channel)."""
    base = contract_clock(PLAYERS, "2026-27")
    noisy = PLAYERS.assign(player_id=PLAYERS["player_id"])
    noisy["extra"] = np.arange(len(noisy)) * 7.0
    again = contract_clock(noisy, "2026-27")
    np.testing.assert_array_equal(clock_design(base), clock_design(again))
    older = contract_clock(PLAYERS, "2025-26")
    assert _at(older, 1, "years_since_draft") == 3


# --------------------------------------------------------------------------- fit and gate

def _synthetic_training(n=6000, effect=2.0, seed=0, noise=4.0):
    rng = np.random.default_rng(seed)
    yrs = rng.integers(1, 9, n)
    rnd = rng.integers(1, 3, n)
    pick = rng.integers(1, 61, n)
    p = pd.DataFrame({"player_id": np.arange(n), "draft_year": 2020 - yrs + 1, "draft_round": rnd, "draft_number": pick})
    p = p.astype({c: "Int64" for c in p.columns})
    clock = contract_clock(p, "2020-21")
    X = clock_design(clock)
    y = rng.normal(0, noise, n) + effect * clock.is_contract_year
    return clock, X, y, np.full(n, 50.0), rng.integers(2016, 2026, n)


def test_fit_recovers_a_planted_contract_year_effect():
    clock, X, y, w, s = _synthetic_training()
    fit = fit_adjustment(X, y, w, s)
    assert fit.enabled, fit.diagnostics
    adj = fit.adjustment(clock)
    assert adj[clock.is_contract_year].mean() == pytest.approx(2.0, abs=0.7)
    veterans = clock.years_since_draft > 5
    assert veterans.any() and np.abs(adj[veterans]).max() == 0.0     # no clock, no adjustment


def test_fit_switches_itself_off_on_pure_noise():
    clock, X, y, w, s = _synthetic_training(effect=0.0, seed=3)
    fit = fit_adjustment(X, y, w, s)
    assert not fit.enabled
    assert fit.diagnostics["cv_gain_vs_zero"] < MIN_CV_GAIN
    assert (fit.adjustment(clock) == 0).all()


def test_fit_falls_back_on_too_little_data():
    clock, X, y, w, s = _synthetic_training(n=60)
    fit = fit_adjustment(X, y, w, s)
    assert not fit.enabled and "too few" in fit.diagnostics["reason"]
    assert isinstance(fit, ContractFit) and (fit.adjustment(clock) == 0).all()


def test_fit_needs_more_than_one_season():
    clock, X, y, w, s = _synthetic_training()
    assert not fit_adjustment(X, y, w, np.full(len(y), 2020)).enabled


def test_factor_clipped_and_low_projections_untouched():
    f = factor_from_adjustment(np.array([30.0, 30.0, 0.5, 30.0]), np.array([100.0, -100.0, 5.0, 0.0]))
    assert f[0] == FACTOR_HI and f[1] == FACTOR_LO and f[2] == 1.0 and f[3] == 1.0


def test_build_training_set_uses_only_each_rows_own_season():
    players = _players([(1, 2023, 1, 3), (2, 2019, 1, 9)])
    calls = []

    def project(s):
        calls.append(s)
        return pd.DataFrame({"player_id": [1, 2], "proj_fppg": [20.0, 25.0]})

    actual = pd.DataFrame({"player_id": [1, 2, 1, 2], "s": [2025, 2025, 2026, 2026], "gp": [60, 60, 60, 5],
                           "fppg": [24.0, 24.0, 30.0, 99.0]})
    ts = build_training_set(players, ["2025-26", "2026-27"], project, actual)
    assert calls == ["2025-26", "2026-27"]
    assert len(ts.resid) == 3                     # the 5-game row is dropped
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    # player 1 is in year 3 in 2025-26 (option year) and year 4 in 2026-27 (contract year)
    assert ts.design[0][col["r1_y3"]] == 1 and ts.design[2][col["r1_y4"]] == 1
    assert build_training_set(players, ["2025-26"], lambda s: None, actual) is None
