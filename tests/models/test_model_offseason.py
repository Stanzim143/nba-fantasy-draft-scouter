"""BaselineOffseasonProjector: graceful fallback, planted-signal recovery, the honesty gate, leakage, identities."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import history_for, make_league
from offseason_testkit import make_offseason

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import PROJECTION_STATS, validate_table
from src.models.baseline import BaselineProjector
from src.models.offseason_baseline import DISPLAY_COLUMNS, BaselineOffseasonProjector
from src.models.registry import available_projectors, get_projector
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]
TARGET = "2018-19"
MIN_ROWS = 60      # the synthetic league is small; production uses MIN_FIT_ROWS
WALK_FORWARD = {}  # base projections of past seasons are identical for every variant here: compute them once


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def signal(tables):
    return make_offseason(tables, "signal", seed=1)


@pytest.fixture(scope="module")
def noise(tables):
    return make_offseason(tables, "noise", seed=1)


@pytest.fixture(scope="module")
def base(tables):
    return BaselineProjector().project(history_for(tables))


def projector(data, **kw):
    logs, tg = data
    return BaselineOffseasonProjector(offseason_logs=logs, offseason_team_games=tg, min_fit_rows=MIN_ROWS,
                                      walk_forward_cache=WALK_FORWARD, **kw)


@pytest.fixture(scope="module")
def fitted_signal(tables, signal):
    p = projector(signal)
    return p, p.project(history_for(tables))


# --------------------------------------------------------------------------- registry

def test_registry_exposes_the_offseason_family():
    names = available_projectors()
    for n in ("baseline_offseason", "baseline_summer_league", "baseline_preseason", "baseline_offseason_rich",
              "baseline_summer_league_rich"):
        assert n in names
        assert get_projector(n).name == n
    assert get_projector("baseline_summer_league").contexts == ("summer_league",)
    assert get_projector("baseline_preseason").contexts == ("preseason",)
    assert get_projector("baseline_offseason").contexts == ("summer_league", "preseason")
    assert get_projector("baseline_offseason_rich").components is True
    assert get_projector("baseline_offseason", preseason_fraction=0.5).preseason_fraction == 0.5


# --------------------------------------------------------------------------- fallback

def test_without_offseason_data_the_projection_equals_the_base(tables, base):
    p = BaselineOffseasonProjector(offseason_logs=pd.DataFrame(), offseason_team_games=None)
    out = p.project(history_for(tables))
    assert p.last_fit is None and not out["offseason_enabled"].any()
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True))
    assert (out["offseason_adj"] == 0).all() and (out["offseason_factor"] == 1).all()
    assert (out["model"] == "baseline_offseason").all()
    validate_table(out, "projections")


def test_a_missing_store_falls_back_quietly(tables, base, tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    out = BaselineOffseasonProjector().project(history_for(tables))
    assert not out["offseason_enabled"].any()
    np.testing.assert_allclose(out["proj_fppg"], base["proj_fppg"])


# --------------------------------------------------------------------------- the honesty gate

def test_pure_noise_switches_the_layer_off_and_returns_the_base(tables, base, noise):
    p = projector(noise)
    out = p.project(history_for(tables))
    assert not p.last_fit.enabled and "below" in p.last_fit.diagnostics["reason"]
    assert not out["offseason_enabled"].any()
    np.testing.assert_allclose(out["proj_fppg"], base["proj_fppg"])
    np.testing.assert_allclose(out["proj_total_fp"], base["proj_total_fp"])


def test_a_planted_signal_is_found_and_applied(fitted_signal, base):
    p, out = fitted_signal
    assert p.last_fit.enabled and p.last_fit.diagnostics["cv_gain_vs_zero"] > 0.05
    assert out["offseason_enabled"].all()
    assert (out["offseason_adj"] != 0).sum() > 0.5 * len(out)
    m = out.merge(base[["player_id", "proj_fppg"]], on="player_id", suffixes=("", "_base"))
    np.testing.assert_allclose(m["proj_fppg"] - m["proj_fppg_base"], m["offseason_adj"], atol=1e-9)


def test_the_adjustment_correlates_with_how_the_season_actually_went(tables, fitted_signal, base):
    """The layer's whole purpose: its uplift should point toward what happened (planted, so it must)."""
    _, out = fitted_signal
    gl = tables["game_logs"]
    s = gl[gl["season"] == TARGET].copy()
    s["fp"] = fantasy_points_frame(s, SCORING)
    actual = s.groupby("player_id")["fp"].agg(["mean", "size"])
    actual = actual[actual["size"] >= 10]["mean"]
    m = out.set_index("player_id").join(actual.rename("actual"), how="inner")
    resid = m["actual"] - (m["proj_fppg"] - m["offseason_adj"])
    assert np.corrcoef(m["offseason_adj"], resid)[0, 1] > 0.3
    assert len(m) > 40


def test_stats_scale_together_so_box_score_identities_and_scoring_hold(fitted_signal, base):
    _, out = fitted_signal
    m = out.merge(base, on="player_id", suffixes=("", "_b"))
    ratio = m["proj_pts"] / m["proj_pts_b"]
    for col in PROJECTION_STATS:                                   # one factor per player across every counting stat
        np.testing.assert_allclose(m[col] / m[f"{col}_b"], ratio, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(m["offseason_factor"], ratio, rtol=1e-9)
    np.testing.assert_allclose(out["proj_fppg"], fantasy_points_frame(out, SCORING, prefix="proj_"), rtol=1e-9)
    np.testing.assert_allclose(out["proj_total_fp"], out["proj_fppg"] * out["proj_gp"], rtol=1e-9)
    assert (out["proj_fgm"] <= out["proj_fga"] + 1e-9).all() and (out["proj_ftm"] <= out["proj_fta"] + 1e-9).all()
    assert (out["proj_fg3m"] <= out["proj_fgm"] + 1e-9).all()
    validate_table(out, "projections")
    for c in ("fppg_p10", "fppg_p50", "fppg_p90"):                 # the floor/median/ceiling move with the projection
        np.testing.assert_allclose(m[c] / m[f"{c}_b"], ratio, rtol=1e-9)
    assert (out["fppg_p10"] <= out["fppg_p50"] + 1e-9).all() and (out["fppg_p50"] <= out["fppg_p90"] + 1e-9).all()


def test_games_played_minutes_and_the_player_set_are_untouched(fitted_signal, base):
    _, out = fitted_signal
    assert set(out["player_id"]) == set(base["player_id"]) and len(out) == len(base)
    m = out.merge(base, on="player_id", suffixes=("", "_b"))
    np.testing.assert_allclose(m["proj_gp"], m["proj_gp_b"])
    np.testing.assert_allclose(m["proj_mpg"], m["proj_mpg_b"])
    assert (m["is_rookie"] == m["is_rookie_b"]).all()


def test_players_without_an_offseason_line_are_not_moved(fitted_signal):
    _, out = fitted_signal
    none = out[out[DISPLAY_COLUMNS].isna().all(axis=1)]
    assert len(none) > 0 and (none["offseason_adj"] == 0).all() and (none["offseason_factor"] == 1).all()


def test_display_columns_report_the_players_own_line(tables, fitted_signal, signal):
    _, out = fitted_signal
    logs, _ = signal
    tgt = logs[(logs["event_season"] == TARGET) & (logs["context"] == "preseason")]
    pid = int(tgt["player_id"].iloc[0])
    row = out[out["player_id"] == pid].iloc[0]
    assert row["pre_gp"] == (tgt["player_id"] == pid).sum()
    assert row["pre_mpg"] == pytest.approx(tgt[tgt["player_id"] == pid]["min"].mean())


# --------------------------------------------------------------------------- variants

def test_summer_league_only_ignores_the_preseason(tables, signal):
    out = projector(signal, contexts=("summer_league",)).project(history_for(tables))
    assert out["pre_gp"].isna().all() and out["sl_gp"].notna().any()


def test_preseason_only_ignores_summer_league(tables, signal):
    out = projector(signal, contexts=("preseason",)).project(history_for(tables))
    assert out["sl_gp"].isna().all() and out["pre_gp"].notna().any()


def test_as_of_before_the_preseason_hides_it(tables, signal):
    out = projector(signal, as_of=pd.Timestamp(2018, 9, 15).date()).project(history_for(tables))
    assert out["pre_gp"].isna().all() and out["sl_gp"].notna().any()


def test_a_partial_preseason_shows_fewer_games(tables, signal):
    full = projector(signal).project(history_for(tables)).set_index("player_id")["pre_gp"]
    part = projector(signal, preseason_fraction=0.34).project(history_for(tables)).set_index("player_id")["pre_gp"]
    both = pd.concat([full, part], axis=1, keys=["full", "part"]).dropna()
    assert (both["part"] <= both["full"]).all() and (both["part"] < both["full"]).any()


def test_the_rich_variant_runs_and_fits_the_wider_design(tables, signal):
    logs, tg = signal
    p = BaselineOffseasonProjector(offseason_logs=logs, offseason_team_games=tg, components=True, min_fit_rows=MIN_ROWS,
                                   walk_forward_cache=WALK_FORWARD)
    out = p.project(history_for(tables))
    assert p.last_fit.components is True
    validate_table(out, "projections")
    if p.last_fit.enabled:
        assert len(p.last_fit.coefs) > 13


# --------------------------------------------------------------------------- leakage

def test_offseason_rows_tagged_for_the_target_or_later_are_ignored(tables, signal):
    logs, tg = signal
    clean = projector((logs, tg)).project(history_for(tables))
    future = logs[logs["season"].map(lambda s: int(s[:4])) >= 2018].copy()            # tag >= target: events of 2019-20+
    assert len(future) > 0
    bent = logs.copy()
    bent.loc[future.index, "min"] = bent.loc[future.index, "min"] * 3 + 40
    bent.loc[future.index, "pts"] = bent.loc[future.index, "pts"] + 9
    bent.loc[future.index, "player_id"] = bent.loc[future.index, "player_id"].iloc[::-1].to_numpy()
    out = projector((bent, tg)).project(history_for(tables))
    pd.testing.assert_frame_equal(out.reset_index(drop=True), clean.reset_index(drop=True))


def test_target_season_events_are_used_but_the_next_summers_are_not(tables, signal):
    """July and October before the season are legitimate evidence; the ones after it are not."""
    logs, tg = signal
    p = projector((logs, tg))
    out = p.project(history_for(tables))
    seen = out["pre_gp"].notna().sum()
    assert seen > 0
    only_after = logs[logs["event_season"] != TARGET]
    out2 = projector((only_after, tg)).project(history_for(tables))
    assert out2["pre_gp"].isna().all() and out2["sl_gp"].isna().all()


def test_the_backtest_future_invariance_check_passes_with_extras(tables, signal):
    logs, tg = signal
    with_extras = {**tables, "offseason_logs": logs.copy(), "offseason_team_games": tg.copy()}
    assert_projector_ignores_future(BaselineOffseasonProjector(min_fit_rows=MIN_ROWS), with_extras, TARGET)


def test_the_future_invariance_check_passes_with_injected_frames(tables, signal):
    logs, tg = signal
    logs, tg = logs.copy(), tg.copy()
    proj = BaselineOffseasonProjector(offseason_logs=logs, offseason_team_games=tg, min_fit_rows=MIN_ROWS)
    assert_projector_ignores_future(proj, {**tables, "offseason_logs": logs, "offseason_team_games": tg}, TARGET)


def test_projection_is_deterministic(tables, signal):
    a = projector(signal).project(history_for(tables))
    b = projector(signal).project(history_for(tables))
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- walk-forward memo

class CountingBase(BaselineProjector):
    calls = 0

    def project(self, history):
        CountingBase.calls += 1
        return super().project(history)


def test_walk_forward_projections_are_computed_once_per_season(tables, signal):
    logs, tg = signal
    CountingBase.calls = 0
    p = BaselineOffseasonProjector(base=CountingBase(), offseason_logs=logs, offseason_team_games=tg, min_fit_rows=MIN_ROWS)
    p.project(history_for(tables, "2018-19"))
    train = CountingBase.calls - 1                               # every call but the target's own is a walk-forward season
    assert train >= 3
    p.project(history_for(tables, "2018-19"))
    assert CountingBase.calls == 1 + train + 1                   # same target again: only the target is re-projected
    p.project(history_for(tables, "2019-20"))
    # a later target re-projects itself plus the one season that is new to its walk-forward window (2018-19)
    assert CountingBase.calls == 1 + train + 1 + 2


def test_a_shared_cache_lets_variants_reuse_the_same_walk_forward_projections(tables, signal):
    logs, tg = signal
    shared: dict = {}
    CountingBase.calls = 0
    a = BaselineOffseasonProjector(base=CountingBase(), offseason_logs=logs, offseason_team_games=tg,
                                   min_fit_rows=MIN_ROWS, walk_forward_cache=shared)
    a.project(history_for(tables))
    first = CountingBase.calls
    b = BaselineOffseasonProjector(base=CountingBase(), offseason_logs=logs, offseason_team_games=tg, contexts=("preseason",),
                                   min_fit_rows=MIN_ROWS, walk_forward_cache=shared)
    b.project(history_for(tables))
    assert CountingBase.calls == first + 1                       # the second variant only projected the target itself
