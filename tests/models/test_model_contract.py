"""BaselineContractProjector: contract validity, fallback, planted-signal recovery, the honesty gate, leakage,
identities, registry wiring. Synthetic data only tests the code; the real result is in ADR 0013."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import PROJECTION_STATS, History, Projector, season_start, validate_table
from src.features.contract import contract_clock
from src.models.baseline import BaselineProjector
from src.models.contract_baseline import BaselineContractProjector
from src.models.registry import available_projectors, get_projector
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]
MIN_ROWS = 60      # the synthetic league is small; production uses MIN_FIT_ROWS
WALK_FORWARD: dict = {}   # base projections of past seasons are identical for every variant here: compute once


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def base(tables):
    return BaselineProjector().project(history_for(tables))


def _plant_contract_year_effect(tables, bump: int):
    """Give every first-rounder in his 4th scale season extra free-throw points in every season he had one."""
    t = {k: v.copy() for k, v in tables.items()}
    gl = t["game_logs"]
    starts = gl["season"].map(season_start)
    mask = np.zeros(len(gl), dtype=bool)
    for s in sorted(starts.unique()):
        idx = np.flatnonzero((starts == s).to_numpy())
        pids = gl["player_id"].to_numpy()[idx]
        clock = contract_clock(t["players"], f"{s}-{str(s + 1)[-2:]}", pids)
        mask[idx] = clock.is_contract_year
    for c in ("ftm", "fta", "pts"):
        gl.loc[mask, c] = gl.loc[mask, c] + bump
    return t


def proj(**kw):
    return BaselineContractProjector(min_fit_rows=MIN_ROWS, walk_forward_cache=WALK_FORWARD, **kw)


# --------------------------------------------------------------------------- registry and contract

def test_is_a_projector_and_registered():
    assert isinstance(BaselineContractProjector(), Projector)
    assert "baseline_contract" in available_projectors()
    p = get_projector("baseline_contract")
    assert isinstance(p, BaselineContractProjector) and p.name == "baseline_contract"


def test_output_validates_and_labels_the_model(tables):
    out = proj().project(history_for(tables))
    validate_table(out, "projections")
    assert (out["model"] == "baseline_contract").all() and out["player_id"].is_unique
    assert out[list(PROJECTION_STATS) + ["proj_fppg", "proj_total_fp", "proj_gp", "proj_mpg"]].notna().all().all()
    for c in ("contract_adj", "contract_factor", "contract_enabled", "contract_flag", "contract_scale_year"):
        assert c in out.columns


def test_deterministic(tables):
    h = history_for(tables)
    pd.testing.assert_frame_equal(proj().project(h), proj().project(h), check_exact=True)


# --------------------------------------------------------------------------- fallback

def test_too_little_history_gives_exactly_the_baseline():
    t = make_league(first_start=2018, last_start=2019)
    h = history_for(t, "2019-20")
    p = BaselineContractProjector()
    out = p.project(h)
    base = BaselineProjector().project(h)
    assert not out["contract_enabled"].any() and p.last_fit is not None and not p.last_fit.enabled
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)


def test_missing_draft_columns_are_harmless(tables):
    t = {k: v.copy() for k, v in tables.items()}
    for c in ("draft_year", "draft_round", "draft_number"):
        t["players"][c] = pd.array([pd.NA] * len(t["players"]), dtype="Int64")
    p = proj()
    out = p.project(history_for(t))
    assert not out["contract_enabled"].any()
    assert (out["contract_factor"] == 1).all() and (out["contract_adj"] == 0).all()


# --------------------------------------------------------------------------- planted signal and the gate

def test_recovers_a_planted_contract_year_effect(tables):
    t = _plant_contract_year_effect(tables, bump=8)
    p = BaselineContractProjector(min_fit_rows=MIN_ROWS, walk_forward_cache={})
    out = p.project(history_for(t))
    assert p.last_fit.enabled, p.last_fit.diagnostics
    cy = out["contract_flag"] == "contract_year"
    assert cy.any()
    assert out.loc[cy, "contract_adj"].mean() > 0.5
    no_clock = ~(out["contract_years_since_draft"] <= 5)     # veterans, undrafted, unknown: exactly zero
    assert no_clock.any() and (out.loc[no_clock, "contract_adj"] == 0).all()


def test_adjusted_stats_keep_the_fantasy_point_identity(tables):
    t = _plant_contract_year_effect(tables, bump=8)
    out = BaselineContractProjector(min_fit_rows=MIN_ROWS, walk_forward_cache={}).project(history_for(t))
    assert out["contract_enabled"].all()
    np.testing.assert_allclose(out["proj_fppg"], fantasy_points_frame(out, SCORING, prefix="proj_"), rtol=1e-9)
    np.testing.assert_allclose(out["proj_total_fp"], out["proj_fppg"] * out["proj_gp"], rtol=1e-9)
    assert out["contract_factor"].between(0.8, 1.25).all()
    assert (out["fppg_p10"] <= out["fppg_p90"]).all()


def test_switches_itself_off_when_the_clock_explains_nothing(tables, base):
    p = proj()
    out = p.project(history_for(tables))
    assert not p.last_fit.enabled, p.last_fit.diagnostics
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)


# --------------------------------------------------------------------------- leakage

def test_leak_check_passes_for_a_disabled_and_an_enabled_layer(tables):
    """The exact machinery behind ``--leak-check``: scrambling everything at or after the target changes nothing."""
    assert_projector_ignores_future(BaselineContractProjector(min_fit_rows=MIN_ROWS), tables, TARGET, seed=3)
    planted = _plant_contract_year_effect(tables, bump=8)
    assert_projector_ignores_future(BaselineContractProjector(min_fit_rows=MIN_ROWS), planted, TARGET, seed=3)


def test_future_draftees_do_not_change_projections(tables):
    """A player drafted after the target season is invisible: adding one changes no existing projection."""
    a = proj().project(history_for(tables))
    t = {k: v.copy() for k, v in tables.items()}
    extra = t["players"].iloc[[0]].copy()
    extra["player_id"] = 9_999_301
    extra["player_name"] = "Future Draftee"
    extra["draft_year"] = season_start(TARGET) + 2
    t["players"] = pd.concat([t["players"], extra], ignore_index=True)
    b = proj().project(history_for(t))
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_history_with_future_rows_is_rejected(tables):
    leaky = History(TARGET, tables["game_logs"], tables["team_games"], tables["players"], tables["player_season_bio"])
    with pytest.raises(AssertionError, match="leakage"):
        BaselineContractProjector().project(leaky)
