"""DebutantBaselineProjector: veterans and rookies exactly the baseline's, debutants added and flagged, leakage-safe."""
import numpy as np
import pandas as pd
import pytest
from debutant_testkit import GHOST_ID, STASH_ID, TARGET, UDFA_ID, build
from model_testkit import history_for

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import validate_table
from src.models.baseline import BaselineProjector
from src.models.debutants import DebutantBaselineProjector, DebutantParams, effective_pick, fit_params, p_play
from src.models.registry import available_projectors, get_projector

T = build()


@pytest.fixture(scope="module")
def base():
    return BaselineProjector().project(history_for(T))


@pytest.fixture(scope="module")
def out():
    p = DebutantBaselineProjector(min_analog_seasons=2)
    return p, p.project(history_for(T))


def test_registry_has_the_debutant_family():
    for n in ("baseline_debut", "baseline_offseason_debut", "baseline_origin"):
        assert n in available_projectors()
    assert get_projector("baseline_debut").name == "baseline_debut"


def test_existing_players_are_exactly_the_baselines(base, out):
    _, o = out
    old = o[o["projection_class"].isin(["veteran", "rookie"])].set_index("player_id").sort_index()
    b = base.set_index("player_id").sort_index()
    assert list(old.index) == list(b.index)
    for c in ("proj_fppg", "proj_gp", "proj_mpg", "proj_total_fp", "fppg_p10", "fppg_p90"):
        np.testing.assert_allclose(old[c], b[c], rtol=1e-12)


def test_debutants_are_added_flagged_low_confidence_and_valid(out):
    _, o = out
    validate_table(o, "projections")
    d = o[o["projection_class"].isin(["stash", "undrafted"])].set_index("player_id")
    assert {STASH_ID, UDFA_ID, GHOST_ID} <= set(d.index)
    assert (d["confidence"] == "low").all() and d["p_play"].between(0.02, 0.97).all()
    assert d.loc[STASH_ID, "projection_class"] == "stash" and d.loc[UDFA_ID, "projection_class"] == "undrafted"
    assert (d["proj_gp"] > 0).all() and (d["proj_gp"] <= 82).all()
    assert d.loc[STASH_ID, "proj_gp"] > d.loc[UDFA_ID, "proj_gp"]      # an assumed-to-play stash out-plays a camp long shot


def test_effective_pick_discounts_old_picks_toward_undrafted():
    assert effective_pick(np.array([10.0]), np.array([0.0]), 0.5)[0] == 10.0
    assert effective_pick(np.array([10.0]), np.array([2.0]), 0.5)[0] == pytest.approx(61 - 51 * 0.25)
    assert effective_pick(np.array([10.0]), np.array([3.0]), 1.0)[0] == 10.0


def test_without_profiles_or_offseason_data_the_model_is_just_the_baseline(base):
    tables = {k: v for k, v in T.items() if k not in ("offseason_logs", "offseason_team_games", "player_profiles")}
    p = DebutantBaselineProjector()
    p._get = lambda history, key, explicit, loader: None
    o = p.project(history_for(tables))
    assert len(o) == len(base) and (o["projection_class"] != "stash").all()


def test_fit_params_needs_no_analogues_and_learns_from_them():
    p = DebutantBaselineProjector(min_analog_seasons=2)
    h = history_for(T)
    fitted = p.fit(h)
    assert not fit_params(fitted, None).enabled
    p._debutant_rows(h, fitted)
    prm = p.last_params
    assert prm.enabled and 0 < prm.gp_share["undrafted"] < 1 and prm.n_train["undrafted_camp"] > 0


def test_p_play_is_bounded_and_falls_back_to_the_class_rate():
    prm = DebutantParams()
    c = pd.DataFrame({"klass": ["stash", "undrafted"], "pre_gp_share": [np.nan, np.nan], "pre_mpg": [np.nan, np.nan], "sl_mpg": [np.nan, np.nan]})
    pp = p_play(prm, c)
    assert pp[0] == pytest.approx(0.9) and pp[1] == pytest.approx(prm.p_base)


@pytest.mark.slow
def test_the_projector_ignores_the_future():
    assert_projector_ignores_future(DebutantBaselineProjector(min_analog_seasons=2), T, TARGET)


def test_a_live_roster_snapshot_of_the_target_season_replaces_the_camp():
    snap = pd.DataFrame({"snapshot_date": [pd.Timestamp("2018-09-20")] * 2, "season": [TARGET] * 2, "player_id": [STASH_ID, GHOST_ID],
                         "player_name": ["Stash Guy", "Ghost Guy"], "team_id": [1, 2], "team_abbr": ["A", "B"]})
    o = DebutantBaselineProjector(min_analog_seasons=2, roster=snap).project(history_for(T))
    d = o[o["projection_class"].isin(["stash", "undrafted"])]
    assert set(d["player_id"]) == {STASH_ID, GHOST_ID}                       # UDFA_ID is in camp but not on the roster
    o2 = DebutantBaselineProjector(min_analog_seasons=2, roster=snap.assign(season="2010-11")).project(history_for(T))
    assert UDFA_ID in set(o2["player_id"])                                   # a stale-season snapshot is ignored
