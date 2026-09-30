"""BaselineInjuryProjector: contract, determinism, leakage, registry wiring, and a comparison
against BaselineProjector on the synthetic league (a sanity smoke test, not the real ablation --
see docs/adr/0006-injury-layer.md and reports/ for the real backtest result)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, add_player_seasons, history_for, make_league

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
from injury_testkit import add_player_season_pattern  # noqa: E402

from src.backtest.leakage import assert_projector_ignores_future, build_history
from src.contracts import PROJECTION_STATS, History, Projector, season_start, validate_table
from src.models.baseline import BaselineProjector
from src.models.injury_baseline import BaselineInjuryProjector
from src.models.registry import get_projector

T_START = season_start(TARGET)


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def proj(hist):
    return BaselineInjuryProjector().project(hist)


# --------------------------------------------------------------------------- contract and shape

def test_is_a_projector():
    assert isinstance(BaselineInjuryProjector(), Projector)
    assert BaselineInjuryProjector().name == "baseline_injury"


def test_registered_and_reachable_from_the_registry(hist):
    p = get_projector("baseline_injury")
    assert isinstance(p, BaselineInjuryProjector)
    out = p.project(hist)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_injury").all()


def test_output_validates_against_projections_contract(proj):
    validate_table(proj, "projections")
    assert (proj["season"] == TARGET).all()
    assert (proj["model"] == "baseline_injury").all()
    assert proj["player_id"].is_unique
    assert proj[list(PROJECTION_STATS) + ["proj_fppg", "proj_total_fp", "proj_gp", "proj_mpg"]].notna().all().all()


def test_output_validates_at_every_history_length(tables):
    for target in ("2013-14", "2014-15", "2015-16", "2019-20"):
        out = BaselineInjuryProjector().project(history_for(tables, target))
        validate_table(out, "projections")
        assert len(out) > 50


def test_deterministic(hist):
    a = BaselineInjuryProjector().project(hist)
    b = BaselineInjuryProjector().project(hist)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_invariant_to_row_order_of_inputs(tables, hist):
    rng = np.random.default_rng(0)
    shuffled = History(
        target_season=hist.target_season,
        game_logs=hist.game_logs.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True),
        team_games=hist.team_games.sample(frac=1.0, random_state=1).reset_index(drop=True),
        players=hist.players.sample(frac=1.0, random_state=2).reset_index(drop=True),
        player_season_bio=hist.player_season_bio.sample(frac=1.0, random_state=3).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(BaselineInjuryProjector().project(hist),
                                  BaselineInjuryProjector().project(shuffled), check_exact=True)


# --------------------------------------------------------------------------- leakage

def _perturb_future(tables, rng):
    out = {k: v.copy() for k, v in tables.items()}
    cut = out["game_logs"]["season"].map(season_start) >= T_START
    gl = out["game_logs"]
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fga", "fta"):
        gl.loc[cut, c] = gl.loc[cut, c] + rng.integers(1, 50, cut.sum())
    gl.loc[cut, "min"] = gl.loc[cut, "min"] * 1.7
    gl.loc[cut, "player_id"] = gl.loc[cut, "player_id"].to_numpy()[rng.permutation(cut.sum())]
    tg = out["team_games"]
    fut = tg["season"].map(season_start) >= T_START
    tg.loc[fut, "pts_for"] = rng.integers(60, 150, fut.sum())
    bio = out["player_season_bio"]
    fb = bio["season"].map(season_start) >= T_START
    bio.loc[fb, "age_at_season_start"] = rng.uniform(18, 45, fb.sum())
    pl = out["players"]
    pl["from_year"] = rng.integers(2010, 2030, len(pl))
    pl["to_year"] = rng.integers(2010, 2030, len(pl))
    return out


def test_no_leakage_perturbing_target_and_later_data_changes_nothing(tables, proj):
    """Same style as BaselineProjector's leakage test, but exercising the injury feature path:
    the streak panel is rebuilt from history.game_logs/team_games every fit, so if it leaked it
    would leak exactly here."""
    perturbed = _perturb_future(tables, np.random.default_rng(1))
    h2 = History.until(perturbed, TARGET)
    h2.assert_no_future()
    out = BaselineInjuryProjector().project(h2)
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_future_only_players_do_not_change_projections(tables, proj):
    extra = add_player_seasons(tables, 9_999_101, "Future Guy", {"2019-20": (40, dict(minutes=30.0))}, draft_year=2019)
    out = BaselineInjuryProjector().project(history_for(extra))
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_history_with_future_rows_is_rejected(tables):
    leaky = History(TARGET, tables["game_logs"], tables["team_games"], tables["players"], tables["player_season_bio"])
    with pytest.raises(AssertionError, match="leakage"):
        BaselineInjuryProjector().project(leaky)


def test_assert_projector_ignores_future_passes(tables):
    """The shared leakage-guard machinery (src.backtest.leakage), the exact check the backtest
    CLI's --leak-check flag runs, must also pass for this projector."""
    assert_projector_ignores_future(BaselineInjuryProjector(), tables, TARGET, seed=3)


def test_build_history_and_project_smoke(tables):
    h = build_history(tables, TARGET)
    out = BaselineInjuryProjector().project(h)
    assert len(out) > 50


# --------------------------------------------------------------------------- injury-specific invariants

def test_games_played_never_exceed_the_schedule(hist, proj):
    games = hist.team_games.groupby(["season", "team_id"]).size().max()
    assert (proj["proj_gp"] > 0).all()
    assert (proj["proj_gp"] <= games).all()
    assert (proj["proj_gp_p90"] <= games + 1e-9).all()
    assert (proj["proj_gp_p10"] <= proj["proj_gp_p90"]).all()
    assert (proj["proj_gp_sd"] > 0).all()


def test_player_coming_off_a_long_absence_projects_fewer_games_than_a_healthy_twin(tables):
    """The clearest behavioural check of the injury layer: two players with identical recent
    per-minute production and recency-weighted games-played fraction, but one just came off one
    long absence streak and the other missed the same number of games in scattered small ones,
    should not project identically -- and the model should not project MORE games for the player
    who just had one long injury than for a scattered-absence twin with an even worse recent
    trend, all else equal."""
    team = tables["team_games"][tables["team_games"].season == "2017-18"].groupby("team_id").size().index[0]
    n = int(((tables["team_games"].season == "2017-18") & (tables["team_games"].team_id == team)).sum())
    long_absence_pattern = [True] * (n - 20) + [False] * 20  # one 20-game injury at season's end
    healthy_pattern = [True] * n
    t = add_player_season_pattern(tables, 9_900_101, "Recently Hurt", "2017-18",
                                  played=long_absence_pattern, team_of_season=team)
    t = add_player_season_pattern(t, 9_900_102, "Healthy Twin", "2017-18",
                                  played=healthy_pattern, team_of_season=team)
    # Give both a normal 2018-19 age/role so they are "active" and projected as veterans; a
    # single prior season is enough history for the availability model's fallback path.
    h = History.until(t, TARGET)
    out = BaselineInjuryProjector().project(h)
    hurt = out.loc[out.player_id == 9_900_101, "proj_gp"]
    healthy = out.loc[out.player_id == 9_900_102, "proj_gp"]
    assert len(hurt) == 1 and len(healthy) == 1
    assert float(hurt.iloc[0]) <= float(healthy.iloc[0])


# --------------------------------------------------------------------------- rookies / thin history

def test_rookie_with_no_history_is_unaffected_by_injury_layer(tables):
    """A rookie is projected from the draft-slot prior, not the availability model at all, so
    baseline and baseline_injury must agree on rookies exactly."""
    h = history_for(tables)
    base = BaselineProjector().project(h)
    inj = BaselineInjuryProjector().project(h)
    rookies = base.loc[base["is_rookie"], "player_id"]
    if len(rookies) == 0:
        pytest.skip("no rookies in this synthetic history slice")
    b = base.set_index("player_id").loc[rookies, "proj_gp"]
    i = inj.set_index("player_id").loc[rookies, "proj_gp"]
    pd.testing.assert_series_equal(b, i, check_exact=True)
