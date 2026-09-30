"""NaiveLastSeason benchmark and the projector registry."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league

from src.contracts import Projector, season_start, validate_table
from src.models.baseline import BaselineProjector
from src.models.naive import NaiveLastSeason
from src.models.registry import available_projectors, get_projector, register_projector
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def naive(hist):
    return NaiveLastSeason().project(hist)


def test_naive_validates_and_names_itself(naive):
    validate_table(naive, "projections")
    assert (naive["model"] == "naive_last_season").all()
    assert (naive["season"] == TARGET).all()
    assert naive["player_id"].is_unique


def test_naive_is_exactly_last_seasons_per_game_line(hist, naive):
    last = hist.game_logs[hist.game_logs["season"] == hist.last_season]
    pid = int(last["player_id"].iloc[3])
    mine = last[last["player_id"] == pid]
    row = naive.set_index("player_id").loc[pid]
    assert row["proj_gp"] == len(mine)
    assert row["proj_mpg"] == pytest.approx(mine["min"].mean())
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m"):
        assert row[f"proj_{c}"] == pytest.approx(mine[c].mean())
    fp = fantasy_points_frame(mine, SCORING)
    assert row["proj_fppg"] == pytest.approx(fp.mean())
    assert row["proj_total_fp"] == pytest.approx(fp.sum())
    assert row["fppg_p10"] == pytest.approx(fp.quantile(0.10))
    assert row["fppg_p90"] == pytest.approx(fp.quantile(0.90))
    assert row["fppg_p10"] <= row["fppg_p50"] <= row["fppg_p90"]


def test_naive_has_no_row_for_players_without_a_last_season(hist, naive):
    last_ids = set(hist.game_logs.loc[hist.game_logs["season"] == hist.last_season, "player_id"])
    assert set(naive["player_id"]) == last_ids
    # Rookies of the target season and earlier retirees are absent by design.
    baseline = BaselineProjector().project(hist)
    assert set(baseline["player_id"]) - set(naive["player_id"])       # baseline covers extra players (rookies)


def test_naive_gp_is_bounded_by_schedule(hist, naive):
    assert naive["proj_gp"].max() <= hist.team_games.groupby(["season", "team_id"]).size().max()


def test_naive_ignores_the_future(tables, naive):
    t = {k: v.copy() for k, v in tables.items()}
    cut = t["game_logs"]["season"].map(season_start) >= season_start(TARGET)
    t["game_logs"].loc[cut, "pts"] += 100
    out = NaiveLastSeason().project(history_for(t))
    pd.testing.assert_frame_equal(naive, out, check_exact=True)


def test_naive_honours_the_scoring_argument(hist, naive):
    out = NaiveLastSeason(scoring={"PTS": 1.0}).project(hist)
    np.testing.assert_allclose(out["proj_fppg"], naive["proj_pts"])


# --------------------------------------------------------------------------- registry

def test_registry_lists_the_builtin_models():
    names = available_projectors()
    assert names == sorted(names)
    assert {"naive_last_season", "baseline"} <= set(names)


@pytest.mark.parametrize("name", ["naive_last_season", "baseline"])
def test_registry_projectors_follow_the_protocol(name, hist):
    p = get_projector(name)
    assert isinstance(p, Projector) and p.name == name
    out = p.project(hist)
    validate_table(out, "projections")
    assert (out["model"] == name).all()


def test_registry_returns_fresh_instances_and_passes_kwargs(hist):
    a, b = get_projector("baseline"), get_projector("baseline")
    assert a is not b
    p = get_projector("baseline", scoring={"PTS": 1.0})
    out = p.project(hist)
    np.testing.assert_allclose(out["proj_fppg"], out["proj_pts"])


def test_registry_unknown_name_lists_the_choices():
    with pytest.raises(KeyError, match="baseline"):
        get_projector("does_not_exist")


def test_registry_can_register_variants_but_not_silently_overwrite():
    register_projector("baseline_variant_for_test", lambda **kw: BaselineProjector(name="baseline_variant_for_test", **kw))
    try:
        assert "baseline_variant_for_test" in available_projectors()
        assert get_projector("baseline_variant_for_test").name == "baseline_variant_for_test"
        with pytest.raises(ValueError, match="already registered"):
            register_projector("baseline_variant_for_test", lambda **kw: BaselineProjector())
        register_projector("baseline_variant_for_test", lambda **kw: BaselineProjector(**kw), overwrite=True)
    finally:
        from src.models import registry

        registry._REGISTRY.pop("baseline_variant_for_test", None)
