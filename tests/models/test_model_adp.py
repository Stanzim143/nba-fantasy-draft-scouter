"""ADP-aware projectors (ADR 0032): registry, contract, degradation without ADP, use of the target season's ADP, no future."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league, spearman

from src.backtest import actuals as act
from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import Projector, season_start, validate_table
from src.models.adp_baseline import AdpDebutantProjector, BaselineAdpProjector
from src.models.baseline import BaselineProjector
from src.models.registry import get_projector
from src.value.league import load_league

SCORING = load_league()["scoring"]
EMPTY = pd.DataFrame({"season": pd.Series([], dtype=object), "player_id": pd.Series([], dtype="int64"),
                      "adp": pd.Series([], dtype="float64")})


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


def _oracle_adp(tables, seasons, noise=0.0):
    """ADP = rank of the realised season total (an oracle), optionally noised; every player with games is listed."""
    rows = []
    rng = np.random.default_rng(3)
    for s in seasons:
        a = act.season_actuals(tables["game_logs"], s, SCORING)
        score = a["total_fp"].to_numpy() * (1 + noise * rng.standard_normal(len(a)))
        rank = pd.Series(-score).rank(method="first").to_numpy()
        rows.append(pd.DataFrame({"season": s, "player_id": a["player_id"].to_numpy(), "adp": rank}))
    return pd.concat(rows, ignore_index=True)


def test_registry_and_contract(hist):
    for name in ("baseline_adp", "baseline_hurdle_adp"):
        p = get_projector(name, adp=EMPTY)
        assert isinstance(p, Projector) and p.name == name and isinstance(p, BaselineAdpProjector)
        validate_table(p.project(hist), "projections")
    assert isinstance(get_projector("baseline_hurdle_adp").config.appearance_hurdle, bool)
    deb = get_projector("baseline_hurdle_adp_offseason_debut", offseason_logs=None)
    assert deb.name == "baseline_hurdle_adp_offseason_debut" and isinstance(deb.base, AdpDebutantProjector)


def test_without_any_adp_it_is_the_base_model(hist):
    base = BaselineProjector().project(hist)
    out = BaselineAdpProjector(adp=EMPTY).project(hist)
    cols = [c for c in base.columns if c not in ("model",)]
    assert np.allclose(out[cols].select_dtypes("number").to_numpy(), base[cols].select_dtypes("number").to_numpy(),
                       atol=1e-6, equal_nan=True)


def test_oracle_adp_moves_the_projection_toward_reality(tables, hist):
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, season_start(TARGET) + 1)]
    adp = _oracle_adp(tables, seasons)
    base = BaselineProjector().project(hist).set_index("player_id")
    withadp = BaselineAdpProjector(adp=adp).project(hist).set_index("player_id")
    actual = act.season_actuals(tables["game_logs"], TARGET, SCORING).set_index("player_id")["total_fp"]
    common = base.index.intersection(actual.index)
    assert spearman(withadp.loc[common, "proj_total_fp"], actual[common]) > spearman(base.loc[common, "proj_total_fp"], actual[common])


def test_the_target_seasons_adp_is_used_and_later_seasons_are_not(tables, hist):
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, season_start(TARGET) + 2)]
    adp = _oracle_adp(tables, seasons)
    ref = BaselineAdpProjector(adp=adp).project(hist)
    later = adp["season"].map(season_start) > season_start(TARGET)
    perturbed = adp.copy()
    perturbed.loc[later, "adp"] = perturbed.loc[later, "adp"].sample(frac=1.0, random_state=1).to_numpy()
    same = BaselineAdpProjector(adp=perturbed).project(hist)
    pd.testing.assert_frame_equal(ref, same)                               # seasons after the target are invisible
    now = adp["season"] == TARGET
    changed = adp.copy()
    changed.loc[now, "adp"] = changed.loc[now, "adp"].iloc[::-1].to_numpy()
    assert not np.allclose(BaselineAdpProjector(adp=changed).project(hist)["proj_total_fp"], ref["proj_total_fp"])


def test_adp_projector_ignores_the_future_of_the_game_data(tables):
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, season_start(TARGET) + 1)]
    adp = _oracle_adp(tables, seasons[:-1])      # ADP through the season before the target: no realised-outcome oracle for TARGET
    assert_projector_ignores_future(BaselineAdpProjector(adp=adp), tables, TARGET)


def test_hurdle_adp_adds_the_interval_columns(hist):
    out = get_projector("baseline_hurdle_adp", adp=EMPTY).project(hist)
    assert {"proj_p_appear", "proj_total_fp_p10", "proj_total_fp_p90"} <= set(out.columns)
