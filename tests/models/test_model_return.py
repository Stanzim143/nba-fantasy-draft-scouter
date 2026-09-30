"""BaselineReturnProjector family (ADR 0022): contract, registry, width consistency, leakage, and the guarantee that plain
``baseline`` is untouched. Synthetic league only; the real ablation is in docs/adr/0022-return-health-feature.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, add_player_seasons, history_for, make_league

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
from injury_testkit import add_player_season_pattern  # noqa: E402

from src.backtest.leakage import assert_projector_ignores_future  # noqa: E402
from src.contracts import History, Projector, season_start, validate_table  # noqa: E402
from src.features import return_health as RH  # noqa: E402
from src.models.availability import AvailabilityModel  # noqa: E402
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig  # noqa: E402
from src.models.registry import available_projectors, get_projector  # noqa: E402
from src.models.return_baseline import BaselineInjuryReturnProjector, BaselineReturnProjector  # noqa: E402

T_START = season_start(TARGET)
VARIANTS = ("baseline_return", "baseline_injury_return")


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def proj(hist):
    return BaselineReturnProjector().project(hist)


# --------------------------------------------------------------------------- baseline is untouched

def test_plain_baseline_matches_the_pre_change_golden_values(hist):
    """Frozen from the code before ADR 0022 (which changes no baseline file): plain ``baseline`` on the synthetic league.

    Since ADR 0030 the default tunes shrinkage on predicted MPG; ``tune_with_predicted_mpg=False`` must still reproduce the
    original numbers exactly, and the default's own values are frozen in the next test."""
    out = BaselineProjector(config=BaselineConfig(tune_with_predicted_mpg=False)).project(hist).sort_values("player_id")
    assert len(out) == 177
    assert out["proj_gp"].sum() == pytest.approx(8791.25771623194, rel=1e-9)
    assert out["proj_total_fp"].sum() == pytest.approx(176889.8969018681, rel=1e-9)
    assert (out["proj_gp"].to_numpy() * np.arange(len(out))).sum() == pytest.approx(774490.5046580521, rel=1e-9)


def test_plain_baseline_golden_values_with_predicted_mpg_tuning(hist):
    """Default since ADR 0030: games played are untouched (availability does not use the rate tuning), total FP moves ~0.01%."""
    out = BaselineProjector().project(hist).sort_values("player_id")
    assert len(out) == 177
    assert out["proj_gp"].sum() == pytest.approx(8791.25771623194, rel=1e-9)
    assert out["proj_total_fp"].sum() == pytest.approx(176866.94064208702, rel=1e-9)


def test_plain_baseline_has_no_extra_availability_columns_and_is_deterministic(hist):
    fitted = BaselineProjector().fit(hist)
    assert fitted.injury_features is None and fitted.availability.n_extra == 0
    a, b = BaselineProjector().project(hist), get_projector("baseline").project(hist)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_baseline_is_bit_identical_to_a_subclass_whose_hook_returns_none(hist):
    class Hooked(BaselineProjector):
        def _build_injury_features(self, history, sub):
            return None
    pd.testing.assert_frame_equal(BaselineProjector().project(hist), Hooked().project(hist), check_exact=True)


def test_baseline_is_unchanged_by_importing_and_running_the_return_variants(hist):
    before = BaselineProjector().project(hist)
    BaselineReturnProjector().project(hist)
    BaselineInjuryReturnProjector().project(hist)
    pd.testing.assert_frame_equal(before, BaselineProjector().project(hist), check_exact=True)


# --------------------------------------------------------------------------- contract, registry

def test_is_a_projector_with_the_right_names():
    assert isinstance(BaselineReturnProjector(), Projector)
    assert BaselineReturnProjector().name == "baseline_return" and BaselineInjuryReturnProjector().name == "baseline_injury_return"


@pytest.mark.parametrize("name", VARIANTS)
def test_registered_and_projects_a_valid_table(name, hist):
    assert name in available_projectors()
    p = get_projector(name)
    out = p.project(hist)
    validate_table(out, "projections")
    assert (out["model"] == name).all() and out["player_id"].is_unique
    assert (out["proj_gp"] > 0).all() and (out["proj_gp"] <= hist.team_games.groupby(["season", "team_id"]).size().max()).all()
    assert np.allclose(out["proj_total_fp"], out["proj_fppg"] * out["proj_gp"], rtol=1e-6, atol=1e-3)


def test_offseason_debut_variant_is_registered_and_builds_on_the_return_base():
    assert "baseline_return_offseason_debut" in available_projectors()
    p = get_projector("baseline_return_offseason_debut")
    assert p.name == "baseline_return_offseason_debut"
    assert isinstance(p.base, BaselineProjector)
    from src.models.return_baseline import ReturnDebutantProjector
    assert isinstance(p.base, ReturnDebutantProjector)
    # the board's default is a different object built on plain baseline, and is unchanged
    assert not isinstance(get_projector("baseline_offseason_debut").base, ReturnDebutantProjector)


def test_no_existing_registry_entry_was_replaced():
    for name in ("baseline", "baseline_injury", "baseline_offseason_debut", "baseline_offseason"):
        assert name in available_projectors()
    assert type(get_projector("baseline")) is BaselineProjector


def test_output_validates_at_every_history_length(tables):
    for target in ("2013-14", "2014-15", "2015-16", "2019-20"):
        for name in VARIANTS:
            out = get_projector(name).project(history_for(tables, target))
            validate_table(out, "projections")
            assert len(out) > 50


def test_deterministic_and_row_order_invariant(hist):
    a = BaselineReturnProjector().project(hist)
    pd.testing.assert_frame_equal(a, BaselineReturnProjector().project(hist), check_exact=True)
    shuffled = History(target_season=hist.target_season,
                       game_logs=hist.game_logs.sample(frac=1.0, random_state=5).reset_index(drop=True),
                       team_games=hist.team_games.sample(frac=1.0, random_state=1).reset_index(drop=True),
                       players=hist.players.sample(frac=1.0, random_state=2).reset_index(drop=True),
                       player_season_bio=hist.player_season_bio.sample(frac=1.0, random_state=3).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, BaselineReturnProjector().project(shuffled), check_exact=True)


def test_rookies_are_identical_to_baseline(hist):
    base, ret = BaselineProjector().project(hist), BaselineReturnProjector().project(hist)
    rookies = base.loc[base["is_rookie"], "player_id"]
    if len(rookies) == 0:
        pytest.skip("no rookies in this synthetic history slice")
    pd.testing.assert_series_equal(base.set_index("player_id").loc[rookies, "proj_gp"],
                                   ret.set_index("player_id").loc[rookies, "proj_gp"], check_exact=True)


# --------------------------------------------------------------------------- width consistency

@pytest.mark.parametrize("name, n_extra", [("baseline_return", RH.N_COLS), ("baseline_injury_return", 4 + RH.N_COLS)])
def test_fit_and_predict_widths_agree(name, n_extra, hist):
    fitted = get_projector(name).fit(hist)
    assert fitted.availability.n_extra == n_extra and fitted.injury_features.n_cols == n_extra
    assert fitted.availability.scaler.n_features_in_ == 7 + n_extra
    frame = fitted.panel.iloc[:5]
    x = fitted.injury_features.build(frame["player_id"].to_numpy("int64"), frame["s"].to_numpy("int64") + 1, np.full(5, 26.0))
    assert x.shape == (5, n_extra)
    core = fitted._veterans()
    assert np.isfinite(core["mu_f"]).all() and core["mu_f"].between(0.02, 0.985).all()


def test_availability_model_rejects_a_mismatched_extra_width():
    rng = np.random.default_rng(0)
    n = 200
    f_lags = np.column_stack([rng.uniform(0.3, 1, n), rng.uniform(0.3, 1, n)])
    m = AvailabilityModel.fit(f_lags, rng.uniform(0.3, 1, n), np.full(n, 26.0), np.full(n, 25.0), decay=0.6, C=1.0, min_rows=60,
                              extra=rng.random((n, RH.N_COLS)))
    assert m.n_extra == RH.N_COLS
    with pytest.raises(ValueError, match="columns"):
        m.predict_mean(f_lags, np.full(n, 26.0), np.full(n, 25.0), extra=rng.random((n, 3)))
    assert m.predict_mean(f_lags, np.full(n, 26.0), np.full(n, 25.0), extra=rng.random((n, RH.N_COLS))).shape == (n,)


def test_return_features_keep_every_training_row(hist):
    """The features are never NaN, so unlike the injury layer no training row is dropped for them."""
    fitted = get_projector("baseline_return").fit(hist)
    plain = BaselineProjector().fit(hist)
    assert fitted.availability.n_train == plain.availability.n_train


# --------------------------------------------------------------------------- behaviour

def test_a_returner_and_a_spread_absence_twin_both_get_a_finite_projection(tables):
    """Smoke test on a lead-block-then-healthy player and a scattered-absence player with similar games played last season.
    It asserts no direction or size (that is what the real ablation is for)."""
    team = tables["team_games"][tables["team_games"].season == "2017-18"].groupby("team_id").size().index[0]
    n = int(((tables["team_games"].season == "2017-18") & (tables["team_games"].team_id == team)).sum())
    lead = [False] * (n - 25) + [True] * 25
    spread = [i % 3 != 0 for i in range(n)]
    spread = (spread + [False] * n)[:n]
    t = add_player_season_pattern(tables, 9_900_201, "Returner", "2017-18", played=lead, team_of_season=team)
    t = add_player_season_pattern(t, 9_900_202, "Scatterer", "2017-18", played=spread, team_of_season=team)
    h = History.until(t, TARGET)
    out = BaselineReturnProjector().project(h).set_index("player_id")
    assert np.isfinite(out.loc[[9_900_201, 9_900_202], "proj_gp"]).all()
    assert (out["proj_gp"] > 0).all()


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
    # scramble only what reveals the future (a past from_year is legitimate history, and this projector reads it to decide who is a veteran)
    frm, to = pd.to_numeric(pl["from_year"], errors="coerce"), pd.to_numeric(pl["to_year"], errors="coerce")
    fm, tm = (frm >= T_START).fillna(False).to_numpy(bool), (to >= T_START).fillna(False).to_numpy(bool)
    pl.loc[fm, "from_year"] = T_START + rng.integers(0, 7, int(fm.sum()))
    pl.loc[tm, "to_year"] = T_START + rng.integers(0, 7, int(tm.sum()))
    return out


@pytest.mark.parametrize("cls", [BaselineReturnProjector, BaselineInjuryReturnProjector])
def test_no_leakage_perturbing_target_and_later_data_changes_nothing(cls, tables, hist):
    base = cls().project(hist)
    h2 = History.until(_perturb_future(tables, np.random.default_rng(1)), TARGET)
    h2.assert_no_future()
    pd.testing.assert_frame_equal(base, cls().project(h2), check_exact=True)


def test_future_only_players_do_not_change_projections(tables, proj):
    extra = add_player_seasons(tables, 9_999_101, "Future Guy", {"2019-20": (40, dict(minutes=30.0))}, draft_year=2019)
    pd.testing.assert_frame_equal(proj, BaselineReturnProjector().project(history_for(extra)), check_exact=True)


def test_history_with_future_rows_is_rejected(tables):
    leaky = History(TARGET, tables["game_logs"], tables["team_games"], tables["players"], tables["player_season_bio"])
    with pytest.raises(AssertionError, match="leakage"):
        BaselineReturnProjector().project(leaky)


@pytest.mark.parametrize("name", VARIANTS)
def test_assert_projector_ignores_future_passes(name, tables):
    """The exact check the backtest CLI's --leak-check runs."""
    assert_projector_ignores_future(get_projector(name), tables, TARGET, seed=3)
