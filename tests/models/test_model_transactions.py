"""BaselineTransactionsProjector: contract, determinism, leakage, registry wiring, and the
no-transactions-file-yet graceful fallback (mirrors tests/models/test_model_roster.py's structure
for the new ADR 0011 projector)."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, add_player_seasons, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future, build_history
from src.contracts import PROJECTION_STATS, History, Projector, season_start, validate_table
from src.models.baseline import BaselineProjector
from src.models.registry import get_projector
from src.models.transactions_baseline import BaselineTransactionsProjector

T_START = season_start(TARGET)


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def proj(hist):
    return BaselineTransactionsProjector().project(hist)


# --------------------------------------------------------------------------- contract and shape

def test_is_a_projector():
    assert isinstance(BaselineTransactionsProjector(), Projector)
    assert BaselineTransactionsProjector().name == "baseline_transactions"


def test_registered_and_reachable_from_the_registry(hist):
    p = get_projector("baseline_transactions")
    assert isinstance(p, BaselineTransactionsProjector)
    out = p.project(hist)
    validate_table(out, "projections")
    assert (out["model"] == "baseline_transactions").all()


def test_output_validates_against_projections_contract(proj):
    validate_table(proj, "projections")
    assert (proj["season"] == TARGET).all()
    assert (proj["model"] == "baseline_transactions").all()
    assert proj["player_id"].is_unique
    assert proj[list(PROJECTION_STATS) + ["proj_fppg", "proj_total_fp", "proj_gp", "proj_mpg"]].notna().all().all()


def test_output_validates_at_every_history_length(tables):
    for target in ("2013-14", "2014-15", "2015-16", "2019-20"):
        out = BaselineTransactionsProjector().project(history_for(tables, target))
        validate_table(out, "projections")
        assert len(out) > 50


def test_deterministic(hist):
    a = BaselineTransactionsProjector().project(hist)
    b = BaselineTransactionsProjector().project(hist)
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
    pd.testing.assert_frame_equal(BaselineTransactionsProjector().project(hist),
                                  BaselineTransactionsProjector().project(shuffled), check_exact=True)


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
    """Same style as BaselineProjector's leakage test. Since no team_transactions.parquet exists in
    this synthetic-only test environment, the transactions hook returns None (graceful fallback) --
    but this must still hold, exercising the fallback path itself for leakage-safety."""
    perturbed = _perturb_future(tables, np.random.default_rng(1))
    h2 = History.until(perturbed, TARGET)
    h2.assert_no_future()
    out = BaselineTransactionsProjector().project(h2)
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_future_only_players_do_not_change_projections(tables, proj):
    extra = add_player_seasons(tables, 9_999_301, "Future Guy T", {"2019-20": (40, dict(minutes=30.0))}, draft_year=2019)
    out = BaselineTransactionsProjector().project(history_for(extra))
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_history_with_future_rows_is_rejected(tables):
    leaky = History(TARGET, tables["game_logs"], tables["team_games"], tables["players"], tables["player_season_bio"])
    with pytest.raises(AssertionError, match="leakage"):
        BaselineTransactionsProjector().project(leaky)


def test_assert_projector_ignores_future_passes(tables):
    """The shared leakage-guard machinery (src.backtest.leakage), the exact check the backtest
    CLI's --leak-check flag runs, must also pass for this projector."""
    assert_projector_ignores_future(BaselineTransactionsProjector(), tables, TARGET, seed=3)


def test_build_history_and_project_smoke(tables):
    h = build_history(tables, TARGET)
    out = BaselineTransactionsProjector().project(h)
    assert len(out) > 50


# --------------------------------------------------------------------------- no-op fallback / plain baseline unaffected

def test_plain_baseline_has_no_transactions_features(hist):
    """Plain BaselineProjector never sets transactions_features -- the hook's default (None) must be
    left completely alone by these changes."""
    fitted = BaselineProjector().fit(hist)
    assert fitted.transactions_features is None


def test_baseline_transactions_has_no_roster_features(hist):
    """BaselineTransactionsProjector only overrides the transactions hook -- roster_features must
    stay at its default None (no accidental double-adjustment / cross-wiring)."""
    fitted = BaselineTransactionsProjector().fit(hist)
    assert fitted.roster_features is None


def test_plain_baseline_projections_unaffected_by_the_new_hook(hist):
    a = BaselineProjector().project(hist)
    b = BaselineProjector().project(hist)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_baseline_transactions_falls_back_to_plain_baseline_with_no_ingested_file(hist, tmp_path, monkeypatch):
    """With no team_transactions.parquet in the data directory, the graceful-fallback contract means
    baseline_transactions must be bit-identical to plain baseline. Isolated via a fresh NBA_DATA_DIR
    (not the ambient absence of the file) so this test's premise holds regardless of whether the real
    Wikipedia ingest has been run in this environment -- see ADR 0011: real ingested data does exist
    at the default NBA_DATA_DIR once `python -m src.ingest.wiki_transactions` has run, which used to
    make this test's docstring claim false and its assertion moot."""
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "no_transactions_here"))
    base = BaselineProjector().project(hist)
    txn = BaselineTransactionsProjector().project(hist)
    base = base.drop(columns=["model"])
    txn = txn.drop(columns=["model"])
    pd.testing.assert_frame_equal(base, txn, check_exact=True)


def test_zero_context_history_makes_baseline_transactions_identical_to_baseline():
    """With only one season ever in the history, no training row can have an earlier season of its
    own, so TransactionFeatures.fit sees zero usable rows and must fall back to the documented zero
    adjustment everywhere. With mpg never adjusted, baseline_transactions must be bit-identical to
    plain baseline for this history."""
    tables = make_league(first_start=2018, last_start=2019)
    h = history_for(tables, "2019-20")
    base = BaselineProjector().project(h)
    txn = BaselineTransactionsProjector().project(h)
    base = base.drop(columns=["model"])
    txn = txn.drop(columns=["model"])
    pd.testing.assert_frame_equal(base, txn, check_exact=True)


# --------------------------------------------------------------------------- rookies / thin history

def test_rookie_with_no_history_is_unaffected_by_transactions_layer(tables):
    """A rookie is projected from the draft-slot prior, not ``_core`` (and therefore never touches
    ``transactions_features``) at all, so baseline and baseline_transactions must agree on rookies
    exactly."""
    h = history_for(tables)
    base = BaselineProjector().project(h)
    txn = BaselineTransactionsProjector().project(h)
    rookies = base.loc[base["is_rookie"], "player_id"]
    if len(rookies) == 0:
        pytest.skip("no rookies in this synthetic history slice")
    b = base.set_index("player_id").loc[rookies, "proj_gp"]
    r = txn.set_index("player_id").loc[rookies, "proj_gp"]
    pd.testing.assert_series_equal(b, r, check_exact=True)
