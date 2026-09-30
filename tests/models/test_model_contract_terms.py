"""BaselineContractTermsProjector: contract validity, graceful fallback, planted-effect recovery, the honesty
gate, leakage (structural and bend-only-the-future), identities, registry wiring (ADR 0019). Synthetic data only
tests the code; the real result is in ADR 0019."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import PROJECTION_STATS, History, Projector, season_start, season_str, validate_table
from src.features.contract_terms import slice_contracts, terms_features
from src.ingest.wiki_contracts import PLAYER_CONTRACTS_COLUMNS
from src.models.baseline import BaselineProjector
from src.models.contract_terms_baseline import BaselineContractTermsProjector
from src.models.registry import available_projectors, get_projector
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]
MIN_ROWS = 60
WALK_FORWARD: dict = {}      # base projections of past seasons are identical for every variant on the plain league
PLANTED: dict = {}          # ... and for every variant on the planted league (the key includes a fingerprint of the game logs)


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def base(tables):
    return BaselineProjector().project(history_for(tables))


def make_contracts(tables, seed=1, p_sign=0.6) -> pd.DataFrame:
    """A plausible contract history: each player without a running deal signs one in July of a season he plays,
    for 1 to 3 years (dated 1 Jul, so it covers that season and is visible to it)."""
    rng = np.random.default_rng(seed)
    gl = tables["game_logs"]
    starts = gl["season"].map(season_start)
    rows = []
    end: dict[int, int] = {}
    for s in sorted(starts.unique()):
        for pid in sorted(gl.loc[starts == s, "player_id"].unique()):
            if end.get(pid, -1) >= s or rng.random() > p_sign:
                continue
            yrs = int(rng.integers(1, 4))
            end[pid] = s + yrs - 1
            d = pd.Timestamp(s, 7, 1)
            rows.append(dict(season=season_str(s - 1), page_season=season_str(s), start_year=s, team_id=1, player_id=int(pid),
                             event="sign", contract_type="standard", years=yrs, days=pd.NA, amount_usd=np.nan,
                             signed_date=d, date_source="cell", is_two_way=False, is_ten_day=False, is_exhibit10=False,
                             is_minimum=False, is_extension=False, is_rookie=False, non_guaranteed=False, has_option=False,
                             multiyear_unspecified=False, section="add", terms_source="cell"))
    df = pd.DataFrame(rows)[PLAYER_CONTRACTS_COLUMNS]
    df["years"] = pd.array(df["years"], dtype="Int64")
    df["team_id"] = pd.array(df["team_id"], dtype="Int64")
    df["days"] = pd.array(df["days"], dtype="Int64")
    return df


def with_contracts(tables, contracts):
    t = dict(tables)
    t["player_contracts"] = contracts
    return t


def plant_contract_year_effect(tables, contracts, bump: int):
    """Extra free-throw points in every season a player is in the final year of a known deal."""
    t = {k: v.copy() for k, v in tables.items()}
    gl = t["game_logs"]
    starts = gl["season"].map(season_start)
    mask = np.zeros(len(gl), dtype=bool)
    for s in sorted(starts.unique()):
        idx = np.flatnonzero((starts == s).to_numpy())
        pids = gl["player_id"].to_numpy()[idx]
        mask[idx] = terms_features(contracts, season_str(s), pids).contract_year
    for c in ("ftm", "fta", "pts"):
        gl.loc[mask, c] = gl.loc[mask, c] + bump
    return t


def proj(contracts=None, **kw):
    return BaselineContractTermsProjector(contracts=contracts, min_fit_rows=MIN_ROWS, walk_forward_cache=WALK_FORWARD, **kw)


@pytest.fixture(scope="module")
def contracts(tables):
    return make_contracts(tables)


# --------------------------------------------------------------------------- registry and contract

def test_is_a_projector_and_registered():
    assert isinstance(BaselineContractTermsProjector(), Projector)
    assert "baseline_contract_terms" in available_projectors()
    p = get_projector("baseline_contract_terms")
    assert isinstance(p, BaselineContractTermsProjector) and p.name == "baseline_contract_terms"
    assert "baseline_contract" in available_projectors()        # the ADR 0013 model is untouched


def test_output_validates_and_labels_the_model(tables, contracts):
    out = proj(contracts).project(history_for(with_contracts(tables, contracts)))
    validate_table(out, "projections")
    assert (out["model"] == "baseline_contract_terms").all() and out["player_id"].is_unique
    assert out[list(PROJECTION_STATS) + ["proj_fppg", "proj_total_fp", "proj_gp", "proj_mpg"]].notna().all().all()
    for c in ("terms_adj", "terms_factor", "terms_enabled", "terms_status", "terms_years_remaining", "terms_flag",
              "terms_basis", "terms_contract_year"):
        assert c in out.columns
    assert set(out["terms_status"]) <= {"known", "no_length", "lapsed", "unknown"}


def test_deterministic(tables, contracts):
    h = history_for(with_contracts(tables, contracts))
    pd.testing.assert_frame_equal(proj().project(h), proj().project(h), check_exact=True)


# --------------------------------------------------------------------------- fallback

def test_no_contract_data_gives_exactly_the_baseline(tables, base, monkeypatch):
    import src.ingest.wiki_contracts as wc

    def missing(*a, **k):
        raise FileNotFoundError("no player_contracts.parquet")

    monkeypatch.setattr(wc, "read_player_contracts", missing)
    p = BaselineContractTermsProjector(min_fit_rows=MIN_ROWS, walk_forward_cache=WALK_FORWARD)
    out = p.project(history_for(tables))
    assert not out["terms_enabled"].any() and (out["terms_status"] == "unknown").all()
    assert (out["terms_adj"] == 0).all() and not out["terms_contract_year"].any()
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)
    assert p.last_fit is not None and not p.last_fit.enabled


def test_empty_contract_table_is_the_same_as_none(tables, base):
    empty = make_contracts(tables).iloc[0:0]
    out = proj(empty).project(history_for(tables))
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)


def test_too_little_history_gives_exactly_the_baseline():
    t = make_league(first_start=2018, last_start=2019)
    c = make_contracts(t)
    h = history_for(with_contracts(t, c), "2019-20")
    p = BaselineContractTermsProjector(contracts=c)
    out = p.project(h)
    base = BaselineProjector().project(h)
    assert not out["terms_enabled"].any() and not p.last_fit.enabled
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)


# --------------------------------------------------------------------------- planted effect and the gate

def test_recovers_a_planted_contract_year_effect(tables, contracts):
    planted = plant_contract_year_effect(tables, contracts, bump=8)
    p = BaselineContractTermsProjector(contracts=contracts, min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED)
    out = p.project(history_for(with_contracts(planted, contracts)))
    assert p.last_fit.enabled, p.last_fit.diagnostics
    assert p.last_fit.diagnostics["coefficients_fppg"]["cy"] > 1.0
    cy = out["terms_contract_year"].to_numpy()
    assert cy.any() and out.loc[cy, "terms_adj"].mean() > 0.5
    unknown = ((out["terms_status"] == "unknown") & (out["terms_basis"] == "")).to_numpy()   # no wiki event, no rookie clock
    assert unknown.any() and (out.loc[unknown, "terms_adj"] == 0).all()     # such players are never adjusted


def test_adjusted_stats_keep_the_fantasy_point_identity(tables, contracts):
    planted = plant_contract_year_effect(tables, contracts, bump=8)
    out = BaselineContractTermsProjector(contracts=contracts, min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED).project(
        history_for(with_contracts(planted, contracts)))
    assert out["terms_enabled"].all()
    np.testing.assert_allclose(out["proj_fppg"], fantasy_points_frame(out, SCORING, prefix="proj_"), rtol=1e-9)
    np.testing.assert_allclose(out["proj_total_fp"], out["proj_fppg"] * out["proj_gp"], rtol=1e-9)
    assert out["terms_factor"].between(0.8, 1.25).all() and (out["fppg_p10"] <= out["fppg_p90"]).all()


def test_switches_itself_off_when_the_terms_explain_nothing(tables, base, contracts):
    p = proj(contracts)
    out = p.project(history_for(with_contracts(tables, contracts)))
    assert not p.last_fit.enabled, p.last_fit.diagnostics
    core = [c for c in base.columns if c != "model"]
    pd.testing.assert_frame_equal(out[core].reset_index(drop=True), base[core].reset_index(drop=True), check_exact=True)


def test_display_columns_carry_unknown_as_unknown(tables, contracts):
    out = proj(contracts).project(history_for(with_contracts(tables, contracts)))
    unknown = out["terms_status"] == "unknown"
    assert unknown.any()
    assert out.loc[unknown, "terms_years_remaining"].isna().all()           # never 0 or "not a contract year"
    assert (out.loc[unknown & (out["terms_basis"] == ""), "terms_flag"] == "").all()
    known = out["terms_status"] == "known"
    assert (out.loc[known, "terms_years_remaining"] >= 1).all()
    assert out.loc[out["terms_contract_year"], "terms_years_remaining"].eq(1).all()


# --------------------------------------------------------------------------- leakage

def test_leak_check_passes_for_a_disabled_and_an_enabled_layer(tables, contracts):
    """The machinery behind ``--leak-check``: scrambling everything at or after the target (contract rows included,
    since they travel as History.extras) changes nothing."""
    assert_projector_ignores_future(BaselineContractTermsProjector(min_fit_rows=MIN_ROWS), with_contracts(tables, contracts), TARGET, seed=3)
    planted = plant_contract_year_effect(tables, contracts, bump=8)
    assert_projector_ignores_future(BaselineContractTermsProjector(min_fit_rows=MIN_ROWS), with_contracts(planted, contracts), TARGET, seed=3)


def _future_bent(contracts, target):
    """Same table, but every row tagged at or after the target is rewritten: new players, waived, other lengths."""
    c = contracts.copy()
    m = (c["season"].map(season_start) >= season_start(target)).to_numpy()
    assert m.any()
    c.loc[m, "years"] = pd.array([1] * int(m.sum()), dtype="Int64")
    c.loc[m, "event"] = np.where(np.arange(int(m.sum())) % 2 == 0, "waived", "extension")
    c.loc[m, "player_id"] = c.loc[m, "player_id"].to_numpy()[::-1]
    extra = c[m].iloc[:20].copy()
    extra["player_id"] = extra["player_id"] + 100_000
    return pd.concat([c, extra], ignore_index=True)


def test_bending_only_the_future_changes_nothing(tables, contracts):
    """Through History (structural slice) and through a caller that hands the projector the whole, unsliced table."""
    planted = plant_contract_year_effect(tables, contracts, bump=8)
    bent = _future_bent(contracts, TARGET)
    a = BaselineContractTermsProjector(min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED).project(history_for(with_contracts(planted, contracts)))
    b = BaselineContractTermsProjector(min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED).project(history_for(with_contracts(planted, bent)))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a["terms_enabled"].all()                          # the comparison is against an ACTIVE layer, not a no-op
    h = history_for(planted)                                  # no extras: contracts handed straight to the projector
    c = BaselineContractTermsProjector(contracts=contracts, min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED).project(h)
    d = BaselineContractTermsProjector(contracts=bent, min_fit_rows=MIN_ROWS, walk_forward_cache=PLANTED).project(h)
    pd.testing.assert_frame_equal(c, d, check_exact=True)


def test_history_slices_the_contract_table_by_its_season_tag(tables, contracts):
    h = History.until(with_contracts(tables, contracts), TARGET)
    got = h.extras["player_contracts"]
    assert (got["season"].map(season_start) < season_start(TARGET)).all()
    assert len(got) == len(slice_contracts(contracts, TARGET)) < len(contracts)
    # an event dated in July of the target season is visible to it (tag = previous season)
    july = contracts[contracts["signed_date"] == pd.Timestamp(season_start(TARGET), 7, 1)]
    assert len(july) and (july["season"] == season_str(season_start(TARGET) - 1)).all()
    assert july.index.isin(got.index).all()
    assert h.extras["player_contracts"]["signed_date"].max() < pd.Timestamp(season_start(TARGET), 10, 1)


def test_future_signings_do_not_flag_anyone_in_the_target_season(tables, contracts):
    """A player whose only deal is signed after opening night is unknown for the target season."""
    only_future = contracts.copy()
    only_future["season"] = season_str(season_start(TARGET))            # everything tagged as following the target
    out = proj(only_future).project(history_for(with_contracts(tables, only_future)))
    assert (out["terms_status"] == "unknown").all() and not out["terms_contract_year"].any()
