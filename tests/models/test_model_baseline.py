"""BaselineProjector: contract, determinism, leakage, invariants, edge cases."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, add_player_seasons, history_for, make_league

from src.contracts import PROJECTION_STATS, History, Projector, season_start, validate_table
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.value.league import load_league
from src.value.points import fantasy_points

T_START = season_start(TARGET)


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


@pytest.fixture(scope="module")
def proj(hist):
    return BaselineProjector().project(hist)


# --------------------------------------------------------------------------- contract and shape

def test_is_a_projector():
    assert isinstance(BaselineProjector(), Projector)
    assert BaselineProjector().name == "baseline"


def test_output_validates_against_projections_contract(proj):
    validate_table(proj, "projections")
    assert (proj["season"] == TARGET).all()
    assert (proj["model"] == "baseline").all()
    assert proj["player_id"].is_unique
    assert proj[list(PROJECTION_STATS) + ["proj_fppg", "proj_total_fp", "proj_gp", "proj_mpg"]].notna().all().all()


def test_output_validates_at_every_history_length(tables):
    for target in ("2013-14", "2014-15", "2015-16", "2019-20"):
        out = BaselineProjector().project(history_for(tables, target))
        validate_table(out, "projections")
        assert len(out) > 50


def test_deterministic(hist):
    a = BaselineProjector().project(hist)
    b = BaselineProjector().project(hist)
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
    pd.testing.assert_frame_equal(BaselineProjector().project(hist), BaselineProjector().project(shuffled), check_exact=True)


# --------------------------------------------------------------------------- leakage

def _perturb_future(tables, rng):
    """Scramble everything dated at or after the target season, plus future-only player attributes."""
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
    perturbed = _perturb_future(tables, np.random.default_rng(1))
    h2 = History.until(perturbed, TARGET)
    h2.assert_no_future()
    out = BaselineProjector().project(h2)
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_future_only_players_do_not_change_projections(tables, proj):
    """A player row for someone who debuts after the target season must not matter."""
    extra = add_player_seasons(tables, 9_999_001, "Future Guy", {"2019-20": (40, dict(minutes=30.0))}, draft_year=2019)
    out = BaselineProjector().project(history_for(extra))
    pd.testing.assert_frame_equal(proj, out, check_exact=True)


def test_history_with_future_rows_is_rejected(tables):
    leaky = History(TARGET, tables["game_logs"], tables["team_games"], tables["players"], tables["player_season_bio"])
    with pytest.raises(AssertionError, match="leakage"):
        BaselineProjector().project(leaky)


# --------------------------------------------------------------------------- invariants

def test_games_played_never_exceed_the_schedule(hist, proj):
    games = hist.team_games.groupby(["season", "team_id"]).size().max()
    assert (proj["proj_gp"] > 0).all()
    assert (proj["proj_gp"] <= games).all()
    assert (proj["proj_gp_p90"] <= games + 1e-9).all()
    assert (proj["proj_gp_p10"] <= proj["proj_gp_p90"]).all()
    assert (proj["proj_gp_sd"] > 0).all()


def test_schedule_length_can_be_overridden(hist):
    out = BaselineProjector(config=BaselineConfig(season_games=40)).project(hist)
    assert (out["proj_gp"] <= 40).all()


def test_total_is_rate_times_games(proj):
    np.testing.assert_allclose(proj["proj_total_fp"], proj["proj_fppg"] * proj["proj_gp"], rtol=1e-12)


def test_floor_median_ceiling_are_ordered(proj):
    assert (proj["fppg_p10"] <= proj["fppg_p50"]).all()
    assert (proj["fppg_p50"] <= proj["fppg_p90"]).all()
    assert (proj["fppg_p10"] < proj["proj_fppg"]).all()
    assert (proj["fppg_p90"] > proj["proj_fppg"]).all()
    assert (proj["proj_fppg_sd"] > 0).all()


def test_box_score_identities_hold_in_projections(proj):
    np.testing.assert_allclose(proj["proj_pts"], 2 * proj["proj_fgm"] + proj["proj_fg3m"] + proj["proj_ftm"], rtol=1e-12)
    assert (proj["proj_fg3m"] <= proj["proj_fgm"] + 1e-12).all()
    assert (proj["proj_fgm"] <= proj["proj_fga"] + 1e-12).all()
    assert (proj["proj_ftm"] <= proj["proj_fta"] + 1e-12).all()
    assert (proj[list(PROJECTION_STATS)] >= 0).all().all()
    assert (proj["proj_mpg"] <= 44.0).all()


def test_fppg_uses_league_config_scoring(proj):
    scoring = load_league()["scoring"]
    row = proj.iloc[7]
    stats = {k: row[f"proj_{c}"] for k, c in
             {"PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "FGM": "fgm",
              "FGA": "fga", "FTM": "ftm", "FTA": "fta", "3PM": "fg3m"}.items()}
    assert row["proj_fppg"] == pytest.approx(fantasy_points(stats, scoring))


def test_scoring_change_changes_fp_as_expected(hist, proj):
    doubled_steals = {**load_league()["scoring"], "STL": 8}
    out = BaselineProjector(scoring=doubled_steals).project(hist)
    # Stat projections do not depend on scoring; only fantasy points do, by exactly the weight delta.
    for c in PROJECTION_STATS:
        pd.testing.assert_series_equal(out[c], proj[c], check_exact=True)
    np.testing.assert_allclose(out["proj_fppg"] - proj["proj_fppg"], 4 * proj["proj_stl"], atol=1e-9)
    pts_only = BaselineProjector(scoring={"PTS": 1.0}).project(hist)
    np.testing.assert_allclose(pts_only["proj_fppg"], proj["proj_pts"], rtol=1e-12)


def test_unknown_scoring_key_fails_loudly(hist):
    with pytest.raises(KeyError):
        BaselineProjector(scoring={"PTS": 1, "DD": 5}).project(hist)


# --------------------------------------------------------------------------- who is projected

def test_only_plausibly_active_players_are_projected(tables, hist, proj):
    logs = hist.game_logs.assign(s=hist.game_logs["season"].map(season_start))
    last_seen = logs.groupby("player_id")["s"].max()
    retired = last_seen[last_seen < T_START - 2].index
    assert len(retired) > 10, "fixture should contain retired players"
    assert not set(retired) & set(proj["player_id"])
    veterans = proj[~proj["is_rookie"]]
    assert set(veterans["player_id"]) == set(last_seen[last_seen >= T_START - 2].index)


def test_rookies_are_projected_with_low_confidence_and_flagged(tables, hist, proj):
    players = hist.players
    rookies_expected = set(players.loc[players["draft_year"] == T_START, "player_id"]) - set(hist.game_logs["player_id"])
    assert len(rookies_expected) > 3
    r = proj[proj["is_rookie"]]
    assert set(r["player_id"]) == rookies_expected
    assert (r["confidence"] == "low").all()
    assert (r["n_hist_seasons"] == 0).all()
    assert (r["proj_mpg"] > 0).all() and (r["proj_fppg"] > 0).all()
    # A rookie has no track record, so he projects below the typical established player.
    assert r["proj_fppg"].median() < proj.loc[~proj["is_rookie"], "proj_fppg"].median()


def test_undrafted_players_without_history_are_excluded(tables):
    extra = add_player_seasons(tables, 9_999_002, "Undrafted", {"2018-19": (40, dict(minutes=20.0))}, draft_year=None)
    out = BaselineProjector().project(history_for(extra))
    assert 9_999_002 not in set(out["player_id"])


def test_draft_slot_prior_is_learned_from_history():
    """Give history's rookies a draft order that tracks their first-season production; picks then matter."""
    t = make_league(first_start=2010, n_teams=24, games_per_team=30, seed=11)
    pl = t["players"]
    gl = t["game_logs"]
    first = gl.groupby("player_id")["pts"].agg("mean")
    rookies = pl[(pl["draft_year"].notna()) & (pl["draft_year"] <= 2017)].copy()
    rookies = rookies[rookies["draft_year"] == rookies["from_year"]]
    order = first.reindex(rookies["player_id"]).rank(ascending=False, method="first").to_numpy()
    pl.loc[rookies.index, "draft_number"] = np.clip(order, 1, 60).astype("int64")
    tgt = pl["draft_year"] == T_START
    ids = pl.loc[tgt, "player_id"].to_numpy()
    assert len(ids) >= 6
    pl.loc[tgt, "draft_number"] = np.clip(np.arange(len(ids)) * 9 + 1, 1, 60)
    out = BaselineProjector().project(history_for(t))
    r = out[out["player_id"].isin(ids)].set_index("player_id")
    picks = pl.set_index("player_id").loc[r.index, "draft_number"]
    assert picks.corr(r["proj_mpg"], method="spearman") < -0.3


def test_replacement_prior_used_when_history_has_few_rookies(tables):
    h = history_for(tables)
    p = BaselineProjector(config=BaselineConfig(min_rookie_rows=10_000))
    fit = p.fit(h)
    assert fit.rookie_prior.mode == "replacement"
    out = fit.predict()
    r = out[out["is_rookie"]]
    assert len(r) > 3 and r["proj_mpg"].nunique() == 1  # everyone gets the same explicit prior
    assert (r["confidence"] == "low").all()


def test_injured_last_season_player_is_kept_and_not_written_off(tables):
    rng_pid = 9_999_010
    healthy = {f"{y}-{(y + 1) % 100:02d}": (55, dict(minutes=32.0, stl=1, blk=1)) for y in (2015, 2016, 2017)}
    base = add_player_seasons(tables, rng_pid, "Ironman", healthy)
    hurt = dict(healthy)
    hurt["2017-18"] = (6, dict(minutes=32.0, stl=1, blk=1))
    inj = add_player_seasons(tables, rng_pid, "Ironman", hurt)
    out_ok = BaselineProjector().project(history_for(base)).set_index("player_id").loc[rng_pid]
    out_inj = BaselineProjector().project(history_for(inj)).set_index("player_id").loc[rng_pid]
    # Projected per-game output is not collapsed by a 6-game season ...
    assert out_inj["proj_mpg"] > 0.85 * out_ok["proj_mpg"]
    assert out_inj["proj_fppg"] > 0.85 * out_ok["proj_fppg"]
    # ... availability is lower than the healthy twin's but well above last season's 6 games.
    assert out_inj["proj_gp"] < out_ok["proj_gp"]
    assert out_inj["proj_gp"] > 6 * 2


def test_player_absent_last_season_but_present_before_is_still_projected(tables):
    pid = 9_999_011
    t = add_player_seasons(tables, pid, "Returner", {"2016-17": (50, dict(minutes=28.0)), "2015-16": (50, dict(minutes=28.0))})
    out = BaselineProjector().project(history_for(t))
    row = out[out["player_id"] == pid]
    assert len(row) == 1
    assert row.iloc[0]["n_hist_seasons"] == 2


def test_single_season_history_uses_documented_fallbacks(tables):
    h = history_for(tables, "2013-14")
    assert h.game_logs["season"].nunique() == 1
    fit = BaselineProjector().fit(h)
    assert not any(f.fitted for f in fit.rates.values())
    assert not fit.availability.fitted
    out = fit.predict()
    validate_table(out, "projections")


def test_empty_history_is_an_error(tables):
    h = history_for(tables, "2012-13")
    assert h.game_logs.empty
    with pytest.raises(ValueError, match="no game logs"):
        BaselineProjector().project(h)


def test_missing_birthdates_fall_back_to_bio_ages_or_neutral(tables, proj):
    t = {k: v.copy() for k, v in tables.items()}
    t["players"]["birthdate"] = pd.NaT
    out = BaselineProjector().project(history_for(t))
    validate_table(out, "projections")
    assert out["age"].notna().sum() > 0.8 * len(out)   # ages recovered from player_season_bio for veterans


def test_missing_team_games_falls_back_to_observed_season_length(hist):
    h = History(hist.target_season, hist.game_logs, hist.team_games.iloc[0:0], hist.players, hist.player_season_bio)
    out = BaselineProjector().project(h)
    validate_table(out, "projections")
    assert out["proj_gp"].max() <= hist.team_games.groupby(["season", "team_id"]).size().max()


def test_older_players_project_lower_minutes_growth_than_young(hist):
    """Sanity: the fitted minutes age curve is higher for 22-year-olds than for 34-year-olds."""
    fit = BaselineProjector().fit(hist)
    c = fit.minutes.curve
    assert c.delta(22.0) > c.delta(34.0)
