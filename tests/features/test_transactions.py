"""Unit tests for src.features.transactions: build_transactions_panel, TransactionContext,
TransactionFeatures.

Uses tests/models/model_testkit.py's synthetic league for a valid team schedule/skeleton and
tests/features/roster_testkit.py to place players in EXACT known minutes/team patterns, plus
hand-constructed ``team_transactions``-shaped DataFrames (the frozen schema from ADR 0011: season,
team_id, player_id, direction, source_kind, txn_date) -- no dependency on the real ingest module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import make_league  # noqa: E402

from roster_testkit import add_team_season_roster  # noqa: E402

from src.contracts import History, season_start
from src.features.transactions import (
    MIN_FIT_ROWS, MPG_SHIFT_CAP, TransactionContext, TransactionFeatures, build_transactions_panel,
)

TARGET = "2018-19"
PATTERN_SEASON = "2016-17"       # will be player's "previous" season
TXN_SEASON = "2017-18"           # the season whose transactions we test (must precede TARGET, so
                                  # its game_logs rows are actually part of History.until(..., TARGET)
                                  # -- build_transactions_panel only emits rows for seasons a player
                                  # actually appears in game_logs for, same as build_roster_panel)


def _txn_row(season, team_id, player_id, direction, source_kind, txn_date):
    return dict(season=season, team_id=team_id, player_id=player_id, direction=direction,
               source_kind=source_kind, txn_date=pd.Timestamp(txn_date) if txn_date is not None else pd.NaT)


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "team_id", "player_id", "direction", "source_kind", "txn_date"])


@pytest.fixture(scope="module")
def tables():
    return make_league()


# --------------------------------------------------------------------------- team assignment / changed_team

def test_changed_team_flagged_for_dated_preseason_addition(tables):
    """Player has a tracked previous-season team A, actually plays TXN_SEASON for team B (an actual
    historical training row), and has a dated "in" transaction to team B (before Oct 1) ->
    changed_team == 1.0."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_001, name="Mover", position="F", minutes=28.0, n_games=10)],
    )
    other_teams = [tid for tid in t["team_games"]["team_id"].unique() if tid != team_a]
    team_b = other_teams[0]
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_001, name="Mover", position="F", minutes=26.0, n_games=10)],
        team_of_season=team_b,
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_b, 9_930_001, "in", "trade", f"{s}-08-15"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)
    row = panel[(panel.player_id == 9_930_001) & (panel.s == s)]
    assert len(row) == 1
    assert row.iloc[0]["changed_team"] == 1.0


def test_changed_team_zero_when_transaction_matches_prior_team(tables):
    """A dated "in" transaction to the SAME team the player was already on (e.g. a re-signing) must
    not be flagged as a team change."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_002, name="Resigned", position="F", minutes=20.0, n_games=10)],
    )
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_002, name="Resigned", position="F", minutes=20.0, n_games=10)],
        team_of_season=team_a,
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_a, 9_930_002, "in", "addition", f"{s}-07-01"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)
    row = panel[(panel.player_id == 9_930_002) & (panel.s == s)]
    assert len(row) == 1
    assert row.iloc[0]["changed_team"] == 0.0


def test_changed_team_zero_with_no_transaction_continuity(tables):
    """No transaction at all -> falls back to previous-season team, changed_team == 0.0."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_003, name="Stayer", position="F", minutes=22.0, n_games=10)],
    )
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_003, name="Stayer", position="F", minutes=22.0, n_games=10)],
        team_of_season=team_a,
    )
    s = season_start(TXN_SEASON)
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, _empty_transactions())
    row = panel[(panel.player_id == 9_930_003) & (panel.s == s)]
    assert len(row) == 1
    assert row.iloc[0]["changed_team"] == 0.0
    assert row.iloc[0]["team_departures_lost"] == 0.0


# --------------------------------------------------------------------------- the October-1 cutoff

def test_october_1_cutoff_excludes_a_midseason_trade(tables):
    """A transaction dated well INTO the season (a genuine midseason trade) must NOT be treated as
    a preseason arrival -- this would be WRONG if the cutoff were missing or checked against the
    wrong date. Player actually plays TXN_SEASON for team A (continuity, matching PATTERN_SEASON's
    team) despite a dated-midseason "in" transaction pointing at team B: he must still resolve to
    team A (continuity), changed_team == 0.0, NOT team B."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_004, name="MidSeasonTradee", position="F", minutes=24.0, n_games=10)],
    )
    other_teams = [tid for tid in t["team_games"]["team_id"].unique() if tid != team_a]
    team_b = other_teams[0]
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_004, name="MidSeasonTradee", position="F", minutes=24.0, n_games=10)],
        team_of_season=team_a,
    )
    s = season_start(TXN_SEASON)
    # January of the season year following s -- clearly midseason, clearly after Oct 1.
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_b, 9_930_004, "in", "trade", f"{s + 1}-01-15"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)
    row = panel[(panel.player_id == 9_930_004) & (panel.s == s)]
    assert len(row) == 1
    # If the cutoff were missing/wrong, this would incorrectly read changed_team == 1.0.
    assert row.iloc[0]["changed_team"] == 0.0


def test_october_1_cutoff_includes_a_late_august_signing(tables):
    """Sanity check the other direction: a transaction clearly BEFORE October 1 is honored."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_005, name="PreseasonSigning", position="F", minutes=18.0, n_games=10)],
    )
    other_teams = [tid for tid in t["team_games"]["team_id"].unique() if tid != team_a]
    team_b = other_teams[0]
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_005, name="PreseasonSigning", position="F", minutes=18.0, n_games=10)],
        team_of_season=team_b,
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_b, 9_930_005, "in", "addition", f"{s}-09-20"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)
    row = panel[(panel.player_id == 9_930_005) & (panel.s == s)]
    assert len(row) == 1
    assert row.iloc[0]["changed_team"] == 1.0


# --------------------------------------------------------------------------- team_departures_lost

def test_team_departures_lost_sums_departing_role_share(tables):
    """Two players on team A in PATTERN_SEASON; one of them departs (dated "out") before the target
    season's cutoff. A third player actually arrives and plays for team A in TXN_SEASON. His
    team_departures_lost must equal the departing player's own PATTERN_SEASON role_share."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[
            dict(player_id=9_930_010, name="Departing Star", position="G", minutes=32.0, n_games=10,
                fga=18, oreb=1, tov=2, fta=6),
            dict(player_id=9_930_011, name="Remaining Player", position="F", minutes=20.0, n_games=10),
        ],
    )
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_012, name="Arriving Player", position="G", minutes=26.0, n_games=10)],
        team_of_season=team_a,
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_a, 9_930_010, "out", "subtraction", f"{s}-07-10"),
        _txn_row(TXN_SEASON, team_a, 9_930_012, "in", "addition", f"{s}-07-10"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)

    # Hand-computed role_share for the departing star in PATTERN_SEASON:
    # team_min_per_game = 32 + 20 = 52; role_share = 32 / (52/5)
    expected_role_share = 32.0 / ((32.0 + 20.0) / 5.0)

    arriving_row = panel[(panel.player_id == 9_930_012) & (panel.s == s)]
    assert len(arriving_row) == 1
    assert arriving_row.iloc[0]["team_departures_lost"] == pytest.approx(expected_role_share)
    assert arriving_row.iloc[0]["changed_team"] == 1.0  # no prior season, has an "in" txn


def test_departing_player_with_no_prior_role_share_contributes_zero(tables):
    """A departing player with no PATTERN_SEASON-1 game_logs data (unresolvable role_share)
    contributes 0 to the sum, not a crash and not NaN."""
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_020, name="Remaining", position="F", minutes=20.0, n_games=10)],
    )
    t, _ = add_team_season_roster(
        t, TXN_SEASON,
        players=[dict(player_id=9_930_021, name="Arriving", position="G", minutes=22.0, n_games=10)],
        team_of_season=team_a,
    )
    s = season_start(TXN_SEASON)
    # 9_930_099 has never appeared in game_logs at all -- unresolvable departing player.
    txn = pd.DataFrame([
        _txn_row(TXN_SEASON, team_a, 9_930_099, "out", "subtraction", f"{s}-07-10"),
        _txn_row(TXN_SEASON, team_a, 9_930_021, "in", "addition", f"{s}-07-10"),
    ])
    h = History.until(t, TARGET)
    panel = build_transactions_panel(h, txn)
    row = panel[(panel.player_id == 9_930_021) & (panel.s == s)]
    assert len(row) == 1
    val = row.iloc[0]["team_departures_lost"]
    assert not np.isnan(val)
    assert val == 0.0


# --------------------------------------------------------------------------- serving the real target season

def test_for_players_serves_the_real_target_season_with_no_game_logs_yet(tables):
    """The whole point of TransactionContext.for_players: query an arbitrary (pids, target_s) pair
    for the REAL target season, whose players have not played yet and so have no row in
    build_transactions_panel's output at all. A player with a tracked TARGET-1 primary team and a
    dated "in" transaction to a different team before TARGET's October 1 must resolve to
    changed_team == 1.0 for TARGET, even though TARGET itself has zero game_logs rows in this
    History (History.until excludes it entirely)."""
    prev_season = "2017-18"  # TARGET - 1
    t, team_a = add_team_season_roster(
        tables, prev_season,
        players=[dict(player_id=9_930_050, name="FutureMover", position="F", minutes=24.0, n_games=10)],
    )
    other_teams = [tid for tid in t["team_games"]["team_id"].unique() if tid != team_a]
    team_b = other_teams[0]
    s = season_start(TARGET)
    txn = pd.DataFrame([_txn_row(TARGET, team_b, 9_930_050, "in", "trade", f"{s}-08-01")])
    h = History.until(t, TARGET)
    # TARGET itself contributes no game_logs rows at all for this player (History.until excludes it).
    assert not ((h.game_logs["player_id"] == 9_930_050) & (h.game_logs["season"] == TARGET)).any()

    ctx = TransactionContext.build(h, txn)
    out = ctx.for_players(np.array([9_930_050]), np.array([s]))
    assert out[0, 0] == 1.0  # changed_team


# --------------------------------------------------------------------------- safe defaults / determinism

def test_no_previous_season_and_no_transaction_gets_safe_zero_defaults():
    """A brand-new player_id, never seen in game_logs, and with no transaction record at all --
    for_players() must return 0.0/0.0, never crash, never NaN."""
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    ctx = TransactionContext.build(h, _empty_transactions())
    out = ctx.for_players(np.array([12345]), np.array([season_start(TARGET)]))
    assert out.shape == (1, 2)
    assert not np.isnan(out).any()
    assert np.all(out == 0.0)


def test_build_transactions_panel_on_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    panel = build_transactions_panel(h, _empty_transactions())
    assert panel.empty
    for c in ("player_id", "s", "changed_team", "team_departures_lost"):
        assert c in panel.columns


def test_build_transactions_panel_deterministic(tables):
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_030, name="Repeat", position="F", minutes=24.0, n_games=10)],
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([_txn_row(TXN_SEASON, team_a, 9_930_030, "in", "addition", f"{s}-08-01")])
    h = History.until(t, TARGET)
    a = build_transactions_panel(h, txn)
    b = build_transactions_panel(h, txn)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# --------------------------------------------------------------------------- TransactionFeatures

def test_transaction_features_fit_falls_back_to_zero_with_too_few_rows(tables):
    h = History.until(tables, TARGET)
    n = MIN_FIT_ROWS - 1
    pids = np.arange(1, n + 1)
    target_s = np.full(n, season_start(TARGET))
    mpg_est = np.full(n, 20.0)
    actual_mpg = mpg_est + 5.0
    weight = np.full(n, 70.0)
    feats = TransactionFeatures.fit(h, _empty_transactions(), pids, target_s, mpg_est, actual_mpg, weight)
    assert np.all(feats.beta == 0.0)
    out = feats.build(pids, target_s)
    assert np.all(out == 0.0)


def test_transaction_features_fit_on_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["season", "game_id", "player_id"])
    h = History(target_season=TARGET, game_logs=empty, team_games=empty, players=empty, player_season_bio=empty)
    feats = TransactionFeatures.fit(h, _empty_transactions(), np.array([1, 2]), np.array([2018, 2018]),
                                    np.array([20.0, 20.0]), np.array([22.0, 18.0]), np.array([70.0, 70.0]))
    assert np.all(feats.beta == 0.0)
    out = feats.build(np.array([1, 2]), np.array([2018, 2018]))
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()


def test_transaction_features_build_clips_to_mpg_shift_cap(tables):
    h = History.until(tables, TARGET)
    ctx = TransactionContext.build(h, _empty_transactions())
    extreme_beta = np.array([1000.0, 1000.0, 1000.0])
    feats = TransactionFeatures(ctx, extreme_beta)
    pids = np.array([1, 2, 3])
    target_s = np.full(3, season_start(TARGET))
    out = feats.build(pids, target_s)
    assert np.all(np.abs(out) <= MPG_SHIFT_CAP + 1e-9)


def test_transaction_features_build_deterministic(tables):
    t, team_a = add_team_season_roster(
        tables, PATTERN_SEASON,
        players=[dict(player_id=9_930_040, name="Repeat2", position="F", minutes=28.0, n_games=15)],
    )
    s = season_start(TXN_SEASON)
    txn = pd.DataFrame([_txn_row(TXN_SEASON, team_a, 9_930_040, "in", "addition", f"{s}-08-01")])
    h = History.until(t, TARGET)
    pids = np.array([9_930_040])
    target_s = np.array([s])
    mpg_est = np.full(1, 20.0)
    actual_mpg = mpg_est + 1.0
    weight = np.full(1, 70.0)
    a = TransactionFeatures.fit(h, txn, pids, target_s, mpg_est, actual_mpg, weight)
    b = TransactionFeatures.fit(h, txn, pids, target_s, mpg_est, actual_mpg, weight)
    out_a = a.build(pids, target_s)
    out_b = b.build(pids, target_s)
    np.testing.assert_array_equal(out_a, out_b)


def test_min_fit_rows_and_mpg_shift_cap_documented_values():
    assert MIN_FIT_ROWS == 60
    assert MPG_SHIFT_CAP == pytest.approx(4.0)
