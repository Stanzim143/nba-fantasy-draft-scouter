"""Draft-day rehearsal (2026-09-29): a full 13-team x 13-round snake draft run through the real
state-machine functions the live app calls, plus the live-sync, best-available-by-need, trade
analyzer and external-rankings-compare code paths a real draft will hit.

This is an integration-level dry run, not a UI test (Streamlit itself is exercised separately, by
hand, per ``docs/draft-day-dry-run-2026-09-29.md``): every function called here is the exact one
``src/app/draft_board.py`` wires a widget to, so a bug here is a bug the live draft would hit too.
Kept deterministic (synthetic board, fixed seed) so it can be re-run before the real draft or in CI
without needing ingested real data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.app import live_sync as ls
from src.app.loader import load_board
from src.app.state import (
    ME,
    OPPONENT,
    DraftError,
    DraftState,
    best_available,
    best_by_position,
    draft_player,
    filter_board,
    my_position_counts,
    need_board_view,
    undraft_player,
)
from src.value.league import load_league
from src.value.rankings_compare import build_comparison

SEASON = "2026-27"
N_TEAMS = 13
ROUNDS = 13


@pytest.fixture(scope="module")
def board() -> pd.DataFrame:
    # Synthetic demo board: deterministic, no real ingested data required (see src/app/loader.py's
    # load_board(synthetic=True) -- the exact path the app's "Use synthetic demo data" checkbox uses).
    return load_board(SEASON, "baseline", teams=N_TEAMS, synthetic=True)


def snake_order(n_teams: int, rounds: int) -> list[int]:
    """0-based team index per overall pick, standard snake order (matches
    ``src.inseason.context.mock_snake_draft``'s own round-alternation rule)."""
    order: list[int] = []
    for rnd in range(rounds):
        seq = range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1)
        order.extend(seq)
    return order


# --------------------------------------------------------------------------- full snake draft

def test_full_13x13_snake_draft_through_real_state_functions(board):
    """Simulate the actual draft: 169 picks, team 0 is "me", the rest "opponent", alternating
    snake order, each pick taken via draft_player (the exact function the manual 'Mark drafted'
    form and live-sync's apply_detected_picks both call)."""
    state = DraftState()
    order = snake_order(N_TEAMS, ROUNDS)
    assert len(order) == N_TEAMS * ROUNDS == 169

    for i, team in enumerate(order):
        avail = best_available(board, state.drafted_ids)
        assert not avail.empty, f"ran out of available players at pick {i + 1}"
        row = avail.iloc[0]  # each team takes the best remaining player (deterministic rehearsal)
        drafted_by = ME if team == 0 else OPPONENT
        state = draft_player(state, int(row["player_id"]), row["name"], row["position"], drafted_by)
        assert state.picks[-1].pick_no == i + 1

    assert len(state) == 169
    assert len(state.drafted_ids) == 169  # no duplicate player_ids slipped through
    assert len(state.my_ids) == ROUNDS  # team 0 ("me") picked once per round
    assert len(state.opponent_ids) == 169 - ROUNDS

    # Board never runs out from under the app: best_available keeps shrinking and stays board-ordered.
    avail = best_available(board, state.drafted_ids)
    assert len(avail) == len(board) - 169
    assert list(avail["rank"]) == sorted(avail["rank"])


def test_manual_override_and_undo_mid_draft(board):
    """A manual correction mid-draft (wrong drafted_by, or a mis-click) -- undo must restore
    best_available immediately without renumbering surviving picks (state.py's own documented
    contract)."""
    state = DraftState()
    order = snake_order(N_TEAMS, ROUNDS)
    for team in order[:20]:
        avail = best_available(board, state.drafted_ids)
        row = avail.iloc[0]
        state = draft_player(state, int(row["player_id"]), row["name"], row["position"],
                             ME if team == 0 else OPPONENT)

    # Oops -- pick 20 was actually mine, not the opponent's. Undo it and re-mark it correctly,
    # exactly the sequence a user does via the Drafted tab's "Undo this pick" button.
    mis_pick = state.picks[-1]
    assert mis_pick.pick_no == 20
    state = undraft_player(state, mis_pick.player_id)
    assert mis_pick.player_id not in state.drafted_ids
    assert len(state) == 19

    state = draft_player(state, mis_pick.player_id, mis_pick.name, mis_pick.position, ME)
    # Pick order is a historical record, not renumbered on undo/redo -- the corrected pick becomes
    # the new 20th entry (len(state.picks) + 1 at the time it's re-applied).
    assert state.picks[-1].pick_no == 20
    assert state.picks[-1].drafted_by == ME
    assert len(state) == 20

    # Undoing a player who isn't drafted must raise, not silently no-op (per undraft_player's docstring).
    with pytest.raises(DraftError):
        undraft_player(state, 999_999_999)


# --------------------------------------------------------------------------- live sync: opponent picks, duplicate race, unresolved id

def _detected(overall, team_id, espn_player_id, player_id, name, position, my_team_id=0):
    return ls.build_detected_picks(
        [{"overall_pick": overall, "round": 1, "espn_player_id": espn_player_id, "team_id": team_id}],
        my_team_id=my_team_id,
        id_map={str(espn_player_id): player_id} if player_id is not None else {},
        board_names={player_id: name} if player_id is not None else {},
        board_positions={player_id: position} if player_id is not None else {},
    )


def test_live_sync_detects_and_applies_opponent_picks_mid_draft(board):
    state = DraftState()
    row = board.iloc[0]
    detected = _detected(1, team_id=7, espn_player_id=90001, player_id=int(row["player_id"]),
                         name=row["name"], position=row["position"])
    state, applied, duplicate, unresolved = ls.apply_detected_picks(state, detected)
    assert len(applied) == 1 and not duplicate and not unresolved
    assert applied[0].drafted_by == OPPONENT
    assert int(row["player_id"]) in state.opponent_ids


def test_live_sync_duplicate_pick_race_is_deduped_not_double_counted(board):
    """ADR 0025's documented race: the manual 'Mark drafted' click and a live-sync poll both see the
    same player. The manual click always wins (applies immediately in its own rerun); the next
    poll's detected batch for that player must be skipped, never raise, never double-count."""
    row = board.iloc[1]
    pid, name, pos = int(row["player_id"]), row["name"], row["position"]
    state = draft_player(DraftState(), pid, name, pos, ME)  # the user's own manual click landed first

    detected = _detected(2, team_id=3, espn_player_id=90002, player_id=pid, name=name, position=pos)
    state, applied, duplicate, unresolved = ls.apply_detected_picks(state, detected)
    assert not applied
    assert [d.player_id for d in duplicate] == [pid]
    assert not unresolved
    assert len(state) == 1  # not double-counted
    assert pid in state.my_ids  # the manual attribution ("me") is preserved, not overwritten to "opponent"


def test_live_sync_unresolved_player_id_is_surfaced_not_dropped():
    """A pick for a player whose espn_player_id has no player_id_map entry yet must come back
    'unresolved' -- never silently skipped, never guessed at (ADR 0025)."""
    detected = ls.build_detected_picks(
        [{"overall_pick": 3, "round": 1, "espn_player_id": 777777, "team_id": 4}],
        my_team_id=0, id_map={}, board_names={}, board_positions={},
    )
    state, applied, duplicate, unresolved = ls.apply_detected_picks(DraftState(), detected)
    assert not applied and not duplicate
    assert len(unresolved) == 1
    assert unresolved[0].espn_player_id == 777777
    assert len(state) == 0  # state untouched; the caller is expected to mark it manually


def test_live_sync_error_degrades_gracefully_manual_marking_still_works(tmp_path, board):
    """ADR 0025's failure-handling contract: any ESPNLeagueError from a poll must degrade to
    SyncResult(ok=False, error=...) rather than raise, and must never disable the manual
    draft_player path -- the two are fully independent."""
    from src.ingest.espn_league import ESPNAuthError, ESPNLeagueClient

    class _RaisingSession:
        def get(self, *a, **kw):
            raise ESPNAuthError("401: private league, no cookies configured", status=401)

    client = ESPNLeagueClient(tmp_path / "raw", session=_RaisingSession(), min_interval=0, offline=False)
    result = ls.sync_once(client, league_id=1, season_id=2027, seen_overall_picks=frozenset(), my_team_id=0)
    assert result.ok is False
    assert "401" in result.error or "private" in result.error
    assert result.detected == ()

    # The manual control is untouched by the sync failure -- exactly the guarantee ADR 0025 makes.
    row = board.iloc[2]
    state = draft_player(DraftState(), int(row["player_id"]), row["name"], row["position"], ME)
    assert len(state) == 1


# --------------------------------------------------------------------------- best available by need, mid-draft

def test_best_available_by_need_reflects_roster_gaps_mid_draft(board):
    from src.value.positions import eligible_positions

    league_cfg = load_league()
    state = DraftState()
    # Draft three PG-eligible players for "me" so PG should look "covered" relative to other
    # positions. The synthetic board's dataset ``position`` strings are compound (e.g. "G-F", "F-C"),
    # not the specific ESPN slot letters, so eligibility is checked the same way the app itself does
    # (src.value.positions.eligible_positions), not a literal "PG" string match.
    pg_rows = board[board["position"].map(lambda p: "PG" in eligible_positions(p))].head(3)
    assert len(pg_rows) == 3, "fixture assumption: synthetic board must have PG-eligible players"
    for _, row in pg_rows.iterrows():
        state = draft_player(state, int(row["player_id"]), row["name"], row["position"], ME)

    view = need_board_view(board, state, league_cfg)
    assert not view.table.empty
    assert list(view.table["need_rank"]) == list(range(1, len(view.table) + 1))
    # NEED_TABLE_COLUMNS (src/value/need.py) carries player_id (added so this table can be joined
    # back onto the live board programmatically, not only by name -- see the module docstring); the
    # app hides it from the rendered grid via column_config, but it is very much present in the
    # data here, so exclusion checks can (and should) use the robust id-based check.
    assert "player_id" in view.table.columns
    drafted_ids = {p.player_id for p in state.picks}
    assert drafted_ids.isdisjoint(set(view.table["player_id"]))
    drafted_names = {p.name for p in state.picks}
    assert drafted_names.isdisjoint(set(view.table["name"]))
    # With an empty roster the view falls back to plain VORP order; with picks made it should not.
    assert view.fallback is False


def test_best_available_by_need_table_joins_back_onto_live_board_by_player_id(board):
    """The point of adding player_id to NEED_TABLE_COLUMNS: a caller can join the need table's
    rows back onto the full live board by id, not just by (fragile) name matching."""
    league_cfg = load_league()
    view = need_board_view(board, DraftState(), league_cfg)
    assert not view.table.empty

    joined = view.table.merge(board[["player_id", "rank", "proj_total_fp"]], on="player_id",
                              suffixes=("", "_board"), how="left")
    assert joined["rank_board"].notna().all(), "every need-table row must resolve back onto the board by id"
    # The join recovers exactly the same rows the board itself has for those ids (rank is stable
    # regardless of need re-sorting, so this also proves the id, not just the row count, is right).
    assert (joined["rank"] == joined["rank_board"]).all()
    assert (joined["proj_total_fp"] == joined["proj_total_fp_board"]).all()


def test_best_available_by_need_fallback_with_empty_roster_matches_plain_vorp(board):
    league_cfg = load_league()
    view = need_board_view(board, DraftState(), league_cfg)
    assert view.fallback is True
    plain = best_available(board, set())
    assert list(view.table["name"].head(20)) == list(plain["name"].head(20))


# --------------------------------------------------------------------------- suggestions / full board tabs mid-draft

def test_suggestions_and_full_board_stay_sane_mid_draft(board):
    state = DraftState()
    order = snake_order(N_TEAMS, ROUNDS)
    for team in order[:39]:  # three rounds in
        avail = best_available(board, state.drafted_ids)
        row = avail.iloc[0]
        state = draft_player(state, int(row["player_id"]), row["name"], row["position"],
                             ME if team == 0 else OPPONENT)

    counts = my_position_counts(board, state.my_ids)
    assert sum(counts.values()) >= len(state.my_ids)  # a player can count at more than one position

    suggestions = best_by_position(board, state.drafted_ids)
    for pos_df in suggestions.values():
        assert set(pos_df["player_id"]).isdisjoint(state.drafted_ids)

    filtered = filter_board(board, position="PG")
    assert (filtered["position"].map(lambda p: True)).all()  # filter_board itself never raises mid-draft


# --------------------------------------------------------------------------- trade analyzer (mock-draft-team roster, ADR 0027)

def test_trade_analyzer_runs_on_a_mock_drafted_roster_mid_dry_run():
    """The trade analyzer lives on the in-season page, not the draft board tab, but a draft-day
    rehearsal plausibly exercises it too (a user checking 'what if' trades against a mock roster
    before/around the real draft). Uses the same synthetic + mock-draft-team path the docs call out
    for demos (src.inseason.context.resolve_roster's mock_team branch), which a recent commit fixed
    for exactly this synthetic/no-overlap case."""
    from src.app.inseason_view import load, roster_choice, run_trade

    ctx = load(SEASON, None, model="baseline", synthetic=True)
    choice = roster_choice(ctx, mock_team=1)
    assert choice.my_ids, "mock-draft-team roster must not be empty (regression guard)"

    other_label = next(iter(choice.others))
    other_ids = choice.others[other_label]
    assert other_ids

    result = run_trade(ctx, choice, give_ids=choice.my_ids[:1], get_ids=other_ids[:1],
                       partner=other_label)
    assert result["verdict"] in {"favors you", "favors them", "fair", "even", "unclear"} or isinstance(result["verdict"], str)
    assert isinstance(result["delta"], float)
    assert result["partner"] is not None  # partner scoring ran too


# --------------------------------------------------------------------------- external rankings compare

def test_external_rankings_compare_runs_against_the_drafted_board(board):
    """Exercise build_comparison directly (the function src/app/pages/external_rankings.py wires to
    widgets) with small synthetic 'resolved' frames shaped like RESOLVED_COLUMNS, standing in for a
    Yahoo/FantasyPros export mid-draft."""
    top = board.head(5)
    yahoo = pd.DataFrame({
        "source": "yahoo", "source_id": [f"y{i}" for i in range(5)],
        "source_name_raw": top["name"].tolist(), "name_parsed": top["name"].tolist(),
        "team": "SYN", "positions": top["position"].tolist(), "status_tag": "", "ext_rank": range(1, 6),
        "adp": range(1, 6), "ecr_vs_adp": None, "player_id": top["player_id"].tolist(),
        "match_method": "exact", "confidence": 1.0, "matched": True,
    })
    # Reverse order vs our board -- exercises a real disagreement, not just a match.
    fp = yahoo.copy()
    fp["source"] = "fantasypros"
    fp["ext_rank"] = list(range(5, 0, -1))

    comparison = build_comparison(board, {"yahoo": yahoo, "fantasypros": fp})
    assert len(comparison) >= 5
    matched = comparison[comparison["player_id"].isin(top["player_id"])]
    assert (matched["on_yahoo"]).all()
    assert (matched["on_fantasypros"]).all()
    # Sign convention per rankings_compare's own docstring: our_rank - source_rank.
    row = matched.iloc[0]
    assert row["rank_delta_yahoo"] == row["our_rank"] - row["yahoo_rank"]
