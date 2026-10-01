"""Draft-in-progress state, board filtering and suggestion logic — pure functions, no Streamlit.
This is the module where a bug would actually bite during a live draft, so it gets the deepest
coverage in this track."""
from __future__ import annotations

import pandas as pd
import pytest

from src.app.state import (
    ME,
    OPPONENT,
    DraftError,
    DraftState,
    best_available,
    best_by_position,
    draft_player,
    dict_to_state,
    filter_board,
    json_to_state,
    my_position_counts,
    reset,
    state_to_dataframe,
    state_to_dict,
    state_to_json,
    undraft_player,
)

# player_id, name, position (dataset string), tier
_ROWS = [
    (1, "Point Guard One", "PG", 1),
    (2, "Shooting Guard One", "SG", 1),
    (3, "Small Forward One", "SF", 2),
    (4, "Power Forward One", "PF", 2),
    (5, "Center One", "C", 3),
    (6, "Guard Flex", "G", 3),
    (7, "Forward Flex", "F", 4),
    (8, "Unknown Position", "???", 4),
]


def make_board() -> pd.DataFrame:
    rows = []
    for i, (pid, name, pos, tier) in enumerate(_ROWS, start=1):
        rows.append({
            "rank": i, "player_id": pid, "name": name, "position": pos, "tier": tier,
            "proj_fppg": 50.0 - i, "proj_gp": 70.0, "proj_total_fp": (50.0 - i) * 70,
            "vorp": 100.0 - i * 5, "vorp_per_game": 1.0, "fppg_p10": 30.0, "fppg_p50": 45.0, "fppg_p90": 60.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def board() -> pd.DataFrame:
    return make_board()


# --------------------------------------------------------------------------- DraftState basics

def test_new_state_is_empty():
    s = DraftState()
    assert len(s) == 0
    assert s.drafted_ids == set()
    assert s.my_ids == set()
    assert s.opponent_ids == set()


def test_draft_player_adds_a_pick_and_is_immutable():
    s0 = DraftState()
    s1 = draft_player(s0, 1, "Point Guard One", "PG", ME)
    assert len(s0) == 0, "original state must not be mutated"
    assert len(s1) == 1
    assert s1.drafted_ids == {1}
    assert s1.my_ids == {1}
    assert s1.picks[0].pick_no == 1


def test_pick_numbers_increment_across_both_teams():
    s = DraftState()
    s = draft_player(s, 1, "A", "PG", ME)
    s = draft_player(s, 2, "B", "SG", OPPONENT)
    s = draft_player(s, 3, "C", "SF", ME)
    assert [p.pick_no for p in s.picks] == [1, 2, 3]
    assert s.my_ids == {1, 3}
    assert s.opponent_ids == {2}


def test_drafting_the_same_player_twice_raises():
    s = draft_player(DraftState(), 1, "A", "PG", ME)
    with pytest.raises(DraftError, match="already drafted"):
        draft_player(s, 1, "A", "PG", OPPONENT)


def test_drafting_with_a_bad_drafted_by_raises():
    with pytest.raises(DraftError, match="drafted_by"):
        draft_player(DraftState(), 1, "A", "PG", "someone_else")


def test_undraft_removes_the_pick():
    s = draft_player(DraftState(), 1, "A", "PG", ME)
    s = draft_player(s, 2, "B", "SG", OPPONENT)
    s2 = undraft_player(s, 1)
    assert s2.drafted_ids == {2}
    assert s.drafted_ids == {1, 2}, "original state must not be mutated"


def test_undraft_unknown_player_raises():
    s = draft_player(DraftState(), 1, "A", "PG", ME)
    with pytest.raises(DraftError, match="not currently drafted"):
        undraft_player(s, 999)


def test_undraft_then_redraft_reuses_the_player_id():
    s = draft_player(DraftState(), 1, "A", "PG", ME)
    s = undraft_player(s, 1)
    s = draft_player(s, 1, "A", "PG", OPPONENT)
    assert s.drafted_ids == {1}
    assert s.opponent_ids == {1}


def test_reset_returns_an_empty_state_regardless_of_input():
    s = draft_player(DraftState(), 1, "A", "PG", ME)
    assert len(reset(s)) == 0


# --------------------------------------------------------------------------- best_available / filter_board

def test_best_available_excludes_drafted_players(board):
    avail = best_available(board, {2, 4})
    assert set(avail["player_id"]) == {1, 3, 5, 6, 7, 8}
    assert list(avail["rank"]) == sorted(avail["rank"]), "keeps board order"


def test_best_available_with_no_drafted_players_returns_everything(board):
    avail = best_available(board, set())
    assert len(avail) == len(board)


def test_best_available_with_everyone_drafted_is_empty(board):
    avail = best_available(board, set(board["player_id"]))
    assert avail.empty


def test_filter_board_by_specific_position(board):
    filtered = filter_board(board, position="PG")
    # Player 6's dataset position "G" is PG/SG-eligible, so it legitimately matches too.
    assert set(filtered["player_id"]) == {1, 6}


def test_filter_board_by_flex_guard_includes_pg_sg_and_g(board):
    filtered = filter_board(board, position="G")
    # G dataset string maps to PG+SG eligibility; specific PG/SG players and the "G" flex player
    # (also PG/SG-eligible) should all show up.
    assert {1, 2, 6} <= set(filtered["player_id"])


def test_filter_board_util_includes_everyone_even_unknown_positions(board):
    # src.value.positions.eligible_slots: every rostered player fills UTIL, known position or not.
    filtered = filter_board(board, position="UTIL")
    assert set(filtered["player_id"]) == set(board["player_id"])


def test_filter_board_all_position_is_a_no_op(board):
    assert len(filter_board(board, position="All")) == len(board)
    assert len(filter_board(board, position=None)) == len(board)


def test_filter_board_by_tier(board):
    filtered = filter_board(board, tier=2)
    assert set(filtered["player_id"]) == {3, 4}


def test_filter_board_tier_all_is_a_no_op(board):
    assert len(filter_board(board, tier="All")) == len(board)


def test_filter_board_by_search_is_case_insensitive_substring(board):
    filtered = filter_board(board, search="guard")
    assert set(filtered["player_id"]) == {1, 2, 6}
    filtered2 = filter_board(board, search="POINT")
    assert set(filtered2["player_id"]) == {1}


def test_filter_board_search_blank_is_a_no_op(board):
    assert len(filter_board(board, search="   ")) == len(board)


def test_filter_board_combines_filters(board):
    filtered = filter_board(board, position="G", search="flex")
    assert set(filtered["player_id"]) == {6}


# --------------------------------------------------------------------------- suggestions

def test_best_by_position_returns_top_n_per_position(board):
    suggestions = best_by_position(board, drafted_ids=set(), n=1)
    assert list(suggestions["PG"]["player_id"]) == [1]
    assert list(suggestions["SG"]["player_id"]) == [2]
    assert list(suggestions["C"]["player_id"]) == [5]


def test_best_by_position_excludes_drafted_players(board):
    suggestions = best_by_position(board, drafted_ids={1}, n=3)
    # player 1 (PG) is drafted; the "G" flex player (6) is PG-eligible and should surface instead.
    assert 1 not in set(suggestions["PG"]["player_id"])
    assert 6 in set(suggestions["PG"]["player_id"])


def test_best_by_position_empty_when_position_exhausted(board):
    drafted = {row["player_id"] for _, row in board.iterrows() if row["position"] in ("C",)}
    suggestions = best_by_position(board, drafted_ids=drafted, n=3)
    assert suggestions["C"].empty


def test_my_position_counts_with_no_picks_is_all_zero(board):
    counts = my_position_counts(board, set())
    assert all(v == 0 for v in counts.values())


def test_my_position_counts_reflects_drafted_eligibility(board):
    # players 1 (PG), 6 (G -> PG/SG) both count toward PG.
    counts = my_position_counts(board, {1, 6})
    assert counts["PG"] == 2
    assert counts["SG"] == 1
    assert counts["C"] == 0


# --------------------------------------------------------------------------- export / import

def test_state_round_trips_through_json():
    s = DraftState()
    s = draft_player(s, 1, "A", "PG", ME)
    s = draft_player(s, 2, "B", "SG", OPPONENT)
    text = state_to_json(s, meta={"season": "2026-27", "model": "baseline"})
    restored, meta = json_to_state(text)
    assert restored.picks == s.picks
    assert meta == {"season": "2026-27", "model": "baseline"}


def test_state_round_trips_through_dict():
    s = draft_player(DraftState(), 3, "C", "SF", ME)
    d = state_to_dict(s)
    restored, meta = dict_to_state(d)
    assert restored.picks == s.picks
    assert meta == {}


def test_import_rejects_invalid_json():
    with pytest.raises(DraftError, match="not valid JSON"):
        json_to_state("{not json")


def test_import_rejects_missing_picks_key():
    with pytest.raises(DraftError):
        dict_to_state({"meta": {}})


def test_import_rejects_duplicate_player_ids():
    data = {"picks": [
        {"player_id": 1, "name": "A", "position": "PG", "drafted_by": "me", "pick_no": 1},
        {"player_id": 1, "name": "A", "position": "PG", "drafted_by": "opponent", "pick_no": 2},
    ]}
    with pytest.raises(DraftError, match="duplicate"):
        dict_to_state(data)


def test_import_rejects_bad_drafted_by():
    data = {"picks": [{"player_id": 1, "name": "A", "position": "PG", "drafted_by": "nobody"}]}
    with pytest.raises(DraftError, match="drafted_by"):
        dict_to_state(data)


def test_import_sorts_picks_by_pick_no_regardless_of_file_order():
    data = {"picks": [
        {"player_id": 2, "name": "B", "position": "SG", "drafted_by": "me", "pick_no": 2},
        {"player_id": 1, "name": "A", "position": "PG", "drafted_by": "me", "pick_no": 1},
    ]}
    restored, _ = dict_to_state(data)
    assert [p.player_id for p in restored.picks] == [1, 2]


def test_state_to_dataframe_empty_state_has_expected_columns():
    df = state_to_dataframe(DraftState())
    assert list(df.columns) == ["pick_no", "player_id", "name", "position", "drafted_by"]
    assert df.empty


def test_state_to_dataframe_orders_by_pick_no():
    s = DraftState()
    s = draft_player(s, 1, "A", "PG", ME)
    s = draft_player(s, 2, "B", "SG", OPPONENT)
    df = state_to_dataframe(s)
    assert list(df["pick_no"]) == [1, 2]
    assert list(df["player_id"]) == [1, 2]


def test_display_columns_add_adp_only_when_the_board_has_it():
    import pandas as pd

    from src.app.state import BOARD_COLUMNS, display_columns

    plain = pd.DataFrame(columns=BOARD_COLUMNS)
    assert display_columns(plain) == BOARD_COLUMNS
    with_adp = pd.DataFrame(columns=[*BOARD_COLUMNS, "adp", "adp_gap"])
    cols = display_columns(with_adp)
    assert cols[7:9] == ["adp", "adp_gap"] and cols[:7] == BOARD_COLUMNS[:7] and cols[9:] == BOARD_COLUMNS[7:]
    only_adp = pd.DataFrame(columns=[*BOARD_COLUMNS, "adp"])
    assert display_columns(only_adp)[7] == "adp" and "adp_gap" not in display_columns(only_adp)


# --------------------------------------------------------------------------- best available by need (ADR 0026)

_LEAGUE_CFG = {
    "league": {
        "format": "h2h_points",
        "roster": {
            "size": 13,
            "starters": {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3},
            "bench": 3,
            "ir": 1,
        },
    }
}


def test_need_board_view_empty_roster_is_fallback_and_vorp_ordered(board):
    from src.app.state import need_board_view

    view = need_board_view(board, DraftState(), _LEAGUE_CFG)
    assert view.fallback is True
    assert view.categories_applicable is False
    # rank-identical to plain vorp ordering (fallback: constant shift for every known-position player)
    known = view.table[view.table["position"] != "???"]
    assert list(known.sort_values("need_rank")["name"]) == \
        list(known.sort_values("vorp", ascending=False)["name"])
    assert "need_score" in view.table.columns and "need_adj_vorp" in view.table.columns
    assert len(view.position_table) == 5


def test_need_board_view_excludes_drafted_players_and_reflects_my_roster(board):
    from src.app.state import need_board_view

    s = draft_player(DraftState(), 5, "Center One", "C", ME)   # I drafted the only center
    view = need_board_view(board, s, _LEAGUE_CFG)
    assert view.fallback is False
    assert "Center One" not in set(view.table["name"])         # drafted players are excluded
    c_need = view.position_table.loc[view.position_table["need_position"] == "C", "need_score"].iloc[0]
    pg_need = view.position_table.loc[view.position_table["need_position"] == "PG", "need_score"].iloc[0]
    assert pg_need > c_need   # PG still wide open, C partially covered by my one drafted center


@pytest.mark.parametrize("data", [
    {"picks": ["not a dict"]},
    {"picks": [5]},
    {"picks": "abc"},
    {"picks": [{"player_id": 1, "drafted_by": "me", "name": 7}]},
    {"picks": [{"player_id": 1, "drafted_by": "me", "pick_no": 1}, {"player_id": 2, "drafted_by": "me", "pick_no": 1}]},
    {"picks": [], "meta": [1]},
    ["picks"],
])
def test_import_malformed_shapes_raise_draft_error_not_attribute_error(data):
    with pytest.raises(DraftError):
        dict_to_state(data)
