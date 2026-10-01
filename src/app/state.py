"""Draft-in-progress state and best-available logic, kept as plain functions/dataclasses so they
are unit-testable without Streamlit (see ``tests/app/test_state.py``). ``draft_board.py`` imports
this module and keeps ``st.session_state`` as the only place these objects live across reruns —
no ``st.*`` calls happen in here.

``DraftState`` is immutable: every mutation (``draft_player``, ``undraft_player``) returns a new
``DraftState`` rather than mutating in place. That makes it trivial to reason about and test, and
plays well with Streamlit's rerun model (reassign ``st.session_state.draft_state = new_state``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Iterable

import pandas as pd

from src.value.need import NEED_TABLE_COLUMNS, compute_need_board, position_need_table
from src.value.positions import ESPN_POSITIONS, eligible_positions, eligible_slots

ME = "me"
OPPONENT = "opponent"
DRAFTED_BY_VALUES = (ME, OPPONENT)


class DraftError(ValueError):
    """A draft-state operation was invalid (e.g. drafting an already-drafted player)."""


@dataclass(frozen=True)
class Pick:
    player_id: int
    name: str
    position: str | None
    drafted_by: str          # "me" | "opponent"
    pick_no: int              # 1-based overall pick order


@dataclass(frozen=True)
class DraftState:
    picks: tuple[Pick, ...] = field(default_factory=tuple)

    @property
    def drafted_ids(self) -> set[int]:
        return {p.player_id for p in self.picks}

    @property
    def my_ids(self) -> set[int]:
        return {p.player_id for p in self.picks if p.drafted_by == ME}

    @property
    def opponent_ids(self) -> set[int]:
        return {p.player_id for p in self.picks if p.drafted_by == OPPONENT}

    def __len__(self) -> int:
        return len(self.picks)


def draft_player(state: DraftState, player_id: int, name: str, position: str | None,
                 drafted_by: str) -> DraftState:
    """Return a new state with ``player_id`` marked drafted. Raises ``DraftError`` if the player
    is already drafted or ``drafted_by`` is not a recognised value."""
    if drafted_by not in DRAFTED_BY_VALUES:
        raise DraftError(f"drafted_by must be one of {DRAFTED_BY_VALUES}, got {drafted_by!r}")
    if player_id in state.drafted_ids:
        raise DraftError(f"player_id {player_id} is already drafted")
    pick = Pick(player_id=int(player_id), name=name, position=position, drafted_by=drafted_by,
               pick_no=len(state.picks) + 1)
    return replace(state, picks=state.picks + (pick,))


def undraft_player(state: DraftState, player_id: int) -> DraftState:
    """Return a new state with ``player_id`` removed from the drafted list (undo a mistaken
    pick). No-op-safe: raises ``DraftError`` if the player was not drafted, so callers notice a
    stale reference rather than silently doing nothing. Remaining picks keep their original
    ``pick_no`` (pick order is a historical record, not renumbered on undo)."""
    if player_id not in state.drafted_ids:
        raise DraftError(f"player_id {player_id} is not currently drafted")
    return replace(state, picks=tuple(p for p in state.picks if p.player_id != player_id))


def reset(state: DraftState) -> DraftState:  # noqa: ARG001 - symmetric with the other verbs
    """A fresh, empty draft state."""
    return DraftState()


# --------------------------------------------------------------------------- board filtering

def best_available(board: pd.DataFrame, drafted_ids: Iterable[int]) -> pd.DataFrame:
    """``board`` (already ranked) with drafted players removed. Preserves the board's order."""
    drafted = set(drafted_ids)
    if not drafted:
        return board.reset_index(drop=True)
    return board[~board["player_id"].isin(drafted)].reset_index(drop=True)


def filter_board(board: pd.DataFrame, *, position: str | None = None, tier: object = None,
                 search: str | None = None, returned_only: bool = False, short_absences_only: bool = False) -> pd.DataFrame:
    """Pure filter used by both the "best available" table and the full board view.

    ``position`` is an ESPN lineup slot (``"PG"``, ``"G"``, ``"UTIL"``, ...) matched against each
    row's dataset position string via ``src.value.positions.eligible_slots`` (specific, flex, and
    UTIL, exactly the slots ESPN lineups use — UTIL always matches, per that module's own rule
    that every rostered player fills UTIL); ``None``/``"All"`` means no filter. ``tier`` filters
    on the exact tier value. ``search`` is a case-insensitive substring match on ``name``. ``returned_only`` keeps only the
    players carrying the advisory return flag (ADR 0023; a board without the column keeps everyone); ``short_absences_only`` does the
    same for the advisory short-absence flag (ADR 0024).
    """
    df = board
    if short_absences_only and "lm_flag" in df.columns:
        df = df[df["lm_flag"].fillna("") != ""]
    if returned_only and "return_flag" in df.columns:
        df = df[df["return_flag"].fillna("") != ""]
    if position and position != "All":
        mask = df["position"].map(lambda p: position in eligible_slots(p))
        df = df[mask]
    if tier not in (None, "All"):
        df = df[df["tier"] == tier]
    if search:
        s = search.strip().lower()
        if s:
            df = df[df["name"].str.lower().str.contains(s, na=False, regex=False)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- suggestions

def best_by_position(board: pd.DataFrame, drafted_ids: Iterable[int], *,
                     positions: tuple[str, ...] = ESPN_POSITIONS, n: int = 3) -> dict[str, pd.DataFrame]:
    """For each specific ESPN position, the top ``n`` remaining (undrafted) players eligible
    there, ranked as the board already ranks them (best VORP first). This is the "next best by
    position" panel — deliberately not a full auto-drafter / roster optimizer."""
    avail = best_available(board, drafted_ids)
    out: dict[str, pd.DataFrame] = {}
    for pos in positions:
        mask = avail["position"].map(lambda p, pos=pos: pos in eligible_positions(p))
        out[pos] = avail[mask].head(n).reset_index(drop=True)
    return out


def my_position_counts(board: pd.DataFrame, my_ids: Iterable[int],
                       positions: tuple[str, ...] = ESPN_POSITIONS) -> dict[str, int]:
    """How many of the user's own drafted players are eligible at each specific position — a
    quick "where am I already covered" readout alongside the suggestions panel."""
    ids = set(my_ids)
    if not ids:
        return {p: 0 for p in positions}
    mine = board[board["player_id"].isin(ids)]
    return {pos: int(mine["position"].map(lambda p, pos=pos: pos in eligible_positions(p)).sum())
           for pos in positions}


# --------------------------------------------------------------------------- best available by need (ADR 0026)

@dataclass(frozen=True)
class NeedBoardView:
    """Everything the "Best available by need" tab needs to render, built by ``need_board_view``.
    ``table`` is undrafted players sorted by ``need_adj_vorp`` (best first) with ``need_rank``
    reassigned 1..N over that order; ``position_table`` is the per-position need breakdown;
    ``fallback`` is True when the user has drafted no one yet, in which case ``table`` is
    rank-identical to the plain VORP board (see ``src.value.need`` for why) and the tab should say
    so; ``categories_applicable`` flags whether the league is a categories/roto format (it is not,
    for this league's ``h2h_points`` config — category need is not implemented, see ADR 0026)."""

    table: pd.DataFrame
    position_table: pd.DataFrame
    fallback: bool
    categories_applicable: bool


def need_board_view(board: pd.DataFrame, draft_state: DraftState, league_cfg: dict) -> NeedBoardView:
    """Cross the undrafted board against the user's drafted roster and ``league_cfg`` (roster
    shape) to produce a need-adjusted ranking (ADR 0026). Pure function: ``league_cfg`` must be
    passed in (typically ``src.value.league.load_league()``), never loaded implicitly, so this
    stays testable without touching disk."""
    # The full board (not just the undrafted slice) is needed here: computing how much of each
    # position's capacity the user has already filled requires looking up the *drafted* players'
    # positions too. need_score/need_adj_vorp are then read back out only for the undrafted rows,
    # filtered by the board's original index (best_available() would reset it, breaking the
    # alignment with result.frame).
    result = compute_need_board(board, draft_state.my_ids, league_cfg)
    avail = board[~board["player_id"].isin(draft_state.drafted_ids)]
    table = avail.copy()
    table["need_score"] = result.frame.loc[avail.index, "need_score"].to_numpy()
    table["need_adj_vorp"] = result.frame.loc[avail.index, "need_adj_vorp"].to_numpy()
    table = table.sort_values(["need_adj_vorp", "vorp", "player_id"],
                              ascending=[False, False, True]).reset_index(drop=True)
    table.insert(0, "need_rank", range(1, len(table) + 1))
    cols = [c for c in NEED_TABLE_COLUMNS if c in table.columns]
    return NeedBoardView(table=table[cols], position_table=position_need_table(result),
                         fallback=result.fallback, categories_applicable=result.categories_applicable)


# --------------------------------------------------------------------------- export / import

def state_to_dict(state: DraftState, meta: dict | None = None) -> dict:
    return {
        "meta": meta or {},
        "picks": [
            {"player_id": p.player_id, "name": p.name, "position": p.position,
             "drafted_by": p.drafted_by, "pick_no": p.pick_no}
            for p in state.picks
        ],
    }


def state_to_json(state: DraftState, meta: dict | None = None) -> str:
    return json.dumps(state_to_dict(state, meta), indent=2)


def dict_to_state(data: dict) -> tuple[DraftState, dict]:
    """Inverse of ``state_to_dict``. Raises ``DraftError`` on malformed input (missing keys,
    duplicate player ids, bad ``drafted_by``) rather than constructing an inconsistent state."""
    try:
        raw_picks = data["picks"]
    except (KeyError, TypeError) as e:
        raise DraftError(f"malformed draft state: {e}") from e
    picks: list[Pick] = []
    seen: set[int] = set()
    for i, row in enumerate(raw_picks):
        try:
            pid = int(row["player_id"])
            drafted_by = row["drafted_by"]
            pick_no = int(row.get("pick_no", i + 1))
        except (KeyError, TypeError, ValueError) as e:
            raise DraftError(f"malformed pick at index {i}: {e}") from e
        if drafted_by not in DRAFTED_BY_VALUES:
            raise DraftError(f"pick at index {i} has invalid drafted_by {drafted_by!r}")
        if pid in seen:
            raise DraftError(f"duplicate player_id {pid} in imported draft state")
        seen.add(pid)
        picks.append(Pick(player_id=pid, name=row.get("name", ""), position=row.get("position"),
                          drafted_by=drafted_by, pick_no=pick_no))
    picks.sort(key=lambda p: p.pick_no)
    return DraftState(picks=tuple(picks)), dict(data.get("meta") or {})


def json_to_state(text: str) -> tuple[DraftState, dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DraftError(f"not valid JSON: {e}") from e
    return dict_to_state(data)


def state_to_dataframe(state: DraftState) -> pd.DataFrame:
    """The drafted list as a DataFrame (pick order), for display and CSV export."""
    if not state.picks:
        return pd.DataFrame(columns=["pick_no", "player_id", "name", "position", "drafted_by"])
    rows = [{"pick_no": p.pick_no, "player_id": p.player_id, "name": p.name,
            "position": p.position, "drafted_by": p.drafted_by} for p in state.picks]
    return pd.DataFrame(rows).sort_values("pick_no").reset_index(drop=True)


# Shown under the board tables (ADR 0023): the rank is total fantasy points, never FPPG alone.
RANK_CAPTION = ("Rank is by projected TOTAL fantasy points above replacement (VORP): FPPG x games. "
                "A high FPPG player who misses games ranks lower.")

RETURN_TABLE_COLUMNS = ["rank", "name", "position", "return_block_pct", "return_tail", "proj_gp", "proj_fppg", "proj_total_fp",
                        "return_gp_upside_adv", "return_fp_upside_adv", "vorp", "adp"]


def return_flag_table(board: pd.DataFrame, drafted_ids: Iterable[int] = ()) -> pd.DataFrame:
    """The undrafted players carrying the advisory return flag, in board order, with the columns to show. An empty frame when the board has no return flag."""
    if "return_flag" not in board.columns:
        return pd.DataFrame()
    r = best_available(board[board["return_flag"].fillna("") != ""], drafted_ids)
    return r[[c for c in RETURN_TABLE_COLUMNS if c in r.columns]]


LM_TABLE_COLUMNS = ["rank", "name", "position", "lm_iso_n", "lm_rest_n", "proj_gp", "proj_fppg", "proj_total_fp",
                    "lm_gp_risk_adv", "lm_fp_risk_adv", "vorp", "adp"]


def load_flag_table(board: pd.DataFrame, drafted_ids: Iterable[int] = ()) -> pd.DataFrame:
    """The undrafted players carrying the advisory short-absence flag (ADR 0024), in board order. Empty when the board has no such flag."""
    if "lm_flag" not in board.columns:
        return pd.DataFrame()
    r = best_available(board[board["lm_flag"].fillna("") != ""], drafted_ids)
    return r[[c for c in LM_TABLE_COLUMNS if c in r.columns]]


# --------------------------------------------------------------------------- external rankings disagreement (ADR 0029)

RANKINGS_DISAGREEMENT_TABLE_COLUMNS = [
    "rank", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp", "vorp",
    "yahoo_rank", "fantasypros_rank", "rank_delta_yahoo", "rank_delta_fantasypros", "max_abs_rank_delta",
    "rankings_disagreement_direction", "rankings_disagreement_text", "adp",
]


def rankings_disagreement_table(board: pd.DataFrame, comparison: pd.DataFrame,
                                drafted_ids: Iterable[int] = ()) -> pd.DataFrame:
    """Undrafted board players flagged by ``src.value.rankings_compare.flag_disagreements`` (ADR
    0029: our board rank vs. the external Yahoo/FantasyPros consensus differs by
    ``DISAGREEMENT_THRESHOLD`` rank spots or more, either direction), in board order.

    ``comparison`` is the frame the "External rankings" page builds in this session
    (``st.session_state.rankings_cmp`` -- ``src.value.rankings_compare.build_comparison``, which
    already includes the disagreement columns). An empty/missing comparison (nothing loaded on
    that page this session yet) gives an empty table, not an error -- this is advisory and
    session-scoped like the "Best available by need" tab (ADR 0026), never baked into the stored
    board or its projection/rank columns.
    """
    if comparison is None or comparison.empty or "rankings_disagreement" not in comparison.columns:
        return pd.DataFrame(columns=RANKINGS_DISAGREEMENT_TABLE_COLUMNS)
    flagged = comparison[comparison["rankings_disagreement"]]
    if flagged.empty:
        return pd.DataFrame(columns=RANKINGS_DISAGREEMENT_TABLE_COLUMNS)
    side = flagged.drop_duplicates("player_id").set_index("player_id")[
        ["yahoo_rank", "fantasypros_rank", "rank_delta_yahoo", "rank_delta_fantasypros",
         "max_abs_rank_delta", "rankings_disagreement_direction", "rankings_disagreement_text"]
    ]
    merged = board[board["player_id"].isin(side.index)].join(side, on="player_id")
    merged = best_available(merged, drafted_ids)
    merged = merged.sort_values("max_abs_rank_delta", ascending=False, kind="mergesort").reset_index(drop=True)
    return merged[[c for c in RANKINGS_DISAGREEMENT_TABLE_COLUMNS if c in merged.columns]]


BOARD_COLUMNS = ["rank", "name", "position", "tier", "proj_fppg", "proj_gp", "proj_total_fp",
                 "vorp", "fppg_p10", "fppg_p50", "fppg_p90"]


def display_columns(board: pd.DataFrame) -> list[str]:
    """The board's display columns, with ADP (and the model-vs-market gap) after the projection when the board has it."""
    extra = [c for c in ("adp", "adp_gap", "blend_rank") if c in board.columns]     # ADR 0032: ADP-anchored ordering beside the model's
    # ADR 0031 / 0033: chance of playing at all and the season-total band, only when the model emitted them
    band = [c for c in ("proj_p_appear", "proj_total_fp_p10", "proj_total_fp_p90") if c in board.columns]
    # ADR 0016: how the projection was made and the draft-day risk overlay, only when the board carries them
    tail = [c for c in ("projection_class", "p_play", "risk_level", "risk_flags") if c in board.columns]
    # ADR 0019: display-only, unvalidated contract flag (blank = not known to be, never known not to be)
    tail += [c for c in ("contract_flag",) if c in board.columns]
    # ADR 0023: advisory return-from-absence flag and the games played since (never a projection); placed after vorp below
    ret = [c for c in ("return_flag", "return_tail") if c in board.columns]   # right after vorp, visible without scrolling
    # ADR 0024: advisory short-absence flag (cause unknown; never a projection)
    ret += [c for c in ("lm_flag",) if c in board.columns]
    return BOARD_COLUMNS[:7] + extra + BOARD_COLUMNS[7:8] + ret + BOARD_COLUMNS[8:] + band + tail
