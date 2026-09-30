"""Positional/roster-need scoring for the "best available by need" board view (ADR 0026).

Crosses the ranked board (VORP, from ``src.value.vorp``) against the user's own drafted roster
and the roster shape in ``config/league.yaml`` (source of truth for what "need" means
structurally: starting slots, flex slots, bench). The result is an additive nudge on top of VORP,
never a replacement for it — the raw board is always still available unmodified.

No Streamlit import here on purpose — this is a plain, unit-tested module (``tests/value/test_need.py``),
following the pattern already used by ``src.value.vorp`` and the other advisory overlays
(``src.value.return_flag``, ``src.value.load_flag``).

**Model, in one paragraph.** Every ESPN starting slot (a specific position, a flex ``G``/``F``, or
``UTIL``) is spread as a fractional "capacity" across the specific positions it can be filled by
(a flex ``G`` starter counts as half a PG slot and half an SG slot; ``UTIL`` splits evenly across
all five; bench slots split the same way, since any drafted player can occupy one). A specific
position's need score is how much of that capacity remains unfilled by the user's own drafted
players, clamped to ``[0, 1]`` (0 = fully covered or the league carries no such slot at all, 1 =
none of it is covered yet). A player's own need score is the *best* (highest) need score among the
specific positions he is eligible at — he only has to fill one of them. The board's ``vorp`` gets a
flat, documented bonus (``DEFAULT_NEED_BONUS`` fantasy points) scaled by that need score.

**Why this stays simple on purpose.** True optimal-lineup need (which slot would a given player
*actually* occupy, given every other drafted player and every remaining flex choice) is an
assignment problem; solving it exactly would fold roster construction into the ranking model in a
way the rest of this app deliberately avoids doing without a validated backtest (see ADR 0016's
positional-scarcity toggle for the same caution). This heuristic is the same kind of "count
eligible positions, don't solve an assignment" approximation ``src.value.positions`` already makes
for lineup eligibility, applied to needs instead of legality.

**Fallback for an empty roster.** With no drafted players, every specific position's need score is
exactly 1.0 (nothing filled anywhere), so the bonus becomes the *same* constant added to every
player's ``vorp`` (unless a player's position is unparseable — see ``src.value.positions.is_known``
— which the standard eligibility rule already treats as "no invented positions", so such a player
gets no bonus). A constant additive shift never changes an ordering, so the need-adjusted board is
then rank-identical to the plain VORP board — which is what "sensibly fall back to plain
best-available-by-VORP" means here — and ``NeedResult.fallback`` is set so the app can also say so
explicitly rather than relying on that being obvious.

**Categories.** ``config/league.yaml`` is currently ``format: h2h_points`` (points league):
category needs (e.g. punting a stat category) do not apply, and ``is_categories_league`` is the
single place that would gate a category-need computation if the league ever became a
categories/roto format. No category-need scoring is implemented (there is nothing to validate it
against without a categories league) — see ADR 0026's "explicitly scoped out".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.value.positions import ESPN_POSITIONS, FLEX_SLOTS, UTIL, eligible_positions

#: Flat VORP-points bonus applied at need_score == 1.0 (fully open position), scaled linearly down
#: to 0 at need_score == 0.0. Not empirically calibrated (no backtest validates a specific size —
#: see ADR 0026's limitations); chosen to be noticeable next to a typical late-round VORP gap
#: without being able to vault a clearly-better player at a position that is not needed at all.
DEFAULT_NEED_BONUS = 40.0

#: ``player_id`` sits right after ``rank`` (matching ``src.value.board.BASE_COLUMNS``'s own
#: ordering) so the table can be joined back onto the live board programmatically -- e.g. looked
#: up on another tab, or exported and re-merged -- instead of only by ``name`` (fragile: duplicate
#: names, accented characters, and the like are exactly what ``player_id``/``id_map`` exist to
#: avoid elsewhere in this codebase, see ADR 0007/0028). It stays display-hidden on the "Best
#: available by need" tab itself (``draft_board.py``'s ``column_config`` for this table sets
#: ``player_id: None`` -- present in the data, hidden from the rendered grid) rather than dropped
#: from the columns list the way the main board's ``display_columns()`` omits ``player_id`` from
#: its own output: unlike the main board, nothing here builds a separate "display" subset of
#: ``NEED_TABLE_COLUMNS`` -- ``view.table`` is both the data and the display frame, so
#: column_config is the only place left to hide it from view.
NEED_TABLE_COLUMNS = ["need_rank", "rank", "player_id", "name", "position", "proj_fppg", "proj_gp",
                     "proj_total_fp", "vorp", "need_score", "need_adj_vorp", "tier", "adp"]
#: Named ``need_position`` (not ``position``): this table's rows are the five specific ESPN
#: positions themselves, a different meaning from the board's coarse dataset ``position`` column
#: (which the existing glossary entry already documents), so it gets its own name and entry rather
#: than reusing a tooltip that would then be wrong half the time.
POSITION_NEED_COLUMNS = ["need_position", "need_capacity", "need_filled", "need_score"]


def is_categories_league(league_cfg: Mapping) -> bool:
    """True when ``league_cfg`` (as loaded by ``src.value.league.load_league``) describes a
    categories/roto format rather than points. The current ``config/league.yaml`` is
    ``h2h_points``, so this is ``False`` for this league; the check exists so a future switch to a
    categories format has a single place to gate category-need scoring (not implemented — see the
    module docstring)."""
    fmt = str((league_cfg or {}).get("league", {}).get("format", ""))
    return "categor" in fmt.lower() or "roto" in fmt.lower()


def position_capacity(league_cfg: Mapping) -> dict[str, float]:
    """Fractional roster "capacity" at each specific ESPN position: the specific starting slot,
    plus an even share of every flex slot (``G``, ``F``) and ``UTIL`` that could be filled there,
    plus an even share of the bench (any position can occupy a bench slot). IR is excluded (it is
    not an active lineup slot)."""
    roster = league_cfg["league"]["roster"]
    starters = roster.get("starters", {})
    bench = float(roster.get("bench", 0) or 0)
    cap = {p: float(starters.get(p, 0) or 0) for p in ESPN_POSITIONS}
    for flex, members in FLEX_SLOTS.items():
        n = float(starters.get(flex, 0) or 0)
        if n:
            share = n / len(members)
            for m in members:
                cap[m] += share
    util = float(starters.get(UTIL, 0) or 0)
    if util:
        share = util / len(ESPN_POSITIONS)
        for p in ESPN_POSITIONS:
            cap[p] += share
    if bench:
        share = bench / len(ESPN_POSITIONS)
        for p in ESPN_POSITIONS:
            cap[p] += share
    return cap


def position_fill_counts(board: pd.DataFrame, my_ids: Iterable[int]) -> dict[str, int]:
    """How many of the user's own drafted players (rows of ``board`` whose ``player_id`` is in
    ``my_ids``) are eligible at each specific ESPN position. A multi-position player counts toward
    each position he is eligible at (a demand-side approximation, not an exact lineup
    assignment — see the module docstring)."""
    ids = set(my_ids)
    if not ids:
        return {p: 0 for p in ESPN_POSITIONS}
    mine = board[board["player_id"].isin(ids)]
    return {p: int(mine["position"].map(lambda x, p=p: p in eligible_positions(x)).sum())
           for p in ESPN_POSITIONS}


def position_need_scores(capacity: Mapping[str, float], filled: Mapping[str, float]) -> dict[str, float]:
    """``clamp((capacity - filled) / capacity, 0, 1)`` per specific position; 0 when the league
    carries no capacity there at all (division by zero would otherwise be undefined) or the
    position is already fully covered; 1 when none of the capacity is covered yet."""
    out: dict[str, float] = {}
    for p in ESPN_POSITIONS:
        cap = float(capacity.get(p, 0.0) or 0.0)
        fl = float(filled.get(p, 0.0) or 0.0)
        out[p] = 0.0 if cap <= 0 else float(np.clip((cap - fl) / cap, 0.0, 1.0))
    return out


def player_need_scores(positions: pd.Series, position_need: Mapping[str, float]) -> pd.Series:
    """Each row's own need score: the best (highest) need score among the specific ESPN positions
    it is eligible at, or 0.0 for a player whose position string is unparseable (never invented,
    same rule as ``src.value.positions.is_known``)."""
    def _score(pos: object) -> float:
        elig = eligible_positions(pos)
        if not elig:
            return 0.0
        return max(position_need.get(p, 0.0) for p in elig)
    return positions.map(_score).astype(float)


@dataclass(frozen=True)
class NeedResult:
    """The need computation for one board (already filtered to undrafted players by the caller)."""

    frame: pd.DataFrame            # aligned to the input board's index: need_score, need_adj_vorp
    capacity: dict[str, float]     # specific position -> fractional roster capacity
    filled: dict[str, int]         # specific position -> how many of the user's players fill it
    position_need: dict[str, float]  # specific position -> need score
    fallback: bool                 # True when the user has drafted no one yet (see module docstring)
    categories_applicable: bool    # True if config/league.yaml is a categories/roto format


def compute_need_board(board: pd.DataFrame, my_ids: Iterable[int], league_cfg: Mapping, *,
                       need_bonus: float = DEFAULT_NEED_BONUS) -> NeedResult:
    """The core, pure need computation. ``board`` needs ``position`` and ``vorp`` columns.
    ``my_ids`` is the set of ``player_id`` the user has already drafted (empty -> the documented
    fallback, see the module docstring). Returns a :class:`NeedResult` with a ``frame`` carrying
    ``need_score`` (0..1) and ``need_adj_vorp`` (``vorp`` plus the scaled bonus) aligned to
    ``board``'s index."""
    ids = set(my_ids)
    capacity = position_capacity(league_cfg)
    filled = position_fill_counts(board, ids)
    position_need = position_need_scores(capacity, filled)
    need_score = player_need_scores(board["position"], position_need) if len(board) else pd.Series(dtype=float)
    need_adj_vorp = board["vorp"].astype(float) + need_bonus * need_score
    frame = pd.DataFrame({"need_score": need_score, "need_adj_vorp": need_adj_vorp}, index=board.index)
    return NeedResult(
        frame=frame,
        capacity=capacity,
        filled=filled,
        position_need=position_need,
        fallback=not ids,
        categories_applicable=is_categories_league(league_cfg),
    )


def position_need_table(result: NeedResult) -> pd.DataFrame:
    """A small per-position table (``need_capacity``, ``need_filled``, ``need_score``) for display
    alongside the need-adjusted board, so the score is never a black box."""
    rows = [{"need_position": p, "need_capacity": round(result.capacity.get(p, 0.0), 2),
            "need_filled": result.filled.get(p, 0), "need_score": round(result.position_need.get(p, 0.0), 2)}
           for p in ESPN_POSITIONS]
    return pd.DataFrame(rows, columns=POSITION_NEED_COLUMNS)
