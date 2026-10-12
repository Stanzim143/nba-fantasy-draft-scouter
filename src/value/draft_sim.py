"""Draft-availability simulator: "will he still be there at my next pick?" (ADR 0040).

Given the undrafted pool, the overall pick about to be made, the user's snake slot, and the league
size and rounds, estimate for each available player the probability that the opponents picking
before the user's next turn (and the turn after) leave him on the board.

Model, in one paragraph: each simulated draft gives every available player a score
``market_adp + sd(market_adp) * z`` with ``z`` standard normal, and the opponents take the
lowest-scoring players in order. A player survives to a target pick when fewer than ``k`` players
score below him, where ``k`` is the number of *opponent* picks between now and that target. The
user's own intervening picks are not modelled as taking anyone (the question is "if I pass on him").
``sd`` grows with ADP (``SD_MIN + SD_FRAC * adp``, capped at ``SD_MAX``), and in a share ``SKIP`` of drafts
this league passes on a player entirely. Those values are fitted on four of this league's earlier drafts
(``src.value.draft_sim_calibrate``, ADR 0040); the fit helps only modestly and stays overconfident at the
extremes, so the table also shows the range across plausible noise settings (``BAND_PARAMS``).

Advisory only: nothing here changes a projection, VORP or the board rank. Plain functions, no
Streamlit (``tests/value/test_draft_sim.py``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Draft-position noise: sd = min(SD_MAX, SD_MIN + SD_FRAC * adp), in picks, fitted on the league's past drafts
#: (ADR 0040): about 3.5 picks at ADP 3, 17 at ADP 30, 25 (the cap) from ADP 46.
SD_MIN = 2.0        # fitted on the league's 2021-22 to 2024-25 drafts (ADR 0040)
SD_FRAC = 0.5
SD_MAX = 25.0
SKIP = 0.2          # share of drafts in which this league passes on a player entirely (fitted, at the grid edge)
SKIP_FROM = 0.0     # ... applied only to players whose market ADP is above this (ADR 0040 addendum; set by the fit)
#: (sd_min, sd_frac, sd_max, skip) settings spanning the plausible range; p_next_lo / p_next_hi are the min / max.
BAND_PARAMS = ((1.0, 0.2, 25.0, 0.0), (2.0, 0.3, 15.0, 0.2), (6.0, 0.5, 40.0, 0.2))
SKIP_OFFSET = 1e4   # a skipped player's score is pushed past every real pick
DEFAULT_SIMS = 2000
DEFAULT_SEED = 0

#: Columns the availability table adds next to the board's own identity/value columns.
AVAIL_COLUMNS = ["market_adp", "p_avail_next", "p_next_lo", "p_next_hi", "p_avail_after", "vorp_at_risk"]
TABLE_COLUMNS = ["rank", "name", "position", "vorp", "adp", *AVAIL_COLUMNS]


# --------------------------------------------------------------------------- snake pick math

def snake_pick(round_no: int, slot: int, teams: int) -> int:
    """Overall (1-based) pick number of ``slot`` (1-based) in ``round_no`` (1-based) of a snake."""
    if teams < 1 or not 1 <= slot <= teams or round_no < 1:
        raise ValueError(f"bad snake position: round {round_no}, slot {slot}, teams {teams}")
    base = (round_no - 1) * teams
    return base + (slot if round_no % 2 == 1 else teams - slot + 1)


def my_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    """Every overall pick number held by ``slot`` in a ``teams``-team, ``rounds``-round snake."""
    return [snake_pick(r, slot, teams) for r in range(1, rounds + 1)]


def slot_of_pick(overall: int, teams: int) -> int:
    """The snake slot that makes overall pick ``overall`` (inverse of :func:`snake_pick`)."""
    if overall < 1 or teams < 1:
        raise ValueError(f"bad pick {overall} for {teams} teams")
    round_no, idx = divmod(overall - 1, teams)
    return idx + 1 if round_no % 2 == 0 else teams - idx


def slot_from_pick_order(pick_order: Sequence[int] | None, team_id: int | None) -> int | None:
    """Slot of ``team_id`` in ESPN's ``draftSettings.pickOrder`` (a list of team ids), else ``None``."""
    if not pick_order or team_id is None:
        return None
    try:
        return [int(t) for t in pick_order].index(int(team_id)) + 1
    except ValueError:
        return None


def load_pick_order(league_id: int | None, season: str, data_root: Path | None = None) -> list[int] | None:
    """``draftSettings.pickOrder`` from the saved ESPN league snapshot (``python -m src.ingest.espn_league``),
    or ``None`` when there is no league id, no snapshot, or no order in it. Never raises."""
    if not league_id:
        return None
    try:
        from src.contracts import data_dir, season_start

        path = (data_root or data_dir()) / "processed" / "espn_league" / f"{league_id}_{season_start(season) + 1}.json"
        order = json.loads(path.read_text(encoding="utf-8"))["settings"]["draft"]["pick_order"]
        return [int(t) for t in order] if order else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def resolve_slot(teams: int, *, manual: int | None = None, pick_order: Sequence[int] | None = None,
                 team_id: int | None = None, my_pick_nos: Iterable[int] = ()) -> tuple[int | None, str]:
    """The user's draft slot and where it came from, in priority order: a manual entry, the ESPN
    league's ``pickOrder`` with the user's team id, or the overall number of his first recorded pick
    (valid only when every pick so far was recorded in order). ``(None, reason)`` when unknown."""
    if manual and 1 <= int(manual) <= teams:
        return int(manual), "entered by hand"
    s = slot_from_pick_order(pick_order, team_id)
    if s is not None and s <= teams:
        return s, f"ESPN league draft order (team id {team_id})"
    nos = sorted(int(p) for p in my_pick_nos)
    if nos:
        return slot_of_pick(nos[0], teams), f"your first recorded pick (#{nos[0]})"
    return None, "unknown: enter your draft slot, or set ESPN_LEAGUE_ID / ESPN_TEAM_ID and sync the league"


def upcoming_turns(current_pick: int, my_picks: Sequence[int]) -> tuple[bool, list[int]]:
    """``(on_the_clock, targets)``: the user's next two turns that a pass *now* would wait for.

    On the clock (``current_pick`` is the user's), the targets are his following two picks; otherwise
    they are his next two picks from ``current_pick`` on. Fewer than two when the draft runs out."""
    later = sorted(p for p in my_picks if p >= current_pick)
    on_clock = bool(later) and later[0] == current_pick
    return on_clock, (later[1:3] if on_clock else later[:2])


def market_picks_before(target: int, current_pick: int, my_picks: Iterable[int]) -> int:
    """How many picks in ``[current_pick, target)`` can take a player the user is passing on.

    The user's own picks take someone else (the best remaining by market order), so a pick of his still
    removes a player ahead of the target unless every pick after it is also his: the target survives iff
    fewer players score below him than the picks up to and including the last opponent pick."""
    mine = set(my_picks)
    opp = [p for p in range(current_pick, target) if p not in mine]
    return (opp[-1] - current_pick + 1) if opp else 0


def opponent_picks_before(target: int, current_pick: int, my_picks: Iterable[int]) -> int:
    """Number of opponent picks in ``[current_pick, target)``; the user's own picks are excluded."""
    mine = set(my_picks)
    return sum(1 for p in range(current_pick, target) if p not in mine)


# --------------------------------------------------------------------------- market order

def market_adp(pool: pd.DataFrame, *, current_pick: int = 1) -> tuple[np.ndarray, str]:
    """The ADP the simulated opponents draft by, one value per ``pool`` row, and a note on its source.

    * Rows with an ADP use it, ties (ESPN compresses the tail near 140) broken by board rank.
    * Rows without one are assumed to go late: after every known ADP (and after ``current_pick``),
      in board-rank order.
    * With no ADP at all (synthetic demo, no ingested ADP), the board rank is the proxy, which says
      how fast *our* order would empty, not the market's.
    """
    n = len(pool)
    rank = (pd.to_numeric(pool["rank"], errors="coerce").to_numpy(float) if "rank" in pool.columns
            else np.arange(1, n + 1, dtype=float))
    rank = np.where(np.isnan(rank), np.nanmax(rank, initial=0) + 1, rank)
    order = rank.argsort(kind="stable").argsort(kind="stable")  # 0..n-1 by board rank
    tie = order * 1e-4
    adp = pd.to_numeric(pool["adp"], errors="coerce").to_numpy(float) if "adp" in pool.columns else np.full(n, np.nan)
    known = ~np.isnan(adp)
    if not known.any():
        return rank.astype(float) + tie, "no ADP on this board: opponents are assumed to draft by our board rank"
    out = adp + tie
    if (~known).any():
        start = max(float(np.nanmax(adp)), float(current_pick))
        late_order = order[~known].argsort(kind="stable").argsort(kind="stable")
        out[~known] = start + 1.0 + late_order
    note = "opponents draft by ESPN ADP with noise"
    if (~known).any():
        note += f"; {int((~known).sum())} players with no ADP are assumed to go after everyone with one"
    return out, note


def adp_sd(adp: np.ndarray, *, sd_min: float = SD_MIN, sd_frac: float = SD_FRAC, sd_max: float = SD_MAX) -> np.ndarray:
    return np.minimum(sd_max, sd_min + sd_frac * np.maximum(np.asarray(adp, float), 0.0))


# --------------------------------------------------------------------------- simulation

def simulate_order(market: np.ndarray, *, n_sims: int = DEFAULT_SIMS, seed: int = DEFAULT_SEED, sd_min: float = SD_MIN,
                   sd_frac: float = SD_FRAC, sd_max: float = SD_MAX, df: float | None = None,
                   skip: float = SKIP, skip_from: float = SKIP_FROM) -> np.ndarray:
    """The simulated drafts themselves: ``order[s, r]`` is the index of the player taken ``r``-th (0-based)
    in draw ``s``, shape ``(n_sims, len(market))``, int32. Shared by :func:`survival_probabilities` and the
    take-now-or-wait panel (``src.value.pick_decision``, ADR 0044) so both read the same draws."""
    market = np.asarray(market, float)
    n = market.size
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, (n_sims, n)) if df else rng.standard_normal((n_sims, n))  # df: Student-t noise
    scores = market[None, :] + adp_sd(market, sd_min=sd_min, sd_frac=sd_frac, sd_max=sd_max)[None, :] * z
    if skip > 0:  # this league passes on him entirely in a share `skip` of drafts (injury news, points-league fit)
        hit = rng.random((n_sims, n)) < skip          # always drawn, so the rest of the stream does not depend on skip_from
        scores = scores + SKIP_OFFSET * (hit & (market > skip_from)[None, :])   # stars are never passed on entirely
    return scores.argsort(axis=1).astype(np.int32)


def survival_from_order(order: np.ndarray, opponent_picks: Sequence[int]) -> np.ndarray:
    """``P(player i survives k picks)`` per ``k`` from :func:`simulate_order`'s draws."""
    n_sims, n = order.shape
    ks = [int(k) for k in opponent_picks]
    if n == 0:
        return np.zeros((len(ks), 0))
    pos = np.empty((n_sims, n), dtype=np.int32)  # pos[s, i] = how many players score below i in sim s
    np.put_along_axis(pos, order, np.arange(n, dtype=np.int32)[None, :], axis=1)
    return np.vstack([(pos >= k).mean(axis=0) for k in ks]) if ks else np.zeros((0, n))


def survival_probabilities(market: np.ndarray, opponent_picks: Sequence[int], *, n_sims: int = DEFAULT_SIMS,
                           seed: int = DEFAULT_SEED, sd_min: float = SD_MIN, sd_frac: float = SD_FRAC,
                           sd_max: float = SD_MAX, df: float | None = None, skip: float = SKIP,
                           skip_from: float = SKIP_FROM) -> np.ndarray:
    """``P(player i is still available after k opponent picks)`` for each ``k`` in ``opponent_picks``.

    Returns an array of shape ``(len(opponent_picks), len(market))``. The same simulated drafts serve
    every ``k``, so the probabilities are non-increasing in ``k`` by construction."""
    market = np.asarray(market, float)
    if market.size == 0:
        return np.zeros((len(list(opponent_picks)), 0))
    order = simulate_order(market, n_sims=n_sims, seed=seed, sd_min=sd_min, sd_frac=sd_frac, sd_max=sd_max,
                           df=df, skip=skip, skip_from=skip_from)
    return survival_from_order(order, opponent_picks)


@dataclass
class AvailabilityResult:
    table: pd.DataFrame                 # TABLE_COLUMNS (+ player_id), board order
    current_pick: int
    on_the_clock: bool
    targets: list[int]                  # overall pick numbers of the next turn and the one after
    opponent_picks: list[int]           # opponent picks before each target
    market_picks: list[int] = field(default_factory=list)  # picks simulated before each target (incl. the user's)
    notes: list[str] = field(default_factory=list)
    order: np.ndarray | None = None     # the draws (simulate_order), columns = rows of ``table``; reused by ADR 0044
    band_orders: list[np.ndarray] = field(default_factory=list)  # draws at each BAND_PARAMS setting, same seed


def availability_table(board: pd.DataFrame, drafted_ids: Iterable[int], *, current_pick: int, slot: int,
                       teams: int, rounds: int, n_sims: int = DEFAULT_SIMS, seed: int = DEFAULT_SEED) -> AvailabilityResult:
    """P(available at my next turn / the turn after) for every undrafted board player. Never mutates ``board``."""
    drafted = set(drafted_ids)
    pool = board[~board["player_id"].isin(drafted)].copy() if "player_id" in board.columns else board.copy()
    if "rank" in pool.columns:
        pool = pool.sort_values("rank", kind="stable")
    pool = pool.reset_index(drop=True)
    total = teams * rounds
    mine = my_pick_numbers(slot, teams, rounds)
    on_clock, targets = upcoming_turns(current_pick, mine)
    ks = [opponent_picks_before(t, current_pick, mine) for t in targets]
    ms = [market_picks_before(t, current_pick, mine) for t in targets]
    market, note = market_adp(pool, current_pick=current_pick)
    notes = [note]
    if current_pick > total:
        notes.append(f"the draft is over ({total} picks)")
    elif not targets:
        notes.append("you have no picks left after this one")
    order = simulate_order(market, n_sims=n_sims, seed=seed)
    probs = survival_from_order(order, ms)
    band_orders = [simulate_order(market, n_sims=n_sims, seed=seed, sd_min=a, sd_frac=b, sd_max=c, skip=d)
                   for a, b, c, d in BAND_PARAMS] if ms else []
    band = [survival_from_order(o, ms[:1]) for o in band_orders]

    table = pd.DataFrame({c: pool[c] if c in pool.columns else np.nan for c in ("player_id", "rank", "name", "position", "vorp", "adp")})
    table["market_adp"] = np.round(market, 1)
    table["p_avail_next"] = probs[0] if len(ks) > 0 else np.nan
    table["p_next_lo"] = np.min([probs[:1], *band], axis=0)[0] if band else np.nan
    table["p_next_hi"] = np.max([probs[:1], *band], axis=0)[0] if band else np.nan
    table["p_avail_after"] = probs[1] if len(ks) > 1 else np.nan
    vorp = pd.to_numeric(table["vorp"], errors="coerce").clip(lower=0)
    table["vorp_at_risk"] = (vorp * (1 - table["p_avail_next"])).round(1)
    return AvailabilityResult(table=table, current_pick=current_pick, on_the_clock=on_clock, targets=targets,
                              opponent_picks=ks, market_picks=ms, notes=notes, order=order, band_orders=band_orders)
