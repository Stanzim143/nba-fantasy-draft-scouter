"""Replacement level, derived from the league config (never hard-coded to a team count).

Definitions
-----------
*Rostered pool.* The players a league of ``teams`` teams actually holds and uses::

    R = teams * (starter_slots + w * bench_slots)

``starter_slots`` is the sum over ``roster.starters`` (10 for PG SG SF PF C G F UTIL x3);
``bench_slots`` is ``roster.bench`` (IR does not count). Bench spots are worth less than starting
spots because a bench player only produces when a starter is unavailable, so the bench is
weighted by ``w`` in [0, 1]. ``w`` is derived from the projections themselves: on a typical day
``starters * (1 - availability)`` starting slots are vacated, and the bench covers up to
``bench_slots`` of them::

    w = min(1, starter_slots * (1 - availability) / bench_slots)

with ``availability = mean(proj_gp) / season_games`` over the top ``teams * starter_slots``
players. (Pass ``bench_weight`` to override.)

*Replacement value.* The best player **outside** the rostered pool, i.e. the (R+1)-th best in the
ranking (linear interpolation when R is fractional). That is who you get from waivers for free.
If the projection frame holds fewer than R players the replacement level is its minimum.

Total-based VORP subtracts the replacement player's projected season total; the per-game variant
subtracts the same replacement player's FPPG (same player identity, expressed per game).

Positional scarcity
-------------------
ESPN's flexible slots (G, F, 3 x UTIL) mostly erase positional scarcity, but that is checked
rather than assumed: ``positional_replacement`` fills the league's slots greedily from the best
player down (specific positions, then G/F, then UTIL, then the weighted bench) and reports, for each
of PG/SG/SF/PF/C, the best *unrostered* eligible player. ``scarcity_index`` is the spread of those
five levels relative to a typical starter's VORP; ``>= scarcity_threshold`` counts as material.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.value.league import load_league
from src.value.positions import ESPN_POSITIONS, FLEX_SLOTS, UTIL, eligible_positions

DEFAULT_SCARCITY_THRESHOLD = 0.10


@dataclass(frozen=True)
class LeagueShape:
    teams: int
    starter_slots: dict[str, int]
    bench: int

    @property
    def n_starters(self) -> int:
        return int(sum(self.starter_slots.values()))

    @property
    def roster_size(self) -> int:
        return self.n_starters + self.bench


def league_shape(cfg: dict | None = None, *, teams: int | None = None) -> LeagueShape:
    """Read team count and slots from the league config (``teams`` overrides; the config value is
    13, confirmed live against the real ESPN league; see ADR 0008)."""
    cfg = cfg or load_league()
    lg = cfg["league"]
    n = int(teams if teams is not None else lg["teams"])
    if n < 1:
        raise ValueError("teams must be >= 1")
    slots = {k: int(v) for k, v in lg["roster"]["starters"].items()}
    return LeagueShape(teams=n, starter_slots=slots, bench=int(lg["roster"].get("bench", 0)))


def derive_bench_weight(shape: LeagueShape, availability: float) -> float:
    """Share of bench spots that behave like starting spots, from average availability (see module doc)."""
    if shape.bench <= 0:
        return 0.0
    vacated = shape.n_starters * (1.0 - float(np.clip(availability, 0.0, 1.0)))
    return float(np.clip(vacated / shape.bench, 0.0, 1.0))


def replacement_rank(shape: LeagueShape, bench_weight: float) -> float:
    """R = teams * (starter_slots + w * bench_slots)."""
    if not 0.0 <= bench_weight <= 1.0:
        raise ValueError("bench_weight must be within [0, 1]")
    return shape.teams * (shape.n_starters + bench_weight * shape.bench)


def value_at_rank(values, rank: float, *, by=None) -> float:
    """Value of the player at 0-based position ``rank`` in the descending order (interpolated).

    By default ``values`` are ranked by themselves. With ``by`` (an array aligned with ``values``) the
    players are ranked by ``by`` instead and the value is read off that ordering, i.e. the value of the
    *same player* who sits at ``rank`` in the ``by`` ranking (interpolated between neighbours). Players
    whose ``by`` or ``values`` entry is not finite are dropped.
    """
    v = np.asarray(values, float)
    if by is None:
        v = np.sort(v)[::-1]
        v = v[np.isfinite(v)]
    else:
        b = np.asarray(by, float)
        keep = np.isfinite(b) & np.isfinite(v)
        v = v[keep][np.argsort(-b[keep], kind="stable")]
    if len(v) == 0:
        return 0.0
    if rank >= len(v) - 1:
        return float(v[-1])
    lo = int(np.floor(rank))
    frac = rank - lo
    return float(v[lo] * (1 - frac) + v[lo + 1] * frac)


@dataclass(frozen=True)
class ReplacementLevel:
    shape: LeagueShape
    bench_weight: float
    availability: float | None
    rank: float
    total: float          # replacement player's projected season total
    per_game: float       # same player, per game

    def as_dict(self) -> dict:
        return {"teams": self.shape.teams, "starter_slots": self.shape.n_starters, "bench": self.shape.bench,
                "bench_weight": self.bench_weight, "availability": self.availability,
                "rank": self.rank, "total": self.total, "per_game": self.per_game}


def replacement_level(total_fp, fppg, gp, shape: LeagueShape, *, bench_weight: float | None = None,
                      season_games: float = 82.0) -> ReplacementLevel:
    """Replacement level for a set of projections (arrays aligned by player)."""
    total, fppg, gp = (np.asarray(a, float) for a in (total_fp, fppg, gp))
    order = np.argsort(-total, kind="stable")
    availability = None
    if bench_weight is None:
        top = order[: max(shape.teams * shape.n_starters, 1)]
        availability = float(np.mean(gp[top]) / season_games)
        bench_weight = derive_bench_weight(shape, availability)
    R = replacement_rank(shape, bench_weight)
    return ReplacementLevel(
        shape=shape, bench_weight=float(bench_weight), availability=availability, rank=float(R),
        total=value_at_rank(total, R), per_game=value_at_rank(fppg, R, by=total),
    )


# --------------------------------------------------------------------------- positional scarcity

def _fill_slots(values: np.ndarray, elig: list[tuple[str, ...]], shape: LeagueShape, n_bench: int) -> np.ndarray:
    """Greedy league-wide slot filling. Returns a boolean mask of players who end up rostered.

    Players are taken best first. Each goes to the eligible *specific* position with the most open
    league-wide slots, else an open flex slot (G/F), else UTIL, else the weighted bench, else the
    free-agent pool. Greedy is a heuristic (optimal assignment is a matching problem), but with
    interchangeable flex slots the difference is small and the result only feeds a diagnostic.
    """
    open_slots = {k: v * shape.teams for k, v in shape.starter_slots.items()}
    bench_left = n_bench
    rostered = np.zeros(len(values), dtype=bool)
    for i in np.argsort(-values, kind="stable"):
        specific = [p for p in elig[i] if open_slots.get(p, 0) > 0]
        if specific:
            open_slots[max(specific, key=lambda p: (open_slots[p], -ESPN_POSITIONS.index(p)))] -= 1
            rostered[i] = True
            continue
        placed = False
        for flex, members in FLEX_SLOTS.items():
            if open_slots.get(flex, 0) > 0 and any(m in elig[i] for m in members):
                open_slots[flex] -= 1
                placed = rostered[i] = True
                break
        if placed:
            continue
        if open_slots.get(UTIL, 0) > 0:
            open_slots[UTIL] -= 1
            rostered[i] = True
        elif bench_left > 0:
            bench_left -= 1
            rostered[i] = True
    return rostered


@dataclass(frozen=True)
class PositionalReport:
    levels: dict[str, float]        # position -> replacement total FP
    global_level: float
    spread: float                   # max - min of the five position levels
    scarcity_index: float           # spread / typical starter VORP
    material: bool
    threshold: float

    def as_dict(self) -> dict:
        return {"levels": dict(self.levels), "global_level": self.global_level, "spread": self.spread,
                "scarcity_index": self.scarcity_index, "material": self.material, "threshold": self.threshold}


def positional_replacement(total_fp, positions, shape: LeagueShape, *, bench_weight: float,
                           global_level: float, threshold: float = DEFAULT_SCARCITY_THRESHOLD) -> PositionalReport:
    """Best unrostered eligible player per ESPN position after a greedy league-wide fill."""
    values = np.asarray(total_fp, float)
    elig = [eligible_positions(p) for p in positions]
    n_bench = int(round(shape.teams * shape.bench * bench_weight))
    rostered = _fill_slots(values, elig, shape, n_bench)
    levels: dict[str, float] = {}
    floor = float(values.min()) if len(values) else 0.0
    for pos in ESPN_POSITIONS:
        pool = [values[i] for i in range(len(values)) if not rostered[i] and pos in elig[i]]
        levels[pos] = float(max(pool)) if pool else floor
    spread = max(levels.values()) - min(levels.values())
    starters = np.sort(values)[::-1][: shape.teams * shape.n_starters]
    typical = float(np.mean(starters) - global_level) if len(starters) else 0.0
    index = spread / typical if typical > 0 else 0.0
    return PositionalReport(levels, float(global_level), float(spread), float(index), bool(index >= threshold), threshold)


def positional_replacement_per_player(positions, report: PositionalReport, default: float) -> np.ndarray:
    """A player's own replacement level: the lowest level among his eligible positions.

    A multi-eligible player is slotted where replacement is weakest (largest gain). Players with an
    unknown position get ``default`` (the global level).
    """
    out = np.empty(len(positions))
    for i, p in enumerate(positions):
        elig = eligible_positions(p)
        out[i] = min(report.levels[e] for e in elig) if elig else default
    return out

