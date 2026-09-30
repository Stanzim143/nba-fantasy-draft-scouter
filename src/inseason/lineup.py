"""Slot-aware roster value: what a roster is worth over the rest of the season given the league's lineup.

The league starts PG, SG, SF, PF, C, G, F and 3 UTIL (10 slots), with 3 bench spots and 1 IR spot
(``config/league.yaml``). A roster's value is::

    value = sum of ROS totals of the best starting assignment  +  w * sum of ROS totals of the best bench

* **Starting assignment**: a maximum-weight matching of players to the 10 slots (``scipy``'s
  ``linear_sum_assignment``), where a player may fill a slot only if his position makes him eligible
  (``src.value.positions.eligible_slots``: ``G`` = PG or SG, ``F`` = SF or PF, ``UTIL`` = anyone). Empty
  slots are allowed (a dummy zero-value player), so a roster with no centre simply leaves C empty.
* **Bench weight** ``w``: the same quantity the value engine derives for replacement level
  (``src.value.replacement.derive_bench_weight``): on a typical day a share of starting slots is vacated
  by injuries and rest, and the bench covers up to its size. Extra players beyond bench size are worth 0.
* **IR**: up to ``ir`` players marked injured-reserve do not use a roster spot; they are valued like
  bench players (they return, and their ROS total already discounts the games they miss).

Approximation, stated plainly: daily lineups and game-by-game injuries are not simulated. Each starter
is assumed to occupy his slot for every game he is expected to play, so slot value is the player's
expected ROS total. This is the same "expected volume" logic as the draft board's VORP, and it is what
makes freed and filled roster spots comparable.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.value.positions import eligible_slots
from src.value.replacement import LeagueShape, derive_bench_weight

FORBIDDEN = 1e9
GENERIC_POSITION = "PG-SG-SF-PF-C"   # a virtual replacement player fits any slot


def slot_names(shape: LeagueShape) -> list[str]:
    return [slot for slot, n in shape.starter_slots.items() for _ in range(n)]


@dataclass
class LineupResult:
    value: float
    starter_value: float
    bench_value: float
    starters: dict[str, int | None]      # slot label (PG, ..., UTIL#2) -> player_id (None = empty slot)
    bench: list[int]
    ir: list[int]
    excess: list[int]                    # rostered but worth nothing (over the bench size)
    empty_slots: list[str] = field(default_factory=list)


def bench_weight_for(ros: pd.DataFrame, shape: LeagueShape) -> float:
    """Bench weight from the ROS frame: mean availability of the players a full league starts."""
    top = ros.sort_values("ros_total_fp", ascending=False).head(max(shape.teams * shape.n_starters, 1))
    return derive_bench_weight(shape, float(top["avail"].mean()) if len(top) else 1.0)


@lru_cache(maxsize=None)
def _elig_row(position, slots: tuple) -> tuple:
    have = set(eligible_slots(position))
    return tuple(slot in have for slot in slots)


def lineup_value(roster: pd.DataFrame, shape: LeagueShape, bench_weight: float, *, ir_ids=(), n_ir: int = 1) -> LineupResult:
    """Value of ``roster`` (columns ``player_id, position, value``) under the league's slots."""
    return lineup_from_arrays(roster["player_id"].to_numpy(), roster["value"].to_numpy(float),
                              roster["position"].to_numpy(object), shape, bench_weight, ir_ids=ir_ids, n_ir=n_ir)


def lineup_from_arrays(pids, vals_all, pos_all, shape: LeagueShape, bench_weight: float, *, ir_ids=(),
                       n_ir: int = 1) -> LineupResult:
    """Array form of :func:`lineup_value` (no pandas in the loop; used by the free-agent search)."""
    ir_ids = set(ir_ids)
    is_ir = np.array([p in ir_ids for p in pids], dtype=bool)
    ir_idx = np.flatnonzero(is_ir)
    ir_idx = ir_idx[np.argsort(-vals_all[ir_idx], kind="stable")][:n_ir]
    keep = np.ones(len(pids), dtype=bool)
    keep[ir_idx] = False
    act = np.flatnonzero(keep)
    slots = tuple(slot_names(shape))
    n, s = len(act), len(slots)
    vals = vals_all[act]
    elig = np.array([_elig_row(pos_all[i], slots) for i in act], dtype=bool).reshape(n, s)
    cost = np.zeros((n + s, s))            # rows n..n+s-1 are dummy empty-slot fillers (value 0)
    cost[:n] = np.where(elig, -vals[:, None], FORBIDDEN)
    rows, cols = linear_sum_assignment(cost)
    labels, seen = [], {}
    for slot in slots:
        seen[slot] = seen.get(slot, 0) + 1
        labels.append(slot if slots.count(slot) == 1 else f"{slot}#{seen[slot]}")
    starters: dict[str, int | None] = {lab: None for lab in labels}
    used: set[int] = set()
    starter_value = 0.0
    for r, c in zip(rows, cols):
        if r < n and cost[r, c] < FORBIDDEN / 2:
            starters[labels[c]] = int(pids[act[r]])
            used.add(r)
            starter_value += vals[r]
    empty = [lab for lab in labels if starters[lab] is None]
    rest = [i for i in range(n) if i not in used]
    rest.sort(key=lambda i: (-vals[i], pids[act[i]]))
    bench, excess = rest[:shape.bench], rest[shape.bench:]
    bench_sum = float(sum(vals[i] for i in bench)) + float(vals_all[ir_idx].sum())
    bench_value = bench_weight * bench_sum
    return LineupResult(
        value=float(starter_value) + bench_value, starter_value=float(starter_value), bench_value=float(bench_value),
        starters=starters, bench=[int(pids[act[i]]) for i in bench], ir=[int(pids[i]) for i in ir_idx],
        excess=[int(pids[act[i]]) for i in excess], empty_slots=empty)


def roster_frame(ros_idx: pd.DataFrame, ids, value_col: str = "ros_total_fp") -> pd.DataFrame:
    """Roster rows from the ROS frame indexed by player_id; unknown ids are an error, never a silent zero."""
    ids = list(ids)
    missing = [i for i in ids if i not in ros_idx.index]
    if missing:
        raise KeyError(f"players not in the ROS projection: {missing}")
    r = ros_idx.loc[ids]
    return pd.DataFrame({"player_id": r.index.to_numpy(), "position": r["position"].to_numpy(),
                         "value": r[value_col].to_numpy(float)})


def replacement_player(ros: pd.DataFrame, value: float, pid: int = -1) -> dict:
    return {"player_id": pid, "position": GENERIC_POSITION, "value": float(value)}


def roster_capacity(shape: LeagueShape) -> int:
    return shape.roster_size


def fit_to_capacity(frame: pd.DataFrame, shape: LeagueShape, bench_weight: float, pool: pd.DataFrame | None,
                    repl_value: float, *, ir_ids=(), n_ir: int = 1, max_pool: int = 60) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Make ``frame`` exactly a full roster: drop the least valuable players when over, add the best free
    agents (greedy, by marginal lineup value; a generic replacement-level player if no pool) when under.

    Returns ``(frame, dropped_ids, added_ids)``; added ids of generic replacements are negative.
    """
    frame = frame.copy()
    dropped: list[int] = []
    added: list[int] = []
    ir_ids = set(ir_ids)
    cap = roster_capacity(shape)

    def n_active(f: pd.DataFrame) -> int:
        ir_here = f[f["player_id"].isin(ir_ids)]
        return len(f) - min(len(ir_here), n_ir)

    while n_active(frame) > cap:
        best = None
        base = lineup_value(frame, shape, bench_weight, ir_ids=ir_ids, n_ir=n_ir).value
        for pid in frame["player_id"]:
            if pid in ir_ids:
                continue
            loss = base - lineup_value(frame[frame["player_id"] != pid], shape, bench_weight, ir_ids=ir_ids, n_ir=n_ir).value
            if best is None or loss < best[0]:
                best = (loss, pid)
        frame = frame[frame["player_id"] != best[1]]
        dropped.append(int(best[1]))
    fake = -1
    while n_active(frame) < cap:
        cands = []
        if pool is not None and len(pool):
            cands = pool[~pool["player_id"].isin(set(frame["player_id"]))].sort_values(
                "value", ascending=False).head(max_pool).to_dict("records")
        cands.append(replacement_player(None, repl_value, fake))
        base = lineup_value(frame, shape, bench_weight, ir_ids=ir_ids, n_ir=n_ir).value
        best = max(cands, key=lambda c: lineup_value(pd.concat([frame, pd.DataFrame([c])], ignore_index=True), shape,
                                                     bench_weight, ir_ids=ir_ids, n_ir=n_ir).value - base)
        frame = pd.concat([frame, pd.DataFrame([best])], ignore_index=True)
        added.append(int(best["player_id"]))
        if best["player_id"] == fake:
            fake -= 1
    return frame.reset_index(drop=True), dropped, added


# --------------------------------------------------------------------------- names

def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


class NameError_(ValueError):
    """A player name matched nothing or several players; the message lists the candidates."""


def resolve_name(ros: pd.DataFrame, query: str) -> int:
    """Player id for a (partial, accent-insensitive) name. Exact match wins; otherwise the query must match
    exactly one player as a substring."""
    q = normalize_name(query)
    norm = ros["name"].map(normalize_name)
    exact = ros[norm == q]
    if len(exact) == 1:
        return int(exact["player_id"].iloc[0])
    hit = ros[norm.str.contains(re.escape(q), na=False)] if q else ros.iloc[0:0]
    if len(hit) == 1:
        return int(hit["player_id"].iloc[0])
    if len(hit) == 0:
        raise NameError_(f"no player matches {query!r}")
    top = ", ".join(f"{n} ({p})" for n, p in zip(hit["name"].head(6), hit["position"].head(6)))
    raise NameError_(f"{query!r} is ambiguous: {top}")


def resolve_names(ros: pd.DataFrame, text: str) -> list[int]:
    return [resolve_name(ros, part) for part in text.split(",") if part.strip()]
