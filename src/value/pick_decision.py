"""Take now or wait? A two-pick lookahead on the draft simulator's draws (ADR 0044).

On the clock at pick ``p`` with the user's next turn ``t`` and ``k`` opponent picks in between, for each
undrafted candidate ``c`` with value ``v_c`` (VORP, or the need-adjusted VORP of ADR 0026):

* ``next_best(c)`` = E[ max v_j over the players still there at ``t`` | I take ``c`` now ], estimated on the
  draft simulator's own draws (``AvailabilityResult.order``, ADR 0040): in each draw the opponents take the
  ``k`` lowest-scoring players of the pool *without c*; the best remaining value is what the next pick gets.
* ``two_pick(c)`` = ``v_c + next_best(c)``. **The recommendation maximises it.**
* ``wait_cost(c)`` = ``two_pick(c) - max_{d != c} two_pick(d)``: the expected two-pick value given up by
  passing on ``c`` now and playing the best other plan (which still gets ``c`` back at ``t`` in the draws
  where he survives). It is positive only for the recommended pick (its margin) and ``<= 0`` for everyone
  else. This is the spec's "VORP(c) - E[value at my next pick | I pass on c]" made consistent: the literal
  form, with the pool otherwise intact, subtracts the same number from everyone and only re-sorts by VORP.
* ``pos_next_best(c)`` = E[best value at ``c``'s coarse position (first letter of G/F/C) at ``t`` | I take
  ``c`` now], for the plain-English reason ("next-best at his position is 40 lower").

Horizon: two picks only. Later turns, opponents' roster needs and the change in the user's own need after
taking ``c`` are not modelled. When nobody at all is left at ``t`` the next pick is worth 0 (replacement).
``playoff_fp`` (fantasy-playoff points), when the board has it, is carried as a tiebreak column only:
it orders exact ties of ``two_pick`` and is never added to the objective.

Advisory, unvalidated heuristic on top of an unvalidated noise model; nothing here changes a projection,
VORP or the board rank. Plain functions, no Streamlit (``tests/value/test_pick_decision.py``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.value.draft_sim import AvailabilityResult

TOP_N = 5
#: Columns of the decision table (``playoff_fp`` only when the board carries it).
DECISION_COLUMNS = ["name", "position", "value_now", "p_avail_next", "next_best", "two_pick", "wait_cost",
                    "pos_next_best", "reason"]
TIEBREAK_COLUMN = "playoff_fp"
#: Labels for the noise-band robustness check: the first and last ``draft_sim.BAND_PARAMS`` settings (the band's
#: low- and high-noise ends). Only the flag "does the pick flip?" uses them.
BAND_LABELS = {"low noise": 0, "high noise": -1}


@dataclass
class DecisionResult:
    status: str                      # ok | back_to_back | last_pick | not_on_clock | empty
    message: str
    table: pd.DataFrame              # DECISION_COLUMNS (+ player_id, playoff_fp), best plan first, top_n rows
    next_pick: int | None = None
    opponent_picks: int | None = None
    value_col: str = "vorp"
    notes: list[str] = field(default_factory=list)
    band: dict[str, str] = field(default_factory=dict)   # noise label -> recommended name under it
    band_flip: bool = False                              # the recommendation changes within the band


def _group(position: object) -> str | None:
    s = str(position or "").strip().upper()
    return s[0] if s[:1] in ("G", "F", "C") else None


def draw_ranks(order: np.ndarray) -> np.ndarray:
    """Inverse of the draws: ``rank[s, i]`` = how many players the opponents take before player ``i`` in draw ``s``."""
    n_sims, n = order.shape
    rank = np.empty((n_sims, n), dtype=np.int32)
    np.put_along_axis(rank, order, np.arange(n, dtype=np.int32)[None, :], axis=1)
    return rank


def expected_next_best(order: np.ndarray, values: np.ndarray, k: int, mask: np.ndarray | None = None, *,
                       candidates: np.ndarray | None = None, rank: np.ndarray | None = None) -> np.ndarray:
    """``E[best value at the next pick | I take c now]`` for each candidate ``c`` (default: every pool player).

    ``order`` is :func:`src.value.draft_sim.simulate_order`'s draws (``order[s, r]`` = the ``r``-th player
    the opponents would take), ``k`` the opponent picks before the next turn. ``mask`` restricts which
    players count as the next pick's options (e.g. one position); a draw where none is left contributes
    ``NaN`` with a mask (averaged out with ``nanmean``; all-``NaN`` stays ``NaN``) and 0 without one.
    ``rank`` (:func:`draw_ranks`) may be passed to share it between calls.

    Taking ``c`` removes him from the pool. If the opponents would have taken him (his draw rank < ``k``),
    they take the ``k`` best-scoring others instead, so the survivors are draw ranks ``>= k + 1``;
    otherwise the survivors are ranks ``>= k`` minus ``c`` himself (the second-best when he was the best).
    """
    n_sims, n = order.shape
    cand = np.arange(n) if candidates is None else np.asarray(candidates, dtype=int)
    if n == 0 or cand.size == 0:
        return np.zeros(cand.size)
    vals = np.asarray(values, float)
    if mask is not None:
        vals = np.where(mask, vals, -np.inf)
    vs = vals[order]                                     # vs[s, r] = value of the r-th player taken
    k = min(max(int(k), 0), n)
    rows = np.arange(n_sims)
    neg = np.full(n_sims, -np.inf)
    if k < n:
        tail = vs[:, k:]
        a = tail.argmax(axis=1)
        m1 = tail[rows, a]
        a_idx = order[rows, k + a]
        tail2 = tail.copy()
        tail2[rows, a] = -np.inf
        m1b = tail2.max(axis=1)
    else:
        m1, m1b, a_idx = neg, neg, np.full(n_sims, -1)
    m2 = vs[:, k + 1:].max(axis=1) if k + 1 < n else neg
    rc = (draw_ranks(order) if rank is None else rank)[:, cand]
    best = np.where(rc < k, m2[:, None], np.where(cand[None, :] == a_idx[:, None], m1b[:, None], m1[:, None]))
    if mask is None:
        return np.where(np.isinf(best), 0.0, best).mean(axis=0)
    best = np.where(np.isinf(best), np.nan, best)
    if not np.isfinite(best).any():
        return np.full(cand.size, np.nan)
    with warnings.catch_warnings():  # an all-NaN column (nobody at his position left in any draw) stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(best, axis=0)


def two_pick_values(order: np.ndarray, values: np.ndarray, k: int, *, keep: int = TOP_N,
                    first: int = 40) -> np.ndarray:
    """``E[next best | take c]`` for every candidate that can reach the top ``keep`` two-pick totals, ``NaN``
    for the rest (exact pruning, not an approximation): the ``first`` highest values are evaluated, then any
    other player whose value plus the largest possible next-pick value (``max(max value, 0)``) still reaches
    the ``keep``-th best total found so far. ``values`` uses ``-inf`` for players without one."""
    vv = np.asarray(values, float)
    ok = np.isfinite(vv)
    nb = np.full(vv.size, np.nan)
    if not ok.any():
        return nb
    rank = draw_ranks(order)
    by_value = np.argsort(-np.where(ok, vv, -np.inf), kind="stable")[: int(ok.sum())]
    head = by_value[:max(first, keep)]
    nb[head] = expected_next_best(order, vv, k, candidates=head, rank=rank)
    two = np.sort(vv[head] + nb[head])[::-1]
    floor = two[min(keep, two.size) - 1]
    bound = max(float(vv[ok].max()), 0.0)
    rest = by_value[max(first, keep):]
    rest = rest[vv[rest] + bound >= floor]
    if rest.size:
        nb[rest] = expected_next_best(order, vv, k, candidates=rest, rank=rank)
    return nb


def band_orders(avail: AvailabilityResult) -> dict[str, np.ndarray]:
    """The simulator's own draws at the low- and high-noise ends of ``draft_sim.BAND_PARAMS``
    (``AvailabilityResult.band_orders``, already drawn for ``p_next_lo``/``p_next_hi``; nothing is re-simulated).
    Empty when there is nothing to wait for or the result carries no band."""
    if avail.table.empty or not avail.on_the_clock or not avail.opponent_picks or avail.opponent_picks[0] == 0:
        return {}
    orders = getattr(avail, "band_orders", None) or []
    return {label: orders[i] for label, i in BAND_LABELS.items()} if orders else {}


def _best_plan(two: np.ndarray, tiebreak: np.ndarray | None, rank: np.ndarray) -> int:
    """Index of the best two-pick plan: highest ``two``, then highest tiebreak, then best board rank."""
    tb = np.zeros(len(two)) if tiebreak is None else np.nan_to_num(np.asarray(tiebreak, float), nan=-np.inf)
    return int(np.lexsort((rank, -tb, -two))[0])


def _reason(row: pd.Series, rec: pd.Series, status: str, target: int | None) -> str:
    if status == "last_pick":
        return "Last pick: highest value left." if row.name == rec.name else \
            f"{-row.wait_cost:.0f} below {rec['name']}; there is no later pick to wait for."
    if status == "back_to_back":
        if row.wait_cost >= -1e-6:
            return f"Take now or at #{target}: no opponent picks in between, so waiting costs nothing."
        return f"Still there at #{target}, but {rec['name']} is worth {-row.wait_cost:.0f} more over the two picks."
    p_there = float(row.p_avail_next)
    p_gone = 1.0 - p_there
    drop = row.value_now - row.pos_next_best if pd.notna(row.pos_next_best) else np.nan
    pos_txt = (f"; next-best {row.position_group} then is {drop:.0f} lower" if pd.notna(drop) and drop > 0.5 else "")
    if row.name == rec.name:
        if p_gone >= 0.5:
            return f"Take now: {p_gone:.0%} gone by #{target}{pos_txt}."
        if row.wait_cost < 1:
            return f"Take now or wait: barely ahead ({row.wait_cost:.0f}); {p_there:.0%} still there at #{target}."
        return f"Take now: best two-pick total (+{row.wait_cost:.0f}); {p_there:.0%} still there at #{target}."
    if p_there >= 0.6:
        return f"Can wait: {p_there:.0%} still there at #{target}."
    return f"{p_gone:.0%} gone by #{target}, but {rec['name']} now is worth {-row.wait_cost:.0f} more over two picks."


def decide(avail: AvailabilityResult, *, values: np.ndarray | pd.Series | None = None, value_col: str = "vorp",
           tiebreak: pd.Series | np.ndarray | None = None, top_n: int = TOP_N,
           alt_orders: dict[str, np.ndarray] | None = None) -> DecisionResult:
    """The take-now-or-wait table from a :func:`src.value.draft_sim.availability_table` result.

    ``values`` (aligned to ``avail.table``'s rows) replaces ``vorp`` as the value, e.g. ``need_adj_vorp``;
    ``tiebreak`` (aligned the same way, e.g. ``playoff_fp``) only orders exact ties. ``alt_orders``
    (:func:`band_orders`) re-runs only the argmax on other draws and flags a recommendation that flips.
    Never mutates inputs."""
    t = avail.table
    if t.empty or avail.current_pick < 1:
        return DecisionResult("empty", "No undrafted players left.", pd.DataFrame(columns=["player_id", *DECISION_COLUMNS]),
                              value_col=value_col)
    if not avail.on_the_clock:
        nxt = f"#{avail.targets[0]}" if avail.targets else "none"
        return DecisionResult("not_on_clock", f"Not your pick yet (current pick #{avail.current_pick}, your next turn "
                              f"{nxt}). The take-now-or-wait advice appears when you are on the clock.",
                              pd.DataFrame(columns=["player_id", *DECISION_COLUMNS]), value_col=value_col)

    v = pd.to_numeric(pd.Series(np.asarray(values if values is not None else t["vorp"], dtype=float)),
                      errors="coerce").to_numpy(float)
    ok = ~np.isnan(v)
    if not ok.any():
        return DecisionResult("empty", "No player on the board has a value.", pd.DataFrame(columns=["player_id", *DECISION_COLUMNS]),
                              value_col=value_col)
    vv = np.where(ok, v, -np.inf)
    target = avail.targets[0] if avail.targets else None
    k = avail.opponent_picks[0] if avail.opponent_picks else None
    groups = t["position"].map(_group) if "position" in t.columns else pd.Series([None] * len(t))

    if target is None:
        status = "last_pick"
        message = "This is your last pick: take the best value left; there is no next pick to wait for."
        nb = np.zeros(len(t))
        pos_nb = np.full(len(t), np.nan)
        p_next = np.full(len(t), np.nan)
    else:
        order = avail.order
        if order is None or order.shape[1] != len(t):
            raise ValueError("availability result carries no simulated draws for this pool")
        nb = two_pick_values(order, vv, k, keep=max(int(top_n), 2))
        pos_nb = np.full(len(t), np.nan)  # filled for the displayed rows below
        p_next = t["p_avail_next"].to_numpy(float)
        if k == 0:
            status = "back_to_back"
            message = (f"Your next pick (#{target}) comes straight after this one with no opponent picks in between: "
                       "waiting costs nothing, so take the top two in either order.")
        else:
            status = "ok"
            message = (f"Two-pick lookahead: this pick and your next one (#{target}, after {k} opponent picks). "
                       "The top row maximises expected value now + best value left at #"
                       f"{target}.")

    two = np.round(np.where(ok & ~np.isnan(nb), v + nb, -np.inf), 6)
    finite = np.isfinite(two)
    srt = np.sort(two[finite])[::-1]
    top1 = srt[0]
    top2 = srt[1] if srt.size > 1 else 0.0  # a lone player: passing on him leaves nothing (value 0)
    wait = np.where(two == top1, two - top2, two - top1)

    out = pd.DataFrame({
        "player_id": t["player_id"].to_numpy() if "player_id" in t.columns else np.arange(len(t)),
        "rank": t["rank"].to_numpy() if "rank" in t.columns else np.arange(1, len(t) + 1),
        "name": t["name"].to_numpy() if "name" in t.columns else "",
        "position": t["position"].to_numpy() if "position" in t.columns else None,
        "position_group": groups.to_numpy(),
        "value_now": v, "p_avail_next": p_next, "next_best": nb, "two_pick": two, "wait_cost": wait,
        "pos_next_best": pos_nb,
    })[finite].copy()
    sort_cols, asc = ["two_pick"], [False]
    if tiebreak is not None:
        out[TIEBREAK_COLUMN] = pd.to_numeric(pd.Series(np.asarray(tiebreak))[finite].to_numpy(), errors="coerce")
        sort_cols.append(TIEBREAK_COLUMN)
        asc.append(False)
    out = out.sort_values([*sort_cols, "rank"], ascending=[*asc, True], kind="stable", na_position="last")
    out = out.head(max(int(top_n), 1))
    if target is not None:
        rank_main = draw_ranks(avail.order)
        idx = out.index.to_numpy()
        for g in ("G", "F", "C"):
            m = (groups == g).to_numpy()
            sel = idx[m[idx]]
            if sel.size:
                out.loc[sel, "pos_next_best"] = expected_next_best(avail.order, vv, k, mask=m, candidates=sel,
                                                                   rank=rank_main)
    rec_idx = int(out.index[0])
    out = out.reset_index(drop=True)
    rec = out.iloc[0]
    out["reason"] = [_reason(r, rec, status, target) for _, r in out.iterrows()]
    for c in ("value_now", "next_best", "two_pick", "wait_cost", "pos_next_best"):
        out[c] = out[c].round(1)
    cols = ["player_id", *DECISION_COLUMNS] + ([TIEBREAK_COLUMN] if tiebreak is not None else [])
    band: dict[str, str] = {}
    flip = False
    if status == "ok" and alt_orders:
        names = t["name"].to_numpy() if "name" in t.columns else np.arange(len(t)).astype(str)
        rank = t["rank"].to_numpy(float) if "rank" in t.columns else np.arange(len(t), dtype=float)
        tb = None if tiebreak is None else pd.to_numeric(pd.Series(np.asarray(tiebreak)), errors="coerce").to_numpy(float)
        for label, o in alt_orders.items():
            alt_nb = two_pick_values(o, vv, k, keep=2)
            alt = np.round(np.where(ok & ~np.isnan(alt_nb), v + alt_nb, -np.inf), 6)
            j = _best_plan(alt, tb, rank)
            band[label] = str(names[j])
            flip = flip or j != rec_idx
    if band:
        if flip:
            alts = ", ".join(f"{n} at {lab}" for lab, n in band.items())
            message += f" Sensitive to the draft-noise assumption: the pick would be {alts}; treat the top options as close."
        else:
            message += " The same pick wins at low and high draft noise."
    return DecisionResult(status, message, out[cols], next_pick=target, opponent_picks=k, value_col=value_col,
                          notes=list(avail.notes), band=band, band_flip=flip)
