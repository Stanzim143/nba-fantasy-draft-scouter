"""Trade analyzer: rest-of-season fantasy points in vs out, adjusted for lineup slots and roster spots.

    python -m src.inseason.trade --season 2026-27 --as-of 2026-12-01 --team-id 3 \
        --give "Tyrese Maxey" --get "Nikola Jokic, Naz Reid"
    python -m src.inseason.trade ... --my-roster "A, B, C, ..."        # explicit roster instead of ESPN
    python -m src.inseason.trade ... --mock-draft-team 5               # demo rosters (mock snake draft)

Library::

    result = evaluate_trade(ros, roster_ids, give_ids, get_ids, shape, bench_weight, pool_ids=free_agents)
    result.delta          # rest-of-season FP gained (+) or lost (-) by the trade, all effects included

What "all effects" means (ADR 0015 D3). Both the roster before and after the trade are valued with
``src.inseason.lineup`` (best assignment of players to the league's PG SG SF PF C G F UTIL x3 slots plus a
weighted bench), and both are first made a *full* roster:

* a 2-for-1 leaves a spot open, which is filled by the best free agent available (greedy by marginal lineup
  value; a generic replacement-level player when no pool is known), so the trade is not credited with the
  empty spot's phantom zero, and is not blamed for it either;
* a 1-for-2 overfills the roster, so the least valuable player (by marginal lineup value) is dropped and
  named in the result;
* positional eligibility matters through the matching: trading your only centre for a guard leaves C empty
  unless another player is eligible there.

``delta`` is ``value(after) - value(before)`` in rest-of-season fantasy points. It is an expected-volume
number like the draft board's VORP (daily lineups and game-by-game injuries are not simulated, see
``lineup.py``). A verdict of "roughly even" is given inside ``tolerance`` FP (default 1.5% of the roster's
value: ROS projections are not that precise, see the backtest evidence in ``docs/inseason.md``).
The same function evaluates the partner's side when you know their roster (``--partner-team-id``).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

import pandas as pd

from src.inseason.context import (
    ContextUnavailable, InSeasonContext, RosterChoice, free_agent_ids, load_context, resolve_roster,
)
from src.inseason.lineup import (
    LineupResult, NameError_, bench_weight_for, fit_to_capacity, lineup_value, resolve_names, roster_frame,
)
from src.value.replacement import LeagueShape, league_shape, replacement_level

DEFAULT_TOLERANCE = 0.015

# --------------------------------------------------------------------------- confidence bucketing (ADR 0027)

#: How many multiples of ``tolerance`` a delta must clear the tolerance band by to count as a "clear"
#: win/loss rather than just a "lean": chosen so the five buckets read as roughly even / a lean either
#: way / a clear verdict either way, on the same tolerance band :func:`evaluate_trade` already uses to
#: call "roughly even" (ROS projections are not precise enough to trust anything inside it).
CONFIDENCE_CLEAR_MULT = 3.0

#: Ordered bucket keys, worst-for-you to best-for-you (stable order for display and tests).
CONFIDENCE_BUCKETS = ["clear_loss", "lean_other_way", "roughly_even", "lean_your_way", "clear_win"]

#: Human, display-ready label for each bucket key.
CONFIDENCE_LABELS: dict[str, str] = {
    "clear_win": "clear win",
    "lean_your_way": "lean your way",
    "roughly_even": "roughly even",
    "lean_other_way": "lean other way",
    "clear_loss": "clear loss",
}


def confidence_bucket(delta: float, tolerance: float, *, clear_mult: float = CONFIDENCE_CLEAR_MULT) -> str:
    """Bucket a trade's ``delta`` against its ``tolerance`` band into one of :data:`CONFIDENCE_BUCKETS`.

    This is a magnitude read on ``TradeResult.verdict``'s own tolerance band, not a new statistic: inside
    the band is "roughly even" (same rule as ``verdict``); up to ``clear_mult`` x the band is a "lean";
    beyond that is a "clear" win or loss. It exists so the app can show a legible tier instead of a raw
    FP float, the same way the draft board shows a tier instead of a raw VORP number.

    NaN-safe: a non-finite ``delta`` reads as no signal (``roughly_even``); a non-finite or non-positive
    ``tolerance`` (e.g. an empty roster) falls back to sign-only bucketing so the function never raises.
    """
    d = float(delta) if math.isfinite(delta) else 0.0
    tol = float(tolerance) if math.isfinite(tolerance) and tolerance > 0 else 0.0
    if tol <= 0:
        if d == 0:
            return "roughly_even"
        return "clear_win" if d > 0 else "clear_loss"
    mag = abs(d) / tol
    if mag <= 1.0:
        return "roughly_even"
    if mag <= clear_mult:
        return "lean_your_way" if d > 0 else "lean_other_way"
    return "clear_win" if d > 0 else "clear_loss"


def confidence_label(bucket: str) -> str:
    """Human label for a :func:`confidence_bucket` key, or the key itself if it is somehow unknown."""
    return CONFIDENCE_LABELS.get(bucket, bucket)


@dataclass
class TradeResult:
    give: list[int]
    get: list[int]
    before: LineupResult
    after: LineupResult
    delta: float
    dropped: list[int] = field(default_factory=list)   # forced drops after the trade (roster overfull)
    added: list[int] = field(default_factory=list)     # free agents added to fill freed spots (<0 = generic)
    added_before: list[int] = field(default_factory=list)
    incoming_ros: float = 0.0
    outgoing_ros: float = 0.0
    verdict: str = ""
    tolerance: float = 0.0


def replacement_value(ros: pd.DataFrame, shape: LeagueShape) -> float:
    """ROS total of a replacement-level player (the free agent you get for free), from the value engine."""
    rep = ros.attrs.get("replacement")
    if rep and "total" in rep:
        return float(rep["total"])
    return replacement_level(ros["ros_total_fp"], ros["ros_fppg"], ros["ros_games"], shape,
                             season_games=max(float(ros["team_games_left"].median()), 1.0)).total


def evaluate_trade(ros: pd.DataFrame, roster_ids, give_ids, get_ids, shape: LeagueShape, bench_weight: float, *,
                   pool_ids=None, ir_ids=(), n_ir: int = 1, tolerance: float = DEFAULT_TOLERANCE,
                   repl_value: float | None = None) -> TradeResult:
    """Value of ``give`` -> ``get`` for the owner of ``roster_ids`` (see the module docstring)."""
    roster_ids, give_ids, get_ids = list(roster_ids), list(give_ids), list(get_ids)
    idx = ros.drop_duplicates("player_id").set_index("player_id")
    if set(give_ids) - set(roster_ids):
        raise ValueError(f"cannot give players who are not on the roster: {sorted(set(give_ids) - set(roster_ids))}")
    if set(get_ids) & set(roster_ids):
        raise ValueError("cannot receive players who are already on the roster")
    if not give_ids and not get_ids:
        raise ValueError("a trade needs at least one player on a side")
    repl = replacement_value(ros, shape) if repl_value is None else repl_value
    pool = None
    if pool_ids is not None:
        excluded = set(roster_ids) | set(get_ids)
        pool = roster_frame(idx, [p for p in pool_ids if p in idx.index and p not in excluded])

    def valued(ids):
        frame = roster_frame(idx, ids)
        fitted, dropped, added = fit_to_capacity(frame, shape, bench_weight, pool, repl, ir_ids=ir_ids, n_ir=n_ir)
        return fitted, lineup_value(fitted, shape, bench_weight, ir_ids=ir_ids, n_ir=n_ir), dropped, added

    _, before, _, added_before = valued(roster_ids)
    after_ids = [p for p in roster_ids if p not in set(give_ids)] + get_ids
    _, after, dropped, added = valued(after_ids)
    delta = after.value - before.value
    tol = tolerance * before.value
    verdict = ("roughly even" if abs(delta) <= tol else ("favours you" if delta > 0 else "favours the other side"))
    return TradeResult(
        give=give_ids, get=get_ids, before=before, after=after, delta=float(delta), dropped=dropped, added=added,
        added_before=added_before, incoming_ros=float(idx.loc[get_ids, "ros_total_fp"].sum()) if get_ids else 0.0,
        outgoing_ros=float(idx.loc[give_ids, "ros_total_fp"].sum()) if give_ids else 0.0,
        verdict=verdict, tolerance=float(tol))


def analyze(ctx: InSeasonContext, choice: RosterChoice, give_ids, get_ids, *, partner: str | None = None) -> dict:
    """Run :func:`evaluate_trade` for the chosen roster (and the partner's side if ``partner`` names one)."""
    shape = league_shape(ctx.cfg)
    n_ir = int(ctx.cfg["league"]["roster"].get("ir", 1))
    w = bench_weight_for(ctx.ros, shape)
    pool = free_agent_ids(ctx, choice.rostered)
    mine = evaluate_trade(ctx.ros, choice.my_ids, give_ids, get_ids, shape, w, pool_ids=pool, ir_ids=choice.ir_ids,
                          n_ir=n_ir)
    out = {"mine": mine, "bench_weight": w, "partner": None, "partner_label": None}
    if partner:
        their = choice.others.get(partner)
        if their is None:
            raise ContextUnavailable(f"unknown partner {partner!r}; known: {sorted(choice.others)}")
        out["partner"] = evaluate_trade(ctx.ros, their, get_ids, give_ids, shape, w, pool_ids=pool, n_ir=n_ir)
        out["partner_label"] = partner
    return out


def describe(ctx: InSeasonContext, res: TradeResult, label: str) -> str:
    names = ctx.ros.drop_duplicates("player_id").set_index("player_id")["name"]

    def nm(ids):
        return ", ".join("a replacement-level free agent" if i < 0 else str(names.get(i, i)) for i in ids) or "nobody"

    lines = [f"{label}: gives {nm(res.give)} ({res.outgoing_ros:.0f} FP ROS), gets {nm(res.get)} ({res.incoming_ros:.0f} FP ROS)",
             f"  roster value over the rest of the season: {res.before.value:.0f} -> {res.after.value:.0f} FP "
             f"({res.delta:+.0f}; {res.verdict}, tolerance +-{res.tolerance:.0f})"]
    if res.added:
        lines.append(f"  freed roster spot(s) filled by: {nm(res.added)}")
    if res.dropped:
        lines.append(f"  roster overfull, drop: {nm(res.dropped)}")
    if res.after.empty_slots:
        lines.append(f"  empty starting slots after the trade: {', '.join(res.after.empty_slots)}")
    return "\n".join(lines)


def _safe_print(text: str, *, file=None) -> None:
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), file=stream)


def add_roster_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--season", default=None)
    ap.add_argument("--as-of", default=None, help="inclusive cutoff date (default: today)")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--league-id", type=int, default=None, help="ESPN league id ($ESPN_LEAGUE_ID)")
    ap.add_argument("--team-id", type=int, default=None, help="your ESPN team id ($ESPN_TEAM_ID)")
    ap.add_argument("--my-roster", default=None, help="comma-separated player names instead of the ESPN roster")
    ap.add_argument("--mock-draft-team", type=int, default=None, help="use team N of a mock snake draft (demo)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic league (demo/testing only)")
    ap.add_argument("--data-dir", default=None)


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m src.inseason.trade", description=__doc__.split("\n\n")[0])
    add_roster_args(ap)
    ap.add_argument("--give", required=True, help="comma-separated names you would send")
    ap.add_argument("--get", required=True, help="comma-separated names you would receive")
    ap.add_argument("--partner", default=None, help="partner team label (see the list printed on error), to score their side")
    args = ap.parse_args(argv)
    try:
        ctx = load_context(args.season, args.as_of, model=args.model, data_dir=Path(args.data_dir) if args.data_dir else None,
                           synthetic=args.synthetic, league_id=args.league_id)
        choice = resolve_roster(ctx, team_id=args.team_id, roster_names=args.my_roster, mock_team=args.mock_draft_team)
        give, get = resolve_names(ctx.ros, args.give), resolve_names(ctx.ros, args.get)
        out = analyze(ctx, choice, give, get, partner=args.partner)
    except (ContextUnavailable, NameError_, ValueError, KeyError) as exc:
        _safe_print(f"error: {exc}", file=sys.stderr)
        return 2
    _safe_print(f"Trade analysis, {ctx.season} as of {ctx.as_of.date()} ({choice.label})")
    for n in ctx.notes:
        _safe_print(f"note: {n}")
    _safe_print(describe(ctx, out["mine"], "You"))
    if out["partner"] is not None:
        _safe_print(describe(ctx, out["partner"], out["partner_label"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
