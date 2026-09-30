"""Waiver / free-agent finder: who to add, who to drop, who to stream this week.

    python -m src.inseason.waivers --season 2026-27 --as-of 2026-12-01 --team-id 3 --top 20
    python -m src.inseason.waivers ... --my-roster "A, B, ..."            # explicit roster instead of ESPN
    python -m src.inseason.waivers ... --mock-draft-team 5                # demo rosters (mock snake draft)
    python -m src.inseason.waivers ... --next-week                        # stream for next matchup week

Three views (ADR 0015 D4), all from the same rest-of-season frame as the trade analyzer:

1. **Adds by ROS value over who they would replace** (``adds``). For each free agent the best drop is
   chosen among your players by marginal lineup value: ``gain = value(roster + FA - drop) - value(roster)``
   in rest-of-season FP, using ``src.inseason.lineup`` (positional eligibility, freed slot, bench weight).
   With an open roster spot the alternative is a replacement-level player, so the gain is measured against
   leaving the spot to a replacement (the drop reads "(open roster spot)").
2. **Flags** on every candidate: rising minutes and rising usage (last 5 games vs earlier this season),
   an injured teammate whose minutes he is likely to inherit (``src.inseason.signals``: game-log absence or
   ESPN ``OUT``), and his own ESPN injury status.
3. **Streaming** (``streams``): free agents ranked by expected fantasy points in the coming matchup week
   (games left in the week x availability x ROS FPPG), with games, back-to-backs and off-night games from
   the schedule, and the weekly gain over the player you would drop (this week's expected FP, not ROS).

The free-agent pool is every NBA-active player (played this season by ``as_of``, or on the latest roster
snapshot) that no league team has rostered; before the draft that is everyone. Free agents are limited to
the top ``candidates`` by ROS total for speed (the best add is never far down that list).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.inseason.context import (
    ContextUnavailable, InSeasonContext, RosterChoice, free_agent_ids, load_context, resolve_roster,
)
from src.inseason.lineup import (
    NameError_, bench_weight_for, fit_to_capacity, lineup_from_arrays, lineup_value, roster_frame,
)
from src.inseason.schedule import ABBR_OF_TEAM_ID, expected_week_games
from src.inseason.signals import absent_players, beneficiaries, trend_signals
from src.inseason.trade import add_roster_args, replacement_value
from src.value.replacement import league_shape

DEFAULT_CANDIDATES = 80


@dataclass
class WaiverReport:
    adds: pd.DataFrame
    streams: pd.DataFrame
    absent: pd.DataFrame
    week: int | None
    week_label: str
    notes: list[str]


def _season_logs(ctx: InSeasonContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.inseason.ros import season_to_date

    return season_to_date(ctx.tables, ctx.season, ctx.as_of, ctx.cfg["scoring"])


def pick_week(ctx: InSeasonContext, *, next_week: bool = False):
    if ctx.calendar is None:
        return None
    day = ctx.week_anchor if ctx.week_anchor is not None else ctx.as_of
    return ctx.calendar.next_week(day) if next_week else ctx.calendar.current_week(day)


def find_waivers(ctx: InSeasonContext, choice: RosterChoice, *, top: int = 25, candidates: int = DEFAULT_CANDIDATES,
                 next_week: bool = False) -> WaiverReport:
    shape = league_shape(ctx.cfg)
    n_ir = int(ctx.cfg["league"]["roster"].get("ir", 1))
    ros = ctx.ros.drop_duplicates("player_id")
    idx = ros.set_index("player_id")
    w = bench_weight_for(ros, shape)
    repl = replacement_value(ros, shape)
    notes: list[str] = []

    gl, tg = _season_logs(ctx)
    trends = trend_signals(gl).set_index("player_id") if len(gl) else pd.DataFrame()
    absent = absent_players(gl, tg, injuries=ctx.injuries)
    positions = idx["position"]
    fpm = (gl.groupby("player_id")["fp"].sum() / gl.groupby("player_id")["min"].sum()) if len(gl) else pd.Series(dtype=float)
    ben = beneficiaries(gl, tg, absent, positions, fp_per_min=fpm) if len(absent) else pd.DataFrame()
    if not len(gl):
        notes.append("no games yet this season: minute trends and injury-replacement flags need game logs")

    week = pick_week(ctx, next_week=next_week)
    week_label = "no schedule"
    if week is not None and ctx.weekly is not None:
        week_label = f"week {week.week} ({week.start} .. {week.end}, {week.kind})"

    mine = roster_frame(idx, choice.my_ids)
    mine, _, _ = fit_to_capacity(mine, shape, w, None, repl, ir_ids=choice.ir_ids, n_ir=n_ir)
    base = lineup_value(mine, shape, w, ir_ids=choice.ir_ids, n_ir=n_ir)
    ir_set = set(choice.ir_ids)
    drop_options = [int(p) for p in mine["player_id"] if p not in ir_set]
    open_spot = False   # open spots are already filled with replacement-level fillers above, and those can be "dropped"

    fa = [p for p in free_agent_ids(ctx, choice.rostered) if p in idx.index and p not in set(choice.my_ids)]
    fa_frame = idx.loc[fa].sort_values("ros_total_fp", ascending=False)
    # candidates: best by ROS total plus anyone flagged rising / inheriting minutes (cheap to include)
    flagged = set()
    if len(trends):
        flagged |= set(trends.index[(trends["rising_minutes"] | trends["rising_usage"])])
    if len(ben):
        flagged |= set(ben["player_id"])
    cand_ids = list(fa_frame.head(candidates).index) + [p for p in fa_frame.index if p in flagged and p not in set(fa_frame.head(candidates).index)][:40]

    names = idx["name"]
    week_games_left = pd.Series(dtype=float)
    week_tbl = pd.DataFrame()
    if week is not None and ctx.weekly is not None:
        week_tbl = ctx.weekly[ctx.weekly["week"] == week.week].set_index("team_id")
        team_of = idx["team_id"]
        remaining_only = not next_week
        wg = expected_week_games(team_of.reindex(idx.index).to_numpy(), ctx.weekly, week.week,
                                 remaining_only=remaining_only, availability=idx["avail"].to_numpy())
        week_games_left = pd.Series(wg, index=idx.index)


    m_pids, m_vals, m_pos = mine["player_id"].to_numpy(), mine["value"].to_numpy(float), mine["position"].to_numpy(object)
    rows = []
    for pid in cand_ids:
        r = idx.loc[pid]
        best = (-1e18, None)
        options = drop_options + ([None] if open_spot else [])
        for d in options:
            keep = np.ones(len(m_pids), dtype=bool) if d is None else (m_pids != d)
            v = lineup_from_arrays(np.append(m_pids[keep], pid), np.append(m_vals[keep], float(r["ros_total_fp"])),
                                   np.append(m_pos[keep], r["position"]), shape, w, ir_ids=choice.ir_ids, n_ir=n_ir).value
            if v - base.value > best[0]:
                best = (v - base.value, d)
        gain, drop = best
        row = {"player_id": int(pid), "name": r["name"], "position": r["position"],
               "team": ABBR_OF_TEAM_ID.get(int(r["team_id"]), r["team_id"]) if pd.notna(r["team_id"]) else "",
               "ros_fppg": r["ros_fppg"], "ros_games": r["ros_games"], "ros_total_fp": r["ros_total_fp"],
               "gain_ros_fp": float(gain),
               "drop": "(open roster spot)" if (drop is None or drop < 0) else names.get(drop, drop),
               "drop_id": None if (drop is None or drop < 0) else drop}
        if pid in trends.index:
            t = trends.loc[pid]
            row.update({"min_delta": t["min_delta"], "usg_delta": t["usg_delta"],
                        "rising_minutes": bool(t["rising_minutes"]), "rising_usage": bool(t["rising_usage"])})
        else:
            row.update({"min_delta": np.nan, "usg_delta": np.nan, "rising_minutes": False, "rising_usage": False})
        if len(ben) and pid in set(ben["player_id"]):
            b = ben[ben["player_id"] == pid].sort_values("gain_mpg", ascending=False).iloc[0]
            row.update({"inherits_from": names.get(int(b["out_player_id"]), b["out_player_id"]),
                        "exp_gain_mpg": float(b["gain_mpg"])})
        else:
            row.update({"inherits_from": "", "exp_gain_mpg": np.nan})
        row["injury"] = "" if ctx.injuries is None else ctx.injuries.get(pid, "")
        if len(week_games_left):
            g = float(week_games_left.get(pid, 0.0))
            wk_fp = g * float(r["ros_fppg"])
            drop_wk = 0.0
            if drop is not None and drop >= 0 and drop in week_games_left.index:
                drop_wk = float(week_games_left[drop] * idx.at[drop, "ros_fppg"])
            elif drop is None:
                drop_wk = 0.0
            tid = r["team_id"]
            b2b = int(week_tbl["b2b"].get(tid, 0)) if len(week_tbl) else 0
            off = int(week_tbl["off_night_games"].get(tid, 0)) if len(week_tbl) else 0
            tgm = float(week_tbl["games_left" if not next_week else "games"].get(tid, 0)) if len(week_tbl) else 0.0
            row.update({"week_team_games": tgm, "week_exp_games": g, "week_exp_fp": wk_fp,
                        "week_gain_fp": wk_fp - drop_wk, "week_b2b": b2b, "week_off_night": off})
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return WaiverReport(table, table, absent, week.week if week else None, week_label, notes)
    # the ESPN injury flag on the *candidate* removes him from streaming unless healthy enough to play this week
    adds = table.sort_values("gain_ros_fp", ascending=False).head(top).reset_index(drop=True)
    if float(adds["gain_ros_fp"].max()) <= 0:
        notes.append("no free agent improves your roster over the rest of the season (every add-drop is <= 0)")
    if "week_exp_fp" in table.columns:
        stream_pool = table[table["injury"].ne("OUT")]
        streams = stream_pool.sort_values(["week_gain_fp", "week_exp_fp"], ascending=False).head(top).reset_index(drop=True)
    else:
        streams = table.iloc[0:0]
        notes.append("no schedule loaded: streaming view unavailable")
    return WaiverReport(adds, streams, absent, week.week if week else None, week_label, notes)


# --------------------------------------------------------------------------- CLI

def _safe_print(text: str, *, file=None) -> None:
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), file=stream)


ADD_COLS = ["name", "position", "team", "ros_fppg", "ros_games", "ros_total_fp", "gain_ros_fp", "drop", "min_delta",
            "rising_minutes", "rising_usage", "inherits_from", "exp_gain_mpg", "injury"]
STREAM_COLS = ["name", "position", "team", "week_team_games", "week_exp_games", "week_b2b", "week_off_night",
               "ros_fppg", "week_exp_fp", "week_gain_fp", "drop", "injury"]


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.waivers", description=__doc__.split("\n\n")[0])
    add_roster_args(ap)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--next-week", action="store_true", help="stream for the next matchup week instead of this one")
    ap.add_argument("--no-injuries", action="store_true", help="ignore the cached ESPN injury snapshot")
    args = ap.parse_args(argv)
    try:
        ctx = load_context(args.season, args.as_of, model=args.model, data_dir=Path(args.data_dir) if args.data_dir else None,
                           synthetic=args.synthetic, league_id=args.league_id, use_injuries=False if args.no_injuries else None)
        choice = resolve_roster(ctx, team_id=args.team_id, roster_names=args.my_roster, mock_team=args.mock_draft_team)
        rep = find_waivers(ctx, choice, top=args.top, next_week=args.next_week)
    except (ContextUnavailable, NameError_, ValueError, KeyError) as exc:
        _safe_print(f"error: {exc}", file=sys.stderr)
        return 2
    _safe_print(f"Waiver finder, {ctx.season} as of {ctx.as_of.date()} ({choice.label}); streaming for {rep.week_label}")
    for n in ctx.notes + rep.notes:
        _safe_print(f"note: {n}")
    fmt = lambda x: f"{x:.1f}"  # noqa: E731
    if len(rep.adds):
        _safe_print("\nBest adds by rest-of-season value over the player they would replace:")
        _safe_print(rep.adds[[c for c in ADD_COLS if c in rep.adds.columns]].to_string(index=False, float_format=fmt))
    if len(rep.streams):
        _safe_print(f"\nStreaming candidates for {rep.week_label} (expected FP this week over the drop):")
        _safe_print(rep.streams[[c for c in STREAM_COLS if c in rep.streams.columns]].to_string(index=False, float_format=fmt))
    if len(rep.absent):
        names = ctx.ros.drop_duplicates("player_id").set_index("player_id")["name"]
        a = rep.absent.assign(name=rep.absent["player_id"].map(names)).sort_values("mpg", ascending=False).head(12)
        _safe_print("\nRotation players currently out (their teammates gain minutes):")
        _safe_print(a[["name", "team_id", "mpg", "streak", "long_term", "source"]].to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
