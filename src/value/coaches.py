"""Coaches for the season about to start: who runs each team, how, and which ranked players sit under a new one (ADR 0020).

    python -m src.value.coaches [--season 2026-27] [--board-csv reports/daily/draft_board_baseline_offseason_debut.csv]

Descriptive, like the draft-day risk overlay: nothing here changes a projection. ``baseline_coach`` (the same information as an
adjustment to minutes) was measured on ten real seasons and did not help (ADR 0020), so this shows the facts and leaves the
judgement to the drafter.

One row per team: the head coach the season opens with (Wikipedia's list of current head coaches, ``src.ingest.wiki_coaches``),
his start date, whether he is new to the team, his career style (league-relative: ``star`` / ``top5`` minutes, rotation
``depth10``, ``pace``, share of minutes to players 31+ (``old``) and 23 and under (``young``); positive = more than the league
average), the team's own style last season, and the **expected shift** a coaching change implies (the coach's earlier style
minus the team's last, damped by how many seasons stand behind him). A first-time head coach has no style to import, so only
"new" is shown. Which axes really follow a coach is measured in ``python -m src.backtest.coach_report``: star and top-five minutes,
rotation depth and three-point share do (star minutes probably); pace and the youth and veteran minute shares do not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.contracts import HISTORY_TABLES, History, season_start
from src.features.coach import CoachContext

SHOW = ("star", "top5", "depth10", "pace", "old", "young")   # shown for context; pace is shown but is not a coach trait here


def _previous(ctx: CoachContext, team_id: int, s: int) -> str | None:
    """The coach the team finished last season with, marked when he was only interim or acting (Portland's Billups was on leave and
    Splitter acted; New Orleans finished with an interim)."""
    key = ctx.closing_of.get((team_id, s - 1))
    if key is None:
        return None
    status = ctx.closing_status.get((team_id, s - 1), "")
    return ctx.name_of.get(key, key) + (f" ({status})" if status in ("interim", "acting") else "")


def coach_table(history: History, team_coaches: pd.DataFrame) -> pd.DataFrame:
    """One row per team for ``history.target_season`` (see the module docstring)."""
    s = season_start(history.target_season)
    ctx = CoachContext.build(history, team_coaches)
    abbr = history.game_logs.drop_duplicates("team_id").set_index("team_id")["team_abbr"]
    cur = team_coaches[(team_coaches["season"] == history.target_season) & team_coaches["is_opening"]].set_index("team_id")
    rows = []
    for team_id, r in cur.iterrows():
        team_id = int(team_id)
        key = r["coach_key"]
        prior = ctx.prior.get((key, s))
        prev = ctx.prev_style.get((team_id, s))
        new = ctx.is_new(team_id, s)
        shift = ctx.shift(team_id, s)
        row = {"team_id": team_id, "team": abbr.get(team_id, str(team_id)), "coach": r["coach_name"],
               "since": r["start_date"], "new_to_team": new, "previous_coach": _previous(ctx, team_id, s),
               "prior_seasons": prior["n"] if prior else 0, "first_time_head_coach": bool(new and not prior)}
        for c in SHOW:
            row[f"career_{c}"] = prior[c] if prior else float("nan")
            row[f"last_{c}"] = prev[c] if prev else float("nan")
            row[f"shift_{c}"] = shift.get(c, 0.0) if new else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["new_to_team", "team"], ascending=[False, True]).reset_index(drop=True)


def affected_players(table: pd.DataFrame, board: pd.DataFrame, roster: pd.DataFrame, *, top: int = 150) -> pd.DataFrame:
    """Ranked players (board rank <= ``top``) whose current team has a new head coach, with the coach's headline tendencies."""
    new = table[table["new_to_team"]][["team_id", "team", "coach", "previous_coach", "first_time_head_coach", "shift_top5", "shift_depth10", "shift_pace"]]
    r = roster[["player_id", "team_id"]].drop_duplicates("player_id")
    b = board[board["rank"] <= top][["player_id", "rank", "name", "position"]]
    out = b.merge(r, on="player_id").merge(new, on="team_id")
    return out.sort_values("rank").reset_index(drop=True)


def describe(row: pd.Series) -> str:
    """One sentence for the app and the CLI: what a coaching change implies, in plain words."""
    if row["first_time_head_coach"]:
        return "first-time head coach: no head-coaching history to import"
    parts = []
    if row["shift_top5"] > 0.5:
        parts.append(f"tighter rotation (top five +{row['shift_top5']:.1f} min)")
    elif row["shift_top5"] < -0.5:
        parts.append(f"more spread minutes (top five {row['shift_top5']:.1f} min)")
    if row["shift_depth10"] > 0.15:
        parts.append(f"deeper rotation (+{row['shift_depth10']:.1f} players over 10 min)")
    elif row["shift_depth10"] < -0.15:
        parts.append(f"shorter rotation ({row['shift_depth10']:.1f} players over 10 min)")
    return "; ".join(parts) or "no large expected change in how minutes are distributed"


def load(season: str, data_dir: Path | None = None) -> tuple[History, pd.DataFrame]:
    from src.ingest.wiki_coaches import read_team_coaches
    from src.store import load_tables

    tables = load_tables(HISTORY_TABLES, base=data_dir)
    return History.until(tables, season), read_team_coaches(data_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.value.coaches", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default="2026-27")
    p.add_argument("--board-csv", type=Path, default=None, help="also list ranked players on teams with a new head coach")
    p.add_argument("--all", action="store_true", help="every team, not only the ones with a new head coach")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        history, coaches = load(args.season, args.data_dir)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    table = coach_table(history, coaches)
    shown = table if args.all else table[table["new_to_team"]]
    print(f"Head coaches for {args.season} ({int(table['new_to_team'].sum())} of {len(table)} teams have a new one)\n")
    for r in shown.itertuples(index=False):
        row = pd.Series(r._asdict())
        tag = "NEW " if r.new_to_team else "    "
        since = "" if pd.isna(r.since) else f" (since {r.since:%Y-%m-%d})"
        prev = f", replacing {r.previous_coach}" if r.new_to_team and r.previous_coach else ""
        print(f"{tag}{r.team:<4} {r.coach}{since}{prev}")
        if r.new_to_team:
            print(f"       {describe(row)}")
        if not pd.isna(r.career_top5):
            print("       career vs league: " + ", ".join(f"{c} {getattr(r, f'career_{c}'):+.2f}" for c in SHOW))
    if args.board_csv is not None:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        board = pd.read_csv(args.board_csv)
        roster = latest_snapshot(read_roster_snapshots(args.data_dir))
        aff = affected_players(table, board, roster)
        print(f"\nRanked players (top 150) under a new head coach: {len(aff)}")
        for r in aff.itertuples(index=False):
            print(f"  #{r.rank:<4} {r.name} ({r.position}) {r.team}: {r.coach}" + (" [first-time HC]" if r.first_time_head_coach else ""))
    print("\nDescriptive only: the same information as a projection input did not help (ADR 0020). "
          "Style axes that follow a coach: top-five minutes, rotation depth, three-point share (star minutes probably); not pace, youth or veteran share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
