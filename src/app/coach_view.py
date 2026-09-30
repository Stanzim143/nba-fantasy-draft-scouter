"""Data prep for the app's Coaches tab (ADR 0020). No Streamlit import: plain functions, tested offline.

Reads the local ``team_coaches`` table and the game history (no network). Everything shown is descriptive; see ``src.value.coaches``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.value import coaches as V

TABLE_COLUMNS = ["team", "coach", "since", "new_to_team", "previous_coach", "prior_seasons", "career_star", "career_top5",
                 "career_depth10", "career_pace", "career_old", "career_young"]


class CoachesUnavailable(Exception):
    """No coach table or history yet; the message is safe to show verbatim."""


def load_table(season: str, data_dir: Path | None = None) -> pd.DataFrame:
    try:
        history, coaches = V.load(season, data_dir)
    except FileNotFoundError as exc:
        raise CoachesUnavailable(f"{exc}") from exc
    if not (coaches["season"] == season).any():
        raise CoachesUnavailable(f"no head coaches for {season} in team_coaches: run python -m src.ingest.wiki_coaches --current {season}")
    return V.coach_table(history, coaches)


def summary_lines(table: pd.DataFrame) -> list[str]:
    """One plain-English line per team with a new head coach."""
    out = []
    for r in table[table["new_to_team"]].itertuples(index=False):
        prev = f" (replacing {r.previous_coach})" if r.previous_coach else ""
        out.append(f"{r.team}: {r.coach}{prev}; " + V.describe(pd.Series(r._asdict())))
    return out


def affected(table: pd.DataFrame, board: pd.DataFrame, data_dir: Path | None = None, top: int = 150) -> pd.DataFrame:
    try:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        roster = latest_snapshot(read_roster_snapshots(data_dir))
    except (FileNotFoundError, OSError):
        return pd.DataFrame()
    return V.affected_players(table, board, roster, top=top)
