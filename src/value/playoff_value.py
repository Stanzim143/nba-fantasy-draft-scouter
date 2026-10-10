"""Advisory fantasy-playoff value for a board (ADR 0039). Projections and ranks never change.

    from src.value.playoff_value import compute_playoff_value, attach_playoff_value
    overlay, notes = compute_playoff_value(board, history, season, data_dir, season_games=82.0)
    board = attach_playoff_value(board, overlay)

The league's fantasy playoffs are its last ``sum(playoff_rounds_weeks)`` matchup weeks (``config/league.yaml``). Which calendar days
those are comes from the league's matchup calendar (:func:`src.inseason.schedule.build_calendar`: ESPN's periods when real, else the
documented derivation in :mod:`src.inseason.weeks`), and each NBA team's games in those weeks from the ``schedule_games`` table (ESPN's
pro schedule), so nothing here builds a second calendar. A player's playoff games are his team's games in those weeks; his team is the
latest roster snapshot of the target season, else the team he ended last season on (as the draft-day risk overlay resolves it).

Columns added by :func:`attach_playoff_value` (NaN where the team or the schedule is unknown):

``playoff_team``    NBA abbreviation of the team used
``playoff_games``   that team's games in the league's playoff weeks
``playoff_avail``   per-game availability: ``proj_gp / season_games``, clipped to [0, 1]. This is the model's own expected share of
                    team games played (it already carries the hurdle model's P(appear) and, for rookies, ADR 0038's), spread
                    evenly over the season; no extra haircut. ADR 0039's late-season study tests whether the playoff weeks
                    deserve a different rate.
``playoff_fp``      ``proj_fppg * playoff_games * playoff_avail``: expected fantasy points in the playoff weeks

The board's ``proj_fppg``, ``proj_gp`` and ``season_games`` are read, never written; ``proj_total_fp``, ``vorp``, ``rank`` and ``tier``
are untouched, and so is every other column.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import History

PLAYOFF_COLUMNS = ["playoff_team", "playoff_games", "playoff_avail", "playoff_fp"]


def playoff_team_games(sched: pd.DataFrame, cal) -> pd.Series:
    """``team_id -> games in the calendar's playoff weeks`` (teams with none get 0)."""
    from src.inseason.schedule import weekly_team_table

    weekly = weekly_team_table(sched, cal)
    if weekly.empty:
        return pd.Series(dtype=float)
    teams = sorted(set(weekly["team_id"]))
    p = weekly[weekly["kind"] == "playoff"]
    return p.groupby("team_id")["games"].sum().reindex(teams, fill_value=0).astype(float)


def per_game_availability(proj_gp, season_games: float) -> np.ndarray:
    """The model's expected share of team games played: ``proj_gp / season_games`` in [0, 1] (NaN stays NaN)."""
    g = np.asarray(proj_gp, dtype=float)
    if not season_games or season_games <= 0:
        return np.full(g.shape, np.nan)
    return np.clip(g / float(season_games), 0.0, 1.0)


def playoff_overlay(board: pd.DataFrame, team_of: pd.Series, team_games: pd.Series, *, season_games: float) -> pd.DataFrame:
    """The overlay (``player_id`` + :data:`PLAYOFF_COLUMNS`) from a resolved ``player_id -> team_id`` and ``team_id -> playoff games``."""
    from src.inseason.schedule import ABBR_OF_TEAM_ID

    ids = board["player_id"].to_numpy("int64")
    team = pd.Series(team_of, dtype="float").reindex(ids).to_numpy()
    games = pd.Series(team).map(team_games).to_numpy(dtype=float) if len(team_games) else np.full(len(ids), np.nan)
    avail = per_game_availability(board["proj_gp"].to_numpy(float), season_games)
    fppg = board["proj_fppg"].to_numpy(float)
    return pd.DataFrame({
        "player_id": ids,
        "playoff_team": [ABBR_OF_TEAM_ID.get(int(t), str(int(t))) if np.isfinite(t) else "" for t in team],
        "playoff_games": games,
        "playoff_avail": avail,
        "playoff_fp": fppg * games * avail,
    })


def current_teams(history: History, season: str, data_dir: Path | None) -> tuple[pd.Series, str | None]:
    """``player_id -> team_id``: the latest roster snapshot of ``season`` over last season's team (the risk overlay's rule)."""
    from src.value.breakouts import last_season_team

    team = last_season_team(history.player_season_bio) if len(history.player_season_bio) else pd.Series(dtype="int64")
    try:
        from src.ingest import nba_incoming

        snap = nba_incoming.read_roster_snapshots(data_dir)
        cur = snap[snap["season"] == season]
        roster = nba_incoming.latest_snapshot(cur) if len(cur) else None
    except (FileNotFoundError, KeyError, ValueError):
        roster = None
    if roster is None or roster.empty:
        return team, f"playoff value: no roster snapshot for {season}, so each player's team is the one he ended last season on"
    return roster.drop_duplicates("player_id").set_index("player_id")["team_id"].combine_first(team), None


def _schedule(season: str, data_dir: Path | None) -> pd.DataFrame | None:
    from src.contracts import ContractError
    from src.inseason.schedule import ScheduleError, from_team_games, read_schedule

    try:
        sched = read_schedule(data_dir, season)
        if len(sched):
            return sched
    except (FileNotFoundError, ContractError):
        pass
    try:   # a finished season's realised schedule (backtests, viewing an old board)
        from src.store import read_table

        return from_team_games(read_table("team_games", data_dir), season)
    except (FileNotFoundError, ScheduleError, KeyError, ContractError):
        return None


def compute_playoff_value(board: pd.DataFrame, history: History, season: str, data_dir: Path | None = None, *,
                          season_games: float = 82.0, league_id: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The overlay for every board player and notes on what was missing; never fails on missing inputs."""
    from src.inseason.schedule import build_calendar
    from src.value.league import load_league

    notes: list[str] = []
    empty = playoff_overlay(board, pd.Series(dtype=float), pd.Series(dtype=float), season_games=season_games)
    sched = _schedule(season, data_dir)
    if sched is None or sched.empty:
        notes.append(f"playoff value unavailable: no schedule for {season} (python -m src.inseason.schedule --season {season} --ingest)")
        return empty, notes
    if league_id is None:
        raw = os.environ.get("ESPN_LEAGUE_ID", "").strip()
        league_id = int(raw) if raw.isdigit() else None
    cal = build_calendar(sched, season, league_id=league_id, cfg=load_league(), data_root=data_dir)
    po = [w for w in cal.weeks if w.kind == "playoff"]
    if not po:
        notes.append("playoff value unavailable: the matchup calendar has no playoff weeks")
        return empty, notes
    tg = playoff_team_games(sched, cal)
    team_of, note = current_teams(history, season, data_dir)
    if note:
        notes.append(note)
    out = playoff_overlay(board, team_of, tg, season_games=season_games)
    notes.append(f"playoff value (advisory, not in the rank): weeks {po[0].week}-{po[-1].week}, {po[0].start} to {po[-1].end} "
                 f"({cal.source} calendar); team games {int(tg.min())}-{int(tg.max())}, availability = proj_gp / {season_games:g} (ADR 0039)")
    return out, notes


def attach_playoff_value(board: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """``board`` with the overlay columns added; row order and every existing column unchanged."""
    o = overlay.drop_duplicates("player_id").set_index("player_id").reindex(board["player_id"].to_numpy("int64"))
    out = board.copy()
    out["playoff_team"] = o["playoff_team"].fillna("").to_numpy(object)
    for c in ("playoff_games", "playoff_avail", "playoff_fp"):
        out[c] = o[c].to_numpy(float)
    out.attrs = dict(board.attrs)
    return out
