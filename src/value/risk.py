"""Attach the draft-day risk overlay to a board (ADR 0016, gap 4). Flags and advisory games only: projections unchanged.

    from src.value.risk import compute_risk, attach_risk
    overlay, notes = compute_risk(projections, board, history, season, data_dir)
    board = attach_risk(board, overlay)

Inputs, each optional (a missing one degrades to a note and fewer flags, never an error):

* the newest ``roster_snapshots`` day **of the target season** (who is on which team now);
* the newest ``espn_status_snapshots`` day (ESPN's current injury status; ``python -m src.ingest.espn_status``);
* the season's preseason box scores (``offseason_logs`` / ``offseason_team_games``), used as of today.

See ``src.features.risk`` for what is validated (preseason absence, measured in ``python -m src.backtest.preseason_availability``)
and what is an assumption (the ESPN status haircuts), and ``docs/offseason.md`` for how to read the columns.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import History, season_start
from src.features import risk as R

RISK_COLUMNS = ["risk_level", "risk_flags", "risk_gp_haircut", "risk_gp"]


def _team_names(history: History, projections: pd.DataFrame) -> pd.Series:
    n = projections.drop_duplicates("player_id").set_index("player_id")["player_name"]
    return n


def compute_risk(projections: pd.DataFrame, board: pd.DataFrame, history: History, season: str, data_dir: Path | None = None,
                 *, today: date | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The overlay frame (``player_id`` + ``RISK_COLUMNS`` + ``pre_flag`` etc.) and human-readable notes on what was missing."""
    from src.ingest import espn_status, nba_incoming, nba_offseason
    from src.value.breakouts import last_season_team

    notes: list[str] = []
    today_ts = pd.Timestamp(today or date.today())
    roster = None
    try:
        snap = nba_incoming.read_roster_snapshots(data_dir)
        cur = snap[snap["season"] == season]
        roster = nba_incoming.latest_snapshot(cur) if len(cur) else None
        if roster is None:
            notes.append(f"no roster snapshot for {season}: team-change flags skipped (run src.ingest.nba_incoming)")
    except FileNotFoundError:
        notes.append("no roster snapshots: team-change flags skipped (run src.ingest.nba_incoming)")
    status = None
    try:
        status = espn_status.latest_status(espn_status.read_status(data_dir))
    except FileNotFoundError:
        notes.append("no ESPN status archive: injury-status flags skipped (run src.ingest.espn_status)")
    pl_games = tm_games = None
    try:
        logs, tg = nba_offseason.read_offseason_logs(data_dir), nba_offseason.read_offseason_team_games(data_dir)
        pl_games, tm_games = R.preseason_usage(logs, tg, season, as_of=today_ts)
        if tm_games.empty:
            pl_games = tm_games = None
            notes.append("no preseason games yet: preseason-absence flags start once teams have played")
    except FileNotFoundError:
        notes.append("no offseason tables: preseason-absence flags skipped (run src.ingest.nba_offseason)")

    last_team = last_season_team(history.player_season_bio)
    ids = board["player_id"].to_numpy("int64")
    frames: dict[str, pd.DataFrame | None] = {"preseason": None, "status": None, "context": None}
    if pl_games is not None:
        team_of = last_team
        if roster is not None:
            team_of = roster.drop_duplicates("player_id").set_index("player_id")["team_id"].combine_first(last_team)
        p = projections.drop_duplicates("player_id").set_index("player_id").reindex(ids)
        frames["preseason"] = R.preseason_flags(ids, team_of,
                                                p["proj_mpg"].to_numpy(float),
                                                p["n_hist_seasons"].fillna(0).to_numpy() if "n_hist_seasons" in p else np.ones(len(ids)),
                                                pl_games, tm_games)
    if status is not None:
        frames["status"] = R.status_flags(ids, status, today=today_ts)
    if roster is not None:
        stars = set(board.loc[board["rank"] <= R.STAR_RANK, "player_id"].astype(int))
        frames["context"] = R.context_flags(roster, last_team, stars, _team_names(history, projections)).flags
    if frames["preseason"] is not None and roster is not None:
        # only players under contract right now: an absent free agent or retiree is not an injury
        on_roster = frames["preseason"]["player_id"].isin(roster["player_id"])
        frames["preseason"].loc[~on_roster, "pre_flag"] = ""
    overlay = R.build_overlay(board[["player_id", "proj_gp"]], preseason=frames["preseason"], status=frames["status"],
                              context=frames["context"], today=today_ts)
    return overlay[["player_id", *RISK_COLUMNS]], notes


def attach_risk(board: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """``board`` with the overlay columns merged in (row order and every existing column unchanged)."""
    out = board.merge(overlay, on="player_id", how="left")
    out["risk_flags"] = out["risk_flags"].fillna("")
    out["risk_level"] = out["risk_level"].fillna("")
    out.attrs = dict(board.attrs)
    return out


def season_is_live(season: str, today: date | None = None) -> bool:
    """True from August of the season's start year to the end of that year: the window in which a draft-day overlay means something."""
    d = today or date.today()
    return d.year == season_start(season) and d.month >= 8
