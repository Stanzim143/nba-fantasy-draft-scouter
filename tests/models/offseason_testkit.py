"""Synthetic Summer League / preseason tables for the offseason-layer tests (imported by name).

``make_offseason(tables, mode)`` builds ``offseason_logs`` / ``offseason_team_games`` from the synthetic
league's own game logs:

* ``"signal"``: each player-season's offseason lines are copies of that season's *actual* box scores, with
  minutes proportional to his actual minutes. The offseason evidence therefore genuinely predicts the
  regular season, so a working layer must find it (a planted, known effect).
* ``"noise"``: the same lines, but each player is handed another player's lines, so there is no relationship
  to his own season. A layer with an honest gate must switch itself off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.ingest.nba_offseason import (
    OFFSEASON_LOGS_COLUMNS, OFFSEASON_TEAM_GAMES_COLUMNS, previous_season, validate_offseason_logs,
    validate_offseason_team_games,
)

BOX = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]


def make_offseason(tables, mode: str = "signal", *, seed: int = 0, sl_share: float = 0.35,
                   min_games: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    gl = tables["game_logs"]
    logs = []
    game_no = 0
    for season, g in gl.groupby("season"):
        y = season_start(season)
        ev = f"{y}-{(y + 1) % 100:02d}"
        per_player = {pid: d for pid, d in g.groupby("player_id") if len(d) >= min_games}
        pids = np.array(sorted(per_player))
        donors = pids if mode == "signal" else rng.permutation(pids)
        team_game_id: dict[tuple[str, int, int], str] = {}
        for pid, donor in zip(pids, donors):
            own = per_player[pid]
            src = per_player[donor]
            team = int(own["team_id"].iloc[0])
            abbr = str(own["team_abbr"].iloc[0])
            contexts = [("preseason", 3)]
            if rng.random() < sl_share:
                contexts.append(("summer_league", 2))
            for ctx, n in contexts:
                for k in range(n):
                    row = src.iloc[int(rng.integers(0, len(src)))]
                    key = (ctx, team, k)
                    if key not in team_game_id:
                        game_no += 1
                        prefix = "152" if ctx == "summer_league" else "001"
                        team_game_id[key] = f"{prefix}{y % 100:02d}{game_no:05d}"
                    date = pd.Timestamp(y, 7, 8 + 2 * k) if ctx == "summer_league" else pd.Timestamp(y, 10, 2 + 2 * k)
                    rec = {c: row[c] for c in BOX}
                    rec.update(season=previous_season(ev), event_season=ev, context=ctx, game_id=team_game_id[key],
                               game_date=date, player_id=int(pid), player_name=str(own["player_name"].iloc[0]),
                               team_id=team, team_abbr=abbr, matchup=f"{abbr} vs. ZZZ",
                               min=float(row["min"]) * float(rng.uniform(0.85, 1.15)), plus_minus=0.0)
                    logs.append(rec)
    df = pd.DataFrame(logs)
    for c, kind in OFFSEASON_LOGS_COLUMNS.items():
        if kind == "int":
            df[c] = df[c].astype("int64")
    df = df[list(OFFSEASON_LOGS_COLUMNS)].sort_values(["game_date", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    validate_offseason_logs(df)

    tg = (df.groupby(["season", "event_season", "context", "game_id", "game_date", "team_id", "team_abbr"])["pts"]
          .sum().rename("pts_for").reset_index())
    tg["is_home"] = True
    tg["pts_against"] = pd.array([pd.NA] * len(tg), dtype="Int64")
    tg["pts_for"] = pd.array(tg["pts_for"], dtype="Int64")
    tg = tg[list(OFFSEASON_TEAM_GAMES_COLUMNS)].reset_index(drop=True)
    validate_offseason_team_games(tg)
    return df, tg
