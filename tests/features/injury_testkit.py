"""Test-only helper: add a player-season with an EXACT played/missed game pattern.

Unlike ``tests.models.model_testkit.add_player_seasons`` (which always plays the first N games of
the season), this lets absence-streak tests construct known streak shapes -- e.g. "missed games
3-7" (one streak of 5) vs "missed games 1, 3, 5, 7, 9" (five streaks of 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import box_line  # noqa: E402


def add_player_season_pattern(tables, player_id: int, name: str, season: str, played: list[bool], *,
                              team_of_season=None, birthdate="1996-03-01", position="F", draft_year=None,
                              minutes: float = 30.0) -> dict:
    """``played[i]`` says whether the player appeared in that season's team schedule game i
    (games ordered by date). Returns new tables (input is not mutated)."""
    t = {k: v.copy() for k, v in tables.items()}
    tg = t["team_games"]
    team = tg[tg["season"] == season].groupby("team_id").size().index[0] if team_of_season is None else team_of_season
    sched = tg[(tg["season"] == season) & (tg["team_id"] == team)].sort_values("game_date").reset_index(drop=True)
    if len(played) > len(sched):
        raise ValueError(f"pattern has {len(played)} games but the schedule only has {len(sched)}")
    rows = []
    for i, p in enumerate(played):
        if not p:
            continue
        g = sched.iloc[i]
        rows.append(dict(season=season, game_id=g["game_id"], game_date=g["game_date"], player_id=player_id,
                         player_name=name, team_id=int(team), team_abbr=g["team_abbr"], matchup=f"{g['team_abbr']} vs. X",
                         plus_minus=0.0, **box_line(minutes)))
    gl = pd.concat([t["game_logs"], pd.DataFrame(rows, columns=[c for c in t["game_logs"].columns])], ignore_index=True) \
        if rows else t["game_logs"]
    for c in t["game_logs"].columns:
        gl[c] = gl[c].astype(t["game_logs"][c].dtype)
    t["game_logs"] = gl
    age = (pd.Timestamp(int(season[:4]), 10, 1) - pd.Timestamp(birthdate)).days / 365.25
    bio = pd.DataFrame([dict(season=season, player_id=player_id, age_at_season_start=age, team_id=int(team))])
    t["player_season_bio"] = pd.concat([t["player_season_bio"], bio[t["player_season_bio"].columns]], ignore_index=True)
    row = {c: None for c in t["players"].columns}
    row.update(player_id=player_id, player_name=name, birthdate=pd.Timestamp(birthdate), position=position,
               height_in=78.0, weight_lb=210.0, draft_year=draft_year, draft_round=1, draft_number=10,
               from_year=int(season[:4]), to_year=int(season[:4]))
    new = pd.DataFrame([row])
    for c in t["players"].columns:
        new[c] = new[c].astype(t["players"][c].dtype)
    t["players"] = pd.concat([t["players"], new], ignore_index=True)
    return t
