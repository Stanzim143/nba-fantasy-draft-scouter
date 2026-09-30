"""Shared helpers for the model tests (imported by name; tests/ has no __init__.py files)."""
from __future__ import annotations

import pandas as pd

from src.contracts import History
from src.synthetic import make_synthetic_tables

TARGET = "2018-19"
LEAGUE_KW = dict(first_start=2012, last_start=2019, n_teams=12, games_per_team=60, seed=5)


def make_league(**overrides) -> dict[str, pd.DataFrame]:
    return make_synthetic_tables(**{**LEAGUE_KW, **overrides})


def history_for(tables, target: str = TARGET) -> History:
    return History.until(tables, target)


def box_line(minutes: float, *, pts_parts=(3, 1, 1), reb=2, ast=1, stl=0, blk=0, tov=1, fga=6, fta=2, fg3a=2, pf=1):
    """A box score obeying the contract identities. pts_parts = (fg2m, fg3m, ftm)."""
    fg2m, fg3m, ftm = pts_parts
    fgm = fg2m + fg3m
    return dict(min=minutes, fgm=fgm, fga=max(fga, fgm), fg3m=fg3m, fg3a=max(fg3a, fg3m), ftm=ftm, fta=max(fta, ftm),
                oreb=0, dreb=reb, reb=reb, ast=ast, stl=stl, blk=blk, tov=tov, pf=pf,
                pts=2 * fgm + fg3m + ftm)


def add_player_seasons(tables, player_id: int, name: str, seasons: dict[str, tuple[int, dict]], *,
                       team_of_season=None, birthdate="1996-03-01", position="F", draft_year=None) -> dict:
    """Return new tables with an extra player.

    ``seasons`` maps season -> (games, box-line kwargs for box_line). The player gets one row per game
    on dates/ids that exist in that season's ``team_games`` for a fixed team, so contract checks pass.
    """
    t = {k: v.copy() for k, v in tables.items()}
    rows, bio = [], []
    for season, (games, kw) in seasons.items():
        tg = t["team_games"]
        team = tg[tg["season"] == season].groupby("team_id").size().index[0] if team_of_season is None else team_of_season
        sched = tg[(tg["season"] == season) & (tg["team_id"] == team)].sort_values("game_date").head(games)
        for _, g in sched.iterrows():
            rows.append(dict(season=season, game_id=g["game_id"], game_date=g["game_date"], player_id=player_id,
                             player_name=name, team_id=int(team), team_abbr=g["team_abbr"], matchup=f"{g['team_abbr']} vs. X",
                             plus_minus=0.0, **box_line(**kw)))
        age = (pd.Timestamp(int(season[:4]), 10, 1) - pd.Timestamp(birthdate)).days / 365.25
        bio.append(dict(season=season, player_id=player_id, age_at_season_start=age, team_id=int(team)))
    gl = pd.concat([t["game_logs"], pd.DataFrame(rows)[t["game_logs"].columns]], ignore_index=True)
    for c in t["game_logs"].columns:
        gl[c] = gl[c].astype(t["game_logs"][c].dtype)
    t["game_logs"] = gl
    t["player_season_bio"] = pd.concat([t["player_season_bio"], pd.DataFrame(bio)[t["player_season_bio"].columns]], ignore_index=True)
    row = {c: None for c in t["players"].columns}
    row.update(player_id=player_id, player_name=name, birthdate=pd.Timestamp(birthdate), position=position,
               height_in=78.0, weight_lb=210.0, draft_year=draft_year, draft_round=1, draft_number=10,
               from_year=int(min(seasons)[:4]), to_year=int(max(seasons)[:4]))
    new = pd.DataFrame([row])
    for c in t["players"].columns:
        new[c] = new[c].astype(t["players"][c].dtype)
    t["players"] = pd.concat([t["players"], new], ignore_index=True)
    return t


def spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
