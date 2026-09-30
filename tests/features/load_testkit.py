"""Test-only helper for the load-management tests: a one-team league where each player's played/missed pattern and the
back-to-back nights are exact."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import box_line  # noqa: E402

TEAM = 1610612737
L = 30


def _pat(missed=(), lead=0):
    a = np.ones(L, bool)
    a[list(missed)] = False
    a[:lead] = False
    return a


def _tables(patterns, *, b2b=(), seasons=("2017-18", "2018-19", "2019-20"), other_team=None):
    """One team, ``L`` games a season, two days apart except the games in ``b2b`` (played the day after the previous game)."""
    tg, gl = [], []
    for season in seasons:
        d = pd.Timestamp(int(season[:4]), 10, 20)
        for g in range(L):
            d = d + pd.Timedelta(days=1 if g in b2b else 2)
            tg.append(dict(season=season, game_id=f"{season}-{g:03d}", game_date=d, team_id=TEAM, team_abbr="AAA", is_home=True,
                           pts_for=100, pts_against=99))
    tg = pd.DataFrame(tg)
    for (pid, season), present in patterns.items():
        rows = tg[tg["season"] == season].reset_index(drop=True)
        for g in np.flatnonzero(present):
            r = rows.iloc[g]
            team = (other_team or {}).get((pid, season, g), TEAM)
            gl.append(dict(season=season, game_id=r["game_id"], game_date=r["game_date"], player_id=pid, player_name=f"P{pid}", team_id=team,
                           team_abbr="AAA", matchup="AAA vs. X", plus_minus=0.0, **box_line(28.0)))
    pids = sorted({p for p, _ in patterns})
    players = pd.DataFrame({"player_id": pids, "player_name": [f"P{p}" for p in pids], "birthdate": pd.Timestamp("1995-01-01"),
                            "position": "F", "height_in": 78.0, "weight_lb": 210.0, "draft_year": 2015, "draft_round": 1,
                            "draft_number": 5, "from_year": 2015, "to_year": 2030})
    bio = pd.DataFrame([dict(season=s, player_id=p, age_at_season_start=24.0, team_id=TEAM) for (p, s) in patterns])
    return {"game_logs": pd.DataFrame(gl), "team_games": tg, "players": players, "player_season_bio": bio}
