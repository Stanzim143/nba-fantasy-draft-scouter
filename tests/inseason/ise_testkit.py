"""Shared fixtures for the in-season tests: a small synthetic league whose last season is the 'live' one."""
from __future__ import annotations

import copy

import pandas as pd

from src.synthetic import make_synthetic_tables
from src.value.league import load_league

SEASON = "2023-24"
N_TEAMS = 14
GAMES = 40


def _cfg_13_teams() -> dict:
    """The synthetic world is sized for 13 rosters; pin it so a change to the real league size in
    config/league.yaml does not silently change the fixture world. Pass as ``load_context(cfg=CFG13)``."""
    cfg = copy.deepcopy(load_league())
    cfg["league"]["teams"] = 13
    return cfg


CFG13 = _cfg_13_teams()


def make_tables(seed: int = 1) -> dict:
    return dict(make_synthetic_tables(first_start=2020, last_start=2023, n_teams=N_TEAMS, games_per_team=GAMES, seed=seed))


def date_after_games(tables: dict, n: int) -> pd.Timestamp:
    """The date the median team had played ``n`` games in SEASON."""
    tg = tables["team_games"]
    tg = tg[tg["season"] == SEASON].sort_values(["team_id", "game_date"])
    d = tg.assign(k=tg.groupby("team_id").cumcount() + 1)
    return pd.Timestamp(d[d["k"] == n]["game_date"].median()).normalize()
