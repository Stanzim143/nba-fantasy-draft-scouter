"""Load-management PROXY features derived purely from ``History`` (ADR 0024).

There is no labelled reason for a missed game anywhere in the data (no rest / injury / suspension / personal code): ``game_logs`` has
only the games a player appeared in and ``team_games`` the team's schedule, so "missed" is ``team_games`` minus ``game_logs``. Load
management therefore cannot be observed, only proxied by the *shape* of the absences. Definitions (fixed in ADR 0024 before any
outcome was computed), all on the player's primary team's schedule in date order (the team he played most for; a traded player is a
documented approximation, ADR 0006):

* **interior run**: a run of consecutive missed team games that starts after game 1 and ends before the last game, i.e. the player
  played the game before and the game after (a minor knock or a rest day, not a lead block, a season-ender or a trade artefact).
* **isolated absence**: a game inside an interior run of length ``<= ISO_MAX`` (2). ``iso_n`` counts them; ``iso_n82`` rescales to an
  82-game schedule (2019-20 and 2020-21 were shorter); ``iso_frac`` = ``iso_n / missed``.
* **rest-like absence**: an interior run of length exactly 1 whose missed game is the second night of a back-to-back (the team's
  previous game was the day before). ``rest_n`` counts them.
* **near-65**: ``65 <= gp <= 67`` in a season of at least 80 team games (the 65-game award-eligibility threshold, 2023-24 onward); a
  descriptive marker, 0 in short seasons.

These are proxies: an interior one-game absence on a back-to-back is also what a minor injury looks like. The module says so and
labels no player as "resting"; ``src.backtest.load_report`` measures whether the proxies carry information beyond the baseline.

Everything is a pure function of ``History.game_logs`` / ``History.team_games`` (sliced to seasons before the target by
``History.until``): point-in-time by construction, like ``src.features.injury``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import History, season_start
from src.models.panel import season_starts

ISO_MAX = 2                 # a "short" absence: at most this many consecutive missed games
NEAR65_LO, NEAR65_HI = 65, 67
NEAR65_MIN_L = 80           # near-65 only makes sense on a (nearly) full schedule
FULL_SEASON = 82.0

LM_COLS = ["player_id", "s", "team_id", "n_teams", "L", "gp", "mpg", "missed", "iso_n", "iso_runs", "rest_n", "iso_n82", "rest_n82",
           "iso_frac", "near65"]


def _runs(missed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, lengths) of each run of consecutive True in a boolean array."""
    if missed.size == 0 or not missed.any():
        return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.concatenate(([0], missed.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def absence_shape(present: np.ndarray, back_to_back: np.ndarray) -> tuple[int, int, int]:
    """``(iso_n, iso_runs, rest_n)`` for one player-season.

    ``present[i]`` = the player appeared in the team's i-th game; ``back_to_back[i]`` = that game was the second night of a
    back-to-back (the previous game was played the day before; False for the first game of the season).
    """
    L = len(present)
    starts, lens = _runs(~present)
    interior = (starts > 0) & (starts + lens < L)
    short = interior & (lens <= ISO_MAX)
    rest = interior & (lens == 1)
    rest_n = int(back_to_back[starts[rest]].sum()) if rest.any() else 0
    return int(lens[short].sum()), int(short.sum()), rest_n


def load_profiles(history: History) -> pd.DataFrame:
    """One row per (player_id, s) of the absence-shape proxies, from ``history`` alone. See the module docstring for the columns."""
    gl, tg = history.game_logs, history.team_games
    if gl.empty or tg.empty:
        return pd.DataFrame(columns=LM_COLS)
    gl = gl[["season", "game_id", "player_id", "team_id", "min"]].drop_duplicates(["player_id", "game_id", "team_id"]).copy()
    gl["s"] = season_starts(gl["season"])
    tg = tg[["season", "game_id", "game_date", "team_id"]].copy()
    tg["s"] = season_starts(tg["season"])
    tg = tg.sort_values(["team_id", "s", "game_date", "game_id"], kind="mergesort")
    tg["idx"] = tg.groupby(["team_id", "s"]).cumcount()
    gap = tg.groupby(["team_id", "s"])["game_date"].diff().dt.days
    tg["b2b"] = (gap == 1).to_numpy()
    sched = {k: g["b2b"].to_numpy() for k, g in tg.groupby(["team_id", "s"], sort=False)}

    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    n_teams = counts.groupby(["player_id", "s"]).size().rename("n_teams")
    primary = (counts.sort_values(["player_id", "s", "n", "team_id"], ascending=[True, True, False, True])
                     .drop_duplicates(["player_id", "s"]))
    mpg = gl.groupby(["player_id", "s"])["min"].mean()
    mine = gl.merge(primary[["player_id", "s", "team_id"]], on=["player_id", "s", "team_id"]) \
             .merge(tg[["team_id", "s", "game_id", "idx"]], on=["team_id", "s", "game_id"], how="inner")
    rows = []
    for (pid, s, team), g in mine.groupby(["player_id", "s", "team_id"], sort=True):
        b2b = sched[(team, s)]
        L = len(b2b)
        present = np.zeros(L, bool)
        present[g["idx"].to_numpy()] = True
        gp = int(present.sum())
        iso_n, iso_runs, rest_n = absence_shape(present, b2b)
        missed = L - gp
        scale = FULL_SEASON / L if L else 0.0
        rows.append((int(pid), int(s), int(team), int(n_teams.loc[(pid, s)]), L, gp, float(mpg.loc[(pid, s)]), missed, iso_n, iso_runs,
                     rest_n, iso_n * scale, rest_n * scale, iso_n / missed if missed else 0.0,
                     bool(NEAR65_LO <= gp <= NEAR65_HI and L >= NEAR65_MIN_L)))
    return pd.DataFrame(rows, columns=LM_COLS)


def prior_load_features(history: History) -> pd.DataFrame:
    """Prior-season (``target - 1``) proxies of every player who appeared then, keyed on ``player_id``, from ``history`` alone.

    Only the prior season's games are profiled (earlier seasons contribute only the veteran flag), so this is cheap and cannot see
    the target season.
    """
    if history.game_logs.empty or history.team_games.empty:
        return pd.DataFrame(columns=[c for c in LM_COLS if c != "s"] + ["veteran", "f"])
    import dataclasses
    from src.backtest.return_report import veteran_flags
    prior = season_start(history.target_season) - 1
    on_prior = lambda df: df["season"].map(season_start) == prior  # noqa: E731
    last = dataclasses.replace(history, game_logs=history.game_logs[on_prior(history.game_logs)],
                               team_games=history.team_games[on_prior(history.team_games)])
    prof = load_profiles(last)
    prof = prof[prof["s"] == prior].copy()
    prof["veteran"] = veteran_flags(history, prof["player_id"].to_numpy(), prof["s"].to_numpy())
    prof["f"] = prof["gp"] / prof["L"]
    return prof.drop(columns="s").reset_index(drop=True)
