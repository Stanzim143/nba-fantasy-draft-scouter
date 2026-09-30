"""Player-season panel and lag helpers shared by every baseline component.

Everything here is a pure function of a ``History``; nothing touches the future. The panel has
one row per (player, season) with box-score totals, games played, age, and the game-level
fantasy-point mean/std. Lags are looked up by (player, season - k) so a missing season simply
contributes zero exposure (a player who sat out a year is not "carried" through it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.contracts import History, season_start
from src.value.frame import fantasy_points_frame
from src.value.positions import position_group

SUM_COLS = ("min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts")


def season_starts(seasons: pd.Series) -> np.ndarray:
    """Vectorised ``season_start`` (maps each unique string once)."""
    lut = {s: season_start(s) for s in pd.unique(seasons)}
    return seasons.map(lut).to_numpy(dtype="int64")


def canonical_logs(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row order, so float sums are bit-identical however the input was shuffled."""
    return game_logs.sort_values(["season", "player_id", "game_id"], kind="mergesort").reset_index(drop=True)


def season_lengths(history: History, panel: pd.DataFrame | None = None) -> pd.Series:
    """Games per team in each historical season (median over teams), indexed by season start."""
    tg = history.team_games
    if len(tg):
        s = season_starts(tg["season"])
        counts = pd.DataFrame({"s": s, "team_id": tg["team_id"].to_numpy()}).groupby(["s", "team_id"]).size()
        return counts.groupby("s").median().astype(float)
    if panel is None:
        raise ValueError("no team_games in history and no panel to infer season length from")
    return panel.groupby("s")["gp"].max().astype(float)


def target_season_games(season_len: pd.Series, n_recent: int = 3) -> int:
    """Expected schedule length of the target season.

    The largest of the last ``n_recent`` completed season lengths, so a shortened season
    (lockout, pandemic) does not drag next season's cap down. A cap, not a forecast of injuries.
    """
    return int(round(float(season_len.sort_index().iloc[-n_recent:].max())))


class AgeLookup:
    """Age (years, as of Oct 1 of the season start) for any player-season.

    Uses ``players.birthdate`` when known, else falls back to the last ``player_season_bio`` age
    rolled forward by season count, else NaN. Never reads anything dated after the history.
    """

    def __init__(self, players: pd.DataFrame, bio: pd.DataFrame):
        self._birth = players.drop_duplicates("player_id").set_index("player_id")["birthdate"]
        self._offset = pd.Series(dtype="float64")
        if len(bio):
            b = bio.assign(s=season_starts(bio["season"])).sort_values(["player_id", "s"])
            last = b.groupby("player_id").tail(1)
            self._offset = pd.Series((last["age_at_season_start"] - last["s"]).to_numpy(), index=last["player_id"].to_numpy())

    def at(self, player_ids, seasons) -> np.ndarray:
        pids = np.asarray(player_ids, dtype="int64")
        s = np.broadcast_to(np.asarray(seasons, dtype="int64"), pids.shape)
        birth = pd.to_datetime(self._birth.reindex(pids).to_numpy())
        oct1 = pd.to_datetime(pd.Series(s).astype(str) + "-10-01").to_numpy()
        age = (oct1 - np.asarray(birth, dtype="datetime64[ns]")) / np.timedelta64(1, "D") / 365.25
        age = np.asarray(age, dtype="float64")
        fallback = self._offset.reindex(pids).to_numpy(dtype="float64") + s
        return np.where(np.isnan(age), fallback, age)


@dataclass
class PanelData:
    df: pd.DataFrame          # player-season rows
    season_len: pd.Series     # games per team by season start
    ages: AgeLookup
    game_fp: pd.DataFrame     # player_id, s, fp: game-level fantasy points (for volatility)


def build_panel(history: History, scoring: Mapping[str, float]) -> PanelData:
    """Player-season panel, season lengths, the age lookup and game-level FP.

    Panel columns: player_id, s (season start year), gp, totals (SUM_COLS), mpg, fp_sum, fppg,
    fp_sd (std of game-level FP, NaN for a single game), season_len, f (gp / season_len), age,
    pos_group.
    """
    gl = canonical_logs(history.game_logs)
    if gl.empty:
        raise ValueError("history has no game logs; nothing to project from")
    gl["s"] = season_starts(gl["season"])
    gl["fp"] = fantasy_points_frame(gl, scoring)
    agg = gl.groupby(["player_id", "s"], sort=True).agg(
        gp=("game_id", "size"),
        **{c: (c, "sum") for c in SUM_COLS},
        fp_sum=("fp", "sum"),
        fp_sd=("fp", "std"),
    ).reset_index()
    for c in SUM_COLS:
        agg[c] = agg[c].astype("float64")
    slen = season_lengths(history, agg)
    agg["season_len"] = agg["s"].map(slen).astype("float64")
    agg["mpg"] = agg["min"] / agg["gp"]
    agg["fppg"] = agg["fp_sum"] / agg["gp"]
    agg["f"] = np.minimum(agg["gp"] / agg["season_len"], 1.0)
    ages = AgeLookup(history.players, history.player_season_bio)
    agg["age"] = ages.at(agg["player_id"].to_numpy(), agg["s"].to_numpy())
    pos = history.players.drop_duplicates("player_id").set_index("player_id")["position"]
    agg["pos_group"] = [position_group(p) for p in pos.reindex(agg["player_id"]).to_numpy()]
    return PanelData(agg, slen, ages, gl[["player_id", "s", "fp"]].copy())


def index_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.set_index(["player_id", "s"]).sort_index()


def lag_matrix(pidx: pd.DataFrame, pids: np.ndarray, target_s: np.ndarray, n_lags: int, cols: list[str]) -> dict[str, np.ndarray]:
    """``out[col][i, k-1]`` = ``pidx[col]`` for (pids[i], target_s[i] - k); NaN where that season is absent."""
    n = len(pids)
    out = {c: np.full((n, n_lags), np.nan) for c in cols}
    for k in range(1, n_lags + 1):
        got = pidx.reindex(pd.MultiIndex.from_arrays([pids, target_s - k]))
        for c in cols:
            out[c][:, k - 1] = got[c].to_numpy(dtype="float64")
    return out
