"""Shared helpers for the backtest tests (unique module name: tests have no __init__.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.benchmarks import stats_to_projection
from src.contracts import STAT_COLUMN_MAP, History, season_start
from src.value.league import load_league

SCORING = load_league()["scoring"]
SMALL = dict(first_start=2015, last_start=2018, n_teams=8, games_per_team=30, seed=1)
STAT_COLS = list(STAT_COLUMN_MAP.values())


def make_logs(spec: list[tuple], season: str = "2020-21", team_id: int = 1, start: str | None = None) -> pd.DataFrame:
    """Contract-valid game_logs from ``[(player_id, n_games, pts, reb, ast), ...]`` (pts must be even)."""
    rows = []
    d0 = pd.Timestamp(start or f"{season_start(season)}-10-22")
    for pid, n, pts, reb, ast in spec:
        for g in range(n):
            rows.append({
                "season": season, "game_id": f"{season}-{g:03d}-{team_id}", "game_date": d0 + pd.Timedelta(days=g),
                "player_id": pid, "player_name": f"P{pid}", "team_id": team_id, "team_abbr": f"T{team_id}",
                "matchup": f"T{team_id} vs. X", "min": 30.0,
                "fgm": pts // 2, "fga": pts // 2, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0,
                "oreb": 0, "dreb": reb, "reb": reb, "ast": ast, "stl": 0, "blk": 0, "tov": 0, "pf": 0,
                "pts": pts, "plus_minus": 0.0,
            })
    return pd.DataFrame(rows)


def oracle_stats(tables, season: str) -> pd.DataFrame:
    """Per-player realised per-game stat lines + GP for ``season`` (peeks at the future)."""
    gl = tables["game_logs"]
    gl = gl[gl["season"] == season]
    g = gl.groupby("player_id")
    st = g[STAT_COLS].mean().add_prefix("proj_")
    st["proj_gp"] = g["game_id"].count().astype(float)
    st["proj_mpg"] = g["min"].mean()
    st["player_name"] = g["player_name"].last()
    return st.reset_index()


class OracleProjector:
    """Cheats on purpose: holds a reference to the full tables and projects the truth."""
    name = "oracle"

    def __init__(self, tables, scoring=SCORING):
        self._tables, self._scoring = tables, scoring

    def project(self, history: History) -> pd.DataFrame:
        st = oracle_stats(self._tables, history.target_season)
        return stats_to_projection(st, season=history.target_season, model=self.name, scoring=self._scoring)


class NoiseProjector:
    """Oracle with a fixed per-player multiplicative log-normal error of scale ``sigma``."""

    def __init__(self, tables, sigma: float, seed: int = 0, scoring=SCORING):
        self._tables, self.sigma, self.seed, self._scoring = tables, sigma, seed, scoring
        self.name = f"noise_{sigma:g}"

    def project(self, history: History) -> pd.DataFrame:
        st = oracle_stats(self._tables, history.target_season).sort_values("player_id").reset_index(drop=True)
        z = np.random.default_rng([self.seed, season_start(history.target_season)]).standard_normal(len(st))
        f = np.exp(self.sigma * z)
        for c in [c for c in st.columns if c.startswith("proj_") and c not in ("proj_gp", "proj_mpg")]:
            st[c] = st[c] * f
        return stats_to_projection(st, season=history.target_season, model=self.name, scoring=self._scoring)


def frame_from_arrays(pred_total, actual_total, *, played=None, projected=None, pred_gp=None, actual_gp=None):
    """A minimal evaluation frame for metric tests."""
    n = len(pred_total)
    pred_total = np.asarray(pred_total, float)
    actual_total = np.asarray(actual_total, float)
    projected = ~np.isnan(pred_total) if projected is None else np.asarray(projected, bool)
    played = (~np.isnan(actual_total)) & (actual_total > 0) if played is None else np.asarray(played, bool)
    gp = np.where(played, 60.0, 0.0) if actual_gp is None else np.asarray(actual_gp, float)
    pg = np.full(n, 60.0) if pred_gp is None else np.asarray(pred_gp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({
            "season": "2020-21", "player_id": np.arange(1, n + 1), "player_name": [f"P{i}" for i in range(n)],
            "projected": projected, "played": played,
            "proj_gp": np.where(projected, pg, np.nan), "proj_mpg": np.nan,
            "proj_fppg": pred_total / pg, "proj_total_fp": pred_total,
            "fppg_p10": np.nan, "fppg_p50": np.nan, "fppg_p90": np.nan,
            "actual_gp": gp, "actual_min": gp * 30.0, "actual_mpg": np.where(gp > 0, 30.0, np.nan),
            "actual_fppg": np.where(gp > 0, actual_total / np.where(gp > 0, gp, 1), np.nan),
            "actual_total_fp": actual_total, "team_id": 1, "n_teams": 1,
            "band_games": 0, "n_below_p10": 0, "n_above_p90": 0, "p50_games": 0, "n_below_p50": 0,
        })
