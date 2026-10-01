"""Realised fantasy results per player-season, and the evaluation universe.

Evaluation-universe rules (also documented in docs/backtest.md):

1. **Projected and played**  -> compared normally.
2. **Projected but played 0 games** -> counted with ``actual_gp = 0`` and
   ``actual_total_fp = 0``.  They are NOT dropped: a model that projects an injured
   player at 30 FPPG x 70 GP must pay for it.  ``actual_fppg`` is undefined (NaN) for them, so
   FPPG-based metrics exclude them while total-FP, GP and rank metrics include them.
3. **Played but not projected** -> a *coverage miss*.  They are kept in the frame with NaN
   predictions, are never selectable as a predicted top-K player, can occupy the *actual*
   top-K (lowering the achievable hit rate), and are reported explicitly (count, share of
   league fantasy points, how many sit in the actual top-100).

Actuals use the league scoring from ``config/league.yaml`` via ``fantasy_points_frame`` so the
target is exactly what the league pays out.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.backtest.errors import BacktestError
from src.contracts import season_start
from src.value.frame import fantasy_points_frame

ACTUAL_COLUMNS = ["season", "player_id", "player_name", "team_id", "n_teams",
                  "gp", "minutes", "mpg", "total_fp", "fppg"]

PRED_COLUMNS = ["proj_gp", "proj_mpg", "proj_fppg", "proj_total_fp",
                "fppg_p10", "fppg_p50", "fppg_p90"]

# Extra projection columns some models emit (ADR 0031/0033): carried into the eval frame when present so the
# appearance and season-interval checks can score them. They are never required.
OPTIONAL_PRED_COLUMNS = ["proj_p_appear", "proj_gp_p10", "proj_gp_p90", "proj_gp_sd",
                         "proj_total_fp_p10", "proj_total_fp_p50", "proj_total_fp_p90"]


def _season_rows(game_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    season_start(season)  # validates the format
    rows = game_logs[game_logs["season"] == season]
    if rows.empty:
        raise BacktestError(
            f"no game_logs rows for season {season}; cannot score a backtest for it "
            f"(available: {sorted(game_logs['season'].unique())[:3]}...)")
    return rows


def game_fp(game_logs: pd.DataFrame, season: str, scoring: Mapping[str, float]) -> pd.DataFrame:
    """One row per game played in ``season``: ``player_id, game_date, fp`` (league scoring)."""
    rows = _season_rows(game_logs, season)
    return pd.DataFrame({
        "player_id": rows["player_id"].to_numpy(),
        "game_date": rows["game_date"].to_numpy(),
        "fp": fantasy_points_frame(rows, scoring).to_numpy(),
    })


def season_actuals(game_logs: pd.DataFrame, season: str, scoring: Mapping[str, float]) -> pd.DataFrame:
    """Per player-season realised results.

    Columns: ``season, player_id, player_name, team_id`` (team of the player's last game that
    season), ``n_teams`` (distinct teams played for), ``gp`` (games played), ``minutes`` (total),
    ``mpg``, ``total_fp``, ``fppg`` (= total_fp / gp).  Only players with >= 1 game appear
    (the contract has no row for a game not played); ``build_eval_frame`` adds the zero-game
    projected players.
    """
    rows = _season_rows(game_logs, season).copy()
    rows["fp"] = fantasy_points_frame(rows, scoring).to_numpy()
    rows = rows.sort_values(["player_id", "game_date", "game_id"], kind="stable")
    g = rows.groupby("player_id", sort=True)
    out = pd.DataFrame({
        "player_name": g["player_name"].last(),
        "team_id": g["team_id"].last(),
        "n_teams": g["team_id"].nunique(),
        "gp": g["game_id"].count(),
        "minutes": g["min"].sum(),
        "total_fp": g["fp"].sum(),
    }).reset_index()
    out["mpg"] = out["minutes"] / out["gp"]
    out["fppg"] = out["total_fp"] / out["gp"]
    out.insert(0, "season", season)
    out["gp"] = out["gp"].astype("int64")
    return out[ACTUAL_COLUMNS]


def band_counts(games: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """Per player: how many of his actual games fell below p10 / p50 and above p90.

    ``fppg_p10/p50/p90`` are *game-level* quantiles (see the projections contract), so
    calibration must be judged on game-level outcomes, not on the season mean.  Returns
    ``player_id, band_games, n_below_p10, n_above_p90, p50_games, n_below_p50``; the count
    columns are 0 for players whose band is missing (they contribute nothing).
    """
    bands = projections[["player_id", "fppg_p10", "fppg_p50", "fppg_p90"]]
    m = games.merge(bands, on="player_id", how="inner")
    has_band = m["fppg_p10"].notna() & m["fppg_p90"].notna()
    has_p50 = m["fppg_p50"].notna()
    m["band_games"] = has_band.astype("int64")
    m["n_below_p10"] = (has_band & (m["fp"] < m["fppg_p10"])).astype("int64")
    m["n_above_p90"] = (has_band & (m["fp"] > m["fppg_p90"])).astype("int64")
    m["p50_games"] = has_p50.astype("int64")
    m["n_below_p50"] = (has_p50 & (m["fp"] < m["fppg_p50"])).astype("int64")
    cols = ["band_games", "n_below_p10", "n_above_p90", "p50_games", "n_below_p50"]
    return m.groupby("player_id")[cols].sum().reset_index()


def build_eval_frame(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
    games: pd.DataFrame,
    season: str,
    *,
    rank_only: bool = False,
) -> pd.DataFrame:
    """Join one season's projections to its actuals over the evaluation universe.

    Universe = every projected player (zero-game players get actual 0 FP) plus every player who
    played but was not projected.  See the module docstring.  ``rank_only`` projections carry
    only an ordering score in ``proj_total_fp``; every other prediction column is set to NaN.

    Output columns: ``season, player_id, player_name, projected, played`` + the seven
    prediction columns + ``actual_gp, actual_min, actual_mpg, actual_fppg, actual_total_fp,
    team_id, n_teams`` + the calibration counts from ``band_counts``.  Sorted by ``player_id``.
    """
    proj = projections.copy()
    if rank_only:
        for c in PRED_COLUMNS:
            if c == "proj_total_fp":
                continue
            proj[c] = np.nan
    keep = PRED_COLUMNS + ([] if rank_only else [c for c in OPTIONAL_PRED_COLUMNS if c in proj.columns])
    proj = proj[["player_id", "player_name"] + keep]
    proj["projected"] = True

    act = actuals[["player_id", "player_name", "team_id", "n_teams", "gp", "minutes", "mpg", "total_fp", "fppg"]].rename(
        columns={"player_name": "_name_act", "gp": "actual_gp", "minutes": "actual_min",
                 "mpg": "actual_mpg", "total_fp": "actual_total_fp", "fppg": "actual_fppg"})
    act["played"] = True

    df = proj.merge(act, on="player_id", how="outer")
    df["projected"] = df["projected"].fillna(False).astype(bool)
    df["played"] = df["played"].fillna(False).astype(bool)
    df["player_name"] = df["player_name"].where(df["player_name"].notna(), df["_name_act"])
    df = df.drop(columns="_name_act")

    # Rule 2: projected players with no games score zero total FP and zero GP.
    dnp = df["projected"] & ~df["played"]
    df.loc[dnp, ["actual_gp", "actual_min", "actual_total_fp"]] = 0.0
    df["actual_gp"] = df["actual_gp"].astype("float64")

    if not rank_only:
        bc = band_counts(games, projections)
        df = df.merge(bc, on="player_id", how="left")
    else:
        for c in ("band_games", "n_below_p10", "n_above_p90", "p50_games", "n_below_p50"):
            df[c] = 0
    for c in ("band_games", "n_below_p10", "n_above_p90", "p50_games", "n_below_p50"):
        df[c] = df[c].fillna(0).astype("int64")

    df.insert(0, "season", season)
    df = df.sort_values("player_id", kind="stable").reset_index(drop=True)
    front = ["season", "player_id", "player_name", "projected", "played"]
    return df[front + [c for c in df.columns if c not in front]]
