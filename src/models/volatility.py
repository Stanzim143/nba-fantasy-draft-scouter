"""Per-game fantasy-point volatility and the floor / median / ceiling columns.

Three estimated pieces, all from the history:

1. ``sd_prior(fppg) = a + b * fppg``: a weighted regression of a player-season's game-level FP
   standard deviation on its mean (spread scales with level; a 45-FPPG star swings more than a
   20-FPPG role player).
2. A player's *ratio* ``z = observed sd / sd_prior`` is recency-weighted across seasons and shrunk
   toward 1 with the same empirical-Bayes machinery as the rates (constant prior 1, exposure =
   games). A player with a short record is treated as average volatility for his level.
3. Standardised quantiles ``q10, q50, q90`` carrying the *shape* (right skew, thin left tail)
   learned from data instead of assuming a normal distribution. Here they are pooled over every
   game as ``(fp - player-season mean) / player-season sd`` (a fall-back). ``BaselineProjector``
   replaces them with a calibrated version taken around the *projection* rather than the realised
   season mean (``FittedBaseline.calibrate_quantiles``), so the floor and ceiling also cover
   projection error and about 10% of games land below the floor.

Projection: ``sd = z_hat * sd_prior(proj_fppg)`` and ``p_q = proj_fppg + q_q * sd``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.shrink import wls

SD_FLOOR = 1.0
DEFAULT_QUANTILES = (-1.28155, 0.0, 1.28155)   # normal fall-back when there are no games at all


@dataclass
class VolatilityPrior:
    a: float
    b: float
    q: tuple[float, float, float]     # standardised quantiles at the configured probabilities
    n_rows: int

    def sd_prior(self, fppg) -> np.ndarray:
        return np.maximum(self.a + self.b * np.asarray(fppg, float), SD_FLOOR)


def fit_volatility_prior(panel: pd.DataFrame, game_fp: pd.DataFrame, *, min_games: int,
                         quantile_min_games: int, quantiles: tuple[float, float, float]) -> VolatilityPrior:
    ok = (panel["gp"] >= min_games) & panel["fp_sd"].notna()
    if ok.sum() >= 5:
        X = np.column_stack([np.ones(ok.sum()), panel.loc[ok, "fppg"].to_numpy(float)])
        a, b = wls(X, panel.loc[ok, "fp_sd"].to_numpy(float), panel.loc[ok, "gp"].to_numpy(float))
    else:
        a, b = 8.0, 0.0
    q = _pooled_quantiles(panel, game_fp, quantile_min_games, quantiles)
    return VolatilityPrior(float(a), float(b), q, int(ok.sum()))


def _pooled_quantiles(panel, game_fp, min_games, quantiles):
    keep = panel[(panel["gp"] >= min_games) & (panel["fp_sd"] > 0)][["player_id", "s", "fppg", "fp_sd"]]
    if keep.empty:
        return DEFAULT_QUANTILES
    g = game_fp.merge(keep, on=["player_id", "s"])
    z = ((g["fp"] - g["fppg"]) / g["fp_sd"]).to_numpy(float)
    q = np.quantile(z, quantiles)
    return (float(q[0]), float(q[1]), float(q[2]))


def add_ratio_columns(panel: pd.DataFrame, prior: VolatilityPrior, min_games: int) -> pd.DataFrame:
    """Add ``z_gp`` (= games * sd/sd_prior) and ``gpz`` (= games, when the season is long enough)."""
    out = panel.copy()
    valid = (out["gp"] >= min_games) & out["fp_sd"].notna()
    z = out["fp_sd"] / prior.sd_prior(out["fppg"].to_numpy())
    out["gpz"] = np.where(valid, out["gp"], 0.0)
    out["z_gp"] = np.where(valid, z * out["gp"], 0.0)
    return out


def floor_median_ceiling(fppg, sd, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fppg, sd = np.asarray(fppg, float), np.asarray(sd, float)
    return fppg + q[0] * sd, fppg + q[1] * sd, fppg + q[2] * sd
