"""Season-total fantasy-point intervals (ADR 0033).

``fppg_p10/p90`` are *game-level* quantiles (calibrated, see docs/backtest.md): they say how one game can go, not how a
season's total can. A season total is ``T = Fbar * G`` with

* ``G``     games played: zero with probability ``1 - p`` (the appearance stage, ADR 0031), else ``L * Beta`` (availability.py);
* ``Fbar``  the season-mean FPPG over the games he plays, which misses the projection for two reasons: true talent differs
            from the projected one (relative spread ``tau``, estimated below) and the average of ``G`` noisy games is itself
            noisy (game-level ``sd / sqrt(G)``).

``SeasonFppgUncertainty`` estimates ``tau`` from the history's own player-seasons (projection from lags alone, as forecasting
does, versus the realised season mean) by experience bucket, subtracting the expected game-noise share so it is not counted
twice. ``simulate_total_quantiles`` then draws ``G`` and ``Fbar`` (independent, deterministic per player) and returns the
quantiles of their product. ``Fbar`` and ``G`` are treated as independent: a player who plays few games through injury is not
modelled as also playing worse; that is checked, not assumed, by the coverage table in the backtest report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

from src.models.availability import MU_CLIP, AvailabilityModel

DEFAULT_TAU = 0.15      # relative sd used only when the history has no usable rows at all
TAU_FLOOR = 0.02        # relative sd of talent misprojection is never below 2% (avoids a degenerate, overconfident band)
MIN_BUCKET_ROWS = 40
N_DRAWS = 1000
PRED_FLOOR = 5.0        # projections below this FPPG give unstable relative residuals and are excluded from the fit
BUCKETS = (0, 1, 2, 3)  # seasons of history, capped: 0 = no NBA history (rookie / debutant)


def bucket(n_hist) -> np.ndarray:
    return np.clip(np.asarray(n_hist, int), 0, BUCKETS[-1])


@dataclass
class SeasonFppgUncertainty:
    tau: dict[int, float]      # bucket -> relative sd of the talent misprojection
    n_rows: dict[int, int]

    def rel_tau(self, n_hist) -> np.ndarray:
        b = bucket(n_hist)
        return np.array([self.tau[int(i)] for i in b])

    @classmethod
    def fit(cls, pred, actual, gp, sd_game, n_hist, *, min_gp: int = 10) -> "SeasonFppgUncertainty":
        """Per-bucket ``tau`` from ``(actual - pred) / pred`` over player-seasons with ``gp >= min_gp``.

        ``E[r^2] = tau^2 + (sd_game / pred)^2 / gp``; the second term is subtracted, the result floored at ``TAU_FLOOR``.
        A bucket with too few rows borrows the pooled estimate of all buckets that have enough.
        """
        pred, actual, gp, sd = (np.asarray(x, float) for x in (pred, actual, gp, sd_game))
        b = bucket(n_hist)
        ok = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(sd) & (pred >= PRED_FLOOR) & (gp >= min_gp)
        r2 = np.full(len(pred), np.nan)
        noise = np.full(len(pred), np.nan)
        r2[ok] = ((actual[ok] - pred[ok]) / pred[ok]) ** 2
        noise[ok] = (sd[ok] / pred[ok]) ** 2 / gp[ok]

        def est(mask):
            m = mask & ok
            if not m.any():
                return DEFAULT_TAU, 0
            return float(np.sqrt(max(np.mean(r2[m]) - np.mean(noise[m]), TAU_FLOOR ** 2))), int(m.sum())

        pooled = est(np.ones(len(pred), bool))
        tau, n_rows = {}, {}
        for i in BUCKETS:
            t, n = est(b == i)
            tau[i], n_rows[i] = (t, n) if n >= MIN_BUCKET_ROWS else (pooled[0], n)
        return cls(tau, n_rows)


def simulate_total_quantiles(player_ids, fppg, mu_f, p_appear, season_games: float, availability: AvailabilityModel,
                             tau_rel, sd_game, *, quantiles=(0.10, 0.50, 0.90), n_draws: int = N_DRAWS, seed: int = 0):
    """Quantiles of season total FP, one row per player; independent of the order or the set of players.

    Each player's draws come from ``default_rng([seed, player_id])``, so a projection does not change when another
    player is added or removed. ``G = 0`` with probability ``1 - p``; otherwise ``L * Beta(mu*nu, (1-mu)*nu)`` through the
    inverse CDF of a uniform. ``Fbar ~ Normal(fppg, sqrt((tau * fppg)^2 + sd_game^2 / max(G, 1)))``, clipped at zero.
    """
    pid = np.asarray(player_ids, "int64")
    fppg, mu, p = (np.asarray(x, float) for x in (fppg, mu_f, p_appear))
    tau, sd = np.asarray(tau_rel, float), np.asarray(sd_game, float)
    mu = np.clip(mu, *MU_CLIP)
    nu = 1.0 / availability.phi - 1.0
    L = float(season_games)
    out = np.empty((len(pid), len(quantiles)))
    for i in range(len(pid)):
        rng = np.random.default_rng([seed, int(pid[i])])
        u, z = rng.random(n_draws), rng.standard_normal(n_draws)
        appears = u >= 1.0 - p[i]
        q = np.clip((u - (1.0 - p[i])) / max(p[i], 1e-12), 1e-9, 1 - 1e-9)
        g = np.where(appears, L * beta_dist.ppf(q, mu[i] * nu, (1 - mu[i]) * nu), 0.0)
        fbar = np.maximum(fppg[i] + z * np.sqrt((tau[i] * fppg[i]) ** 2 + sd[i] ** 2 / np.maximum(g, 1.0)), 0.0)
        out[i] = np.quantile(fbar * g, quantiles)
    return out
