"""Appearance (hurdle) stage of availability: the probability a veteran plays at all in the target season (ADR 0031).

``AvailabilityModel`` (``availability.py``) is fit on player-seasons with at least one game, so its mean ``mu`` is the
expected share of the schedule *given that the player appears*. A returner from a long injury, a player with a
retirement or a suspension risk, or someone who never signs is invisible to it, which biases ``proj_gp`` high for them
(docs/limitations.md: returners project 15.3 GP too high, only 41% appear).

This module adds the missing first stage::

    P(appear)   logistic regression on the same games-missed history (f_bar, f_last, f_min, absent, age, age^2, minutes)
                plus how many of the last K seasons the player played and how long ago the last one was
    GP          = 0 with probability 1 - p, else L * Beta(mu * nu, (1 - mu) * nu)   (the conditional Beta of availability.py)

so ``proj_gp = p * mu * L`` is the unconditional expectation and ``proj_total_fp = proj_fppg * proj_gp`` stays the
contract's product. Training rows are every (player, season) in the history where the player was *eligible* in the
same sense the projector uses (played in one of the ``active_seasons`` seasons before it), labelled by whether the player
has any game in that season. Everything is a pure function of the ``History``-derived panel, so it is leakage-safe by
construction exactly like the rest of the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.availability import MU_CLIP, AvailabilityModel, availability_features

P_CLIP = (0.02, 1.0)
DEFAULT_C = 1.0


def appearance_features(f_lags: np.ndarray, decay: float, age: np.ndarray, mpg: np.ndarray) -> np.ndarray:
    """Availability features plus seasons-present count and seasons since the last appearance."""
    f_lags = np.asarray(f_lags, float)
    base = availability_features(f_lags, decay, np.asarray(age, float), np.asarray(mpg, float))
    present = np.isfinite(f_lags)
    k = f_lags.shape[1]
    n_present = present.sum(axis=1).astype(float)
    last_idx = np.where(present.any(axis=1), present.argmax(axis=1) + 1, k + 1).astype(float)   # 1 = played last season
    return np.column_stack([base, n_present, last_idx])


def _with_extra(X: np.ndarray, extra: np.ndarray | None, n_extra: int) -> np.ndarray:
    """Append ``extra`` columns (an injury- or ADP-aware caller's feature matrix) or ``n_extra`` NaN columns when absent."""
    if extra is None:
        return X if n_extra == 0 else np.column_stack([X, np.full((X.shape[0], n_extra), np.nan)])
    extra = np.asarray(extra, float)
    extra = extra[:, None] if extra.ndim == 1 else extra
    if n_extra and extra.shape[1] != n_extra:
        raise ValueError(f"extra has {extra.shape[1]} columns, model was fit with {n_extra}")
    return np.column_stack([X, extra])


@dataclass
class AppearanceModel:
    scaler: StandardScaler | None
    clf: LogisticRegression | None
    base_rate: float
    decay: float
    n_train: int
    n_appeared: int
    n_extra: int = 0

    @property
    def fitted(self) -> bool:
        return self.clf is not None

    @classmethod
    def fit(cls, f_lags, appeared, age, mpg, *, decay: float, C: float = DEFAULT_C,
            min_rows: int = 100, extra: np.ndarray | None = None) -> "AppearanceModel":
        """``appeared`` is 1.0 where the player has >= 1 game in the target season of that row, else 0.0.

        ``extra`` (optional, ``(n, m)``) carries the same additional feature columns the availability model gets (absence
        streaks, ADP, labelled injuries); a row with a non-finite extra value is dropped from the fit.
        """
        y = np.asarray(appeared, float)
        n_extra = 0 if extra is None else (1 if np.asarray(extra).ndim == 1 else np.asarray(extra).shape[1])
        X = _with_extra(appearance_features(f_lags, decay, age, mpg), extra, n_extra)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        base_rate = float(y[ok].mean()) if ok.any() else 0.9
        pos, neg = int((y[ok] > 0.5).sum()), int((y[ok] < 0.5).sum())
        if ok.sum() < min_rows or pos < 10 or neg < 10:     # too little evidence of either outcome to fit a logit
            return cls(None, None, base_rate, decay, int(ok.sum()), pos, n_extra)
        scaler = StandardScaler().fit(X[ok])
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X[ok]), (y[ok] > 0.5).astype(int))
        return cls(scaler, clf, base_rate, decay, int(ok.sum()), pos, n_extra)

    def predict(self, f_lags, age, mpg, extra: np.ndarray | None = None) -> np.ndarray:
        """P(player has at least one game in the target season), in ``P_CLIP``."""
        n = len(np.asarray(f_lags))
        if self.clf is None:
            return np.full(n, float(np.clip(self.base_rate, *P_CLIP)))
        X = _with_extra(appearance_features(f_lags, self.decay, age, mpg), extra, self.n_extra)
        Xs = np.nan_to_num(self.scaler.transform(X), nan=0.0)   # a missing feature takes its training-set column mean
        return np.clip(self.clf.predict_proba(Xs)[:, 1], *P_CLIP)


def gp_mixture(availability: AvailabilityModel, mu: np.ndarray, p: np.ndarray, season_games: float,
               quantiles=(0.10, 0.90)):
    """``(mean, sd, lo, hi)`` of games played under ``0 w.p. 1-p, else season_games * Beta(mu*nu, (1-mu)*nu)``.

    The mean is ``p * mu * L``. A quantile below the point mass at zero (``q <= 1 - p``) is 0; above it, it is the
    conditional Beta quantile at ``(q - (1 - p)) / p``. With ``p = 1`` this reduces exactly to ``availability.distribution``.
    """
    mu = np.clip(np.asarray(mu, float), *MU_CLIP)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    nu = 1.0 / availability.phi - 1.0
    L = float(season_games)
    cond_mean = mu * L
    cond_var = L * L * availability.phi * mu * (1 - mu)
    mean = p * cond_mean
    var = p * (cond_var + cond_mean ** 2) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    out = []
    for q in quantiles:
        q_cond = np.clip((q - (1.0 - p)) / np.maximum(p, 1e-12), 0.0, 1.0)
        val = L * beta_dist.ppf(q_cond, mu * nu, (1 - mu) * nu)
        out.append(np.where(q <= 1.0 - p, 0.0, val))
    return mean, sd, out[0], out[1]
