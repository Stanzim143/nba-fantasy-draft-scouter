"""Availability: games played as a distribution, not a point estimate.

Model. Let ``f = GP / L`` be the fraction of the schedule a player appears in (``L`` = team games,
so ``GP <= L`` holds by construction). The mean of ``f`` is a *fractional logistic regression*
(quasi-binomial: the rows are duplicated with label 1 weighted ``f`` and label 0 weighted ``1-f``)
on the games-missed history and context::

    f_bar    recency-weighted fraction of the schedule played over the last seasons
    f_last   last season's fraction (equals f_bar when the player sat out last season)
    f_min    worst recent season (a proxy for injury proneness)
    absent   1 if the player has no games last season
    age, age^2, projected minutes per game

The spread is Beta-shaped: ``f ~ Beta(mu*nu, (1-mu)*nu)`` with ``Var(f) = phi * mu * (1 - mu)``
and ``phi`` estimated from the training residuals. That gives an asymmetric distribution bounded by
the schedule, and ``proj_gp = mu * L`` is its mean. ``mu`` is conditional on the player being
active in the target season (a player who never plays is invisible to the training rows), so a
retirement or season-ending injury before opening night is *not* in ``proj_gp``; the injury layer
described in docs/modeling.md is where that risk belongs.

``AvailabilityModel.fit``/``predict_mean`` accept an optional ``extra`` matrix of additional
feature columns (same row count as ``f_lags``), appended after the columns above before scaling
and fitting. ``BaselineProjector`` never passes one (``extra=None``), so its behaviour is
unchanged; ``src.models.injury_baseline.BaselineInjuryProjector`` (docs/adr/0006-injury-layer.md)
is the one caller that does, with absence-streak-derived columns from ``src.features.injury``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MU_CLIP = (0.02, 0.985)
PHI_CLIP = (0.01, 0.6)
DEFAULT_PHI = 0.12


def availability_features(f_lags: np.ndarray, decay: float, age: np.ndarray, mpg: np.ndarray) -> np.ndarray:
    """Feature matrix from lagged fractions ``f_lags`` (n, K; NaN = season absent)."""
    present = np.isfinite(f_lags)
    f0 = np.where(present, f_lags, 0.0)
    w = decay ** np.arange(f_lags.shape[1]) * present
    wsum = w.sum(axis=1)
    f_bar = np.where(wsum > 0, (w * f0).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)
    f_last = np.where(present[:, 0], f0[:, 0], f_bar)
    f_min = np.where(present.any(axis=1), np.where(present, f_lags, np.inf).min(axis=1), np.nan)
    age_c = np.where(np.isfinite(age), (age - 27.0) / 5.0, 0.0)
    mpg_c = (mpg - 20.0) / 10.0
    return np.column_stack([f_bar, f_last, f_min, (~present[:, 0]).astype(float), age_c, age_c ** 2, mpg_c])


def _with_extra(X: np.ndarray, extra: np.ndarray | None, n_extra: int, fill: float = 0.0) -> np.ndarray:
    """Append ``extra`` columns to ``X``, or ``n_extra`` filled columns if ``extra`` is None.

    Keeps the feature matrix width consistent between fit and predict regardless of whether a
    caller passes ``extra`` (the plain ``AvailabilityModel`` never does; an injury-aware caller
    always does with the same width it fitted with).
    """
    if extra is None:
        if n_extra == 0:
            return X
        return np.column_stack([X, np.full((X.shape[0], n_extra), fill)])
    extra = np.asarray(extra, float)
    if extra.ndim == 1:
        extra = extra[:, None]
    if n_extra and extra.shape[1] != n_extra:
        raise ValueError(f"extra has {extra.shape[1]} columns, model was fit with {n_extra}")
    return np.column_stack([X, extra])


@dataclass
class AvailabilityModel:
    scaler: StandardScaler | None
    clf: LogisticRegression | None
    phi: float
    league_mean: float
    decay: float
    n_train: int
    n_extra: int = 0

    @property
    def fitted(self) -> bool:
        return self.clf is not None

    @classmethod
    def fit(cls, f_lags, f_target, age, mpg, *, decay: float, C: float, min_rows: int,
            extra: np.ndarray | None = None) -> "AvailabilityModel":
        """Fit on rows that have at least one earlier season and played in the target season.

        ``extra``, if given, is an (n, m) matrix of additional feature columns aligned with
        ``f_lags`` rows (e.g. absence-streak features); it is appended to the base feature
        matrix before scaling. Rows with a non-finite ``extra`` value are dropped from the fit
        just like the base features.
        """
        f_lags, f_target = np.asarray(f_lags, float), np.asarray(f_target, float)
        n_extra = 0 if extra is None else (1 if np.asarray(extra).ndim == 1 else np.asarray(extra).shape[1])
        X = availability_features(f_lags, decay, np.asarray(age, float), np.asarray(mpg, float))
        X = _with_extra(X, extra, n_extra, fill=np.nan)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(f_target) & (f_target > 0)
        league_mean = float(np.mean(f_target[ok])) if ok.any() else 0.75
        if ok.sum() < min_rows:
            return cls(None, None, DEFAULT_PHI, league_mean, decay, int(ok.sum()), n_extra)
        X, y = X[ok], np.clip(f_target[ok], 0.0, 1.0)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(np.vstack([Xs, Xs]), np.r_[np.ones(len(y)), np.zeros(len(y))],
                sample_weight=np.r_[y, 1.0 - y])
        mu = np.clip(clf.predict_proba(Xs)[:, 1], *MU_CLIP)
        phi = float(np.clip(np.sum((y - mu) ** 2) / np.sum(mu * (1 - mu)), *PHI_CLIP))
        return cls(scaler, clf, phi, league_mean, decay, int(ok.sum()), n_extra)

    def predict_mean(self, f_lags, age, mpg, extra: np.ndarray | None = None) -> np.ndarray:
        """Expected fraction of the schedule played, in (0, 1)."""
        X = availability_features(np.asarray(f_lags, float), self.decay, np.asarray(age, float), np.asarray(mpg, float))
        X = _with_extra(X, extra, self.n_extra, fill=np.nan)
        if self.clf is None:
            # Too little history to fit: half own recent rate, half league mean.
            f_bar = np.where(np.isfinite(X[:, 0]), X[:, 0], self.league_mean)
            return np.clip(0.5 * f_bar + 0.5 * self.league_mean, *MU_CLIP)
        Xs = self.scaler.transform(np.nan_to_num(X, nan=self.league_mean))
        return np.clip(self.clf.predict_proba(Xs)[:, 1], *MU_CLIP)

    def distribution(self, mu: np.ndarray, season_games: float, quantiles=(0.10, 0.90)):
        """``(sd, lo, hi)`` of games played under Beta(mu*nu, (1-mu)*nu) scaled by the schedule."""
        mu = np.clip(np.asarray(mu, float), *MU_CLIP)
        nu = 1.0 / self.phi - 1.0
        sd = season_games * np.sqrt(self.phi * mu * (1 - mu))
        qs = [season_games * beta_dist.ppf(q, mu * nu, (1 - mu) * nu) for q in quantiles]
        return sd, qs[0], qs[1]
