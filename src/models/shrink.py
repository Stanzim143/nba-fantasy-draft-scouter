"""Empirical-Bayes shrinkage helpers.

A player's estimate ``est`` (built from ``n`` units of exposure: minutes, attempts, games) is
pulled toward a prior mean ``prior`` with weight ``n / (n + kappa)``. ``kappa`` is the exposure at
which own data and prior get equal say; it is *fitted* by minimising next-season prediction
error on pairs inside the history, not hard-coded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

KAPPA_LO, KAPPA_HI = 1e-2, 1e7


def shrink(est, prior, n, kappa):
    """``prior + n/(n+kappa) * (est - prior)``; n = 0 returns the prior exactly."""
    est, prior, n = np.asarray(est, float), np.asarray(prior, float), np.asarray(n, float)
    lam = n / (n + kappa)
    return prior + lam * (est - prior)


def kappa_loss(kappa: float, est, prior, n, y, w) -> float:
    pred = shrink(est, prior, n, kappa)
    return float(np.sum(w * (y - pred) ** 2))


def fit_kappa(est, prior, n, y, w, *, default: float, min_rows: int = 30) -> tuple[float, float, bool]:
    """Best shrinkage constant on log scale.

    Returns ``(kappa, loss, fitted)``. ``fitted`` is False when there were too few usable rows, in
    which case ``kappa`` is ``default`` and ``loss`` is the loss at that default (or inf).
    """
    est, prior, n, y, w = (np.asarray(a, float) for a in (est, prior, n, y, w))
    ok = np.isfinite(est) & np.isfinite(prior) & np.isfinite(n) & np.isfinite(y) & np.isfinite(w) & (w > 0) & (n > 0)
    if ok.sum() < min_rows:
        return float(default), float("inf"), False
    est, prior, n, y, w = est[ok], prior[ok], n[ok], y[ok], w[ok]
    # Normalise weights so the loss scale is comparable across decay candidates.
    w = w / w.sum()
    grid = np.exp(np.linspace(np.log(KAPPA_LO), np.log(KAPPA_HI), 80))
    losses = np.array([kappa_loss(k, est, prior, n, y, w) for k in grid])
    i = int(np.argmin(losses))
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    res = minimize_scalar(lambda lk: kappa_loss(float(np.exp(lk)), est, prior, n, y, w),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    k, loss = float(np.exp(res.x)), float(res.fun)
    if losses[i] < loss:
        k, loss = float(grid[i]), float(losses[i])
    return k, loss, True


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Weighted least squares with a tiny ridge on non-intercept columns (column 0 is the intercept)."""
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(X).all(axis=1)
    X, y, w = X[ok], y[ok], w[ok]
    if len(y) == 0:
        raise ValueError("no usable rows for regression")
    XtW = X.T * w
    A = XtW @ X
    pen = np.eye(X.shape[1]) * ridge * max(np.trace(A) / X.shape[1], 1e-12)
    pen[0, 0] = 0.0
    return np.linalg.solve(A + pen + 1e-12 * np.eye(X.shape[1]), XtW @ y)
