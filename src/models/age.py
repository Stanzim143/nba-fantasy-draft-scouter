"""Age curves estimated from the history with the delta method.

For every player observed in two consecutive seasons we take ``delta = rate(t+1) - rate(t)`` and
regress it on the age at ``t``, weighting each pair by the harmonic mean of the two exposures
(inverse-variance for a Poisson-type rate). The fitted polynomial is the expected one-year change
at each age. Its antiderivative shifts a rate observed at age ``a`` to what it "would be" at the
target age. Nothing about the shape is assumed beyond a low-degree polynomial.

Known bias (documented in docs/modeling.md): only players who stay in the league appear in the
pairs (survivorship), which flatters late-career curves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AgeCurve:
    """Expected yearly change ``delta(age)``; a polynomial in ``(age - center)``, held flat outside [lo, hi]."""

    coefs: tuple[float, ...] = (0.0,)   # low -> high order
    center: float = 27.0
    lo: float = 19.0
    hi: float = 40.0
    n_pairs: int = 0

    @property
    def is_flat(self) -> bool:
        return self.n_pairs == 0

    def delta(self, age):
        a = np.clip(np.asarray(age, float), self.lo, self.hi)
        return np.polynomial.polynomial.polyval(a - self.center, self.coefs)

    def cumulative(self, age):
        """Antiderivative ``C(age)``; extended linearly (constant delta) beyond [lo, hi]."""
        age = np.asarray(age, float)
        a = np.clip(age, self.lo, self.hi)
        integ = np.polynomial.polynomial.polyint(self.coefs)
        base = np.polynomial.polynomial.polyval(a - self.center, integ)
        return base + self.delta(a) * (age - a)

    def shift(self, age_from, age_to):
        """Expected change in the rate between ``age_from`` and ``age_to`` (NaN ages give 0)."""
        out = self.cumulative(age_to) - self.cumulative(age_from)
        return np.where(np.isfinite(out), out, 0.0)


FLAT = AgeCurve()


def consecutive_pairs(panel: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """One row per (player, t, t+1) with ``age`` (at t), ``delta`` of the rate num/den, and a weight."""
    p = panel[panel[den] > 0][["player_id", "s", "age", num, den]].copy()
    p["rate"] = p[num] / p[den]
    nxt = p[["player_id", "s", "rate", den]].copy()
    nxt["s"] = nxt["s"] - 1
    m = p.merge(nxt, on=["player_id", "s"], suffixes=("", "_next"))
    m = m[np.isfinite(m["age"])]
    e0, e1 = m[den].to_numpy(float), m[f"{den}_next"].to_numpy(float)
    return pd.DataFrame({
        "player_id": m["player_id"].to_numpy(),
        "s": m["s"].to_numpy(),
        "age": m["age"].to_numpy(float),
        "delta": (m["rate_next"] - m["rate"]).to_numpy(float),
        "weight": e0 * e1 / (e0 + e1),
    })


def fit_age_curve(pairs: pd.DataFrame, *, degree: int = 2, min_pairs: int = 40, center: float = 27.0) -> AgeCurve:
    """Weighted polynomial fit of delta on age. Too little data returns the flat (zero) curve."""
    d = pairs[np.isfinite(pairs["delta"]) & (pairs["weight"] > 0)]
    ages = d["age"].to_numpy(float)
    if len(d) < min_pairs or len(np.unique(np.round(ages))) < degree + 2:
        return FLAT
    w = d["weight"].to_numpy(float)
    order = np.argsort(ages)
    cw = np.cumsum(w[order]) / w.sum()
    lo = float(ages[order][np.searchsorted(cw, 0.025)])
    hi = float(ages[order][min(np.searchsorted(cw, 0.975), len(ages) - 1)])
    deg = min(degree, max(len(np.unique(np.round(ages))) - 2, 0))
    coefs = np.polynomial.polynomial.polyfit(ages - center, d["delta"].to_numpy(float), deg, w=np.sqrt(w))
    return AgeCurve(coefs=tuple(float(c) for c in coefs), center=center, lo=lo, hi=hi, n_pairs=int(len(d)))
