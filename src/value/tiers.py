"""Gap-based tiers: group players by value cliffs, not by rank buckets.

Sort players by value (VORP), look at the drop between neighbours, and start a new tier where the
drop is a *cliff*: much larger than the typical drop in that neighbourhood.

1. ``gap_i = v_i - v_{i+1}`` for consecutive players (descending order).
2. ``scale_i`` = rolling median of gaps in a window around ``i`` (adapts to the density: gaps are
   naturally large at the very top, tiny in the deep middle).
3. ``i`` is a cliff candidate when ``gap_i >= gap_factor * scale_i`` and ``gap_i`` is at least
   ``min_gap_frac`` of the total value range (ignores cliffs that are large only relative to a
   nearly flat neighbourhood).
4. Candidates are accepted strongest first, skipping any that would create a tier smaller than
   ``min_tier_size``, until ``max_tiers - 1`` cliffs are used (at most ``max_tiers`` tiers above replacement).
5. Players at or below ``floor`` (default 0 VORP = replacement level) form one last tier of
   their own: "replaceable". That tier is added on top of the ``max_tiers`` cap, so a board has at most
   ``max_tiers + 1`` tiers.

Tier numbers start at 1 and never decrease with rank (property-tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_tiers(values, *, floor: float | None = 0.0, gap_factor: float = 2.5, window: int = 15,
                 min_tier_size: int = 3, max_tiers: int = 12, min_gap_frac: float = 0.01) -> np.ndarray:
    """Tier (1 = best) for each entry of ``values``, aligned with the input order.

    ``floor=None`` tiers every player; otherwise players with ``value <= floor`` all get the
    final tier. ``max_tiers`` caps the tiers above the floor only, so the result has at most ``max_tiers + 1``
    distinct tiers when a floor is used.
    """
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.lexsort((np.arange(n), -v))            # descending value, stable on input order
    sv = v[order]
    n_top = n if floor is None else int(np.sum(sv > floor))
    tiers_sorted = np.ones(n, dtype=int)
    bounds: list[int] = []                            # accepted cliff positions: gap between i and i+1
    if n_top >= 2 * min_tier_size and max_tiers > 1:
        top = sv[:n_top]
        gaps = top[:-1] - top[1:]
        scale = pd.Series(gaps).rolling(window * 2 + 1, center=True, min_periods=3).median().to_numpy()
        scale = np.maximum(scale, 1e-9)
        span = top[0] - top[-1]
        cand = np.where((gaps >= gap_factor * scale) & (gaps >= min_gap_frac * span) & (gaps > 0))[0]
        for i in cand[np.argsort(-(gaps[cand] / scale[cand]), kind="stable")]:
            if len(bounds) >= max_tiers - 1:
                break
            edges = sorted(bounds + [int(i)])
            starts = [0] + [e + 1 for e in edges]
            ends = [e + 1 for e in edges] + [n_top]
            if all(b - a >= min_tier_size for a, b in zip(starts, ends)):
                bounds.append(int(i))
        bounds.sort()
    for b in bounds:
        tiers_sorted[b + 1:n_top] += 1
    if n_top < n:
        tiers_sorted[n_top:] = (tiers_sorted[n_top - 1] + 1) if n_top else 1
    out = np.empty(n, dtype=int)
    out[order] = tiers_sorted
    return out
