"""Vectorized fantasy points over DataFrames (the per-row twin of ``points.fantasy_points``)."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.contracts import STAT_COLUMN_MAP


def fantasy_points_frame(
    df: pd.DataFrame,
    scoring: Mapping[str, float],
    *,
    prefix: str = "",
    columns: Mapping[str, str] = STAT_COLUMN_MAP,
) -> pd.Series:
    """Fantasy points for every row of ``df``.

    ``prefix`` selects the column family: ``""`` for ``game_logs``-style columns (``pts``),
    ``"proj_"`` for projection columns (``proj_pts``). Raises KeyError naming any scored stat
    whose column is absent, so a mis-shaped frame fails loudly instead of scoring as zero.
    """
    unknown = [k for k in scoring if k not in columns]
    if unknown:
        raise KeyError(f"scoring keys with no column mapping: {unknown}")
    cols = {k: f"{prefix}{columns[k]}" for k in scoring}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise KeyError(f"frame missing stat columns: {missing}")
    total = pd.Series(0.0, index=df.index)
    for key, weight in scoring.items():
        total = total + df[cols[key]].astype("float64") * weight
    return total
