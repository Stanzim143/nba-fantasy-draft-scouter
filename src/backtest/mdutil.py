"""Tiny markdown helpers (no ``tabulate`` dependency)."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd


def _fmt(v, floatfmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return floatfmt.format(v)
    return str(v).replace("|", "\\|").replace("\n", " ")


def df_to_markdown(df: pd.DataFrame, *, floatfmt: str = "{:.3f}", index: bool = True,
                   formats: Mapping[str, Callable | str] | None = None) -> str:
    """Render ``df`` as a GitHub markdown table.  ``formats`` overrides per column (format string
    or callable).  NaN renders as ``n/a``."""
    formats = formats or {}
    d = df.reset_index() if index else df
    if index and df.index.name is None and not isinstance(df.index, pd.MultiIndex):
        d = d.rename(columns={"index": ""})
    head = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for _, row in d.iterrows():
        cells = []
        # Index by position (row.iloc[i]), not by name (row[c]): if two columns share a name --
        # e.g. resetting an unnamed index into a column called "" collides with a caller's own
        # "" column -- `row[c]` returns a Series instead of a scalar, and every cell in that
        # column silently renders as a pandas repr ("Name: 0, dtype: ...") instead of the value.
        # This happened for real in write_report's "Run" table (found manually, not by a test).
        for i, c in enumerate(d.columns):
            v, f = row.iloc[i], formats.get(c)
            if callable(f):
                cells.append(str(f(v)))
            elif isinstance(f, str) and not (isinstance(v, float) and np.isnan(v)):
                cells.append(f.format(v))
            else:
                cells.append(_fmt(v, floatfmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
