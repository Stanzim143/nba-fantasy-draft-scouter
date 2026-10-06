"""A one-line, plain-English reason for each player's model rank (display only).

The board ranks by ``vorp`` (value over replacement of the season total). This module only *describes* that
result from columns the board already carries; it never changes a projection or an ordering. It cannot know
causes the model never saw (a coach's plan, camp news), so where ADP disagrees strongly it says so rather
than guessing a cause.
"""
from __future__ import annotations

import pandas as pd

ADP_GAP_NOTE = 20          # |adp - rank| (spots) from which the market-vs-model disagreement is worth a clause
LOW_GP, HIGH_GP = 60.0, 72.0
WIDE_BAND = 1.0            # (p90 - p10) / p50 of the season total above this = a very wide range
MAX_FLAG_CHARS = 90


def _num(row: pd.Series, key: str) -> float | None:
    v = row.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _txt(row: pd.Series, key: str) -> str:
    v = row.get(key)
    return v.strip() if isinstance(v, str) else ""


def reason_for(row: pd.Series, season_games: float = 82.0) -> str:
    """The reason text for one board row (any missing column is simply left out)."""
    fppg, gp, total, vorp = (_num(row, k) for k in ("proj_fppg", "proj_gp", "proj_total_fp", "vorp"))
    parts: list[str] = []
    if None not in (fppg, gp, total, vorp):
        parts.append(f"{fppg:.1f} FP/game x {gp:.0f} games = {total:.0f} FP, "
                     f"{'+' if vorp >= 0 else '-'}{abs(vorp):.0f} vs a replacement player")
    cls = _txt(row, "projection_class")
    if cls and cls != "veteran":
        parts.append(f"{cls}: projected from limited or indirect evidence, so treat it as rougher")
    if gp is not None:
        missed = season_games - gp
        if gp < LOW_GP:
            parts.append(f"model expects ~{missed:.0f} missed games (injury history or low appearance rate), which cuts the total")
        elif gp >= HIGH_GP:
            parts.append("projected to be durable, so the full per-game output counts")
    age = _num(row, "age")
    if age is not None and age >= 33:
        parts.append(f"age {age:.0f}: ageing curve trims output")
    elif age is not None and age <= 22 and cls in ("", "veteran"):
        parts.append(f"age {age:.0f}: young players are projected to grow")
    lo, hi, mid = (_num(row, k) for k in ("proj_total_fp_p10", "proj_total_fp_p90", "proj_total_fp_p50"))
    if None not in (lo, hi, mid) and mid > 0 and (hi - lo) / mid > WIDE_BAND:
        parts.append("very wide outcome range (boom or bust)")
    gap = _num(row, "adp_gap")
    adp = _num(row, "adp")
    if gap is not None and adp is not None and abs(gap) >= ADP_GAP_NOTE:
        if gap > 0:
            parts.append(f"model ranks him {gap:.0f} spots above the market (ADP {adp:.0f}); check for news the model can't see")
        else:
            parts.append(f"market drafts him {-gap:.0f} spots higher (ADP {adp:.0f}): ADP may price role or health news the model can't see")
    for key in ("risk_flags", "return_flag", "lm_flag"):
        t = _txt(row, key)
        if t:
            parts.append(t if len(t) <= MAX_FLAG_CHARS else t[: MAX_FLAG_CHARS - 1].rstrip() + "…")
    return "; ".join(parts)


def add_rank_reason(board: pd.DataFrame, season_games: float = 82.0) -> pd.DataFrame:
    """``board`` plus a ``rank_reason`` column (a copy; attrs are kept). Every row gets a string."""
    out = board.copy()
    out.attrs.update(board.attrs)
    out["rank_reason"] = [reason_for(r, season_games) for _, r in out.iterrows()]
    return out
