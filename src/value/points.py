"""Fantasy points engine: stat line x league scoring weights."""
from __future__ import annotations

from typing import Mapping

STAT_KEYS = ("PTS", "REB", "AST", "STL", "BLK", "TO", "FGM", "FGA", "FTM", "FTA", "3PM")


def fantasy_points(stats: Mapping[str, float], scoring: Mapping[str, float]) -> float:
    """Total fantasy points for one stat line (per game, per season, whatever the stats are)."""
    missing = [k for k in scoring if k not in stats]
    if missing:
        raise KeyError(f"stat line missing scored stats: {missing}")
    return sum(stats[k] * w for k, w in scoring.items())


def breakdown(stats: Mapping[str, float], scoring: Mapping[str, float]) -> dict[str, float]:
    """Per-stat contribution, useful for explaining why a player ranks where they do."""
    return {k: stats[k] * w for k, w in scoring.items()}
