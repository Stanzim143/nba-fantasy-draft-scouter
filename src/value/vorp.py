"""Value over replacement player (VORP) for a points league.

Two variants are always computed:

* ``vorp``           = ``proj_total_fp`` - replacement season total (what you draft on)
* ``vorp_per_game``  = ``proj_fppg`` - the same replacement player's FPPG (rate value, ignores health)

Positional scarcity modes (``positional``):

* ``"off"``   one league-wide replacement level.
* ``"on"``    each player's replacement level is the lowest positional level among his eligible
              ESPN positions (see ``replacement.positional_replacement_per_player``).
* ``"auto"``  (default) ``"on"`` only when the data-driven scarcity check says positional
              differences are material for this league, else ``"off"``. The check and the verdict
              are always returned so the decision is visible, never silent.

Per-game VORP is always league-wide: positional replacement levels are season totals from a
rostered-pool simulation and there is no equally clean per-game analogue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.value.replacement import (
    DEFAULT_SCARCITY_THRESHOLD, LeagueShape, PositionalReport, ReplacementLevel, positional_replacement,
    positional_replacement_per_player, replacement_level,
)

POSITIONAL_MODES = ("off", "on", "auto")


@dataclass
class VorpResult:
    frame: pd.DataFrame                       # index aligned with the input; vorp, vorp_per_game, repl_total
    replacement: ReplacementLevel
    positional: PositionalReport | None       # None when no positions were supplied
    positional_used: bool


def compute_vorp(proj: pd.DataFrame, shape: LeagueShape, *, bench_weight: float | None = None,
                 season_games: float = 82.0, positional: str = "auto",
                 scarcity_threshold: float = DEFAULT_SCARCITY_THRESHOLD) -> VorpResult:
    """VORP for every row of ``proj`` (needs proj_total_fp, proj_fppg, proj_gp; ``position`` optional)."""
    if positional not in POSITIONAL_MODES:
        raise ValueError(f"positional must be one of {POSITIONAL_MODES}")
    need = ["proj_total_fp", "proj_fppg", "proj_gp"]
    missing = [c for c in need if c not in proj.columns]
    if missing:
        raise KeyError(f"projections missing columns {missing}")
    if positional == "on" and "position" not in proj.columns:
        raise ValueError("positional='on' needs a 'position' column")

    total = proj["proj_total_fp"].to_numpy(float)
    repl = replacement_level(total, proj["proj_fppg"].to_numpy(float), proj["proj_gp"].to_numpy(float),
                             shape, bench_weight=bench_weight, season_games=season_games)
    report = None
    if "position" in proj.columns and len(proj):
        report = positional_replacement(total, proj["position"].to_numpy(), shape,
                                        bench_weight=repl.bench_weight, global_level=repl.total,
                                        threshold=scarcity_threshold)
    use_positional = positional == "on" or (positional == "auto" and report is not None and report.material)
    if use_positional:
        repl_total = positional_replacement_per_player(proj["position"].to_numpy(), report, repl.total)
    else:
        repl_total = np.full(len(proj), repl.total)
    frame = pd.DataFrame({
        "vorp": total - repl_total,
        "vorp_per_game": proj["proj_fppg"].to_numpy(float) - repl.per_game,
        "repl_total": repl_total,
    }, index=proj.index)
    return VorpResult(frame, repl, report, bool(use_positional))
