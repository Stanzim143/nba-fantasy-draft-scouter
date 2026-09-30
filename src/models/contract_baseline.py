"""``BaselineContractProjector``: the baseline plus a rookie-scale clock adjustment (ADR 0013).

A *stacked* projector, built the way ``BaselineOffseasonProjector`` is: it wraps a base projector
(default ``BaselineProjector``) and multiplies each player's projected counting stats by a factor learned
from how the rookie-scale clock (``src.features.contract``: scale year, option year, contract year,
extension-eligible year, post-scale year, second-round early years) explained the base model's *past*
misses. Games played, minutes and volatility are untouched, so the ablation isolates exactly this layer.

The clock is a function of ``players`` (draft year, round, pick) and the target season only, so it needs
no data beyond the ``History`` the base projector already receives. If the history is too short, too few
players carry a clock, or leave-one-season-out cross-validation shows no gain over "no adjustment", the
result is the base projection unchanged and ``contract_enabled`` is False: the same graceful fallback
every feature layer here uses.

Extra output columns (allowed by the ``projections`` contract): ``contract_adj`` (fantasy points per game
added), ``contract_factor``, ``contract_enabled``, and the display columns ``contract_years_since_draft``,
``contract_scale_year`` (1..4 for first-rounders on the scale) and ``contract_flag`` (``"contract_year"``,
``"option_year"``, ``"post_scale"``, ``"rookie_scale"``, or empty).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.contracts import PROJECTION_STATS, History, Projector, season_start, season_str, validate_table
from src.features.contract import (
    ContractFit, contract_clock, factor_from_adjustment, fit_adjustment, build_training_set,
)
from src.models.baseline import BaselineProjector
from src.models.panel import canonical_logs
from src.value.frame import fantasy_points_frame
from src.value.league import load_league


def _fingerprint(tables: Mapping[str, pd.DataFrame]) -> tuple:
    gl = tables["game_logs"]
    return (len(gl), int(gl["pts"].sum()) if len(gl) else 0, len(tables["players"]),
            int(pd.to_numeric(tables["players"]["draft_year"], errors="coerce").fillna(0).sum()))


def _flags(clock) -> np.ndarray:
    out = np.full(len(clock.player_id), "", dtype=object)
    out[np.isfinite(clock.scale_year)] = "rookie_scale"
    out[clock.is_post_scale_year] = "post_scale"
    out[clock.is_option_year] = "option_year"
    out[clock.is_contract_year] = "contract_year"
    return out


class BaselineContractProjector:
    """Registered as ``"baseline_contract"`` in ``src.models.registry``."""

    name = "baseline_contract"

    def __init__(self, base: Projector | None = None, scoring: Mapping[str, float] | None = None, *,
                 min_fit_rows: int | None = None, name: str | None = None,
                 walk_forward_cache: dict | None = None):
        self.scoring: Mapping[str, float] = dict(scoring) if scoring is not None else dict(load_league()["scoring"])
        self.base: Projector = base if base is not None else BaselineProjector(self.scoring)
        self.min_fit_rows = min_fit_rows
        if name is not None:
            self.name = name
        self.last_fit: ContractFit | None = None
        # Base projections of past seasons depend only on the data before them, not on the target, so they are
        # memoised (and can be shared between variants run over the same data).
        self._walk_forward: dict[tuple, pd.DataFrame] = walk_forward_cache if walk_forward_cache is not None else {}

    def _history_tables(self, history: History) -> dict[str, pd.DataFrame]:
        return {"game_logs": history.game_logs, "team_games": history.team_games, "players": history.players,
                "player_season_bio": history.player_season_bio, **history.extras}

    def _base_projection(self, tables: dict[str, pd.DataFrame], season: str) -> pd.DataFrame | None:
        h = History.until(tables, season)
        if h.game_logs.empty:
            return None
        key = (season, type(self.base).__name__, _fingerprint({"game_logs": h.game_logs, "players": h.players}))
        if key not in self._walk_forward:
            self._walk_forward[key] = self.base.project(h)
        return self._walk_forward[key]

    def _fit(self, history: History) -> ContractFit:
        tables = self._history_tables(history)
        gl = canonical_logs(history.game_logs)
        gl = gl.assign(fp=fantasy_points_frame(gl, self.scoring), s=gl["season"].map(season_start))
        actual = gl.groupby(["player_id", "s"]).agg(gp=("game_id", "size"), fppg=("fp", "mean")).reset_index()
        seasons = sorted(gl["s"].unique())
        train_seasons = [season_str(int(s)) for s in seasons[1:]]  # the first season has nothing before it to project from
        ts = build_training_set(history.players, train_seasons, lambda s: self._base_projection(tables, s), actual)
        if ts is None:
            return ContractFit(None, None, 0.0, False, {"reason": "no training seasons", "train_rows": 0})
        kw = {} if self.min_fit_rows is None else {"min_rows": self.min_fit_rows}
        return fit_adjustment(ts.design, ts.resid, ts.weight, ts.season, **kw)

    def project(self, history: History) -> pd.DataFrame:
        history.assert_no_future()
        base = self.base.project(history)
        out = base.copy()
        clock = contract_clock(history.players, history.target_season, out["player_id"].to_numpy("int64"))
        out["contract_years_since_draft"] = clock.years_since_draft
        out["contract_scale_year"] = clock.scale_year
        out["contract_flag"] = _flags(clock)
        out["contract_adj"] = 0.0
        out["contract_factor"] = 1.0
        out["contract_enabled"] = False
        fit = self._fit(history)
        self.last_fit = fit
        if fit.enabled:
            adj = fit.adjustment(clock)
            factor = factor_from_adjustment(out["proj_fppg"].to_numpy(float), adj)
            for col in PROJECTION_STATS:
                out[col] = out[col].to_numpy(float) * factor
            new_fppg = fantasy_points_frame(out, self.scoring, prefix="proj_").to_numpy()
            out["contract_adj"] = new_fppg - base["proj_fppg"].to_numpy(float)
            out["contract_factor"] = factor
            out["proj_fppg"] = new_fppg
            out["proj_total_fp"] = out["proj_fppg"] * out["proj_gp"]
            for c in ("fppg_p10", "fppg_p50", "fppg_p90", "proj_fppg_sd"):
                if c in out.columns:
                    out[c] = out[c].to_numpy(float) * factor
            out["contract_enabled"] = True
        out["model"] = self.name
        return validate_table(out, "projections")
