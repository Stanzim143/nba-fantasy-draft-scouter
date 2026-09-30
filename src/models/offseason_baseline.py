"""``BaselineOffseasonProjector``: the baseline plus a Summer League / preseason adjustment (ADR 0012).

A *stacked* projector: it wraps any base projector (default ``BaselineProjector``), and multiplies each
player's projected counting stats by a factor learned from how Summer League and preseason performance
explained the base model's past misses (see ``src.features.offseason`` for the method and the leakage
argument). Everything else in the base projection (games played, minutes, volatility, rookie prior) is
untouched, so the ablation isolates exactly this layer.

Data comes from ``History.extras`` when present (already sliced by ``History.until``, and covered by the
backtest's leakage checks), else from the parquet files ``python -m src.ingest.nba_offseason`` writes, sliced
here by the same rule. With neither, or with too little history to fit, the result is the base projection
unchanged and ``offseason_enabled`` is False, the same graceful fallback every feature layer here uses.

Extra output columns (allowed by the ``projections`` contract): ``offseason_adj`` (fantasy points per game
added), ``offseason_factor``, and the player's own summer/preseason line (``sl_*``, ``pre_*``) for display.
"""
from __future__ import annotations

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from src.contracts import PROJECTION_STATS, History, Projector, season_start, season_str, validate_table
from src.features.offseason import (
    MIN_FIT_ROWS, PRESEASON, SUMMER_LEAGUE, OffseasonFit, assert_offseason_no_future, build_training_set, event_player_table, factor_from_adjustment,
    event_columns, fit_adjustment, slice_offseason,
)
from src.models.baseline import BaselineProjector
from src.models.panel import AgeLookup, canonical_logs
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

DISPLAY_COLUMNS = ["sl_gp", "sl_mpg", "sl_fp36", "sl_z", "pre_gp", "pre_mpg", "pre_fp36", "pre_z"]


def _fingerprint(tables: Mapping[str, pd.DataFrame]) -> tuple:
    gl = tables["game_logs"]
    return (len(gl), int(gl["pts"].sum()) if len(gl) else 0, len(tables["players"]),
            int(pd.to_numeric(tables["players"]["draft_year"], errors="coerce").fillna(0).sum()))


class BaselineOffseasonProjector:
    """Registered as ``"baseline_offseason"`` in ``src.models.registry``."""

    name = "baseline_offseason"

    def __init__(self, base: Projector | None = None, scoring: Mapping[str, float] | None = None, *,
                 offseason_logs: pd.DataFrame | None = None, offseason_team_games: pd.DataFrame | None = None,
                 as_of: date | None = None, contexts: tuple[str, ...] = (SUMMER_LEAGUE, PRESEASON),
                 preseason_fraction: float = 1.0, components: bool = False, min_fit_rows: int = MIN_FIT_ROWS,
                 name: str | None = None, walk_forward_cache: dict | None = None):
        self.scoring: Mapping[str, float] = dict(scoring) if scoring is not None else dict(load_league()["scoring"])
        self.base: Projector = base if base is not None else BaselineProjector(self.scoring)
        self._logs = offseason_logs
        self._team_games = offseason_team_games
        self.as_of = as_of
        self.contexts = tuple(contexts)
        self.preseason_fraction = float(preseason_fraction)
        self.components = bool(components)
        self.min_fit_rows = min_fit_rows
        if name is not None:
            self.name = name
        self.last_fit: OffseasonFit | None = None
        # Base projections of past seasons do not depend on the offseason data or on the target, so callers that
        # run several variants over the same data (the ablation, the tests) can share one cache between them.
        self._walk_forward: dict[tuple, pd.DataFrame] = walk_forward_cache if walk_forward_cache is not None else {}

    # ------------------------------------------------------------------ data access

    def _offseason_tables(self, history: History) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        logs = self._logs if self._logs is not None else history.extras.get("offseason_logs")
        tg = self._team_games if self._team_games is not None else history.extras.get("offseason_team_games")
        if logs is None:
            try:
                from src.ingest.nba_offseason import read_offseason_logs, read_offseason_team_games

                logs = read_offseason_logs()
                tg = tg if tg is not None else read_offseason_team_games()
            except (ImportError, FileNotFoundError):
                return None, None
        logs = slice_offseason(logs, history.target_season)
        tg = slice_offseason(tg, history.target_season) if tg is not None else None
        assert_offseason_no_future(logs, history.target_season)
        return logs, tg

    def _history_tables(self, history: History) -> dict[str, pd.DataFrame]:
        return {"game_logs": history.game_logs, "team_games": history.team_games, "players": history.players,
                "player_season_bio": history.player_season_bio, **history.extras}

    def _base_projection(self, tables: dict[str, pd.DataFrame], season: str) -> pd.DataFrame | None:
        """Base projection of ``season`` from data before it, memoised (it does not depend on the target)."""
        h = History.until(tables, season)
        if h.game_logs.empty:
            return None
        key = (season, type(self.base).__name__, _fingerprint({"game_logs": h.game_logs, "players": h.players}))
        if key not in self._walk_forward:
            self._walk_forward[key] = self.base.project(h)
        return self._walk_forward[key]

    # ------------------------------------------------------------------ projection

    def project(self, history: History) -> pd.DataFrame:
        history.assert_no_future()
        base = self.base.project(history)
        out = base.copy()
        out["offseason_adj"] = 0.0
        out["offseason_factor"] = 1.0
        for c in DISPLAY_COLUMNS:
            out[c] = np.nan
        out["offseason_enabled"] = False
        logs, team_games = self._offseason_tables(history)
        if logs is None or logs.empty:
            self.last_fit = None
            return self._finish(out)

        events = event_player_table(logs, team_games, self.scoring, as_of=self.as_of, contexts=self.contexts,
                                    preseason_fraction=self.preseason_fraction)
        target = history.target_season
        tgt_ev = events[events["event_season"] == target].drop(columns="event_season")
        merged = out[["player_id"]].merge(tgt_ev, on="player_id", how="left")
        for c in DISPLAY_COLUMNS:
            out[c] = merged[c].to_numpy()

        fit = self._fit(history, events)
        self.last_fit = fit
        if fit.enabled:
            ev_cols = event_columns(self.components)
            for c in ev_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
            adj = fit.adjustment(merged[ev_cols], out["age"].to_numpy(float), out["is_rookie"].to_numpy(bool).astype(float))
            factor = factor_from_adjustment(out["proj_fppg"].to_numpy(float), adj)
            for col in PROJECTION_STATS:
                out[col] = out[col].to_numpy(float) * factor
            new_fppg = fantasy_points_frame(out, self.scoring, prefix="proj_").to_numpy()
            out["offseason_adj"] = new_fppg - base["proj_fppg"].to_numpy(float)
            out["offseason_factor"] = factor
            out["proj_fppg"] = new_fppg
            out["proj_total_fp"] = out["proj_fppg"] * out["proj_gp"]
            for c in ("fppg_p10", "fppg_p50", "fppg_p90", "proj_fppg_sd"):
                if c in out.columns:
                    out[c] = out[c].to_numpy(float) * factor
            out["offseason_enabled"] = True
        return self._finish(out)

    def _finish(self, out: pd.DataFrame) -> pd.DataFrame:
        out["model"] = self.name
        return validate_table(out, "projections")

    def _fit(self, history: History, events: pd.DataFrame) -> OffseasonFit:
        tables = self._history_tables(history)
        gl = canonical_logs(history.game_logs)
        gl = gl.assign(fp=fantasy_points_frame(gl, self.scoring), s=gl["season"].map(season_start))
        actual = gl.groupby(["player_id", "s"]).agg(gp=("game_id", "size"), fppg=("fp", "mean")).reset_index()
        seasons = sorted(gl["s"].unique())
        train_seasons = [season_str(int(s)) for s in seasons[1:]]  # the first season has no prior data to project from
        ages = AgeLookup(history.players, history.player_season_bio)

        def project(s: str) -> pd.DataFrame | None:
            return self._base_projection(tables, s)

        def events_for(s: str) -> pd.DataFrame:
            return events[events["event_season"] == s].drop(columns="event_season")

        ts = build_training_set(tables, train_seasons, project, events_for,
                                lambda pids, sy: ages.at(pids, sy), actual, self.components)
        if ts is None:
            return OffseasonFit(None, None, None, 0.0, 0.0, False, {"reason": "no training seasons", "train_rows": 0}, self.components)
        return fit_adjustment(ts.events, ts.age, ts.rookie, ts.resid, ts.weight, ts.season, min_rows=self.min_fit_rows,
                              components=self.components)
