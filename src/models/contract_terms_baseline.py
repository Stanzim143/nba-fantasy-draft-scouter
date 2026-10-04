"""``BaselineContractTermsProjector``: the baseline plus a veteran contract-terms adjustment (ADR 0019).

A stacked projector in the ADR 0013 mould (it subclasses ``BaselineContractProjector`` only to reuse its
memoised walk-forward base projections and history plumbing; the rookie-scale clock itself is *inside* the
terms features, applied only where no Wikipedia-stated contract is known). For each earlier season the base
projector is re-run on ``History.until(s)`` and its ``proj_fppg`` residual against the season's actual FPPG is
regressed, by ridge WLS, on the as-of-season-start terms features of ``src.features.contract_terms``:
final year of a known contract, years left, extension, new deal signed this offseason, two-way / minimum
deal, lapsed contract, and the nominal rookie-scale year for first-rounders with no known deal. Only the
contrast against players with no known state is applied, as a per-player multiplier on every counting stat
clipped to [0.8, 1.25]. The layer switches itself off (base projection, ``terms_enabled = False``) unless
leave-one-season-out cross-validation beats "no adjustment" by 0.05%, or too few rows / seasons / marked rows
exist. Games played, minutes and volatility are untouched.

The contract events come from ``History.extras["player_contracts"]`` (already sliced by ``History.until``: the
table's ``season`` tag is the season the event follows) or, when the caller did not supply it, from
``player_contracts.parquet`` (``python -m src.ingest.wiki_contracts``), sliced here by the same tag rule. With
neither, the result is the base projection with the display columns and ``terms_enabled = False``.

Extra output columns (allowed by the ``projections`` contract): ``terms_adj``, ``terms_factor``, ``terms_enabled``
and the display columns ``terms_status`` (known / no_length / lapsed / unknown), ``terms_years_remaining``,
``terms_flag`` (``contract_year``, ``extension``, ``new_deal``, ``rookie_contract_year``, ``rookie_option_year``
or empty), ``terms_basis`` (``wiki`` / ``rookie_scale`` / empty) and ``terms_contract_year`` (True only when a
final year is *known*; False means "not known to be", never "known not to be").
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.contracts import PROJECTION_STATS, History, Projector, season_start, season_str, validate_table
from src.features.contract import ContractFit, contract_clock, factor_from_adjustment
from src.features.contract_terms import (
    build_terms_training_set, fit_terms_adjustment, slice_contracts, terms_features,
)
from src.models.contract_baseline import BaselineContractProjector
from src.models.panel import canonical_logs
from src.value.frame import fantasy_points_frame


class BaselineContractTermsProjector(BaselineContractProjector):
    """Registered as ``"baseline_contract_terms"`` in ``src.models.registry``."""

    name = "baseline_contract_terms"

    def __init__(self, base: Projector | None = None, scoring: Mapping[str, float] | None = None, *,
                 contracts: pd.DataFrame | None = None, min_fit_rows: int | None = None, min_cv_gain: float | None = None,
                 name: str | None = None, walk_forward_cache: dict | None = None):
        super().__init__(base, scoring, min_fit_rows=min_fit_rows, name=name, walk_forward_cache=walk_forward_cache)
        self._contracts = contracts
        self.min_cv_gain = min_cv_gain          # None = the ADR 0013 gate (0.05%); a diagnostic can force the layer on

    # ------------------------------------------------------------------ data access

    def _contract_events(self, history: History) -> pd.DataFrame | None:
        c = self._contracts if self._contracts is not None else history.extras.get("player_contracts")
        if c is None:
            try:
                from src.ingest.wiki_contracts import read_player_contracts

                c = read_player_contracts()
            except (ImportError, FileNotFoundError):
                return None
        return slice_contracts(c, history.target_season)

    # ------------------------------------------------------------------ fit

    def _fit_terms(self, history: History, events: pd.DataFrame) -> ContractFit:
        tables = self._history_tables(history)
        gl = canonical_logs(history.game_logs)
        gl = gl.assign(fp=fantasy_points_frame(gl, self.scoring), s=gl["season"].map(season_start))
        actual = gl.groupby(["player_id", "s"]).agg(gp=("game_id", "size"), fppg=("fp", "mean")).reset_index()
        seasons = sorted(gl["s"].unique())
        train_seasons = [season_str(int(s)) for s in seasons[1:]]
        ts = build_terms_training_set(history.players, events, train_seasons,
                                      lambda s: self._base_projection(tables, s), actual)
        if ts is None:
            return ContractFit(None, None, 0.0, False, {"reason": "no training seasons", "train_rows": 0})
        fit = fit_terms_adjustment(ts, min_rows=self.min_fit_rows, min_gain=self.min_cv_gain)
        fit.diagnostics["known_rows"] = int(ts.known.sum())
        fit.diagnostics["rows"] = int(len(ts.known))
        return fit

    # ------------------------------------------------------------------ projection

    def project(self, history: History) -> pd.DataFrame:
        history.assert_no_future()
        base = self.base.project(history)
        out = base.copy()
        events = self._contract_events(history)
        pids = out["player_id"].to_numpy("int64")
        clock = contract_clock(history.players, history.target_season, pids)
        tf = terms_features(events, history.target_season, pids, clock)
        out["terms_status"] = tf.status
        out["terms_years_remaining"] = tf.years_remaining
        out["terms_flag"] = tf.flag()
        out["terms_basis"] = tf.basis()
        out["terms_contract_year"] = tf.contract_year
        out["terms_adj"] = 0.0
        out["terms_factor"] = 1.0
        out["terms_enabled"] = False
        if events is None or events.empty:
            self.last_fit = ContractFit(None, None, 0.0, False, {"reason": "no contract events", "train_rows": 0})
            out["model"] = self.name
            return validate_table(out, "projections")
        fit = self._fit_terms(history, events)
        self.last_fit = fit
        if fit.enabled:
            adj = fit.adjustment_from_design(tf.design())
            factor = factor_from_adjustment(out["proj_fppg"].to_numpy(float), adj)
            for col in PROJECTION_STATS:
                out[col] = out[col].to_numpy(float) * factor
            new_fppg = fantasy_points_frame(out, self.scoring, prefix="proj_").to_numpy()
            out["terms_adj"] = new_fppg - base["proj_fppg"].to_numpy(float)
            out["terms_factor"] = factor
            out["proj_fppg"] = new_fppg
            out["proj_total_fp"] = out["proj_fppg"] * out["proj_gp"]
            for c in ("fppg_p10", "fppg_p50", "fppg_p90", "proj_fppg_sd",
                      "proj_total_fp_p10", "proj_total_fp_p50", "proj_total_fp_p90"):
                if c in out.columns:
                    out[c] = out[c].to_numpy(float) * factor
            out["terms_enabled"] = True
        out["model"] = self.name
        return validate_table(out, "projections")
