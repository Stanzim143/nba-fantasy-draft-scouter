"""Baselines with extra preseason context on availability and minutes: ADP (ADR 0032) and labelled injuries (ADR 0033).

``ContextHooks`` is a mixin for any ``BaselineProjector`` subclass. Through the existing feature hooks it supplies

* availability (and, for the hurdle stage, appearance) extra columns: ``AdpFeatures.build`` (``[covered, listed, z]``) and/or
  ``InjuryLabelFeatures.build`` (``[covered, inj_out, rest_out, other_out, longest, n_spells]``), concatenated in that order;
* a fitted additive minutes adjustment from ADP (the ``_build_coach_features`` slot is generic: any object with
  ``build(pids, target_s)`` that returns an mpg delta).

ADP comes from the constructor (``adp``: ``season, player_id, adp``) or the ingested ``adp`` table; the target season's own ADP
is preseason information and is used, later seasons never are. Injury reports come from ``History.extras['injury_reports']``
(sliced to seasons before the target by ``History.until``) or the ingested ``injury_reports`` table, sliced the same way. With no
data a projector degrades to its base model (every feature column is zero), it does not fail.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.contracts import History, season_start
from src.features.adp import AdpFeatures, load_store_adp
from src.features.injury_labels import InjuryLabelFeatures
from src.features.return_health import ConcatFeatures
from src.models.baseline import BaselineProjector
from src.models.debutants import DebutantBaselineProjector

LOGGER = logging.getLogger(__name__)

ADP_COLS, LABEL_COLS = 3, 6       # widths of AdpFeatures.build / InjuryLabelFeatures.build


class _AdpMinutes:
    """Adapter giving ``AdpFeatures`` the ``build(pids, target_s)`` shape of a minutes adjustment."""

    def __init__(self, feats: AdpFeatures):
        self.feats = feats

    def build(self, pids, target_s) -> np.ndarray:
        return self.feats.build_minutes(pids, target_s)


class _Zeros:
    """Stand-in for a missing data source: ``n_cols`` zero columns, so fit and predict always see the same matrix width."""

    def __init__(self, n_cols: int):
        self.n_cols = n_cols

    def build(self, pids, target_s, age=None) -> np.ndarray:
        return np.zeros((len(pids), self.n_cols))


class ContextHooks:
    use_adp = False
    use_labels = False

    def __init__(self, *args, adp: pd.DataFrame | None = None, injury_reports: pd.DataFrame | None = None, **kw):
        super().__init__(*args, **kw)
        self._adp, self._reports = adp, injury_reports
        self._adp_feats: AdpFeatures | None = None
        self._adp_target: str | None = None

    # -- ADP
    def _adp_table(self) -> pd.DataFrame | None:
        if self._adp is None:
            try:
                self._adp = load_store_adp()
            except (FileNotFoundError, ImportError, OSError) as e:
                LOGGER.warning("ADP table unavailable (%s): the ADP-aware model runs WITHOUT ADP (zeroed ADP columns)", e)
                return None
        return self._adp

    def _adp_features(self, history: History) -> AdpFeatures | None:
        if self._adp_target != history.target_season:
            adp = self._adp_table()
            self._adp_feats = None if adp is None else AdpFeatures.from_table(adp, history.target_season)
            self._adp_target = history.target_season
        return self._adp_feats

    # -- labelled injuries
    def _label_features(self, history: History) -> InjuryLabelFeatures | None:
        reports = history.extras.get("injury_reports") if self._reports is None else self._reports
        if reports is None:
            try:
                from src.ingest.nba_injury_reports import read_injury_reports

                reports = read_injury_reports()
            except (FileNotFoundError, ImportError, OSError) as e:
                LOGGER.warning("labelled injury reports unavailable (%s): the model runs WITHOUT injury-label features", e)
                return None
        cutoff = season_start(history.target_season)
        reports = reports[reports["season"].map(season_start) < cutoff]
        if reports.empty or history.team_games.empty:
            return None
        return InjuryLabelFeatures.from_reports(reports, history.team_games, history.target_season,
                                                n_lags=self.config.n_lags, decay=self.config.availability_decay)

    def _build_injury_features(self, history: History, sub):
        if not self.use_adp and not self.use_labels:
            return None
        parts = []
        if self.use_adp:
            adp_feats = self._adp_features(history)
            if adp_feats is None or int(season_start(history.target_season)) not in adp_feats.covered:
                LOGGER.warning("no ADP covers target season %s: ADP is ignored by the model", history.target_season)
            parts.append(adp_feats or _Zeros(ADP_COLS))
        if self.use_labels:
            parts.append(self._label_features(history) or _Zeros(LABEL_COLS))
        return parts[0] if len(parts) == 1 else ConcatFeatures(parts)

    def _build_coach_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                              weight: np.ndarray) -> _AdpMinutes | None:
        if not self.use_adp:
            return None
        feats = self._adp_features(history)
        if feats is None:
            return None
        fitted = AdpFeatures(feats.table, feats.covered).fit_minutes(sub.pids, sub.target_s, mpg_est, actual_mpg, weight)
        return _AdpMinutes(fitted)


class AdpHooks(ContextHooks):
    use_adp = True


class LabelHooks(ContextHooks):
    use_labels = True


class AdpLabelHooks(ContextHooks):
    use_adp = True
    use_labels = True


class BaselineAdpProjector(AdpHooks, BaselineProjector):
    """Registered as ``baseline_adp`` (and, with the hurdle config, ``baseline_hurdle_adp``)."""

    name = "baseline_adp"


class BaselineLabelsProjector(LabelHooks, BaselineProjector):
    """Registered as ``baseline_labels`` / ``baseline_hurdle_labels``."""

    name = "baseline_labels"


class BaselineAdpLabelsProjector(AdpLabelHooks, BaselineProjector):
    """Registered as ``baseline_hurdle_adp_labels``."""

    name = "baseline_adp_labels"


class AdpDebutantProjector(AdpHooks, DebutantBaselineProjector):
    """``DebutantBaselineProjector`` whose veterans come from the ADP-aware baseline; the base of the board stack."""

    name = "baseline_adp_debut"


class AdpLabelsDebutantProjector(AdpLabelHooks, DebutantBaselineProjector):
    """Debutant stack whose veterans use ADP and the labelled injury features."""

    name = "baseline_adp_labels_debut"
