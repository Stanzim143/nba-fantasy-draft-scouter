"""``BaselineDepthProjector``: BaselineProjector + departed-minutes / playoff-closing-role minutes adjustment (ADR 0035).

Same wiring as ``BaselineTransactionsProjector`` (the ``_build_transactions_features`` hook); the two never coexist. Missing
``team_transactions`` or ``playoff_logs`` degrade to that input being absent (zero feature), and with neither the model is
bit-identical to plain ``baseline``.
"""
from __future__ import annotations

import numpy as np

from src.contracts import History
from src.features.depth import DepthFeatures
from src.models.adp_baseline import AdpHooks
from src.models.baseline import BaselineProjector
from src.models.debutants import DebutantBaselineProjector


class DepthHooks:
    """Mixin: fills the (otherwise unused) transactions slot of ``BaselineProjector`` with :class:`DepthFeatures`."""

    def _build_transactions_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                                     weight: np.ndarray) -> DepthFeatures:
        transactions = playoffs = None
        try:
            from src.ingest.wiki_transactions import read_team_transactions
            transactions = read_team_transactions()
        except (ImportError, FileNotFoundError):
            pass
        try:
            from src.ingest.nba_playoffs import read_playoff_logs
            playoffs = read_playoff_logs()
        except FileNotFoundError:
            pass
        return DepthFeatures.fit(history, transactions, playoffs, sub.pids, sub.target_s, mpg_est, actual_mpg, weight)


class BaselineDepthProjector(DepthHooks, BaselineProjector):
    """Registered as ``"baseline_depth"`` in ``src.models.registry``."""

    name = "baseline_depth"


class AdpDepthDebutantProjector(DepthHooks, AdpHooks, DebutantBaselineProjector):
    """The board stack's base (``AdpDebutantProjector``) plus the depth minutes layer."""

    name = "baseline_adp_depth_debut"


class DepthDebutantProjector(DepthHooks, DebutantBaselineProjector):
    """The board stack's base without ADP-as-input: debutants plus the depth minutes layer (ADR 0043 drop-one variant)."""

    name = "baseline_depth_debut"
