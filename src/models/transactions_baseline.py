"""``BaselineTransactionsProjector``: BaselineProjector + transactions-context adjustment on minutes
(ADR 0011).

Identical to ``BaselineProjector`` in every component (per-minute rates, availability, volatility,
rookies) except minutes, which gets an additive adjustment built by
``src.features.transactions.TransactionFeatures`` from the dated ``team_transactions`` source (see
that module's docstring): a fitted, directly-dated (not lagged) team-transactions signal
(``changed_team``, ``team_departures_lost``) predicting the residual between a player's actual mpg
and the context-free minutes prior.

Wiring uses the single ``BaselineProjector._build_transactions_features`` hook (returns ``None`` in
the base class); nothing else about ``BaselineProjector`` or ``FittedBaseline`` needed to change for
plain ``baseline`` (or ``baseline_roster``) runs, which leave ``transactions_features`` at its
default ``None``. This is an independent SIBLING of ``BaselineRosterProjector`` (ADR 0010) -- it
subclasses plain ``BaselineProjector``, not ``BaselineRosterProjector`` -- so ``baseline_roster``'s
own already-reported result stays exactly as ablated (ADR 0011 D5).

``team_transactions.parquet`` is produced by ``src.ingest.wiki_transactions`` (a concurrent,
independently-built ingest module per ADR 0011). If that file has not been ingested yet (or the
module itself doesn't exist yet), ``_build_transactions_features`` returns ``None`` -- the same
graceful "no enough data" fallback every other feature layer in this codebase uses -- so
``baseline_transactions`` degrades to plain ``baseline`` rather than crashing.
"""
from __future__ import annotations

import numpy as np

from src.contracts import History
from src.features.transactions import TransactionFeatures
from src.models.baseline import BaselineProjector


class BaselineTransactionsProjector(BaselineProjector):
    """Registered as ``"baseline_transactions"`` in ``src.models.registry``."""

    name = "baseline_transactions"

    def _build_transactions_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                                     weight: np.ndarray) -> TransactionFeatures | None:
        try:
            from src.ingest.wiki_transactions import read_team_transactions
        except ImportError:
            # src/ingest/wiki_transactions.py is built by a concurrent track; not there yet.
            return None
        try:
            transactions = read_team_transactions()
        except FileNotFoundError:
            # Real ingest hasn't been run yet -- graceful no-op, same philosophy as every other
            # "not enough data" fallback in this codebase.
            return None
        return TransactionFeatures.fit(history, transactions, sub.pids, sub.target_s, mpg_est, actual_mpg, weight)
