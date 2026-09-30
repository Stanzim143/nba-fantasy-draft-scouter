"""``BaselineRosterProjector``: BaselineProjector + team-context adjustment on minutes (ADR 0010).

Identical to ``BaselineProjector`` in every component (per-minute rates, availability, volatility,
rookies) except minutes, which gets an additive adjustment built by
``src.features.roster.RosterFeatures`` from the history's own ``game_logs``/``team_games`` (see
that module's docstring): a fitted, recency-weighted, lagged team-context signal (pace, role share,
positional crowding) predicting the residual between a player's actual mpg and the context-free
minutes prior.

Wiring uses the single ``BaselineProjector._build_roster_features`` hook (returns ``None`` in the
base class); nothing else about ``BaselineProjector`` or ``FittedBaseline`` needed to change for
plain ``baseline`` runs, which leave ``roster_features`` at its default ``None``.
"""
from __future__ import annotations

import numpy as np

from src.contracts import History
from src.features.roster import RosterFeatures
from src.models.baseline import BaselineProjector


class BaselineRosterProjector(BaselineProjector):
    """Registered as ``"baseline_roster"`` in ``src.models.registry``."""

    name = "baseline_roster"

    def _build_roster_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                               weight: np.ndarray) -> RosterFeatures:
        return RosterFeatures.fit(
            history, sub.pids, sub.target_s, mpg_est, actual_mpg, weight,
            n_lags=self.config.n_lags, decay=self.config.default_decay)
