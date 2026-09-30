"""``BaselineInjuryProjector``: BaselineProjector + absence-streak features on availability.

ADR 0006. Identical to ``BaselineProjector`` in every component (minutes, rates, volatility,
rookies) except the availability model, which gets four extra columns built by
``src.features.injury.InjuryFeatures`` from the history's own ``game_logs``/``team_games`` (see
that module's docstring): recency-weighted absence-streak length, recency-weighted absence-streak
frequency, a "coming off a long absence last season" flag, and an age-adjusted absence residual.

Wiring uses the single ``BaselineProjector._build_injury_features`` hook (returns ``None`` in the
base class); nothing else about ``BaselineProjector`` or ``FittedBaseline`` needed to change for
plain ``baseline`` runs, which pass ``extra=None`` through ``AvailabilityModel`` unchanged.
"""
from __future__ import annotations

from src.contracts import History
from src.features.injury import InjuryFeatures
from src.models.baseline import BaselineProjector


class BaselineInjuryProjector(BaselineProjector):
    """Registered as ``"baseline_injury"`` in ``src.models.registry``."""

    name = "baseline_injury"

    def _build_injury_features(self, history: History, sub) -> InjuryFeatures:
        return InjuryFeatures.fit(
            history, sub.age, sub.pids, sub.target_s,
            n_lags=self.config.n_lags, decay=self.config.availability_decay)
