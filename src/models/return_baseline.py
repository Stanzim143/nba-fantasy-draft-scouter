"""``BaselineReturnProjector``: BaselineProjector + 'returned and healthy' features on availability (ADR 0022).

Identical to ``BaselineProjector`` in every component except the availability model, which gets the seven extra columns built by
``src.features.return_health.ReturnFeatures`` (lead-block share of the season, tail health after the first appearance, Task A's
return flag; see that module). Wiring uses the single ``BaselineProjector._build_injury_features`` hook, exactly like
``BaselineInjuryProjector`` (ADR 0006); ``baseline`` itself is untouched and bit-identical.

Variants (all registered in ``src.models.registry``):

* ``baseline_return``               return features only.
* ``baseline_injury_return``        the four injury-layer columns plus the seven return columns (one ``extra`` matrix).
* ``baseline_return_offseason_debut``  the board's stack (debutants + Summer League / preseason) on top of ``baseline_return``
  instead of ``baseline``; available to the board, not the default.
"""
from __future__ import annotations

from src.contracts import History
from src.features.injury import InjuryFeatures
from src.features.return_health import ConcatFeatures, ReturnFeatures
from src.models.baseline import BaselineProjector
from src.models.debutants import DebutantBaselineProjector


def _return_features(proj: BaselineProjector, history: History) -> ReturnFeatures:
    return ReturnFeatures.fit(history, n_lags=proj.config.n_lags, decay=proj.config.availability_decay)


class BaselineReturnProjector(BaselineProjector):
    """Registered as ``"baseline_return"``."""

    name = "baseline_return"

    def _build_injury_features(self, history: History, sub) -> ReturnFeatures:
        return _return_features(self, history)


class BaselineInjuryReturnProjector(BaselineProjector):
    """Registered as ``"baseline_injury_return"``: ADR 0006's injury columns followed by the return columns."""

    name = "baseline_injury_return"

    def _build_injury_features(self, history: History, sub) -> ConcatFeatures:
        injury = InjuryFeatures.fit(history, sub.age, sub.pids, sub.target_s,
                                    n_lags=self.config.n_lags, decay=self.config.availability_decay)
        return ConcatFeatures([injury, _return_features(self, history)])


class ReturnDebutantProjector(DebutantBaselineProjector):
    """``DebutantBaselineProjector`` whose veterans come from ``baseline_return``; the base of ``baseline_return_offseason_debut``."""

    name = "baseline_return_debut"

    def _build_injury_features(self, history: History, sub) -> ReturnFeatures:
        return _return_features(self, history)
