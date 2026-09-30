"""Name -> projector registry. The backtest CLI, the ablation runner and the draft board import this.

Stable API (do not change without telling the backtest track)::

    available_projectors() -> list[str]            # sorted names
    get_projector(name, **kwargs) -> Projector      # fresh instance; kwargs go to the constructor
    register_projector(name, factory)               # add a model (e.g. an ablation variant)

Built-ins: ``"naive_last_season"``, ``"baseline"``, ``"baseline_injury"`` (ADR 0006), ``"baseline_return"``, ``"baseline_injury_return"`` and ``"baseline_return_offseason_debut"`` (ADR 0022),
``"baseline_roster"`` (ADR 0010), ``"baseline_transactions"`` (ADR 0011), ``"baseline_contract"`` (ADR 0013), ``"baseline_contract_terms"`` (ADR 0019), ``"baseline_coach"`` (ADR 0020) and the offseason family
(ADR 0012): ``"baseline_offseason"`` (Summer League + preseason), ``"baseline_summer_league"`` and
``"baseline_preseason"`` (one context each, for the ablation), and the debutant family (ADR 0016):
``"baseline_debut"`` (baseline plus stash and undrafted debutants) and ``"baseline_offseason_debut"``. Every
projector returned satisfies the ``Projector`` protocol from ``src.contracts`` and its ``.name``
equals the registry name, so the ``model`` column of its projections identifies it.
"""
from __future__ import annotations

from typing import Callable

from src.contracts import Projector
from src.models.baseline import BaselineProjector
from src.models.config import BaselineConfig
from src.models.coach_baseline import BaselineCoachProjector
from src.models.contract_baseline import BaselineContractProjector
from src.models.contract_terms_baseline import BaselineContractTermsProjector
from src.models.debutants import DebutantBaselineProjector
from src.models.injury_baseline import BaselineInjuryProjector
from src.models.naive import NaiveLastSeason
from src.models.offseason_baseline import BaselineOffseasonProjector
from src.models.return_baseline import BaselineInjuryReturnProjector, BaselineReturnProjector, ReturnDebutantProjector
from src.models.roster_baseline import BaselineRosterProjector
from src.models.transactions_baseline import BaselineTransactionsProjector

_REGISTRY: dict[str, Callable[..., Projector]] = {
    "naive_last_season": lambda **kw: NaiveLastSeason(**kw),
    "baseline": lambda **kw: BaselineProjector(**kw),
    "baseline_nopos": lambda **kw: BaselineProjector(
        **{"config": BaselineConfig(use_position_priors=False), "name": "baseline_nopos", **kw}),
    "baseline_injury": lambda **kw: BaselineInjuryProjector(**kw),
    "baseline_return": lambda **kw: BaselineReturnProjector(**kw),
    "baseline_injury_return": lambda **kw: BaselineInjuryReturnProjector(**kw),
    "baseline_roster": lambda **kw: BaselineRosterProjector(**kw),
    "baseline_transactions": lambda **kw: BaselineTransactionsProjector(**kw),
    "baseline_contract": lambda **kw: BaselineContractProjector(**kw),
    "baseline_contract_terms": lambda **kw: BaselineContractTermsProjector(**kw),
    "baseline_coach": lambda **kw: BaselineCoachProjector(**kw),
    "baseline_offseason": lambda **kw: BaselineOffseasonProjector(**kw),
    "baseline_summer_league": lambda **kw: BaselineOffseasonProjector(
        **{"contexts": ("summer_league",), "name": "baseline_summer_league", **kw}),
    "baseline_preseason": lambda **kw: BaselineOffseasonProjector(
        **{"contexts": ("preseason",), "name": "baseline_preseason", **kw}),
    "baseline_debut": lambda **kw: DebutantBaselineProjector(**kw),
    "baseline_origin": lambda **kw: DebutantBaselineProjector(
        **{"add_debutants": False, "origin": True, "name": "baseline_origin", **kw}),
    "baseline_offseason_debut": lambda **kw: BaselineOffseasonProjector(
        **{"base": DebutantBaselineProjector(), "name": "baseline_offseason_debut", **kw}),
    "baseline_return_offseason_debut": lambda **kw: BaselineOffseasonProjector(
        **{"base": ReturnDebutantProjector(), "name": "baseline_return_offseason_debut", **kw}),
    "baseline_offseason_rich": lambda **kw: BaselineOffseasonProjector(
        **{"components": True, "name": "baseline_offseason_rich", **kw}),
    "baseline_summer_league_rich": lambda **kw: BaselineOffseasonProjector(
        **{"contexts": ("summer_league",), "components": True, "name": "baseline_summer_league_rich", **kw}),
}


def available_projectors() -> list[str]:
    return sorted(_REGISTRY)


def register_projector(name: str, factory: Callable[..., Projector], *, overwrite: bool = False) -> None:
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"projector {name!r} already registered")
    _REGISTRY[name] = factory


def get_projector(name: str, **kwargs) -> Projector:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown projector {name!r}; available: {available_projectors()}") from None
    proj = factory(**kwargs)
    if proj.name != name:
        # A registered factory that returns a differently named model would corrupt the model column.
        proj.name = name
    return proj
