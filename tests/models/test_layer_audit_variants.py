"""Drop-one variants of the board stack (ADR 0043): registered, wired as "full stack minus one layer", and runnable."""
import pytest
from model_testkit import history_for, make_league

from src.contracts import validate_table
from src.models.adp_baseline import AdpHooks
from src.models.depth_baseline import DepthHooks
from src.models.offseason_baseline import BaselineOffseasonProjector
from src.models.registry import available_projectors, get_projector

FULL = "baseline_hurdle_adp_depth_rookie_offseason_debut"
DROP_ONE = {
    "hurdle": "baseline_adp_depth_rookie_offseason_debut",
    "adp": "baseline_hurdle_depth_rookie_offseason_debut",
    "depth": "baseline_hurdle_adp_rookie_offseason_debut",
    "rookie": "baseline_hurdle_adp_depth_offseason_debut",
    "offseason": "baseline_hurdle_adp_depth_rookie_debut",
    "debutants": "baseline_hurdle_adp_depth_rookie_offseason",
}


def _layers(p) -> dict[str, bool]:
    offseason = isinstance(p, BaselineOffseasonProjector)
    base = p.base if offseason else p
    return {"hurdle": base.config.appearance_hurdle and base.config.season_intervals,
            "adp": isinstance(base, AdpHooks), "depth": isinstance(base, DepthHooks),
            "rookie": bool(getattr(base, "rookie_appearance", False)), "offseason": offseason,
            "debutants": bool(getattr(base, "add_debutants", False))}


def test_full_stack_has_every_layer():
    assert all(_layers(get_projector(FULL)).values())


@pytest.mark.parametrize("layer,name", DROP_ONE.items())
def test_each_variant_is_the_full_stack_minus_exactly_one_layer(layer, name):
    assert name in available_projectors()
    p = get_projector(name)
    assert p.name == name
    layers = _layers(p)
    assert not layers.pop(layer) and all(layers.values())


@pytest.mark.parametrize("name", DROP_ONE.values())
def test_each_variant_projects_synthetic_data(name, tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "empty"))
    out = get_projector(name).project(history_for(make_league()))
    validate_table(out, "projections")
    assert len(out) and (out["model"] == name).all()
