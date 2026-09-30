"""Load config/league.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "league.yaml"


def load_league(path: Path | str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def starter_slots(cfg: dict) -> int:
    return sum(cfg["league"]["roster"]["starters"].values())


def rostered_players(cfg: dict) -> int:
    """Players across the league with a non-IR roster spot."""
    return cfg["league"]["teams"] * cfg["league"]["roster"]["size"]
