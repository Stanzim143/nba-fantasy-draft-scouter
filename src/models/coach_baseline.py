"""``BaselineCoachProjector``: the baseline plus a coaching-change minutes adjustment (ADR 0020).

Identical to ``BaselineProjector`` in every component except minutes, which gets the additive adjustment built by
``src.features.coach.CoachFeatures``: a fitted, dated (not lagged) signal for teams whose opening head coach is not the
coach they finished last season with, driven by how that coach ran his earlier teams (star and top-five minutes, rotation
depth). Wiring uses the ``BaselineProjector._build_coach_features`` hook (``None`` in the base class), so plain ``baseline``
and every other layer are untouched.

Inputs read from the data directory: ``team_coaches`` (``src.ingest.wiki_coaches``; missing means the layer degrades to plain
``baseline``) and, when present, ``team_transactions`` (ADR 0011) to place players who changed team before the season. For the
live season the newest ``roster_snapshots`` day places every rostered player on his current team.
"""
from __future__ import annotations

import numpy as np

from src.contracts import History, season_start
from src.features.coach import CoachFeatures
from src.models.baseline import BaselineProjector


def _live_team_map(history: History) -> dict:
    """``player_id -> team_id`` from the newest roster snapshot, only when the target season has no games yet (live)."""
    try:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        snap = latest_snapshot(read_roster_snapshots())
    except (FileNotFoundError, ImportError, OSError):
        return {}
    if snap.empty or str(snap["season"].iloc[0]) != history.target_season:
        return {}
    return dict(zip(snap["player_id"].astype(int), snap["team_id"].astype(int)))


class BaselineCoachProjector(BaselineProjector):
    """Registered as ``"baseline_coach"`` in ``src.models.registry``."""

    name = "baseline_coach"

    def _build_coach_features(self, history: History, sub, mpg_est: np.ndarray, actual_mpg: np.ndarray,
                              weight: np.ndarray) -> CoachFeatures | None:
        try:
            from src.ingest.wiki_coaches import read_team_coaches

            coaches = read_team_coaches()
        except (ImportError, FileNotFoundError):
            return None
        transactions = None
        try:
            from src.ingest.wiki_transactions import read_team_transactions

            transactions = read_team_transactions()
        except (ImportError, FileNotFoundError):
            pass
        live = _live_team_map(history) if season_start(history.target_season) >= self._live_start() else {}
        feats = CoachFeatures.build_context(history, coaches, transactions, live_team=live)
        return feats.fit(sub.pids, sub.target_s, mpg_est, actual_mpg, weight)

    @staticmethod
    def _live_start() -> int:
        from src.ingest.nba_offseason import live_season_start

        return live_season_start()
