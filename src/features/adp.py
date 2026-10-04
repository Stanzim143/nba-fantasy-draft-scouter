"""ADP as a preseason information source for minutes and availability (ADR 0032).

An average draft position is a market summary of everything known about a player *before* the season: injury news,
role changes, contract and trade context that the game logs cannot show. ``AdpFeatures`` turns it into two things a
projector can use without touching the rest of the model:

* ``build(pids, target_s, age)``   columns for ``AvailabilityModel``'s ``extra`` matrix: ``[covered, listed, z]`` where
  ``covered`` = the season has ADP data at all, ``listed`` = covered and the player has an ADP, ``z`` = ``listed *
  (log ADP - 4)``. A season without ADP coverage (before 2015-16) is all zeros, so those rows still train the base model
  and only ADP-covered rows identify the ADP coefficients.
* ``build_minutes(pids, target_s)``   an additive minutes-per-game adjustment, a ridge fit on ADP-covered training rows of
  ``actual mpg - predicted mpg`` against ``[1, listed, z]`` (weighted by games played).

ADP for season ``s`` is known before ``s`` starts, so using the *target* season's ADP is not leakage; ADP of any later
season is never read (``slice_adp`` drops it, ``assert_adp_no_future`` checks). The table is ``season, player_id, adp``
(``load_store_adp`` maps the ingested ``adp`` table through ``player_id_map``; the lowest ADP wins a duplicate).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import season_start

log = logging.getLogger(__name__)

MIN_MATCH_CONFIDENCE = 0.9     # player_id_map: exact name matches are 1.0, fuzzy matches below it; the map's own floor is 0.9
LOG_ADP_CENTER = 4.0           # log(55): roughly the middle of the listed ADP range
MIN_COVERED_ROWS = 100         # ADP-covered training rows needed to fit the minutes adjustment
RIDGE = 5.0


def load_store_adp(base: Path | None = None, *, source: str = "espn", min_confidence: float = MIN_MATCH_CONFIDENCE) -> pd.DataFrame:
    """``season, player_id, adp`` from the ingested ``adp`` table mapped through ``player_id_map`` (raises if absent).

    Id matches below ``min_confidence`` (exact = 1.0, fuzzy name matches lower) are dropped, and the number dropped is logged.
    """
    from src.ingest.espn_adp import adp_path
    from src.store import read_table

    path = adp_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.espn_adp` run?")
    raw = pd.read_parquet(path)
    ids = read_table("player_id_map", base)
    ids = ids[ids["source"] == source]
    low = ids[ids["confidence"] < min_confidence]
    ids = ids[ids["confidence"] >= min_confidence]
    raw = raw[raw["source"] == source].copy()
    raw["source_id"] = raw["source_id"].astype(str)
    ids = ids[["source_id", "player_id"]].assign(source_id=lambda d: d["source_id"].astype(str))
    if len(low):
        n_rows = int(raw["source_id"].isin(set(low["source_id"].astype(str))).sum())
        log.warning("ADP: dropped %d id match(es) below confidence %.2f (%d ADP rows)", len(low), min_confidence, n_rows)
    m = raw.merge(ids, on="source_id", how="inner")
    m = m.sort_values(["season", "player_id", "adp"], kind="stable").drop_duplicates(["season", "player_id"])
    out = m[["season", "player_id", "adp"]].reset_index(drop=True)
    out["player_id"] = out["player_id"].astype("int64")
    out["adp"] = out["adp"].astype("float64")
    return out


def slice_adp(adp: pd.DataFrame, target_season: str) -> pd.DataFrame:
    """ADP for seasons up to and including ``target_season`` (preseason information); later seasons are dropped."""
    cutoff = season_start(target_season)
    keep = adp["season"].map(season_start) <= cutoff
    return adp[keep].reset_index(drop=True)


def assert_adp_no_future(adp: pd.DataFrame, target_season: str) -> None:
    if len(adp) and (adp["season"].map(season_start) > season_start(target_season)).any():
        raise AssertionError(f"leakage: ADP contains seasons after {target_season}")


@dataclass
class AdpFeatures:
    table: pd.DataFrame                         # indexed (player_id, s): log_adp, one row per listed player-season
    covered: frozenset                          # season starts with any ADP data
    minutes_coef: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [1, listed, z], applied on covered rows
    n_cols: int = field(default=3, init=False)

    @classmethod
    def from_table(cls, adp: pd.DataFrame, target_season: str) -> "AdpFeatures":
        a = slice_adp(adp, target_season)
        assert_adp_no_future(a, target_season)
        s = a["season"].map(season_start).to_numpy("int64")
        tab = pd.DataFrame({"player_id": a["player_id"].to_numpy("int64"), "s": s,
                            "log_adp": np.log(np.maximum(a["adp"].to_numpy(float), 1.0))})
        tab = tab.sort_values(["player_id", "s", "log_adp"], kind="stable")      # the earliest pick wins a duplicate
        return cls(tab.drop_duplicates(["player_id", "s"]).set_index(["player_id", "s"]).sort_index(),
                   frozenset(int(x) for x in np.unique(s)))

    def _parts(self, pids, target_s):
        pids, ts = np.asarray(pids, "int64"), np.asarray(target_s, "int64")
        covered = np.array([int(t) in self.covered for t in ts], float)
        la = self.table["log_adp"].reindex(pd.MultiIndex.from_arrays([pids, ts])).to_numpy(float)
        listed = covered * np.isfinite(la)
        z = np.where(listed > 0, la - LOG_ADP_CENTER, 0.0)
        return covered, listed, z

    def build(self, pids, target_s, age=None) -> np.ndarray:
        """``(n, 3)`` availability-extra columns ``[covered, listed, z]`` (``age`` is accepted for the hook signature)."""
        covered, listed, z = self._parts(pids, target_s)
        return np.column_stack([covered, listed, z])

    def build_minutes(self, pids, target_s) -> np.ndarray:
        covered, listed, z = self._parts(pids, target_s)
        return covered * (self.minutes_coef[0] + self.minutes_coef[1] * listed + self.minutes_coef[2] * z)

    def fit_minutes(self, pids, target_s, mpg_est, actual_mpg, weight) -> "AdpFeatures":
        """Ridge of the minutes residual on ADP over ADP-covered training rows; too few rows leaves the adjustment at zero."""
        covered, listed, z = self._parts(pids, target_s)
        resid = np.asarray(actual_mpg, float) - np.asarray(mpg_est, float)
        w = np.asarray(weight, float)
        ok = (covered > 0) & np.isfinite(resid) & np.isfinite(w) & (w > 0)
        if ok.sum() < MIN_COVERED_ROWS:
            return self
        X = np.column_stack([np.ones(ok.sum()), listed[ok], z[ok]])
        sw = np.sqrt(w[ok] / w[ok].mean())
        pen = RIDGE * np.diag([0.0, 1.0, 1.0])
        Xw = X * sw[:, None]
        self.minutes_coef = np.linalg.solve(Xw.T @ Xw + pen, Xw.T @ (resid[ok] * sw))
        return self
