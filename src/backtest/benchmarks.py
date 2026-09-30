"""Benchmarks and projection-building helpers.

* ``NaiveLastSeason``: the "beat this first" baseline (rank by last season's fantasy points).
  A local implementation so this package's tests do not depend on the model registry; the
  registry's own ``naive_last_season`` (if any) takes precedence in the CLI.
* ``AdpBenchmark`` / ``load_adp``: hook for an external ADP / consensus ranking.  ADP is a
  *ranking*, not a stat projection, so it is a **rank-only** projector: the harness scores only
  the rank metrics (Spearman on total FP, top-K hit rate/capture, NDCG) for it and marks the
  error and calibration metrics NaN.

Expected ADP input (CSV or Parquet), one row per player per season, as known before that
season's draft::

    season     str    '2023-24'
    source     str    id namespace, must exist in player_id_map (e.g. 'espn')
    source_id  str    the source's player identifier
    adp        float  average draft position, lower = drafted earlier
    name       str    (optional) display name; falls back to player_id_map.source_name

``load_adp`` maps ``(source, source_id)`` onto the canonical ``player_id`` through
``player_id_map`` and reports how many rows could not be mapped.  Nothing here downloads data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.backtest.errors import BacktestError
from src.contracts import (PROJECTION_STATS, STAT_COLUMN_MAP, History, ContractError,
                           validate_table)
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

MIN_GAMES_FOR_BAND = 10


# --------------------------------------------------------------------------- projection builder

def stats_to_projection(
    stats: pd.DataFrame,
    *,
    season: str,
    model: str,
    scoring: Mapping[str, float],
) -> pd.DataFrame:
    """Turn per-game stat projections into a contract-valid ``projections`` frame.

    ``stats`` needs ``player_id, player_name, proj_gp, proj_mpg`` and every ``proj_<stat>``
    column; ``fppg_p10/p50/p90`` are optional.  FPPG uses ``scoring`` and total FP is
    ``fppg x gp`` exactly as the contract states.
    """
    df = stats.copy()
    for c in ("fppg_p10", "fppg_p50", "fppg_p90"):
        if c not in df.columns:
            df[c] = np.nan
    df["proj_fppg"] = fantasy_points_frame(df, scoring, prefix="proj_")
    df["proj_total_fp"] = df["proj_fppg"] * df["proj_gp"]
    df.insert(0, "season", season)
    df["model"] = model
    cols = ["season", "player_id", "player_name", "model", "proj_gp", "proj_mpg", *PROJECTION_STATS,
            "proj_fppg", "proj_total_fp", "fppg_p10", "fppg_p50", "fppg_p90"]
    out = df[cols].reset_index(drop=True)
    for c in ("proj_gp", "proj_mpg", *PROJECTION_STATS, "proj_fppg", "proj_total_fp"):
        out[c] = out[c].astype("float64")
    return validate_table(out, "projections")


def empty_projection(season: str, model: str) -> pd.DataFrame:
    """A valid, empty ``projections`` frame (for projectors with no history to work from)."""
    stats = pd.DataFrame({"player_id": pd.Series([], dtype="int64"),
                          "player_name": pd.Series([], dtype="object"),
                          "proj_gp": pd.Series([], dtype="float64"),
                          "proj_mpg": pd.Series([], dtype="float64"),
                          **{c: pd.Series([], dtype="float64") for c in PROJECTION_STATS}})
    return stats_to_projection(stats, season=season, model=model, scoring=load_league()["scoring"])


# --------------------------------------------------------------------------- naive last season

class NaiveLastSeason:
    """Project each player at exactly his last season's per-game line and games played.

    Ranking by ``proj_total_fp`` is therefore "rank by last season's fantasy points".  Floor /
    ceiling are the empirical game-level p10/p50/p90 of last season (players with >= 10 games).
    Players absent last season (rookies, returners) are not projected: a real coverage miss.
    """

    name = "naive_last_season"
    fingerprint = "naive_last_season/v1"

    def __init__(self, scoring: Mapping[str, float] | None = None):
        self.scoring = dict(scoring if scoring is not None else load_league()["scoring"])

    def project(self, history: History) -> pd.DataFrame:
        last = history.last_season
        if last is None:
            return empty_projection(history.target_season, self.name)
        gl = history.game_logs[history.game_logs["season"] == last]
        stat_cols = list(STAT_COLUMN_MAP.values())
        g = gl.groupby("player_id", sort=True)
        stats = g[stat_cols].mean().add_prefix("proj_")
        stats["proj_gp"] = g["game_id"].count().astype("float64")
        stats["proj_mpg"] = g["min"].mean()
        stats["player_name"] = g["player_name"].last()
        fp = pd.Series(fantasy_points_frame(gl, self.scoring).to_numpy(), index=gl.index)
        q = fp.groupby(gl["player_id"]).quantile([0.1, 0.5, 0.9]).unstack()
        enough = stats["proj_gp"] >= MIN_GAMES_FOR_BAND
        stats["fppg_p10"] = q[0.1].where(enough)
        stats["fppg_p50"] = q[0.5].where(enough)
        stats["fppg_p90"] = q[0.9].where(enough)
        stats = stats.reset_index()
        return stats_to_projection(stats, season=history.target_season, model=self.name, scoring=self.scoring)


# --------------------------------------------------------------------------- ADP hook

ADP_REQUIRED = ("season", "source", "source_id", "adp")


@dataclass(frozen=True)
class AdpData:
    """Mapped ADP rows plus an audit of what could not be mapped."""
    frame: pd.DataFrame          # season, player_id, player_name, adp
    n_rows: int
    n_unmapped: int
    n_low_confidence: int
    n_duplicates: int

    @property
    def unmapped_share(self) -> float:
        return self.n_unmapped / self.n_rows if self.n_rows else float("nan")


def load_adp(path: str | Path, player_id_map: pd.DataFrame, *, source: str = "espn",
             min_confidence: float = 0.0) -> AdpData:
    """Load an ADP file and map it onto canonical ``player_id``.

    Fails clearly (``BacktestError``) if the file is missing, unreadable, lacks a required
    column, or the id map has no rows for ``source``.  Rows whose ``source_id`` is not in the map,
    or whose mapping confidence is below ``min_confidence``, are dropped and counted.  If two
    source rows map to one player-season, the earliest pick (lowest ADP) is kept.
    """
    p = Path(path)
    if not p.exists():
        raise BacktestError(
            f"ADP file {p} not found. The ADP benchmark needs an external file with columns "
            f"{list(ADP_REQUIRED)} (see src/backtest/benchmarks.py); nothing is downloaded automatically. "
            f"Run without an ADP benchmark, or supply the file.")
    try:
        raw = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p, dtype={"source_id": str})
    except Exception as exc:  # noqa: BLE001 - surface any parse problem with the path
        raise BacktestError(f"could not read ADP file {p}: {exc}") from exc
    missing = [c for c in ADP_REQUIRED if c not in raw.columns]
    if missing:
        raise BacktestError(f"ADP file {p} is missing columns {missing}; expected {list(ADP_REQUIRED)}")
    if raw["adp"].isna().any():
        raise BacktestError(f"ADP file {p} has {int(raw['adp'].isna().sum())} rows with a null adp")
    try:
        validate_table(player_id_map, "player_id_map")
    except ContractError as exc:
        raise BacktestError(f"player_id_map invalid: {exc}") from exc
    ids = player_id_map[player_id_map["source"] == source]
    if ids.empty:
        raise BacktestError(f"player_id_map has no rows for source {source!r}")
    raw = raw[raw["source"] == source].copy()
    raw["source_id"] = raw["source_id"].astype(str)
    m = raw.merge(ids[["source_id", "player_id", "source_name", "confidence"]], on="source_id", how="left")
    n_unmapped = int(m["player_id"].isna().sum())
    low = m["player_id"].notna() & (m["confidence"] < min_confidence)
    m = m[m["player_id"].notna() & ~low].copy()
    m["player_id"] = m["player_id"].astype("int64")
    name = m["name"] if "name" in m.columns else m["source_name"]
    m["player_name"] = name.where(name.notna(), m["source_name"])
    m = m.sort_values(["season", "player_id", "adp"], kind="stable")
    n_dup = int(m.duplicated(["season", "player_id"]).sum())
    m = m.drop_duplicates(["season", "player_id"], keep="first")
    frame = m[["season", "player_id", "player_name", "adp"]].reset_index(drop=True)
    frame["adp"] = frame["adp"].astype("float64")
    return AdpData(frame, len(raw), n_unmapped, int(low.sum()), n_dup)


class AdpBenchmark:
    """Rank-only projector serving an external ADP ranking (lower ADP = better = ranked first).

    ``proj_total_fp`` carries the ordinal score ``-adp`` (higher = better); it is NOT a fantasy
    point projection, which is why ``rank_only`` makes the harness score only rank metrics.
    Seasons absent from the file yield an empty projection (and the harness will say so).
    """

    name = "adp"
    rank_only = True

    def __init__(self, adp: AdpData | pd.DataFrame):
        self.frame = adp.frame if isinstance(adp, AdpData) else adp
        self.fingerprint = f"adp/{len(self.frame)}"

    def project(self, history: History) -> pd.DataFrame:
        f = self.frame[self.frame["season"] == history.target_season]
        return pd.DataFrame({
            "season": history.target_season,
            "player_id": f["player_id"].to_numpy(),
            "player_name": f["player_name"].to_numpy(),
            "model": self.name,
            "proj_total_fp": -f["adp"].to_numpy(dtype="float64"),
        })
