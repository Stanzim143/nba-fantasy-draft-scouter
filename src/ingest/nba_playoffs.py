"""Playoff box scores (stats.nba.com ``leaguegamelog``, ``SeasonType="Playoffs"``) -> ``playoff_logs``.

    python -m src.ingest.nba_playoffs --seasons 2025-26 [--offline] [--refresh]

Why this exists: ``game_logs`` is regular season only by contract (ESPN H2H points ignores playoffs), so a breakout
playoff run (a bench player starting in a conference final) is invisible to the model. This table feeds the descriptive
``playoff_flag`` in the draft-day risk overlay (``src.features.risk.playoff_flags``); it does **not** feed any projection
(a handful of games in an injury-driven role is not a validated predictor, and nothing in the backtest tests it).

``season`` is the season the playoffs belong to, so ``History.until`` semantics hold: they are visible to a projection of
season S only when ``season < S``. Standalone parquet (like ``offseason_logs``, ADR 0011 D1), not in ``src.contracts.TABLES``.
Play-in games (id prefix 005) are excluded; only ids ``004<yy>xxxxx`` are kept. Same source and accepted-risk stance as
``nba_stats`` (ADR 0005 R1): private, non-commercial, nothing committed. One request per season.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd

from src.contracts import data_dir, season_start
from src.ingest import nba_transform as tf
from src.ingest.nba_client import NBAClient, NBAClientError
from src.ingest.nba_stats import gamelog_params, parse_seasons

TABLE = "playoff_logs"
SEASON_TYPE = "Playoffs"
MIN_SEASON_START = 2015
KEEP = ["season", "game_id", "game_date", "player_id", "player_name", "team_id", "team_abbr", "min", *tf._INT_STATS]


def playoff_params(season: str) -> dict:
    return {**gamelog_params(season, "P"), "SeasonType": SEASON_TYPE}


def is_playoff_game_id(game_id: str, season: str) -> bool:
    return bool(re.fullmatch(rf"004{season_start(season) % 100:02d}\d{{5}}", game_id))


def transform_playoff_logs(payload: dict, season: str) -> pd.DataFrame:
    """Player-game rows of one season's playoffs; rows without minutes or box-score fields are dropped."""
    raw = tf.result_set(payload)
    tf._require(raw, list(tf._GL_SOURCE), f"playoff logs {season}")
    df = raw.rename(columns=tf._GL_SOURCE)[list(tf._GL_SOURCE.values())].copy()
    df["min"] = tf.parse_minutes(raw["MIN"] if "MIN" in raw.columns else pd.Series(float("nan"), index=raw.index))
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df = df[[is_playoff_game_id(g, season) for g in df["game_id"]]]
    df = df[df["min"].notna() & (df["min"] > 0)].copy()
    df["game_date"] = tf.parse_game_date(df["game_date"])
    for c in ("player_id", "team_id", *tf._INT_STATS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["player_id", "team_id", *tf._INT_STATS])
    df = df.sort_values(["game_id", "player_id", "min"]).drop_duplicates(["game_id", "player_id"], keep="last")
    df["season"] = season
    for c in ("player_id", "team_id", *tf._INT_STATS):
        df[c] = df[c].astype("int64")
    return df[KEEP].reset_index(drop=True)


def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def read_playoff_logs(base: Path | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.nba_playoffs` run?")
    return pd.read_parquet(path)


def run_ingest(seasons: list[str], client: NBAClient, base: Path | None = None, *, refresh: bool = False,
               log: Callable[[str], None] = print) -> pd.DataFrame:
    for s in seasons:
        if season_start(s) < MIN_SEASON_START:
            raise ValueError(f"{s}: earlier than the {MIN_SEASON_START} start of this project's window")
    base = base or data_dir()
    frames = []
    for s in seasons:
        payload = client.get("leaguegamelog", playoff_params(s), refresh=refresh and not client.offline)
        df = transform_playoff_logs(payload, s)
        log(f"[{s} playoffs] {len(df):,} player-games, {df['game_id'].nunique()} games, {df['player_id'].nunique()} players")
        frames.append(df)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP)
    path = table_path(base)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["season"].isin(seasons)], new], ignore_index=True)
    new = new.sort_values(["season", "game_date", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        new.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return new


def main(argv: list[str] | None = None, *, client: NBAClient | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.ingest.nba_playoffs",
                                description="Ingest playoff box scores from stats.nba.com (unofficial) into the data store.")
    p.add_argument("--seasons", default="2025-26", help="e.g. 2025-26 or 2023-24:2025-26 (default 2025-26)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=6)
    args = p.parse_args(argv)
    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        p.error(str(exc))
    base = args.data_dir or data_dir()
    if client is None:
        client = NBAClient(base / "raw" / "nba_api", offline=True if args.offline else None,
                           min_interval=max(args.min_interval, 0.8), max_retries=args.max_retries)
    try:
        run_ingest(seasons, client, base, refresh=args.refresh)
    except (NBAClientError, tf.TransformError, ValueError) as exc:
        print(f"ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
