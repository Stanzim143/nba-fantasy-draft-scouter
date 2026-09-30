"""Dated ESPN injury-status snapshots -> ``espn_status_snapshots`` (ADR 0016, gap 4).

    python -m src.ingest.espn_status [--season 2026-27] [--offline] [--data-dir ...]

ESPN's public player universe (the same payload ``espn_adp`` downloads, cached under ``raw/espn/players``) carries an
``injuryStatus`` (``ACTIVE`` / ``DAY_TO_DAY`` / ``OUT`` / ...), an ``injured`` flag and ``lastNewsDate`` per player. It is
a *current* value: ESPN keeps no history, so it cannot be back-tested and can only be archived from now on. This step
appends one row per player per day (re-running a day replaces that day), which is the only way to get a point-in-time
record of the preseason and opening-week statuses.

It reads the cache and never downloads: ``python -m src.ingest.preseason_refresh`` refreshes the ESPN payload (its ``adp``
step), and its ``status`` step then archives what that refresh saw. Same accepted-risk stance as ADR 0005 R2 (personal,
non-commercial, low volume, local only).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.contracts import data_dir, season_str
from src.ingest import espn_adp
from src.ingest.http_cache import CachedHttpClient, default_cache_dir
from src.ingest.nba_offseason import live_season_start

TABLE = "espn_status_snapshots"
COLUMNS = ["snapshot_date", "season", "espn_id", "player_id", "player_name", "pro_team_id", "injury_status", "injured",
           "last_news_date"]
KEY = ("snapshot_date", "espn_id")


def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def status_frame(raw: list[dict[str, Any]], season: str, snapshot_date: date, id_map: pd.DataFrame | None) -> pd.DataFrame:
    """One snapshot day from the ESPN player list. ``player_id`` is null where ESPN's id is not in ``player_id_map``."""
    rows = []
    for p in raw:
        if p.get("id") is None or not p.get("fullName"):
            continue
        ln = p.get("lastNewsDate")
        rows.append({
            "snapshot_date": pd.Timestamp(snapshot_date), "season": season, "espn_id": str(p["id"]),
            "player_name": p["fullName"], "pro_team_id": p.get("proTeamId"),
            "injury_status": p.get("injuryStatus") or "UNKNOWN", "injured": bool(p.get("injured", False)),
            "last_news_date": pd.to_datetime(ln, unit="ms", errors="coerce") if ln else pd.NaT,
        })
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "player_id"])
    if id_map is not None and len(id_map):
        em = id_map[id_map["source"] == "espn"].drop_duplicates("source_id")
        m = pd.Series(em["player_id"].to_numpy(), index=em["source_id"].astype(str).to_numpy())
        df["player_id"] = pd.array(df["espn_id"].map(m), dtype="Int64")
    else:
        df["player_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["pro_team_id"] = pd.array(pd.to_numeric(df["pro_team_id"], errors="coerce"), dtype="Int64")
    df["snapshot_date"] = df["snapshot_date"].astype("datetime64[ms]")
    df["last_news_date"] = pd.to_datetime(df["last_news_date"]).astype("datetime64[ms]")
    return df[COLUMNS]


def write_snapshot(day: pd.DataFrame, base: Path | None = None) -> Path:
    """Append one day; re-running the same day replaces it. Atomic."""
    if day.duplicated(list(KEY)).any():
        raise ValueError("duplicate espn ids in a status snapshot")
    path = table_path(base)
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old[~old["snapshot_date"].isin(set(day["snapshot_date"]))], day], ignore_index=True)
    else:
        merged = day
    merged = merged.sort_values(["snapshot_date", "espn_id"], kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_status(base: Path | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.espn_status`")
    return pd.read_parquet(path)


def latest_status(snapshots: pd.DataFrame) -> pd.DataFrame:
    return snapshots[snapshots["snapshot_date"] == snapshots["snapshot_date"].max()].reset_index(drop=True)


def run_ingest(season: str, client: CachedHttpClient, base: Path | None = None, *, snapshot_date: date | None = None,
               log=print) -> dict[str, Any]:
    base = base or data_dir()
    raw = espn_adp.fetch_espn_players(client, espn_adp.espn_season_id(season), refresh=False)
    id_map = None
    try:
        from src.store import read_table

        id_map = read_table("player_id_map", base, validate=False)
    except FileNotFoundError:
        pass
    day = status_frame(raw, season, snapshot_date or date.today(), id_map)
    write_snapshot(day, base)
    counts = day["injury_status"].value_counts().to_dict()
    log(f"  espn_status_snapshots: {len(day)} players on {day['snapshot_date'].iloc[0].date()}: {counts}")
    return {"players": len(day), "mapped": int(day["player_id"].notna().sum()), "by_status": counts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.espn_status", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default=season_str(live_season_start()))
    p.add_argument("--offline", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    return p


def main(argv=None, *, client: CachedHttpClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    client = client or CachedHttpClient(default_cache_dir(espn_adp.CACHE_DIR_NAME), offline=True if args.offline else None,
                                        min_interval=2.0)
    try:
        r = run_ingest(args.season, client, base)
    except (espn_adp.EspnAdpError, OSError, ValueError) as exc:
        print(f"status failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {r['players']} players ({r['mapped']} mapped to NBA ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
