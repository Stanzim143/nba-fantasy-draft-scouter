"""Static player profiles (country, previous organisation, origin class) and draft-combine measurements (ADR 0016).

    python -m src.ingest.nba_profiles [--season 2026-27] [--offline] [--no-combine] [--no-birthdates]

Two additive tables, both point-in-time safe because they describe facts fixed before any season they are used for:

* ``player_profiles``: one row per player in stats.nba.com's ``playerindex`` (the payload the roster update already
  downloads and caches: **no new request** for the bulk of it). ``country``, ``prev_org`` (the index calls it
  COLLEGE, but it holds whatever the player came from: a college, a pro club or a high school), ``origin``
  (``usa`` / ``intl`` by country), ``prev_org_type`` (``college`` when the organisation is one many players share,
  else ``other``: club, prep school or none), plus size, position and draft slot. Birthdates for players who are on a
  roster but not in ``players`` (stash debutants, undrafted signees) come from ``commonplayerinfo`` (one polite
  request each, resumable).
* ``draft_combine``: ``draftcombinestats`` per draft year (12 requests): measured height, weight, wingspan, standing
  reach, jumps, agility. Only about 60% of draftees attend; stashes and internationals mostly do not.

Same source, same accepted-risk stance as every other stats.nba.com pull (ADR 0005 R1: private, non-commercial,
uncommitted). ``players`` is never modified. Both tables are written atomically; an existing table is backed up once
(``*.bak_pre_profiles``) before the first rewrite.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.contracts import data_dir, season_start, season_str
from src.ingest import nba_transform as tf
from src.ingest.nba_client import NBAClient, NBAClientError, atomic_write_bytes
from src.ingest.nba_offseason import live_season_start
from src.ingest.nba_stats import common_player_info_params, player_index_params

PROFILES_TABLE = "player_profiles"
COMBINE_TABLE = "draft_combine"
REPORT_NAME = "nba_profiles_report.json"
MIN_COMBINE_YEAR = 2015
COLLEGE_MIN_PLAYERS = 3   # an organisation that at least this many USA-born players list is treated as a college

PROFILE_COLUMNS = ["player_id", "player_name", "birthdate", "country", "prev_org", "origin", "prev_org_type",
                   "height_in", "weight_lb", "position", "draft_year", "draft_round", "draft_number", "from_year"]
COMBINE_COLUMNS = ["draft_year", "player_id", "height_wo_shoes_in", "weight_lb", "wingspan_in", "standing_reach_in",
                   "standing_vertical_in", "max_vertical_in", "lane_agility_s", "sprint_s"]
_COMBINE_MAP = {"HEIGHT_WO_SHOES": "height_wo_shoes_in", "WEIGHT": "weight_lb", "WINGSPAN": "wingspan_in",
                "STANDING_REACH": "standing_reach_in", "STANDING_VERTICAL_LEAP": "standing_vertical_in",
                "MAX_VERTICAL_LEAP": "max_vertical_in", "LANE_AGILITY_TIME": "lane_agility_s", "THREE_QUARTER_SPRINT": "sprint_s"}


class ProfilesError(RuntimeError):
    """The pulled profile data is unusable."""


# --------------------------------------------------------------------------- transforms

def build_profiles(payload: dict) -> pd.DataFrame:
    """``playerindex`` payload -> ``player_profiles`` rows (birthdate empty; filled from ``players`` / CPI later)."""
    raw = tf.result_set(payload, "PlayerIndex")
    tf._require(raw, ["PERSON_ID", "COUNTRY"], "playerindex")
    attrs = tf.player_index_frame(payload)
    extra = pd.DataFrame({
        "player_id": pd.to_numeric(raw["PERSON_ID"]).astype("int64"),
        "country": raw["COUNTRY"].fillna("").astype(str).str.strip(),
        "prev_org": (raw["COLLEGE"].fillna("").astype(str).str.strip() if "COLLEGE" in raw else ""),
    }).drop_duplicates("player_id", keep="first")
    df = attrs.merge(extra, on="player_id", how="left")
    df["country"] = df["country"].replace("", np.nan)
    df["prev_org"] = df["prev_org"].replace("", np.nan)
    us = df[df["country"] == "USA"]["prev_org"].value_counts()
    colleges = set(us[us >= COLLEGE_MIN_PLAYERS].index)
    df["origin"] = np.where(df["country"].isna(), None, np.where(df["country"] == "USA", "usa", "intl"))
    df["prev_org_type"] = np.where(df["prev_org"].isin(colleges), "college", "other")
    df["birthdate"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return df[PROFILE_COLUMNS]


def build_combine(payload: dict) -> pd.DataFrame:
    """One ``draftcombinestats`` payload -> ``draft_combine`` rows."""
    raw = tf.result_set(payload, "DraftCombineStats")
    if raw.empty:
        return pd.DataFrame(columns=COMBINE_COLUMNS)
    tf._require(raw, ["SEASON", "PLAYER_ID"], "draftcombinestats")
    out = pd.DataFrame({"draft_year": pd.to_numeric(raw["SEASON"], errors="coerce").astype("Int64"),
                        "player_id": pd.to_numeric(raw["PLAYER_ID"], errors="coerce").astype("Int64")})
    for src, dst in _COMBINE_MAP.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    out = out.dropna(subset=["draft_year", "player_id"])
    out["draft_year"], out["player_id"] = out["draft_year"].astype("int64"), out["player_id"].astype("int64")
    return out.drop_duplicates(["draft_year", "player_id"], keep="first")[COMBINE_COLUMNS].reset_index(drop=True)


def fill_birthdates(profiles: pd.DataFrame, players: pd.DataFrame | None, extra: dict[int, Any]) -> pd.DataFrame:
    """Birthdates from ``players`` first, then from ``commonplayerinfo`` results (``extra``: player_id -> date-like)."""
    p = profiles.copy()
    bd = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns]")
    if players is not None and len(players):
        m = players.drop_duplicates("player_id").set_index("player_id")["birthdate"]
        bd = pd.to_datetime(p["player_id"].map(m)).astype("datetime64[ns]")
    for pid, v in extra.items():
        if v is None or pd.isna(v):
            continue
        idx = p.index[p["player_id"] == pid]
        bd.loc[idx] = pd.Timestamp(v)
    p["birthdate"] = bd.astype("datetime64[ns]")
    return p


# --------------------------------------------------------------------------- storage

def table_path(name: str, base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{name}.parquet"


def report_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / REPORT_NAME


def write_parquet_atomic(df: pd.DataFrame, path: Path, *, backup_suffix: str = ".bak_pre_profiles") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak = path.with_name(path.name + backup_suffix)
        if not bak.exists():
            shutil.copy2(path, bak)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_profiles(base: Path | None = None) -> pd.DataFrame:
    path = table_path(PROFILES_TABLE, base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.nba_profiles`")
    return pd.read_parquet(path)


def read_combine(base: Path | None = None) -> pd.DataFrame:
    path = table_path(COMBINE_TABLE, base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.nba_profiles`")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- pipeline

@dataclass
class ProfilesResult:
    season: str
    profiles: int = 0
    with_birthdate: int = 0
    candidates_needing_birthdate: int = 0
    combine_rows: int = 0
    combine_years: list[int] = field(default_factory=list)
    combine_failed_years: list[int] = field(default_factory=list)
    network_requests: int = 0


def combine_params(year: int) -> dict[str, Any]:
    return {"LeagueID": "00", "SeasonYear": f"{year}-{(year + 1) % 100:02d}"}


def run_ingest(season: str, client: NBAClient, base: Path | None = None, *, combine: bool = True,
               birthdates: bool = True, log: Callable[[str], None] = print) -> ProfilesResult:
    from src.store import read_table, table_exists

    season_start(season)
    base = base or data_dir()
    result = ProfilesResult(season=season)
    payload = client.get("playerindex", player_index_params(season))
    prof = build_profiles(payload)
    if prof.empty:
        raise ProfilesError("the player index is empty; refusing to write an empty profile table")
    players = read_table("players", base, validate=False) if table_exists("players", base) else None
    extra: dict[int, Any] = {}
    snap_path = base / "processed" / "roster_snapshots.parquet"
    if birthdates and snap_path.exists():
        snap = pd.read_parquet(snap_path)
        latest = snap[snap["snapshot_date"] == snap["snapshot_date"].max()]
        known = set(players["player_id"].astype(int)) if players is not None else set()
        need = [int(p) for p in latest["player_id"] if int(p) not in known]
        result.candidates_needing_birthdate = len(need)
        todo = sum(client.peek("commonplayerinfo", common_player_info_params(p)) is None for p in need)
        if todo:
            log(f"  fetching CommonPlayerInfo for {todo} rostered players without history (~{todo * (client.min_interval + 0.5) / 60:.1f} min); resumable")
        for pid in need:
            try:
                extra[pid] = tf.player_attributes_from_common_player_info(
                    client.get("commonplayerinfo", common_player_info_params(pid))).get("birthdate")
            except (tf.TransformError, NBAClientError) as exc:
                log(f"  no birthdate for {pid}: {exc}")
    prof = fill_birthdates(prof, players, extra)
    write_parquet_atomic(prof, table_path(PROFILES_TABLE, base))
    result.profiles, result.with_birthdate = len(prof), int(prof["birthdate"].notna().sum())
    log(f"  player_profiles: {len(prof)} rows ({result.with_birthdate} with a birthdate)")

    if combine:
        frames = []
        for y in range(MIN_COMBINE_YEAR, season_start(season) + 1):
            try:
                frames.append(build_combine(client.get("draftcombinestats", combine_params(y))))
                result.combine_years.append(y)
            except (NBAClientError, tf.TransformError) as exc:
                result.combine_failed_years.append(y)
                log(f"  draft combine {y} failed: {exc}")
        path = table_path(COMBINE_TABLE, base)
        if result.combine_failed_years and not path.exists():
            log(f"  draft_combine not written: years {result.combine_failed_years} failed and there is no earlier table; re-run")
        elif frames:
            comb = pd.concat(frames, ignore_index=True)
            if result.combine_failed_years:      # keep the failed years' rows from the previous table
                old = pd.read_parquet(path)
                comb = pd.concat([old[~old["draft_year"].isin(result.combine_years)], comb], ignore_index=True)
                comb = comb.sort_values(["draft_year", "player_id"], kind="stable").reset_index(drop=True)
            write_parquet_atomic(comb, path)
            result.combine_rows = len(comb)
            log(f"  draft_combine: {len(comb)} rows for {len(result.combine_years)} draft years")
    result.network_requests = client.stats.network_requests
    _write_report(base, result)
    return result


def _write_report(base: Path, result: ProfilesResult) -> None:
    path = report_path(base)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}
    report["runs"][result.season] = {k: v for k, v in result.__dict__.items() if k != "network_requests"}
    report["runs"][result.season]["run_date"] = str(date.today())
    atomic_write_bytes(path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.nba_profiles",
                                description="Player origin profiles and draft-combine measurements (stats.nba.com).")
    p.add_argument("--season", default=season_str(live_season_start()))
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-combine", action="store_true")
    p.add_argument("--no-birthdates", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None, *, client: NBAClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    if client is None:
        client = NBAClient(base / "raw" / "nba_api", offline=True if args.offline else None,
                           min_interval=max(args.min_interval, 0.8))
    try:
        r = run_ingest(args.season, client, base, combine=not args.no_combine, birthdates=not args.no_birthdates)
    except (NBAClientError, ProfilesError, tf.TransformError, ValueError) as exc:
        print(f"profiles failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {r.profiles} profiles, {r.combine_rows} combine rows, {r.network_requests} network requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
