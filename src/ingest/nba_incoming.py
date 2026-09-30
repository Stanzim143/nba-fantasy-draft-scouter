"""Incoming draft class + current rosters (stats.nba.com ``playerindex``) -> ``players`` and ``roster_snapshots``.

    python -m src.ingest.nba_incoming [--season 2026-27] [--offline] [--no-birthdates]

The preseason roster update (ADR 0012). It fixes two gaps that only exist for a season that has not
started yet:

1. **The incoming draft class was not in ``players``.** ``nba_transform.build_players`` only creates rows
   for people who appear in NBA game logs, so a rookie who has not played a game cannot exist there. The
   historical backtest never noticed, because every past rookie eventually debuted. For the live season
   that meant *no* rookie could be projected, and every rookie ESPN lists an ADP for stayed unmapped.
   This module adds each drafted, currently rostered rookie of the target draft year, with real draft
   slot, position, size and birthdate (``commonplayerinfo``), so the rookie prior in
   ``src.models.rookies`` can see them.
2. **Nothing recorded who is on which roster, and when.** ``roster_snapshots`` is an append-only, dated
   table (one row per rostered player per snapshot day), so preseason cuts, signings and trades can be
   diffed day by day until the draft (:func:`roster_moves`) and each player's current team is known.

Scope decisions, all deliberate:

* Only players with ``ROSTER_STATUS == 1`` (on a team's current roster) count. ``TEAM_ID`` is filled
  even for retired players, so it cannot be used for this.
* Only the *drafted* class of the target year is added to ``players``. Undrafted signees are not: the
  backtest can never project an undrafted debutant (``History`` drops anyone neither drafted by the
  season's draft nor already in game logs), so adding them would make the live board behave in a way no
  backtest ever measured. They remain visible in ``roster_snapshots`` and in the Summer League tables.
* Existing ``players`` rows are never modified (``merge_players`` lets a null in the new frame never
  erase a stored value, and only new player ids are passed in).

Same source and accepted-risk stance as ``nba_stats`` (ADR 0005 R1): private, non-commercial, uncommitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.contracts import ContractError, data_dir, season_start, season_str, validate_table
from src.ingest import nba_transform as tf
from src.ingest.nba_client import NBAClient, NBAClientError, atomic_write_bytes
from src.ingest.nba_offseason import live_season_start
from src.ingest.nba_stats import common_player_info_params, merge_players, player_index_params
from src.store import read_table, table_exists, write_table

REPORT_NAME = "nba_incoming_report.json"
SNAPSHOT_TABLE = "roster_snapshots"

ROSTER_SNAPSHOT_COLUMNS: dict[str, str] = {
    "snapshot_date": "date", "season": "str", "player_id": "int", "player_name": "str",
    "team_id": "int", "team_abbr": "str", "position": "str?", "draft_year": "int?", "draft_number": "int?",
}
SNAPSHOT_KEY = ("snapshot_date", "player_id")


class IncomingError(RuntimeError):
    """The pulled roster data is unusable."""


# --------------------------------------------------------------------------- transforms

def _index(payload: dict) -> pd.DataFrame:
    """``playerindex`` payload -> one row per player: ``players``-style attributes plus current-roster fields."""
    raw = tf.result_set(payload, "PlayerIndex")
    tf._require(raw, ["PERSON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "ROSTER_STATUS"], "playerindex")
    attrs = tf.player_index_frame(payload)
    extra = pd.DataFrame({
        "player_id": pd.to_numeric(raw["PERSON_ID"]).astype("int64"),
        "team_id": pd.to_numeric(raw["TEAM_ID"], errors="coerce"),
        "team_abbr": raw["TEAM_ABBREVIATION"],
        "roster_status": pd.to_numeric(raw["ROSTER_STATUS"], errors="coerce"),
    }).drop_duplicates("player_id", keep="first")
    return attrs.merge(extra, on="player_id", how="left")


def rostered(index: pd.DataFrame) -> pd.DataFrame:
    """Players on a team's current roster (``ROSTER_STATUS == 1``). See the module docstring for why."""
    return index[(index["roster_status"] == 1.0) & index["team_id"].notna() & (index["team_id"] != 0)].copy()


def incoming_class(index: pd.DataFrame, season: str, known_player_ids: set[int]) -> pd.DataFrame:
    """Rostered players drafted in ``season``'s draft who are not yet in ``players``."""
    year = season_start(season)
    r = rostered(index)
    dy = pd.to_numeric(r["draft_year"], errors="coerce")
    return r[(dy == year) & ~r["player_id"].isin(known_player_ids)].reset_index(drop=True)


def roster_snapshot_frame(index: pd.DataFrame, season: str, snapshot_date: date) -> pd.DataFrame:
    """Validated ``roster_snapshots`` rows for one day."""
    r = rostered(index)
    out = pd.DataFrame({
        "snapshot_date": pd.Timestamp(snapshot_date),
        "season": season,
        "player_id": r["player_id"].astype("int64").to_numpy(),
        "player_name": r["player_name"].astype(str).to_numpy(),
        "team_id": r["team_id"].astype("int64").to_numpy(),
        "team_abbr": r["team_abbr"].astype(str).to_numpy(),
        "position": r["position"].astype("object").to_numpy(),
        "draft_year": pd.array(pd.to_numeric(r["draft_year"], errors="coerce").round(), dtype="Int64"),
        "draft_number": pd.array(pd.to_numeric(r["draft_number"], errors="coerce").round(), dtype="Int64"),
    })
    out = out.sort_values(["team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    return validate_roster_snapshots(out)


def roster_moves(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """What changed between two snapshots: ``arrived`` (on a roster now, not before), ``departed``, ``moved``.

    Columns: ``player_id``, ``player_name``, ``kind``, ``from_team``, ``to_team`` (abbreviations; null where n/a).
    """
    p = previous.drop_duplicates("player_id").set_index("player_id")
    c = current.drop_duplicates("player_id").set_index("player_id")
    rows: list[dict[str, Any]] = []
    for pid in sorted(set(c.index) - set(p.index)):
        rows.append({"player_id": pid, "player_name": c.at[pid, "player_name"], "kind": "arrived",
                     "from_team": None, "to_team": c.at[pid, "team_abbr"]})
    for pid in sorted(set(p.index) - set(c.index)):
        rows.append({"player_id": pid, "player_name": p.at[pid, "player_name"], "kind": "departed",
                     "from_team": p.at[pid, "team_abbr"], "to_team": None})
    for pid in sorted(set(p.index) & set(c.index)):
        if int(p.at[pid, "team_id"]) != int(c.at[pid, "team_id"]):
            rows.append({"player_id": pid, "player_name": c.at[pid, "player_name"], "kind": "moved",
                         "from_team": p.at[pid, "team_abbr"], "to_team": c.at[pid, "team_abbr"]})
    return pd.DataFrame(rows, columns=["player_id", "player_name", "kind", "from_team", "to_team"])


def new_player_rows(cls: pd.DataFrame, cpi_attrs: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """``players`` rows for the incoming class: index attributes first, ``commonplayerinfo`` fills the gaps
    (same precedence as ``nba_transform.build_players``); birthdate only ever comes from ``commonplayerinfo``."""
    cols = ["position", "height_in", "weight_lb", "draft_year", "draft_round", "draft_number", "from_year", "to_year"]
    rows = []
    for _, r in cls.iterrows():
        extra = cpi_attrs.get(int(r["player_id"]), {})
        row: dict[str, Any] = {"player_id": int(r["player_id"]), "player_name": r["player_name"],
                               "birthdate": extra.get("birthdate")}
        for c in cols:
            v = r.get(c)
            row[c] = extra.get(c) if (v is None or pd.isna(v)) else v
        rows.append(row)
    df = pd.DataFrame(rows)
    df["birthdate"] = pd.to_datetime(df["birthdate"]).astype("datetime64[ns]")
    return df[["player_id", "player_name", "birthdate", *cols]]


# --------------------------------------------------------------------------- snapshot storage (ad-hoc table)

def snapshots_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{SNAPSHOT_TABLE}.parquet"


def report_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / REPORT_NAME


def validate_roster_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    problems: list[str] = []
    missing = [c for c in ROSTER_SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    for col, kind in ROSTER_SNAPSHOT_COLUMNS.items():
        if col in df.columns and not kind.endswith("?") and df[col].isna().any():
            problems.append(f"{col}: nulls in a non-nullable column")
    if not missing and df.duplicated(list(SNAPSHOT_KEY)).any():
        problems.append(f"duplicate rows on key {SNAPSHOT_KEY}")
    if problems:
        raise IncomingError("roster_snapshots: " + "; ".join(problems))
    return df


def read_roster_snapshots(base: Path | None = None) -> pd.DataFrame:
    path = snapshots_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.nba_incoming` run?")
    return pd.read_parquet(path)


def latest_snapshot(snapshots: pd.DataFrame, before: date | None = None) -> pd.DataFrame:
    """The rows of the most recent snapshot day (strictly before ``before`` when given)."""
    s = snapshots if before is None else snapshots[snapshots["snapshot_date"] < pd.Timestamp(before)]
    if s.empty:
        return s
    return s[s["snapshot_date"] == s["snapshot_date"].max()].reset_index(drop=True)


def write_roster_snapshot(day: pd.DataFrame, base: Path | None = None) -> Path:
    """Append one day; re-running the same day replaces that day, never duplicates it. Atomic."""
    validate_roster_snapshots(day)
    path = snapshots_path(base)
    if path.exists():
        old = pd.read_parquet(path)
        days = set(day["snapshot_date"])
        merged = pd.concat([old[~old["snapshot_date"].isin(days)], day], ignore_index=True)
    else:
        merged = day
    merged = merged.sort_values(["snapshot_date", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
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


# --------------------------------------------------------------------------- pipeline

@dataclass
class IncomingResult:
    season: str
    snapshot_date: str
    rostered: int = 0
    added_players: int = 0
    already_known: int = 0
    undrafted_rostered_not_in_players: int = 0
    earlier_draft_debuts_not_in_players: int = 0
    earlier_draft_debut_names: list[str] = field(default_factory=list)
    missing_birthdates: int = 0
    moves_since_previous: dict[str, int] = field(default_factory=dict)
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(
    season: str,
    client: NBAClient,
    base: Path | None = None,
    *,
    snapshot_date: date | None = None,
    birthdates: bool = True,
    log: Callable[[str], None] = print,
) -> IncomingResult:
    season_start(season)
    base = base or data_dir()
    day = snapshot_date or date.today()
    log(f"[{season}] pulling the current player index")
    payload = client.get("playerindex", player_index_params(season), refresh=not client.offline)
    index = _index(payload)
    ros = rostered(index)
    if ros.empty:
        raise IncomingError(f"{season}: the player index lists nobody on a roster; refusing to write an empty snapshot")

    existing = read_table("players", base, validate=False) if table_exists("players", base) else None
    known = set(existing["player_id"].astype(int)) if existing is not None else set()
    result = IncomingResult(season=season, snapshot_date=str(day), rostered=len(ros),
                            already_known=int(ros["player_id"].isin(known).sum()))
    not_known = ros[~ros["player_id"].isin(known)]
    dy = pd.to_numeric(not_known["draft_year"], errors="coerce")
    result.undrafted_rostered_not_in_players = int(dy.isna().sum())
    debuts = not_known[(dy < season_start(season)).fillna(False).to_numpy()]
    result.earlier_draft_debuts_not_in_players = len(debuts)
    result.earlier_draft_debut_names = sorted(debuts["player_name"].astype(str))

    cls = incoming_class(index, season, known)
    if len(cls):
        cpi: dict[int, dict[str, Any]] = {}
        if birthdates:
            todo = sum(client.peek("commonplayerinfo", common_player_info_params(int(p))) is None for p in cls["player_id"])
            if todo:
                log(f"  fetching CommonPlayerInfo for {todo} rookies (~{todo * (client.min_interval + 0.5) / 60:.1f} min); resumable")
            for pid in cls["player_id"]:
                try:
                    cpi[int(pid)] = tf.player_attributes_from_common_player_info(
                        client.get("commonplayerinfo", common_player_info_params(int(pid))))
                except tf.TransformError:
                    continue  # an empty CommonPlayerInfo just means "no birthdate known"
        new_rows = new_player_rows(cls, cpi)
        result.missing_birthdates = int(new_rows["birthdate"].isna().sum())
        players = merge_players(existing, new_rows)
        try:
            validate_table(players, "players")
        except ContractError as exc:
            raise IncomingError(f"merged players table failed its contract: {exc}") from exc
        write_table(players, "players", base)
        result.added_players = len(new_rows)
        log(f"  players: +{len(new_rows)} incoming rookies ({result.missing_birthdates} without a birthdate)")
    else:
        log("  players: no new drafted rookies to add")

    snap = roster_snapshot_frame(index, season, day)
    prev = None
    if snapshots_path(base).exists():
        prev = latest_snapshot(read_roster_snapshots(base), before=day)
    write_roster_snapshot(snap, base)
    if prev is not None and len(prev):
        moves = roster_moves(prev, snap)
        result.moves_since_previous = {k: int(v) for k, v in moves["kind"].value_counts().sort_index().items()}
        log(f"  roster moves since {prev['snapshot_date'].iloc[0].date()}: {result.moves_since_previous or 'none'}")
    if result.earlier_draft_debut_names:
        log(f"  note: {len(result.earlier_draft_debut_names)} rostered players drafted in earlier years have never played and are "
            f"not projected (the model projects only the current draft class): {', '.join(result.earlier_draft_debut_names)}")
    log(f"  roster_snapshots: {len(snap)} rostered players on {day}")
    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    _write_report(base, result)
    return result


def _write_report(base: Path, result: IncomingResult) -> None:
    path = report_path(base)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}
    report["runs"][result.season] = {k: v for k, v in result.__dict__.items() if k not in ("network_requests", "cache_hits")}
    atomic_write_bytes(path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.nba_incoming",
        description="Add the incoming draft class to `players` and record a dated roster snapshot (stats.nba.com).",
    )
    p.add_argument("--season", default=season_str(live_season_start()),
                   help="the season about to start (default: the live one); its draft class is added")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--no-birthdates", action="store_true", help="skip the per-rookie CommonPlayerInfo pulls")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0, help="minimum seconds between network requests (default 1.0)")
    p.add_argument("--max-retries", type=int, default=6)
    return p


def main(argv: list[str] | None = None, *, client: NBAClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_interval < 0.8 and not args.offline:
        print(f"warning: --min-interval {args.min_interval} is below the 0.8 s politeness floor; using 0.8", file=sys.stderr)
        args.min_interval = 0.8
    base = args.data_dir or data_dir()
    if client is None:
        client = NBAClient(base / "raw" / "nba_api", offline=True if args.offline else None,
                           min_interval=args.min_interval, max_retries=args.max_retries)
    try:
        result = run_ingest(args.season, client, base, birthdates=not args.no_birthdates)
    except (NBAClientError, IncomingError, tf.TransformError, ValueError) as exc:
        print(f"ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {result.rostered} rostered, +{result.added_players} rookies, "
          f"{result.network_requests} network requests, {result.cache_hits} cache hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
