"""Summer League and preseason box scores (stats.nba.com) -> ``offseason_logs`` / ``offseason_team_games``.

    python -m src.ingest.nba_offseason --seasons 2015-16:2026-27 [--offline] [--refresh]

Why this exists (ADR 0012): the regular-season tables know nothing about what a player did *between*
seasons. Summer League (July, ``LeagueID=15``) and preseason (October, ``SeasonType="Pre Season"``) are
the only pre-draft evidence about rookies and young players, the population where a breakout is most
likely and least priced in.

Season labelling -- read this before touching the leakage logic
---------------------------------------------------------------
Every row carries two season labels:

* ``event_season`` -- the NBA season the offseason event feeds into, exactly as stats.nba.com labels it
  (July 2026 Summer League = ``"2026-27"``).
* ``season`` -- the **completed regular season the event follows** (July 2026 = ``"2025-26"``).

``season`` is what ``History.until`` slices on, so an event is available to a projection of season S
precisely when ``season < S`` -- i.e. the July and October before S starts are visible, and nothing from
S itself is. The leak guard is therefore structural, the same as every other ``History`` table.

Tables (standalone parquet files, ADR 0011 D1 precedent: not in ``src.contracts.TABLES``)
-----------------------------------------------------------------------------------------
``offseason_logs``        one row per player per game actually played (minutes > 0)
``offseason_team_games``  one row per team per game, so a player's share of his team's games is knowable

Data-quality rules, each counted in the report and never silent:

* game ids must carry the expected prefix (``152`` Summer League, ``001`` preseason) and season digits;
* preseason rows dated outside September 1 - December 31 of the start year are dropped (the 2019-20
  pull contains July 2020 bubble games under the preseason type; they are *after* that season began);
* Summer League rows outside June 1 - September 30 of the start year are dropped;
* rows with no minutes, missing box-score fields, or an impossible box-score identity are dropped;
* player rows whose team has no team-game row (exhibitions against non-NBA clubs) are dropped;
* duplicate (game, player) rows keep the one with the most minutes.

Sources: ``leaguegamelog`` supplies the rows (real NBA person ids, names, teams); ``playergamelogs``
only refines integer minutes into fractional ones where a row matches exactly, because for most
pre-2024 summers it reports players under placeholder ids that match no NBA ``PERSON_ID``.

Stats.nba.com terms: same source and same accepted-risk stance as ``nba_stats`` (ADR 0005 R1): private,
non-commercial, nothing committed. Summer League 2020 was not held; that season is legitimately empty.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
from src.ingest.nba_stats import gamelog_params, parse_seasons, player_gamelogs_params

SUMMER_LEAGUE = "summer_league"
PRESEASON = "preseason"
CONTEXTS = (SUMMER_LEAGUE, PRESEASON)

SUMMER_LEAGUE_ID = "15"
PRESEASON_SEASON_TYPE = "Pre Season"
GAME_ID_PREFIX = {SUMMER_LEAGUE: "152", PRESEASON: "001"}
REPORT_NAME = "nba_offseason_report.json"
MIN_SEASON_START = 2015  # earliest season this project ingests; the endpoints themselves go back further

_INT_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]

OFFSEASON_LOGS_COLUMNS: dict[str, str] = {
    "season": "str", "event_season": "str", "context": "str", "game_id": "str", "game_date": "date",
    "player_id": "int", "player_name": "str", "team_id": "int", "team_abbr": "str", "matchup": "str",
    "min": "float", **{c: "int" for c in _INT_STATS}, "plus_minus": "float?",
}
OFFSEASON_TEAM_GAMES_COLUMNS: dict[str, str] = {
    "season": "str", "event_season": "str", "context": "str", "game_id": "str", "game_date": "date",
    "team_id": "int", "team_abbr": "str", "is_home": "bool", "pts_for": "int?", "pts_against": "int?",
}
LOG_KEY = ("game_id", "player_id")
TEAM_GAME_KEY = ("game_id", "team_id")

_SOURCE_COLUMNS = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team_abbr",
    "GAME_ID": "game_id", "GAME_DATE": "game_date", "MATCHUP": "matchup", "PLUS_MINUS": "plus_minus",
    **{c.upper(): c for c in _INT_STATS},
}


class OffseasonError(RuntimeError):
    """The pulled offseason data is unusable."""


# --------------------------------------------------------------------------- labels and windows

def previous_season(event_season: str) -> str:
    """``"2026-27" -> "2025-26"``: the completed season an offseason event follows (see module docstring)."""
    return season_str(season_start(event_season) - 1)


def event_window(context: str, event_season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive calendar window in which a real event of ``context`` for ``event_season`` can fall."""
    y = season_start(event_season)
    if context == SUMMER_LEAGUE:
        return pd.Timestamp(y, 6, 1), pd.Timestamp(y, 9, 30)
    if context == PRESEASON:
        return pd.Timestamp(y, 9, 1), pd.Timestamp(y, 12, 31)
    raise ValueError(f"unknown context {context!r}; expected one of {CONTEXTS}")


def live_season_start(today: date | None = None) -> int:
    """Start year of the NBA season whose offseason events are still unfolding or fresh.

    From June 1 onward the new season's Summer League and preseason are the live ones; before that the
    season that started the previous autumn is.
    """
    t = today or date.today()
    return t.year if t.month >= 6 else t.year - 1


# --------------------------------------------------------------------------- endpoint parameters

def player_logs_params(event_season: str, context: str) -> dict[str, Any]:
    """``leaguegamelog`` (player rows): the **canonical** row source, with real NBA person ids and names.

    ``playergamelogs`` looks like the better endpoint (fractional minutes) but for most summers before
    2024 it reports many players under placeholder ids that match no NBA ``PERSON_ID`` at all (2015 and
    2016: zero overlap), and blank names/teams. Rows come from here; see :func:`fractional_minutes_params`.
    """
    return _apply_context(gamelog_params(event_season, "P"), context)


def fractional_minutes_params(event_season: str, context: str) -> dict[str, Any]:
    """``playergamelogs``: used only to refine ``leaguegamelog``'s integer minutes into fractional ones."""
    return _apply_context(player_gamelogs_params(event_season), context)


def team_logs_params(event_season: str, context: str) -> dict[str, Any]:
    """``leaguegamelog`` (team rows) for one context of one season."""
    return _apply_context(gamelog_params(event_season, "T"), context)


def _apply_context(params: dict[str, Any], context: str) -> dict[str, Any]:
    out = dict(params)
    if context == SUMMER_LEAGUE:
        out["LeagueID"] = SUMMER_LEAGUE_ID
    elif context == PRESEASON:
        out["SeasonType"] = PRESEASON_SEASON_TYPE
    else:
        raise ValueError(f"unknown context {context!r}; expected one of {CONTEXTS}")
    return out


# --------------------------------------------------------------------------- transforms

@dataclass
class OffseasonDropReport:
    """Counts of rows a transform discarded, by reason. Same spirit as ``nba_transform.DropReport``."""

    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    info: dict[str, int] = field(default_factory=dict)  # facts about kept rows; not drops

    def add(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rows_in": self.rows_in, "rows_out": self.rows_out,
                               "dropped": dict(sorted(self.dropped.items()))}
        if self.info:
            out["info"] = dict(sorted(self.info.items()))
        return out


def _id_regex(context: str, event_season: str) -> re.Pattern[str]:
    yy = f"{season_start(event_season) % 100:02d}"
    return re.compile(rf"^{GAME_ID_PREFIX[context]}{yy}\d+$")


def _identity_violations(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-row box-score identities, mirroring ``src.contracts`` for regular-season logs -- with one
    deliberate difference: points must be *at least* ``2*fgm + fg3m + ftm``, not exactly equal.

    Exact equality holds in every real season except the July 2026 Summer League, where points exceed
    the made-shot points on every team in every game (average +7.8 per team-game, tracking free
    throws: ~40% of free throws are absent from ``ftm``/``fta`` while the official points include
    them). Player sums match team totals exactly, so ``pts`` is trustworthy and the source's FT
    counters are what is short. Fewer points than made shots would be impossible, so that direction
    is still a hard violation; the excess is measured and reported by :func:`unrecorded_points`.
    """
    return {
        "made_exceeds_attempted": (df["fgm"] > df["fga"]) | (df["fg3m"] > df["fgm"]) | (df["fg3a"] > df["fga"])
        | (df["ftm"] > df["fta"]),
        "rebound_identity": df["reb"] != df["oreb"] + df["dreb"],
        "points_below_made_shots": df["pts"] < 2 * df["fgm"] + df["fg3m"] + df["ftm"],
        "negative_counting_stat": (df[["fgm", "fga", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"]] < 0).any(axis=1),
    }


def unrecorded_points(df: pd.DataFrame) -> pd.Series:
    """Points beyond what the recorded made shots explain (see :func:`_identity_violations`); 0 in normal data."""
    return df["pts"] - (2 * df["fgm"] + df["fg3m"] + df["ftm"])


def apply_fractional_minutes(logs: pd.DataFrame, fractional_payload: dict | None,
                             tolerance: float = 1.0) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace integer minutes with ``playergamelogs``' fractional minutes where the row matches exactly.

    A match is the same ``(game_id, player_id)`` with the two minute values within ``tolerance``. Rows
    that do not match keep their integer minutes; ``playergamelogs`` rows with no counterpart (the
    placeholder-id players described in :func:`player_logs_params`) are counted, not used.
    """
    info = {"fractional_minutes_used": 0, "integer_minutes_only": len(logs),
            "fractional_rows_without_real_id": 0, "fractional_minutes_out_of_tolerance": 0}
    if fractional_payload is None or logs.empty:
        return logs, info
    raw = tf.result_set(fractional_payload)
    if raw.empty or not {"GAME_ID", "PLAYER_ID"} <= set(raw.columns):
        return logs, info
    minutes = raw["MIN"] if "MIN" in raw.columns else pd.Series(np.nan, index=raw.index)
    if minutes.isna().all() and "MIN_SEC" in raw.columns:
        minutes = raw["MIN_SEC"]
    frac = pd.DataFrame({
        "game_id": raw["GAME_ID"].astype(str).str.strip(),
        "player_id": pd.to_numeric(raw["PLAYER_ID"], errors="coerce"),
        "min_frac": tf.parse_minutes(minutes),
    }).dropna(subset=["player_id", "min_frac"])
    frac = frac[frac["min_frac"] > 0].astype({"player_id": "int64"}).drop_duplicates(["game_id", "player_id"])
    merged = logs.merge(frac, on=["game_id", "player_id"], how="left")
    close = (merged["min_frac"] - merged["min"]).abs() <= tolerance
    known = set(zip(logs["game_id"], logs["player_id"]))
    info["fractional_rows_without_real_id"] = int(sum(k not in known for k in zip(frac["game_id"], frac["player_id"])))
    info["fractional_minutes_out_of_tolerance"] = int((merged["min_frac"].notna() & ~close).sum())
    use = merged["min_frac"].notna() & close
    out = logs.copy()
    out["min"] = np.where(use.to_numpy(), merged["min_frac"].to_numpy(), out["min"].to_numpy()).astype("float64")
    info["fractional_minutes_used"] = int(use.sum())
    info["integer_minutes_only"] = int((~use).sum())
    return out, info


def transform_offseason_logs(payload: dict, event_season: str, context: str, *,
                             fractional_payload: dict | None = None) -> tuple[pd.DataFrame, OffseasonDropReport]:
    """One ``leaguegamelog`` (player rows) payload -> validated ``offseason_logs`` rows for (season, context).

    ``fractional_payload`` (a ``playergamelogs`` payload) refines integer minutes where rows match
    exactly, see :func:`apply_fractional_minutes`. A blank player name becomes ``"player <id>"`` and is
    counted: the stats are real and the name is display-only, so dropping the row would only lose data.
    """
    season_start(event_season)  # validates the format
    lo, hi = event_window(context, event_season)
    raw = tf.result_set(payload)
    report = OffseasonDropReport(rows_in=len(raw))
    if raw.empty:  # e.g. the 2020 Summer League was never held
        return empty_table(OFFSEASON_LOGS_COLUMNS), report
    where = f"{context} {event_season} player logs"
    tf._require(raw, list(_SOURCE_COLUMNS), where)
    df = raw.rename(columns=_SOURCE_COLUMNS)[list(_SOURCE_COLUMNS.values())].copy()
    df["min"] = tf.parse_minutes(raw["MIN"])

    df["game_id"] = df["game_id"].astype(str).str.strip()
    id_ok = pd.Series([bool(_id_regex(context, event_season).match(g)) for g in df["game_id"]], index=df.index, dtype=bool)
    report.add("unexpected_game_id", (~id_ok).sum())
    df = df[id_ok]

    df["game_date"] = tf.parse_game_date(df["game_date"])
    in_window = df["game_date"].between(lo, hi)
    report.add("outside_event_window", (~in_window).sum())
    df = df[in_window]

    no_min = df["min"].isna() | (df["min"] <= 0)
    report.add("no_minutes_played", no_min.sum())
    df = df[~no_min]

    for col in ("player_id", "team_id", *_INT_STATS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    incomplete = df[["player_id", "team_id", *_INT_STATS]].isna().any(axis=1)
    report.add("missing_box_score_fields", incomplete.sum())
    df = df[~incomplete]
    for col in ("player_id", "team_id", *_INT_STATS):
        df[col] = df[col].astype("int64")

    drop = pd.Series(False, index=df.index)
    for reason, mask in _identity_violations(df).items():
        report.add(reason, (mask & ~drop).sum())     # a row violating several rules is counted once, under the first
        drop |= mask
    df = df[~drop]

    excess = unrecorded_points(df)
    if (excess > 0).any():
        report.info["rows_with_unrecorded_points"] = int((excess > 0).sum())
        report.info["unrecorded_points_total"] = int(excess.sum())
    df["plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").astype("float64")
    for col in ("player_name", "team_abbr", "matchup"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col in ("team_abbr", "matchup"):
        if df[col].isna().any():
            raise OffseasonError(f"{where}: {int(df[col].isna().sum())} rows have no {col}")
    unnamed = df["player_name"].isna()
    if unnamed.any():
        report.info["names_unresolved"] = int(unnamed.sum())
        df.loc[unnamed, "player_name"] = "player " + df.loc[unnamed, "player_id"].astype(str)
    for col in ("player_name", "team_abbr", "matchup"):
        df[col] = df[col].astype(str)

    key = list(LOG_KEY)
    before = len(df)
    df = df.drop_duplicates()
    report.add("duplicate_row_identical", before - len(df))
    if df.duplicated(key).any():
        before = len(df)
        df = df.sort_values(key + ["min", "pts"], ascending=[True, True, False, False]).drop_duplicates(key)
        report.add("duplicate_row_conflicting", before - len(df))

    df, minutes_info = apply_fractional_minutes(df, fractional_payload)
    report.info.update(minutes_info)
    df["season"] = previous_season(event_season)
    df["event_season"] = event_season
    df["context"] = context
    out = df[list(OFFSEASON_LOGS_COLUMNS)].sort_values(["game_date", "game_id", "team_id", "player_id"],
                                                        kind="mergesort").reset_index(drop=True)
    report.rows_out = len(out)
    validate_offseason_logs(out)
    return out, report


def transform_offseason_team_games(payload: dict, event_season: str, context: str) -> tuple[pd.DataFrame, OffseasonDropReport]:
    """One ``leaguegamelog`` (team rows) payload -> validated ``offseason_team_games`` rows."""
    season_start(event_season)
    lo, hi = event_window(context, event_season)
    raw = tf.result_set(payload)
    report = OffseasonDropReport(rows_in=len(raw))
    if raw.empty:
        return empty_table(OFFSEASON_TEAM_GAMES_COLUMNS), report
    where = f"{context} {event_season} team games"
    tf._require(raw, ["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS"], where)
    df = pd.DataFrame({
        "game_id": raw["GAME_ID"].astype(str).str.strip(),
        "game_date": tf.parse_game_date(raw["GAME_DATE"]),
        "team_id": pd.to_numeric(raw["TEAM_ID"], errors="coerce"),
        "team_abbr": raw["TEAM_ABBREVIATION"].astype("string").str.strip().astype(str),
        "matchup": raw["MATCHUP"].astype(str),
        "pts": pd.to_numeric(raw["PTS"], errors="coerce"),
    })
    id_ok = pd.Series([bool(_id_regex(context, event_season).match(g)) for g in df["game_id"]], index=df.index, dtype=bool)
    report.add("unexpected_game_id", (~id_ok).sum())
    df = df[id_ok]
    in_window = df["game_date"].between(lo, hi)
    report.add("outside_event_window", (~in_window).sum())
    df = df[in_window]
    bad = df["team_id"].isna()
    report.add("missing_team_id", bad.sum())
    df = df[~bad].copy()
    df["team_id"] = df["team_id"].astype("int64")
    df = df.drop_duplicates(["game_id", "team_id"])

    df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    totals = df.groupby("game_id")["pts"].agg(["sum", "size"])
    df["opp_pts"] = df["game_id"].map(totals["sum"]) - df["pts"]
    complete = df["game_id"].map(totals["size"]) == 2  # opponent points are only knowable when both rows are present
    df["pts_for"] = df["pts"].where(df["pts"].notna())
    df["pts_against"] = df["opp_pts"].where(complete & df["pts"].notna())
    for col in ("pts_for", "pts_against"):
        df[col] = pd.array(df[col].round(), dtype="Int64")
    df["season"] = previous_season(event_season)
    df["event_season"] = event_season
    df["context"] = context
    out = df[list(OFFSEASON_TEAM_GAMES_COLUMNS)].sort_values(["game_date", "game_id", "team_id"],
                                                             kind="mergesort").reset_index(drop=True)
    report.rows_out = len(out)
    validate_offseason_team_games(out)
    return out, report


# --------------------------------------------------------------------------- validation (ad-hoc tables)

def _kind_ok(kind: str, s: pd.Series) -> bool:
    base = kind.rstrip("?")
    nonnull = s.dropna()
    if not len(nonnull):
        return True
    if base == "int":
        return pd.api.types.is_integer_dtype(nonnull.dtype)
    if base == "float":
        return pd.api.types.is_float_dtype(nonnull.dtype)
    if base == "str":
        return pd.api.types.is_string_dtype(nonnull.dtype) or pd.api.types.is_object_dtype(nonnull.dtype)
    if base == "bool":
        return pd.api.types.is_bool_dtype(nonnull.dtype)
    if base == "date":
        return pd.api.types.is_datetime64_any_dtype(nonnull.dtype)
    raise ValueError(kind)


def _validate(df: pd.DataFrame, name: str, columns: dict[str, str], key: tuple[str, ...]) -> pd.DataFrame:
    problems: list[str] = []
    missing = [c for c in columns if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    for col, kind in columns.items():
        if col not in df.columns:
            continue
        if not kind.endswith("?") and df[col].isna().any():
            problems.append(f"{col}: {int(df[col].isna().sum())} nulls but column is not nullable")
        if not _kind_ok(kind, df[col]):
            problems.append(f"{col}: dtype {df[col].dtype} is not {kind}")
    if not missing and df.duplicated(list(key)).any():
        problems.append(f"{int(df.duplicated(list(key)).sum())} duplicate rows on key {key}")
    if problems:
        raise OffseasonError(f"{name}: " + "; ".join(problems))
    return df


def validate_offseason_logs(df: pd.DataFrame) -> pd.DataFrame:
    _validate(df, "offseason_logs", OFFSEASON_LOGS_COLUMNS, LOG_KEY)
    if len(df):
        if not df["context"].isin(CONTEXTS).all():
            raise OffseasonError("offseason_logs: unknown context value")
        if (df["min"] <= 0).any():
            raise OffseasonError("offseason_logs: rows with non-positive minutes")
        starts = df["event_season"].map(season_start)
        if not (df["season"].map(season_start) == starts - 1).all():
            raise OffseasonError("offseason_logs: `season` must be the season before `event_season`")
        for reason, mask in _identity_violations(df).items():
            if mask.any():
                raise OffseasonError(f"offseason_logs: {int(mask.sum())} rows violate {reason}")
    return df


def validate_offseason_team_games(df: pd.DataFrame) -> pd.DataFrame:
    _validate(df, "offseason_team_games", OFFSEASON_TEAM_GAMES_COLUMNS, TEAM_GAME_KEY)
    if len(df) and not df["context"].isin(CONTEXTS).all():
        raise OffseasonError("offseason_team_games: unknown context value")
    return df


# --------------------------------------------------------------------------- storage

def _table_path(name: str, base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{name}.parquet"


def write_offseason_table(df: pd.DataFrame, name: str, base: Path | None = None) -> Path:
    """Atomic parquet write; these are ad-hoc tables (ADR 0011 D1), so this does not go through ``store``."""
    if name == "offseason_logs":
        validate_offseason_logs(df)
    elif name == "offseason_team_games":
        validate_offseason_team_games(df)
    else:
        raise KeyError(name)
    path = _table_path(name, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_offseason_logs(base: Path | None = None) -> pd.DataFrame:
    path = _table_path("offseason_logs", base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.nba_offseason` run?")
    return pd.read_parquet(path)


def read_offseason_team_games(base: Path | None = None) -> pd.DataFrame:
    path = _table_path("offseason_team_games", base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.nba_offseason` run?")
    return pd.read_parquet(path)


def report_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / REPORT_NAME


def _merge_events(existing: pd.DataFrame | None, new: pd.DataFrame, events: set[tuple[str, str]]) -> pd.DataFrame:
    """Replace exactly the ``(event_season, context)`` events that were just pulled; keep the rest."""
    if existing is not None and len(existing):
        pair = list(zip(existing["event_season"], existing["context"]))
        keep = existing[[p not in events for p in pair]]
        frames = [f for f in (keep, new) if len(f)]
        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else (frames[0].copy() if frames else new)
    else:
        merged = new
    order = merged["event_season"].map(season_start)
    sort_cols = [c for c in ("game_date", "game_id", "team_id", "player_id") if c in merged.columns]
    return (merged.assign(_s=order).sort_values(["_s", "context", *sort_cols], kind="mergesort")
            .drop(columns="_s").reset_index(drop=True))


# --------------------------------------------------------------------------- pipeline

@dataclass
class OffseasonIngestResult:
    events: dict[str, dict[str, Any]] = field(default_factory=dict)  # "<season>/<context>" -> report
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    network_requests: int = 0
    cache_hits: int = 0


def _fetch(client: NBAClient, endpoint: str, params: dict[str, Any], *, refresh: bool) -> dict:
    """Cached fetch; a refresh is silently downgraded to the cache when the client is offline."""
    return client.get(endpoint, params, refresh=refresh and not client.offline)


def run_ingest(
    seasons: list[str],
    client: NBAClient,
    base: Path | None = None,
    *,
    contexts: tuple[str, ...] = CONTEXTS,
    refresh: bool = False,
    live_start: int | None = None,
    log: Callable[[str], None] = print,
) -> OffseasonIngestResult:
    """Pull, transform, validate and merge the offseason events of ``seasons`` into the store.

    Events whose season is the *live* one (``season_start >= live_start``) are always re-downloaded:
    a cached "0 rows" preseason from September must not mask the games played in October.
    """
    if not seasons:
        raise OffseasonError("no seasons to ingest")
    for s in seasons:
        if season_start(s) < MIN_SEASON_START:
            raise OffseasonError(f"{s}: earlier than the {season_str(MIN_SEASON_START)} start of this project's window")
    base = base or data_dir()
    live = live_season_start() if live_start is None else live_start
    result = OffseasonIngestResult()
    new_logs: list[pd.DataFrame] = []
    new_teams: list[pd.DataFrame] = []
    pulled: set[tuple[str, str]] = set()

    for season in seasons:
        for context in contexts:
            do_refresh = refresh or season_start(season) >= live
            log(f"[{season} {context}] pulling{' (refresh)' if do_refresh else ''}")
            p_payload = _fetch(client, "leaguegamelog", player_logs_params(season, context), refresh=do_refresh)
            f_payload = _fetch(client, "playergamelogs", fractional_minutes_params(season, context), refresh=do_refresh)
            t_payload = _fetch(client, "leaguegamelog", team_logs_params(season, context), refresh=do_refresh)
            logs, l_rep = transform_offseason_logs(p_payload, season, context, fractional_payload=f_payload)
            teams, t_rep = transform_offseason_team_games(t_payload, season, context)
            known = set(zip(teams["game_id"], teams["team_id"]))
            orphan = pd.Series([k not in known for k in zip(logs["game_id"], logs["team_id"])], index=logs.index, dtype=bool)
            if orphan.any():
                # Exhibitions against non-NBA clubs: player rows exist for the foreign side, team rows do not.
                l_rep.add("no_matching_team_game", int(orphan.sum()))
                l_rep.rows_out -= int(orphan.sum())
                logs = logs[~orphan].reset_index(drop=True)
            result.events[f"{season}/{context}"] = {
                "player_logs": l_rep.as_dict(), "team_games": t_rep.as_dict(),
                "n_games": int(teams["game_id"].nunique()), "n_players": int(logs["player_id"].nunique()),
                "first_date": None if logs.empty else str(logs["game_date"].min().date()),
                "last_date": None if logs.empty else str(logs["game_date"].max().date()),
            }
            log(f"[{season} {context}] {len(logs):,} player-games, {teams['game_id'].nunique()} games, "
                f"{logs['player_id'].nunique()} players; dropped {l_rep.dropped or 'nothing'}")
            new_logs.append(logs)
            new_teams.append(teams)
            pulled.add((season, context))

    def _existing(name: str) -> pd.DataFrame | None:
        path = _table_path(name, base)
        return pd.read_parquet(path) if path.exists() else None

    def _concat(frames: list[pd.DataFrame], cols: dict[str, str]) -> pd.DataFrame:
        live_frames = [f for f in frames if len(f)]
        return pd.concat(live_frames, ignore_index=True) if live_frames else empty_table(cols)

    logs_all = _merge_events(_existing("offseason_logs"), _concat(new_logs, OFFSEASON_LOGS_COLUMNS), pulled)
    teams_all = _merge_events(_existing("offseason_team_games"), _concat(new_teams, OFFSEASON_TEAM_GAMES_COLUMNS), pulled)
    for name, df in (("offseason_logs", logs_all), ("offseason_team_games", teams_all)):
        old = _existing(name)
        changed = old is None or list(old.columns) != list(df.columns) or len(old) != len(df) or not _same(old, df)
        path = write_offseason_table(df, name, base) if changed else _table_path(name, base)
        result.tables[name] = {"rows": len(df), "changed": bool(changed), "path": str(path)}
        log(f"  {name}: {len(df):,} rows ({'written' if changed else 'unchanged'})")
    _write_report(base, result)
    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    return result


_EMPTY_DTYPE = {"str": "object", "int": "int64", "float": "float64", "bool": "bool", "date": "datetime64[ns]"}
_EMPTY_NULLABLE_INT = "Int64"


def empty_table(columns: dict[str, str]) -> pd.DataFrame:
    """A zero-row frame with the table's exact dtypes (nullable ints where the schema says ``int?``)."""
    return pd.DataFrame({
        c: pd.Series(dtype=_EMPTY_NULLABLE_INT if k == "int?" else _EMPTY_DTYPE[k.rstrip("?")])
        for c, k in columns.items()
    })


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        return False


def _write_report(base: Path, result: OffseasonIngestResult) -> None:
    """Merged by event and free of timestamps, so an identical re-run leaves the file byte-identical."""
    path = report_path(base)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"events": {}}
    report["events"].update(json.loads(json.dumps(result.events, default=str)))
    report["events"] = dict(sorted(report["events"].items(), key=lambda kv: (season_start(kv[0].split("/")[0]), kv[0])))
    atomic_write_bytes(path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.nba_offseason",
        description="Ingest Summer League and preseason box scores from stats.nba.com (unofficial) into the data store.",
    )
    p.add_argument("--seasons", default=f"{season_str(MIN_SEASON_START)}:{season_str(live_season_start())}",
                   help="event seasons, e.g. 2015-16:2026-27 (default: every season from 2015-16 to the live one). "
                        "July 2026 Summer League and October 2026 preseason are both '2026-27'.")
    p.add_argument("--context", choices=[*CONTEXTS, "all"], default="all")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--refresh", action="store_true", help="re-download every requested event, not only the live season's")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0, help="minimum seconds between network requests (default 1.0)")
    p.add_argument("--max-retries", type=int, default=6)
    return p


def main(argv: list[str] | None = None, *, client: NBAClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_interval < 0.8 and not args.offline:
        print(f"warning: --min-interval {args.min_interval} is below the 0.8 s politeness floor; using 0.8", file=sys.stderr)
        args.min_interval = 0.8
    base = args.data_dir or data_dir()
    if client is None:
        client = NBAClient(base / "raw" / "nba_api", offline=True if args.offline else None,
                           min_interval=args.min_interval, max_retries=args.max_retries)
    contexts = CONTEXTS if args.context == "all" else (args.context,)
    try:
        result = run_ingest(seasons, client, base, contexts=contexts, refresh=args.refresh)
    except (NBAClientError, OffseasonError, tf.TransformError) as exc:
        print(f"ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {len(result.events)} events, {result.network_requests} network requests, {result.cache_hits} cache hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
