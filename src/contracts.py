"""Shared data contract for the whole project.

Every track (ingest, features, models, value, backtest, app) codes against this
module, never against each other's internals. Changing a schema here is a
cross-track decision: update docs/adr/0001-data-contract.md in the same change.

Conventions
-----------
* Seasons are strings like ``"2023-24"``. ``season_start("2023-24") == 2023``.
* Only **regular-season** games exist in these tables (ESPN H2H points ignores playoffs).
* Player identity is the NBA ``PERSON_ID`` (``player_id``). Other sources are
  mapped onto it through the ``player_id_map`` table.
* A player who did not play (DNP / inactive) has **no** ``game_logs`` row.
  Availability is derived from ``team_games`` minus a player's played games.
* Column names are lowercase snake_case. Scoring keys used by the league config
  (``3PM``, ``TO``) differ from column names (``fg3m``, ``tov``); use
  ``STAT_COLUMN_MAP`` to translate.
* Leakage rule: anything used to project season S may only use information dated
  before season S starts. ``History`` enforces this structurally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import pandas as pd

# --------------------------------------------------------------------------- seasons

FIRST_BACKTEST_START = 2015  # 2015-16
LAST_COMPLETED_START = 2025  # 2025-26 (finished before this project began)


def season_str(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start(season: str) -> int:
    """'2023-24' -> 2023. Raises ValueError on malformed input."""
    try:
        start, end = season.split("-")
        if (len(start) != 4 or len(end) != 2 or not (start.isascii() and start.isdigit() and end.isascii() and end.isdigit())
                or int(end) != (int(start) + 1) % 100):
            raise ValueError
        return int(start)
    except (ValueError, AttributeError):
        raise ValueError(f"malformed season {season!r}; expected like '2023-24'") from None


def seasons_between(first_start: int, last_start: int) -> list[str]:
    """Inclusive list of season strings."""
    return [season_str(y) for y in range(first_start, last_start + 1)]


BACKTEST_SEASONS = seasons_between(FIRST_BACKTEST_START, LAST_COMPLETED_START)

# --------------------------------------------------------------------------- scoring keys

# league.yaml scoring key -> game_logs column
STAT_COLUMN_MAP: dict[str, str] = {
    "PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov",
    "FGM": "fgm", "FGA": "fga", "FTM": "ftm", "FTA": "fta", "3PM": "fg3m",
}

# --------------------------------------------------------------------------- storage

def data_dir() -> Path:
    """Shared data root. Lives OUTSIDE the repo and OneDrive so every worktree sees the same data.

    Override with ``NBA_DATA_DIR``. Layout: ``raw/<source>/...`` and ``processed/<table>.parquet``.
    """
    return Path(os.environ.get("NBA_DATA_DIR", Path.home() / "dev-data" / "nba-fantasy-2026"))


def raw_dir(source: str) -> Path:
    return data_dir() / "raw" / source


def table_path(name: str, base: Path | None = None) -> Path:
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; known: {sorted(TABLES)}")
    return (base or data_dir()) / "processed" / f"{name}.parquet"


# --------------------------------------------------------------------------- table specs

class ContractError(ValueError):
    """A DataFrame does not satisfy its table contract."""


# kind -> predicate on a pandas dtype
_KINDS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "str": lambda dt: pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt),
    "bool": pd.api.types.is_bool_dtype,
    "date": pd.api.types.is_datetime64_any_dtype,
}


@dataclass(frozen=True)
class Col:
    kind: str                 # int | float | str | bool | date
    nullable: bool = False
    doc: str = ""


@dataclass(frozen=True)
class TableSpec:
    name: str
    key: tuple[str, ...]      # unique row identity
    columns: Mapping[str, Col]
    doc: str = ""


def _ints(*names: str) -> dict[str, Col]:
    return {n: Col("int") for n in names}


TABLES: dict[str, TableSpec] = {}


def _register(spec: TableSpec) -> TableSpec:
    TABLES[spec.name] = spec
    return spec


GAME_LOGS = _register(TableSpec(
    "game_logs", key=("game_id", "player_id"),
    doc="One row per player per regular-season game actually played (minutes > 0).",
    columns={
        "season": Col("str"),
        "game_id": Col("str", doc="NBA game id, shared by both teams' rows"),
        "game_date": Col("date"),
        "player_id": Col("int"),
        "player_name": Col("str"),
        "team_id": Col("int"),
        "team_abbr": Col("str"),
        "matchup": Col("str", doc="e.g. 'BOS vs. NYK' (home) or 'BOS @ NYK' (away)"),
        "min": Col("float", doc="minutes played, fractional"),
        **_ints("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
                "ast", "stl", "blk", "tov", "pf", "pts"),
        "plus_minus": Col("float", nullable=True),
    },
))

TEAM_GAMES = _register(TableSpec(
    "team_games", key=("game_id", "team_id"),
    doc="Every regular-season game each team played. Needed to compute games a player missed.",
    columns={
        "season": Col("str"),
        "game_id": Col("str"),
        "game_date": Col("date"),
        "team_id": Col("int"),
        "team_abbr": Col("str"),
        "is_home": Col("bool"),
        "pts_for": Col("int", nullable=True),
        "pts_against": Col("int", nullable=True),
    },
))

PLAYERS = _register(TableSpec(
    "players", key=("player_id",),
    doc="Static player attributes, one row per player.",
    columns={
        "player_id": Col("int"),
        "player_name": Col("str"),
        "birthdate": Col("date", nullable=True),
        "position": Col("str", nullable=True, doc="source position string, e.g. 'G', 'F-C'"),
        "height_in": Col("float", nullable=True),
        "weight_lb": Col("float", nullable=True),
        "draft_year": Col("int", nullable=True, doc="null = undrafted"),
        "draft_round": Col("int", nullable=True),
        "draft_number": Col("int", nullable=True),
        "from_year": Col("int", nullable=True, doc="first NBA season start year"),
        "to_year": Col("int", nullable=True),
    },
))

PLAYER_SEASON_BIO = _register(TableSpec(
    "player_season_bio", key=("season", "player_id"),
    doc="Per-season attributes that change over time.",
    columns={
        "season": Col("str"),
        "player_id": Col("int"),
        "age_at_season_start": Col("float", doc="years, as of Oct 1 of the season's start year"),
        "team_id": Col("int", nullable=True, doc="team at end of regular season"),
    },
))

PLAYER_ID_MAP = _register(TableSpec(
    "player_id_map", key=("source", "source_id"),
    doc="Maps every external source's player identifier onto the canonical NBA player_id.",
    columns={
        "player_id": Col("int"),
        "source": Col("str", doc="espn | spotrac | hoopshype | bbref | ..."),
        "source_id": Col("str"),
        "source_name": Col("str"),
        "match_method": Col("str", doc="exact | normalized | manual | fuzzy"),
        "confidence": Col("float"),
    },
))

# Per-game stat projections use the scoring keys lower-cased with a proj_ prefix.
PROJECTION_STATS = tuple(f"proj_{c}" for c in STAT_COLUMN_MAP.values())

PROJECTIONS = _register(TableSpec(
    "projections", key=("season", "player_id", "model"),
    doc="A model's preseason projection for one player-season. FP columns use league scoring.",
    columns={
        "season": Col("str"),
        "player_id": Col("int"),
        "player_name": Col("str"),
        "model": Col("str"),
        "proj_gp": Col("float", doc="expected games played (mean of availability distribution)"),
        "proj_mpg": Col("float"),
        **{c: Col("float") for c in PROJECTION_STATS},
        "proj_fppg": Col("float", doc="projected fantasy points per game played"),
        "proj_total_fp": Col("float", doc="proj_fppg * proj_gp"),
        "fppg_p10": Col("float", nullable=True, doc="floor: 10th percentile of game-level FP"),
        "fppg_p50": Col("float", nullable=True),
        "fppg_p90": Col("float", nullable=True, doc="ceiling: 90th percentile of game-level FP"),
    },
))


def validate_table(df: pd.DataFrame, name: str, *, allow_extra: bool = True) -> pd.DataFrame:
    """Raise ContractError unless ``df`` satisfies table ``name``. Returns df for chaining."""
    spec = TABLES[name]
    problems: list[str] = []
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in spec.columns]
        if extra:
            problems.append(f"unexpected columns: {extra}")
    for col, c in spec.columns.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not c.nullable and s.isna().any():
            problems.append(f"{col}: {int(s.isna().sum())} nulls but column is not nullable")
        # Nullable int columns are float/Int64 with NaN, so check the non-null values.
        check = s.dropna()
        if len(check) and not _KINDS[c.kind](check.dtype):
            if not (c.kind == "int" and c.nullable and _all_integral(check)):
                problems.append(f"{col}: dtype {s.dtype} is not {c.kind}")
        elif c.kind == "str" and len(check) and pd.api.types.is_object_dtype(check.dtype)                 and not check.map(lambda v: isinstance(v, str)).all():
            problems.append(f"{col}: object column holds non-string values")   # e.g. ints mixed into a str column
    if not missing:
        dup = df.duplicated(list(spec.key)).sum()
        if dup:
            problems.append(f"{int(dup)} duplicate rows on key {spec.key}")
    if not problems and name == "game_logs":
        problems.extend(_game_log_invariants(df))
    if problems:
        raise ContractError(f"{name}: " + "; ".join(problems))
    return df


def _all_integral(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and bool(((s % 1) == 0).all())


def _game_log_invariants(df: pd.DataFrame) -> list[str]:
    """Box-score identities that must hold in real data too; violations mean a bad ingest."""
    bad: list[str] = []
    checks = {
        "fgm > fga": df["fgm"] > df["fga"],
        "fg3m > fgm": df["fg3m"] > df["fgm"],
        "fg3a > fga": df["fg3a"] > df["fga"],
        "ftm > fta": df["ftm"] > df["fta"],
        "reb != oreb + dreb": df["reb"] != df["oreb"] + df["dreb"],
        "pts != 2*fgm + fg3m + ftm": df["pts"] != 2 * df["fgm"] + df["fg3m"] + df["ftm"],
        "min <= 0": df["min"] <= 0,
        "negative counting stat": (df[["fgm", "fga", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"]] < 0).any(axis=1),
    }
    for label, mask in checks.items():
        n = int(mask.sum())
        if n:
            bad.append(f"{n} rows violate '{label}'")
    return bad


# --------------------------------------------------------------------------- History (leakage guard)

HISTORY_TABLES = ("game_logs", "team_games", "players", "player_season_bio")


def _sanitize_players(players: pd.DataFrame, cutoff: int, known_player_ids: pd.Index | set) -> pd.DataFrame:
    """Return ``players`` as it could have been known before season-start-year ``cutoff``.

    ``players`` is static (one row per player across all seasons ever ingested), so unlike the
    other tables it cannot be fixed by dropping rows with ``season >= cutoff`` — it has no season
    column. Two of its columns are nonetheless derived from the future: ``to_year`` (last season
    played) and ``from_year`` (first season played) reveal exactly who is still active, and who
    is about to debut, in ``target_season``.

    A player is knowable before ``target_season`` if either is true:

    * they were **drafted** by this season's draft (``draft_year <= cutoff``) — the draft happens
      before the season, so this is legitimately known in advance, even for a player who hasn't
      played a single NBA game yet; or
    * they have **actually played** before ``target_season`` — i.e. ``player_id`` appears in
      ``game_logs`` or ``player_season_bio`` restricted to seasons < ``cutoff`` (``known_player_ids``,
      computed by the caller from the *already season-sliced* tables).

    Deliberately not ``from_year >= cutoff`` for this decision: ``from_year`` is a *derived*
    summary column and is only as trustworthy as whatever computed it, whereas actually appearing
    in the season-sliced game log tables is true by construction. Everyone else is dropped.
    Kept rows still get ``from_year >= cutoff`` blanked to NaN and ``to_year`` capped at
    ``cutoff - 1``, purely as defense in depth (e.g. against a caller who reads these columns
    without re-deriving them from game logs).

    Never mutates the input. Idempotent given the same ``known_player_ids``.
    """
    p = players.copy()
    draft = pd.to_numeric(p["draft_year"], errors="coerce")
    frm = pd.to_numeric(p["from_year"], errors="coerce")
    to = pd.to_numeric(p["to_year"], errors="coerce")
    drafted_in_time = (draft <= cutoff).fillna(False)
    already_played = p["player_id"].isin(known_player_ids)
    keep = (drafted_in_time | already_played).to_numpy()
    p, frm, to = p[keep].copy(), frm[keep], to[keep]
    p["from_year"] = frm.where(~(frm >= cutoff)).astype("Int64")
    p["to_year"] = to.clip(upper=cutoff - 1).astype("Int64")
    return p.reset_index(drop=True)


@dataclass
class History:
    """Everything a projector is allowed to see when projecting ``target_season``.

    Build it with ``History.until(tables, target_season)``: that slices every
    season-indexed table to seasons strictly before the target, and sanitises the static
    ``players`` table (see ``_sanitize_players``) so its ``from_year``/``to_year``/``draft_year``
    columns cannot reveal who debuts, retires, or gets drafted in ``target_season`` or later.
    Projectors receive only a History, so they cannot read the future by construction.
    """

    target_season: str
    game_logs: pd.DataFrame
    team_games: pd.DataFrame
    players: pd.DataFrame
    player_season_bio: pd.DataFrame
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def until(cls, tables: Mapping[str, pd.DataFrame], target_season: str) -> "History":
        cutoff = season_start(target_season)

        def before(df: pd.DataFrame) -> pd.DataFrame:
            if "season" not in df.columns:
                return df
            starts = df["season"].map(season_start)
            return df[starts < cutoff].reset_index(drop=True)

        extras = {k: before(v) for k, v in tables.items() if k not in HISTORY_TABLES}
        game_logs = before(tables["game_logs"])
        player_season_bio = before(tables["player_season_bio"])
        known_player_ids = set(game_logs["player_id"]) | set(player_season_bio["player_id"])
        return cls(
            target_season=target_season,
            game_logs=game_logs,
            team_games=before(tables["team_games"]),
            players=_sanitize_players(tables["players"], cutoff, known_player_ids),
            player_season_bio=player_season_bio,
            extras=extras,
        )

    def assert_no_future(self) -> None:
        """Raise if any season-indexed frame, or ``players``, reveals target_season or later."""
        cutoff = season_start(self.target_season)
        for name in ("game_logs", "team_games", "player_season_bio"):
            df = getattr(self, name)
            if len(df) and (df["season"].map(season_start) >= cutoff).any():
                raise AssertionError(f"leakage: {name} contains seasons >= {self.target_season}")
        if len(self.players):
            for col, bad in (
                ("from_year", lambda x: x >= cutoff),
                ("to_year", lambda x: x >= cutoff),
                ("draft_year", lambda x: x > cutoff),
            ):
                v = pd.to_numeric(self.players[col], errors="coerce")
                if bad(v).any():
                    raise AssertionError(
                        f"leakage: players.{col} reveals seasons >= {self.target_season}")

    @property
    def last_season(self) -> str | None:
        if self.game_logs.empty:
            return None
        return season_str(int(self.game_logs["season"].map(season_start).max()))


@runtime_checkable
class Projector(Protocol):
    """Anything the backtest and the draft board can consume."""

    name: str

    def project(self, history: History) -> pd.DataFrame:
        """Return a ``projections`` table (see PROJECTIONS) for ``history.target_season``."""
        ...
