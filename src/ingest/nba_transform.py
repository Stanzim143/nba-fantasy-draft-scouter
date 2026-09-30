"""Pure functions: raw stats.nba.com JSON -> contract DataFrames.

No IO, no network, no clocks. Everything here takes already-parsed payloads (dicts) and returns
DataFrames plus a small *drop report* saying what was discarded and why. Every produced table is
validated with ``validate_table`` before it is returned, so a bad transform fails here, loudly,
not in some consumer.

Real-response facts this code is built around (observed 2026-09, see docs/data-quality.md):

* ``playergamelogs`` (all players, one season, one call) has fractional ``MIN`` (a float such as
  ``36.4516``); ``leaguegamelog`` with ``PlayerOrTeam=P`` has the same rows but *integer* minutes,
  so it is only used as a cross-check. ``leaguegamelog`` with ``PlayerOrTeam=T`` is the team-game
  source (two rows per game, ``MIN`` is 240 + 25 per overtime).
* ``GAME_DATE`` is ``"2019-04-10T00:00:00"`` in ``playergamelogs`` and ``"2019-04-10"`` in
  ``leaguegamelog``. ``GAME_ID`` is ``"002"`` + two-digit season start year + five-digit sequence.
* Draft fields are strings (``"2014"``, ``"Undrafted"``) in some endpoints and floats/NaN in
  others; heights are ``"6-9"``.
* ``team_id`` is the stable identity; abbreviations are whatever the source reports per row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.contracts import TABLES, season_start, validate_table


class TransformError(ValueError):
    """Raw data cannot be interpreted (missing columns, unparseable values, wrong season)."""


# --------------------------------------------------------------------------- drop report

@dataclass
class DropReport:
    """Counts of rows discarded by a transform, keyed by reason. Never silently lossy."""

    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    info: dict[str, int] = field(default_factory=dict)  # facts about dropped rows; not extra drops

    def add(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def note(self, key: str, n: int) -> None:
        if n:
            self.info[key] = self.info.get(key, 0) + int(n)

    def merge(self, other: "DropReport") -> "DropReport":
        out = DropReport(self.rows_in + other.rows_in, self.rows_out + other.rows_out, dict(self.dropped), dict(self.info))
        for k, v in other.dropped.items():
            out.add(k, v)
        for k, v in other.info.items():
            out.note(k, v)
        return out

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rows_in": self.rows_in, "rows_out": self.rows_out, "dropped": dict(self.dropped)}
        if self.info:
            d["info"] = dict(self.info)
        return d


# --------------------------------------------------------------------------- generic parsing

def result_set(payload: Mapping[str, Any], name: str | None = None) -> pd.DataFrame:
    """Turn one named ``resultSets`` entry (headers + rowSet) into a DataFrame.

    ``name=None`` takes the first set. Raises TransformError when the payload has no such set.
    """
    if not isinstance(payload, Mapping):
        raise TransformError(f"payload is {type(payload).__name__}, expected a JSON object")
    sets = payload.get("resultSets", payload.get("resultSet"))
    if isinstance(sets, Mapping):
        sets = [sets]
    if not isinstance(sets, list) or not sets:
        raise TransformError("payload has no resultSets")
    if name is None:
        chosen = sets[0]
    else:
        matches = [s for s in sets if s.get("name") == name]
        if not matches:
            raise TransformError(f"no result set named {name!r}; available: {[s.get('name') for s in sets]}")
        chosen = matches[0]
    headers, rows = chosen.get("headers"), chosen.get("rowSet")
    if headers is None or rows is None:
        raise TransformError(f"result set {chosen.get('name')!r} lacks headers/rowSet")
    return pd.DataFrame(rows, columns=headers)


def _require(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise TransformError(f"{where}: missing columns {missing}; got {list(df.columns)}")


_MMSS = re.compile(r"^\s*(\d+):(\d{1,2})(?:\.(\d+))?\s*$")


def parse_minutes(values: pd.Series) -> pd.Series:
    """Minutes as float. Accepts numbers, ``"MM:SS"`` / ``"MM:SS.ff"`` strings, numeric strings.

    Empty / ``None`` / ``NaN`` / ``"-"`` become NaN (= no minutes recorded).
    """
    def one(v: Any) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip()
        if s in ("", "-", "--", "None", "nan", "NaN"):
            return np.nan
        m = _MMSS.match(s)
        if m:
            secs = int(m.group(2)) + (float("0." + m.group(3)) if m.group(3) else 0.0)
            return int(m.group(1)) + secs / 60.0
        try:
            return float(s)
        except ValueError:
            raise TransformError(f"unparseable minutes value {v!r}") from None

    return pd.Series([one(v) for v in values], index=values.index, dtype="float64")


def parse_game_date(values: pd.Series) -> pd.Series:
    """ISO ``YYYY-MM-DD`` (optionally with a ``T00:00:00`` tail) or ``"APR 10, 2019"`` -> datetime64."""
    text = values.astype("string").str.strip()
    out = pd.to_datetime(text.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    bad = out.isna()
    if bad.any():  # older/other endpoints: "APR 10, 2019"
        out[bad] = pd.to_datetime(text[bad], format="%b %d, %Y", errors="coerce")
    if out.isna().any():
        samples = text[out.isna()].head(3).tolist()
        raise TransformError(f"{int(out.isna().sum())} unparseable game dates, e.g. {samples}")
    return out.astype("datetime64[ns]")


def parse_height_inches(values: pd.Series) -> pd.Series:
    """``"6-9"`` -> 81.0; anything else (blank, ``"6-"``, NaN) -> NaN."""
    def one(v: Any) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) if v > 30 else np.nan  # already inches
        m = re.match(r"^\s*(\d+)\s*-\s*(\d{1,2})\s*$", str(v))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

    return pd.Series([one(v) for v in values], index=values.index, dtype="float64")


def _nullable_int(values: pd.Series) -> pd.Series:
    """Numeric-with-junk (``"Undrafted"``, ``""``, NaN, ``"2014"``) -> pandas ``Int64``."""
    num = pd.to_numeric(values, errors="coerce")
    return num.round().astype("Int64")


def _clean_str(values: pd.Series) -> pd.Series:
    s = values.astype("string").str.strip()
    return s.mask(s.isin(["", "None", "nan", "NaN"]))


_POSITION_WORDS = {"Guard": "G", "Forward": "F", "Center": "C"}


def normalize_position(value: Any) -> str | None:
    """``"Guard-Forward"`` -> ``"G-F"``; ``"G-F"`` stays; blank -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    return "-".join(_POSITION_WORDS.get(p.strip(), p.strip()) for p in s.split("-"))


# --------------------------------------------------------------------------- season / game id logic

def game_id_prefix(season: str) -> str:
    """Regular-season game ids for season ``"2018-19"`` start with ``"00218"``."""
    return f"002{season_start(season) % 100:02d}"


_REGULAR_GAME_ID = re.compile(r"^002\d{2}\d{5}$")


def is_regular_season_game_id(game_id: str, season: str) -> bool:
    """True only for ids like ``0021800001`` for this season.

    Excludes preseason (001), All-Star (003), playoffs (004), play-in (005), NBA Cup knockout
    (006) and other leagues/seasons.
    """
    return bool(_REGULAR_GAME_ID.match(game_id)) and game_id.startswith(game_id_prefix(season))


# --------------------------------------------------------------------------- game_logs

_INT_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
              "ast", "stl", "blk", "tov", "pf", "pts"]
_GL_SOURCE = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team_abbr", "GAME_ID": "game_id", "GAME_DATE": "game_date",
    "MATCHUP": "matchup", "PLUS_MINUS": "plus_minus",
    **{c.upper(): c for c in _INT_STATS},
}
_GAME_LOG_SORT = ["game_date", "game_id", "team_id", "player_id"]


def transform_game_logs(payload: Mapping[str, Any], season: str) -> tuple[pd.DataFrame, DropReport]:
    """One season of all players' game rows -> ``game_logs`` (validated).

    Accepts a ``playergamelogs`` payload (fractional MIN) or a ``leaguegamelog`` ``PlayerOrTeam=P``
    payload (integer MIN). Drops, with counts: non-regular-season game ids, rows with no/zero
    minutes (did not play), rows missing box-score fields, and duplicate (game, player) rows
    (identical duplicates first; conflicting duplicates keep the row with the most minutes).
    """
    season_start(season)  # validates format
    raw = result_set(payload)
    report = DropReport(rows_in=len(raw))
    _require(raw, list(_GL_SOURCE), f"game logs {season}")
    df = raw.rename(columns=_GL_SOURCE)[list(_GL_SOURCE.values())].copy()

    # Minutes: prefer MIN; fall back to MIN_SEC ("MM:SS") if MIN is absent or entirely null.
    minutes = raw["MIN"] if "MIN" in raw.columns else pd.Series(np.nan, index=raw.index)
    if minutes.isna().all() and "MIN_SEC" in raw.columns:
        minutes = raw["MIN_SEC"]
    df["min"] = parse_minutes(minutes)

    src_season = None
    if "SEASON_YEAR" in raw.columns:
        src_season = raw["SEASON_YEAR"]
    if src_season is not None and (src_season.astype(str) != season).any():
        wrong = sorted(set(src_season.astype(str)) - {season})
        raise TransformError(f"game logs for {season} contain rows for other seasons {wrong}")

    df["game_id"] = df["game_id"].astype(str).str.strip()
    ok_id = pd.Series([is_regular_season_game_id(g, season) for g in df["game_id"]], index=df.index, dtype=bool)
    report.add("not_regular_season_game_id", (~ok_id).sum())
    df = df[ok_id]

    no_min = df["min"].isna() | (df["min"] <= 0)
    report.add("no_minutes_played", no_min.sum())
    if no_min.any():  # e.g. a 0:00 row that still credits a free throw: dropped (contract: min > 0) but reported
        stat_cols = [c for c in _INT_STATS if c not in ("reb",)]
        recorded = df.loc[no_min, stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=1) > 0
        report.note("no_minutes_rows_with_recorded_stats", recorded.sum())
        report.note("no_minutes_points_lost", pd.to_numeric(df.loc[no_min, "pts"], errors="coerce").fillna(0).sum())
    df = df[~no_min]

    df["game_date"] = parse_game_date(df["game_date"])
    for col in ("player_id", "team_id", *_INT_STATS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    incomplete = df[["player_id", "team_id", *_INT_STATS]].isna().any(axis=1)
    report.add("missing_box_score_fields", incomplete.sum())
    df = df[~incomplete]
    for col in ("player_id", "team_id", *_INT_STATS):
        df[col] = df[col].astype("int64")
    df["plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").astype("float64")
    for col in ("player_name", "team_abbr", "matchup"):
        df[col] = df[col].astype("string").str.strip().astype(str)
    df["season"] = season

    # Duplicates: identical first, then conflicting (keep most minutes, deterministic).
    key = ["game_id", "player_id"]
    before = len(df)
    df = df.drop_duplicates()
    report.add("duplicate_row_identical", before - len(df))
    if df.duplicated(key).any():
        before = len(df)
        df = df.sort_values(key + ["min", "pts"], ascending=[True, True, False, False]).drop_duplicates(key)
        report.add("duplicate_row_conflicting", before - len(df))

    cols = list(TABLES["game_logs"].columns)
    df = df[cols].sort_values(_GAME_LOG_SORT, kind="mergesort").reset_index(drop=True)
    report.rows_out = len(df)
    validate_table(df, "game_logs")
    return df, report


def crosscheck_game_logs(primary: pd.DataFrame, secondary: pd.DataFrame, *, minute_tolerance: float = 1.0) -> dict[str, Any]:
    """Compare two independently pulled versions of the same player rows (e.g. the fractional-minute
    ``playergamelogs`` source vs the integer-minute ``leaguegamelog`` source).

    Both inputs are ``game_logs`` frames (after transform). Returns counts; nothing raises, the
    caller decides what is acceptable.
    """
    key = ["game_id", "player_id"]
    a = primary.set_index(key)
    b = secondary.set_index(key)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    both = a.index.intersection(b.index)
    stat_cols = [c for c in _INT_STATS if c in a.columns] + ["team_id"]
    mism = {}
    for c in stat_cols:
        n = int((a.loc[both, c].to_numpy() != b.loc[both, c].to_numpy()).sum())
        if n:
            mism[c] = n
    dmin = (a.loc[both, "min"].to_numpy() - b.loc[both, "min"].to_numpy())
    # The integer-minute source drops players who logged under 30 seconds (MIN rounds to 0), so rows
    # only in the primary source are expected iff their minutes are < 0.5.
    unexpected_only_a = a.loc[only_a][a.loc[only_a, "min"] >= 0.5].index
    return {
        "rows_primary": len(a), "rows_secondary": len(b), "rows_common": len(both),
        "only_in_primary": [tuple(x) for x in only_a[:20]], "n_only_in_primary": len(only_a),
        "n_only_in_primary_over_half_minute": len(unexpected_only_a),
        "only_in_secondary": [tuple(x) for x in only_b[:20]], "n_only_in_secondary": len(only_b),
        "stat_mismatches": mism,
        "minutes_out_of_tolerance": int((np.abs(dmin) > minute_tolerance).sum()) if len(both) else 0,
        "minutes_max_abs_diff": float(np.abs(dmin).max()) if len(both) else 0.0,
    }


# --------------------------------------------------------------------------- team_games

def transform_team_games(payload: Mapping[str, Any], season: str) -> tuple[pd.DataFrame, DropReport]:
    """One season of ``leaguegamelog`` (``PlayerOrTeam=T``) -> ``team_games`` (validated).

    ``is_home`` comes from the MATCHUP string (``"BOS vs. NYK"`` home, ``"BOS @ NYK"`` away).
    ``pts_against`` is the opponent's PTS from the other row of the same game; it is null only if a
    game has a single team row.
    """
    season_start(season)
    raw = result_set(payload)
    report = DropReport(rows_in=len(raw))
    _require(raw, ["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS"], f"team games {season}")
    df = pd.DataFrame({
        "game_id": raw["GAME_ID"].astype(str).str.strip(),
        "game_date": raw["GAME_DATE"],
        "team_id": pd.to_numeric(raw["TEAM_ID"], errors="coerce"),
        "team_abbr": raw["TEAM_ABBREVIATION"].astype("string").str.strip().astype(str),
        "matchup": raw["MATCHUP"].astype("string"),
        "pts": pd.to_numeric(raw["PTS"], errors="coerce"),
    })
    ok_id = pd.Series([is_regular_season_game_id(g, season) for g in df["game_id"]], index=df.index, dtype=bool)
    report.add("not_regular_season_game_id", (~ok_id).sum())
    df = df[ok_id].copy()

    bad = df["team_id"].isna()
    report.add("missing_team_id", bad.sum())
    df = df[~bad].copy()
    df["team_id"] = df["team_id"].astype("int64")
    df["game_date"] = parse_game_date(df["game_date"])

    n = len(df)
    df = df.drop_duplicates(["game_id", "team_id"])
    report.add("duplicate_row", n - len(df))

    is_vs = df["matchup"].str.contains(" vs. ", regex=False)
    is_at = df["matchup"].str.contains(" @ ", regex=False)
    unknown = ~(is_vs ^ is_at)
    if unknown.any():
        raise TransformError(f"{int(unknown.sum())} team rows with unrecognised MATCHUP, e.g. {df.loc[unknown, 'matchup'].head(3).tolist()}")
    df["is_home"] = is_vs.to_numpy()

    # groupby(...).sum() silently skips NaN by default: if one team's row in a game has a missing
    # `pts` (a malformed source row), a naive sum would understate the group total and make the
    # *other* team's computed pts_against come out as a wrong non-null number (e.g. 0) instead of
    # unknown. Guard explicitly: any missing pts in the game nulls pts_against for both rows.
    pts_sum = df.groupby("game_id")["pts"].transform("sum")
    n_rows = df.groupby("game_id")["pts"].transform("size")
    has_missing_pts = df.groupby("game_id")["pts"].transform(lambda s: s.isna().any())
    df["pts_for"] = df["pts"]
    df["pts_against"] = (pts_sum - df["pts"]).where((n_rows == 2) & ~has_missing_pts)
    for col in ("pts_for", "pts_against"):
        s = df[col]
        df[col] = s.astype("int64") if not s.isna().any() else s.round().astype("Int64")
    df["season"] = season

    cols = list(TABLES["team_games"].columns)
    df = df[cols].sort_values(["game_date", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)
    report.rows_out = len(df)
    validate_table(df, "team_games")
    return df, report


# --------------------------------------------------------------------------- players / bio

def parse_birthdate(value: Any) -> pd.Timestamp | None:
    """``"1984-12-30T00:00:00"`` / ``"1984-12-30"`` -> Timestamp; blank / junk -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()[:10]
    if not s:
        return None
    ts = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(ts) or ts.year < 1900 or ts.year > date.today().year:
        return None
    return ts


def birthdates_from_common_player_info(payload: Mapping[str, Any]) -> tuple[int, pd.Timestamp | None]:
    """A ``commonplayerinfo`` payload -> ``(player_id, birthdate or None)``."""
    info = result_set(payload, "CommonPlayerInfo")
    if info.empty:
        raise TransformError("commonplayerinfo returned no CommonPlayerInfo row")
    _require(info, ["PERSON_ID", "BIRTHDATE"], "commonplayerinfo")
    row = info.iloc[0]
    return int(row["PERSON_ID"]), parse_birthdate(row["BIRTHDATE"])


def player_attributes_from_common_player_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Static attributes from a ``commonplayerinfo`` payload, in ``players`` column terms."""
    info = result_set(payload, "CommonPlayerInfo")
    if info.empty:
        raise TransformError("commonplayerinfo returned no CommonPlayerInfo row")
    r = info.iloc[0]
    def num(col):
        v = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]
        return None if pd.isna(v) else v
    return {
        "player_id": int(r["PERSON_ID"]),
        "birthdate": parse_birthdate(r.get("BIRTHDATE")),
        "position": normalize_position(r.get("POSITION")),
        "height_in": parse_height_inches(pd.Series([r.get("HEIGHT")])).iloc[0],
        "weight_lb": num("WEIGHT"),
        "draft_year": num("DRAFT_YEAR"), "draft_round": num("DRAFT_ROUND"), "draft_number": num("DRAFT_NUMBER"),
        "from_year": num("FROM_YEAR"), "to_year": num("TO_YEAR"),
    }


def player_index_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """``playerindex`` payload -> attributes keyed by player_id (columns are ``players``-style, unfiltered)."""
    raw = result_set(payload, "PlayerIndex")
    _require(raw, ["PERSON_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME"], "playerindex")
    out = pd.DataFrame({
        "player_id": pd.to_numeric(raw["PERSON_ID"]).astype("int64"),
        "player_name": (raw["PLAYER_FIRST_NAME"].fillna("").astype(str).str.strip() + " "
                        + raw["PLAYER_LAST_NAME"].fillna("").astype(str).str.strip()).str.strip(),
        "position": pd.Series([normalize_position(v) for v in raw["POSITION"]], index=raw.index, dtype="object") if "POSITION" in raw else None,
        "height_in": parse_height_inches(raw["HEIGHT"]) if "HEIGHT" in raw else np.nan,
        "weight_lb": pd.to_numeric(raw["WEIGHT"], errors="coerce") if "WEIGHT" in raw else np.nan,
        "draft_year": _nullable_int(raw["DRAFT_YEAR"]) if "DRAFT_YEAR" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "draft_round": _nullable_int(raw["DRAFT_ROUND"]) if "DRAFT_ROUND" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "draft_number": _nullable_int(raw["DRAFT_NUMBER"]) if "DRAFT_NUMBER" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "from_year": _nullable_int(raw["FROM_YEAR"]) if "FROM_YEAR" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "to_year": _nullable_int(raw["TO_YEAR"]) if "TO_YEAR" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
    })
    return out.drop_duplicates("player_id", keep="first").reset_index(drop=True)


_BIO_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "AGE"]


def bio_stats_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """``leaguedashplayerbiostats`` payload -> player_id, integer season age, height/weight/draft."""
    raw = result_set(payload, "LeagueDashPlayerBioStats")
    _require(raw, _BIO_COLS, "leaguedashplayerbiostats")
    out = pd.DataFrame({
        "player_id": pd.to_numeric(raw["PLAYER_ID"]).astype("int64"),
        "bio_age": pd.to_numeric(raw["AGE"], errors="coerce"),
        "height_in": pd.to_numeric(raw["PLAYER_HEIGHT_INCHES"], errors="coerce") if "PLAYER_HEIGHT_INCHES" in raw else np.nan,
        "weight_lb": pd.to_numeric(raw["PLAYER_WEIGHT"], errors="coerce") if "PLAYER_WEIGHT" in raw else np.nan,
        "draft_year": _nullable_int(raw["DRAFT_YEAR"]) if "DRAFT_YEAR" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "draft_round": _nullable_int(raw["DRAFT_ROUND"]) if "DRAFT_ROUND" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
        "draft_number": _nullable_int(raw["DRAFT_NUMBER"]) if "DRAFT_NUMBER" in raw else pd.array([pd.NA] * len(raw), dtype="Int64"),
    })
    return out.drop_duplicates("player_id", keep="first").reset_index(drop=True)


def age_at_season_start(birthdate: pd.Series, season: str) -> pd.Series:
    """Exact age in years as of **October 1** of the season's start year.

    ``(Oct 1 - birthdate).days / 365.25``. The only error is the 365.25 approximation (< 0.01 y).
    """
    oct1 = pd.Timestamp(year=season_start(season), month=10, day=1)
    return (oct1 - pd.to_datetime(birthdate)).dt.days / 365.25


# Oct 1 -> Jun 30 of the following year is 272 days. AGE = floor(age at Jun 30), whose expectation
# is AGE + 0.5, so E[age at Oct 1] = AGE + 0.5 - 272/365.25 = AGE - 0.2447.
AGE_OCT1_OFFSET = 272 / 365.25 - 0.5

_PLAYER_ATTR_COLS = ["position", "height_in", "weight_lb", "draft_year", "draft_round", "draft_number", "from_year", "to_year"]


def build_players(
    game_logs: pd.DataFrame,
    player_index: pd.DataFrame | None = None,
    bio_stats: Mapping[str, pd.DataFrame] | None = None,
    birthdates: Mapping[int, pd.Timestamp | None] | None = None,
    extra_attrs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Static player table for every player in ``game_logs`` (validated).

    Attribute precedence per column, first non-null wins: ``player_index`` (PlayerIndex), then
    ``extra_attrs`` (CommonPlayerInfo rows), then the most recent season of ``bio_stats`` (LeagueDashPlayer
    BioStats; height/weight/draft only). ``player_name`` is the most recent name in ``game_logs``.
    ``birthdate`` comes only from ``birthdates`` (null when unknown; never guessed).
    """
    latest = (game_logs.sort_values(["game_date", "game_id"], kind="mergesort")
              .drop_duplicates("player_id", keep="last")[["player_id", "player_name"]])
    out = latest.set_index("player_id")
    idx = out.index

    def take(source: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
        if source is None or source.empty:
            return pd.DataFrame(index=idx, columns=cols)
        s = source.drop_duplicates("player_id", keep="last").set_index("player_id")
        return s.reindex(idx)[[c for c in cols if c in s.columns]].reindex(columns=cols)

    layers = [take(player_index, _PLAYER_ATTR_COLS), take(extra_attrs, _PLAYER_ATTR_COLS)]
    if bio_stats:
        latest_first = [bio_stats[s] for s in sorted(bio_stats, key=season_start, reverse=True)]
        for b in latest_first:
            layers.append(take(b, ["height_in", "weight_lb", "draft_year", "draft_round", "draft_number"])
                          .reindex(columns=_PLAYER_ATTR_COLS))
    merged = layers[0]
    for layer in layers[1:]:
        merged = merged.combine_first(layer) if not merged.empty else layer
    merged = merged.reindex(index=idx, columns=_PLAYER_ATTR_COLS)

    out["birthdate"] = pd.Series({pid: bd for pid, bd in (birthdates or {}).items() if bd is not None}).reindex(idx)
    out["birthdate"] = pd.to_datetime(out["birthdate"]).astype("datetime64[ns]")
    for c in _PLAYER_ATTR_COLS:
        out[c] = merged[c]
    out["position"] = out["position"].astype("object").where(out["position"].notna(), None)
    for c in ("height_in", "weight_lb"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ("draft_year", "draft_round", "draft_number", "from_year", "to_year"):
        out[c] = _nullable_int(out[c])
    out = out.reset_index()
    cols = list(TABLES["players"].columns)
    out = out[cols].sort_values("player_id").reset_index(drop=True)
    return validate_table(out, "players")


def build_player_season_bio(
    game_logs: pd.DataFrame,
    birthdates: Mapping[int, pd.Timestamp | None] | None = None,
    bio_stats: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, DropReport]:
    """One row per (season, player) present in ``game_logs`` (validated).

    * ``team_id`` = team of the player's **last game** of the season (latest ``game_date``, ties by
      ``game_id``) = the team he finished the regular season with, however many teams he played for.
    * ``age_at_season_start`` (as of Oct 1 of the start year), in order of preference:
        1. exact, from ``birthdates``;
        2. from the source's integer season ``AGE`` (LeagueDashPlayerBioStats). Verified against
           1,000+ real birthdates: ``AGE == floor(age on June 30 of the season's END year)`` (99.9%).
           Oct 1 is ``AGE_OCT1_OFFSET`` (0.245) years before the *expected* value of that
           quantity, so ``age_oct1 ~= AGE - 0.245`` with a worst-case error of +/- 0.5 year;
        3. shifted from that player's ``AGE`` in another season (``AGE + seasons_apart - 0.245``),
           same bound.
      A player-season with none of these is dropped and counted (``no_age_source``): the contract
      forbids a null age and we do not invent one.
    """
    report = DropReport()
    if game_logs.empty:
        empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                              [("season", "str"), ("player_id", "int64"), ("age_at_season_start", "float64"), ("team_id", "int64")]})
        return validate_table(empty, "player_season_bio"), report

    last = (game_logs.sort_values(["game_date", "game_id"], kind="mergesort")
            .drop_duplicates(["season", "player_id"], keep="last")[["season", "player_id", "team_id"]])
    bio = last.reset_index(drop=True)
    report.rows_in = len(bio)

    known = {int(pid): b for pid, b in (birthdates or {}).items() if b is not None and not pd.isna(b)}
    bio["birthdate"] = pd.to_datetime(pd.Series([known.get(int(p)) for p in bio["player_id"]], index=bio.index))
    age = pd.Series(np.nan, index=bio.index, dtype="float64")
    for season, grp in bio[bio["birthdate"].notna()].groupby("season"):
        age.loc[grp.index] = age_at_season_start(grp["birthdate"], season).to_numpy()

    if bio_stats:
        rows = []
        for s, b in bio_stats.items():
            t = b[["player_id", "bio_age"]].dropna()
            t = t.assign(source_season=s)
            rows.append(t)
        if rows:
            ages = pd.concat(rows, ignore_index=True)
            ages["src_start"] = ages["source_season"].map(season_start)
            # step 2: same season
            same = ages.set_index(["source_season", "player_id"])["bio_age"]
            key = pd.MultiIndex.from_frame(bio[["season", "player_id"]])
            direct = pd.Series(same.reindex(key).to_numpy(), index=bio.index) - AGE_OCT1_OFFSET
            need = age.isna()
            age[need] = direct[need]
            # step 3: nearest other season
            need = age.isna()
            if need.any():
                bio_start = bio["season"].map(season_start)
                ages_by_player = {pid: g for pid, g in ages.groupby("player_id")}
                for i in bio.index[need]:
                    g = ages_by_player.get(int(bio.at[i, "player_id"]))
                    if g is None:
                        continue
                    delta = (g["src_start"] - bio_start.at[i]).abs()
                    j = delta.idxmin()
                    age.at[i] = g.at[j, "bio_age"] + (bio_start.at[i] - g.at[j, "src_start"]) - AGE_OCT1_OFFSET

    bio["age_at_season_start"] = age
    n_missing = int(age.isna().sum())
    report.add("no_age_source", n_missing)
    bio = bio[age.notna()][["season", "player_id", "age_at_season_start", "team_id"]]
    bio = bio.sort_values(["season", "player_id"], key=lambda s: s.map(season_start) if s.name == "season" else s,
                          kind="mergesort").reset_index(drop=True)
    bio["team_id"] = bio["team_id"].astype("int64")
    report.rows_out = len(bio)
    return validate_table(bio, "player_season_bio"), report


def age_fallback_calibration(bio_frame: pd.DataFrame, season: str,
                             birthdates: Mapping[int, pd.Timestamp | None]) -> dict[str, Any]:
    """How good is the integer-``AGE`` fallback? Compare it with exact ages wherever both exist.

    ``bio_frame`` is one season of ``bio_stats_frame``. Returns the number of comparable pairs, the
    share where ``AGE == floor(exact age on June 30 of the season's end year)`` (the hypothesis the
    fallback rests on), and the error of the Oct-1 estimate ``AGE - AGE_OCT1_OFFSET`` against the
    exact Oct-1 age. Empty dict when nothing is comparable.
    """
    known = {int(k): v for k, v in birthdates.items() if v is not None and not pd.isna(v)}
    b = bio_frame.dropna(subset=["bio_age"]).copy()
    b["birth"] = pd.to_datetime(pd.Series([known.get(int(p)) for p in b["player_id"]], index=b.index))
    b = b[b["birth"].notna()]
    if b.empty:
        return {}
    end_year = season_start(season) + 1
    jun30 = pd.Timestamp(year=end_year, month=6, day=30)
    whole = (jun30.year - b["birth"].dt.year) - (((b["birth"].dt.month > 6) | ((b["birth"].dt.month == 6) & (b["birth"].dt.day > 30))).astype(int))
    err = (b["bio_age"] - AGE_OCT1_OFFSET) - age_at_season_start(b["birth"], season)
    return {
        "pairs": int(len(b)),
        "floor_age_on_jun30_matches": int((whole == b["bio_age"]).sum()),
        "mean_error": round(float(err.mean()), 4),
        "max_abs_error": round(float(err.abs().max()), 4),
        "within_half_year": int((err.abs() <= 0.5 + 0.005).sum()),
    }


# --------------------------------------------------------------------------- cross-table consistency

def check_consistency(game_logs: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, Any]:
    """Cross-table sanity between ``game_logs`` and ``team_games`` (pure; nothing raises).

    * ``orphan_player_rows``: player rows whose (game_id, team_id) has no ``team_games`` row. Any
      such row breaks the availability computation (games missed = team games - games played).
    * ``team_games_without_players``: team games with no player row at all.
    * ``team_points_mismatch``: team-games where the sum of player points != team points. A few are
      expected in real data (the NBA occasionally credits points to no one, e.g. team-level events).
    * ``team_minutes_off_grid``: team-games whose summed player minutes are not within 1.5 minutes of
      ``240 + 25 * overtimes``. Indicates missing player rows (e.g. a player logged 0:00).
    """
    tg_keys = team_games[["game_id", "team_id"]].drop_duplicates()
    gl_keys = game_logs[["game_id", "team_id"]].drop_duplicates()
    merged = gl_keys.merge(tg_keys, how="left", indicator=True)
    orphan_keys = merged[merged["_merge"] == "left_only"][["game_id", "team_id"]]
    orphan_rows = int(game_logs.merge(orphan_keys, on=["game_id", "team_id"]).shape[0]) if len(orphan_keys) else 0
    no_players = tg_keys.merge(gl_keys, how="left", indicator=True)
    no_players = no_players[no_players["_merge"] == "left_only"]

    agg = game_logs.groupby(["game_id", "team_id"]).agg(pts=("pts", "sum"), minutes=("min", "sum")).reset_index()
    both = agg.merge(team_games[["game_id", "team_id", "pts_for"]], on=["game_id", "team_id"])
    pts_bad = both[both["pts"] != both["pts_for"]]
    overtimes = ((both["minutes"] - 240) / 25).round().clip(lower=0)
    grid_bad = both[(both["minutes"] - (240 + 25 * overtimes)).abs() > 1.5]
    return {
        "orphan_player_rows": orphan_rows,
        "team_games_without_players": int(len(no_players)),
        "team_points_mismatch": int(len(pts_bad)),
        "team_points_mismatch_examples": pts_bad.head(5).to_dict("records"),
        "team_minutes_off_grid": int(len(grid_bad)),
        "team_minutes_off_grid_examples": grid_bad.head(5).to_dict("records"),
        "team_games_checked": int(len(both)),
    }
