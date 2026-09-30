"""NBA stats ingest: pull (cached, polite), transform, validate, and merge into the shared store.

    python -m src.ingest.nba_stats --seasons 2015-16:2025-26 [--offline] [--birthdates]

What it produces (contract tables, see docs/adr/0001 and docs/adr/0002-nba-ingest.md):
``game_logs``, ``team_games``, ``players``, ``player_season_bio``, written through
``store.write_table`` so every table is validated. Re-running with the same inputs changes nothing
(tables are rebuilt per season, merged deterministically, and a table is only rewritten when its
content actually differs).

Endpoints (all stats.nba.com, unofficial; see the ADR):

=========================== ==============================================================
``playergamelogs``          all players' games in a season, fractional minutes -> game_logs
``leaguegamelog`` (T)       every team game -> team_games
``leaguegamelog`` (P)       same player rows, integer minutes -> cross-check only
``leaguedashplayerbiostats`` per-season integer AGE, height, weight, draft -> age fallback
``playerindex``             position, height, weight, draft, from/to year -> players
``commonplayerinfo``        BIRTHDATE (one call per player; only with ``--birthdates``)
=========================== ==============================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.contracts import (
    ContractError, TABLES, data_dir, season_start, season_str, seasons_between, validate_table,
)
from src.ingest import nba_transform as tf
from src.ingest.nba_client import NBAClient, NBAClientError, atomic_write_bytes
from src.store import read_table, table_exists, write_table

MIN_SEASON_START = 1996  # playergamelogs has no data before 1996-97
REPORT_NAME = "nba_ingest_report.json"


class IngestError(RuntimeError):
    """The pulled data is unusable (empty season, broken referential integrity, ...)."""


# --------------------------------------------------------------------------- endpoint parameters
# Parameter dicts are kept explicit so the cache key is stable; tests compare them with the
# nba_api endpoint classes so a typo in a name/value cannot go unnoticed.

def gamelog_params(season: str, player_or_team: str) -> dict[str, Any]:
    """``leaguegamelog``: ``player_or_team`` is ``"P"`` (player rows) or ``"T"`` (team rows)."""
    if player_or_team not in ("P", "T"):
        raise ValueError("player_or_team must be 'P' or 'T'")
    season_start(season)
    return {"Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": player_or_team,
            "Season": season, "SeasonType": "Regular Season", "Sorter": "DATE", "DateFrom": "", "DateTo": ""}


def player_gamelogs_params(season: str) -> dict[str, Any]:
    season_start(season)
    return {"DateFrom": "", "DateTo": "", "GameSegment": "", "LastNGames": "", "LeagueID": "00", "Location": "",
            "MeasureType": "Base", "Month": "", "OpponentTeamID": "0", "Outcome": "", "PORound": "",
            "PerMode": "Totals", "Period": "", "PlayerID": "", "Season": season, "SeasonSegment": "",
            "SeasonType": "Regular Season", "ShotClockRange": "", "TeamID": "", "VsConference": "",
            "VsDivision": ""}


def bio_stats_params(season: str) -> dict[str, Any]:
    season_start(season)
    return {"LeagueID": "00", "PerMode": "Totals", "Season": season, "SeasonType": "Regular Season",
            "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "", "Division": "",
            "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "", "Height": "",
            "LastNGames": "", "Location": "", "Month": "", "OpponentTeamID": "", "Outcome": "", "PORound": "",
            "Period": "", "PlayerExperience": "", "PlayerPosition": "", "SeasonSegment": "",
            "ShotClockRange": "", "StarterBench": "", "TeamID": "", "VsConference": "", "VsDivision": "",
            "Weight": ""}


def player_index_params(season: str) -> dict[str, Any]:
    """``Historical=1`` returns every player ever; attributes do not depend on ``Season``."""
    season_start(season)
    return {"Active": "", "AllStar": "", "College": "", "Country": "", "DraftPick": "", "DraftYear": "",
            "Height": "", "PlayerPosition": "", "Historical": 1, "LeagueID": "00", "Season": season,
            "TeamID": "", "Weight": ""}


def common_player_info_params(player_id: int) -> dict[str, Any]:
    return {"PlayerID": int(player_id), "LeagueID": "00"}


# --------------------------------------------------------------------------- season parsing

_SEASON_TOKEN = re.compile(r"^\d{4}-\d{2}$")


def _one_season(token: str) -> str:
    token = token.strip()
    if not _SEASON_TOKEN.match(token):
        raise ValueError(f"malformed season {token!r}; expected like '2023-24'")
    start = season_start(token)  # also checks the two-digit end year
    if start < MIN_SEASON_START:
        raise ValueError(f"season {token} is before {season_str(MIN_SEASON_START)}, which the source does not cover")
    return token


def parse_seasons(spec: str) -> list[str]:
    """``"2015-16:2025-26"`` (inclusive range), ``"2023-24"``, or a comma list mixing both.

    Returns sorted unique season strings. Raises ValueError with a readable message on bad input.
    """
    if not spec or not spec.strip():
        raise ValueError("no seasons given; expected like '2015-16:2025-26'")
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in season list {spec!r}")
        if ":" in part:
            a, _, b = part.partition(":")
            first, last = season_start(_one_season(a)), season_start(_one_season(b))
            if last < first:
                raise ValueError(f"reversed season range {part!r}")
            out.update(seasons_between(first, last))
        else:
            out.add(_one_season(part))
    return sorted(out, key=season_start)


# --------------------------------------------------------------------------- merge helpers

def _sorted_by_season(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    order = df["season"].map(season_start)
    return (df.assign(_s=order).sort_values(["_s", *cols], kind="mergesort")
            .drop(columns="_s").reset_index(drop=True))


def merge_seasons(existing: pd.DataFrame | None, new: pd.DataFrame, seasons: list[str], sort_cols: list[str]) -> pd.DataFrame:
    """Replace ``seasons`` in ``existing`` with ``new`` and keep every other season untouched."""
    if existing is None or existing.empty:
        return _sorted_by_season(new, sort_cols)
    kept = existing[~existing["season"].isin(seasons)]
    frames = [f for f in (kept, new) if len(f)]
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    return _sorted_by_season(merged, sort_cols)


def conform_players(df: pd.DataFrame) -> pd.DataFrame:
    """Force the exact ``players`` dtypes (merging can widen nullable ints/datetimes)."""
    out = df.copy()
    out["player_id"] = out["player_id"].astype("int64")
    out["birthdate"] = pd.to_datetime(out["birthdate"]).astype("datetime64[ns]")
    for c in ("height_in", "weight_lb"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ("draft_year", "draft_round", "draft_number", "from_year", "to_year"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    out["position"] = out["position"].astype("object").where(out["position"].notna(), None)
    return out[list(TABLES["players"].columns)].sort_values("player_id").reset_index(drop=True)


def merge_players(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union by player_id. ``new`` wins where it has a value; a null in ``new`` never erases a value
    already stored (e.g. a birthdate fetched on an earlier ``--birthdates`` run)."""
    if existing is None or existing.empty:
        return conform_players(new)
    n, o = new.set_index("player_id"), existing.set_index("player_id")
    merged = n.combine_first(o)  # per-column, new wins where non-null (player_name included)
    return conform_players(merged.reset_index())


def frames_equal(a: pd.DataFrame | None, b: pd.DataFrame) -> bool:
    if a is None or list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        return False


# --------------------------------------------------------------------------- pipeline

@dataclass
class IngestResult:
    seasons: list[str]
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> {rows, changed, path}
    season_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    birthdates: dict[str, int] = field(default_factory=dict)          # this run: cache vs download counts
    birthdate_coverage: dict[str, int] = field(default_factory=dict)  # stored players table: known / unknown
    network_requests: int = 0
    cache_hits: int = 0


def _fetch_birthdates(
    client: NBAClient, ids: list[int], fetch_ids: set[int], fetch: bool, log: Callable[[str], None],
) -> tuple[dict[int, pd.Timestamp | None], list[dict[str, Any]], dict[str, int]]:
    """Birthdates + CommonPlayerInfo attributes. Cached responses are always used; new ones are only
    downloaded when ``fetch`` is true (and only for ``fetch_ids``)."""
    birth: dict[int, pd.Timestamp | None] = {}
    attrs: list[dict[str, Any]] = []
    stats = {"from_cache": 0, "downloaded": 0, "no_birthdate": 0, "not_fetched": 0}
    todo = [pid for pid in ids if pid in fetch_ids] if fetch else []
    n_fetch_target = sum(1 for pid in todo if client.peek("commonplayerinfo", common_player_info_params(pid)) is None)
    if n_fetch_target:
        log(f"  fetching CommonPlayerInfo for {n_fetch_target} players (~{n_fetch_target * (client.min_interval + 0.5) / 60:.0f} min); resumable")
    done = 0
    for pid in ids:
        params = common_player_info_params(pid)
        payload = client.peek("commonplayerinfo", params)
        if payload is not None:
            stats["from_cache"] += 1
        elif fetch and pid in fetch_ids:
            payload = client.get("commonplayerinfo", params)
            stats["downloaded"] += 1
            done += 1
            if done % 100 == 0:
                log(f"    commonplayerinfo {done}/{n_fetch_target}")
        else:
            stats["not_fetched"] += 1
            continue
        try:
            row = tf.player_attributes_from_common_player_info(payload)
        except tf.TransformError:
            # Some players have an empty CommonPlayerInfo; that just means "no birthdate known".
            stats["no_birthdate"] += 1
            continue
        birth[pid] = row["birthdate"]
        if row["birthdate"] is None:
            stats["no_birthdate"] += 1
        attrs.append(row)
    return birth, attrs, stats


def run_ingest(
    seasons: list[str],
    client: NBAClient,
    base: Path | None = None,
    *,
    birthdates: bool = False,
    refresh: bool = False,
    crosscheck: bool = True,
    log: Callable[[str], None] = print,
) -> IngestResult:
    """Pull, transform, validate and merge ``seasons`` into the store under ``base``."""
    if not seasons:
        raise IngestError("no seasons to ingest")
    base = base or data_dir()
    result = IngestResult(seasons=list(seasons))

    new_logs, new_teams, bio_frames = [], [], {}
    for season in seasons:
        log(f"[{season}] pulling")
        gl_payload = client.get("playergamelogs", player_gamelogs_params(season), refresh=refresh)
        tm_payload = client.get("leaguegamelog", gamelog_params(season, "T"), refresh=refresh)
        bio_payload = client.get("leaguedashplayerbiostats", bio_stats_params(season), refresh=refresh)

        gl, gl_rep = tf.transform_game_logs(gl_payload, season)
        tg, tg_rep = tf.transform_team_games(tm_payload, season)
        if gl.empty or tg.empty:
            raise IngestError(f"{season}: the source returned no regular-season games (season not played yet?)")
        cons = tf.check_consistency(gl, tg)
        if cons["orphan_player_rows"]:
            raise IngestError(
                f"{season}: {cons['orphan_player_rows']} player rows reference a (game, team) missing from team_games; "
                "refusing to write a table that would break availability calculations")
        entry: dict[str, Any] = {
            "game_logs": gl_rep.as_dict(), "team_games": tg_rep.as_dict(), "consistency": cons,
            "n_games": int(tg["game_id"].nunique()),
        }
        if crosscheck:
            p_payload = client.get("leaguegamelog", gamelog_params(season, "P"), refresh=refresh)
            other, _ = tf.transform_game_logs(p_payload, season)
            entry["crosscheck"] = tf.crosscheck_game_logs(gl, other)
        result.season_reports[season] = entry
        new_logs.append(gl)
        new_teams.append(tg)
        bio_frames[season] = tf.bio_stats_frame(bio_payload)
        log(f"[{season}] {len(gl):,} player-games, {tg['game_id'].nunique():,} games, "
            f"{gl['player_id'].nunique()} players; dropped {gl_rep.dropped or 'nothing'}")

    game_logs = merge_seasons(_existing("game_logs", base), pd.concat(new_logs, ignore_index=True), seasons,
                              ["game_date", "game_id", "team_id", "player_id"])
    team_games = merge_seasons(_existing("team_games", base), pd.concat(new_teams, ignore_index=True), seasons,
                               ["game_date", "game_id", "team_id"])
    validate_table(game_logs, "game_logs")
    validate_table(team_games, "team_games")

    # ---- static player attributes and birthdates
    run_logs = game_logs[game_logs["season"].isin(seasons)]
    all_ids = sorted(game_logs["player_id"].unique().tolist())
    fetch_ids = set(run_logs["player_id"].unique().tolist())
    index_payload = client.get("playerindex", player_index_params(seasons[-1]), refresh=refresh)
    player_index = tf.player_index_frame(index_payload)
    birth, cpi_rows, bstats = _fetch_birthdates(client, all_ids, fetch_ids, birthdates, log)
    result.birthdates = bstats
    existing_players = _existing("players", base)
    if existing_players is not None:  # birthdates learned on an earlier run are never forgotten
        for pid, bd in zip(existing_players["player_id"], existing_players["birthdate"]):
            if pd.notna(bd) and birth.get(int(pid)) is None:
                birth[int(pid)] = bd
    extra = pd.DataFrame(cpi_rows) if cpi_rows else None
    players_new = tf.build_players(game_logs, player_index, bio_frames, birth, extra)
    players = merge_players(existing_players, players_new)
    validate_table(players, "players")
    result.birthdate_coverage = {"players": len(players), "with_birthdate": int(players["birthdate"].notna().sum()),
                                 "without_birthdate": int(players["birthdate"].isna().sum())}

    # ---- per-season bio (rebuilt for the requested seasons only)
    run_birth = {pid: bd for pid, bd in birth.items() if pid in fetch_ids}
    bio_new, bio_rep = tf.build_player_season_bio(run_logs, run_birth, bio_frames)
    for season in seasons:  # per-season bio drop counts, for the report
        _, one = tf.build_player_season_bio(run_logs[run_logs["season"] == season], run_birth, bio_frames)
        result.season_reports[season]["player_season_bio"] = one.as_dict()
        result.season_reports[season]["age_fallback_calibration"] = tf.age_fallback_calibration(
            bio_frames[season], season, run_birth)
    bio = merge_seasons(_existing("player_season_bio", base), bio_new, seasons, ["player_id"])
    validate_table(bio, "player_season_bio")

    for name, df in (("game_logs", game_logs), ("team_games", team_games), ("players", players), ("player_season_bio", bio)):
        old = _existing(name, base)
        changed = not frames_equal(old, df)
        path = write_table(df, name, base) if changed else _path(name, base)
        result.tables[name] = {"rows": len(df), "changed": changed, "path": str(path)}
        log(f"  {name}: {len(df):,} rows ({'written' if changed else 'unchanged'})")

    _write_report(base, seasons, result)
    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    return result


def _path(name: str, base: Path) -> Path:
    from src.contracts import table_path
    return table_path(name, base)


def _existing(name: str, base: Path) -> pd.DataFrame | None:
    return read_table(name, base, validate=False) if table_exists(name, base) else None


def report_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / REPORT_NAME


def read_report(base: Path | None = None) -> dict[str, Any]:
    p = report_path(base)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"seasons": {}}


def _write_report(base: Path, seasons: list[str], result: IngestResult) -> None:
    """Per-season drop/consistency/cross-check numbers, merged by season and deterministic
    (no timestamps) so re-runs leave the file byte-identical."""
    report = read_report(base)
    for s in seasons:
        entry = json.loads(json.dumps(result.season_reports[s], default=str))
        previous = report.setdefault("seasons", {}).get(s, {})
        if "crosscheck" not in entry and "crosscheck" in previous:  # e.g. --no-crosscheck: keep the last result
            entry["crosscheck"] = previous["crosscheck"]
        report["seasons"][s] = entry
    report["birthdates"] = result.birthdate_coverage  # table-level, so identical on a re-run
    report["seasons"] = dict(sorted(report["seasons"].items(), key=lambda kv: season_start(kv[0])))
    atomic_write_bytes(report_path(base), (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.nba_stats",
        description="Ingest NBA regular-season data from stats.nba.com (unofficial) into the shared data store.",
    )
    p.add_argument("--seasons", required=True, help="e.g. 2015-16:2025-26, 2023-24, or 2015-16:2017-18,2020-21")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network; fail with a clear error if a response is not cached "
                        "(also enabled by NBA_OFFLINE=1)")
    p.add_argument("--birthdates", action="store_true",
                   help="download CommonPlayerInfo per player for exact birthdates (~1 request/player, resumable). "
                        "Without it, cached birthdates are still used and ages fall back to the integer season AGE.")
    p.add_argument("--refresh", action="store_true",
                   help="re-download season-level responses even if cached (for an in-progress season)")
    p.add_argument("--no-crosscheck", action="store_true", help="skip the integer-minute leaguegamelog cross-check pull")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override the data root (default: NBA_DATA_DIR or ~/dev-data/nba-fantasy-2026)")
    p.add_argument("--min-interval", type=float, default=1.0, help="minimum seconds between network requests (default 1.0)")
    p.add_argument("--max-retries", type=int, default=6)
    return p


def main(argv: list[str] | None = None, *, client: NBAClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2
    if args.min_interval < 0.8 and not args.offline:
        print(f"warning: --min-interval {args.min_interval} is below the 0.8 s politeness floor; using 0.8", file=sys.stderr)
        args.min_interval = 0.8
    base = args.data_dir or data_dir()
    if client is None:
        cache = base / "raw" / "nba_api"
        client = NBAClient(cache, offline=True if args.offline else None,
                           min_interval=args.min_interval, max_retries=args.max_retries)
    try:
        result = run_ingest(seasons, client, base, birthdates=args.birthdates, refresh=args.refresh,
                            crosscheck=not args.no_crosscheck)
    except (NBAClientError, IngestError, tf.TransformError, ContractError) as exc:
        print(f"ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {len(seasons)} seasons, {result.network_requests} network requests, {result.cache_hits} cache hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
