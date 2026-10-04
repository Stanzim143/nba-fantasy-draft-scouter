"""Shared loading for the in-season tools: tables, ROS frame, schedule, matchup calendar, league rosters.

Everything is read from the local data directory (nothing here touches the network); ESPN data comes
from the caches the ingest modules already wrote (``src.ingest.espn_league``, ``src.ingest.espn_adp``,
``src.inseason.schedule --ingest``). A missing optional input never blocks a tool, it produces a
``notes`` line saying what is missing and what the tool did instead.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import HISTORY_TABLES, ContractError, data_dir, season_start
from src.inseason.ros import RosParams, build_ros
from src.inseason.schedule import (
    ScheduleError, build_calendar, from_team_games, read_schedule, weekly_team_table,
)
from src.inseason.weeks import MatchupCalendar
from src.value.league import load_league


class ContextUnavailable(Exception):
    """A tool cannot run for these inputs; the message is safe to show to a user."""


@dataclass
class InSeasonContext:
    season: str
    as_of: pd.Timestamp
    cfg: dict
    tables: dict
    ros: pd.DataFrame
    schedule: pd.DataFrame | None = None
    calendar: MatchupCalendar | None = None
    weekly: pd.DataFrame | None = None
    league: dict | None = None
    id_map: dict[str, int] = field(default_factory=dict)   # ESPN player id (str) -> NBA player_id
    injuries: pd.Series | None = None                       # player_id -> ESPN injury status
    notes: list[str] = field(default_factory=list)
    data_root: Path | None = None
    week_anchor: pd.Timestamp | None = None                 # day that picks the "current" matchup week (default: as_of)


def _int_env(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


def load_tables(season: str, data_root: Path | None, synthetic: bool) -> dict:
    if synthetic:
        from src.synthetic import make_synthetic_tables

        s = season_start(season)
        return dict(make_synthetic_tables(first_start=s - 4, last_start=s, n_teams=14, games_per_team=60, seed=0))
    from src.store import load_tables as _load

    try:
        return dict(_load(HISTORY_TABLES, base=data_root))
    except FileNotFoundError as exc:
        raise ContextUnavailable(f"real data isn't available: {exc}. Ingest it with `python -m src.ingest.nba_stats`, "
                                 "or use --synthetic for a demo.") from exc
    except ContractError as exc:
        raise ContextUnavailable(f"stored data failed validation: {exc}") from exc


def _schedule_for(tables: dict, season: str, data_root: Path | None, notes: list[str], *,
                  use_store: bool = True, as_of: pd.Timestamp | None = None) -> pd.DataFrame | None:
    if use_store:
        try:
            sch = read_schedule(data_root, season)
            if len(sch):
                return sch
        except (FileNotFoundError, ContractError):
            pass
    tg = tables["team_games"]
    tgs = tg[tg["season"] == season]
    # Judged against ``as_of`` (never the wall clock, so a replay is reproducible): complete when the table already holds
    # games after ``as_of`` (a replay of a finished season) or its last game is over a month before ``as_of``.
    last = pd.Timestamp(tgs["game_date"].max()) if len(tgs) else None
    ref = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    finished = last is not None and (last > ref or last < ref - pd.Timedelta(days=30))
    if finished:  # a finished season: its realised team_games are its schedule; a live one's are only the games so far
        try:
            notes.append(f"no stored schedule_games for {season}; using the realised team_games schedule "
                         "(python -m src.inseason.schedule --ingest stores the ESPN one)")
            return from_team_games(tg, season)
        except ScheduleError:
            return None
    notes.append(f"no schedule for {season} (run `python -m src.inseason.schedule --season {season} --ingest`); "
                 "games remaining fall back to season length minus games played and weekly tools are unavailable")
    return None


def _teams_series(tables: dict, season: str, as_of: pd.Timestamp, data_root: Path | None, *,
                  use_store: bool = True) -> pd.Series | None:
    """player_id -> team_id for players with no game yet: latest roster snapshot on or before ``as_of``."""
    try:
        if not use_store:
            raise FileNotFoundError
        from src.ingest.nba_incoming import read_roster_snapshots

        snap = read_roster_snapshots(data_root)
        snap = snap[snap["snapshot_date"] <= as_of]
        if len(snap):
            last = snap[snap["snapshot_date"] == snap["snapshot_date"].max()]
            return last.drop_duplicates("player_id").set_index("player_id")["team_id"]
    except (FileNotFoundError, KeyError, ValueError):
        pass
    bio = tables["player_season_bio"]
    if len(bio):
        b = bio[bio["team_id"].notna()].sort_values("season")
        return b.drop_duplicates("player_id", keep="last").set_index("player_id")["team_id"]
    return None


def load_league_json(league_id: int | None, season: str, data_root: Path | None) -> dict | None:
    if league_id is None:
        return None
    path = (data_root or data_dir()) / "processed" / "espn_league" / f"{league_id}_{season_start(season) + 1}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_id_map(data_root: Path | None) -> dict[str, int]:
    try:
        from src.store import read_table

        m = read_table("player_id_map", data_root)
    except (FileNotFoundError, ContractError):
        return {}
    m = m[m["source"] == "espn"]
    return {str(s): int(p) for s, p in zip(m["source_id"], m["player_id"])}


def load_injuries(season: str, data_root: Path | None, id_map: dict[str, int]) -> pd.Series | None:
    """ESPN injury status per player from the cached player universe (no network); ``None`` if not cached."""
    from src.ingest.http_cache import CachedHttpClient, HttpCacheError
    from src.contracts import raw_dir

    sid = season_start(season) + 1
    client = CachedHttpClient((data_root / "raw" / "espn") if data_root else raw_dir("espn"), offline=True,
                              write_fetch_log=False)
    try:
        payload = client.get_json(f"players/{sid}", f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{sid}/players",
                                  {"view": "kona_player_info", "limit": 600})
    except HttpCacheError:
        return None
    out = {}
    for p in payload if isinstance(payload, list) else []:
        pid = id_map.get(str(p.get("id")))
        st = p.get("injuryStatus")
        if pid is not None and st and st != "ACTIVE":
            out[pid] = st
    return pd.Series(out, dtype="object", name="injury_status")


def mapped_teams(league: dict | None, id_map: dict[str, int]) -> tuple[dict[int, dict], int]:
    """League teams with rosters mapped to NBA ``player_id``: ``{team_id: {name, players: [(pid, slot)]}}``.

    Returns the mapping and the number of rostered players that could not be mapped (reported, never guessed).
    """
    teams: dict[int, dict] = {}
    unmapped = 0
    for t in (league or {}).get("teams", []) or []:
        players = []
        for e in t.get("roster", []) or []:
            pid = id_map.get(str(e.get("espn_player_id")))
            if pid is None:
                unmapped += 1
                continue
            players.append((pid, e.get("lineup_slot")))
        teams[int(t["team_id"])] = {"name": t.get("name"), "players": players}
    return teams, unmapped


def load_context(season: str | None = None, as_of=None, *, model: str = "baseline", mode: str = "blend",
                 data_dir: Path | None = None, synthetic: bool = False, league_id: int | None = None,
                 use_injuries: bool | None = None, params: RosParams | None = None,
                 tables: dict | None = None, prior: pd.DataFrame | None = None,
                 cfg: dict | None = None) -> InSeasonContext:
    """Everything the tools need for ``season`` as of ``as_of`` (default today, inclusive)."""
    cfg = cfg if cfg is not None else load_league()
    season = season or cfg["league"]["season"]
    try:
        season_start(season)
    except ValueError as exc:
        raise ContextUnavailable(str(exc)) from exc
    as_of_ts = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today().normalize())
    notes: list[str] = []
    tables = tables if tables is not None else load_tables(season, data_dir, synthetic)
    schedule = _schedule_for(tables, season, data_dir, notes, use_store=not synthetic, as_of=as_of_ts)
    teams = _teams_series(tables, season, as_of_ts, data_dir, use_store=not synthetic)
    try:
        ros = build_ros(tables, season, as_of_ts, prior=prior, model=model, cfg=cfg, schedule=schedule,
                        params=RosParams(mode=mode) if params is None else params, teams=teams)
    except (KeyError, ValueError) as exc:
        raise ContextUnavailable(f"could not build the rest-of-season projection: {exc}") from exc
    if int((ros["gp"] > 0).sum()) == 0:
        notes.append(f"no {season} games on or before {as_of_ts.date()}: this is the preseason projection "
                     "with all games remaining")

    league_id = league_id if league_id is not None else _int_env("ESPN_LEAGUE_ID")
    league = None if synthetic else load_league_json(league_id, season, data_dir)
    id_map = {} if synthetic else load_id_map(data_dir)
    if use_injuries is None:
        use_injuries = (not synthetic) and as_of_ts >= pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    injuries = load_injuries(season, data_dir, id_map) if use_injuries and id_map else None

    calendar = weekly = None
    if schedule is not None and len(schedule):
        calendar = build_calendar(schedule, season, league_id=None if synthetic else league_id, cfg=cfg,
                                  data_root=data_dir)
        weekly = weekly_team_table(schedule, calendar, as_of_ts)
        if calendar.source == "derived":
            notes.append("matchup weeks are derived (Monday-Sunday, All-Star week merged), not read from ESPN's "
                         "matchupPeriods (unpublished or not synced for this league/season)")
    return InSeasonContext(season=season, as_of=as_of_ts, cfg=cfg, tables=tables, ros=ros, schedule=schedule,
                           calendar=calendar, weekly=weekly, league=league, id_map=id_map, injuries=injuries,
                           notes=notes, data_root=data_dir)


def rostered_ids(ctx: InSeasonContext) -> set[int]:
    teams, _ = mapped_teams(ctx.league, ctx.id_map)
    return {pid for t in teams.values() for pid, _ in t["players"]}


def active_universe(ctx: InSeasonContext) -> set[int]:
    """Players who can plausibly be on a roster now: played this season up to ``as_of`` or on a current NBA
    roster (latest snapshot on or before ``as_of``). Keeps retired players in the projection tail out of the
    free-agent pool.

    The roster-snapshot ids are intersected with ``ros``'s own player universe before being unioned in: the
    snapshot file is real-NBA-player data shared across every worktree (``NBA_DATA_DIR``), so when ``ros`` is
    a synthetic demo frame (``--synthetic``/"Use synthetic demo data") its fake ids never overlap the real
    snapshot's ids at all. Without the intersection, a real snapshot file just sitting in the shared data dir
    (put there by ordinary real-data use elsewhere) would silently make this return only those non-existent
    real ids, leaving the synthetic universe empty -- and everything downstream that filters through it (the
    free-agent pool, ``mock_snake_draft``'s player pool) empty too, with no error. Intersecting first means a
    snapshot that shares no id with ``ros`` contributes nothing, so the ``not ids`` fallback below still
    reliably catches it and returns the full synthetic universe instead of an empty one."""
    ros = ctx.ros
    all_ids = set(ros["player_id"])
    ids = set(ros.loc[ros["gp"] > 0, "player_id"])
    try:
        from src.ingest.nba_incoming import read_roster_snapshots

        snap = read_roster_snapshots(ctx.data_root)
        snap = snap[snap["snapshot_date"] <= ctx.as_of]
        if len(snap):
            ids |= set(snap[snap["snapshot_date"] == snap["snapshot_date"].max()]["player_id"]) & all_ids
    except (FileNotFoundError, KeyError, ValueError):
        pass
    if not ids:
        ids = all_ids
    return ids


def mock_snake_draft(order_values: pd.Series, n_teams: int, roster_size: int) -> dict[int, list[int]]:
    """A deterministic snake draft by descending value: ``{team_index: [player_id, ...]}``.

    Used by tests and the real-data smoke run to give every team a plausible roster when no real league
    rosters exist (before the draft, or for a past season). Never presented as anyone's real roster.
    """
    ids = list(order_values.sort_values(ascending=False, kind="mergesort").index)
    teams: dict[int, list[int]] = {t: [] for t in range(n_teams)}
    i = 0
    for rnd in range(roster_size):
        seq = range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1)
        for t in seq:
            if i < len(ids):
                teams[t].append(int(ids[i]))
                i += 1
    return teams


def ensure_finite(x) -> float:
    return float(x) if np.isfinite(x) else 0.0


@dataclass
class RosterChoice:
    """Whose roster a tool is working on and who else is rostered (so the free-agent pool is known)."""
    my_ids: list[int]
    ir_ids: list[int]
    rostered: set[int]
    label: str
    others: dict[str, list[int]] = field(default_factory=dict)   # team label -> roster (for trade partners)


def resolve_roster(ctx: InSeasonContext, *, team_id: int | None = None, roster_names: str | None = None,
                   mock_team: int | None = None) -> RosterChoice:
    """Pick the roster to analyse: a real ESPN team (``team_id`` or ``$ESPN_TEAM_ID``), explicit player names,
    or team ``mock_team`` (1-based) of a deterministic mock snake draft on the ROS board (demos, tests and
    the real-data smoke run: before the draft and for past seasons there is no real roster to read)."""
    from src.inseason.lineup import resolve_names
    from src.value.replacement import league_shape

    ros = ctx.ros
    uni = active_universe(ctx)
    teams, unmapped = mapped_teams(ctx.league, ctx.id_map)
    real = {t: v for t, v in teams.items() if v["players"]}
    if roster_names:
        mine = resolve_names(ros, roster_names)
        others = {f"{v['name']} (ESPN team {t})": [p for p, _ in v["players"]] for t, v in real.items()}
        rostered = set(mine) | {p for ps in others.values() for p in ps}
        return RosterChoice(mine, [], rostered, "the roster you listed", others)
    if mock_team is not None:
        shape = league_shape(ctx.cfg)
        pool = ros[ros["player_id"].isin(uni)].set_index("player_id")["ros_total_fp"]
        mock = mock_snake_draft(pool, shape.teams, shape.roster_size)
        if not 1 <= mock_team <= shape.teams:
            raise ContextUnavailable(f"--mock-draft-team must be 1..{shape.teams}")
        ctx.notes.append("rosters are a MOCK snake draft on the rest-of-season board, not real league rosters")
        others = {f"mock team {t + 1}": ids for t, ids in mock.items() if t + 1 != mock_team}
        return RosterChoice(mock[mock_team - 1], [], {p for ids in mock.values() for p in ids},
                            f"mock team {mock_team}", others)
    if team_id is None:
        team_id = _int_env("ESPN_TEAM_ID")
    if team_id is None or team_id not in teams:
        avail = ", ".join(f"{t}={v['name']}" for t, v in teams.items()) or "none (run src.ingest.espn_league first)"
        raise ContextUnavailable("say whose roster to use: --team-id N (or $ESPN_TEAM_ID; league teams: "
                                 f"{avail}), --my-roster 'Name, Name, ...', or --mock-draft-team N for a demo")
    mine = teams[team_id]
    if not mine["players"]:
        raise ContextUnavailable(f"ESPN team {team_id} ({mine['name']}) has no players yet (the draft has not "
                                 "happened). Use --my-roster or --mock-draft-team to try the tool.")
    if unmapped:
        ctx.notes.append(f"{unmapped} rostered players could not be mapped to NBA ids (player_id_map) and are ignored")
    ids = [p for p, _ in mine["players"]]
    ir = [p for p, slot in mine["players"] if slot == "IR"]
    others = {f"{v['name']} (ESPN team {t})": [p for p, _ in v["players"]] for t, v in real.items() if t != team_id}
    return RosterChoice(ids, ir, {p for v in real.values() for p, _ in v["players"]}, f"{mine['name']} (ESPN team {team_id})", others)


def free_agent_ids(ctx: InSeasonContext, rostered: set[int]) -> list[int]:
    """Players in the ROS projection who can be picked up: active in the NBA and on nobody's roster."""
    uni = active_universe(ctx)
    return [int(p) for p in ctx.ros["player_id"] if p in uni and p not in rostered]
