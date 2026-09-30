"""Unattended in-season nightly job (ADR 0018; extends the pre-draft automation of ADR 0014).

    python -m src.ops.nightly [--season 2026-27] [--force] [--as-of 2026-12-10] [--offline] [--dry-run]

Runs once a day, after the previous US night's games are final. In order:

1. ``games``     incremental, idempotent pull of the live season's game logs and team games (``src.ops.live_ingest``);
2. ``roster``    today's NBA roster snapshot (``nba_incoming``);
3. ``adp``       a fresh ESPN player universe and the player id map (needed for injuries and league rosters);
4. ``status``    the dated ESPN injury-status archive;
5. ``snapshot``  a dated compact archive of the ESPN payload (point-in-time injuries, ADR 0005);
6. ``league``    read-only ESPN league sync: rosters, free agents, transactions, the real matchup periods once published;
7. ``schedule``  refresh the ``schedule_games`` table (one ESPN request), reporting moved and added games;
8. ``analysis``  rest-of-season projection as of the last completed game day and the artifacts of ADR 0018: waiver and
   streaming recommendations, rising-minutes and injury-beneficiary alerts, this/next week's games per team, start/sit hints.

Then it diffs against the previous night and writes ``reports/nightly/<date>.md``, ``latest.md``, ``latest.json`` and CSVs under
``reports/nightly/data/``. Same guarantees as ADR 0014: single-instance lock (shared with the daily refresh), one subprocess and
one timeout per step, a failing step never stops the others, structured ``status.json``, rotating log, ``ALERT.txt`` plus a
Windows toast on consecutive failures or stale data. **Season-window gate**: active from opening night through the fantasy
final day plus two days of grace; outside it the command exits 0 having done nothing (``--force`` overrides).

Exit codes: 0 ok (or outside the window), 1 a step failed or was degraded, 2 configuration error, 3 another run holds the lock.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.ops import daily_refresh as dr
from src.ops import refresh_diff as rd
from src.ops.nightly_analysis import completed_through
from src.ops.nightly_report import render_report
from src.ops.runlock import LockBusy, RunLock

LOGGER = logging.getLogger("nightly")
REPO_ROOT = dr.REPO_ROOT
STEP_ORDER = ("games", "roster", "adp", "status", "snapshot", "league", "schedule", "analysis")
DEFAULT_TIMEOUTS = {"games": 900, "roster": 300, "adp": 300, "status": 60, "snapshot": 60, "league": 180, "schedule": 120, "analysis": 900}
OPENING_NIGHTS = {"2026-27": date(2026, 10, 20)}   # used only until the schedule table can say (user, 2026-09-25)
FALLBACK_FANTASY_DAYS = 167                        # the league's finalScoringPeriod (one scoring period per day)
GRACE_DAYS = 2                                     # the last games of the final day are ingested by a run after it
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_BUSY = dr.EXIT_OK, dr.EXIT_FAILED, dr.EXIT_CONFIG, dr.EXIT_BUSY
SETTINGS_FILE = "nightly.json"
BAD = ("failed", "timeout", "degraded")
StepOutcome = dr.StepOutcome
ConfigError = dr.ConfigError


# --------------------------------------------------------------------------- settings and the season window

@dataclass
class Settings:
    season: str
    data_dir: Path
    reports_dir: Path
    repo_root: Path
    window_start: date
    window_end: date
    window_source: str
    league_id: int | None = None
    team_id: int | None = None
    mock_team: int | None = None
    as_of: date | None = None
    offline: bool = False
    budget_s: float = 60 * 60
    lock_wait_s: float = 0.0                       # how long to wait for a busy shared lock (the daily job) before skipping
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    stale_hours: float = 36.0
    alert_after: int = 2
    top: int = 15
    steps: tuple[str, ...] = STEP_ORDER

    @property
    def ops_dir(self) -> Path:
        return self.data_dir / "nightly"

    @property
    def lock_path(self) -> Path:                   # shared with the daily refresh: they touch the same tables and caches
        return self.data_dir / "daily_refresh" / "lock.json"


def _read_settings_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def season_window(season: str, data_dir: Path, league_id: int | None, *, start_override: date | None = None,
                  end_override: date | None = None) -> tuple[date, date, str]:
    """``(first day, last day, how it was derived)`` scheduled runs do anything.

    Start: the first game in the ``schedule_games`` table, else the known opening night, else October 20 of the start year.
    End: the last day of the league's final matchup period (ESPN's real periods when published, else the derived calendar
    that reproduces ``finalScoringPeriod``), else the last scheduled game, else 167 days after opening night; plus two days
    of grace. Missing schedule, league or config never raises: the fallback is used and named.
    """
    from src.contracts import season_start

    sched = None
    try:
        from src.inseason.schedule import read_schedule

        sched = read_schedule(data_dir, season)
        if sched is not None and sched.empty:
            sched = None
    except Exception:                                    # noqa: BLE001 - no table yet is the normal pre-season state
        sched = None
    how = []
    if start_override:
        start = start_override
        how.append("start: flag")
    elif sched is not None:
        start = sched["game_date"].min().date()
        how.append("start: first game in schedule_games")
    else:
        start = OPENING_NIGHTS.get(season) or date(season_start(season), 10, 20)
        how.append("start: opening-night constant" if season in OPENING_NIGHTS else "start: October 20 fallback")
    if end_override:
        return start, end_override, "; ".join(how + ["end: flag"])
    end = None
    if sched is not None:
        try:
            from src.inseason.schedule import build_calendar
            from src.value.league import load_league

            end = build_calendar(sched, season, league_id=league_id, cfg=load_league(), data_root=data_dir).last_day
            how.append("end: league final matchup period")
        except Exception:                                # noqa: BLE001
            end = sched["game_date"].max().date()
            how.append("end: last scheduled game (no league calendar)")
    if end is None:
        end = start + timedelta(days=FALLBACK_FANTASY_DAYS - 1)
        how.append(f"end: {FALLBACK_FANTASY_DAYS} days after the start (no schedule)")
    return start, end + timedelta(days=GRACE_DAYS), "; ".join(how)


def resolve_settings(args: argparse.Namespace, env: Mapping[str, str], now: datetime) -> Settings:
    """Flag > environment > ``<data>/nightly.json`` > ``<data>/daily_refresh.json`` (league id, team id) > default."""
    from src.contracts import data_dir as default_data_dir
    from src.contracts import season_start
    from src.ingest.nba_offseason import live_season_start

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    local = {**_read_settings_file(data_dir / dr.SETTINGS_FILE), **_read_settings_file(data_dir / SETTINGS_FILE)}
    from src.contracts import season_str

    season = args.season or local.get("season") or season_str(live_season_start(now.date()))
    try:
        season_start(season)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    def pick(flag, env_key, file_key):
        if flag not in (None, ""):
            return flag
        if env.get(env_key):
            return env[env_key]
        return local.get(file_key)

    def as_int(v, what):
        try:
            return int(v) if v not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{what} {v!r} is not an integer") from exc

    def as_day(v, what):
        try:
            return date.fromisoformat(str(v).strip()) if v not in (None, "") else None
        except ValueError as exc:
            raise ConfigError(f"{what}: {v!r} is not a date like 2026-10-20") from exc

    repo_root = Path(args.repo_root) if getattr(args, "repo_root", None) else REPO_ROOT
    league_id = as_int(pick(args.league_id, "ESPN_LEAGUE_ID", "league_id"), "league id") or dr._league_id_from_dotenv(repo_root)
    team_id = as_int(pick(args.team_id, "ESPN_TEAM_ID", "team_id"), "team id")
    start_o = as_day(pick(args.window_start or args.season_start, "NBA_WINDOW_START", "window_start"), "window start")
    end_o = as_day(pick(args.window_end, "NBA_WINDOW_END", "window_end"), "window end")
    try:
        start, end, source = season_window(season, data_dir, league_id, start_override=start_o, end_override=end_o)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if end < start:
        raise ConfigError(f"window end {end} is before window start {start}")
    steps = tuple(s.strip() for s in (args.only or ",".join(STEP_ORDER)).split(",") if s.strip())
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    unknown = [s for s in (*steps, *skip) if s not in STEP_ORDER]
    if unknown:
        raise ConfigError(f"unknown step(s) {unknown}; expected some of {list(STEP_ORDER)}")
    steps = tuple(s for s in STEP_ORDER if s in steps and s not in skip)
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo_root / "reports" / "nightly"
    return Settings(
        season=season, data_dir=data_dir, reports_dir=reports_dir, repo_root=repo_root, window_start=start, window_end=end,
        window_source=source, league_id=league_id, team_id=team_id, mock_team=args.mock_team,
        as_of=as_day(args.as_of, "--as-of"), offline=bool(args.offline), budget_s=args.budget_minutes * 60.0, lock_wait_s=args.lock_wait_minutes * 60.0,
        stale_hours=args.stale_hours, alert_after=args.alert_after, top=args.top, steps=steps)


def in_window(day: date, s: Settings) -> bool:
    return s.window_start <= day <= s.window_end


def run_days(s: Settings, now: datetime) -> tuple[date, date, date]:
    """``(gate day, completed_through, slate)``: the last fully final US game day and the next one to be played.

    ``--as-of D`` means "pretend D is the last completed day" (a replay); the gate then judges the slate day, not today.
    """
    through = s.as_of or completed_through(now)
    slate = through + timedelta(days=1)
    return (slate if s.as_of else now.date()), through, slate


# --------------------------------------------------------------------------- subprocess runner

class SubprocessRunner:
    """Runs ``snapshot`` in-process (a cache read) and every other step as ``python -m src.ops.nightly --worker``."""

    def __init__(self, s: Settings, through: date, slate: date, *, process_runner: Callable = dr.run_worker_process):
        self.s, self.through, self.slate, self._proc = s, through, slate, process_runner

    def __call__(self, step: str, timeout: float) -> StepOutcome:
        if step == "snapshot":
            return dr.worker_snapshot(self.s)                     # type: ignore[arg-type] - duck-typed (data_dir, season)
        fd, name = tempfile.mkstemp(prefix=f"nightly_{step}_", suffix=".json")
        os.close(fd)
        result = Path(name)
        cmd = [dr.worker_python(), "-m", "src.ops.nightly", "--worker", step, "--result-file", str(result),
               "--season", self.s.season, "--data-dir", str(self.s.data_dir), "--reports-dir", str(self.s.reports_dir),
               "--through", self.through.isoformat(), "--slate", self.slate.isoformat(), "--top", str(self.s.top)]
        if self.s.offline:
            cmd.append("--offline")
        if self.s.as_of:
            cmd.append("--replay")                                 # the games step bounds its window at --through
        if self.s.league_id:
            cmd += ["--league-id", str(self.s.league_id)]
        if self.s.team_id:
            cmd += ["--team-id", str(self.s.team_id)]
        if self.s.mock_team:
            cmd += ["--mock-team", str(self.s.mock_team)]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "NBA_DATA_DIR": str(self.s.data_dir)}
        try:
            code, output = self._proc(cmd, timeout, cwd=self.s.repo_root, env=env)
            for line in output.splitlines()[-200:]:
                LOGGER.info("[%s] %s", step, line.rstrip())
            if code is None:
                return StepOutcome("timeout", error=f"exceeded its {timeout:.0f}s timeout and was killed")
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                tail = " | ".join(x for x in output.splitlines()[-3:] if x.strip())
                return StepOutcome("failed", error=f"worker exited {code} without a result: {tail[:400]}")
            if payload.get("ok"):
                degraded = payload.get("degraded") or []
                return StepOutcome("degraded" if degraded else "ok", payload.get("summary") or {}, payload.get("state") or {},
                                   error="; ".join(degraded) if degraded else None, note=payload.get("note"))
            return StepOutcome("failed", payload.get("summary") or {}, error=payload.get("error") or f"worker exited {code}")
        finally:
            result.unlink(missing_ok=True)


# --------------------------------------------------------------------------- worker bodies (subprocess)

StepError = dr.StepError


def _opening_night(base: Path, season: str) -> date | None:
    try:
        from src.inseason.schedule import read_schedule

        sch = read_schedule(base, season)
        return None if sch.empty else sch["game_date"].min().date()
    except Exception:                                                # noqa: BLE001
        return OPENING_NIGHTS.get(season)


def worker_games(a: argparse.Namespace) -> tuple[dict, dict]:
    from src.ingest.nba_client import NBAClient
    from src.ops.live_ingest import run_live_ingest

    client = NBAClient(a.data_dir / "raw" / "nba_api", offline=True if a.offline else None, min_interval=1.0)
    r = run_live_ingest(a.season, client, a.data_dir, completed_through=date.fromisoformat(a.through),
                        opening_night=_opening_night(a.data_dir, a.season), refresh=not a.offline, log=print,
                        date_to=date.fromisoformat(a.through) if getattr(a, "replay", False) else None)
    summary = {"skipped": r.skipped, "date_from": r.date_from, "last_game_date": r.last_game_date, "new_games": r.new_games,
               "new_player_rows": r.new_player_rows, "new_players": r.new_players, "pending_games": r.pending_games, "network_requests": r.network_requests,
               "tables_changed": [k for k, v in r.tables.items() if v.get("changed")]}
    return summary, {"games": summary}


def worker_schedule(a: argparse.Namespace) -> tuple[dict, dict]:
    import shutil

    import pandas as pd

    from src.inseason import schedule as sch
    from src.ingest.http_cache import CachedHttpClient

    client = CachedHttpClient(a.data_dir / "raw" / sch.CACHE_SOURCE, offline=True if a.offline else None, min_interval=2.0)
    new = sch.parse_pro_schedule(sch.fetch_pro_schedule(client, a.season, refresh=not a.offline), a.season)
    try:
        old = sch.read_schedule(a.data_dir, a.season)
    except (FileNotFoundError, sch.ContractError):
        old = new.iloc[0:0]
    if len(old) and len(new) < 0.95 * len(old):
        raise StepError(f"the fetched schedule has {len(new)} games but {len(old)} are stored; refusing to replace it")
    o, n = old.set_index("game_id")["game_date"], new.set_index("game_id")["game_date"]
    moved = [{"game_id": g, "from": str(o[g].date()), "to": str(n[g].date())} for g in sorted(set(o.index) & set(n.index)) if o[g] != n[g]]
    added, removed = sorted(set(n.index) - set(o.index)), sorted(set(o.index) - set(n.index))
    changed = bool(moved or added or removed) or not len(old)
    if changed:
        path = sch.table_path(a.data_dir)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".prev_nightly"))
        sch.write_schedule(new, a.data_dir)
    del pd
    summary = {"games": len(new), "changed": changed, "moved": len(moved), "added": len(added), "removed": len(removed),
               "requests": client.stats.network_requests}
    return summary, {"schedule": {**summary, "moved_games": moved[:20]}}


def worker_league(a: argparse.Namespace) -> tuple[dict, dict]:
    summary, state = dr.worker_league(a)
    from src.contracts import season_start

    path = a.data_dir / "processed" / "espn_league" / f"{a.league_id}_{season_start(a.season) + 1}.json"
    rep = json.loads(path.read_text("utf-8"))
    names: dict[str, str] = {}
    rosters = {}
    for t in rep.get("teams") or []:
        rosters[str(t["team_id"])] = {"name": t.get("name"), "players": [str(e["espn_player_id"]) for e in t.get("roster") or []]}
        names.update({str(e["espn_player_id"]): e.get("name") for e in t.get("roster") or []})
    names.update({str(f["espn_player_id"]): f.get("name") for f in rep.get("free_agents") or []})
    tx = [{"id": t.get("id"), "type": t.get("type"), "status": t.get("status"), "team_id": t.get("team_id"),
           "items": [{"type": i.get("type"), "player": str(i.get("playerId")), "team": i.get("toTeamId") if i.get("type") == "ADD" else i.get("fromTeamId")}
                     for i in (t.get("items") or [])]} for t in rep.get("transactions") or []]
    state = {**state, "league_state": {"rosters": rosters, "names": names, "transactions": tx, "fetched_at": rep.get("fetched_at"),
                                       "free_agent_espn_ids": [str(f["espn_player_id"]) for f in rep.get("free_agents") or []]}}
    return summary, state


def worker_analysis(a: argparse.Namespace) -> tuple[dict, dict]:
    from src.ops import nightly_analysis as na

    through, slate = date.fromisoformat(a.through), date.fromisoformat(a.slate)
    ctx = na.build_context(a.season, through, slate, data_dir=a.data_dir, league_id=a.league_id or None)
    fa_espn = None
    if ctx.league and ctx.id_map:
        fa_espn = {ctx.id_map[str(f["espn_player_id"])] for f in ctx.league.get("free_agents") or [] if str(f["espn_player_id"]) in ctx.id_map}
    art = na.build_artifacts(ctx, slate, team_id=a.team_id or None, mock_team=a.mock_team or None, top=a.top, espn_fa_ids=fa_espn)
    out = a.reports_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in art.frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, encoding="utf-8")
        files[name] = str(p)
    tables = {n: dr._clean(df.head(a.top * 2 if n != "ros" else 30).to_dict("records")) for n, df in art.frames.items()}
    report = {"meta": art.meta, "tables": tables, "notes": art.notes, "errors": art.errors, "files": files}
    summary = {"tables": {n: len(df) for n, df in art.frames.items()}, "errors": art.errors, "team": art.meta.get("team_label"),
               "week": (art.meta.get("week") or {}).get("this_week")}
    degraded = [f"{k}: {v}" for k, v in art.errors.items()]
    state = {"analysis": dr._clean(art.state), "_report": dr._clean(report)}
    return dr._clean(summary), {**state, "_degraded": degraded}


WORKERS = {"games": worker_games, "roster": dr.worker_roster, "adp": dr.worker_adp, "status": dr.worker_status,
           "league": worker_league, "schedule": worker_schedule, "analysis": worker_analysis}


def run_worker(a: argparse.Namespace) -> int:
    a.data_dir, a.reports_dir = Path(a.data_dir), Path(a.reports_dir)
    result: dict[str, Any]
    try:
        summary, state = WORKERS[a.worker](a)
        degraded = state.pop("_degraded", None) if isinstance(state, dict) else None
        result = {"ok": True, "summary": dr._clean(summary), "state": dr._clean(state), "degraded": degraded or []}
    except Exception as exc:                                        # noqa: BLE001 - the parent records it and moves on
        traceback.print_exc()
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(a.result_file).write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


# --------------------------------------------------------------------------- diffs, freshness, alerts

def _league_diff(prev: Mapping[str, Any], cur: Mapping[str, Any], mine: str | None) -> dict[str, Any]:
    names = {**prev.get("names", {}), **cur.get("names", {})}
    moves = []
    for tid, now in (cur.get("rosters") or {}).items():
        before = (prev.get("rosters") or {}).get(tid)
        if before is None:
            continue
        add = sorted(set(now["players"]) - set(before["players"]))
        drop = sorted(set(before["players"]) - set(now["players"]))
        if add or drop:
            moves.append({"team_id": tid, "team": now.get("name"), "mine": tid == mine,
                          "added": [names.get(p, p) for p in add], "dropped": [names.get(p, p) for p in drop]})
    moves.sort(key=lambda m: (not m["mine"], str(m["team"])))
    seen = {t["id"] for t in prev.get("transactions") or []}
    new_tx = [{"type": t["type"], "status": t["status"], "team": (cur["rosters"].get(str(t["team_id"])) or {}).get("name"),
               "items": [f"{i['type']} {names.get(i['player'], i['player'])}" for i in t["items"]]}
              for t in cur.get("transactions") or [] if t["id"] not in seen]
    return {"moves": moves, "new_transactions": new_tx}


def _keyed_diff(prev: Mapping[str, Any] | None, cur: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if prev is None or cur is None:
        return None
    return {"new": [{"player_id": k, **v} for k, v in cur.items() if k not in prev],
            "gone": [{"player_id": k, **v} for k, v in prev.items() if k not in cur]}


def _ros_movers(prev: Mapping[str, Any] | None, cur: Mapping[str, Any] | None, min_move: int = 8) -> dict[str, Any] | None:
    if not prev or not cur:
        return None
    risers, fallers = [], []
    for k, v in cur.items():
        if k in prev:
            d = v["rank"] - prev[k]["rank"]
            (risers if d <= -min_move else fallers if d >= min_move else []).append({"name": v["name"], "from": prev[k]["rank"], "to": v["rank"]})
    return {"risers": sorted(risers, key=lambda r: r["to"] - r["from"])[:10], "fallers": sorted(fallers, key=lambda r: r["from"] - r["to"])[:10],
            "entered": [v["name"] for k, v in cur.items() if k not in prev][:10], "left_top": [v["name"] for k, v in prev.items() if k not in cur][:10]}


def build_diffs(states: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    prev = dr.newest_with(states, "roster_map")
    if prev and state.get("roster_map") is not None:
        diffs["roster"] = rd.roster_diff(prev["roster_map"], state["roster_map"])
    prev = dr.newest_with(states, "snapshot")
    if prev and state.get("snapshot"):
        cur_p, prev_p = Path(state["snapshot"]), Path(prev["snapshot"])
        if cur_p == prev_p:
            diffs["injuries"] = []
        elif cur_p.exists() and prev_p.exists():
            rows = rd.injury_changes(rd.read_snapshot(prev_p), rd.read_snapshot(cur_p))
            mine = set(((state.get("league_state") or {}).get("my_espn_ids")) or [])
            for r in rows:
                r["mine"] = r["id"] in mine
            rows.sort(key=lambda r: (not r["mine"], -(r.get("pct_owned") or 0)))
            diffs["injuries"] = rows
    prev = dr.newest_with(states, "league_state")
    if prev and state.get("league_state"):
        diffs["league"] = _league_diff(prev["league_state"], state["league_state"], state["league_state"].get("my_team_id"))
    a_now = state.get("analysis") or {}
    prev = dr.newest_with(states, "analysis")
    a_prev = (prev or {}).get("analysis") or {}
    if prev:
        diffs["rising"] = _keyed_diff(a_prev.get("rising_minutes"), a_now.get("rising_minutes"))
        diffs["beneficiaries"] = _keyed_diff(a_prev.get("beneficiaries"), a_now.get("beneficiaries"))
        diffs["adds"] = _keyed_diff(a_prev.get("adds_top"), a_now.get("adds_top"))
        diffs["ros"] = _ros_movers(a_prev.get("ros_rank"), a_now.get("ros_rank"))
        if a_prev.get("calendar_source") and a_now.get("calendar_source") and a_prev["calendar_source"] != a_now["calendar_source"]:
            diffs["calendar"] = {"from": a_prev["calendar_source"], "to": a_now["calendar_source"]}
        if a_prev.get("my_roster") is not None and a_now.get("my_roster") is not None and a_prev["my_roster"] != a_now["my_roster"]:
            diffs["my_roster"] = {"added": sorted(set(a_now["my_roster"]) - set(a_prev["my_roster"])),
                                  "dropped": sorted(set(a_prev["my_roster"]) - set(a_now["my_roster"]))}
    return diffs


def _latest_game_date(base: Path, season: str) -> date | None:
    try:
        import pandas as pd

        df = pd.read_parquet(base / "processed" / "team_games.parquet", columns=["season", "game_date"])
        df = df[df["season"] == season]
        return None if df.empty else pd.Timestamp(df["game_date"].max()).date()
    except Exception:                                               # noqa: BLE001 - freshness is advisory
        return None


def _scheduled_through(base: Path, season: str, through: date) -> date | None:
    try:
        from src.inseason.schedule import read_schedule

        sch = read_schedule(base, season)
        past = sch[sch["game_date"] <= str(through)]
        return None if past.empty else past["game_date"].max().date()
    except Exception:                                               # noqa: BLE001
        return None


def _newest_snapshot_time(s: Settings) -> str | None:
    snaps = rd.list_snapshots(s.data_dir, s.season)
    if not snaps:
        return None
    try:
        return datetime.strptime(snaps[-1].name.split(".")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _league_fetched_at(s: Settings) -> str | None:
    from src.contracts import season_start

    rep = dr._read_json(s.data_dir / "processed" / "espn_league" / f"{s.league_id}_{season_start(s.season) + 1}.json") or {}
    return rep.get("fetched_at")


def freshness_and_reasons(s: Settings, state: dict[str, Any], now_utc: datetime, failures: int, through: date) -> tuple[dict[str, str], list[str]]:
    fresh: dict[str, str] = {}
    reasons: list[str] = []
    if failures >= s.alert_after:
        reasons.append(f"{s.alert_after} or more consecutive runs had failing steps")
    if s.as_of:
        return {"replay": f"as-of {s.as_of}: freshness checks skipped"}, reasons
    got = _latest_game_date(s.data_dir, s.season)
    want = _scheduled_through(s.data_dir, s.season, through)
    if want is not None:
        lag = (want - got).days if got else None
        fresh["Games ingested through"] = f"{got or 'nothing'} (schedule has games through {want}; completed through {through})"
        if "games" in s.steps and (got is None or (lag is not None and lag > 1)):
            reasons.append(f"game data is stale: newest stored game {got or 'none'}, but games were scheduled through {want}")
    elif got:
        fresh["Games ingested through"] = f"{got} (no schedule to compare against)"
    taken = state.get("snapshot_taken_at") or _newest_snapshot_time(s)          # a failed step falls back to what is on disk
    if taken:
        age = (now_utc - datetime.fromisoformat(taken)).total_seconds() / 3600
        fresh["ESPN player snapshot"] = f"{taken} ({age:.1f} h old)"
        if age > s.stale_hours:
            reasons.append(f"ESPN data is {age:.0f} h old (limit {s.stale_hours:.0f} h): the adp step is not refreshing it")
    elif "snapshot" in s.steps:
        fresh["ESPN player snapshot"] = "none archived yet"
        reasons.append("no ESPN snapshot has been archived yet")
    lg = (state.get("league_state") or {}).get("fetched_at") or (_league_fetched_at(s) if s.league_id else None)
    if lg:
        age = (now_utc - datetime.fromisoformat(lg)).total_seconds() / 3600
        fresh["ESPN league sync"] = f"{lg} ({age:.1f} h old)"
        if age > s.stale_hours:
            reasons.append(f"the ESPN league sync is {age:.0f} h old (limit {s.stale_hours:.0f} h)")
    elif s.league_id and "league" in s.steps:
        fresh["ESPN league sync"] = "never synced"
        reasons.append("the ESPN league has never been synced")
    rdate = dr.latest_roster_date(s.data_dir)
    if rdate:
        age_d = max(0, (now_utc.astimezone().date() - date.fromisoformat(rdate)).days)
        fresh["NBA roster snapshot"] = f"{rdate} ({age_d} day(s) old)"
        if "roster" in s.steps and age_d * 24 > s.stale_hours:
            reasons.append(f"the NBA roster snapshot is from {rdate}, {age_d} days old")
    return fresh, reasons


def update_alert(s: Settings, reasons: list[str], run_id: str, notifier: Callable[[str, str], None], failures: int = 0) -> bool:
    path = s.reports_dir / "ALERT.txt"
    if not reasons:
        if path.exists():
            path.unlink()
            LOGGER.info("alert cleared: %s removed", path)
        return False
    body = (f"NBA fantasy nightly job needs attention (run {run_id}, season {s.season}).\n\n"
            + "\n".join(f"- {r}" for r in reasons)
            + f"\n\nConsecutive runs with failing steps so far: {failures}.\nSee reports/nightly/latest.md, {s.ops_dir / 'status.json'} and the log in {s.ops_dir / 'logs'}.\n"
            "Re-run by hand: python -m src.ops.nightly --force   (or ./dev schedule run-now --job nightly).\n"
            "This file is removed automatically by the next fully successful run.\n")
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    dr.atomic_write_text(path, body)
    if previous is None or dr._reasons_of(previous) != reasons:
        notifier("NBA fantasy nightly job needs attention", reasons[0])
    return True


# --------------------------------------------------------------------------- orchestration

def execute(s: Settings, runner: Callable[[str, float], StepOutcome], *, now: datetime, through: date, slate: date,
            notifier: Callable[[str, str], None] = dr.toast, lock_factory: Callable[[Path, float], RunLock] | None = None,
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> int:
    ops = s.ops_dir
    run_id = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    make_lock = lock_factory or (lambda path, stale: RunLock(path, stale_after=stale))
    lock = make_lock(s.lock_path, s.budget_s + 15 * 60)
    waited_from = clock()
    while True:
        try:
            lock.acquire()
            break
        except LockBusy as exc:
            if clock() - waited_from >= s.lock_wait_s:
                LOGGER.warning("skipped after waiting %.0f min: %s", (clock() - waited_from) / 60, exc)
                return EXIT_BUSY
            LOGGER.info("shared lock busy (%s); waiting", exc)
            sleep(min(30.0, max(1.0, s.lock_wait_s - (clock() - waited_from))))
    if lock.recovered_stale:
        LOGGER.warning("recovered a stale lock; the old one was kept as %s", lock.recovered_stale)
    started = clock()
    try:
        prior_status = dr._read_json(ops / "status.json") or {}
        dr.atomic_write_text(ops / "status.json", json.dumps(
            {**prior_status, "run_id": run_id, "outcome": "running", "started_at": now.isoformat(), "pid": os.getpid()}, indent=2))
        outcomes: dict[str, StepOutcome] = {}
        for step in s.steps:
            if step == "league" and not s.league_id:
                outcomes[step] = StepOutcome("skipped", note="no ESPN league id configured (ESPN_LEAGUE_ID, --league-id, or league_id in nightly.json)")
                continue
            remaining = s.budget_s - (clock() - started)
            if remaining < 5:
                outcomes[step] = StepOutcome("skipped", error="run-time budget exhausted before this step")
                LOGGER.error("%s: budget exhausted", step)
                continue
            timeout = min(s.timeouts.get(step, 600), remaining)
            LOGGER.info("== %s (timeout %.0fs)", step, timeout)
            t0 = clock()
            try:
                outcome = runner(step, timeout)
            except Exception as exc:                                # noqa: BLE001 - one step never stops the others
                LOGGER.exception("%s crashed", step)
                outcome = StepOutcome("failed", error=f"{type(exc).__name__}: {exc}")
            outcome.seconds = outcome.seconds or clock() - t0
            outcomes[step] = outcome
            LOGGER.info("%s: %s (%.0fs)%s", step, outcome.status, outcome.seconds, f" {outcome.error}" if outcome.error else "")
        return _finish(s, outcomes, now, run_id, started, clock, notifier, through, slate)
    finally:
        lock.release()


def _finish(s: Settings, outcomes: dict[str, StepOutcome], now: datetime, run_id: str, started: float,
            clock: Callable[[], float], notifier: Callable[[str, str], None], through: date, slate: date) -> int:
    ops = s.ops_dir
    failed = [n for n, o in outcomes.items() if o.status in BAD or (o.status == "skipped" and o.error)]
    ok_any = any(o.status in ("ok", "degraded") for o in outcomes.values())
    outcome = "ok" if not failed else ("partial" if ok_any else "failed")
    state: dict[str, Any] = {}
    for o in outcomes.values():
        if o.status in ("ok", "degraded"):
            state.update(o.state)
    league_state = state.get("league_state")
    if league_state and s.team_id is not None:                  # my ESPN ids, for marking injury changes that concern me
        mine = (league_state.get("rosters") or {}).get(str(s.team_id)) or {}
        league_state["my_team_id"] = str(s.team_id)
        league_state["my_espn_ids"] = mine.get("players", [])
    report_part = state.pop("_report", None)
    states = dr.load_prior_states(ops / "runs", run_id)
    diffs = build_diffs(states, state)
    history = dr.read_history(ops / "history.jsonl") + [{"run_id": run_id, "ok": not failed}]
    failures = dr.consecutive_failures(history)
    now_utc = now.astimezone(timezone.utc)
    fresh, reasons = freshness_and_reasons(s, state, now_utc, failures, through)
    team_problem = team_setup_problem(s)
    if team_problem and not s.as_of and draft_done(s):          # before the draft there is no roster to be missing
        reasons.append(team_problem)
    prev_state = states[-1] if states else None
    finished = datetime.now(timezone.utc)
    local_now = now
    games = (outcomes.get("games").summary if outcomes.get("games") and outcomes["games"].status == "ok" else None)
    run = {
        "run_id": run_id, "date": slate.isoformat() if s.as_of else local_now.strftime("%Y-%m-%d"), "season": s.season, "outcome": outcome,
        "started_at": now.isoformat(), "finished_at": finished.isoformat(), "finished_local": finished.astimezone().strftime("%Y-%m-%d %H:%M"),
        "seconds": clock() - started, "completed_through": through.isoformat(), "slate": slate.isoformat(),
        "window": [s.window_start.isoformat(), s.window_end.isoformat()], "window_source": s.window_source,
        "days_left": (s.window_end - slate).days, "replay": s.as_of is not None,
        "prev_run_id": prev_state["run_id"] if prev_state else None, "prev_time": prev_state.get("finished_local") if prev_state else None,
        "steps": [{"name": n, "status": o.status, "seconds": o.seconds, "error": o.error, "note": o.note} for n, o in outcomes.items()],
        "diffs": diffs, "games": games,
        "schedule": (outcomes["schedule"].summary if outcomes.get("schedule") and outcomes["schedule"].status == "ok" else None),
        "league": (outcomes["league"].summary if outcomes.get("league") and outcomes["league"].status == "ok" else None),
        "analysis": report_part, "freshness": fresh, "team_setup": team_problem, "alert": {"reasons": reasons},
        "files": {"data": str(s.reports_dir / "data"), "json": str(s.reports_dir / "latest.json"), "status": str(ops / "status.json"),
                  "log": str(ops / "logs" / "nightly.log")},
    }
    report = render_report(run)
    dr.atomic_write_text(s.reports_dir / f"{run['date']}.md", report)
    dr.atomic_write_text(s.reports_dir / "latest.md", report)
    dr.atomic_write_text(s.reports_dir / "runs" / f"{run_id}.md", report)
    dr.atomic_write_text(s.reports_dir / "latest.json", json.dumps(run, indent=1, default=str))
    run_state = {**state, "run_id": run_id, "finished_local": run["finished_local"], "outcome": outcome}
    dr.atomic_write_text(ops / "runs" / f"{run_id}.json", json.dumps(run_state, default=str))
    with open(ops / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": run_id, "finished_at": run["finished_at"], "ok": not failed, "outcome": outcome, "failed_steps": failed}) + "\n")
    alerting = update_alert(s, reasons, run_id, notifier, failures) if not s.as_of else False
    prior_ok = dr._read_json(ops / "status.json") or {}
    dr.atomic_write_text(ops / "status.json", json.dumps({
        "run_id": run_id, "outcome": outcome, "ok": not failed, "exit_code": EXIT_OK if not failed else EXIT_FAILED,
        "started_at": now.isoformat(), "finished_at": run["finished_at"], "seconds": run["seconds"], "season": s.season,
        "completed_through": through.isoformat(), "slate": slate.isoformat(), "window": run["window"],
        "consecutive_failures": failures, "last_success_at": run["finished_at"] if not failed else prior_ok.get("last_success_at"),
        "alert": alerting, "alert_reasons": reasons, "team_setup": team_problem, "replay": s.as_of is not None,
        "steps": {n: {"status": o.status, "seconds": round(o.seconds, 1), "error": o.error, "summary": dr._small(o.summary)} for n, o in outcomes.items()},
        "report": str(s.reports_dir / "latest.md")}, indent=2, default=str))
    LOGGER.info("run %s finished: %s%s; report %s", run_id, outcome, f" (failed: {', '.join(failed)})" if failed else "", s.reports_dir / "latest.md")
    return EXIT_OK if not failed else EXIT_FAILED


# --------------------------------------------------------------------------- which team is mine

def league_file(s: Settings) -> Path:
    from src.contracts import season_start

    return s.data_dir / "processed" / "espn_league" / f"{s.league_id}_{season_start(s.season) + 1}.json"


def league_teams(s: Settings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(teams, league report)`` from the last league sync; ``([], {})`` when there is none yet."""
    if not s.league_id:
        return [], {}
    rep = dr._read_json(league_file(s)) or {}
    members = {m["id"]: m.get("display_name") for m in rep.get("members") or []}
    rows = [{"team_id": int(t["team_id"]), "name": t.get("name"), "abbrev": t.get("abbrev"),
             "owners": ", ".join(str(members.get(o) or o) for o in t.get("owner_ids") or []) or "(unowned)",
             "players": len(t.get("roster") or [])} for t in rep.get("teams") or []]
    return sorted(rows, key=lambda r: r["team_id"]), rep


def team_setup_problem(s: Settings) -> str | None:
    """Why the team-specific sections cannot be built, phrased as an action; ``None`` when the team is set and valid."""
    hint = "run `python -m src.ops.nightly --set-team` to list the league's teams, then `--set-team N`"
    teams, rep = league_teams(s)
    if s.mock_team is not None:
        return None
    if s.team_id is None:
        return f"YOUR TEAM IS NOT SET, so lineup hints, waiver adds and 'mine' tags are off; {hint}"
    if teams and s.team_id not in {t["team_id"] for t in teams}:
        return f"configured team id {s.team_id} is not a team of league {s.league_id} (teams: {', '.join(str(t['team_id']) for t in teams)}); {hint}"
    return None


def draft_done(s: Settings) -> bool:
    return bool(league_teams(s)[1].get("draft_completed"))


def write_team_setting(data_dir: Path, team_id: int) -> Path:
    path = data_dir / SETTINGS_FILE
    current = _read_settings_file(path)
    current["team_id"] = int(team_id)
    dr.atomic_write_text(path, json.dumps(current, indent=2) + "\n")
    return path


def set_team_command(s: Settings, value: str, out: Callable[[str], None] = print) -> int:
    """``--set-team`` lists the league's teams; ``--set-team N`` validates N against the last league sync and saves it."""
    teams, rep = league_teams(s)
    if not s.league_id:
        out("error: no ESPN league id configured (ESPN_LEAGUE_ID, --league-id, or league_id in nightly.json / daily_refresh.json)")
        return EXIT_CONFIG
    if not teams:
        out(f"error: no synced league data at {league_file(s)}; run `python -m src.ingest.espn_league --league-id {s.league_id}` "
            "(or `python -m src.ops.nightly --force --only league`) first")
        return EXIT_CONFIG
    listing = ["", f"League {s.league_id} ({(rep.get('settings') or {}).get('name')}), synced {rep.get('fetched_at')}; draft "
               + ("done" if rep.get("draft_completed") else "NOT held yet (rosters are empty until it is)"),
               "  id  abbrev  players  team / owner (ESPN's public data has no 'this is me' flag; match your team name in the ESPN app)"]
    listing += [f"  {t['team_id']:>2}  {str(t['abbrev']):<6}  {t['players']:>7}  {t['name']}  /  {t['owners']}" for t in teams]
    if value in ("", "list"):
        out("\n".join(listing + ["", "Save yours with: python -m src.ops.nightly --set-team N"]))
        cur = s.team_id
        out(f"currently configured: {cur if cur is not None else 'nothing'}")
        return EXIT_OK
    try:
        n = int(value)
    except ValueError:
        out(f"error: {value!r} is not a team id number")
        return EXIT_CONFIG
    match = [t for t in teams if t["team_id"] == n]
    if not match:
        out("\n".join([f"error: team id {n} is not in league {s.league_id}"] + listing))
        return EXIT_CONFIG
    path = write_team_setting(s.data_dir, n)
    t = match[0]
    out(f"saved team_id {n} ({t['name']}, owner {t['owners']}, {t['players']} players on the roster) to {path}")
    if not t["players"]:
        out("note: that roster is empty because the draft has not been held; team sections start working after it")
    return EXIT_OK


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ops.nightly", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default=None, help="the live season (default: derived from today)")
    p.add_argument("--season-start", default=None, help="YYYY-MM-DD first day of the window (default: first game in schedule_games)")
    p.add_argument("--window-start", default=None, help="same as --season-start")
    p.add_argument("--window-end", default=None, help="YYYY-MM-DD last day (default: end of the league's final matchup period + 2 days)")
    p.add_argument("--force", action="store_true", help="run even outside the season window")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD: replay a night as if this were the last completed game day (no alerts, no freshness checks)")
    p.add_argument("--dry-run", action="store_true", help="print what would run and exit; no lock, no network, no writes")
    p.add_argument("--offline", action="store_true", help="never touch the network (cached data only)")
    p.add_argument("--only", default=None, help=f"comma-separated subset of {list(STEP_ORDER)}")
    p.add_argument("--skip", default=None, help="comma-separated steps to skip")
    p.add_argument("--league-id", type=int, default=None, help="ESPN league id (or ESPN_LEAGUE_ID, or league_id in nightly.json / daily_refresh.json)")
    p.add_argument("--team-id", type=int, default=None, help="your ESPN team id (or ESPN_TEAM_ID, or team_id in nightly.json)")
    p.add_argument("--set-team", nargs="?", const="list", default=None, metavar="N",
                   help="with no value: list the league's teams; with N: validate it against the last league sync and save it as your team in nightly.json")
    p.add_argument("--mock-team", type=int, default=None, help="demo only: use team N of a mock snake draft as your roster")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--reports-dir", type=Path, default=None, help="default: <repo>/reports/nightly (gitignored)")
    p.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--top", type=int, default=15, help="rows per recommendation table")
    p.add_argument("--budget-minutes", type=float, default=60.0, help="whole-run time budget")
    p.add_argument("--lock-wait-minutes", type=float, default=20.0,
                   help="wait this long for the shared lock (the pre-draft refresh may still be running at the handoff) before skipping the night")
    p.add_argument("--stale-hours", type=float, default=36.0, help="alert when ESPN data is older than this")
    p.add_argument("--alert-after", type=int, default=2, help="alert after this many consecutive failing runs")
    p.add_argument("--quiet", action="store_true", help="console shows warnings only (the log file always has everything)")
    p.add_argument("--worker", choices=sorted(WORKERS), default=None, help=argparse.SUPPRESS)
    p.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--replay", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--through", default=None, help=argparse.SUPPRESS)
    p.add_argument("--slate", default=None, help=argparse.SUPPRESS)
    return p


def describe_plan(s: Settings, now: datetime) -> str:
    gate, through, slate = run_days(s, now)
    inside = in_window(gate, s)
    return "\n".join([
        f"nightly for {s.season}: {gate} is {'INSIDE' if inside else 'OUTSIDE'} the window {s.window_start} .. {s.window_end} ({s.window_source})",
        f"games completed through {through}; next slate {slate}; data {s.data_dir}; reports {s.reports_dir}",
        f"steps: {', '.join(s.steps)}; league id: {s.league_id or 'not configured (league step skipped)'}; team id: {s.team_id or 'not configured (team artifacts skipped)'}"
        f"{'; MOCK team ' + str(s.mock_team) if s.mock_team else ''}; offline: {s.offline}; replay: {s.as_of is not None}",
        f"budget {s.budget_s / 60:.0f} min; timeouts {', '.join(f'{k} {int(v)}s' for k, v in s.timeouts.items() if k in s.steps)}"])


def main(argv: Sequence[str] | None = None, *, runner: Callable[[str, float], StepOutcome] | None = None,
         now: datetime | None = None, notifier: Callable[[str, str], None] = dr.toast, env: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not args.result_file or not args.data_dir or not args.through or not args.slate:
            print("error: --worker needs --result-file, --data-dir, --through and --slate", file=sys.stderr)
            return EXIT_CONFIG
        args.league_id = args.league_id or None
        args.reports_dir = args.reports_dir or REPO_ROOT / "reports" / "nightly"
        return run_worker(args)
    stamp = now or datetime.now().astimezone()
    try:
        s = resolve_settings(args, os.environ if env is None else env, stamp)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if args.set_team is not None:
        return set_team_command(s, args.set_team)
    if args.dry_run:
        print(describe_plan(s, stamp))
        return EXIT_OK
    dr.setup_logging(s.ops_dir / "logs", quiet=args.quiet, logger=LOGGER, filename="nightly.log")
    LOGGER.info("start: %s", describe_plan(s, stamp).replace("\n", " | "))
    gate, through, slate = run_days(s, stamp)
    if not args.force and not in_window(gate, s):
        LOGGER.info("outside the season window %s .. %s: nothing to do", s.window_start, s.window_end)
        return EXIT_OK
    code = execute(s, runner or SubprocessRunner(s, through, slate), now=stamp, through=through, slate=slate, notifier=notifier)
    if sys.stdout is not None:
        try:
            print(f"nightly: exit {code}; report {s.reports_dir / 'latest.md'}")
        except UnicodeEncodeError:
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
