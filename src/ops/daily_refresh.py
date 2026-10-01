"""Unattended daily refresh for the weeks before the draft (ADR 0014).

    python -m src.ops.daily_refresh [--draft-date 2026-10-17] [--force] [--offline] [--dry-run]

One command, safe to run twice a day, that does in order:

1. ``roster``     today's NBA roster snapshot + newly drafted rookies (``nba_incoming``);
2. ``offseason``  Summer League and preseason box scores, re-downloaded (``nba_offseason``);
3. ``adp``        a fresh ESPN player universe, ADP table and id map (``espn_adp``);
4. ``profiles``   player country / previous-organisation profiles and draft-combine rows (ADR 0016);
5. ``status``     the dated ESPN injury-status archive ``espn_status_snapshots`` (ADR 0016; ESPN keeps no history);
6. ``snapshot``   a dated, compact archive of that ESPN payload (ADR 0005: our only point-in-time ADP/injury history);
7. ``league``     the read-only ESPN league sync, only when a league id is configured (ADR 0008);
8. ``watchlist``  the breakout watchlist with the model chosen from the data available (ADR 0012);
9. ``board``      draft-board CSVs (the plain ``baseline`` and the information-matched offseason model).

Then it diffs against the previous run and writes ``reports/daily/<date>.md`` and ``latest.md`` (roster moves, injury
status changes, ADP risers and fallers, players newly on or off the watchlist, new preseason games, failures).

Operational guarantees: a single-instance lock (stale-lock aware, never deleted silently); each step runs in its own
subprocess with a timeout and a whole-run budget, and a failing step never stops the others; a structured
``status.json``; a rolling log; a date window (``--window-start`` / ``--window-end`` / ``--draft-date``) outside which a
scheduled invocation exits 0 having done nothing; and an ``ALERT.txt`` plus a best-effort Windows toast when runs keep
failing or the data goes stale.

Exit codes: 0 ok (or outside the window), 1 one or more steps failed, 2 configuration error, 3 another run holds the lock.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.ops import refresh_diff as rd
from src.ops.daily_report import render_report
from src.ops.runlock import LockBusy, RunLock

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_ORDER = ("roster", "offseason", "adp", "profiles", "status", "snapshot", "league", "watchlist", "board", "transactions", "coaches")
DEFAULT_TIMEOUTS = {"roster": 300, "offseason": 600, "adp": 300, "profiles": 300, "status": 60, "snapshot": 60, "league": 180, "watchlist": 600, "board": 600, "transactions": 300, "coaches": 120}
FALLBACK_DRAFT_DATE = date(2026, 10, 18)     # only used if config/league.yaml, the env and the local json all lack a draft date
DEFAULT_DRAFT_TZ = "Pacific/Auckland"        # the zone the user's draft date is written in (config draft.timezone overrides)
WINDOW_DAYS_BEFORE = 30                       # default window opens this long before the draft date
WINDOW_DAYS_AFTER = 1                         # and closes this long after it
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_BUSY = 0, 1, 2, 3
SETTINGS_FILE = "daily_refresh.json"
LOGGER = logging.getLogger("daily_refresh")


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- settings

@dataclass
class Settings:
    season: str
    data_dir: Path
    reports_dir: Path
    repo_root: Path
    draft_date: date
    draft_date_source: str
    window_start: date
    window_end: date
    league_id: int | None = None
    draft_start: datetime | None = None           # tz-aware instant of the draft start, if known
    draft_tz: str = DEFAULT_DRAFT_TZ
    offline: bool = False
    budget_s: float = 45 * 60
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    stale_hours: float = 36.0
    alert_after: int = 2
    board_models: tuple[str, ...] = ("baseline", "auto", "baseline_offseason_debut", "baseline_hurdle_adp_offseason_debut")
    steps: tuple[str, ...] = STEP_ORDER

    @property
    def ops_dir(self) -> Path:
        return self.data_dir / "daily_refresh"


def _parse_date(text: str, what: str) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError as exc:
        raise ConfigError(f"{what}: {text!r} is not a date like 2026-10-17") from exc


def _parse_instant(text: str, what: str) -> datetime:
    try:
        dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(f"{what}: {text!r} is not a timestamp like 2026-10-16T18:00:00Z") from exc
    if dt.tzinfo is None:
        raise ConfigError(f"{what}: {text!r} needs a UTC offset or Z (e.g. 2026-10-16T18:00:00Z)")
    return dt


def config_draft(repo_root: Path) -> dict[str, Any]:
    """The ``league.draft`` block of ``<repo>/config/league.yaml`` ({} if absent or unreadable)."""
    try:
        import yaml

        cfg = yaml.safe_load((repo_root / "config" / "league.yaml").read_text(encoding="utf-8")) or {}
        block = (cfg.get("league") or {}).get("draft") or {}
        return block if isinstance(block, dict) else {}
    except Exception:                                                  # noqa: BLE001 - a broken yaml must not stop the refresh
        return {}


def resolve_draft(flag_date: str | None, flag_start: str | None, env: Mapping[str, str], local: Mapping[str, Any],
                  repo_root: Path) -> tuple[date | None, datetime | None, str, str | None]:
    """Draft date, start instant, time zone name and the date's source.

    Precedence per value: flag > env (NBA_DRAFT_DATE / NBA_DRAFT_START) > ``daily_refresh.json`` (draft_date / draft_start)
    > ``config/league.yaml`` (draft.date / draft.start_utc / draft.timezone). A config start is ignored if its local calendar
    date disagrees with a date given by a higher-precedence source.
    """
    from zoneinfo import ZoneInfo

    cfg = config_draft(repo_root)
    tzname = str(cfg.get("timezone") or DEFAULT_DRAFT_TZ)
    tz = ZoneInfo(tzname)

    def pick(flag, env_key, file_key, cfg_key):
        for src, val in (("flag", flag), ("env", env.get(env_key)), ("file", local.get(file_key)), ("config", cfg.get(cfg_key))):
            if val:
                return str(val), src
        return None, None

    d_text, d_src = pick(flag_date, "NBA_DRAFT_DATE", "draft_date", "date")
    s_text, s_src = pick(flag_start, "NBA_DRAFT_START", "draft_start", "start_utc")
    start = _parse_instant(s_text, "draft start") if s_text else None
    draft = _parse_date(d_text, "draft date") if d_text else None
    if start is not None and draft is not None and start.astimezone(tz).date() != draft:
        if s_src == "config":
            start = None                               # a higher-precedence date overrides the config start
        elif d_src == "config":
            draft, d_src = None, None                  # an explicit start overrides the config date
    if start is not None and draft is None:
        draft, d_src = start.astimezone(tz).date(), s_src
    return draft, start, tzname, d_src


def _read_local_settings(data_dir: Path) -> dict[str, Any]:
    path = data_dir / SETTINGS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def _league_id_from_dotenv(repo_root: Path) -> int | None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return None
    try:
        from dotenv import dotenv_values

        val = (dotenv_values(env_file).get("ESPN_LEAGUE_ID") or "").strip()
        return int(val) if val else None
    except (ImportError, ValueError, OSError):
        return None


def resolve_settings(args: argparse.Namespace, env: Mapping[str, str], today: date) -> Settings:
    """Flag > environment > ``<data_dir>/daily_refresh.json`` > default, for the draft date, window and league id."""
    from src.contracts import data_dir as default_data_dir
    from src.contracts import season_start, season_str
    from src.ingest.nba_offseason import live_season_start

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    local = _read_local_settings(data_dir)
    season = args.season or local.get("season") or season_str(live_season_start(today))
    try:
        season_start(season)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    def pick(flag, env_key, file_key):
        if flag:
            return flag, "flag"
        if env.get(env_key):
            return env[env_key], "env"
        if local.get(file_key):
            return str(local[file_key]), "file"
        return None, None

    repo_root = Path(args.repo_root) if getattr(args, "repo_root", None) else REPO_ROOT
    draft_found, draft_start, draft_tz, dd_src = resolve_draft(args.draft_date, getattr(args, "draft_start", None), env, local, repo_root)
    draft = draft_found or FALLBACK_DRAFT_DATE
    ws_text, _ = pick(args.window_start, "NBA_WINDOW_START", "window_start")
    we_text, _ = pick(args.window_end, "NBA_WINDOW_END", "window_end")
    start = _parse_date(ws_text, "window start") if ws_text else draft - timedelta(days=WINDOW_DAYS_BEFORE)
    end = _parse_date(we_text, "window end") if we_text else draft + timedelta(days=WINDOW_DAYS_AFTER)
    if end < start:
        raise ConfigError(f"window end {end} is before window start {start}")

    league_text, _ = pick(str(args.league_id) if args.league_id else None, "ESPN_LEAGUE_ID", "league_id")
    try:
        league_id = int(league_text) if league_text else _league_id_from_dotenv(repo_root)
    except ValueError as exc:
        raise ConfigError(f"league id {league_text!r} is not an integer") from exc

    steps = tuple(s.strip() for s in (args.only or ",".join(STEP_ORDER)).split(",") if s.strip())
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    unknown = [s for s in (*steps, *skip) if s not in STEP_ORDER]
    if unknown:
        raise ConfigError(f"unknown step(s) {unknown}; expected some of {list(STEP_ORDER)}")
    steps = tuple(s for s in STEP_ORDER if s in steps and s not in skip)
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo_root / "reports" / "daily"
    return Settings(
        season=season, data_dir=data_dir, reports_dir=reports_dir, repo_root=repo_root, draft_date=draft,
        draft_date_source=dd_src or "fallback", window_start=start, window_end=end, league_id=league_id,
        draft_start=draft_start, draft_tz=draft_tz, offline=bool(args.offline), budget_s=args.budget_minutes * 60.0, stale_hours=args.stale_hours,
        alert_after=args.alert_after, steps=steps,
        board_models=tuple(m.strip() for m in args.board_models.split(",") if m.strip()))


def in_window(today: date, s: Settings, now: datetime | None = None) -> bool:
    """Inside the date window and, when the draft start is known and ``now`` is given, before that instant."""
    if now is not None and s.draft_start is not None and now >= s.draft_start:
        return False
    return s.window_start <= today <= s.window_end


def hours_to_draft(s: Settings, now: datetime) -> float | None:
    return None if s.draft_start is None else (s.draft_start - now).total_seconds() / 3600.0


# --------------------------------------------------------------------------- step execution

@dataclass
class StepOutcome:
    status: str                                   # ok | failed | timeout | skipped
    summary: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)   # goes to the run state file, not the status file
    error: str | None = None
    note: str | None = None
    seconds: float = 0.0


def _clean(obj: Any) -> Any:
    """JSON-safe: NaN/NaT to None, numpy scalars to Python, dates to text."""
    import math

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except (ValueError, TypeError):
            pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def worker_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        return str(exe.with_name("python.exe"))
    return str(exe)


def run_worker_process(cmd: Sequence[str], timeout: float, *, cwd: Path, env: Mapping[str, str]) -> tuple[int | None, str]:
    """Run ``cmd`` with a hard timeout. Returns ``(returncode or None on timeout, combined output)``."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(list(cmd), capture_output=True, timeout=timeout, cwd=str(cwd), env=dict(env), creationflags=flags)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        return None, out.decode("utf-8", "replace")
    return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace")


class SubprocessRunner:
    """Runs ``snapshot`` in-process (a local file read) and every other step as ``python -m src.ops.daily_refresh --worker``."""

    def __init__(self, settings: Settings, *, process_runner: Callable = run_worker_process):
        self.s = settings
        self._proc = process_runner

    def __call__(self, step: str, timeout: float) -> StepOutcome:
        if step == "snapshot":
            return worker_snapshot(self.s)
        fd, result_name = tempfile.mkstemp(prefix=f"daily_{step}_", suffix=".json")
        os.close(fd)
        result = Path(result_name)
        cmd = [worker_python(), "-m", "src.ops.daily_refresh", "--worker", step, "--result-file", str(result),
               "--season", self.s.season, "--data-dir", str(self.s.data_dir), "--reports-dir", str(self.s.reports_dir),
               "--board-models", ",".join(self.s.board_models)]
        if self.s.offline:
            cmd.append("--offline")
        if self.s.league_id:
            cmd += ["--league-id", str(self.s.league_id)]
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
                tail = " | ".join(line for line in output.splitlines()[-3:] if line.strip())
                return StepOutcome("failed", error=f"worker exited {code} without a result: {tail[:400]}")
            if payload.get("ok"):
                return StepOutcome("ok", payload.get("summary") or {}, payload.get("state") or {}, note=payload.get("note"))
            return StepOutcome("failed", payload.get("summary") or {}, error=payload.get("error") or f"worker exited {code}")
        finally:
            result.unlink(missing_ok=True)


# --------------------------------------------------------------------------- worker bodies (run in a subprocess)

class StepError(RuntimeError):
    pass


def _clients(base: Path, offline: bool):
    from src.ingest import espn_adp
    from src.ingest.http_cache import CachedHttpClient, default_cache_dir
    from src.ingest.nba_client import NBAClient

    off = True if offline else None
    return dict(nba=NBAClient(base / "raw" / "nba_api", offline=off, min_interval=1.0),
                espn=CachedHttpClient(default_cache_dir(espn_adp.CACHE_DIR_NAME), offline=off, min_interval=2.0),
                fp=CachedHttpClient(default_cache_dir(espn_adp.FP_CACHE_DIR_NAME), offline=off, min_interval=5.0))


def _refresh_step(name: str, season: str, base: Path, offline: bool) -> dict[str, Any]:
    from src.ingest import preseason_refresh as pr

    (r,) = pr.run_refresh(season, base, steps=(name,), **_clients(base, offline))
    if not r.ok:
        raise StepError(r.error or f"{name} failed")
    return r.summary


def worker_roster(a: argparse.Namespace) -> tuple[dict, dict]:
    from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

    summary = _refresh_step("roster", a.season, a.data_dir, a.offline)
    snap = latest_snapshot(read_roster_snapshots(a.data_dir))
    summary["snapshot_date"] = str(snap["snapshot_date"].iloc[0].date())
    return summary, {"roster_map": rd.roster_map(snap.to_dict("records"))}


def worker_offseason(a: argparse.Namespace) -> tuple[dict, dict]:
    summary = _refresh_step("offseason", a.season, a.data_dir, a.offline)
    return summary, {"offseason": summary}


def worker_adp(a: argparse.Namespace) -> tuple[dict, dict]:
    return _refresh_step("adp", a.season, a.data_dir, a.offline), {}


def worker_league(a: argparse.Namespace) -> tuple[dict, dict]:
    from src.ingest import espn_league as el

    if not a.league_id:
        raise StepError("no league id configured")

    class FreshClient(el.ESPNLeagueClient):
        """The stock CLI serves its disk cache forever; a daily sync must see today's league state."""

        def get(self, league_id, season_id, views, *, refresh=False, **kw):
            return super().get(league_id, season_id, views, refresh=refresh or not self.offline, **kw)

    client = FreshClient(cache_dir=a.data_dir / "raw" / el.SOURCE, offline=True if a.offline else None)
    argv = ["--league-id", str(a.league_id), "--season", a.season, "--data-dir", str(a.data_dir)]
    if a.offline:
        argv.append("--offline")
    rc = el.main(argv, client=client)
    if rc != 0:
        raise StepError(f"espn_league exited {rc}")
    from src.contracts import season_start

    report = json.loads((a.data_dir / "processed" / "espn_league" / f"{a.league_id}_{season_start(a.season) + 1}.json").read_text("utf-8"))
    rec = report.get("reconciliation") or []
    summary = {"name": report["settings"].get("name"), "teams": report["settings"].get("team_count"),
               "draft_completed": bool(report.get("draft_completed")),
               "draft_in_progress": bool((report.get("draft") or {}).get("in_progress")),
               "n_transactions": len(report.get("transactions") or []),
               "discrepancies": sum(1 for r in rec if not r.get("match")), "fetched_at": report.get("fetched_at")}
    return summary, {"league": summary}


def worker_watchlist(a: argparse.Namespace) -> tuple[dict, dict]:
    from src.value.breakouts import assemble_watchlist

    res = assemble_watchlist(a.season, data_dir=a.data_dir)
    wl = res.watchlist
    a.reports_dir.mkdir(parents=True, exist_ok=True)
    out = a.reports_dir / "watchlist.csv"
    wl.to_csv(out, index=False, encoding="utf-8")
    rows = [{"player_id": r["player_id"], "rank": int(r["watch_rank"]), "name": r["name"], "team": r.get("team"),
             "uplift": r.get("uplift"), "useful_prob": r.get("useful_prob"), "adp": r.get("adp"),
             "evidence": r.get("evidence")} for r in wl.to_dict("records")]
    summary = {"model": res.model, "rows": len(rows), "calibrated": res.calibrated, "notes": res.notes, "csv": str(out)}
    return _clean(summary), _clean({"watchlist": rows, "model": res.model})


def worker_board(a: argparse.Namespace) -> tuple[dict, dict]:
    import pandas as pd

    from src.ingest.nba_offseason import read_offseason_logs
    from src.value import board
    from src.value.breakouts import choose_model

    try:
        logs = read_offseason_logs(a.data_dir)
    except FileNotFoundError:
        logs = None
    models: list[str] = []
    for m in a.board_models.split(","):
        m = choose_model(logs, a.season) if m == "auto" else m
        if m and m not in models:
            models.append(m)
    adp_file = a.data_dir / "processed" / "adp.parquet"
    boards: dict[str, Any] = {}
    errors = []
    for m in models:
        out = a.reports_dir / f"draft_board_{m}.csv"
        argv = ["--season", a.season, "--model", m, "--out", str(out), "--data-dir", str(a.data_dir)]
        if adp_file.exists():
            argv += ["--adp", str(adp_file)]
        rc = board.main(argv)
        if rc != 0:
            errors.append(f"{m}: board exited {rc}")
            continue
        df = pd.read_csv(out)
        boards[m] = {"path": str(out), "rows": len(df),
                     "top": [{"rank": int(r["rank"]), "name": r["name"]} for r in df.sort_values("rank").head(15).to_dict("records")]}
    if errors:
        raise StepError("; ".join(errors))
    return _clean({"boards": boards}), _clean({"boards": boards})


def worker_profiles(a: argparse.Namespace) -> tuple[dict, dict]:
    summary = _refresh_step("profiles", a.season, a.data_dir, a.offline)
    return _clean(summary), {"profiles": _clean(summary)}


def worker_status(a: argparse.Namespace) -> tuple[dict, dict]:
    summary = _refresh_step("status", a.season, a.data_dir, a.offline)
    return _clean(summary), {"status_archive": _clean({"players": summary.get("players"), "by_status": summary.get("by_status")})}


def worker_coaches(a: argparse.Namespace) -> tuple[dict, dict]:
    """Refresh the live season's head coaches (ADR 0020); the state is the team -> coach map the next run diffs against."""
    summary = _refresh_step("coaches", a.season, a.data_dir, a.offline)
    cmap = summary.pop("coaches", {})
    return _clean(summary), _clean({"coach_map": cmap})


def worker_transactions(a: argparse.Namespace) -> tuple[dict, dict]:
    """Refresh the league-transactions ledger, then render what is new this run against the boards this run just built."""
    import pandas as pd

    from src.features import txn_impact as ti
    from src.ingest.espn_transactions import read_ledger

    summary = _refresh_step("transactions", a.season, a.data_dir, a.offline)
    ledger = read_ledger(a.data_dir)
    new = ledger[ledger["first_seen"] == ledger["first_seen"].max()] if summary.get("new_rows") else ledger.iloc[0:0]
    board = None
    for name in ("draft_board_baseline_offseason_debut.csv", "draft_board_baseline.csv"):
        path = a.reports_dir / name
        if path.exists():
            board = pd.read_csv(path)
            break
    roster = None
    try:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        roster = latest_snapshot(read_roster_snapshots(a.data_dir))
    except (FileNotFoundError, OSError):
        pass
    ann = ti.annotate(ti.events(new), board, roster)
    staff = ti.staff_events(new)
    state = {"new_rows": int(summary.get("new_rows") or 0), "ranked": board is not None,
             "lines": ti.digest(ann, relevant_only=board is not None, limit=40),
             "staff": [f"{r.team_abbr} {r.kind} {r.person} ({r.role.replace('_', ' ')})" for r in staff.itertuples(index=False)][:15],
             "ledger_through": str(ledger["txn_date"].max().date())}
    return _clean(summary), _clean({"transactions": state})


WORKERS = {"roster": worker_roster, "offseason": worker_offseason, "adp": worker_adp,
           "profiles": worker_profiles, "status": worker_status, "league": worker_league,
           "watchlist": worker_watchlist, "board": worker_board, "transactions": worker_transactions, "coaches": worker_coaches}


def run_worker(a: argparse.Namespace) -> int:
    """Subprocess entry: run one step and write ``{"ok", "summary", "state", "error"}`` to ``--result-file``."""
    a.data_dir = Path(a.data_dir)
    a.reports_dir = Path(a.reports_dir)
    result: dict[str, Any]
    try:
        summary, state = WORKERS[a.worker](a)
        result = {"ok": True, "summary": _clean(summary), "state": _clean(state)}
    except Exception as exc:                                  # noqa: BLE001 - the parent records it and moves on
        traceback.print_exc()
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(a.result_file).write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


def worker_snapshot(s: Settings) -> StepOutcome:
    """Archive the ESPN payload the ``adp`` step just refreshed (read from its cache; no network)."""
    from src.ingest import espn_adp
    from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir

    t0 = time.monotonic()
    sid = espn_adp.espn_season_id(s.season)
    try:
        client = CachedHttpClient(default_cache_dir(espn_adp.CACHE_DIR_NAME), offline=True)
        raw = espn_adp.fetch_espn_players(client, sid)
        cache = client.cache_path(f"players/{sid}", {"view": "kona_player_info", "limit": 600})
        taken = datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc)
        path, written = rd.write_snapshot(s.data_dir, s.season, raw, taken)
    except (HttpCacheError, espn_adp.EspnAdpError, OSError, ValueError) as exc:
        return StepOutcome("failed", error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - t0)
    summary = {"path": str(path), "written": written, "taken_at": taken.isoformat(), "players": len(raw)}
    return StepOutcome("ok", summary, {"snapshot": str(path), "snapshot_taken_at": taken.isoformat()},
                       note=None if written else "identical to the previous snapshot; not duplicated",
                       seconds=time.monotonic() - t0)


# --------------------------------------------------------------------------- run state, status, history

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_prior_states(runs_dir: Path, exclude: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []:
        if p.stem == exclude:
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out                                                # oldest first


def newest_with(states: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for st in reversed(states):
        if st.get(key) is not None:
            return st
    return None


def read_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def consecutive_failures(history: list[dict[str, Any]]) -> int:
    n = 0
    for row in reversed(history):
        if row.get("ok"):
            break
        n += 1
    return n


def build_diffs(states: list[dict[str, Any]], state: dict[str, Any], s: Settings) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    prev = newest_with(states, "roster_map")
    if prev and state.get("roster_map") is not None:
        diffs["roster"] = rd.roster_diff(prev["roster_map"], state["roster_map"])
    prev = newest_with(states, "snapshot")
    if prev and state.get("snapshot"):
        cur_path, prev_path = Path(state["snapshot"]), Path(prev["snapshot"])
        if cur_path == prev_path:
            diffs["injuries"], diffs["adp"] = [], {"risers": [], "fallers": [], "priced": [], "unpriced": []}
        elif cur_path.exists() and prev_path.exists():
            a, b = rd.read_snapshot(prev_path), rd.read_snapshot(cur_path)
            diffs["injuries"], diffs["adp"] = rd.injury_changes(a, b), rd.adp_moves(a, b)
    prev = newest_with(states, "watchlist")
    if prev and state.get("watchlist") is not None:
        diffs["watchlist"] = rd.watchlist_diff(prev["watchlist"], state["watchlist"])
    if state.get("status_archive") is not None:
        prev_st = newest_with(states, "status_archive")
        cur = state["status_archive"].get("by_status") or {}
        old = (prev_st["status_archive"].get("by_status") or {}) if prev_st else None
        diffs["status_archive"] = {"players": state["status_archive"].get("players"), "by_status": cur,
                                   "changed": None if old is None else {k: [old.get(k, 0), cur.get(k, 0)] for k in sorted({*old, *cur}) if old.get(k, 0) != cur.get(k, 0)}}
    if state.get("transactions") is not None:
        diffs["transactions"] = state["transactions"]
    prev_c = newest_with(states, "coach_map")
    if prev_c and state.get("coach_map") is not None:
        old, cur = prev_c["coach_map"], state["coach_map"]
        from src.ingest.espn_transactions import ESPN_TEAMS

        abbr = {str(tid): a for tid, a in ESPN_TEAMS.values()}
        diffs["coaches"] = [{"team_id": t, "team": abbr.get(t, t), "from": old.get(t), "to": cur.get(t)}
                            for t in sorted(cur) if old.get(t) != cur.get(t)]
    if state.get("offseason") is not None:
        prev_off = newest_with(states, "offseason")
        diffs["games"] = rd.games_delta(prev_off["offseason"] if prev_off else None, state["offseason"])
    return diffs


def latest_roster_date(base: Path) -> str | None:
    try:
        import pandas as pd

        df = pd.read_parquet(base / "processed" / "roster_snapshots.parquet", columns=["snapshot_date"])
        return str(pd.to_datetime(df["snapshot_date"]).max().date())
    except Exception:                                         # noqa: BLE001 - freshness is advisory
        return None


def freshness_and_reasons(s: Settings, state: dict[str, Any], now_utc: datetime, failures: int) -> tuple[dict[str, str], list[str]]:
    fresh: dict[str, str] = {}
    reasons: list[str] = []
    if failures >= s.alert_after:
        reasons.append(f"{failures} consecutive runs had failing steps (alert threshold {s.alert_after})")
    taken = state.get("snapshot_taken_at")
    if taken:
        age = (now_utc - datetime.fromisoformat(taken)).total_seconds() / 3600
        fresh["ESPN player snapshot"] = f"{taken} ({age:.1f} h old)"
        if age > s.stale_hours:
            reasons.append(f"ESPN data is {age:.0f} h old (limit {s.stale_hours:.0f} h): the adp step is not refreshing it")
    else:
        fresh["ESPN player snapshot"] = "none archived this run"
        if "snapshot" in s.steps:
            reasons.append("no ESPN snapshot could be archived this run")
    rdate = latest_roster_date(s.data_dir)
    if rdate:
        age_d = max(0, (now_utc.astimezone().date() - date.fromisoformat(rdate)).days)
        fresh["NBA roster snapshot"] = f"{rdate} ({age_d} day(s) old)"
        if age_d * 24 > s.stale_hours:
            reasons.append(f"the NBA roster snapshot is from {rdate}, {age_d} days old")
    return fresh, reasons


# --------------------------------------------------------------------------- alert + toast

def toast(title: str, message: str) -> None:
    """Best-effort Windows notification via PowerShell; silently does nothing anywhere else or on any error."""
    if sys.platform != "win32":
        return
    t, m = title.replace("'", "''"), message.replace("'", "''")
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
        "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('ToastText02');"
        f"$x.GetElementsByTagName('text')[0].AppendChild($x.CreateTextNode('{t}')) > $null;"
        f"$x.GetElementsByTagName('text')[1].AppendChild($x.CreateTextNode('{m}')) > $null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('NBA Fantasy Daily Refresh')"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($x))")
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout=20,
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        pass


def update_alert(s: Settings, reasons: list[str], run_id: str, notifier: Callable[[str, str], None]) -> bool:
    path = s.reports_dir / "ALERT.txt"
    if not reasons:
        if path.exists():
            path.unlink()
            LOGGER.info("alert cleared: %s removed", path)
        return False
    body = (f"NBA fantasy daily refresh needs attention (run {run_id}, draft {s.draft_date}).\n\n"
            + "\n".join(f"- {r}" for r in reasons)
            + f"\n\nSee reports/daily/latest.md, {s.ops_dir / 'status.json'} and the log in {s.ops_dir / 'logs'}.\n"
            "Re-run by hand: python -m src.ops.daily_refresh --force   (or ./dev schedule run-now).\n"
            "This file is removed automatically by the next fully successful run.\n")
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    atomic_write_text(path, body)
    if previous is None or _reasons_of(previous) != reasons:
        notifier("NBA fantasy refresh needs attention", reasons[0])
    return True


def _reasons_of(text: str) -> list[str]:
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]


# --------------------------------------------------------------------------- logging

def setup_logging(log_dir: Path, *, quiet: bool, logger: logging.Logger | None = None, filename: str = "daily_refresh.log") -> None:
    """Rotating file log (1 MB x 5) plus a console handler. ``logger``/``filename`` let the nightly job (ADR 0018) reuse it."""
    lg = logger or LOGGER
    lg.setLevel(logging.INFO)
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / filename, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except OSError as exc:
        print(f"warning: cannot write the log file: {exc}", file=sys.stderr)
    if sys.stderr is not None:                                # pythonw has no console
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.WARNING if quiet else logging.INFO)
        ch.setFormatter(fmt)
        try:
            sys.stderr.reconfigure(errors="replace")          # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
        lg.addHandler(ch)


# --------------------------------------------------------------------------- orchestration

def execute(s: Settings, runner: Callable[[str, float], StepOutcome], *, now: datetime,
            notifier: Callable[[str, str], None] = toast, lock_factory: Callable[[Path, float], RunLock] | None = None,
            clock: Callable[[], float] = time.monotonic) -> int:
    ops = s.ops_dir
    run_id = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    make_lock = lock_factory or (lambda path, stale: RunLock(path, stale_after=stale))
    lock = make_lock(ops / "lock.json", s.budget_s + 15 * 60)
    try:
        lock.acquire()
    except LockBusy as exc:
        LOGGER.warning("skipped: %s", exc)
        return EXIT_BUSY
    if lock.recovered_stale:
        LOGGER.warning("recovered a stale lock; the old one was kept as %s", lock.recovered_stale)
    started = clock()
    try:
        prior_status = _read_json(ops / "status.json") or {}
        atomic_write_text(ops / "status.json", json.dumps(
            {**prior_status, "run_id": run_id, "outcome": "running", "started_at": now.isoformat(), "pid": os.getpid()}, indent=2))
        outcomes: dict[str, StepOutcome] = {}
        for step in s.steps:
            if step == "league" and not s.league_id:
                outcomes[step] = StepOutcome("skipped", note="no ESPN league id configured (ESPN_LEAGUE_ID, --league-id or league_id in daily_refresh.json)")
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
            except Exception as exc:                          # noqa: BLE001 - one step never stops the others
                LOGGER.exception("%s crashed", step)
                outcome = StepOutcome("failed", error=f"{type(exc).__name__}: {exc}")
            outcome.seconds = outcome.seconds or clock() - t0
            outcomes[step] = outcome
            LOGGER.info("%s: %s (%.0fs)%s", step, outcome.status, outcome.seconds, f" {outcome.error}" if outcome.error else "")
        return _finish(s, outcomes, now, run_id, started, clock, notifier)
    finally:
        lock.release()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finish(s: Settings, outcomes: dict[str, StepOutcome], now: datetime, run_id: str, started: float,
            clock: Callable[[], float], notifier: Callable[[str, str], None]) -> int:
    ops = s.ops_dir
    ran = [o for o in outcomes.values() if o.status != "skipped" or o.error]
    failed = [n for n, o in outcomes.items() if o.status in ("failed", "timeout") or (o.status == "skipped" and o.error)]
    ok_any = any(o.status == "ok" for o in outcomes.values())
    outcome = "ok" if not failed else ("partial" if ok_any else "failed")
    state: dict[str, Any] = {}
    for o in outcomes.values():
        if o.status == "ok":
            state.update(o.state)
    states = load_prior_states(ops / "runs", run_id)
    diffs = build_diffs(states, state, s)
    history = read_history(ops / "history.jsonl") + [{"run_id": run_id, "ok": not failed}]
    failures = consecutive_failures(history)
    now_utc = now.astimezone(timezone.utc)
    fresh, reasons = freshness_and_reasons(s, state, now_utc, failures)
    prev_state = states[-1] if states else None
    finished = datetime.now(timezone.utc)
    league = outcomes.get("league")
    boards = (outcomes["board"].summary.get("boards") if "board" in outcomes and outcomes["board"].status == "ok" else None)
    local_now = now
    run = {
        "run_id": run_id, "date": local_now.strftime("%Y-%m-%d"), "season": s.season, "outcome": outcome,
        "started_at": now.isoformat(), "finished_at": finished.isoformat(), "finished_local": finished.astimezone().strftime("%Y-%m-%d %H:%M"),
        "seconds": clock() - started, "draft_date": s.draft_date.isoformat(), "draft_date_source": s.draft_date_source,
        "days_to_draft": (s.draft_date - local_now.date()).days,
        "hours_to_draft": hours_to_draft(s, now), "draft_start": s.draft_start.isoformat() if s.draft_start else None,
        "prev_run_id": prev_state["run_id"] if prev_state else None, "prev_time": prev_state.get("finished_local") if prev_state else None,
        "steps": [{"name": n, "status": o.status, "seconds": o.seconds, "error": o.error, "note": o.note} for n, o in outcomes.items()],
        "diffs": diffs, "extras": {n: o.summary for n, o in outcomes.items() if n == "profiles" and o.status == "ok"}, "model": state.get("model"), "watchlist_top": (state.get("watchlist") or [])[:10],
        "league": league.summary if league and league.status == "ok" else None, "boards": boards,
        "freshness": fresh, "alert": {"reasons": reasons},
        "files": {"watchlist": str(s.reports_dir / "watchlist.csv"), "status": str(ops / "status.json"), "log": str(ops / "logs" / "daily_refresh.log")},
    }
    report = render_report(run)
    atomic_write_text(s.reports_dir / f"{run['date']}.md", report)
    atomic_write_text(s.reports_dir / "latest.md", report)
    atomic_write_text(s.reports_dir / "runs" / f"{run_id}.md", report)
    run_state = {**state, "run_id": run_id, "finished_local": run["finished_local"], "outcome": outcome}
    atomic_write_text(ops / "runs" / f"{run_id}.json", json.dumps(run_state, default=str))
    with open(ops / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": run_id, "finished_at": run["finished_at"], "ok": not failed, "outcome": outcome, "failed_steps": failed}) + "\n")
    alerting = update_alert(s, reasons, run_id, notifier)
    prior_ok = _read_json(ops / "status.json") or {}
    atomic_write_text(ops / "status.json", json.dumps({
        "run_id": run_id, "outcome": outcome, "ok": not failed, "exit_code": EXIT_OK if not failed else EXIT_FAILED,
        "started_at": now.isoformat(), "finished_at": run["finished_at"], "seconds": run["seconds"], "season": s.season,
        "draft_date": s.draft_date.isoformat(), "draft_date_source": s.draft_date_source, "days_to_draft": run["days_to_draft"],
        "window": [s.window_start.isoformat(), s.window_end.isoformat()], "consecutive_failures": failures,
        "last_success_at": run["finished_at"] if not failed else prior_ok.get("last_success_at"),
        "alert": alerting, "alert_reasons": reasons, "model": state.get("model"),
        "steps": {n: {"status": o.status, "seconds": round(o.seconds, 1), "error": o.error, "summary": _small(o.summary)} for n, o in outcomes.items()},
        "report": str(s.reports_dir / "latest.md")}, indent=2, default=str))
    LOGGER.info("run %s finished: %s%s; report %s", run_id, outcome, f" (failed: {', '.join(failed)})" if failed else "", s.reports_dir / "latest.md")
    del ran
    return EXIT_OK if not failed else EXIT_FAILED


def _small(summary: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(summary, default=str)
    return dict(summary) if len(text) < 1500 else {"truncated": text[:300] + "..."}


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ops.daily_refresh", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default=None, help="the season about to start (default: the live one)")
    p.add_argument("--draft-date", default=None, help="YYYY-MM-DD (default: draft.date in config/league.yaml; or NBA_DRAFT_DATE, or draft_date in <data>/daily_refresh.json)")
    p.add_argument("--draft-start", default=None, help="draft start instant with offset, e.g. 2026-10-16T18:00:00Z (default: draft.start_utc in config/league.yaml; or NBA_DRAFT_START, or draft_start in daily_refresh.json)")
    p.add_argument("--window-start", default=None, help="first day scheduled runs do anything (default: draft date - 30 days)")
    p.add_argument("--window-end", default=None, help="last day scheduled runs do anything (default: draft date + 1 day)")
    p.add_argument("--force", action="store_true", help="run even outside the window")
    p.add_argument("--dry-run", action="store_true", help="print what would run and exit; no lock, no network, no writes")
    p.add_argument("--offline", action="store_true", help="never touch the network (cached data only)")
    p.add_argument("--only", default=None, help=f"comma-separated subset of {list(STEP_ORDER)}")
    p.add_argument("--skip", default=None, help="comma-separated steps to skip")
    p.add_argument("--league-id", type=int, default=None, help="ESPN league id (or ESPN_LEAGUE_ID, or league_id in daily_refresh.json)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--reports-dir", type=Path, default=None, help="default: <repo>/reports/daily (gitignored)")
    p.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--budget-minutes", type=float, default=45.0, help="whole-run time budget")
    p.add_argument("--stale-hours", type=float, default=36.0, help="alert when data is older than this")
    p.add_argument("--alert-after", type=int, default=2, help="alert after this many consecutive failing runs")
    p.add_argument("--board-models", default="baseline,auto,baseline_offseason_debut,baseline_hurdle_adp_offseason_debut", help="projectors for the draft-board CSVs; 'auto' = the watchlist's model; the *_debut boards carry the debutants and undrafted signees, and baseline_hurdle_adp_offseason_debut is the best measured model (ADR 0031/0032)")
    p.add_argument("--quiet", action="store_true", help="console shows warnings only (the log file always has everything)")
    p.add_argument("--worker", choices=sorted(WORKERS), default=None, help=argparse.SUPPRESS)
    p.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    return p


def describe_plan(s: Settings, today: date, now: datetime | None = None) -> str:
    inside = in_window(today, s, now)
    lines = [f"daily refresh for {s.season}: today {today} is {'INSIDE' if inside else 'OUTSIDE'} the window {s.window_start} .. {s.window_end}",
             f"draft date {s.draft_date} ({s.draft_date_source}){_start_text(s, now)}; data {s.data_dir}; reports {s.reports_dir}",
             f"steps: {', '.join(s.steps)}; league id: {s.league_id or 'not configured (league step skipped)'}; offline: {s.offline}",
             f"budget {s.budget_s / 60:.0f} min; timeouts {', '.join(f'{k} {int(v)}s' for k, v in s.timeouts.items() if k in s.steps)}"]
    return "\n".join(lines)


def _start_text(s: Settings, now: datetime | None) -> str:
    if s.draft_start is None:
        return ""
    from zoneinfo import ZoneInfo

    loc = s.draft_start.astimezone(ZoneInfo(s.draft_tz))
    txt = f", start {s.draft_start.astimezone(timezone.utc):%Y-%m-%d %H:%M}Z = {loc:%Y-%m-%d %H:%M} {s.draft_tz}"
    h = hours_to_draft(s, now) if now else None
    return txt + ("" if h is None else f", {h:.1f} h from now" if h > 0 else ", already started")


def main(argv: Sequence[str] | None = None, *, runner: Callable[[str, float], StepOutcome] | None = None,
         now: datetime | None = None, notifier: Callable[[str, str], None] = toast, env: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not args.result_file or not args.data_dir:
            print("error: --worker needs --result-file and --data-dir", file=sys.stderr)
            return EXIT_CONFIG
        args.league_id = args.league_id or None
        args.reports_dir = args.reports_dir or REPO_ROOT / "reports" / "daily"
        return run_worker(args)
    stamp = now or datetime.now().astimezone()
    try:
        s = resolve_settings(args, os.environ if env is None else env, stamp.date())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if args.dry_run:
        print(describe_plan(s, stamp.date(), stamp))
        return EXIT_OK
    setup_logging(s.ops_dir / "logs", quiet=args.quiet)
    LOGGER.info("start: %s", describe_plan(s, stamp.date(), stamp).replace("\n", " | "))
    if not args.force and not in_window(stamp.date(), s, stamp):
        LOGGER.info("outside the window %s .. %s (or at/after the draft start %s): nothing to do", s.window_start, s.window_end, s.draft_start)
        return EXIT_OK
    code = execute(s, runner or SubprocessRunner(s), now=stamp, notifier=notifier)
    if sys.stdout is not None:
        try:
            print(f"daily refresh: exit {code}; report {s.reports_dir / 'latest.md'}")
        except UnicodeEncodeError:
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
