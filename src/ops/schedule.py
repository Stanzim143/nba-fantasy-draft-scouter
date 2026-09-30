"""Register the daily refresh with Windows Task Scheduler (per user, no admin) (ADR 0014).

    python -m src.ops.schedule install  [--times 07:30,19:00] [--final-runs-before-hours 15.5,1.75] [--draft-date ...] [--dry-run]
    python -m src.ops.schedule status | remove | run-now [--job daily|nightly|all] [--direct]

``status`` with no ``--job`` prints BOTH jobs (daily and nightly); every other action defaults to ``daily``.

(also ``./dev schedule ...``). The task runs the venv's ``pythonw.exe -m src.ops.daily_refresh`` from the PRIMARY
checkout (main), so it survives worktree removal. It is generated as Task Scheduler XML and registered with
``schtasks /Create /XML``: an interactive-token (run only while you are logged on, no password, no elevation) task with two
daily triggers, ``StartWhenAvailable`` (catch up after sleep), run only if a network is available, one instance at a
time, a one hour execution limit, and an ``EndBoundary`` after which the task stops firing on its own. The draft date and
start instant default to ``draft.date`` / ``draft.start_utc`` in config/league.yaml; the daily triggers end AT the draft start
and extra one-time "final pre-draft" triggers are placed ``--final-runs-before-hours`` before it (ADR 0014). The script has
the same window as an in-code gate, so even a stray invocation after the window does nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape

TASK_NAME = "NBA Fantasy 2026 Daily Refresh"
DEFAULT_TIMES = "07:30,19:00"     # machine-local; about when US evening games and their injury news have landed (ADR 0014)
DEFAULT_FINAL_HOURS = "15.5,1.75"   # hours before the draft start: one after the last useful US preseason games, one ~105 min before
NIGHTLY_TASK_NAME = "NBA Fantasy 2026 Nightly In-Season"
# Machine-local (Pacific/Auckland, UTC+13 from late September to early April). NBA games are dated in US Eastern time; the
# last tip-offs finish about 03:30 ET, which is 20:30 local. 09:30 local is 16:30 ET on the previous US day: the day before's
# games have been final for 13 hours and that evening's have not started, so the run sees the newest complete game day and the
# freshest overnight injury news before the day's lineups lock (ADR 0018). Override with --times.
NIGHTLY_TIMES = "09:30"
JOBS = {
    "daily": {"task": TASK_NAME, "module": "src.ops.daily_refresh", "times": DEFAULT_TIMES, "limit": "PT1H", "reports": "daily", "ops": "daily_refresh",
              "what": "Daily pre-draft refresh of NBA data, ESPN snapshot, breakout watchlist and draft board (ADR 0014)"},
    "nightly": {"task": NIGHTLY_TASK_NAME, "module": "src.ops.nightly", "times": NIGHTLY_TIMES, "limit": "PT90M", "reports": "nightly", "ops": "nightly",
                "what": "In-season nightly job: game-log ingest, ESPN league sync, rest-of-season projection, waiver/streaming/lineup recommendations (ADR 0018)"},
}


class ScheduleError(RuntimeError):
    pass


def parse_times(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        part = part.strip()
        try:
            t = datetime.strptime(part, "%H:%M")
        except ValueError as exc:
            raise ScheduleError(f"bad time {part!r}; expected HH:MM, comma separated") from exc
        out.append(t.strftime("%H:%M"))
    if not out:
        raise ScheduleError("no times given")
    return out


def parse_hours(text: str) -> list[float]:
    """``"15.5,1.75"`` -> [15.5, 1.75]; ``"none"``/empty -> []."""
    if text.strip().lower() in ("", "none", "off"):
        return []
    try:
        vals = [float(p) for p in text.split(",")]
    except ValueError as exc:
        raise ScheduleError(f"bad --final-runs-before-hours {text!r}; expected e.g. 15.5,1.75 or none") from exc
    if any(v <= 0 for v in vals):
        raise ScheduleError("--final-runs-before-hours values must be positive")
    return sorted(vals, reverse=True)


def final_run_times(draft_start: datetime, hours: Sequence[float], tz: tzinfo, now: datetime | None = None) -> list[datetime]:
    """Naive machine-local run times ``h`` hours before the draft start (rounded to the minute), dropping those already past."""
    out = []
    for h in hours:
        t = (draft_start - timedelta(hours=h)).astimezone(tz)
        t = (t + timedelta(seconds=30)).replace(second=0, microsecond=0)
        if now is None or t > now.astimezone(tz):
            out.append(t.replace(tzinfo=None))
    return sorted(set(out))


def build_arguments(data_dir: Path, draft_date: date | None, extra: Sequence[str] = (), module: str = "src.ops.daily_refresh",
                    draft_start: datetime | None = None) -> str:
    args = ["-m", module, "--quiet", "--data-dir", f'"{data_dir}"']
    if draft_date:
        args += ["--draft-date", draft_date.isoformat()]
    if draft_start:
        args += ["--draft-start", draft_start.isoformat().replace("+00:00", "Z")]
    args += list(extra)
    return " ".join(args)


def build_task_xml(*, python_exe: Path, repo: Path, data_dir: Path, times: Sequence[str], start: date, end: date | datetime,
                   draft_date: date | None, author: str = "nba-fantasy-2026", job: str = "daily",
                   final_runs: Sequence[datetime] = (), draft_start: datetime | None = None) -> str:
    """Task Scheduler 1.2 XML (returned as text; written UTF-16 by :func:`register`).

    ``end`` is a date (end of that day) or a naive machine-local datetime (exactly then). ``final_runs`` are naive
    machine-local datetimes that get an extra one-time trigger each.
    """
    end_text = end.strftime("%Y-%m-%dT%H:%M:%S") if isinstance(end, datetime) else f"{end.isoformat()}T23:59:59"
    if end_text < f"{start.isoformat()}T00:00:00":
        raise ScheduleError(f"end {end_text} is before start date {start}")
    spec = JOBS[job]
    triggers = "\n".join(f"""    <CalendarTrigger>
      <StartBoundary>{start.isoformat()}T{t}:00</StartBoundary>
      <EndBoundary>{end_text}</EndBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>""" for t in times)
    if final_runs:
        triggers += "\n" + "\n".join(f"""    <TimeTrigger>
      <StartBoundary>{f.strftime("%Y-%m-%dT%H:%M:%S")}</StartBoundary>
      <EndBoundary>{end_text}</EndBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>""" for f in final_runs)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>{escape(author)}</Author>
    <Description>{escape(spec['what'])}. Active {start} to {end_text}. Report: {escape(str(repo / 'reports' / spec['reports'] / 'latest.md'))}</Description>
  </RegistrationInfo>
  <Triggers>
{triggers}
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>{spec['limit']}</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(python_exe))}</Command>
      <Arguments>{escape(build_arguments(data_dir, draft_date, module=spec['module'], draft_start=draft_start))}</Arguments>
      <WorkingDirectory>{escape(str(repo))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


# --------------------------------------------------------------------------- environment discovery

def venv_python(windowless: bool = True) -> Path:
    venv = Path(os.environ.get("NBA_VENV") or Path.home() / ".venvs" / "nba-fantasy-2026")
    for name in (("pythonw.exe", "python.exe") if windowless else ("python.exe",)):
        p = venv / "Scripts" / name
        if p.exists():
            return p
    for cand in (venv / "bin" / "python",):
        if cand.exists():
            return cand
    return Path(sys.executable)


def primary_checkout(start: Path | None = None) -> Path:
    """The primary worktree (first entry of ``git worktree list``), which is on the integration branch."""
    here = start or Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "worktree", "list", "--porcelain"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScheduleError(f"cannot locate the primary checkout via git: {exc}") from exc
    for line in out.splitlines():
        if line.startswith("worktree "):
            return Path(line[len("worktree "):])
    raise ScheduleError("git reported no worktrees")


def machine_tz(name: str, injected: tzinfo | None = None) -> tzinfo:
    """DST-aware zone for turning UTC instants into the machine-local wall clock the triggers use.

    ``datetime.astimezone()`` would give a FIXED offset (today's), wrong across a DST change (NZ moves to NZDT on
    2026-09-27, after the install date). So use the named zone, and refuse if the machine's current UTC offset disagrees.
    """
    if injected is not None:
        return injected
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(name)
    real = datetime.now().astimezone().utcoffset()
    if datetime.now(tz).utcoffset() != real:
        raise ScheduleError(f"this machine's UTC offset ({real}) differs from {name} ({datetime.now(tz).utcoffset()}); "
                            "pass --tz <the machine's IANA zone>")
    return tz


def resolve_draft_window(a: argparse.Namespace, data_dir: Path, repo: Path, today: date, injected_tz: tzinfo | None, now: datetime | None
                         ) -> tuple[date, date | datetime, date | None, datetime | None, list[datetime]]:
    """(window start, trigger end, baked draft date, baked draft start, final-run times) for the daily job.

    Draft date/start: flag > env > ``daily_refresh.json`` > config/league.yaml (same as the run itself). The trigger end is
    the draft START in machine-local time when it is known (so nothing fires after the draft), else the end of the draft
    date + 1 day; an explicit --end-date wins. Only a flag/env value is baked into the task's arguments; otherwise the run
    reads the config itself, so the config stays the single source.
    """
    from src.ops.daily_refresh import FALLBACK_DRAFT_DATE, WINDOW_DAYS_AFTER, ConfigError, resolve_draft

    try:
        local = json.loads((data_dir / "daily_refresh.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local = {}
    date_txt = a.draft_date or os.environ.get("NBA_DRAFT_DATE")
    start_txt = a.draft_start or os.environ.get("NBA_DRAFT_START")
    try:
        draft, dstart, tzname, _src = resolve_draft(a.draft_date, a.draft_start, os.environ, local if isinstance(local, dict) else {}, repo)
        baked_date = date.fromisoformat(date_txt) if date_txt else None
        baked_start = datetime.fromisoformat(start_txt.replace("Z", "+00:00")) if start_txt else None
    except (ConfigError, ValueError) as exc:
        raise ScheduleError(str(exc)) from exc
    tz = machine_tz(a.tz or tzname, injected_tz)
    now = now or datetime.now(tz)
    start = date.fromisoformat(a.start_date) if a.start_date else today
    if a.end_date:
        end: date | datetime = date.fromisoformat(a.end_date)
    elif dstart is not None:
        end = dstart.astimezone(tz).replace(tzinfo=None)
    else:
        end = (draft or FALLBACK_DRAFT_DATE) + timedelta(days=WINDOW_DAYS_AFTER)
    finals = final_run_times(dstart, parse_hours(a.final_runs_before_hours), tz, now) if dstart else []
    return start, end, baked_date, baked_start, finals


# --------------------------------------------------------------------------- registration

Runner = Callable[[list[str]], "tuple[int, str]"]


def default_runner(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def register(xml: str, task_name: str, runner: Runner, workdir: Path, filename: str = "daily_refresh_task.xml") -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / filename
    path.write_bytes(b"\xff\xfe" + xml.encode("utf-16-le"))     # schtasks wants UTF-16 with a BOM
    rc, out = runner(["schtasks", "/Create", "/TN", task_name, "/XML", str(path), "/F"])
    if rc != 0:
        raise ScheduleError(f"schtasks /Create failed ({rc}): {out}")


STATUS_PS = (
    "$t = Get-ScheduledTask -TaskName '{n}' -ErrorAction Stop; $i = $t | Get-ScheduledTaskInfo;"
    "[pscustomobject]@{{State=[string]$t.State; LastRunTime=$i.LastRunTime; LastTaskResult=$i.LastTaskResult; NextRunTime=$i.NextRunTime;"
    "NumberOfMissedRuns=$i.NumberOfMissedRuns; Triggers=@($t.Triggers | ForEach-Object {{ $_.StartBoundary + ' -> ' + $_.EndBoundary }});"
    "Action=($t.Actions | ForEach-Object {{ $_.Execute + ' ' + $_.Arguments }}); WorkingDirectory=($t.Actions | Select-Object -First 1).WorkingDirectory}}"
    " | ConvertTo-Json -Compress")


def query_task(task_name: str, runner: Runner) -> dict | None:
    rc, out = runner(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", STATUS_PS.format(n=task_name.replace("'", "''"))])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def describe_health(data_dir: Path, reports_dir: Path, now: datetime | None = None, ops: str = "daily_refresh") -> list[str]:
    now = now or datetime.now().astimezone()
    lines = []
    status = data_dir / ops / "status.json"
    try:
        st = json.loads(status.read_text(encoding="utf-8"))
        finished = datetime.fromisoformat(st["finished_at"]) if st.get("finished_at") else None
        age = f", {(now - finished).total_seconds() / 3600:.1f} h ago" if finished else ""
        lines.append(f"last run: {st.get('outcome')} (run {st.get('run_id')}{age}); consecutive failing runs: {st.get('consecutive_failures', 0)}")
        bad = [f"{k}={v['status']}" for k, v in (st.get("steps") or {}).items() if v.get("status") not in ("ok", "skipped")]
        if bad:
            lines.append("steps not ok: " + ", ".join(bad))
        rep = st.get("report")
        if not rep:
            lines.append("report: none recorded")
        else:
            rp = Path(rep)
            if not rp.is_absolute():
                rp = reports_dir.parents[1] / rp          # recorded relative to the checkout that ran (reports/<job>/latest.md)
            lines.append(f"report: {rep}" if rp.exists()
                         else f"report: {rep} -- REPORT FILE MISSING at {rp} (the run may have come from another checkout or worktree)")
    except (OSError, ValueError, KeyError):
        lines.append(f"no status file yet at {status}")
    if (reports_dir / "ALERT.txt").exists():
        lines.append(f"ALERT present: {reports_dir / 'ALERT.txt'}")
    stale = sorted((data_dir / "daily_refresh").glob("lock.json.stale-*"))
    if stale:
        lines.append(f"{len(stale)} stale lock(s) were recovered and kept, e.g. {stale[-1].name}")
    return lines


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ops.schedule", description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["install", "status", "remove", "run-now"])
    p.add_argument("--job", choices=sorted(JOBS) + ["all"], default=None,
                   help="daily = pre-draft refresh (ADR 0014); nightly = in-season job (ADR 0018); all = both. "
                        "Default: `status` shows both jobs, every other action uses daily")
    p.add_argument("--times", default=None, help=f"daily run times, machine-local HH:MM list (default {DEFAULT_TIMES} for daily, {NIGHTLY_TIMES} for nightly)")
    p.add_argument("--draft-date", default=None, help="YYYY-MM-DD; default: draft.date in config/league.yaml (also see NBA_DRAFT_DATE)")
    p.add_argument("--draft-start", default=None, help="draft start instant, e.g. 2026-10-16T18:00:00Z; default: draft.start_utc in config/league.yaml (also NBA_DRAFT_START)")
    p.add_argument("--tz", default=None, help="IANA zone of this machine's clock (default: draft.timezone from config, Pacific/Auckland)")
    p.add_argument("--final-runs-before-hours", default=DEFAULT_FINAL_HOURS,
                   help=f"extra one-time daily-job runs this many hours before the draft start (default {DEFAULT_FINAL_HOURS}; 'none' disables)")
    p.add_argument("--start-date", default=None, help="first day the triggers fire (default: today)")
    p.add_argument("--end-date", default=None, help="last day, end of day (default: the draft start instant when known, else draft date + 1)")
    p.add_argument("--repo", type=Path, default=None, help="checkout to run from (default: the primary checkout)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--task-name", default=None, help="default depends on --job")
    p.add_argument("--season", default=None, help="nightly: season whose window sets the trigger dates (default: the live one)")
    p.add_argument("--dry-run", action="store_true", help="install: print the XML and the command, register nothing")
    p.add_argument("--direct", action="store_true", help="run-now: run in this console with --force instead of via Task Scheduler")
    return p


def main(argv: Sequence[str] | None = None, *, runner: Runner = default_runner, today: date | None = None,
         python_exe: Path | None = None, out: Callable[[str], None] = print, local_tz: tzinfo | None = None,
         now: datetime | None = None) -> int:
    a = build_parser().parse_args(argv)
    if runner is default_runner and sys.platform != "win32" and not (a.action == "install" and a.dry_run):
        print("error: `schedule` manages Windows Task Scheduler entries only. On macOS/Linux run the same\n"
              "jobs from cron or systemd, e.g. `30 9 * * * cd /path/to/repo && .venv/bin/python -m src.ops.nightly`\n"
              "(daily pre-draft job: `python -m src.ops.daily_refresh`). See docs/inseason.md.", file=sys.stderr)
        return 2
    if a.job is None:
        a.job = "all" if a.action == "status" else "daily"
    if a.job == "all" and a.action != "status":
        print("error: --job all is only valid for `status`", file=sys.stderr)
        return 2
    today = today or date.today()
    from src.contracts import data_dir as default_data_dir

    data_dir = a.data_dir or default_data_dir()
    if a.job == "all":
        try:
            repo = a.repo or primary_checkout()
        except (ScheduleError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        rcs = []
        for name in sorted(JOBS):
            out(f"== {name}: {JOBS[name]['task']}")
            rcs.append(_status(name, JOBS[name]["task"] if a.task_name is None else a.task_name, repo, data_dir, runner, out))
        return 0 if all(r == 0 for r in rcs) else 1
    spec = JOBS[a.job]
    task_name = a.task_name or spec["task"]
    try:
        repo = a.repo or primary_checkout()
        if a.action == "install":
            times = parse_times(a.times or spec["times"])
            draft = dstart = None
            finals: list[datetime] = []
            if a.job == "nightly":
                start, end = nightly_window(data_dir, a.season, today, a.start_date, a.end_date)
            else:
                start, end, draft, dstart, finals = resolve_draft_window(a, data_dir, repo, today, local_tz, now)
            py = python_exe or venv_python()
            xml = build_task_xml(python_exe=py, repo=repo, data_dir=data_dir, times=times, start=start, end=end, draft_date=draft, job=a.job,
                                 final_runs=finals, draft_start=dstart)
            extra = f" plus final runs at {', '.join(f.strftime('%Y-%m-%d %H:%M') for f in finals)}" if finals else ""
            if a.dry_run:
                out(f"# would register task {task_name!r}: {py} {build_arguments(data_dir, draft, module=spec['module'], draft_start=dstart)}\n# in {repo}, "
                    f"daily at {', '.join(times)} from {start} to {end}{extra}\n{xml}")
                return 0
            module_file = repo / (spec["module"].replace(".", "/") + ".py")
            if not module_file.exists():
                raise ScheduleError(f"{repo} does not contain {module_file.relative_to(repo)}; is the feature merged into that checkout?")
            register(xml, task_name, runner, data_dir / spec["ops"], f"{spec['ops']}_task.xml")
            out(f"registered {task_name!r}: daily at {', '.join(times)} (local time), {start} to {end}{extra}, running {py} from {repo}")
            return 0
        if a.action == "remove":
            rc, text = runner(["schtasks", "/Delete", "/TN", task_name, "/F"])
            out(text or ("removed" if rc == 0 else "failed"))
            return 0 if rc == 0 else 1
        if a.action == "run-now":
            if a.direct:
                return subprocess.call([str(venv_python(False)), "-m", spec["module"], "--force", "--data-dir", str(data_dir)], cwd=str(repo))
            rc, text = runner(["schtasks", "/Run", "/TN", task_name])
            out(text or (f"started; follow reports/{spec['reports']}/latest.md and the log" if rc == 0 else "failed"))
            return 0 if rc == 0 else 1
        return _status(a.job, task_name, repo, data_dir, runner, out)
    except (ScheduleError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _status(job: str, task_name: str, repo: Path, data_dir: Path, runner: Runner, out: Callable[[str], None]) -> int:
    spec = JOBS[job]
    info = query_task(task_name, runner)
    if info is None:
        out(f"task {task_name!r} is not registered (install it with: ./dev schedule install --job {job})")
    else:
        out(json.dumps(info, indent=2))
    for line in describe_health(data_dir, repo / "reports" / spec["reports"], ops=spec["ops"]):
        out(line)
    return 0 if info is not None else 1


def nightly_window(data_dir: Path, season: str | None, today: date, start_date: str | None, end_date: str | None) -> tuple[date, date]:
    """Trigger window of the nightly task: the job's own season window (opening night .. fantasy final day + grace)."""
    from src.ops import nightly as nt

    argv = ["--data-dir", str(data_dir)] + (["--season", season] if season else [])
    if start_date:
        argv += ["--window-start", start_date]
    if end_date:
        argv += ["--window-end", end_date]
    now = datetime.combine(today, datetime.min.time()).astimezone()
    s = nt.resolve_settings(nt.build_parser().parse_args(argv), os.environ, now)
    return s.window_start, s.window_end


if __name__ == "__main__":
    raise SystemExit(main())
