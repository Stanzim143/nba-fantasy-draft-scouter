# ADR 0014: Daily refresh automation and its Windows scheduling

**Status:** accepted, 2026-09-24
**Code:** `src/ops/daily_refresh.py` (orchestrator, CLI, step workers), `src/ops/refresh_diff.py` (ESPN snapshot archive and diffs),
`src/ops/daily_report.py` (report), `src/ops/runlock.py` (lock), `src/ops/schedule.py` (Task Scheduler), `./dev schedule`;
tests in `tests/ops/`.
**Builds on:** ADR 0005 (our own daily ESPN snapshots are the only point-in-time ADP and injury source), ADR 0007 (ADP ingest),
ADR 0008 (read-only league sync), ADR 0012 (`preseason_refresh`, breakout watchlist).

## Context

The draft is on 2026-10-17 in the morning New Zealand time, i.e. the evening of 2026-10-16 in central Europe (set 2026-09-26; before that it was "Oct 17 or 18", and the first version of this ADR assumed the latest, Oct 18). See the dated note at the end. The preseason starts in early October and the breakout
watchlist only becomes trustworthy once preseason games exist (ADR 0012), so the data has to be refreshed repeatedly in the last
weeks, including by nobody at the keyboard. ADR 0005 also requires that we archive ESPN's player universe ourselves from now on:
ESPN wipes prior seasons and publishes no history, so a snapshot we do not take today is gone.

## Decision

1. **One unattended command**, `python -m src.ops.daily_refresh`, runs in order `roster`, `offseason`, `adp` (the existing
   `preseason_refresh` steps), `profiles` and `status` (the ADR 0016 extras: player profiles/draft combine and the dated
   `espn_status_snapshots` archive; added later so the risk overlay accrues history without a manual `--with-extras` run), `snapshot` (dated archive), `league` (read-only ESPN league sync, skipped cleanly when no league id
   is configured), `watchlist` (auto model selection) and `board` (CSVs for `baseline`, the information-matched model and, since 2026-09-25, `baseline_offseason_debut`: the debut board is the one that contains the debutants and undrafted signees, which the plain `baseline` and `auto` boards omit by design, ADR 0016), then, since 2026-09-25, `transactions` (ADR 0017: the league-transactions ledger and a report section ranked against this run's boards, so it runs after `board`), then `coaches` (ADR 0020: the live season's head coaches from Wikipedia's current-coaches list, one request; the report lists changes since the previous run).
2. **Snapshot archive**: `<data>/archive/espn_players/<season>/<UTC stamp>.json.gz`, one compact record per player (id, name, team,
   injury status, ADP, percent owned and started, last news time). A run whose payload is identical to the newest snapshot does not
   write a duplicate; nothing is ever overwritten or deleted. Never committed (the data dir is outside the repo).
3. **Report**: `reports/daily/<date>.md`, `latest.md` and `reports/daily/runs/<run id>.md` (the directory is gitignored). It diffs
   against the previous run: roster moves, injury status changes, ADP risers and fallers (2+ picks), players newly on or dropped from
   the watchlist, new preseason games, step failures, data freshness, and a draft countdown. Each run stores a small state file
   (`<data>/daily_refresh/runs/<run id>.json`); a section is diffed against the newest earlier run that has it, so one failed step
   does not blind the next report.
4. **Operational guarantees.**
   * Single instance: `<data>/daily_refresh/lock.json` created with `O_EXCL`. A lock whose owner pid is gone, or that is older than
     budget plus 15 minutes, is *renamed* to `lock.json.stale-<stamp>` (kept as evidence, reported by `schedule status`), never
     silently deleted. A live lock makes the second invocation exit 3.
   * Each step except `snapshot` runs in its own subprocess with a timeout (a hung stats.nba.com call is killed, not waited for), inside a
     45 minute whole-run budget. A failing, crashing or timed-out step never stops the others; the report says which data is stale.
   * Politeness is inherited: the NBA client (1 s), ESPN (2 s) and FantasyPros (5 s) minimum intervals and caches. One run makes about
     one ESPN players request, four league requests and the NBA roster/Summer League/preseason pulls.
   * Structured `<data>/daily_refresh/status.json` (outcome, per-step status and seconds, consecutive failures, last success,
     alert reasons), `history.jsonl`, and a rotating log (1 MB x 5) at `<data>/daily_refresh/logs/daily_refresh.log`.
   * Exit codes: 0 ok (or outside the window), 1 a step failed, 2 configuration error, 3 lock busy.
   * **Window gate**: `--window-start`, `--window-end`, `--draft-date` (or `NBA_DRAFT_DATE`, or `draft_date` in the optional untracked
     `<data>/daily_refresh.json`). Default window is draft date minus 30 days to draft date plus 1 day. Since the 2026-09-26 note the
     default source of the date is `draft.date` in `config/league.yaml`; only if that too is absent is 2026-10-18 assumed, and the
     report says so. Outside the window a scheduled invocation exits 0 having done nothing; `--force` overrides.
   * **Alerts**: `reports/daily/ALERT.txt` plus a best-effort Windows toast (PowerShell, never fatal) when 2 consecutive runs have
     failing steps, or the ESPN snapshot or NBA roster snapshot is older than 36 hours. The next fully healthy run removes the file.
5. **Scheduling on this machine: Windows Task Scheduler, per user, no admin.** `./dev schedule install|status|remove|run-now`
   (`src/ops/schedule.py`) generates Task Scheduler XML and registers it with `schtasks /Create /XML`: interactive-token principal
   with least privilege (runs while the user is logged on; no password stored), two daily triggers (default 07:30 and 19:00
   machine-local, `--times` to change), `StartWhenAvailable` (catch up after sleep), `RunOnlyIfNetworkAvailable`, one instance,
   1 hour limit, and an `EndBoundary` on the triggers (originally draft date plus one day; since the 2026-09-26 note the draft start
   instant) after which the task stops firing by itself. The
   action is the shared venv's `pythonw.exe -m src.ops.daily_refresh --quiet --data-dir <dir> [--draft-date D]` with the working
   directory set to the **primary checkout** (`main`), so it keeps working after task worktrees are removed. `--dry-run` prints the XML.
   Re-running `install` (for example once the draft date is known) replaces the task.
   The default times assume the machine's clock is in Auckland, where US evening preseason games end in the local late afternoon:
   the 07:30 run catches the previous day's news and the 19:00 run the just-finished games.

## Why not GitHub Actions cron

Rejected. stats.nba.com blocks datacenter IPs (ADR 0002/0005), so the roster and box-score pulls would fail from a hosted runner;
ESPN's terms restrict our use to personal, local access (ADR 0005, accepted by the user); and the point-in-time archive and raw pulls
must never be committed, which is what a hosted runner would need to persist state. Cron on the user's own machine has none of
these problems, at the price of running only while the computer is on and the user is logged in (`StartWhenAvailable` covers sleep).

## Consequences

* The user has to do far less by hand before the draft; what remains is listed in `docs/offseason.md`.
* Steps run as subprocesses, so step summaries cross a small JSON boundary (`--worker`, `--result-file`); that is internal.
* The league step forces a network refresh of the four league requests (the stock `espn_league` CLI serves its disk cache forever)
  by subclassing the client inside the worker; `espn_league.py` is untouched. The two standing config "discrepancies" it reports
  are the known `matchup_limit` and `trades.deadline` unit/timezone caveats of ADR 0008.
* Not built: in-season nightly projections and rest-of-season rankings, trade/waiver tools, schedule awareness (roadmap phase 5).
  The same command and task keep the archive and reports going in-season if `--window-end` / `install --end-date` are extended.
* Ownership: `src/ops/` and `tests/ops/` belong to this track. Only `dev` (a `schedule` subcommand) and docs were touched outside it.
* Limits: a run does nothing while the machine is off or the user is logged out; the alert is only raised by a run that happens, so
  `./dev schedule status` (which prints the last run's age and any ALERT) is the check for a task that stopped firing.

## Note 2026-09-26: the draft date and time are known

The user reported: the draft is on **2026-10-17 in the morning, New Zealand time**, which is the **evening of 2026-10-16 in CET
(CEST, UTC+2)**. The clock time is not given; the consistent window is about 18:00-20:00 UTC on 2026-10-16. We plan against the
**earliest** plausible start, **2026-10-16T18:00Z = 2026-10-17 07:00 NZDT = 2026-10-16 20:00 CEST** (the latest, 20:00Z, is 09:00
NZDT / 22:00 CEST). The user authorised setting it in the shared `config/league.yaml`, done in the task that carries this note
(`draft.date: 2026-10-17`, `draft.timezone: Pacific/Auckland`, `draft.start_utc: "2026-10-16T18:00:00Z"`; the extra keys are
additive, nothing else reads the block except `espn_league`'s four setting comparisons).

Time zone arithmetic (checked in `tests/ops/test_ops_draft_time.py`):

* NZ changes to NZDT (UTC+13) on Sunday 2026-09-27 02:00 (= 2026-09-26 14:00Z); on 2026-09-26 the machine is still on NZST (UTC+12).
  Anything computed on 2026-09-26 with the machine's *current* offset would put October triggers an hour early, so `install` converts
  with the named zone (`draft.timezone`, or `--tz`) and refuses if the machine's current UTC offset disagrees with it.
* Central Europe stays on CEST (UTC+2) until Sunday 2026-10-25 01:00Z, after the draft; the evening of 2026-10-16 is CEST.
* 2026-10-16T18:00Z = 07:00 NZDT 2026-10-17 = 20:00 CEST 2026-10-16.

Decisions:

1. **Resolution order** for the draft date and the start instant (`src.ops.daily_refresh.resolve_draft`): flag (`--draft-date`,
   `--draft-start`) > environment (`NBA_DRAFT_DATE`, `NBA_DRAFT_START`) > `<data>/daily_refresh.json` (`draft_date`, `draft_start`) >
   `config/league.yaml` > fallback 2026-10-18. A start instant needs an offset or `Z`. A config start whose NZ calendar date disagrees
   with a date from a higher source is ignored (date-only mode); an explicit start overrides a config date. The local
   `daily_refresh.json` on this machine carries only `league_id` (no stale draft date), and a test asserts it stays that way.
2. **The run gate ends at the draft.** With a known start instant a scheduled run at or after it exits 0 doing nothing (the date
   window alone would still allow the draft day); `--force` overrides. Reports say "draft in N day(s)" / "DRAFT DAY" and the hours
   to the start.
3. **Schedule.** The last US preseason games that can matter are the evening (US) of 2026-10-15: east-coast games end about
   22:00 EDT = 15:00 NZDT on 2026-10-16, west-coast ones about 22:30 PDT = 18:30 NZDT. Games on 2026-10-16 (US) end after the
   earliest start. So, in NZDT: the daily 07:30 and 19:00 runs continue until the end; **final pre-draft runs** are added as one-time
   triggers `--final-runs-before-hours 15.5,1.75` before the start (default; `none` disables): **2026-10-16 15:30** (east-coast
   games final) and **2026-10-17 05:15** (105 minutes before the earliest start, so a slow run of up to an hour still finishes, and
   the 19:00 run of the 16th has already caught the west-coast games). The trigger end boundary is the draft start
   (2026-10-17T07:00 NZDT), which covers 05:15 and excludes the 07:30 run of the 17th. Task times are machine-local wall clock; the
   default hours are relative to the start, so a corrected `draft.start_utc` plus re-running `install` moves the final runs.
4. **Reliability.** The task keeps `StartWhenAvailable` (a missed run fires when the machine is next available and the user is
   logged on) and `WakeToRun` stays false: waking a machine is governed by the power plan's wake-timer setting, not by this repo,
   and a surprise wake at 05:15 is not worth it. The machine must be on, awake and the user logged in around 15:30 and 05:15 NZDT on
   the 16th/17th; if it sleeps overnight, wake it before the draft and run `./dev schedule run-now --job daily` (about 20-30 minutes).

## Note 2026-10-02: the clock time is confirmed

The draft start is confirmed as **2026-10-17 07:00 NZDT = 2026-10-16T18:00Z = 20:00 CEST**, i.e. exactly the instant the schedule
already planned against (the earlier "earliest plausible" start). `draft.start_utc` is unchanged, so the installed triggers and the
final pre-draft runs (15:30 and 05:15 NZDT) stand and no reinstall is needed; only the "earliest/unknown" wording was removed.
