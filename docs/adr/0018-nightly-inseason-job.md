# ADR 0018: The in-season nightly job

**Status:** accepted, 2026-09-25; live ingest proven on historical nights and hardened 2026-09-26
**Code:** `src/ops/nightly.py` (orchestrator, gate, workers, diffs, alerts), `src/ops/live_ingest.py` (incremental game ingest),
`src/ops/nightly_analysis.py` (artifacts), `src/ops/nightly_report.py` (report), `src/ops/schedule.py` (`--job nightly`);
tests `tests/ops/test_ops_nightly*.py`, `test_ops_live_ingest.py`, `test_ops_schedule.py`. Two small edits outside `src/ops/`:
`InSeasonContext.week_anchor` (`src/inseason/context.py`) and its use in `pick_week` (`src/inseason/waivers.py`).
**Builds on:** ADR 0014 (the pre-draft automation whose primitives are reused), ADR 0015 (the tools it runs), ADR 0008 (league sync),
ADR 0005 (ESPN and NBA sources, politeness). **Roadmap:** phase 5 "Runs unattended for a full week".

## Context

ADR 0014 keeps the data fresh until the draft and stops at the draft start (the daily task's `EndBoundary`: 2026-10-17 07:00 NZDT = 2026-10-16T18:00Z, from `config/league.yaml`; ADR 0014 note 2026-09-26). ADR 0015
built the in-season tools but nothing runs them. Opening night is 2026-10-20 and the fantasy final day (the league's
`finalScoringPeriod` 167) is 2027-04-04. Nothing in this ADR could be run against a live season when it was written.

## Decision

1. **A second command, not a mode of the first.** `python -m src.ops.nightly` reuses the ADR 0014 primitives by import
   (`StepOutcome`, subprocess-with-timeout runner, `RunLock`, atomic writes, rotating-log setup, toast, snapshot archive and diff
   helpers) and keeps its own state under `<data>/nightly/` and reports under `reports/nightly/`. `daily_refresh` and its installed
   task are unchanged except that `setup_logging` accepts a logger and file name. The two commands **share one lock file**
   (`<data>/daily_refresh/lock.json`): they write the same tables and caches, and the handoff days may overlap.
2. **Steps**, each in its own subprocess with a timeout inside a 60 minute budget; a failing one never stops the others:
   `games` (incremental ingest), `roster`, `adp` (ESPN universe and id map), `status` (injury archive), `snapshot` (dated ESPN
   archive), `league` (read-only sync, skipped without a league id), `schedule` (refresh `schedule_games`, report moved and added
   games, refuse a schedule that shrank below 95%, keep a `.prev_nightly` copy), `analysis`. A step can also end `degraded`
   (it ran but one artifact failed); that counts as failing for the alert.
3. **Incremental, idempotent ingest** (`live_ingest.py`): pull `playergamelogs` and `leaguegamelog` (T) with `DateFrom` = the newest
   stored game minus 3 days, replace exactly that window in the season's `game_logs` and `team_games` using the ingest's own
   transforms and validation, and refresh `players` and the season bio (two more requests) only when a player appears who is not
   in `players`. The overlap repairs corrected box scores and half-published game days. It refuses to write if the window is smaller
   than 80% of what is stored for the same dates (the error names stored games absent from the pull; a stored game that vanishes upstream is never deleted, and the guard clears once the window moves past it). A game whose player rows lack team rows is held pending like a game without a result and the other final games are stored; only when every game in the window is like that does it refuse (a broken team endpoint). An unchanged table is not rewritten; a
   changed one keeps one rolling `.prev_nightly` copy. Before opening night (or while the source is still empty) it makes no
   request and says so. Two requests a night at the NBA client's 1 s minimum interval, plus about six ESPN requests at 2 s.
4. **"Completed through" and the slate.** NBA games are dated in US Eastern time and the last one of a night ends about 03:30 ET.
   `completed_through(now)` = `(now_ET - 3.5 h).date() - 1 day` (Eastern DST is computed by rule; this venv has no tz database).
   The **slate** is the next date to be played. The ROS projection is built as of `completed_through`; "this week" is chosen by the
   slate (`week_anchor`), so on the Monday a new matchup week starts the recommendations are for the new week.
5. **Artifacts** (`nightly_analysis.py`, each built in isolation, reusing the ADR 0015 tools): `ros` (top 300 to CSV, rank diff), waiver
   adds and streaming for this week and next (`find_waivers`), rising-minutes and injury-beneficiary alerts league-wide tagged mine /
   free agent / other team (`signals`), games per team this and next week and the slate, and start/sit hints (a weekly best lineup
   from games left x availability x FPPG, plus the slate's lineup; hints for OUT players, no games left, bench players who play on the
   slate, swaps for starters with no game, stream suggestions worth 3+ FP). Without a team id, or before the draft, team artifacts are
   skipped with the reason in the report and the league-wide ones still run.
6. **Report and structured output.** `reports/nightly/<date>.md`, `latest.md`, `runs/<run id>.md`, `latest.json` (the full run record)
   and CSVs in `reports/nightly/data/`, plus `<data>/nightly/status.json`, `history.jsonl` and per-run state files. The report diffs
   against the previous run: ESPN injury changes (yours first), league roster moves and transactions, NBA roster moves, new
   rising-minutes and beneficiary flags, new top waiver adds, ROS rank movers, schedule changes, a matchup-calendar source change
   (derived to ESPN), games ingested, data freshness.
7. **Season-window gate.** Active from the first game in `schedule_games` (2026-10-20; the constant is the fallback) to the end of the
   league's final matchup period (ESPN's periods when real, else the derived calendar that reproduces `finalScoringPeriod`, else the
   last scheduled game, else start + 166 days) plus 2 days of grace: 2026-10-20 .. 2027-04-06. Outside it the command exits 0 doing
   nothing. `--force` overrides; `--season-start` and `--window-end` set it explicitly; a missing schedule, league or config never
   raises (the fallback is named in the report). Days with no games run normally (injury news, waivers) and say when the next game
   day is.
8. **Alerts** (`reports/nightly/ALERT.txt` plus a best-effort Windows toast, only when the reasons change, cleared by the next healthy
   run): 2+ consecutive runs with failing or degraded steps; the newest stored game more than 1 day behind the newest scheduled game
   at or before `completed_through`; ESPN snapshot or league sync older than 36 h; NBA roster snapshot older than 36 h.
   `--as-of D` replays a night as if D were the last completed day: no alerts, no freshness checks, the report is dated by the slate.
9. **Scheduling**: `./dev schedule install --job nightly` registers "NBA Fantasy 2026 Nightly In-Season": per user, interactive token,
   least privilege, one daily trigger at **09:30 machine-local**, `StartWhenAvailable`, network required, one instance, 90 minute
   limit, trigger window = the job's season window, `pythonw.exe -m src.ops.nightly --quiet` from the primary checkout. `--times`
   changes it. The pre-draft task is untouched.
10. **Why 09:30 (worked out again 2026-09-26; unchanged).** The machine is on New Zealand time, UTC+13 from 2026-09-27 to 2027-04-04
    (then UTC+12). A run at 09:30 on NZ day D is 20:30 UTC on D-1, which is 16:30 EDT (15:30 EST after 2026-11-01, EDT again from
    2027-03-14) on US day D-1. US game day D-2 ended at the latest about 03:30 ET on D-1 (a 10:30 pm ET tip with overtime), which is
    07:30 UTC = 20:30 NZDT on D-1: **13 hours before the run** (12 hours in the worst DST combination). US game day D-1 tips from 19:00
    ET (23:00 UTC, **12:00 NZDT on D**), so the run precedes its first evening tip by 2.5 to 3 hours: the report is the "before
    lineups lock" report for slate D-1, with the newest complete box scores (D-2) and the game-day injury news. Opening night, Tuesday
    2026-10-20 (tips 19:30 ET = 12:30 NZDT on 10-21): the run of 2026-10-20 has an empty slate (season not started); the run of
    **2026-10-21 09:30 NZDT is the first that reports opening night as the slate**; the run of **2026-10-22 09:30 NZDT is the first
    that ingests its games** (the last ended about 07:30 UTC on 10-21). Alternatives rejected: an evening run (after 20:30 NZDT) sees the
    same box scores about 13 hours sooner but its injury news is 8+ hours older when the next slate's lineups are set; a 07:00 run
    would sit in the middle of weekend matinee tips (05:00 to 07:00 NZDT) for no gain. **Games still in progress are never stored**:
    a Sunday 1 pm ET tip is still running at 15:30 EST (09:30 NZDT during US standard time), so `live_ingest` keeps only games whose
    two team rows carry a result (`WL`) and whose player points add up to each team's points, and reports the rest as pending; the next
    night's 3-day overlap picks them up.
11. **Contention and catch-up.** The shared lock is waited for (default 20 minutes, `--lock-wait-minutes`; 20 + 60 budget < the 90 minute
    task limit) before a night is skipped, so a run still finishing when the two tasks meet does not cost a night (in practice they do not meet: nothing runs between the draft start, 2026-10-17 07:00 NZDT, and the nightly's first run, 2026-10-20 09:30 NZDT). A
    machine that slept through several nights gets one `StartWhenAvailable` run, and because the ingest window starts 3 days before
    the newest *stored* game, that single run recovers every missed night (tested; Task Scheduler runs one catch-up, not one per night).
12. **Your team.** ESPN's public league data has no "this is me" flag (only anonymous member ids and display names), so the team cannot
    be found without credentials and is never guessed. `python -m src.ops.nightly --set-team` lists teams (id, abbreviation, roster
    size, team name, owner display name); `--set-team N` validates N against the last league sync and writes it into
    `<data>/nightly.json` (other keys kept). Until it is set, or when the saved id is not in the league, the report starts with
    "ACTION NEEDED: YOUR TEAM IS NOT SET" and, once the draft is complete, the run raises an alert (toast once, `ALERT.txt` until fixed).
13. **Housekeeping.** The raw NBA cache files of the nightly date windows are never re-read; those older than 14 days are deleted (about
    250 KB a night otherwise). Logs rotate (1 MB x 5). A new player triggers the player index, the season bio and one `commonplayerinfo`
    per new player (exact birthdate, as the full ingest has it). Known limit: `player_season_bio.team_id` (a player's latest team of the
    season) is refreshed only on nights with a new player.

## Consequences and limits (what has and has not been exercised)

* **Exercised live on 2026-09-25, before the season:** every step in `--force` mode against the real data directory: `games`
  (correctly skipped before opening night, zero requests), `roster`, `adp`, `status`, `snapshot`, `league` (the real ESPN league),
  `schedule` (1,200 games, unchanged), `analysis` (2026-27 preseason projection, league-wide artifacts, no team configured). Exit 0, 93 s.
* **Proven live on 2026-09-26 (real stats.nba.com, historical nights, fake clock via `--as-of`, scratch copy of the data):** the
  `MM/DD/YYYY` `DateFrom`/`DateTo` windows are accepted and honoured by `playergamelogs` and `leaguegamelog`. The real nightly command
  (`--force --as-of D --only games`, one subprocess per night, real requests, the client's rate limit) was replayed night by night and
  each night re-run for idempotency, from a copy truncated before the window: 2026-01-12..01-19 (8 nights), 2025-12-08..12-17 (10 nights
  including the NBA Cup knockout games, which carry `006` game ids and are correctly excluded, and the day of the Cup final with no
  regular-season game), 2026-02-09..02-20 (the All-Star break: 6 nights without games, 2 requests each, nothing changed), plus three
  short windows containing debuting players (2025-11-26..12-02 with the Thanksgiving gap, 2026-02-10..02-13, 2026-01-18..01-19).
  **Result:** all night runs and reruns exit 0; every rerun is a no-op (tables not rewritten); the resulting `game_logs` rows (1,283 + 732
  + 1,078 in the three main windows, 1,862 more in the short ones) and `team_games` rows (118 + 70 + 100, 174 more) equal the canonical
  full-season pull **row for row, every column, same dtypes** (fractional minutes, game ids like `0022501225`, ET game dates, `is_home`,
  `pts_against`), and every row outside the window is unchanged. Debuting players were re-discovered on their first night (5 + 4 + 1)
  with the player index, bio and CommonPlayerInfo pulls; the `players` table then matches the canonical one except `to_year` for 6 to 7
  players (today's player index versus the index of the earlier canonical pull: source drift, not logic) and `player_season_bio.team_id`
  (latest team as of the replayed night versus the full season, expected). **Discrepancies the live run revealed and fixed:** (1) a new
  player had no birthdate, so `age_at_season_start` was off by up to 0.2 years: the ingest now fetches CommonPlayerInfo for each new
  player; (2) nothing stopped a game still in progress from being stored: the final-game check above; (3) a replay had no upper bound
  (`DateTo`), so past nights could not be replayed faithfully: `--as-of` now passes `DateTo`. Pinned in
  `tests/ops/test_ops_nightly_finish.py`.
* **Exercised on historical data only:** the full artifact set for a team (mock snake-draft roster) as of 2026-01-15 and again as of
  2026-01-19 (on tables just rebuilt by the live replay and the real ESPN 2025-26 schedule), offline, in a scratch copy of the data
  directory. The ROS, waiver and signal tools themselves were validated in ADR 0015.
* **Simulated only (fake stats.nba.com, fake steps):** outage isolation, the alert after two bad nights and its clearing, a stored game
  never removed by a bad refresh, the lock wait, catch-up after missed nights, the gate, the Task Scheduler XML. HTTP 429 and timeouts are
  handled by the shared NBA client (exponential backoff, `Retry-After`, six retries; unit-tested, never provoked against the live API); a
  `games` step that still fails is reported, leaves the tables as they were, and alerts on the second night.
* **Unknowable until the season, checklist for the user:**
  1. after the draft: sync the league (the nightly `league` step, or `python -m src.ingest.espn_league --league-id 1234567890`), then
     `python -m src.ops.nightly --set-team` and `--set-team N`. Until then every report says so at the top.
  2. **2026-10-20 morning:** the first scheduled run (empty slate, "the season opens" note in the games step). `./dev schedule status
     --job nightly` should show a run today with exit 0.
  3. **2026-10-21 morning:** `reports/nightly/latest.md` lists the opening-night slate and your lineup hints.
  4. **2026-10-22 morning:** the first ingest of real games: "N new game(s)" equal to the games played on 10-20 (and 10-21), new players
     added, no `ALERT.txt`. Cross-check one score on nba.com.
  5. **Weeks 1 to 2:** whether ESPN's real `matchupPeriods` replace the derived calendar (the report announces "MATCHUP CALENDAR SOURCE
     CHANGED ... espn"); if the weeks differ, check the boundaries in "The week ahead".
  6. **A week in:** `status.json` `consecutive_failures` is 0, no toast appeared, `nightly.log` has one block per night. That is the "runs
     unattended for a full week" criterion of PLANNING.md phase 5; it is not met until then.
* The task only runs while the computer is on and the user is logged in (`StartWhenAvailable` catches up);
  `./dev schedule status --job nightly` is the check for a task that stopped firing.
* Not built: any write to ESPN (the client is read-only, ADR 0008); auto-executed adds or trades. NBA Cup knockout games (`006` ids) are
  not in `game_logs` (nor in any earlier season); whether ESPN counts them for fantasy is unknown.

## Addendum 2026-10-01

An `injuries` step (after `games`) was added to the nightly job by [ADR 0033](0033-season-intervals-and-injury-labels.md): it archives the league's
official injury reports for the season's game dates not yet fetched (at most 60 requests a night, incremental parse). The 300 s step timeout and
the step list in this ADR otherwise stand; the first real run is, like the rest of the job, opening night.
