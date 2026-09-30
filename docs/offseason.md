# Summer League, preseason and the breakout watchlist

The user guide and draft-day runbook for ADR [0012](adr/0012-offseason-layer.md) (the decisions and the evidence live there).
Everything here reads the same real data as the board; nothing is committed (raw pulls stay in `~/dev-data/nba-fantasy-2026`).

## What you get

| You want | Run | What it does |
|---|---|---|
| Bring everything current | `python -m src.ingest.preseason_refresh` | Roster snapshot + new rookies, Summer League + preseason box scores, ADP, in dependency order. Idempotent, cached, polite. About 15 seconds when cached. |
| The breakout watchlist | `python -m src.value.breakouts --season 2026-27 --out reports/watchlist.csv` | Young, under-the-radar players ranked by how far their offseason evidence moves the projection, with team, pick, ADP, their own lines and a calibrated probability. |
| The board with all of it | `streamlit run src/app/draft_board.py`, load `2026-27` | Board now includes the rookie class and ADP columns; the **Breakouts** tab shows the watchlist and hides players you have drafted. |
| A board from the offseason model | pick `baseline_offseason` in the sidebar, or `python -m src.value.board --season 2026-27 --model baseline_offseason --adp ...` | The baseline with the Summer League / preseason adjustment applied. |
| The evidence | `python -m src.backtest.breakouts --model baseline_offseason` | Walk-forward breakout evaluation across ten seasons. |

## Automated daily refresh (ADR 0014)

A Windows Task Scheduler entry runs `python -m src.ops.daily_refresh` twice a day (07:30 and 19:00 local) until the day after the
draft. Each run does the refresh above, archives a dated ESPN snapshot (`~/dev-data/nba-fantasy-2026/archive/espn_players/`), refreshes player profiles and appends the day's ESPN injury statuses to `espn_status_snapshots` (the draft-board risk overlay's history, ADR 0016), syncs the
league read-only, rebuilds the watchlist (`reports/daily/watchlist.csv`) and the draft-board CSVs
(`reports/daily/draft_board_*.csv`), and writes `reports/daily/latest.md`: roster moves, injury status changes, ADP risers and
fallers, players newly on or off the watchlist, new preseason games, new league transactions (trades, signings, waivers, coach moves, ranked against the board; ADR 0017, `docs/transactions.md`) and any failure, since the previous run.

| You want | Run |
|---|---|
| Read the latest report | open `reports/daily/latest.md` (a dated copy is in `reports/daily/<date>.md`) |
| Is it healthy? | `./dev schedule status` (task state, last/next run, last result, alert) and `~/dev-data/nba-fantasy-2026/daily_refresh/status.json` |
| Run it right now | `./dev schedule run-now` (via Task Scheduler) or `python -m src.ops.daily_refresh --force` |
| Change the draft date or time | edit `draft.date` / `draft.start_utc` in `config/league.yaml` (currently 2026-10-17, earliest start 2026-10-16T18:00Z = 07:00 NZDT), then re-run `./dev schedule install --job daily` from the primary checkout; `--draft-date`, `--draft-start`, `NBA_DRAFT_DATE`, `NBA_DRAFT_START` or `draft_date`/`draft_start` in `~/dev-data/nba-fantasy-2026/daily_refresh.json` override the config |
| Turn it off | `./dev schedule remove` (or Task Scheduler, task "NBA Fantasy 2026 Daily Refresh", Disable) |

It only runs while the computer is on and you are logged in (missed runs catch up when it wakes). If it fails twice in a row or the
data goes stale you get `reports/daily/ALERT.txt` and a Windows notification; the log is
`~/dev-data/nba-fantasy-2026/daily_refresh/logs/daily_refresh.log`. The draft date and earliest start come from `config/league.yaml`; the task stops firing at that start (2026-10-17 07:00 NZDT) and a stray later run does nothing.

### Handoff to the nightly job (ADR 0018)

The pre-draft task stops firing at the draft start, 2026-10-17 07:00 NZDT (its `EndBoundary`; it was 2026-10-19 before the draft date was known, ADR 0014 note 2026-09-26). On 2026-10-20 the in-season task "NBA Fantasy 2026 Nightly In-Season" (`./dev schedule install --job nightly`, daily 09:30 local, until two days after the fantasy final day, 2027-04-06) takes over: `python -m src.ops.nightly`, report `reports/nightly/latest.md`. Both tasks stay registered; they share one lock so an overlap cannot corrupt a table (none is expected: nothing runs between 2026-10-17 07:00 and 2026-10-20 09:30 NZDT). Before the first nightly run, tell it which team is yours (`{"team_id": N}` in `~/dev-data/nba-fantasy-2026/nightly.json`, or `ESPN_TEAM_ID`) once the draft has happened; without it the team sections are skipped and the report says so. See [`docs/inseason.md`](inseason.md).

## What changed on draft day (ADR 0016)

| You want | Run | What it does |
|---|---|---|
| The debutants on the board | pick `baseline_offseason_debut` (or `baseline_debut`) in the sidebar, or `python -m src.value.board --season 2026-27 --model baseline_offseason_debut --adp ... --out board.csv` | Adds the 5 stash players (Sorber, Marković, Toohey, Diop, Biberovic) and the undrafted signees (about 50 at 2026-09-25), each with `projection_class`, `p_play` and `confidence = low`. Under `baseline_debut` veterans and rookies are the baseline's numbers exactly; under `baseline_offseason_debut` they differ from `baseline_offseason` because debutant rows enter the offseason layer's training residuals (ADR 0016). Measured 2026-09-25 on the 739 shared players: vs `baseline_offseason` 186 differ, max |dFPPG| 1.07, 1 above 1 FPPG; vs `baseline` 186 differ, max 3.24, 23 above 1 FPPG (this includes the offseason layer's intended adjustment). This board is also the one the daily refresh writes with the debutants and undrafted signees (`reports/daily/draft_board_baseline_offseason_debut.csv`); the daily `baseline` and `auto` boards omit them. |
| Origin profiles and the draft combine | `python -m src.ingest.nba_profiles` (or `preseason_refresh --with-extras`) | `player_profiles` (country, previous organisation, origin, birthdate for rostered players without history) and `draft_combine`. One-off: 65 requests, then cached. |
| ESPN injury status, archived | `python -m src.ingest.espn_status` (or `preseason_refresh --with-extras`) | Appends today's ESPN status for 600 players to `espn_status_snapshots`. Run after the refresh's ADP step so the payload is fresh. |
| Risk flags | automatic on the board, watchlist and CLI in the live window | `risk_level` (watch / high), `risk_flags`, `risk_gp` (advisory games after a haircut). **Projections are not changed.** The **Debutants & risk** app tab lists them. |

How to read the new columns:

* `projection_class`: `veteran`, `rookie` (this year's class), `stash` (drafted earlier, first NBA season), `undrafted`. `p_play` is the chance he plays at all;
  `proj_gp` already includes it. **Stash `p_play` is a fixed 0.90 assumption**, not an estimate (history cannot identify it, ADR 0016 D1). Treat every debutant row as the least
  certain on the board. The undrafted rows are small on purpose: about one in three camp players plays at all.
* `risk_flags`: `ESPN out` / `day-to-day` / `suspended` (current ESPN status; `suspended` is level `watch` like day-to-day; the 20% (OUT), 3% (day-to-day) and 10% (suspended) game haircuts behind `risk_gp` are assumptions, and a status with news older than 45 days is shown as stale and
  not discounted); `played 0 of N preseason games` (warning only, no discount: history cannot separate injury from a player not under contract); `new team (from XXX)`; `star arrived/left:`
  (a top-60 board player joined or left his team since last season). The overlay is meaningful from August of the season's year; before preseason games exist the preseason flag is silent.
* `return_flag` (`returned-healthy`), `return_block_pct`, `return_tail` (e.g. `16/20`), `return_gp_upside_adv`, `return_fp_upside_adv` (ADR 0023): he missed the first 25%+ of last season's team
  games, then played 75%+ of at least 15 remaining games (Tatum: 62 of 82, then 16 of 20). The same sentence is appended to `risk_flags`; `risk_level` and `risk_gp` do not change. The advisory
  upside is a **judgement, not a projection**: +5 games (and +5 x `proj_fppg` total FP) for a block of 60%+, else 0, because ADR 0021 found no significant absolute under-projection for this group
  (+1.0 GP, CI -4.1 to +5.8). `proj_gp`, `proj_total_fp` and the rank are exactly what they were. The sentence in `risk_flags` appears only in the August-December live window (when the risk overlay exists); the `return_*` columns need no roster snapshot and are shown at any time of year.
* `lm_flag` (`short-absences`), `lm_iso_n`, `lm_rest_n`, `lm_gp_risk_adv`, `lm_fp_risk_adv` (ADR 0024): a rotation veteran (single team, >= 20 mpg, >= 41 GP) who missed 6+ games last season in 1-2 game absences.
  The cause is unknown (rest or minor injury; `lm_rest_n` counts the one-game back-to-back absences, which alone showed no effect). The study found such players land about 2.6 GP [-5.1, -0.2] below the
  baseline on average, but the association did not improve out-of-sample forecasts, so the advisory -2 GP (and -2 x `proj_fppg` total FP) is a judgement only; the sentence is appended to `risk_flags`,
  and `proj_gp`, `proj_total_fp`, the rank and `risk_*` are exactly what they were. About 29% of the 2026-27 universe (54 of 185) carries it, so use it as a tiebreaker, not a warning.
* **What was checked**: the debutant treatment and origin proxies were walk-forward tested; ESPN status and preseason absence could not be (no history exists). The NBA's injury-report PDFs do not exist
  before opening week (probed for 2024 and 2025), so nothing official is available before the 2026-10-17 draft.

## Draft-day runbook (draft 2026-10-17 morning NZ = 2026-10-16 evening CEST)

Timeline (earliest plausible start 2026-10-16 18:00 UTC; the real start is somewhere in 18:00-20:00 UTC, i.e. 07:00-09:00 NZDT):

| Event | NZDT (UTC+13) | UTC | CEST (UTC+2) |
|---|---|---|---|
| Daily run | Thu 2026-10-15 07:30, 19:00 | Wed 10-14 18:30; Thu 10-15 06:00 | Wed 10-14 20:30; Thu 10-15 08:00 |
| Last full US preseason night that counts: games of Thu 2026-10-15 (US), east coast final about | Fri 10-16 15:00 | 10-16 02:00 | 10-16 04:00 |
| Final run 1 (after the east-coast games) | Fri 10-16 15:30 | 10-16 02:30 | 10-16 04:30 |
| Daily run (west-coast games final about 18:30 NZDT) | Fri 10-16 19:00 | 10-16 06:00 | 10-16 08:00 |
| **Final pre-draft run** (105 min before the earliest start) | Sat 10-17 05:15 | Fri 10-16 16:15 | Fri 10-16 18:15 |
| You: read `reports/daily/latest.md`, open the app, load the board | Sat 10-17 05:30-06:45 | 10-16 16:30-17:45 | 10-16 18:30-19:45 |
| **Earliest plausible draft start**; scheduled runs end here | Sat 10-17 07:00 | Fri 10-16 18:00 | Fri 10-16 20:00 |
| Latest plausible draft start | Sat 10-17 09:00 | Fri 10-16 20:00 | Fri 10-16 22:00 |
| US preseason games of Fri 10-16 (US) end: too late to use | Sat 10-17 afternoon | after the draft | after the draft |
| Opening night; nightly job takes over (first run 09:30 NZDT) | Tue 10-20 | | |

Preseason games of the last US night are the most valuable single data point: the Breakouts tab switches itself from the
Summer-League-only model to the full model (`baseline_offseason`) as soon as any preseason game exists (from about Oct 2), and the
signal is in the preseason, not Summer League. Machine requirements: on, awake and you logged in at the trigger times (see the
"only runs while the computer is on" note above; `WakeToRun` is off on purpose). Set the power plan to not sleep the evening of
the 16th and the night to the 17th (NZ), or run `./dev schedule run-now --job daily` by hand around 05:15 NZDT.

By hand, in order: (1) about 05:30 NZDT check `./dev schedule status` (last run `ok`, no ALERT) and read `reports/daily/latest.md`
(roster moves, injuries, ADP movers, watchlist changes); if the 05:15 run failed or was skipped, run `./dev schedule run-now --job daily`
(a manual run works right up to the draft, and the scheduled gate does not block `--force`). (2) Open the app, pick the model
(`baseline_offseason`, or `baseline_offseason_debut` for the debutants), and check the Breakouts tab says it used the full model.
(3) Have the board and watchlist open before 06:45 NZDT; the draft can start at 07:00.

The daily automation covers "refresh the data and rebuild the watchlist and board CSVs". What stays manual: reading the report,
judging the watchlist, and loading the board in the app. Steps 2 and 3 below are done for you at the scheduled times (07:30 and
19:00 NZ, plus the two final runs in the table above); still confirm the last one finished (`./dev schedule status`) so nothing is a
few hours old.

1. **Now (late September).** Nothing is required. The watchlist already works but can only use Summer League, which the
   backtest found to be close to noise on its own. Do not draft off it yet.
2. **From about Oct 2, as preseason games are played.** `python -m src.ingest.preseason_refresh` any time you like; it prints
   how many preseason games exist and who moved teams since the last snapshot. Trades, cuts and signings show up in
   "roster moves since the previous snapshot".
3. **The day before the draft (NZ: Fri 2026-10-16), and again just before it (NZ: Sat 2026-10-17, about 05:30).** Run the refresh, then the watchlist. Once any preseason game of the
   season exists the tool switches itself to the full model (`baseline_offseason`); the app's Breakouts tab prints which model
   it used. The preseason carries most of the signal, so this is the run that matters. A draft held before the preseason is
   half over sees materially less (top-ten lift +8.7pp instead of +13.7pp in the backtest).
4. **In the app.** Load the board with `baseline_offseason` if you want the adjustment in the ranking; keep `baseline` if you
   want the exact model every earlier result was measured on. The Breakouts tab is independent of that choice.

## Reading the watchlist

Columns: `useful_prob` and `breakout_prob` (below), `uplift` (the model's FPPG change from the baseline), `base_fppg` and
`layer_fppg`, `board_rank` (where the board ranks him, lower is better), `adp` and `adp_gap` (the market's price and the gap to the
board), `team` and `changed_team` (from the roster snapshot vs last season's ending team), `draft_pick`, and `evidence`, his own
lines, e.g. `SL 5g 28mpg z+1.5 | pre 3g 24mpg z+0.8` (games, minutes per game, and production per 36 minutes as a z-score against
that event's cohort).

* **Young** = age 23 or under. **Under the radar** = no ADP, or ADP rank worse than 100. Toggle both off in the app or with
  `--all-ages --include-priced`. A player is only listed if his uplift is positive; no positive evidence, no flag.
* **`breakout_prob`**: chance he beats the baseline by at least 4 FPPG and 25%, given he plays 20 or more games.
  **`useful_prob`**: chance that breakout also lands him in the rostered top 169. This is the one to rank by; the default sort.
* **What the numbers mean.** Backtested over ten seasons, the top ten flagged young under-the-radar players became a useful
  breakout about **24%** of the time against a **10%** base rate (about 2.3 times), with a wide margin for error
  (+13.7 percentage points, 95% interval +5.8 to +22.1). It is an edge, not a lock: most flagged players do not break out.
  The probabilities are conditional on playing 20 games; `proj_gp` says how likely that is.
* **Sort by `sl_z` or `pre_z`** to see raw Summer League or preseason production with no model in the way. Worth doing; the model
  found Summer League to be weak evidence, and you may know things the box score does not.
* The list is mostly rookies before the preseason (they are who plays Summer League). After it, second- and third-year players
  whose October minutes jumped can show up too.

## What the evidence says (short version)

* **Preseason** minutes and production carry the signal: rookies AUC 0.66 for a useful breakout, top-ten lift +15.4pp.
* **Summer League alone** is close to noise for this purpose (AUC 0.52-0.57, top-ten lift indistinguishable from zero), in ten
  summers, with two feature designs. It is ingested, in the model, and shown as evidence, but the model's own estimate of its
  value is small. That contradicts the premise the feature was requested on; ADR 0012 has the numbers and the caveats.
* Whole-league: better per-game and VORP-weighted accuracy; total-FP rank order slightly noisier overall and better among the
  players who played 20+ games. See the ADR's results table.

## The data, and how it stays honest

| Table (in `~/dev-data/nba-fantasy-2026/processed/`) | Written by | Contents |
|---|---|---|
| `offseason_logs`, `offseason_team_games` | `nba_offseason` | Every Summer League (2015-16 to 2026-27, no 2020) and preseason (through the live season) game: 39,836 player-games. |
| `roster_snapshots` | `nba_incoming` | One row per rostered player per snapshot day (current NBA team). |
| `players` (+52 rows) | `nba_incoming` | The 2026 draft class with real slots and birthdates. |
| `breakout_calibration_<model>.json` | `python -m src.backtest.breakouts --save-calibration` | The logistic calibration behind the probabilities. |
| `nba_offseason_report.json`, `nba_incoming_report.json` | the ingests | What was dropped and why, per event. |

* **Leakage.** Each row is tagged with the season it *follows*, so July 2026 is `2025-26` and `History.until("2026-27")` sees it
  while `History.until("2025-26")` does not. The future-invariance check, a bend-only-the-future test, and `--leak-check` in the
  real backtest all cover it.
* **Data-quality findings** (all counted in the reports, all pinned by tests): `playergamelogs` has placeholder player ids for
  most pre-2024 summers, so `leaguegamelog` is the row source; the 2019-20 preseason pull contains July 2020 bubble games (dropped
  by date window); the July 2026 Summer League has roughly 40% of free throws missing from the FT counters while points are correct
  (kept, excess reported); 2020 Summer League was never held.

## Troubleshooting

* **"no preseason games yet"** in the refresh summary: expected until October. The watchlist uses Summer League only and says so.
* **A rookie is missing from the board.** He is not on an NBA roster yet (`ROSTER_STATUS`), or he was drafted in an earlier year and
  is only now debuting (five this season: Sorber, Marković, Toohey, Diop, Biberovic). The plain `baseline` and `baseline_offseason` boards still omit
  them by design (their numbers are pinned); use `baseline_offseason_debut` to see them, flagged low confidence.
* **Probabilities are blank.** No calibration file: `python -m src.backtest.breakouts --model baseline_offseason --save-calibration`
  (and once for `baseline_summer_league`). About 90 seconds each. Needed again only if the model or its definitions change.
* **`reports/daily/ALERT.txt` exists**: read it, then `reports/daily/latest.md` for the failing step and the log. Re-run with `./dev schedule run-now`; the file disappears after the next fully successful run.
* **The task did not run**: `./dev schedule status` shows the last run time and result; the computer must be on and you logged in (it catches up on wake). Outside the date window a run does nothing by design.
* **The ADP step failed** (ESPN down or blocked): the other steps still ran and the last good ADP is still used. Re-run later.
* **The Breakouts tab asks you to press "Load the breakout watchlist"**: it fits the ten walk-forward seasons (about a minute the first
  time), on request so it never slows a pick, then re-filters instantly.
* **Everything is empty**: `python -m src.ingest.nba_offseason` has never run, so there are no offseason tables (the tab says so).

## Not built (yet)

College and international *production* (no compliant, machine-readable, historically complete source: ADR 0016 D3), validated injury / preseason-absence effects (no history: ADR 0016 D4;
`roster_snapshots` and `espn_status_snapshots` accumulate for next year), and a per-team context feature. Debutants and origin *proxies* are built (ADR 0016).
