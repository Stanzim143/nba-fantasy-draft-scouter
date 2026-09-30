# In-season tools

Phase 5 of the roadmap ([ADR 0015](adr/0015-inseason-tools.md)): schedule awareness, a rest-of-season (ROS) projection, a trade
analyzer and a waiver / free-agent finder. Everything is a CLI (`python -m src.inseason.<name>`) and a page in the Streamlit app.
The season has not started (opening night 2026-10-20), so the tools were tested on synthetic data and smoke-tested on the real
2025-26 season "as if" mid-season, using only games up to a cutoff date. **What the nightly refresh does is
[ADR 0018](adr/0018-nightly-inseason-job.md) (see "The nightly job" below), not this page** — this page is
the CLI/manual-use guide for the same underlying tools.

Every tool takes `--as-of DATE` (default today, inclusive): games dated after it never influence the output. Before opening night
that means the preseason projection with every game remaining.

## Quick start

```bash
# once: store the ESPN pro-team schedule (one cached request) and, when you have it, your league's real matchup periods
python -m src.inseason.schedule --season 2026-27 --ingest
python -m src.ingest.espn_league --league-id 1234567890           # ADR 0008; caches the real matchupPeriods

python -m src.inseason.schedule --season 2026-27 --as-of 2026-11-02          # games per team per matchup week, flags
python -m src.inseason.ros --season 2026-27 --as-of 2026-12-01 --top 30       # rest-of-season board
python -m src.inseason.trade --as-of 2026-12-01 --team-id 3 --give "A" --get "B, C"
python -m src.inseason.waivers --as-of 2026-12-01 --team-id 3 --top 20 [--next-week]
streamlit run src/app/draft_board.py       # then open the "inseason" page in the sidebar
```

Whose roster: `--team-id N` (or `$ESPN_TEAM_ID`; needs `$ESPN_LEAGUE_ID` and a synced league with a drafted roster), or
`--my-roster "Name, Name, ..."`, or `--mock-draft-team N` (a deterministic snake draft on the ROS board, for demos and before the
draft; the output says the rosters are a mock). Names are matched accent-insensitively; an ambiguous name lists the candidates.

## Schedule awareness (`src/inseason/schedule.py`, `weeks.py`)

* **`schedule_games`** (standalone parquet table, [ADR 0015 D1](adr/0015-inseason-tools.md)): one row per game. 2026-27: 1,200
  games, 80 per team (NBA Cup games are not resolved yet; ESPN adds them later, so the week of 2026-12-07 reads 0.93 games per
  team and the Cup week before it 1.93 until then).
* **Matchup weeks**: ESPN's `matchupPeriods` when they look real. On 2026-09-24 they did not (a one-day-per-period placeholder), so
  the calendar is **derived**: Monday-Sunday from opening night (week 1 is 6 days), the All-Star week merged into the one before it
  (a 14-day matchup), 20 regular weeks then 3 playoff weeks ending 2027-04-04. That end date equals the league's
  `finalScoringPeriod` (167), the one independent check available. Every report prints the source; re-run the league sync once
  the season starts and the real periods are used automatically.
* **Weekly table** per (week, team): `games`, `games_left` (after `as_of`), `b2b` (second nights), `off_night_games` (games on a
  league-wide slate of few games, threshold = 35th percentile of slate sizes), `heavy` (4+ games in 7 days), `light` (2 or fewer).
  The playoff weeks (21-23 for 2026-27) print games per team and total, for playoff-run planning.
* Per-player weekly games for streaming = team games left in the week x the player's availability (in the waiver finder).

## Rest-of-season projection (`src/inseason/ros.py`)

Method: for each player, three shrinkage averages `(k*prior + n*sample)/(k + n)` between the preseason projection (built only from
earlier seasons) and this season's games so far: per-minute production (n = minutes played, k = 300), minutes per game (n = games,
k = 3) and availability (n = team games, k = 5). `ros_fppg = fp_per_min x mpg`, `ros_games = availability x team games left`
(from the schedule), `ros_total_fp = ros_fppg x ros_games`. `ros_vorp` reuses the value engine's replacement-level machinery
over the games left. Players with no preseason projection get a weak bench-level prior (`has_prior = False`).

### Evidence: does blending beat the preseason projection?

`python -m src.inseason.ros_eval` (real data, walk-forward, 10 seasons 2016-17..2025-26, cutoffs after 15/30/45/60 team games).
Target: what the player actually scored after the cutoff (a player who did not play scores 0). Universe from cutoff-time information:
preseason top 300 plus anyone with 10+ games. Pseudo-counts were chosen on the train seasons (2016-17..2020-21, lowest ROS-total MAE
over a 100-config grid) and reported out of sample on 2021-22..2025-26:

| Test seasons 2021-22..2025-26, mean over cutoffs | Spearman (ROS total) | MAE ROS total FP | Top-50 hit rate | MAE ROS FPPG (10+ games) |
|---|---|---|---|---|
| Preseason projection only | 0.554 | 307.4 | 0.576 | 5.35 |
| Season-to-date only | 0.722 | 273.5 | 0.606 | 4.52 |
| **Blend (default: 300 / 3 / 5)** | **0.730** | **255.0** | **0.612** | **4.17** |
| Direct FPPG shrinkage variant (k = 5) | 0.721 | 251.6 | 0.628 | 4.18 |

Paired by season on the test seasons, the blend beats preseason-only in **5/5 seasons on every metric** (MAE -52 FP, Spearman
+0.176, FPPG MAE -1.18) and beats season-to-date-only in **5/5 seasons** on MAE (-18 FP), Spearman (+0.008) and FPPG MAE (-0.35). By
cutoff the blend's edge over season-to-date-only is largest early (MAE 380 vs 435 after 15 games) and vanishes by 60 games (139.2 vs
139.4), and its rank-correlation gain is small everywhere. Honest read: **in season, this season's games are a far better predictor
than the preseason projection (Spearman 0.72 vs 0.55), and the preseason projection still adds a real but modest amount early on.**
Caveats: the chosen games and availability pseudo-counts sit on the edge of the grid (the sample dominates those two quickly, so a
wider grid could move them); the direct-FPPG variant is a statistical tie (better MAE and top-50 out of sample, worse Spearman), so
neither shrinkage form is clearly better; ROS "actuals" include injuries, which no projection foresees.

## Trade analyzer (`src/inseason/trade.py`, `lineup.py`)

```
Trade analysis, 2025-26 as of 2026-01-15 (mock team 5)
You: gives Cade Cunningham (1785 FP ROS), gets Nikola Jokic, Naz Reid (3220 FP ROS)
  roster value over the rest of the season: 12765 -> 13411 FP (+646; favours you, tolerance +-191)
  roster overfull, drop: Moses Moody
```

The roster is valued as the best assignment of players to PG/SG/SF/PF/C/G/F/UTIL x3 (real eligibility rules) plus a bench worth `w`
times its ROS total, `w` from the value engine's own derivation. Both the roster before and after are first made full: a freed
spot is filled with the best free agent, a 1-for-2's extra player forces a named drop, so **the freed or filled roster spot is part
of the answer**, and giving up your only centre costs the C slot unless someone else is eligible. `--partner ...` scores the other
side too. Verdict "roughly even" inside +-1.5% of roster value. Limits: expected volume only (no daily-lineup or injury simulation);
positional eligibility comes from the dataset position string (ESPN's own eligibility is not used); ROS totals are estimates with
the error shown above.

## Waiver / free-agent finder (`src/inseason/waivers.py`, `signals.py`)

* **Best adds**: each free agent's ROS gain over the player he would replace (best drop among yours, positional and bench effects
  included; with an open spot the comparison is a replacement-level filler). Negative gains everywhere mean nothing helps.
* **Flags**: `rising_minutes` / `rising_usage` (last 5 games vs earlier games this season, needs 8+ earlier games, z >= 1.5 and
  a +3 minutes / +2.0 usage-per-36 rise), `inherits_from` + `exp_gain_mpg` (a rotation teammate is out: absent from the team's
  last game(s), or ESPN `OUT`), and his own injury status. ESPN status comes from the cached universe snapshot only when `as_of` is
  within 3 days of today; refresh it with the ADP ingest or the nightly job.
* **Streaming**: free agents ranked by expected FP in this (or `--next-week`) matchup week, showing team games left, back-to-backs
  and off-night games, and the gain over the player you would drop.
* The minute-redistribution rule was checked on real data (`python -m src.inseason.signals --eval`, 10 seasons):
  1,359 star-absence events (a 25+ mpg player missing 5+ games) and 18,699 teammate pairs. The rule's predicted minute gain
  correlates with the observed with/without gain at Spearman 0.40 (Pearson 0.39); for teammates the rule expects to gain 1.5+
  minutes the observed gain was positive 80% of the time and averaged +4.4 minutes against 2.5 predicted (the rule under-predicts
  the size: summed over teammates it gives 23 minutes per event against 38 observed, because other rotation changes coincide).
  So the *ranking* of who benefits is informative, the magnitude is conservative. In-sample by construction (same-season with/without),
  so it validates the shape of the rule, not a forecast.
  These flags are alerts for a human, not model inputs.

## Real-data smoke (2026-09-24)

* `python -m src.inseason.schedule --season 2026-27 --ingest`: 1,200 games, 80 per team, 0 network requests (already cached), a
  23-period derived calendar ending 2027-04-04.
* `python -m src.inseason.ros --season 2025-26 --as-of 2026-01-15`: 770 players, 509 with games so far, replacement 738 FP; top of the
  board Jokic, Gilgeous-Alexander, Doncic, Maxey, Cunningham.
* `python -m src.inseason.ros --season 2026-27`: the preseason board with all games left (Jokic 3,579, Wembanyama 3,488, Doncic 3,388).
* `trade`, `waivers` on the 2025-26 as-of run with a mock-draft roster; `--team-id 1` against the real league correctly reports
  that the draft has not happened.

## The nightly job (ADR 0018)

`python -m src.ops.nightly` runs all of the above unattended every morning (Task Scheduler, daily 09:30 local from 2026-10-20 to 2027-04-06; `./dev schedule install --job nightly`). Order: incremental game ingest, NBA roster snapshot, ESPN universe + injury archive, league sync, schedule refresh, then the analysis as of the last completed US game day. Read `reports/nightly/latest.md`:

| Section | What it answers |
|---|---|
| What changed | injury changes (yours first), league roster moves and transactions, new rising-minutes / beneficiary flags, new top adds, ROS rank movers, schedule moves |
| The week ahead | games left this week and games next week per team, heavy/light teams, the next slate |
| Recommendations | best adds by ROS gain over your drop, streaming this and next week |
| Player alerts | rising minutes and injury beneficiaries anywhere in the league, tagged mine / free agent / other team |
| Lineup hints | your weekly best lineup, who plays on the next slate, OUT starters, swaps |

Set your team once after the draft: `python -m src.ops.nightly --set-team` lists the league's teams, `--set-team N` validates and saves it (`~/dev-data/nba-fantasy-2026/nightly.json`, or `ESPN_TEAM_ID`); ESPN's public data cannot say which team is yours, and until it is set every report starts with "ACTION NEEDED: YOUR TEAM IS NOT SET". Health: `./dev schedule status --job nightly`, `~/dev-data/nba-fantasy-2026/nightly/status.json`, `reports/nightly/ALERT.txt`. Replay a past night: `python -m src.ops.nightly --force --as-of 2026-01-15 --season 2025-26 --offline --only analysis --mock-team 5 --data-dir <scratch copy> --reports-dir <scratch>` (never point a replay at the real data directory). Status: all steps run live before the season (2026-09-25); the incremental ingest was proven on 2026-09-26 against the real stats.nba.com by replaying 8 + 10 + 6 real nights of 2025-26 (January, the December NBA Cup week, the All-Star break) plus three short windows with debuting players: every night idempotent, the rebuilt `game_logs` and `team_games` rows equal to the canonical pull row for row. Still unproven until 2026-10-20 to 10-22: your real roster, ESPN's real matchup weeks and the first real night; the first-week checklist is in ADR 0018. Run time 09:30 local (NZDT) is deliberate: it is 13 hours after the previous US game day ends and 3 hours before the evening slate tips. Details and limits: [ADR 0018](adr/0018-nightly-inseason-job.md).

## Not done / open

Real league rosters are untested against real data until the draft; matchup weeks stay derived until ESPN publishes them; the
nightly job is built (ADR 0018), proven on replayed historical nights but not yet on a live season; no auto-execution of adds or trades (the ESPN client is read-only by
design, ADR 0008).
