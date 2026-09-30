# ADR 0012: Preseason roster update, Summer League and preseason evidence, and a breakout watchlist

**Status:** accepted, 2026-09-24
**Code:** `src/ingest/nba_offseason.py`, `src/ingest/nba_incoming.py`, `src/ingest/preseason_refresh.py`,
`src/features/offseason.py`, `src/models/offseason_baseline.py`, `src/models/rookies.py` (availability cap),
`src/backtest/breakouts.py`, `src/value/breakouts.py`, `src/app/` (Breakouts tab, ADP columns).
**Usage and runbook:** [`docs/offseason.md`](../offseason.md). **Evidence:** the real walk-forward runs listed at the end
(`reports/` is gitignored; every command below reproduces its numbers).

## Context

PLANNING.md's risk table said preseason trades and role changes after the data freeze needed "a preseason roster/minutes
update step", and the user asked for two things on top: include Summer League data (the last ten summers plus this one), and
use it to find young, under-the-radar players about to break out. Answering that turned up three things nobody had looked at:

1. **The 2026 draft class did not exist in the data.** `nba_transform.build_players` only creates rows for people who appear
   in NBA game logs, so a rookie who has not played a game cannot be in `players`. The backtest never noticed, because every
   past rookie eventually debuted. For the live season it meant the 2026-27 board contained **no rookies at all**, and the 14
   "unmapped" ESPN ADP rows that had been described as fringe players were in fact the entire rookie class (Dybantsa, Boozer,
   Wilson, Peterson, ...). (An earlier note in PLANNING.md said otherwise; that was wrong and is corrected there.)
2. **Summer League and preseason box scores are available** from the same stats.nba.com source the project already uses, under
   the same accepted-risk stance (ADR 0005 R1: private, non-commercial, nothing committed). `docs/research/data-sources.md`
   4.9 had recorded "nothing legal and machine-readable found"; that referred to depth charts and minutes projections, not
   box scores.
3. **Once rookies were on the board, the rookie prior showed a flaw**: it projected ~81 games for picks 1-3 (see D7).

## Decisions

### D1. Two standalone tables, tagged so the leak guard is structural

`offseason_logs` (one row per player per game played, minutes > 0) and `offseason_team_games`, standalone parquet files in the
ADR 0011 D1 pattern (not in `src.contracts.TABLES`; each module validates its own schema).

Every row carries two season labels: `event_season`, exactly as stats.nba.com labels it (July 2026 = `"2026-27"`), and
`season`, the completed season the event **follows** (July 2026 = `"2025-26"`). `History.until` slices extras on `season`, so an
event is visible to a projection of season S exactly when it happened before S started, with no new leakage code. Verified by
the repo's own `assert_projector_ignores_future` (it scrambles every frame with a `season` column, which includes these), by a
test that bends only future-tagged rows and requires identical output, and by `--leak-check` in the real backtest.

### D2. `leaguegamelog` is the canonical row source; `playergamelogs` only refines minutes

Found on the real data, and the reason a "better" endpoint is not used: for most summers before 2024, `playergamelogs`
reports many players under **placeholder ids that match no NBA `PERSON_ID`** (2015-16 and 2016-17: zero overlap with the same
games in `leaguegamelog`) and with blank names, teams and matchups. `leaguegamelog` (player rows) has the real ids and names for
every season but rounds minutes to integers. So rows come from `leaguegamelog`, and fractional minutes are taken from
`playergamelogs` only where the same `(game_id, player_id)` exists in both and the two minute values agree within one minute.
Unmatched `playergamelogs` rows are counted, never used. Of 39,836 stored rows, 50-70% of Summer League rows and roughly 90% or more of
preseason rows belong to players in the `players` table (the rest never played an NBA game).

### D3. Data-quality rules, each counted in a JSON report

* **Event windows.** Preseason rows outside Sep 1 - Dec 31 of the start year, Summer League rows outside Jun 1 - Sep 30, are
  dropped. Needed: the 2019-20 "preseason" pull contains 822 rows of July 2020 bubble games, which happened after that season began.
* **Game-id prefix and season digits** (`152yy...`, `001yy...`); rows with no minutes or missing box-score fields; duplicate
  `(game, player)` keeps the row with most minutes; player rows for teams with no team-game row (exhibitions against non-NBA
  clubs) are dropped.
* **Points identity is one-sided.** Points must be at least `2*fgm + fg3m + ftm`. Exact equality holds in every event except the
  **July 2026 Summer League, where official points exceed the made-shot points on every team in every game** (+7.8 per
  team-game on average, correlating about 0.8 with recorded free throws: roughly 40% of free throws appear to be absent from
  `ftm`/`fta` while the points include them). Player sums equal team totals exactly, so `pts` is trustworthy and the FT counters are what is short.
  Dropping those 715 rows would have discarded most of the most important summer, so they are kept and the excess is reported
  (`rows_with_unrecorded_points`, `unrecorded_points_total`). A real-data test pins this: only that one event may have excess.
* The 2020 Summer League was never held; that event is legitimately empty and typed as such.
* The live season's events are always re-downloaded. A cached "0 preseason games" from September must not mask October's games.

### D4. The preseason roster update: incoming class in `players`, dated roster snapshots

`nba_incoming` reads `playerindex` for the target season (which lists each player's **current team and `ROSTER_STATUS`**;
`TEAM_ID` alone is filled even for retired players), adds every rostered player drafted in the target year who is not yet in
`players` (real draft slot, position, size, and birthdate via `commonplayerinfo`), and appends a dated row per rostered player to
`roster_snapshots` (re-running a day replaces that day). `roster_moves` diffs two snapshots into arrivals, departures and team
changes. Result on 2026-09-24: 584 rostered players, +52 rookies, ADP unmapped rows for 2026-27 down from 14 to 2 (both
non-NBA internationals), 739 players on the board instead of 687.

Deliberate scope: only the **drafted** class is added. The backtest can never project an undrafted debutant (`History` keeps a
player only if drafted by the season's draft or already in game logs), so adding undrafted signees would make the live board
behave in a way no backtest ever measured. Five rostered players drafted in earlier years are also making their NBA debut
(Sorber, Marković, Toohey, Diop, Biberovic); the rookie path is keyed on `draft_year == target` and stashed-then-arrived players
are unprojectable in the backtest for the same reason. They are **not projected**, and the ingest and the refresh summary print
their names instead of hiding the gap.

`preseason_refresh` runs roster -> Summer League/preseason -> ADP in that order (the ADP mapping needs the rookies in `players`),
refreshes only the live season's ESPN payload (ESPN has already wiped 2025-26; history is never re-downloaded), and never lets
one failing step stop the others.

### D5. A stacked residual model, gated by cross-validation

`baseline_offseason` wraps the baseline (any base projector). For every earlier season `s` the base is re-run on
`History.until(s)` (memoised, shareable across variants) and its `proj_fppg` compared with what each player averaged in `s`
(players with 10+ games). Those residuals are regressed on features of the Summer League and preseason **preceding** `s`:
fantasy points per 36 minutes relative to that summer's cohort (minutes-weighted z-score, clipped at 3) shrunk by
`minutes / (minutes + K)`, and minutes per game, each also interacted with a youth weight (1 at age 20 or younger, 0 at 26 or
older) and, for the production term, with rookie status. Ridge strength and `K` are chosen by leave-one-season-out
cross-validation, and **the adjustment switches itself off unless that cross-validation beats "no adjustment" by 0.2%** (tested:
on a planted signal it turns on with 42% gain; on a shuffled control it turns itself off with -0.6%). The fitted adjustment
becomes one per-player multiplier (clipped to 0.6-1.6) on every counting-stat projection, so box-score identities and league
scoring hold exactly, games played and minutes are untouched, and the floor/median/ceiling scale with it.

Why stacking rather than new baseline inputs: the baseline already prices age, draft slot, minutes history and regression to the
mean, so the only honest claim an offseason signal can make is that it explains what is left over, and stacking lets the ablation
test that claim directly against the unadjusted baseline.

### D6. Breakout evaluation, with definitions fixed before looking

`python -m src.backtest.breakouts` (population: players the baseline projected who played 20+ games):

* **Breakout**: actual FPPG beats the *baseline* projection by at least 4 FPPG and at least 25% of it (defined against the base, not
  the model under test, so the comparison is not circular).
* **Useful breakout**: a breakout that also finished in the league's top 169 by season total fantasy points (13 teams x 13 spots).
  Added after the first pass because a 9 to 13 FPPG jump is a breakout by the first definition and still leaves a player nobody
  rosters; it is the outcome a drafter cares about.
* **Young**: age 23 or under. **Under the radar**: no ADP, or ADP rank worse than 100. **Score**: the model's own uplift.
* Reported: subgroup FPPG accuracy (paired bootstrap), AUC and precision@K against each season's base rate (bootstrap over seasons,
  each season is one draft), and a season-by-season list of who was flagged and what happened.

Variants evaluated once each, by rule: Summer League only, preseason only, both, both with only the first half of each
preseason (a draft held mid-preseason), and a richer component set (efficiency, usage, assists, rebounds, stocks, each z-scored)
for Summer League and for both. The richer set was defined up front as the one fair second attempt at giving Summer League a
chance; it was evaluated once and reported whatever happened.

### D7. Rookie availability capped at the top-10 picks' own plateau

`fit_rookie_prior` regressed the games-played fraction linearly on log(pick). Walk-forward (2017-18 to 2025-26, actual games
including rookies who never played) it projected **~78 games for picks 1-3 against ~59 played (+18.2 games, MAE 18.4)**; the
other slots were calibrated. Simmons (0 games), Fultz (14), Williamson (24), Holmgren (0), Wiseman (39) are why. The fix caps the
predicted fraction at the mean fraction of the history's own top-10 picks, estimated from the history like everything else in the
model: picks 1-3 bias +5.1 games (MAE 12.9), every other bucket unchanged (a quadratic and a logit form were tried and lost). The
headline baseline metrics did not move (Spearman 0.784, MAE 422.6).

### D8. Watchlist, information-matched model, calibrated probability

`python -m src.value.breakouts` lists young under-the-radar players with positive uplift: current team (roster snapshot), draft
pick, board rank, ADP, base and adjusted FPPG, the player's own Summer League and preseason lines, and two calibrated
probabilities (of a breakout, and of a *useful* breakout, both conditional on playing 20+ games) from a logistic fit on the
walk-forward, out-of-sample scores. The model is chosen from the evidence that exists: `baseline_summer_league` until any preseason
game of the season exists, then `baseline_offseason`, each with its own calibration. Sort modes include raw Summer League and
preseason z-scores, model-free, so the user can look at who popped without trusting the model. The Streamlit app gets a
**Breakouts** tab (re-filtering is instant; projections are cached) and ADP / ADP-gap columns on the board.

## Results

All numbers: real data, walk-forward, 2016-17 through 2025-26 (10 target seasons), `--leak-check` passed.

**Whole league** (`python -m src.backtest --ablate baseline,baseline_summer_league,baseline_offseason ...`; players projected by
both, paired bootstrap):

| Model | Spearman total FP | Spearman FPPG | Top-100 hit | NDCG@100 | MAE FPPG | MAE total FP | VORP-weighted MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.784 | 0.797 | 0.670 | 0.873 | 4.672 | 422.6 | 669.7 |
| + Summer League only | 0.786 | 0.797 | 0.670 | 0.873 | 4.684 | 422.1 | 668.6 |
| + Preseason only | 0.778 | 0.814 | 0.683 | 0.882 | 4.501 | 412.8 | 645.7 |
| + Summer League + preseason | 0.781 | 0.813 | 0.682 | 0.881 | 4.511 | 413.1 | 646.5 |

Paired, baseline vs Summer League + preseason: MAE on total FP improves by 9.5 (CI [+7.9, +10.9], 8/10 seasons); Spearman on total
FP is **-0.0028** (CI [-0.0047, -0.0009], 3/10 seasons: verdict `hurts`); top-50 hit rate no significant change. Summer League
alone: Spearman +0.0022 and MAE +0.5, both statistically real and practically nothing. **Where the rank-order "hurt" lives**
(diagnosed, not assumed): among players who played 20+ games the layer *improves* Spearman on total FP (0.737 to 0.749, +0.013;
+0.016 for players 23 and under); the whole drop is among the 41% of projected players who barely played, where the ordering of
near-replacement players is noise. Read the headline as: better per-game and VORP-weighted accuracy, neutral-to-positive on the
players who matter, slightly noisier ordering of the ones who do not.

*How to reproduce the 20+ games figures* (re-derived on 2026-09-24 from fresh runs, since the first reports were not kept):
run `python -m src.backtest --model baseline ...` and `--model baseline_offseason ...` over `2016-17:2025-26`, join the two
`players.parquet` files on `(season, player_id)`, keep rows projected by both, and take the mean over seasons of the Spearman between
`proj_total_fp` and `actual_total_fp`. Among players with `actual_gp >= 20` (4,139 player-seasons) that gives 0.7368 (baseline) and
0.7494 (offseason); among those who played 1 to 19 games 0.304 and 0.321. 40.9% of projected player-seasons are under 20 games (29.8%
never played). The +0.016 for players 23 and under comes from the original run and was not re-derived in that check.

**Breakout detection** (base rate of a breakout among young players who played: 21.4%, 216 of 1,010; of a useful breakout: 12.5%).
AUC 0.5 is no skill; "p@10" is the share of each season's top ten by score who broke out:

| Population, useful-breakout target | Model | AUC [95% CI] | p@10 vs base rate | Lift @10 [CI] |
|---|---|---|---|---|
| Young under the radar (base 10.3%) | Summer League + preseason | 0.623 [0.563, 0.684] | 24% | **+13.7pp** [+5.8, +22.1] |
| | Preseason only | 0.637 [0.555, 0.717] | 22% | +11.7pp [+3.6, +19.7] |
| | **Summer League only** | 0.521 [0.461, 0.583] | 15% | +4.7pp [-0.9, +10.7] |
| | + first half of preseason only | 0.606 [0.563, 0.647] | 19% | +8.7pp [+1.5, +15.8] |
| Rookies (base 13.6%) | Summer League + preseason | 0.645 [0.574, 0.715] | 24% | +10.4pp [+3.8, +17.5] |
| | Preseason only | 0.664 [0.574, 0.749] | 29% | +15.4pp [+8.3, +21.6] |
| | **Summer League only** | 0.544 [0.495, 0.595] | 16% | +2.4pp [-4.0, +9.6] |

On the plain breakout target the picture is the same: young under-the-radar AUC 0.583 [0.524, 0.642] with p@10 35% vs 20.6%
(+14.4pp [+3.5, +25.2]); Summer League alone 0.541 [0.511, 0.579] with p@10 lift +3.4pp [-4.9, +11.8].
Restricted to actual Summer League participants the result does not change (AUC 0.548 [0.522, 0.576], p@10 lift +0.4pp).
The richer component set did not help: Summer League only 0.539 / 0.541 / 0.565 (young / under the radar / rookies) versus 0.537 /
0.541 / 0.557 for the simple feature; both contexts 0.571 / 0.587 / 0.614 versus 0.573 / 0.583 / 0.619. It is registered
(`baseline_offseason_rich`, `baseline_summer_league_rich`) as a complete, tested negative result and not recommended.

FPPG accuracy among players who played (MAE gain over baseline, positive is better): all players +0.136 FPPG [+0.093, +0.180];
players with a Summer League or preseason line +0.145 [+0.098, +0.191]; rookies +0.204 [-0.012, +0.421]; young under the radar
+0.023 [-0.096, +0.149] (preseason only: rookies +0.198 [+0.021, +0.372]). The accuracy gain is broad rather than concentrated in the young players (the young subgroup's
gain, +0.058 [-0.063, +0.173], is not significant on its own).

## The finding that contradicts the premise

**Summer League, on its own, is close to noise for predicting a breakout in the data we have.** Ten summers, two feature designs,
two outcome definitions, four subgroups: AUC 0.52-0.57 with lower bounds at or near 0.5, top-ten lifts indistinguishable from
zero. The signal is in the **preseason**: how many minutes a young player is actually given by a coach in October, and how he
produces in them. Plausible reasons (not tested): a Summer League line is 4-6 games against a mix of rookies, G-Leaguers and
fringe players, its usage is set by roster construction rather than merit, and the baseline already prices draft slot, age and
prior minutes, which is most of what a Summer League line implies about a rookie. So the honest shape of the deliverable is: the
ingest and features are built and evaluated, Summer League is in the model and in the watchlist evidence, and the model's own
estimate of its worth is small; the tool becomes useful **when preseason games exist**, which for a draft on 2026-10-17 (NZ morning; fixed 2026-09-26, ADR 0014) they
will (recent preseasons ran from the first days of October to about the 17th). Timing matters: with only the first half of the preseason, the top-ten lift
falls from +13.7pp to +8.7pp.

## Limitations

* **Preseason timing is assumed, not measured.** The backtest gives every season its full preseason (before the opener). A draft
  held earlier sees less; the half-preseason row above is the sensitivity.
* **A modest edge, not a lock.** The top ten flagged players became a useful breakout about 24% of the time against 10%. Most
  flagged players do not break out. The probabilities are calibrated to that, not to hope.
* **Conditional on playing.** A breakout requires 20+ games; availability is modelled separately and is still about 70% of the
  biggest misses.
* **Multiple looks.** Six variants, three populations and two targets were examined. The headline claims are the ones stable
  across them (preseason drives the signal; Summer League alone does not); no single CI should be read in isolation.
* **Not modelled:** rookies drafted in earlier years who debut now (5 players this season), undrafted signees (the backtest cannot
  score them either), scouting, college or international production, and injuries during the preseason.
* The July 2026 Summer League has free throws missing from its FT counters (D3); fantasy points use `pts`, so the effect on the
  layer is small, but that summer's z-scores are computed from the source's own numbers.

## Reproduce

```bash
python -m src.ingest.preseason_refresh                                   # roster, Summer League + preseason, ADP
python -m src.backtest --ablate baseline,baseline_summer_league,baseline_offseason --benchmark naive_last_season \
    --adp-file ~/dev-data/nba-fantasy-2026/processed/adp.parquet --seasons 2016-17:2025-26 --n-boot 1000 --leak-check --out reports
python -m src.backtest.breakouts --model baseline_offseason --save-calibration       # and --model baseline_summer_league
python -m src.backtest.breakouts --model baseline_preseason                          # the decomposition rows
python -m src.backtest.breakouts --model baseline_offseason --preseason-fraction 0.5
python -m src.value.breakouts --season 2026-27 --out reports/watchlist.csv
```
