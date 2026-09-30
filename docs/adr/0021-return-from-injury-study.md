# ADR 0021: Return-from-injury study: does the availability model over-discount a long absence followed by a healthy tail?

**Status:** accepted, 2026-09-26 (the definitions below were committed before any outcome was computed; results follow them)
**Code:** `src/backtest/return_report.py` (analysis only; no model, board or contract changes).
**Evidence:** real walk-forward run over 2016-17..2025-26, `reports/return_<date>/` (gitignored; the command below reproduces it).

## Context

On the 2026-27 draft board (`baseline_offseason_debut`, which wraps plain `baseline`) Jayson Tatum is projected at 50.1 games and 41.0
FPPG (board rank 53, ADP 9.7). Last season he played 16 games, all after returning from an Achilles injury: his first game was
2026-03-06, and he then played 16 of Boston's remaining 20. `AvailabilityModel` (ADR 0003) sees only a *season-level* fraction of the
schedule played (16/82 = 0.20) plus its recency-weighted mean, last-season value, worst-season value, age and projected minutes. It has
no notion of "a long absence, then healthy". The injury layer (ADR 0006) adds streak length and frequency, but its features are also
season-level summaries and it is not part of the board model.

The metric that matters for a points-league draft is **expected total fantasy points** (`proj_fppg x proj_gp`, and value over
replacement built on it), not FPPG: a high-FPPG player who does not play is worthless. So this study judges everything on games-played
(GP) error and total-FP error.

## Question

For a player in Tatum's position (one long absence block, then a healthy tail), does the current `baseline` availability model
under-project next season's games and total fantasy points, and by how much? Is the discount too harsh?

## Pre-registered design (fixed before any outcome was inspected)

Only cohort *sizes* and a name list were looked at while building the code; no projection error, actual GP or total FP was computed
for any cohort before this section was committed (git history: the pre-registration commit precedes the code).

**Model under test:** `baseline` (the plain model; the board model adds only Summer League/preseason/debutant context on top and is
run as a secondary check). Walk-forward over 2016-17..2025-26 with the existing harness conventions: the projection for target season
`s` comes from `History.until(tables, s)`; every cohort feature is computed from that same history (seasons `< s`), the *prior* season
`p = s-1` in particular; `s` itself only supplies the outcome.

**Universe:** player-target-seasons that `baseline` projected, in which the player appeared in season `p`, had at least one NBA season
before `p` (`players.from_year < p`, so a debut year is not "absence"), played for a single team in `p` (a mid-season trade
misattributes the earlier team's games as missed, ADR 0006 limitation), and averaged at least 15 minutes per game in `p`. Zero-game
target seasons count as 0 GP and 0 FP (the harness rule).

**Per-player season profile (`p`):** the primary team's schedule from `team_games`, in date order (`L` games); each game is
played/missed. The *block* is the longest run of consecutive missed games.

**Primary cohort (Tatum-like, "returner with a healthy tail"), all of:**

1. the block starts at the first game of the season (the player's first appearance came after the block: `lead` type) and has
   length `>= 25%` of `L`;
2. the tail (team games from the first appearance to season end) is `>= 15` games;
3. the player played `>= 75%` of the tail (`healthy tail`).

Together these say the absence is one contiguous block and everything after it is healthy. Tatum (block 62/82 = 76%, tail 16/20 = 80%)
is inside.

**Sensitivity grid (reported in full, never tuned on):** block threshold `T1 in {25%, 40%, 60%}` x tail health `X in {60%, 75%,
90%}`, plus the `mid` type (the block does not start at game 1 and does not reach the last game: the player was playing, missed one
long block, and returned; health is judged over all games outside the block, tail `>= 15`). Also: including multi-team players,
including players under 15 mpg, excluding the 2020-21 target (the 2019-20 bubble season makes "games missed" partly non-injury), and
the tail-length threshold at 10 and 25.

**Matched controls (same target season):** players not in the cohort with a *spread* absence (longest missed run `<= 60%` of their
missed games, i.e. many short absences rather than one block), matched by k=3 nearest neighbours (with replacement) on prior-season
fraction played (caliper 0.10), age (caliper 2 years) and prior-season minutes per game (caliper 5). Sensitivity: controls = any
non-cohort player. Cohort rows with no control inside the calipers are dropped from the matched comparison and counted.

**Outcomes (signed *actual minus projected*; positive = the model under-projected):** GP, total FP; also MAE of both, the share of
cohort rows with `actual_gp > proj_gp`, and FPPG bias among those who played `>= 10` games (reported, secondary).

**Inference:** percentile bootstrap, 95%, 2000 resamples, resampling rows within each season (stratified by season, as ADRs 0004 and
0006); the cohort-minus-control difference resamples cohort rows together with their matched controls. The verdict rule is fixed:
"discount too harsh" requires the paired cohort-minus-control **total-FP** bias CI to exclude zero on the primary cohort with the
positive sign; an interval that includes zero is reported as inconclusive with its `n`, not as "no effect". Grid cells are reported as
they fall; there is no selection of a favourable cell.

**Informational table:** players currently (history through 2025-26) in the primary cohort, their board `proj_gp` and the
cohort-implied correction. It changes nothing.

## Decisions

* **D1. Analysis only.** No model, board or contract changes. The module reads the walk-forward projections of the registered
  projectors and never writes to them. A correction, if one were warranted, would be a separate ADR with its own ablation.
* **D2. Pre-registered primary cohort** (above). Everything else is a sensitivity cell, reported whole.
* **D3. Two comparators.** *Matched controls* answer "is the cohort treated differently from players with a similar season total?";
  the *raw* cohort bias answers the question the draft cares about, "is the cohort's projection wrong on average, and by how much?".
  They are not the same question (see the verdict).
* **D4. Bootstrap unit.** Rows are resampled within each target season and pooled (not season means averaged), because cohorts are 1
  to 11 rows a season and averaging tiny season means would over-weight a season with one player. The CIs are over players, not
  seasons, so they understate season-level dependence (see Limitations).
* **D5. Point-in-time features.** `prior_features(History.until(tables, s))`: the same history object the projector receives, so the
  features and the projection cannot disagree about what is known. `tests/backtest/test_return_report.py` scrambles every row of
  season `>= s` in place (`scrambled_future`, the tool behind `assert_projector_ignores_future`) and separately extends the tables
  with a later season; the features are identical in both cases.

## Result (real walk-forward, 2016-17 through 2025-26, model `baseline`)

Reproduce: see the last section; the run takes about four minutes. Study universe: 5,386 projected player-seasons. Over *everyone*
the baseline over-projects by 9.6 GP and 129 total FP (its `proj_gp` is conditional on playing, so zero-game seasons and
retirements are pure misses), so every comparison below is read against that background.

**Primary cohort** (lead block >= 25% of the schedule, then >= 15 tail games at >= 75% health; veteran, single team, >= 15 mpg):
**n = 60** over 10 seasons (1 to 11 a season); 37 of them have a matched control (pool of 1,436 spread-absence players). Cohort means:
projected 45.9 GP, actual 46.9 GP; projected 912 total FP, actual 974.

| metric (actual - projected, positive = under-projected) | cohort [95% CI] | matched controls | cohort - controls [95% CI] |
|---|---|---|---|
| bias, games played | **+1.0 [-4.1, +5.8]** | -8.8 [-14.3, -3.7] | +10.7 [+3.5, +17.7] |
| bias, total FP | **+62 [-80, +196]** | -133 [-261, -11] | +251 [+70, +438] |
| MAE, games played | 20.3 | 22.1 | -2.6 [-7.4, +2.0] |
| MAE, total FP | 526 | 521 | ~0 (under 1 in magnitude) [-127, +126] |
| share with actual GP > projected | 62% [50%, 72%] | 45% [34%, 57%] | +17 points [+1, +32] |
| bias, FPPG (>= 10 GP; 31 paired) | -0.11 [-1.45, +1.25] | -0.14 | +0.40 [-1.51, +2.24] |

Per season the cohort's GP bias was +13.7, +8.4, +16.3, **-31.0**, +6.4, -6.1, +7.1, +5.2, +19.6 (n = 1), +0.4 for 2016-17 through
2025-26 (n = 4, 7, 5, 7, 4, 8, 7, 6, 1, 11): eight of ten seasons positive, but the 2019-20 target season (the bubble year; 7 cohort
players, including a retirement and a second ACL tear) is far below the rest. Leaving each season out in turn (a fixed procedure,
added after seeing this table and not part of the verdict) moves the raw cohort bias between -0.4 and +5.2 GP (dropping 2019-20:
+5.2 [-0.3, +10.4] GP, +183 [+36, +340] FP) and the paired excess between +9.0 and +13.6 GP; the FP-excess CI includes zero only
when 2025-26 is dropped.

**Sensitivity grid** (18 cells; the full table is in the generated report). The raw cohort GP bias sits between -3.4 and +10.1 across
cells and its CI excludes zero in only two cells (lead block >= 60% with tail >= 75%: +8.0 [+0.4, +15.5], n = 21; and the mid type
with block >= 40%: +5.7 [+0.2, +10.6], n = 13). The paired total-FP excess CI excludes zero, positive, in 9 of 18 cells (8 of 17 without the mid cell with block >= 25% and tail >= 90%, which the pre-registered grid did not list and which I added afterwards; it is one of the 9); the cells
with the longest blocks have only 6 to 10 matched controls and are inconclusive on the paired test. The cell closest to Tatum (block
>= 60%, tail >= 75%) has raw bias +8.0 GP and +210 total FP [+25, +385], but its paired excess (+7.7 GP [-3.3, +18.7], +121 FP
[-77, +311], 9 matched) is inconclusive, and it is one of 18 cells: a hint, not a finding. Using any non-cohort player as the control
(rather than spread-absence players) shrinks the excess to +4.4 GP [-2.5, +11.3] and +87 FP [-111, +295], also inconclusive.

**Calibration.** In projected-GP deciles the whole universe is over-projected in every decile (-18 GP in the lowest, -4.5 in the
highest). The cohort sits in deciles 3 to 6 (projected 38 to 56 GP), where its bias is +4.6, -3.6, +2.1 and +6.5 GP (n = 18, 19, 12,
8) against -13.3, -10.6, -10.7 and -7.3 GP for those deciles overall: roughly on the line while its neighbours lie below it. The
other five cohort rows sit elsewhere and do not look on the line: decile 2 (n = 1, -3.4 GP) and deciles 7 and 8 (n = 1 each, -15.4 and
-13.8 GP). By projected-GP band the cohort's bias is -7.9 (n = 2), +2.0 (n = 39) and -0.1 (n = 19) GP.

**Other projectors, same primary cohort.** `baseline_injury`: bias +0.6 GP [-4.6, +5.4], +54 FP. `baseline_offseason_debut` (the board
model): +1.0 GP [-4.1, +5.8], +52 FP. Neither changes the picture; the streak features of ADR 0006 do not express "healthy since
return" either.

**Tatum and the current cohort** (informational, from data through 2025-26). Nine players qualify: Jayson Tatum (block 62 of 82
games, tail 16 of 20), Scoot Henderson, De'Anthony Melton, Grant Williams, Max Strus, Killian Hayes, Micah Potter, Josh Green and
Cameron Payne. Tatum's board projection is 50.1 GP, 41.0 FPPG, 2,053 total FP. The cohort-implied correction is **+1.0 GP (about +41
FP)** if the cohort's raw bias is applied, and at most **+10.7 GP (about +437 FP)** if the whole excess over matched controls were
applied, which the data do not support as an absolute correction (see the verdict). The top of the CI on the raw bias, +5.8 GP,
would put Tatum near 56 GP (about +240 FP).

## Verdict

**By the rule fixed in advance the answer is "too harsh": the cohort-minus-control total-FP bias is +251 [+70, +438]. That statement
is about the model's treatment of the cohort relative to comparable players; it does not mean Tatum is under-projected.**

1. **In absolute terms the discount is about right, not too harsh.** For the primary cohort the baseline's projected games are within
   one game of the truth on average (+1.0 GP, CI [-4.1, +5.8]) and total FP within 62 (CI [-80, +196]). Both CIs include zero and the
   upper end is about 6 GP. A Tatum-like player is not systematically under-projected by this model.
2. **The relative gap is real and comes from the other side.** Matched players with the same season total but spread-out absences
   are *over*-projected by 8.8 GP; the cohort is not. The model gives both the same discount for a low games-played fraction, but a
   contiguous injury followed by health carries no extra risk while spread-out absence (a marginal or injury-prone player) does.
   Encoding "long absence, then healthy" would therefore mostly help by taking games away from the controls, not by adding them to
   the cohort.
3. **A Tatum-specific hint exists and is not established.** The longest-block cell (>= 60% of the season missed, n = 21) shows +8.0 GP
   [+0.4, +15.5] raw. It is one cell of 18, its paired test is inconclusive, and Tatum (a star with an Achilles injury) is not the
   average of that cell, which holds many role players and non-roster signings.
4. **Recommendation:** do not change Tatum's projection or add a model term on this evidence. If the board is later given a return
   flag it should be advisory (like `risk_flags`), sized by judgement, not derived: the >= 60% cell's +8.0 GP [+0.4, +15.5] is one of 18 cells and its paired test is inconclusive, and the
   >= 25 mpg rows are over- rather than under-projected, so a figure of a few games (about +5, roughly the midpoint of the cell and the
   cohort's +1.0) is the most the data could be argued to support, with the CI shown.

## Limitations

* **Small cohorts.** n = 60 (37 matched) for the primary cohort, 1 to 11 a season; the ten-season total is what gives any power, and
  the CIs are wide (the raw GP bias could be anywhere from -4 to +6 games).
* **Non-injury absences are in the cohort.** "Missed the first games" includes players who signed mid-season, were waived, sat out
  contract disputes or were in the G League; `game_logs` cannot separate these from injuries (the injury-report PDFs start
  2018-12-19, ADR 0005). A debut year is excluded (`from_year < p`) and a >= 15 mpg filter removes most fringe players, but the
  result is about "long absence, then healthy" as the data can see it, not about injury type. How many cohort rows are real injuries to established players
  was not classified. What the frame does show: 12 of the 60 averaged >= 25 mpg in the prior season and 36 averaged >= 20 mpg; 4 played
  no game the next season. The 12 highest-minutes rows were over-projected (-7.5 GP, n = 12) while the other 48 were under-projected
  (+3.1 GP), so the mix matters and a Tatum-like star is closer to the first group (descriptive, not tested).
* **Team attribution.** Single-team players only in the primary cell (a trade makes the earlier team's games look missed); including
  multi-team players moves the raw bias to -3.4 GP [-7.3, +0.1] with an inconclusive paired excess, a mixed group by construction.
* **Season structure.** 2019-20 (a suspended season; bubble teams played different numbers of games) and 2020-21 (72 games) make
  "team games" and next-season targets irregular; excluding the 2020-21 target changes nothing, but the 2019-20 target moves the raw
  bias by about 4 GP.
* **Controls are a choice.** The "spread absence" rule and the calipers (0.10 fraction, 2 years, 5 mpg) were fixed in advance; the
  alternative (any non-cohort control) gives a smaller, inconclusive excess. The paired test uses only the 37 matched of 60 rows.
* **Bootstrap over players, not seasons.** It ignores season-level shocks (a league-wide injury year); the per-season table is the
  safeguard.
* **The pre-registered rule tests a relative statement.** A cleaner rule for the draft question would have used the raw bias. The
  rule was left as registered, and the raw bias is reported beside it and drives the verdict text.
* **The projection under test is `baseline`,** not a board with risk haircuts; `risk_gp` on the board is advisory and 0 for Tatum.

## Reproduce

```
python -m src.backtest.return_report --seasons 2016-17:2025-26 \
    --board reports/daily/draft_board_baseline_offseason_debut.csv --out reports/return_2026-09-26
python -m pytest tests/backtest/test_return_report.py
```

The report (`return_report.md`), `primary.csv`, `sensitivity.csv`, `calibration.csv`, `current_cohort.csv` and the row-level
`study_frame.parquet` are written to `--out` (gitignored). `--compare` selects the other projectors and `--model` the one under
test; without `--board` the current projection is the live model's. Only local data is read; nothing is downloaded.

## Consequences

* A reusable, leakage-tested analysis (`src/backtest/return_report.py`) exists for re-running with any projector, cohort definition
  or, in season, new data.
* No model or board output changed; the draft board's Tatum row (rank 53, 50.1 GP) stands.
* Open, precisely: whether a Tatum-scale (>= 60% block) subgroup is under-projected by 5 to 10 games needs more seasons, or injury
  type from a source not yet available, to settle; this sample cannot.
