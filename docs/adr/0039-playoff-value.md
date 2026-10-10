# ADR 0039: Fantasy-playoff value (advisory) and a late-season availability study

**Status:** accepted, 2026-10-10
**Code:** `src/value/playoff_value.py` (board overlay), `src/backtest/playoff_availability.py` (study); wired into
`src/value/board.py` (CLI and `daily_refresh` board CSVs), `src/app/loader.py`, `src/app/state.py`, `src/app/glossary.py`.
**Follows:** ADR 0015 (matchup calendar, `src/inseason/weeks.py`; schedule, `src/inseason/schedule.py`), ADR 0023/0024 (advisory
overlays that never change a rank), ADR 0031/0038 (hurdle `proj_gp` with P(appear)).

## Context

The league is H2H points; 8 of 14 teams play the fantasy playoffs, which are the league's last `sum(playoff_rounds_weeks)` = 3 matchup
weeks. On 2026-10-10 ESPN reports 19 regular-season matchups (`matchupPeriodCount`) and `finalScoringPeriod` 160, so the playoffs are
weeks 20-22, 2027-03-08 to 2027-03-28 (ESPN's per-period list is still the one-day placeholder, so the dates come from the documented
derivation in `weeks.py`; it ends on the league's final scoring day exactly). The board ranks by projected season total and ignores
which players' teams play more in those weeks, and whether some players reliably fade late in the season (shutdowns on tanking
teams, accumulated load).

## Decisions

**D1. Part A: an advisory playoff value on the board, never in the rank.** Per player: `playoff_games` = his team's NBA games in the
playoff weeks (the matchup calendar from `schedule.build_calendar` over the stored `schedule_games`; no second calendar),
`playoff_avail` = `clip(proj_gp / season_games, 0, 1)` and `playoff_fp = proj_fppg x playoff_games x playoff_avail`, plus
`playoff_team`. The availability term is the model's own expected share of team games played (it already holds the hurdle P(appear)
and, for rookies, ADR 0038's), spread evenly over the season; no extra haircut is invented. The team is the latest roster snapshot of
the season, else last season's team (the risk overlay's rule). The columns are attached after the board is built, as ADR 0023/0024's
overlays are, so `proj_*`, `vorp`, `rank` and `tier` stay byte-identical (`tests/value/test_playoff_value.py`). Missing schedule or
team degrades to NaN and a note.

**D2. Part B: does late-season availability fall below the model's rate in a way pre-season features predict? Pre-registered before
any result was computed.**

- *Historical playoff weeks.* For each past season the same `weeks.py` fallback (`calendar_from_settings` with no ESPN payload: Monday
  to Sunday weeks from opening night, All-Star week merged, 19 + 3 periods) is applied to that season's `team_games` schedule. A season
  whose playoff weeks average fewer than 8 team games is not scored (expected: 2019-20, suspended in March 2020, and 2020-21, which
  started on 2020-12-22 so 22 weeks run past its end).
- *Universe (pre-season knowable).* Rotation veterans: at least 41 games and 20 mpg the season before, projected by the board model
  `baseline_hurdle_adp_depth_rookie_offseason_debut` (walk-forward, `src.backtest.harness.walk_forward`). Players who then play no game
  stay in (outcome 0).
- *Baseline (what Part A shows).* Expected playoff-week games `E0 = team_po x clip(proj_gp / season_games, 0, 1)`, with `team_po` the
  games of his previous-season team in the target's playoff weeks and `season_games` the board's own schedule length.
- *Label for training.* The share gap `G` = his share of team games played in the playoff weeks minus his share over the regular weeks
  (single-team seasons; a season with no games has `G = 0`).
- *Candidates.* (1) intercept only: `G_hat` = mean `G` of the training rows. (2) OLS of `G` on an intercept plus, all from
  `History.until(target)`: his team's previous-season win% (tanking proxy), age, his own previous-season `G` (0 when unknown), a star
  flag (previous-season FPPG in the top quarter of the universe) times bad team (win% under .400), previous-season mpg and total
  minutes. Both predict `E1 = team_po x clip(avail + G_hat, 0, 1)`.
- *Walk-forward.* Target S is fitted on universe rows of scored seasons strictly before S (features need a previous season, so
  2016-17 is the first row season and the first target with training data is 2017-18; a target with no training rows predicts the
  baseline and is not scored).
- *Metric and rule.* Primary: squared error of playoff-week games per player (the advisory value is an expectation, and the candidates
  are fitted as means, so squared error is the matching score; MAE rewards the median and would penalise a correct mean shift on a
  zero-inflated count). Paired bootstrap over players stratified by season (`src.backtest.metrics.bootstrap_groups`, 2000
  replicates, 95%). **A candidate is adopted only if (a) the CI of the MSE reduction lies entirely above zero, (b) it lowers MSE in at
  least 70% of the scored seasons, and (c) the CI of its MAE change is not entirely on the worse side.** If candidate (2) passes it is
  used, unless candidate (1) also passes and (2) does not beat (1) with a CI above zero, in which case the simpler (1) is used. If
  neither passes, Part A keeps the plain model availability. Secondary (reported, not decisive): MAE and MSE of playoff-week fantasy
  points (`proj_fppg x E` against the realised FP in those weeks).

**D3. Result.** Pending (filled in by the run below).

## Consequences

- Part A is a tie-breaker for a team expecting to make the playoffs, not a re-ranking; the main rank is unchanged by construction.
- The playoff-week dates rest on the derived calendar until ESPN publishes real periods; `daily_refresh` rebuilds the board CSVs, so a
  published calendar is picked up automatically.
