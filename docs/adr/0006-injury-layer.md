# ADR 0006: Injury/availability feature layer

**Status:** accepted, 2026-09-23
**Code:** `src/features/injury.py`, `src/models/injury_baseline.py`; small additive changes to
`src/models/availability.py`, `src/models/baseline.py`, `src/models/registry.py`.
**Evidence:** real walk-forward ablation, `reports/` (gitignored; command below reproduces it),
README's [Results](../project-overview.md#results) table.

## Context

The real backtest (ADR 0004's harness, first run 2026-09-22) found the baseline projector loses to
naive-last-season on total-FP MAE (paired comparison, `hurts`), and that ~70% of the miss error in
the biggest misses is availability-driven: projecting games played for a player who then gets hurt
or DNPs, or the reverse. `docs/modeling.md` section 10 had documented "injury layer" only as an
extension point. PLANNING.md's injury row lists: games missed per season, streak length and
frequency, recency-weighted absence rate, age-adjusted absence rate, and a "coming off a long
absence" flag. ADR 0005 (data sources) found the NBA official injury-report PDF archive only
covers 2018-12-19 onward -- too short a window for most of the 2016-17..2025-26 backtest -- and
recommended the games-missed-derived signal from `game_logs`/`team_games` (available for all 11
ingested seasons) as the backbone, with the PDF archive as a later, seasons-limited enrichment.

## Decisions

### D1. Derive the signal from data already ingested; no new raw source for the primary layer

`team_games` minus `game_logs` already gives, for every season back to 2015-16, exactly which
games a player missed (ADR 0001 rule 5). The *only* new computation needed is the **shape** of
those misses over time -- streak length and frequency -- which a season-total games-missed count
cannot express. `src/features/injury.py` builds this from `History` alone (`build_injury_panel`):
for each player-season, find the player's primary team (most games played for that season),
walk that team's full schedule in date order, and mark each game present/absent. This is a pure
function of already-leakage-safe `History` fields, so it needed no changes to the data contract
and no new table.

**Not pursued as the primary source**: the NBA injury-report PDF archive (ADR 0005 D2), because it
only covers seasons from 2018-12-19, which would make most of the 10-season backtest window
(2016-17 through 2018-19, three of ten seasons) unable to use it at all, and mixing "has real
injury type" and "has only derived signal" seasons inside one ablation variant would confound the
result with coverage rather than model quality. Kept as a documented, not-yet-built option for a
later, seasons-restricted enrichment (D5).

### D2. Four extra features, chosen to match PLANNING.md's injury row exactly

`InjuryFeatures` (`src/features/injury.py`) turns the per-season streak panel into four
recency-weighted, point-in-time features for `AvailabilityModel`:

1. **`streak_bar`** -- recency-weighted longest-absence-streak fraction of the schedule (decay =
   `BaselineConfig.availability_decay`, same as everywhere else in the baseline). This is the
   "length" half of PLANNING.md's ask.
2. **`freq_bar`** -- recency-weighted count of distinct absence streaks. The "frequency" half: a
   player who misses 20 games in one stretch (`streak_bar` high, `freq_bar` low) and a player who
   misses 20 games in ones and twos (`streak_bar` low, `freq_bar` high) now look different to the
   model, where before both only reduced `f_bar` identically.
3. **`long_absence`** -- binary, 1.0 if last season's longest streak was >= 15% of the schedule
   (~12 of 82 games), else 0.0. Exactly PLANNING.md's "coming off a long absence last season" flag.
4. **`age_residual`** -- a player's own recency-weighted absence rate minus a population-level
   `missed_frac ~ a + b*age + c*age^2` curve, fit by weighted least squares once per history (not
   per player) so a rookie or a player with only one season of history still gets a sensible
   age-typical baseline rather than NaN. Positive = missing more than is typical for his age.

Deliberately **not** included: injury *type* (needs the PDF archive, D1) and days-since-last-injury
as a separate continuous feature (redundant with `f_last`, already in the base 7-column feature
matrix, and `streak_bar`/`long_absence` together capture "how bad, how recent").

### D3. Wire in through `AvailabilityModel`'s feature matrix, not a parallel model

`AvailabilityModel.fit`/`predict_mean` (`src/models/availability.py`) gained an optional `extra`
matrix parameter, appended to the existing 7 columns (`f_bar, f_last, f_min, absent, age, age^2,
mpg`) before scaling and fitting; `n_extra` is stored on the fitted model so predict-time width
always matches fit-time width. Passing `extra=None` (what plain `BaselineProjector` always does)
reduces to exactly the old code path -- checked by the full existing test suite passing unchanged,
plus `_with_extra`'s own unit coverage.

`FittedBaseline` gained one optional field, `injury_features` (default `None`), and `_core`'s one
line that computes `mu_f` now passes `extra=self.injury_features.build(...)` when it is set.
`BaselineProjector` gained one hook method, `_build_injury_features(history, sub)` (default:
returns `None`), which `BaselineInjuryProjector` (`src/models/injury_baseline.py`) overrides to
build a real `InjuryFeatures` from the history. No other baseline component (minutes, per-minute
rates, volatility, rookie prior) changes at all, and rookies (who have no availability-model
history to begin with -- they come from the draft-slot prior) are provably identical between
`baseline` and `baseline_injury` (tested).

This was chosen over a **parallel/ensembled availability model** (fit a separate injury-only model
and blend its `mu_f` with the base one) because the shrinkage, Beta-spread and calibration
machinery already in `AvailabilityModel` is exactly what four extra correlated features need --
duplicating it would have been more code with no obvious benefit, and blending weights would be
one more thing to tune on eval data (ADR 0004's warning).

### D4. Registered as `"baseline_injury"`

`src/models/registry.py` adds `"baseline_injury" -> BaselineInjuryProjector`, so
`python -m src.backtest --ablate baseline,baseline_injury ...` and the draft board both pick it up
by name with no other wiring.

### D5. PDF injury-type enrichment: not built, and correctly so given the primary result (D-Result)

Per the task brief, this was only worth building if it could be shown to add lift over the
derived-from-game-logs signal on the seasons where both exist (2018-19 onward), and only after the
primary layer's own result was in. Given the result below (no lift from the primary,
already-cheap-to-add signal), spending the remaining budget on parsing and point-in-time-aligning
a PDF archive for a *narrower* season window, in the hope it beats a layer that itself showed no
lift, was not a good bet. **Not built.** If a future ablation of the primary layer showed real lift
and the team wanted to push further, D5 in ADR 0005 has the parsing approach already scoped.

## Result (real ablation, 2016-17 through 2025-26)

Reproduce: `python -m src.backtest --ablate baseline,baseline_injury --seasons 2016-17:2025-26
--leak-check --out reports/`. Ran on 2026-09-23; report at
`reports/baseline_injury_2016-17_2025-26_b30db614/report.md` (generated reports are gitignored;
numbers below are copied from that run). `--leak-check` passed on the last season.

**Honest read.** The injury layer gives a real, statistically significant lift over plain
`baseline` on every metric tested -- and it is small. Paired bootstrap over players, stratified
by season (positive = `baseline_injury` better; same methodology as the README's baseline-vs-naive
comparison):

| metric | lift | 95% CI | p(lift<=0) | seasons won | verdict |
|---|---:|---|---:|---:|---|
| Spearman, total FP | +0.0009 | [+0.0002, +0.0018] | 0.006 | 5/10 | improves |
| Top-50 hit rate | -0.0020 | [-0.0120, +0.0120] | 0.487 | 1/10 | no significant change |
| MAE, total FP | +1.2297 | [+0.7551, +1.7409] | 0.002 | 7/10 | improves |
| MAE, games played | +0.1004 | [+0.0751, +0.1308] | 0.002 | 8/10 | improves |

(MAE, games played is not one of the harness's three default significance metrics; it was added to
the significance set for this ADR specifically because the task brief and README's Results section
both call out GP/availability error as the metric that matters most here -- computed with the same
`src.backtest.ablation.run_ablation`, `n_boot=500`, `seed=0`, on the same two variants and seasons.)

Every CI excludes zero on the metrics that moved, so this is not noise -- but look at the
magnitude, not just the stars: MAE on games played drops from 19.24 to 19.14 (about **0.5%
relative**), and MAE on total FP drops from 422.6 to 421.4 (about **0.3% relative**). Spearman rank
correlation is essentially unchanged (+0.0009 on a base of 0.784) and only wins 5 of 10 seasons
outright (the CI is still entirely positive because the bootstrap is over players within each
season, not across seasons -- a small, consistent per-player effect can be significant even when
season-level point estimates split close to 50/50). Top-50 hit rate shows no significant change at
all.

**What this means in practice.** The absence-streak/recency/age features do capture *something*
real about availability that the base `f_bar/f_last/f_min` features did not -- the direction is
right and consistent (8/10 seasons for games-played MAE, not a coin flip) -- but they are not the
fix for the ~70% availability-driven miss error the README's Results section documents. Four extra
columns nudging a logistic regression's probability of missing games is a small lever on a problem
that is fundamentally about *unpredictable* events (an ACL tear in December is not foreshadowed by
last season's absence pattern in any of these features, and can't be from box-score data alone).
The biggest-misses table in the generated report still shows availability as the dominant miss
category (69.5% of the top misses' absolute error, materially unchanged from the pre-injury-layer
baseline's ~70%) -- this layer measurably helps, it does not solve the problem PLANNING.md's
roadmap flagged it to solve.

**No tuning was done to reach this number.** `LONG_ABSENCE_THRESHOLD` (0.15) and the reuse of
`BaselineConfig.availability_decay`/`n_lags` for the injury features (rather than a separate
search grid) were fixed before this ablation ran, on modeling-approach grounds (see D2), not
adjusted afterward to improve the result. This is the first and only real ablation run for this
layer.

## Consequences

* `AvailabilityModel` and `FittedBaseline` carry small, additive, backward-compatible API surface
  area beyond what ADR 0003 originally shipped; every other consumer of `BaselineProjector` is
  unaffected (`extra`/`injury_features` default to `None`/off).
* `docs/modeling.md` section 10 now describes the injury layer as implemented, with the roster and
  contract layers still open extension points.
  *Note, 2026-09-25:* ADR 0013 built the rookie-scale contract proxy and it showed no lift; that closes the contract row for what is testable with the available data.
* The known team-attribution approximation (a traded player's absences are computed against his
  season-primary team, misattributing games around the trade) is accepted, not fixed, because the
  current data contract has no point-in-time roster-tenure table to do better with; flagged for
  whichever future ADR adds one (naturally the roster-context layer's territory).
