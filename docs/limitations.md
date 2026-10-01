# Known methodological limitations

What the backtest and the projections do not (yet) establish. Numbers are from the real-data run in
[ADR 0030](adr/0030-methodology-critique-fixes.md).

## The model does not beat ADP at the top of the board

On the ADP universe, ADP-only (a fit of season FP on log-ADP) has a higher top-12/24/50/100 hit rate and value capture than the
baseline model (top-12 hit 0.583 vs 0.528; top-50 0.722 vs 0.653). Adding the model to ADP does not detectably improve on ADP for
ranking, only for total-FP error. Use ADP as the anchor at the top; the model adds depth, minutes, games-played and floor/ceiling.
Nine comparable seasons is little, so top-12 differences are not statistically resolvable on their own.

## Availability: the plain baseline is conditional on appearing; the hurdle models are not (ADR 0031)

`src/models/availability.py` fits games played only on player-seasons with at least one game, so the plain `baseline`'s `proj_gp` is biased high for anyone
with a real chance of not appearing. Backtest by risk group (`--method-checks`), plain baseline: returners after missing over half the previous season
appear in only about 41% of cases and are projected 15.3 GP too high; part-season players +9.5 GP; rookies +3.1. The `baseline_hurdle*` models add a
first stage, P(appear), and project the unconditional expectation: returners +1.5 GP, part-season +2.3, total-FP bias +134 to +14 (ADR 0031). What is
still not modelled: **rookies' and debutants' own chance of not appearing** (about 10% of drafted rookies never play; they keep their draft-slot prior),
and retirements or moves overseas before opening night beyond what the games-missed history predicts. "Prior injury" is still proxied by games missed
for the models without labelled data; the injury-report features (ADR 0033) add cause and duration from 2018-19 on.

## ADP-aware models read the target season's own ADP

`baseline_adp*` and the board stack use ADP as a preseason signal (ADR 0032). ADP of the target season is information available before the season, so
it is not leakage, but it makes these models depend on an ingested ADP table: without one they degrade to their base model. Historical ADP rows come
from ESPN's player feed as ingested (2025-26 was rebuilt from FantasyPros because ESPN's was wiped, docs/research/data-sources.md); their exact
timestamp relative to each draft is not recorded, so the backtest assumes they are draft-time values. The blend's coefficients are refit from completed
seasons, so they lag the market's current mix of information by construction.

## Positions are not point-in-time

Historical position groups come from the current `players.position` label; no dated position history is ingested. A `baseline_nopos`
ablation (all players `U`) matches the shipped model (Spearman 0.785 vs 0.784, MAE total FP 422.8 vs 422.5), so any leak through the
position priors has no measurable effect; the priors also show no measurable benefit. A point-in-time position source would therefore have nothing to
recover, and none is built (ADR 0033 records the decision).

## Shrinkage tuning

Rate shrinkage (decay, kappa) is tuned on the training history using the minutes model's predicted MPG, as forecasting does
(previously the realised MPG; no material change). Age curves, kappa and decay are still estimated on the same history that is then
projected from, without a nested holdout; the walk-forward outer loop keeps this from touching the scored season.

## Season-total intervals (ADR 0033): calibrated for the hurdle models, conservative for returners, too narrow for rookies

`fppg_p10/p90` are game-level quantiles (calibrated, see docs/backtest.md). The hurdle models also emit `proj_total_fp_p10/p50/p90`: a deterministic
simulation of season-mean FPPG (a fitted relative misprojection spread by seasons of history, plus game noise) times games played (the hurdle mixture).
Out of sample (walk-forward, ten seasons) the nominal 80% band holds 85% of realised totals overall, 82% for prime players, about 90% for returners
(conservative: a zero floor and a wide top), and only 68% for rookies, whose chance of not appearing is not modelled and whose talent spread is the largest.
Season-mean FPPG and games played are treated as independent. The plain `baseline` has no season-total band. The board rank is still a point-estimate
ordering of expected season totals.

## Few seasons, dependence between them

Headline results average about ten seasons, and bootstrap intervals over players are optimistic (a player appears in several
seasons). The report's leave-one-season-out table shows how far the means move when one season is dropped (Spearman 0.777 to 0.790,
top-12 hit 0.519 to 0.546, top-50 hit 0.647 to 0.664); paired ADP comparisons use season-level t intervals for that reason.
