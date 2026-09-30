# Known methodological limitations

What the backtest and the projections do not (yet) establish. Numbers are from the real-data run in
[ADR 0030](adr/0030-methodology-critique-fixes.md).

## The model does not beat ADP at the top of the board

On the ADP universe, ADP-only (a fit of season FP on log-ADP) has a higher top-12/24/50/100 hit rate and value capture than the
baseline model (top-12 hit 0.583 vs 0.528; top-50 0.722 vs 0.653). Adding the model to ADP does not detectably improve on ADP for
ranking, only for total-FP error. Use ADP as the anchor at the top; the model adds depth, minutes, games-played and floor/ceiling.
Nine comparable seasons is little, so top-12 differences are not statistically resolvable on their own.

## Availability is conditional on appearing

`src/models/availability.py` fits games played only on player-seasons with at least one game; a season with zero games (a full-year
injury, retirement, suspension, a player who never signs) is invisible to it, so `proj_gp` is biased high for anyone with a real
chance of not appearing. Backtest by risk group (`--method-checks`): returners after missing over half of the previous season
appear in only about 41% of cases and are projected 15.3 GP too high; part-season players +9.5 GP; rookies +3.1. Among players who do
appear the bias is about +2 to +3 GP. The injury and return layers do not fix this. Treat proj_gp for those groups as an upper bound.
"Prior injury" is proxied by games missed, not by injury records.

## Positions are not point-in-time

Historical position groups come from the current `players.position` label; no dated position history is ingested. A `baseline_nopos`
ablation (all players `U`) matches the shipped model (Spearman 0.785 vs 0.784, MAE total FP 422.8 vs 422.5), so any leak through the
position priors has no measurable effect; the priors also show no measurable benefit.

## Shrinkage tuning

Rate shrinkage (decay, kappa) is tuned on the training history using the minutes model's predicted MPG, as forecasting does
(previously the realised MPG; no material change). Age curves, kappa and decay are still estimated on the same history that is then
projected from, without a nested holdout; the walk-forward outer loop keeps this from touching the scored season.

## Season-value uncertainty is not calibrated

`fppg_p10/p90` are game-level quantiles (calibrated, see docs/backtest.md). There is no calibrated interval for the season total
(FPPG x GP): season-mean FPPG uncertainty is not modelled, and the Beta shape of GP uncertainty is not checked for coverage per group.
Do not read `proj_total_fp` spreads as probabilities of a season outcome. The board rank is a point-estimate ordering of expected season totals, not a ranking of outcome distributions.

## Few seasons, dependence between them

Headline results average about ten seasons, and bootstrap intervals over players are optimistic (a player appears in several
seasons). The report's leave-one-season-out table shows how far the means move when one season is dropped (Spearman 0.777 to 0.790,
top-12 hit 0.519 to 0.546, top-50 hit 0.647 to 0.664); paired ADP comparisons use season-level t intervals for that reason.
