# 0030: Methodology critique: honest shrinkage tuning, ADP + model arm, disclosed limits

Status: Accepted (2026-09-30). Extends [0004](0004-backtest.md) (backtest); does not rewrite it.

## Context

An external methodology critique raised five points. Two are correctness/comparison problems that are cheap to fix, the rest are
limits to state plainly.

## Decision

1. **Shrinkage tuning uses predicted minutes.** `fit_spec` tuned each rate's decay and kappa with the prior evaluated at the
   target season's *actual* MPG, while forecasting feeds the prior the *predicted* MPG. Tuning now uses the minutes model's
   prediction from lags alone for every training row (`BaselineConfig.tune_with_predicted_mpg`, default True; False reproduces the
   old behaviour). The default is the honest setting.
2. **Availability calibration by risk group** is added to the backtest report (`--method-checks`): rookies, returners after missed
   time, part-season last year, age 33+, prime. The availability model stays conditional on appearing; that limitation is
   disclosed in [../limitations.md](../limitations.md), not fixed.
3. **Position priors.** No point-in-time position history is ingested (`roster_snapshots` covers only the current preseason), so
   `players.position` is a current label. Instead of a fix, a leak-free arm exists: `BaselineConfig.use_position_priors=False`,
   registered as `baseline_nopos`, sets every group to `U`. Disclosed.
4. **ADP + model arm and per-tier comparison** (`src/backtest/method_checks.py`). ADP-only is a regression of season total FP on
   log-ADP; ADP + model adds the model's `proj_total_fp`. Coefficients are fit on earlier seasons only, the first season is dropped
   from all arms, and all arms share the ADP universe. Top-N (12/24/50/100) hit and value capture are compared with season-paired
   t intervals (seasons are the independent units).
5. **Leave-one-season-out** sensitivity of the headline means is added (cheap: the walk-forward already yields per-season
   metrics). A calibration check of season-value (FPPG x GP) uncertainty is **not** done: the model has no season-mean FPPG
   uncertainty (its FPPG band is game-level), so it needs new modelling; recorded in PLANNING.md and limitations.

## Results (real data, 2016-17 to 2025-26, run 2026-09-30)

* Tuning fix: no material change (the MPG prior input is a weak lever). Mean over 10 seasons, actual-MPG tuning to predicted-MPG
  tuning: Spearman total FP 0.7836 to 0.7837; top-12 hit 0.5417 to 0.5333; top-50 hit 0.6560 to 0.6540; MAE FPPG 4.672 to 4.670;
  MAE total FP 422.59 to 422.53. Differences are far inside season-to-season noise; the honest setting is the default.
* Position priors off (`baseline_nopos`): Spearman 0.785 vs 0.784, top-50 hit 0.650 vs 0.654, MAE total FP 422.8 vs 422.5. No
  detectable benefit from the position priors, hence no detectable effect of their non-point-in-time source.
* **Model vs ADP at the top of the board: the model does not beat ADP.** On the ADP universe ADP leads the model at every depth
  (top-12 hit 0.583 vs 0.528, top-24 0.644 vs 0.542, top-50 0.722 vs 0.653, top-100 0.737 vs 0.673). The gap is statistically
  clear from top-24 down; at top-12 the hit-rate interval includes zero (9 seasons) but the model won only 1 of 9 seasons, and value
  capture at top-12 favours ADP (0.899 vs 0.843, 0 of 9 seasons won by the model). **ADP + model does not detectably improve on ADP**
  on any rank metric (top-50 hit 0.720 vs 0.722); it does cut total-FP error (MAE 516 vs 550 for ADP-only; 507 for the model
  alone). So the model's value is in magnitude and depth (ADP ranks 37+ and unlisted players), not at the top of the board.
* Risk groups: proj_gp is biased high everywhere once zero-game players count, worst for returners (+15.3 GP; only 41% of them
  appear at all) and part-season players (+9.5); when the player does appear the bias is +2 to +3 GP for veterans.

## Consequences

* Claims that the model "beats naive on rank" stay true; for draft-board ordering defer to ADP at the top and use the model for depth,
  minutes and games-played judgement.
* `BaselineConfig` gains two fields; defaults keep every existing projector identical except the (tiny) tuning change.
* Reproduce: `python -m src.backtest --model baseline --benchmark naive_last_season,baseline_nopos --adp-file <processed>/adp.parquet --method-checks`.
