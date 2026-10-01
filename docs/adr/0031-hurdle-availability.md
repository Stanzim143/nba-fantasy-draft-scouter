# ADR 0031: Hurdle availability: model the chance a player appears at all

**Status:** accepted, 2026-10-01. Extends [0003](0003-baseline-model.md) (baseline) and answers the disclosed limit in
[0030](0030-methodology-critique-fixes.md) and [../limitations.md](../limitations.md) ("Availability is conditional on appearing").
**Code:** `src/models/appearance.py` (`AppearanceModel`, `gp_mixture`), `src/models/baseline.py` (`_fit_appearance`, `p_appear` in `_core`,
mixture in `_assemble`), `src/models/config.py` (`appearance_hurdle`, `appearance_C`, `appearance_min_rows`), `src/models/registry.py`
(`baseline_hurdle`), `src/backtest/method_checks.py` (`appearance_calibration`), `src/backtest/actuals.py` (optional eval columns).
Tests: `tests/models/test_model_hurdle.py`, `tests/backtest/test_method_checks_hurdle.py`.

## Context

`AvailabilityModel` is a fractional logistic fit on player-seasons with at least one game, so `mu` is the expected share of the schedule
*given that the player appears*. Anyone who does not appear (a returner from a long injury, a retirement, a suspension, a player who never
signs) is invisible to it and `proj_gp = mu * L` is biased high for exactly the players whose availability matters most. The backtest's
risk-group table (ADR 0030) measured it: returners after missing over half of the previous season appear in only 41% of cases and were projected
15.3 games too high, part-season players 9.5 too high. About 70% of the model's miss error is availability.

## Decision

Add the missing first stage and keep the second:

```
P(appear)  = logistic( f_bar, f_last, f_min, absent, age, age^2, mpg, seasons played of the last 4, seasons since last game )
GP         = 0 with probability 1 - P(appear), else L * Beta(mu * nu, (1 - mu) * nu)        (the existing conditional model)
proj_gp    = P(appear) * mu * L ;   proj_total_fp = proj_fppg * proj_gp  (the contract's product, unchanged)
```

* **Training rows** are every historical (player, season) where the player was *eligible* by the rule the projector already uses for the
  target season (played in one of the `active_seasons` = 2 seasons before), labelled by whether he has any game that season. Built
  season by season from the panel, so nothing after the target is read.
* `proj_gp_sd`, `proj_gp_p10` and `proj_gp_p90` are the mean, sd and quantiles of the mixture (`gp_mixture`); it reduces exactly to
  `AvailabilityModel.distribution` when `P(appear) = 1`. A player with more than a 10% chance of not playing has a zero floor.
* Rookies and debutants keep their draft-slot prior (`proj_p_appear = 1`): their non-appearance is not modelled here (disclosed below).
* `proj_p_appear` is emitted as an extra projection column (and carried onto the board). It is **off by default**
  (`BaselineConfig.appearance_hurdle=False`), so `baseline` stays bit-identical; `baseline_hurdle` turns it on (together with the
  season-total band of [0033](0033-season-intervals-and-injury-labels.md), which changes no existing column).
* The appearance model takes the same `extra` columns as the availability model, so ADP ([0032](0032-adp-as-input-and-blend.md)) and the
  labelled-injury features (0033) reach it too.

## Results (real data, 2016-17 to 2025-26, walk-forward, 2026-10-01)

| | baseline | baseline_hurdle |
|---|---|---|
| Spearman, total FP | 0.784 | **0.810** (+0.026, CI [+0.023, +0.029], 10/10 seasons) |
| MAE total FP | 422.5 | **367.5** (-55.0, CI [-57.8, -51.8], 9/10) |
| MAE games played | 19.2 | **15.7** (-3.6, 9/10) |
| RMSE total FP | 536.4 | 516.3 |
| bias total FP (pred - actual) | +133.6 | **+13.9** |
| top-12 / top-50 hit | 0.533 / 0.654 | 0.550 / 0.654 (no significant change) |
| VORP-weighted MAE | 662.4 | 708.7 (worse) |

Games-played bias by risk group (projected minus actual, zero-game players included; `bias_gp`), baseline -> hurdle: returners
`+15.3 -> +1.5`, part-season last year `+9.5 -> +2.3`, age 33+ and prime players `+2 to +3 -> +1.9 to +2.3`, rookies `+3.1` unchanged.
`bias_when_played` (the conditional statistic) for returners is `-9.8` for the hurdle model, as it should be: the old model projected
about 29 games for each of them (13.6 actually played on average, 33 among those who appeared at all); the new one projects 15.1 on average
over the group because it says "41% of these players appear at all, and those who do play about a third of the season".

Calibration of `proj_p_appear`, out of sample, 7,000 projected players: bins 0-0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8 / 0.8-0.9 / 0.9-0.95 / 0.95-1
predict 0.068 / 0.310 / 0.499 / 0.702 / 0.854 / 0.916 / 0.985 and observe 0.088 / 0.291 / 0.492 / 0.711 / 0.878 / 0.883 / 0.959; Brier 0.113
against 0.209 for always predicting the base rate. It is slightly over-confident in the top two bins (the eligible pool includes players who
retire or go overseas, which no games-missed history can see).

**VORP-weighted MAE gets worse (662 -> 709) and that is a property of the metric, not a regression.** MAE is minimised by the median, and for
a star with a 60% chance of playing the median outcome is "plays", while the mean is lower; the hurdle model predicts the mean, which is what a
points league pays (season points are a sum) and what cuts RMSE and bias. The metric weights exactly those high-value, high-variance players.
With ADP in the appearance stage (0032) the metric returns to 658, better than the baseline's 662.

## Consequences

* A returner like Tatum is now projected with his real chance of not playing instead of as if certain; `proj_total_fp` and the board rank
  move accordingly for every player with a real chance of missing the year. `proj_gp` for such players is an expected value over a mixture
  with a point mass at zero, not a typical season: read `proj_gp_p10` (0) and `proj_p_appear` beside it.
* The risk overlay (ADR 0016) applies its games-at-risk haircut to `proj_gp`; with the hurdle that can partly double count appearance risk
  for players ESPN lists as OUT. The overlay is advisory and unchanged.
* **Not modelled:** rookies' and debutants' chance of not appearing (about 10% of drafted rookies never play; the season-total band is too
  narrow for them, see 0033), and retirements or overseas moves before opening night beyond what the games-missed history predicts.
* `baseline` remains registered and unchanged so earlier results stay reproducible.

Reproduce: `python -m src.backtest --ablate baseline,baseline_hurdle --sig-metrics spearman_total_fp,top12_hit,top50_hit,mae_total_fp,mae_gp --method-checks`.
