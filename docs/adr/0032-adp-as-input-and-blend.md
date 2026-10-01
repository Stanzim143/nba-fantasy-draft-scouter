# ADR 0032: ADP as an input to availability and minutes, and as a blended board rank

**Status:** accepted, 2026-10-01. Extends [0030](0030-methodology-critique-fixes.md) (ADP as the benchmark to beat) and
[0031](0031-hurdle-availability.md) (the appearance stage that takes ADP). Until now ADP was only an external benchmark the model competed with
(backtest `--adp-file`); this ADR lets the model and the board *use* it.
**Code:** `src/features/adp.py` (`AdpFeatures`, `load_store_adp`, `assert_adp_no_future`), `src/models/adp_baseline.py` (`ContextHooks`,
`BaselineAdpProjector`, `AdpDebutantProjector`), `src/value/adp_blend.py` (`AdpBlend`, `fit_adp_blend`, `blend_totals`), `src/value/board.py`
(`--adp-blend`, `load_blend`), `src/backtest/method_checks.py` (`blend_universe_table`).
Tests: `tests/features/test_adp_features.py`, `tests/models/test_model_adp.py`, `tests/value/test_adp_blend.py`,
`tests/backtest/test_method_checks_hurdle.py`.

## Context

ADP (the market's consensus draft position) beat the model's top-50 hit rate in every comparison (ADR 0030) because it prices what the game logs
cannot see: injuries in the news, offseason role changes, coaches' minutes plans. The model beat ADP over the whole projected universe
(ADP lists only ~300 players a season). Two separate uses follow.

## Decision

1. **ADP as a feature** (`baseline_hurdle_adp`). Per player, `[covered, listed, z]` where `z` is the standardised log ADP of the season being
   projected (0 when not listed). It enters the availability model and the appearance model through their `extra` hooks, and a ridge adjustment
   of the minutes-per-game projection. ADP of the target season is pre-season information, so it is allowed; `assert_adp_no_future` rejects
   any ADP row of a later season, and the projector-ignores-the-future test covers it. Seasons without ADP get all-zero features, so the base
   model still trains on them.
2. **Board blend** (`src/value/adp_blend.py`). For ADP-listed players:
   `total = b0 + b1 ln(adp) + b2 ln(adp)^2 + b3 model_total + b4 [model total missing]`, fit by least squares on *earlier seasons only*
   (players listed but not playing count as 0). Unlisted players keep the model total. The board adds `blend_total_fp`, `blend_vorp`,
   `blend_rank` beside the existing columns (additive; `rank` unchanged); the games-played band, floor and ceiling stay the model's. The fit is
   stored as `adp_blend.json` in the processed data dir (`python -m src.value.adp_blend fit`), loaded with `--adp-blend auto|off|<path>`; a
   missing or corrupt auto file silently means no blend. It must be refit right before the draft.

The board default model becomes `baseline_hurdle_adp_offseason_debut`; `daily_refresh` builds it.

## Results (real data, walk-forward 2016-17 to 2025-26)

Stack versus the stack it replaces (10 seasons, whole projected universe):

| | baseline_offseason_debut | baseline_hurdle_adp_offseason_debut | ADP alone |
|---|---|---|---|
| Spearman, total FP | 0.771 | **0.819** | 0.723 (listed only) |
| MAE total FP | 372 | **316** | n/a |
| MAE FPPG | 4.514 | 4.488 | n/a |
| top-50 hit | 0.658 | 0.674 | 0.728 |
| top-100 hit | 0.683 | 0.687 | 0.736 |

ADP in the appearance stage alone (`baseline_hurdle` -> `baseline_hurdle_adp`): Spearman 0.810 -> 0.823, MAE total FP 367.5 -> 353.2, and the
VORP-weighted MAE that the hurdle alone had worsened returns to 658 (baseline 662).

Blend versus the model it blends (each season fit on earlier seasons only, so 8 scored seasons; the first two lack the 150 listed rows needed
to fit and are skipped, not guessed):

| metric | model | blend | blend - model | 95% CI | seasons won |
|---|---|---|---|---|---|
| Spearman, total FP | 0.832 | 0.827 | -0.004 | [-0.014, +0.005] | 2/8 |
| top-12 hit | 0.510 | 0.552 | +0.042 | [-0.033, +0.116] | 5/8 |
| top-24 hit | 0.568 | 0.641 | +0.073 | [+0.012, +0.134] | 6/8 |
| top-50 hit | 0.677 | 0.725 | +0.047 | [+0.013, +0.082] | 6/8 |
| top-100 hit | 0.688 | 0.716 | +0.029 | [+0.005, +0.053] | 6/8 |
| MAE total FP | 296 | 303 | +6.9 | [-6.4, +20.3] | 5/8 |

Reading it: the blend closes the top-end gap to ADP (top-50 0.725 against ADP's 0.728) at no significant cost elsewhere; it is slightly worse
on whole-universe rank correlation and MAE (not significant), which is why it is a *separate* column and the model's `rank` stays primary for
depth picks. Use `blend_rank` for the first rounds, `rank` for the rest.

## Consequences

* Model and blend can disagree: that disagreement is the information (the model sees a durable starter ADP undervalues, or ADP knows of news the
  model cannot). The board shows both.
* ADP comes from ESPN's pre-season listing, the source already ingested (ADR 0005, `src/ingest/espn_adp.py`); nothing new is fetched.
* The blend has five coefficients fit on ~2,000 listed player-seasons; an old file is stale by construction.
* The backtest still reports ADP as an independent arm, so the benchmark comparison stays honest.

Reproduce: `python -m src.backtest --model baseline_hurdle_adp_offseason_debut --benchmark baseline_offseason_debut --adp-file <processed>/adp.parquet --method-checks`.

## Addendum 2026-10-02: blend/model guard and loud fallbacks

An independent review found that `--adp-blend auto` applied `adp_blend.json` to any board model although its coefficients are only valid
for the model they were fitted on (`AdpBlend.model`). `load_blend(..., model)` now skips an `auto` blend fitted for a different model
(warning) and only warns for an explicitly named path; the board CLI and the app loader pass their model. `adp_blend fit` now defaults to
the board default `baseline_hurdle_adp_offseason_debut`; refit with `python -m src.value.adp_blend fit` when the model or the completed history changes. (A refit on 2026-10-02 reproduced the saved coefficients exactly, so the original "refit right before the draft" instruction above is not needed: the fit uses completed seasons only.) The ADP-aware
projectors also log a warning when the ADP table, the injury-label table, or ADP for the target season is missing, instead of silently
running as the base model.
