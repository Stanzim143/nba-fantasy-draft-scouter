# ADP-gap study: when does the market know something the model doesn't?

Script: `src/backtest/adp_gap.py` (`collect` walks the model forward once and caches the eval frames; `report` prints
every table below from that cache). Test: `tests/backtest/test_adp_gap.py`.

## Pre-registration (written 2026-10-10, before any result was looked at)

**Data.** Board default `baseline_hurdle_adp_depth_rookie_offseason_debut` (ADR 0038), walk-forward 2016-17 to
2025-26, joined to ESPN ADP (`load_store_adp`). Comparable set per season: projected, ADP-listed players.
`rank_delta = model_rank - adp_rank` (ADR 0028; positive = ADP ranks him earlier). "ADP's side wins" a disagreement
(|rank_delta| >= 10) when the realised total-FP rank is closer to the ADP rank than to the model rank. Descriptors,
all fixed in advance: age >= 31, experience <= 1 season (rookie/sophomore), team change (first team this season vs
last team last season; near pre-season, a few early-season trades leak in), last season's GP < 50% of 82,
|preseason FPPG - last season FPPG| >= 5, contract year (ADR 0013/0019). The descriptive tables (by tier top-24 /
25-50 / 51-100 / >100, direction, player type) carry no decision.

**Corrections** (both fit on earlier seasons only, each season re-fit; seasons with < 150 earlier listed rows are
skipped, as in ADR 0032):

* **A, feature blend.** The ADR 0032 regression (`1, ln ADP, ln ADP^2, model total`) plus, for each of the five
  flags (young, age >= 31, team change, prior GP < 50%, big preseason change), the flag and its interactions with
  `ln ADP` and with the model total.
* **B, rank shrink.** For listed players, re-order by `(1 - w) model_rank + w adp_rank` with `w` chosen on earlier
  seasons from {0, 0.1, ..., 1} per cell = tier x (any flag set or not), minimising mean |actual rank - shrunk rank|;
  the blend's own totals are then re-assigned in that order (magnitudes unchanged, only order moves).

**Decision rule.** Comparison: each correction against the current ADR 0032 blend (`blend_rank`), whole projected
universe, paired bootstrap of `src/backtest/ablation.py` (stratified by season; hit rates resample seasons).
A correction **passes** only if all hold: top-50 hit verdict `improves` (95% CI above 0) with at least 5 of the
scored seasons won; top-12 point lift >= 0; Spearman (total FP) verdict not `hurts`. Two corrections are tested, so a
pass by one at the edge is read with that in mind. A passing correction becomes a selectable blend variant; it
replaces the default blend only if top-12 is also `improves`. The board's default *model* is not touched by this
study either way.

## Results

(pending)
