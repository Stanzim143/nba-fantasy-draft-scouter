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

## Results (2026-10-10)

Run: `python -m src.backtest.adp_gap collect` (about 25 minutes, under the compute lock), then
`python -m src.backtest.adp_gap report <cache parquet>` (seconds). The eval frames are cached outside the repo at
`<data dir>/cache/adp-gap/baseline_hurdle_adp_depth_rookie_offseason_debut__<sha>.parquet`. 2,586 listed, projected
player-seasons; ADP lists 147-149 players in 2016-17 and 2017-18 and 230-346 after.

**Where ADP's side wins** (|rank_delta| >= 10; "seasons" = seasons in which ADP wins the majority):

| tier (better of the two ranks) | disagreements | ADP right | seasons | ADP higher than model | ADP lower than model |
|---|---:|---:|---:|---:|---:|
| top-24 | 117 | 0.70 | 8/10 | 0.66 | 0.73 |
| 25-50 | 186 | 0.63 | 9/10 | 0.58 | 0.68 |
| 51-100 | 419 | 0.60 | 10/10 | 0.60 | 0.61 |
| beyond 100 | 1,167 | 0.46 | 2/10 | 0.55 | **0.36** |

Inside the top 100 ADP wins most disagreements in both directions: the market is right more often whether it is
higher or lower than the model, so there is no one-sided bias to correct. Beyond 100 the model wins when it likes a
player ADP has lower (64%): the deep "sleepers" are the model's edge.

By player type (one label per player, priority rookie > sophomore > 31+ > team change > prior GP < 50% > preseason > contract year; descriptors fixed before the run): the only clean, consistent cell is
**rookies ADP ranks above the model** (109 disagreements, ADP right 0.76, 9/9 seasons); rookies ADP ranks *below*
the model go the model's way (0.28). Team changers lean ADP slightly in both directions (0.55, 7/9 seasons). Veterans
31+, sophomores, contract-year players and "other" are coin flips (0.48-0.51). "Preseason up" is 0.83 when ADP is
higher and 0.15 when lower, but on 36 and 26 rows. ("Preseason down" catches 502 rows because starters' preseason
FPPG is routinely lower than their season FPPG; it is not a role signal.) These rates compare against the raw model
rank, not the blend.

**Pre-registered corrections against the ADR 0032 blend** (8 scored seasons 2018-19..2025-26; the blend itself beats
the model by +0.060 top-50 [+0.030, +0.088], 7/8, as in ADR 0032):

| correction | top-12 | top-24 | top-50 | top-100 | Spearman | NDCG@50 |
|---|---|---|---|---|---|---|
| A, feature blend | -0.021 [-0.062, +0.021] 2/8 | -0.068 hurts 0/8 | -0.015 [-0.033, 0.000] 1/8 | -0.007 n.s. | +0.007 improves 8/8 | -0.009 hurts |
| B, rank shrink | +0.010 [-0.042, +0.062] 2/8 | +0.026 [+0.005, +0.047] 4/8 | -0.003 [-0.015, +0.007] 2/8 | +0.013 [+0.001, +0.025] 5/8 | +0.004 improves 5/8 | +0.004 n.s. |

Season means: blend top-12/24/50/100 0.552 / 0.641 / 0.730 / 0.716; B 0.562 / 0.667 / 0.728 / 0.729.

**Verdict: null.** Neither correction meets the rule (top-50 `improves` with >= 5 wins): A is worse at the top
(it over-fits 19 coefficients with interactions to ~2,000 rows), B ties the blend on top-50. B's top-24 and top-100
gains are significant but won in only 4/8 and 5/8 seasons, were not the primary metric, and come from one of two
tested corrections; reading them as real would be tuning on the test. Nothing was implemented; the blend, the board
default model and `blend_rank` are unchanged. The residual structure is real but small: once the blend has ADP in it,
the market's extra knowledge is not predictable from age, experience, team change, prior availability, preseason or
contract year beyond what the blend already uses. The one cell worth a human eye on draft day is a rookie the market
takes well above our rank: the market was right three times in four.
