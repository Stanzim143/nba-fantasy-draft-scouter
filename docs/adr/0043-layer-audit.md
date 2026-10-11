# ADR 0043: Drop-one audit of the board stack's layers

**Status:** accepted, 2026-10-12 (decision rule pre-registered in commit cbf3017 before any run)
**Code:** drop-one variants in `src/models/registry.py`, `DepthDebutantProjector` in `src/models/depth_baseline.py`;
`tests/models/test_layer_audit_variants.py`.
**Follows:** ADR 0012 (offseason), 0016 (debutants), 0031 (hurdle), 0032 (ADP as input), 0035 (depth), 0038 (rookie appearance).

## Context

The board default `baseline_hurdle_adp_depth_rookie_offseason_debut` was built one layer at a time, each ablated against the
stack as it was then. A layer that helped then can be redundant now; ADP as an input, for example, may already carry what the
preseason or depth layers add. This ADR re-tests every layer against the current full stack.

The layers, from the code: **hurdle** (`BaselineConfig(appearance_hurdle, season_intervals)`; the season intervals only widen
bands and do not move the point metrics, so the two are dropped together), **ADP as input** (`AdpHooks`), **depth**
(`DepthHooks`), **rookie appearance** (`rookie_appearance=True`), **offseason** (the `BaselineOffseasonProjector` wrapper) and
**debutants** (`add_debutants`: extra rows for stash and undrafted players).

## Pre-registered decision rule (written before any run)

For each layer, lift = full stack over the stack minus that layer, positive = the layer helps; paired bootstrap over players
stratified by season, 500 replicates, seed 0, 95% CI; seasons 2016-17 to 2025-26; metrics `mae_total_fp`, `mae_gp`,
`spearman_total_fp`, `top50_hit`.

- **Earns its place:** at least one metric's CI lies entirely above zero.
- **Redundant:** no metric's CI lies entirely above zero (every CI includes zero or favours removal).
- **Harmful:** at least one metric's CI lies entirely below zero (removing it improves that metric significantly).
- **The board default changes only** if a layer is harmful on at least one metric, earns on none, and the improvement from
  removing it wins at least 7 of 10 seasons. A redundant but harmless layer is "keep, flagged": no change lands a week before
  the draft. A mixed layer (helps one metric, hurts another) is kept and the trade-off recorded.
- **Debutants are not judged on the paired lift.** The paired comparison only scores players both models project, and the
  debutant layer's purpose is the players only it projects (ADR 0016); its paired lift reflects only its indirect effect
  through the offseason layer's training residuals. It is judged on coverage (unprojected players in the realised top 50) and
  is not a removal candidate on paired metrics alone.

## Results

Real walk-forward, 2016-17 to 2025-26, one frozen copy of the processed data (2026-10-10) for every run. Lift = full stack over
the stack minus the layer (positive = the layer helps). The full stack reproduces ADR 0038 (MAE total FP 313.6, games 14.19,
Spearman 0.820, top-50 0.670).

| layer removed | MAE total FP | MAE games | Spearman total FP | top-50 hit | verdict |
|---|---|---|---|---|---|
| hurdle | +48.4 [+45.7, +50.9] 9/10 | +3.24 [+3.02, +3.45] 9/10 | +0.0403 [+0.0370, +0.0444] 10/10 | +0.002 [0, +0.006] 1/10 | earns, keep |
| ADP as input | +8.96 [+6.72, +11.29] 9/10 | +0.37 [+0.29, +0.47] 8/10 | +0.0101 [+0.0081, +0.0119] 9/10 | +0.014 [+0.004, +0.024] 6/10 | earns, keep |
| depth | +0.81 [+0.30, +1.32] 8/10 | +0.009 [+0.005, +0.014] 9/10 | +0.0004 [+0.0001, +0.0006] 7/10 | -0.004 [-0.010, +0.004] 1/10 | earns (small), keep |
| rookie appearance | +1.86 [+1.39, +2.46] 8/10 | +0.16 [+0.12, +0.21] 8/10 | +0.0010 [+0.0002, +0.0019] 8/10 | 0 (no change) | earns, keep |
| offseason (SL/preseason) | +3.33 [+2.35, +4.40] 7/10 | 0 (no change) | +0.0004 [-0.0005, +0.0012] 5/10 | +0.006 [-0.002, +0.012] 4/10 | earns on MAE only, keep |
| debutants (paired, shared players only) | -0.59 [-0.75, -0.45] 0/10 | 0 | +0.0005 [+0.0003, +0.0007] 8/10 | 0 | judged on coverage, keep |

Cells: lift [95% CI] seasons won. Paired bootstrap over players stratified by season, 500 replicates, seed 0.

**Verdicts.** Hurdle, ADP as input, depth, rookie appearance and offseason each have at least one metric whose CI lies above
zero, so none is redundant and none is harmful; ADP as input does **not** absorb the later layers. Two are flagged as marginal:
depth (+0.8 FP of MAE, +0.0004 Spearman, as in ADR 0035) and offseason, which now earns only on total-FP error (+3.3) and no
longer on rank order once ADP is an input. Debutants: on shared players the layer is mixed (MAE -0.59, Spearman +0.0005, the
indirect effect through the offseason fit that ADR 0016 recorded); without it, 0.2 realised top-50 players a season go
unprojected (2 over ten seasons), against 0 with it. Its purpose is that coverage, so it stays.

**The board default does not change** (`baseline_hurdle_adp_depth_rookie_offseason_debut`); `adp_blend.json` needs no refit.

**Method.** The CLI's `--cache-dir` refuses every registry model (none sets `projector.fingerprint`), and the depth and ADP
inputs are read from the store outside the history hash, so a cache keyed on the history alone could be stale. The audit
therefore ran each model once through `walk_forward` against a frozen copy of the processed data, with a fingerprint set to the
`src` tree hash, and built each pair with `ablation_from_results`: the same numbers as
`python -m src.backtest --ablate <minus>,<full> --seasons 2016-17:2025-26 --sig-metrics mae_total_fp,mae_gp,spearman_total_fp,top50_hit`,
which reproduces any row (it recomputes the full stack each time). Drop-one variant names: `baseline_adp_depth_rookie_offseason_debut`
(hurdle), `baseline_hurdle_depth_rookie_offseason_debut` (ADP), `baseline_hurdle_adp_rookie_offseason_debut` (depth),
`baseline_hurdle_adp_depth_offseason_debut` (rookie), `baseline_hurdle_adp_depth_rookie_debut` (offseason),
`baseline_hurdle_adp_depth_rookie_offseason` (debutants).

## Consequences

- No change before the 2026-10-17 draft. The next audit is due whenever a layer is added or the data gains a season.
- Depth and offseason are the first candidates to drop if maintenance cost matters later; both still pass the rule.
