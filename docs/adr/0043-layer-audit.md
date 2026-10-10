# ADR 0043: Drop-one audit of the board stack's layers

**Status:** proposed, 2026-10-10 (decision rule pre-registered before any run; results pending)
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

Pending.
