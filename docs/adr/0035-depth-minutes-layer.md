# ADR 0035: Departed-minutes and playoff-role as a minutes layer (`baseline_depth`)

**Status:** accepted, 2026-10-08
**Code:** `src/features/depth.py`, `src/models/depth_baseline.py` (`DepthHooks`, `BaselineDepthProjector`,
`AdpDepthDebutantProjector`), `src/models/registry.py`; ingest `src/ingest/nba_playoffs.py` (2015-16 to 2025-26).
**Follows:** ADR 0034 (the same two signals as display flags); ADR 0010/0011 (earlier team-context layers, no lift).

## Context

ADR 0034 showed departed minutes and a playoff closing run only as flags, because nothing had tested them as inputs. The ask was to
apply them to every player in the model, and to move the projection only if the walk-forward backtest supports it.

## Decisions

**D1. Two features per player, both known before the season.** `open_net`: minutes per team game that left the player's team
(dated `out` transactions before 1 October) net of those that arrived, weighted by each mover's previous-season minutes, in tens of
minutes; only for players who stayed on their team and averaged 8-30 mpg the season before (movers are ADR 0011's `changed_team`
problem, which hurt). `po_uplift`: closing-run (last 5 games) playoff mpg of the previous season above the regular-season mpg,
floored at 0, needing 5 playoff games and a regular-season average under 28 mpg. One WLS of `actual_mpg - context_free_mpg` on both
(weighted by games played), clipped at +-4 mpg, intercept not applied: the `TransactionFeatures` contract.

**D2. Registered as `baseline_depth` and, for the board, `baseline_hurdle_adp_depth_offseason_debut`.** It uses the otherwise unused
transactions slot of `BaselineProjector`; `baseline_transactions` and `baseline_depth` never coexist. Missing transactions or
playoff logs zero the feature (bit-identical to the base model, tested).

**D3. Result, real walk-forward ablation 2016-17 to 2025-26** (paired bootstrap over players, stratified by season; positive = better):

| comparison | metric | lift | 95% CI | seasons won | verdict |
|---|---|---:|---|---:|---|
| `baseline_depth` over `baseline` | Spearman, total FP | +0.0016 | [+0.0010, +0.0022] | 9/10 | improves |
| | Top-50 hit | +0.0000 | [-0.0050, +0.0060] | 1/10 | no significant change |
| | MAE, total FP | +0.88 | [+0.29, +1.48] | 7/10 | improves |
| board stack with depth over without | Spearman, total FP | +0.0004 | [+0.0002, +0.0006] | 7/10 | improves |
| | Top-50 hit | -0.0040 | [-0.0100, +0.0040] | 1/10 | no significant change |
| | MAE, total FP | +0.82 | [+0.31, +1.33] | 8/10 | improves |

Reproduce: `python -m src.ingest.nba_playoffs --seasons 2015-16:2025-26`, then `python -m src.backtest --ablate baseline,baseline_depth
--seasons 2016-17:2025-26` and `--ablate baseline_hurdle_adp_offseason_debut,baseline_hurdle_adp_depth_offseason_debut`. The effect is small: about 0.8 FP of total-FP error out of about 316, and no top-50 gain. `--leak-check` passed for both `baseline_depth` and the board stack on 2023-24 to 2025-26
(the ten-season run could not complete in one process on the 8 GB dev machine: out of memory). That check perturbs the game-log
tables only, so the two file inputs have their own guards: feature-level tests (`tests/models/test_depth.py`) show every player's
features are identical with and without rows dated after the 1 October cutoff, next-season moves and target-season playoffs, and
that the check fails if the cutoff is removed. Playoffs are keyed by season (`s - 1` for target `s`), so a target-season playoff
row can never reach a target-season feature even without the extra `season < target` filter.

**D4. The board default becomes `baseline_hurdle_adp_depth_offseason_debut`** (draft board, `daily_refresh` board CSVs, `adp_blend`
default). The ADP blend (ADR 0032) must be refit for it: `python -m src.value.adp_blend fit`.

**D5. The display flags now apply to every player.** The 8-30 projected-mpg band on the depth flag and the 28 mpg regular-season cap
on the playoff flag are removed; both flags show for any player whose team opened 20+ net minutes, or whose closing run is 4+ mpg
above the regular season. Flags stay descriptive and independent of the projection.

## Consequences

Known limits carry over: unsigned free agents count as departed, rookies and two-way players are unmeasured, the signal is not
position-aware, and returning injured players take minutes back. A player's flag and the projection shift are separate: the shift
is small (clipped at 4 mpg, typically well under 1).
