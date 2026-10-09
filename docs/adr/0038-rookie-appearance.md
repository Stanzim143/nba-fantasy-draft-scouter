# ADR 0038: Rookie appearance stage (P(a drafted rookie plays at all))

**Status:** accepted, 2026-10-09
**Code:** `src/models/rookie_appearance.py`; the `rookie_appearance` option of `DebutantBaselineProjector` (`src/models/debutants.py`);
the `FittedBaseline.rookie_p_appear` hook (`src/models/baseline.py`); `src/models/registry.py`.
**Follows:** ADR 0031 (hurdle availability, veterans only), ADR 0033 (season-total band too narrow for rookies), ADR 0012 (preseason
evidence is point-in-time for the draft), ADR 0035 (current board stack).

## Context

The draft-slot prior (`src.models.rookies`) is fit on rookie seasons with at least one game, and the ADR 0031 appearance stage covers
veterans only, so every drafted rookie was projected as certain to play. PLANNING and `docs/limitations.md` listed this as open, with
the rookie season-total band holding only 68% of outcomes against a nominal 80% (ADR 0033).

Most of the missing signal is the preseason. Real data, 2015 to 2025 draft classes (650 rookies with a `players` row): 98% of rookies
with a preseason game play in their first season, against 39% of the 92 without one (mostly stashed second-rounders and injured
picks). By pick: 98% of picks 1-10 appear, 85% of picks 46-60.

## Decisions

**D1. A ridge logistic for P(appear) on log(pick), played-in-preseason, preseason mpg, played-in-Summer-League and Summer League
mpg.** It is fit on every earlier season's draft class in the `History` (rows: `draft_year` equal to the season's start year;
label: any game that season), with the offseason logs sliced to the target (`slice_offseason`). The rookie's `p_appear` multiplies
`proj_gp` exactly as a veteran's does in the hurdle models, so the season-total band gets its point mass at zero. FPPG, veterans,
stash and undrafted debutant rows are unchanged. With fewer than 100 training rookies or 10 non-appearances the stage is off
(every rookie keeps `p_appear = 1`); in the backtest that is 2016-17 only.

**D2. The partial-preseason cut is live only.** A draft held during the preseason has seen only part of it. When a roster snapshot of
the target season exists (the live path, as for the debutant camp), the target's preseason game dates so far are divided by the
median of the last three training preseasons, and every training preseason is cut to that share of its dates, so "no preseason
game yet" means the same in training as on draft day. In a backtest nothing is cut: preseason length varies too much (22 game
dates in 2015-16, 9 in 2020-21) for a date count to tell a finished short preseason from an unfinished one. A first version
compared every target with the training median, which cut five complete backtest preseasons; the independent review caught it
before any result was used. With no target preseason game at all, the preseason columns are dropped.

**D3. Result, real walk-forward ablation 2016-17 to 2025-26** (board stack with the stage over without it; paired bootstrap over
players stratified by season, 500 replicates; positive = better):

| metric | lift | 95% CI | seasons won | verdict |
|---|---:|---|---:|---|
| MAE, total FP | +1.86 | [+1.39, +2.46] | 8/10 | improves |
| MAE, games played | +0.16 | [+0.12, +0.21] | 8/10 | improves |
| Spearman, total FP | +0.0010 | [+0.0002, +0.0019] | 8/10 | improves |
| Top-50 hit | +0.0000 | [+0.0000, +0.0000] | 0/10 | no change |

Mean MAE of total FP 315.5 to 313.6, games played 14.35 to 14.19, total-FP bias +39.3 to +36.4. With the stage, the rookie group
(no prior NBA games, n = 1,693 including debutants) has a games bias of -0.2 and its season-total band holds 80.5% of outcomes (nominal 80%).
Reproduce: `python -m src.backtest --ablate baseline_hurdle_adp_depth_offseason_debut,baseline_hurdle_adp_depth_rookie_offseason_debut
--seasons 2016-17:2025-26 --method-checks --sig-metrics mae_total_fp,mae_gp,spearman_total_fp,top50_hit`. `--leak-check` passed for the
board stack with the stage on 2025-26 (`--model ... --seasons 2025-26 --leak-check`; a ten-season or three-season run with it does
not fit in memory, as in ADR 0035). Feature-level guards live in
`tests/models/test_rookie_appearance.py`: training rows never read target-season games, and the projector passes
`assert_projector_ignores_future`.

**D4. The board default becomes `baseline_hurdle_adp_depth_rookie_offseason_debut`** (draft board, `daily_refresh` board CSVs,
`adp_blend` default). The ADP blend must be refit for it: `python -m src.value.adp_blend fit` (the board does not apply a blend
fitted for another model).

## Consequences

- The top of the board is untouched: early picks almost always play preseason, and nothing changes for veterans. The effect is on
  late picks without a preseason game, which drop to about a third to a half of their earlier projected games.
- Draftees who never play an NBA game have no `players` row, so they are missing from both the training rows and the backtest's
  scoring universe; the fitted rate is if anything too high (conservative). Rookies who skip the preseason through injury and
  return mid-season are under-projected, which is why the stage scored worse than "always appears" on Brier score in 2023-24 to 2025-26 (an
  independent review's probe) while improving total-FP error overall.
- Live, before the preseason ends, a rostered rookie who has not played yet is scored against training preseasons cut to the same
  share; the board's numbers move as the preseason fills in (the daily refresh rebuilds them).
