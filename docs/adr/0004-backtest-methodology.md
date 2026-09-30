# ADR 0004: Backtest methodology

**Status:** accepted, 2026-09-22
**Code:** `src/backtest/`  **Usage and metric definitions:** `docs/backtest.md`

## Context

The backtest is the credibility of the whole project: a projection engine is only as good as the
evidence that it would have worked. Fantasy projection backtests fail in predictable ways: future
information reaches the model, injured busts are dropped from the scoring set, rookies the model never
projected are ignored, one flattering metric is reported, and lift is claimed from noise. The design
below is a defence against each.

## Decisions

1. **Walk-forward, one season at a time.** Season S is projected from seasons < S only, scored
   against S, and the harness repeats for every requested season. Each season is one draft, so
   per-season metrics are averaged with equal weight.

2. **No random or k-fold cross-validation.** Rows are not exchangeable: a player-season is
   correlated with his neighbours in time (career arcs, role continuity), league-wide behaviour drifts
   (pace, three-point volume, rules), and a random split would train on seasons after the ones it is
   tested on, which is leakage by construction and inflates every metric. Walk-forward is also the
   only scheme that matches how the model will actually be used (preseason, once).

3. **Leakage is defended in layers.** (a) Structural: models get only `History`, checked by
   `assert_no_future` plus a check of `extras`. (b) `players` is sanitised by `History.until` itself
   (`src/contracts.py`) — `from_year`/`to_year` (and the mere existence of future draftees and
   undrafted debutants) are derived from future seasons, so `History.until` keeps a player only if
   drafted by season S's draft or actually present in the season-sliced game logs, unconditionally;
   this module's own `build_history`/`sanitize_players` no longer re-sanitise on top of that, and
   exist only for backward compatibility (see the Consequences update below — this used to describe
   the other way around, before the fix). (c) Behavioural: `assert_projector_ignores_future` scrambles
   all season >= S rows **in place** in the caller's frames and requires identical output; in-place is
   the point, because the cheating pattern that matters is a projector holding a reference to the full
   tables. The oracle and partial cheaters are in the test suite and must be caught. (d)
   Non-deterministic projectors are rejected explicitly, so noise cannot masquerade as leakage or hide
   it.

4. **Actuals use the league scoring** (`fantasy_points_frame` with `config/league.yaml`), so the
   target is exactly what the league pays.

5. **Evaluation universe: nobody silently disappears.** A projected player who plays 0 games is
   scored as 0 actual total FP; a player who plays but was not projected is a reported coverage miss
   that can occupy the actual top-K. Rank metrics that would otherwise let a model choose its own
   scoring set (top-K, NDCG) are computed over the union universe; the actual top-K pool is everyone
   who played.

6. **A metric panel, not a single number.** Rank quality (Spearman on total FP and FPPG, top-K hit
   rate for K in 12/50/100, NDCG@K), decision value (top-K value capture, VORP-weighted error), error
   size (MAE/RMSE/bias for FPPG, GP, total FP) and interval calibration (game-level share below p10,
   above p90). Total FP is the primary target because that is what a season-long draft buys; FPPG and
   GP metrics decompose where the error comes from.

7. **VORP weights use `max(pred, actual) - R`.** Weighting by the actual alone rewards models that
   never take risks (weights vanish for players who bust); weighting by the prediction alone ignores
   missed breakouts. Taking the larger of the two charges both false stars and missed stars, and gives
   zero weight to replacement-level filler. R defaults to the realised (N+1)-th best season total
   (N = league roster capacity) and is pluggable so the value engine's replacement level can replace
   it without touching the harness.

8. **Floor/ceiling calibration is judged at game level.** The contract defines `fppg_p10/p90` as
   quantiles of game-level FP, so they are compared with each actual game, not with the season mean.

9. **Uncertainty: stratified player bootstrap; paired for comparisons.** CIs come from resampling
   players within each season; ablation lifts use the same resampled rows for both variants (paired),
   restricted to players both projected, oriented so positive is better, and reported with the number
   of seasons where the lift is positive. Verdicts use the CI, not a p-value alone.

10. **Validate what projectors return, harshly.** Contract conformance, season match, one row per
    player across all `model` values, finiteness, `total = fppg * gp`, FPPG consistent with league
    scoring (a model using the wrong scoring fails at once, not silently), ordered quantiles. Failures
    name projector and season.

11. **Rank-only benchmarks are first-class.** ADP is a ranking, not a stat projection. Rather than
    fabricating fantasy-point values for it, a projector may declare `rank_only` and is scored on rank
    metrics only; error and calibration metrics are NaN, never zero.

12. **Reproducibility over convenience.** Deterministic processing order, order-insensitive data hash
    in every run's metadata, seeded bootstrap, optional projection cache keyed by projector
    fingerprint and the content hash of the history (so a stale cache cannot be served after model
    or data changes), synthetic runs bannered on the first line of their report.

13. **Miss analysis is descriptive.** Tags (rookie, availability, team change, minutes, age) are
    computed from data with the exact rate-vs-games error decomposition; they say what changed, not
    why.

## Alternatives considered

* **Random k-fold / leave-one-season-out CV.** Rejected (decision 2). Leave-one-season-out trains on
  the future of the held-out season.
* **Expanding-window with periodic refit only (e.g. refit every 3 seasons).** Cheaper but
  lets stale models be scored against later seasons and complicates the "one season = one draft"
  interpretation; the harness leaves refit cadence to the projector's own training, which sees a fresh
  `History` every season.
* **Drop zero-game projected players** (common in published fantasy accuracy studies). Rejected: it
  is the single biggest flattering choice, because the misses that matter most (injured stars) are
  exactly the ones removed.
* **Score only on the projected universe for top-K.** Rejected: a model could raise its top-K by
  omitting hard-to-project players. The coverage miss count and top-K cap keep that visible.
* **One headline metric (e.g. Spearman only).** Rejected: it hides scale errors (a ranking can be
  perfect while every projection is 20% high) and ignores where in the ranking errors fall.
* **Weighted-by-actual VORP; DCG with exponential gains.** Rejected for the reasons in decision 7 and
  because FP is already ratio-scale, so linear gains are the honest choice.
* **Compare intervals to the season-mean FPPG.** Rejected (decision 8).
* **Parametric significance tests (paired t-test on per-season metrics).** With about 10 seasons the
  normality assumption is untestable and power is poor; the player bootstrap plus season win-count is
  more informative, though still imperfect (see limitations).
* **Committing generated reports.** Rejected; reports are reproducible from one command and the data
  hash, and generated files would churn.

## Limitations (also in docs/backtest.md)

* Preseason information gap: the backtest cannot know offseason moves that happen after the data
  cut; features that are not as-of-dated would overstate performance.
* About 10 seasons; coarse top-12 metric; regime drift; optimistic bootstrap CIs (independent player
  resampling ignores cross-season correlation); multiple comparisons in the ablation.
* Season-total scoring, not H2H simulation.
* The behavioural leakage check cannot see a projector that reads data from disk on its own or keeps
  a pre-check copy of the tables; it is a tripwire that complements, not replaces, structural
  isolation and review.
* Hyperparameter tuning on the evaluation seasons would make results optimistic; tuning has to be
  nested in the walk-forward or disclosed.

## Consequences

* Every future model or feature layer is scored the same way with one command, and a new layer must
  earn a consistent `improves` verdict to stay in.
* A projector that cannot cover a player pays for it (coverage miss), so rookies/returners need a real
  path rather than being ignored.
* **Update (2026-09-22, post-launch):** the observation above described the state when this ADR was
  written and prompted a fix. `History.until` (`src/contracts.py`) now sanitises the `players` table
  itself, unconditionally — the "safer long-term fix" happened. `build_history` no longer
  re-sanitises on top of it; the `sanitize=`/`sanitize_players=` flags on it and on
  `assert_projector_ignores_future` are kept only for backward compatibility and are no-ops today.
  See `src/backtest/leakage.py`'s module docstring and `docs/backtest.md`'s leakage-guards section
  for the current state.
