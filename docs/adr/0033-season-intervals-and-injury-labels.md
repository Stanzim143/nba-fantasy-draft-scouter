# ADR 0033: Season-total intervals, labelled injury data, and what was deliberately not built

**Status:** accepted, 2026-10-01. Extends [0030](0030-methodology-critique-fixes.md) (its uncalibrated season-value uncertainty) and
[0031](0031-hurdle-availability.md); uses the source verdict of [0005](0005-data-sources.md) (NBA official injury reports, "USE").
**Code:** `src/models/season_interval.py`, `src/models/baseline.py` (`fit_season_uncertainty`, simulation in `_assemble`),
`src/ingest/nba_injury_reports.py`, `src/features/injury_labels.py`, `src/models/adp_baseline.py` (`ContextHooks`),
`src/ops/nightly.py` (`injuries` step), `src/backtest/method_checks.py` (`interval_coverage_table`), `src/backtest/runner.py`.
Tests: `tests/models/test_model_season_interval.py`, `tests/ingest/test_nba_injury_reports.py`, `tests/features/test_injury_labels.py`,
`tests/models/test_model_labels.py`, `tests/ops/test_ops_nightly_injuries.py`, `tests/backtest/test_method_checks_hurdle.py`.

## Part 1: season-total intervals

### Context

`fppg_p10/p50/p90` are quantiles of a *game*; nothing said how wide a *season total* can be. ADR 0030 recorded that this needed new modelling
(season-mean FPPG uncertainty) and that games-played coverage per risk group was unchecked. Head-to-head drafting cares: a 3,000-point player
with a 25% chance of missing the year is not the same asset as a safe 2,600.

### Decision

`BaselineConfig.season_intervals` adds `proj_total_fp_p10`, `proj_total_fp_p50`, `proj_total_fp_p90` (every existing column unchanged):

```
T = Fbar * G
G     ~ 0 w.p. 1 - P(appear), else L * Beta(mu nu, (1 - mu) nu)           the hurdle mixture of ADR 0031
Fbar  ~ Normal( proj_fppg, sqrt( (tau * proj_fppg)^2 + sd_game^2 / max(G, 1) ) ),  clipped at 0
tau   = relative spread of (actual season FPPG - projection) / projection, by seasons of history (0, 1, 2, 3+),
        estimated from the history's own player-seasons projected from their lags (veterans through the model, players with no history
        through the draft-slot prior); game noise sd_game^2 / gp is subtracted so it is not counted twice; floor 2%
```

The quantiles come from 1000 deterministic draws per player (`default_rng([0, player_id])`, so a player's interval never depends on who
else is in the projection). `Fbar` and `G` are independent (checked, not assumed, by the coverage table). The offseason layer scales the three
columns with the FPPG factor it already applies to the game band. `baseline_hurdle*` enable it; the plain `baseline` has no band.

### Results (real data, walk-forward 2016-17 to 2025-26, `baseline_hurdle`; out of sample by construction)

Share of realised season totals inside the nominal 80% band (10% below p10, 10% above p90), zero-game players included:

| group | n | in band | below p10 | above p90 | games-played band |
|---|---|---|---|---|---|
| all projected players | 6,459 | 85.3% | 5.5% | 9.2% | 90.2% |
| prime, played 75%+ last year | 1,736 | 82.4% | 9.3% | 8.3% | 90.0% |
| part-season last year | 1,148 | 83.8% | 5.8% | 10.4% | 91.7% |
| age 33+ | 210 | 93.8% | 2.4% | 3.8% | 97.1% |
| returner (missed over half, or absent) | 2,828 | 90.2% | 1.0% | 8.8% | 91.4% |
| rookie (no prior NBA games) | 537 | **68.2%** | 18.1% | 13.8% | 79.0% |

Reading it: near nominal where the model has history (prime 82%), conservative for returners (a zero floor and a wide top: the mass at zero
makes p10 uninformative, which is the honest statement), **too narrow for rookies**, whose own chance of not appearing is not modelled (the
`players` table holds only players who appeared, so drafted-but-never-played rookies cannot be identified to train on) and whose talent spread is
the largest. Use the rookie band as a lower bound on the real uncertainty.

## Part 2: labelled injury data (item 5)

### Context

The injury features of ADR 0006 and the load-management study of ADR 0024 work from games missed; they cannot tell injury from rest from a
G League assignment, and cannot give a spell's duration. The NBA publishes a report before every game with a reason per player
(docs/research/data-sources.md 4.3: layouts verified, parser matched independent status counts, robots.txt allows `/referee/injury/`,
`Crawl-Delay: 1`, same accepted personal / non-commercial / local-only stance as ADR 0005).

### Decision

* **Ingest** (`python -m src.ingest.nba_injury_reports`): for each NBA game date from 2018-12-19 on, one report (the first existing of
  `05PM`, `06PM`, `01PM`, `03PM`, `08AM`), at most one request per 1.5 s, a User-Agent without an e-mail address, raw PDFs under
  `<data dir>/raw/nba_injury` (never committed), absent dates remembered (a date within the last 5 days is re-probed, since a report may be
  published late), `pdftotext -table` for text, a content-based parser for the three layouts, names mapped to `player_id` (unique names directly,
  ties broken by who played for that team). Resumable; `--offline` parses the cache; incremental runs parse only the days they downloaded.
  The table carries `season`, so it rides in `History.extras` and is sliced to seasons before the target like every other extra.
* **Nightly**: a new `injuries` step (after `games`) keeps the table current in season. The reports cannot be rebuilt later, so the archive has to
  be kept going from opening night.
* **Features** (`src/features/injury_labels.py`): `classify_reason` buckets each line into injury / illness / recovery / rest / G League / personal /
  suspension / covid / not-with-team (and a body part); per (player, season), over the team games that have a report: `inj_out`, `rest_out`,
  `other_out` (shares of those games listed Out), `longest` (longest consecutive injury spell: the duration the report itself never states),
  `n_spells`. Recency-weighted over seasons with reports (a season with under 40% coverage is not trusted), plus a `covered` flag; an uncovered
  row is all zeros, so it still trains the base model. They feed the availability and appearance models (`baseline_labels`,
  `baseline_hurdle_labels`, `baseline_hurdle_adp_labels`).

### Results

Crawl (2026-10-01): 1,108 of 1,211 game dates from 2018-12-19 had a report (103 absent: no report published), 116,959 rows, 98.8% matched to a
`player_id`, 0 parse problems. Because reports start in 2018-19, only seasons from 2020-21 can be scored with a prior season of labels.

Ablation `baseline_hurdle_adp` -> `baseline_hurdle_adp_labels`, 2020-21 to 2025-26 (6 seasons, paired bootstrap over players):

| metric | without | with labels | lift | 95% CI | seasons won |
|---|---|---|---|---|---|
| Spearman, total FP | 0.840 | 0.842 | +0.002 | [+0.001, +0.004] | 3/6 |
| MAE total FP | 327.9 | 324.0 | -3.9 | [-5.4, -2.3] | 5/6 |
| MAE games played | 13.77 | 13.67 | -0.09 | [-0.16, -0.02] | 5/6 |
| top-50 hit | 0.663 | 0.673 | +0.010 | [-0.017, +0.027] | 3/6 |
| bias total FP (pred - actual) | +34.8 | +15.6 | | | |

Verdict: a real but small gain, concentrated in availability and bias, none in the top-end ranking. **Not promoted into the board stack**:
six scored seasons cannot justify adding a runtime dependency on a crawled archive for about 1% of MAE. The variants stay registered
(`baseline_hurdle_labels`, `baseline_hurdle_adp_labels`) and the archive keeps growing nightly, so the question can be re-run with more
seasons (a future promotion needs a `*_labels_offseason_debut` registry entry and a full-stack backtest). Without a table the labelled projectors
reduce exactly to their base model (tested).

## Part 3: point-in-time positions (item 6): decided not to build

Historical position groups come from the current `players.position` label. The `baseline_nopos` ablation (every player `U`) matches the shipped
model (Spearman 0.785 vs 0.784, MAE of total FP 422.8 vs 422.5, ADR 0030): the position priors carry no measurable benefit, so a leak through them has no
measurable effect and a point-in-time version has nothing to recover. No dated position source exists in the data contract (`roster_snapshots` covers
only the current preseason). The item was marked low priority for exactly this reason; it is recorded here as closed by evidence, not as an
oversight. It would be revisited only if a position-aware feature (for example positional scarcity in the value engine, which is already
"weak" for this league) ever showed a measurable effect.

## Consequences

* Board and app rows of hurdle models carry a season-total band; the app's glossary documents every new column.
* The injury archive is a new private data asset (DATA.md row added); `./dev data pack` includes `injury_reports.parquet` (derived) but never the raw PDFs.
* `pdftotext` (poppler/xpdf) must be on PATH for the ingest; nothing is pip-installed, no dependency changed.

Reproduce: `python -m src.backtest --model baseline_hurdle --method-checks` (interval coverage), `python -m src.ingest.nba_injury_reports --seasons 2018-19:2025-26`
(about an hour at the polite rate), then `python -m src.backtest --ablate baseline_hurdle_adp,baseline_hurdle_adp_labels --seasons 2020-21:2025-26`.

## Addendum 2026-10-02: calibration is out-of-fold

`calibrate_quantiles` (game-FP quantile shape) and `fit_season_uncertainty` (`tau`) originally measured residuals of the model on the very
player-seasons it was fit on, which understates the spread. They now use expanding-window out-of-fold projections
(`BaselineProjector.oof_projections`): each of the last `oof_folds` (3) training seasons is projected by a copy of the model refit on the history
strictly before it (`BaselineConfig.oof_calibration`, `oof_min_seasons`; in-sample residuals remain the fall-back when no fold qualifies, and
`oof_calibration=False` restores the old behaviour). Point projections do not change. On the 2026-10-02 real-data fit for 2026-27: game quantile
shape `(-1.289, -0.070, +1.542)` -> `(-1.298, -0.083, +1.581)`; `tau` by seasons of history (0, 1, 2, 3+) `0.473, 0.392, 0.362, 0.270` ->
`0.470, 0.450, 0.417, 0.283`, so season-total bands widen by roughly 10-15% for players with 1-2 seasons of history. The coverage figures in this ADR
and in `docs/categories.md` section 4.4 were measured before this change; re-run `python -m src.backtest --model baseline_hurdle --method-checks`
to refresh them (a refit costs about 3x more because of the inner refits).
