# Backtest methodology and how to run it

Code: `src/backtest/`. Decisions and rejected alternatives: [ADR 0004](adr/0004-backtest-methodology.md).
Tests: `tests/backtest/test_bt_*.py`.

The backtest answers one question honestly: **if this model had produced a preseason ranking for
each of the last N seasons, using only what was knowable before that season, how close would it have
come to what actually happened under this league's scoring?**

## Run it

```bash
# Backtest a registered model over a range, write reports/<run-id>/
python -m src.backtest --model baseline --seasons 2016-17:2025-26 --out reports/

# One season only (the "rerun any season" command)
python -m src.backtest --model baseline --seasons 2021-22

# Against the naive benchmark (paired lift + significance) and an ADP file
python -m src.backtest --model baseline --benchmark naive_last_season --adp-file data/adp.csv

# Ablation: ordered layers, each layer's lift over the previous one
python -m src.backtest --ablate baseline,baseline_injury,baseline_roster,baseline_contract

# Pipeline smoke test on a simulated league. NOT A RESULT: the report says so in its first line.
python -m src.backtest --synthetic --model naive_last_season --seasons 2017-18:2018-19
```

Useful flags: `--leak-check` (run the future-invariance test on the last season), `--ks 12,50,100`,
`--min-gp 10`, `--n-boot 500`, `--seed 0`, `--top-misses 10`, `--cache-dir DIR`, `--run-id NAME`.
Exit codes: 0 ok, 2 user/data error (unknown model, missing tables, bad seasons), 3 leakage detected.

Model names resolve through `src.models.registry.get_projector(name)`, imported lazily; without the
registry only `naive_last_season` (a local implementation) runs from the CLI. Data comes from the
Parquet store (`NBA_DATA_DIR`, see ADR 0001); `--synthetic` builds a league in memory instead.

Programmatic use:

```python
from src.backtest import walk_forward, run_ablation, assert_projector_ignores_future, write_report
from src.store import load_tables

tables = load_tables()
res = walk_forward(tables, my_projector, ["2021-22", "2022-23"])
res.season_metrics      # one row per season
res.summary()           # mean across seasons
res.summary_ci()        # bootstrap CIs
res.players             # player-level frame (projection vs actual, every season)
```

## What is evaluated

### Walk-forward, no random cross-validation

For target season S the projector receives `History.until(tables, S)`: every season-indexed table cut
to seasons strictly before S. It projects S; the projection is scored against S's realised results;
repeat for every requested season. Data is time-ordered, seasons are not exchangeable (careers,
league-wide pace and three-point volume drift), and a random split would train on the future of the
rows it is tested on. See the ADR.

### Actuals

`actuals.season_actuals` computes, per player-season, from `game_logs` under the league scoring in
`config/league.yaml` (through `fantasy_points_frame`): games played (`gp`), total minutes, `mpg`,
total fantasy points, and `fppg = total_fp / gp`. `actuals.game_fp` gives every individual game's
fantasy points (used for floor/ceiling calibration).

### The evaluation universe (the rules that keep the numbers honest)

| Group | Definition | Treatment |
|---|---|---|
| Projected and played | in the projection and has >= 1 game | compared normally |
| **Projected, 0 games** | in the projection, no `game_logs` rows that season | **kept**, actual GP = 0 and actual total FP = **0**. Actual FPPG is undefined (NaN), so FPPG metrics skip them; total-FP, GP and rank metrics include them |
| **Played, not projected** | >= 1 game, no projection row | a **coverage miss**: kept with NaN prediction, never selectable as a predicted top-K player, may sit in the actual top-K (capping the achievable hit rate), and reported (count, share of league FP, how many are in the actual top-100) |

Dropping the second group is the classic way to flatter a model (the injured busts vanish); the
third group is where a last-season-only model bleeds (rookies, returners), so both are explicit.
Tests: `tests/backtest/test_bt_actuals.py`, `test_bt_harness.py`.

### Validation of what a projector returns

Before scoring, `validate_projections` requires: the `projections` contract (columns, dtypes, no
nulls); every row is for the target season; **one row per player across all `model` values**;
finite values; `0 <= proj_gp <= 100`; `proj_total_fp == proj_fppg * proj_gp`; `proj_fppg` equal to the
league scoring applied to the projected stat line (catches a model built on the wrong scoring);
ordered `p10 <= p50 <= p90`. Failures raise `ProjectionValidationError` naming the projector and
season. Projectors with `rank_only = True` (the ADP benchmark) need only
`season, player_id, player_name, model, proj_total_fp`.

## Metrics

All are computed per season, then averaged across seasons (each season is one draft). "Universe" is
defined above. `bias = mean(pred - actual)`; positive means over-projection. Pairs with NaN on either
side are dropped from error metrics; degenerate inputs return NaN rather than a misleading 0.

| Name | Definition | Rows scored |
|---|---|---|
| `spearman_total_fp` | Spearman rank correlation (Pearson on average ranks) of projected vs actual season total FP | projected players (zero-game players count at actual 0) |
| `spearman_fppg` | same, on FPPG | projected players with `actual_gp >= min_gp` (default 10) |
| `top{K}_hit` | \|predicted top-K intersect actual top-K\| / min(K, players who played). K in (12, 50, 100), configurable. Ties break by `player_id` order | universe |
| `top{K}_capture` | actual FP of the predicted top-K / actual FP of the true top-K (draft value captured; 1.0 = perfect picks). A projected player who never played is a 0-value pick | universe |
| `ndcg_{K}` | NDCG@K, linear gains = actual total FP (floored at 0), discount `1/log2(rank+1)`; tied predictions get the expected DCG under random tie-breaking; unprojected players rank last. Top-heavy: the first picks matter most | universe |
| `mae_/rmse_/bias_` `fppg`, `gp`, `total_fp` | mean absolute error, root mean squared error, signed bias | fppg: as `spearman_fppg`; gp and total_fp: projected players |
| `vorp_weighted_mae`, `vorp_weighted_bias` | `sum(w*abs(e))/sum(w)` and `sum(w*e)/sum(w)` with `w = max(max(pred, actual) - R, 0)` on total FP. A player matters if the model or reality put him above replacement level R; errors among filler get weight 0 | projected players |

**Replacement level R.** Default: the (N+1)-th best realised season total FP among everyone who
played, N = teams x roster size from `config/league.yaml` (182 for the current 14-slot league; the
backtest results reported in the README were run at 169, the 13-team league of 2026-09-23; 0.0 if the universe is smaller than N). It is derived from realised
results, not from the model. Pass
`replacement=<float>` or `replacement=<callable values -> float>` to `walk_forward` to substitute the
value engine's replacement level once it exists.

Coverage and calibration columns per season: `n_projected`, `n_played`, `n_projected_no_games`,
`n_coverage_miss`, `coverage_miss_fp_share`, `top{K}_unprojected`, `replacement_level`,
`share_below_p10`, `share_in_p10_p90`, `share_above_p90`, `share_below_p50`.

### Floor and ceiling calibration

`fppg_p10/p50/p90` are quantiles of **game-level** fantasy points (see the contract), so they are
judged against each actual game: the share of a player's games below p10, below p50 and above p90,
pooled over players with a band. A calibrated band leaves about 10% of games below p10, 10% above
p90 (80% inside), 50% below p50. Comparing the season *mean* to a game-level band would be a category
error, so it is not done.

### Uncertainty

* **Bootstrap CI** (`BacktestResult.summary_ci`): percentile bootstrap over players, stratified by
  season (each replicate resamples every season's players and averages the per-season metric).
  Deterministic given `seed`.
* **Caveat:** players are resampled independently although the same player recurs across seasons and
  a season's shocks (a rule change, one superstar's injury) are shared, so the CIs capture player
  sampling noise only and are optimistic. Read them together with the per-season table: a lift that
  holds in 9 of 10 seasons is stronger evidence than a tight CI.

### Ablation

`run_ablation(tables, variants, seasons)` (or `--ablate a,b,c`) runs an ordered list of projectors
over the same seasons. Rows are variants; columns are mean metrics. For each variant after the first
and each metric in `sig_metrics` (default `spearman_total_fp`, `top50_hit`, `mae_total_fp`) it reports
the **lift over the previous variant** with a paired, season-stratified bootstrap:

* lift is oriented so that positive always means better (error metrics are sign-flipped);
* 95% CI of the lift, one-sided bootstrap p-value for "lift <= 0", seasons won out of seasons compared;
* verdict: `improves` (CI wholly > 0), `hurts` (CI wholly < 0), else `no significant change`;
* the paired comparison uses players projected by **both** variants, so both are scored on identical
  rows; differences in coverage show in the un-paired columns (`top{K}_hit`, `n_coverage_miss`) but
  not in the lift;
* several layers and several metrics are tested, so some "significant" results will be chance:
  prefer lifts that are consistent across metrics and seasons. The contract layer was a stated
  hypothesis; keep it only if it earns a consistent `improves`. It did not (ADR 0013): a CI that excludes zero
  on Spearman at +0.0004, in the 3 of 10 seasons where its gate was open, with top-50 and MAE flat, is not a consistent
  `improves`. Note `baseline_contract` is stacked on plain `baseline` (like `baseline_offseason`), so in a four-row
  chain its "lift over the previous variant" is over `baseline_roster`; run `--ablate baseline,baseline_contract` for the
  meaningful comparison.

### Miss analysis

`analyze_misses` lists the top-N absolute total-FP misses per season and tags them with observable
changes, priority `rookie/debut > availability > team change > role/minutes > age > unexplained`:
no earlier games (`rookie/debut`); the games-played component of the error is at least half of it and
the GP gap is >= 20 (`availability`, using the exact decomposition
`error = (actual_fppg - proj_fppg) * actual_gp + proj_fppg * (actual_gp - proj_gp)`); different last
team than the previous season or 2+ teams in the season (`team change`); minutes per game moved >= 5
from the projection (`role/minutes`); age >= 33 or <= 22 (`age`). These are descriptions, not causal
claims.

## Leakage guards

Three layers (`src/backtest/leakage.py`, tests in `test_bt_leakage.py`):

1. **Structural.** `build_history` = `History.until` + `assert_no_future` + a check that `extras`
   contain no future season + **checking the static `players` table is sanitised**. `History.until`
   (`src/contracts.py`) sanitises `players` itself, unconditionally: two of its columns are derived
   from the future — `to_year` (last NBA season) and `from_year` (first) — and `from_year == S` would
   otherwise name every season-S rookie, `to_year >= S` who keeps playing. `History.until` keeps a
   player only if they were drafted by season S's draft or actually appear in the season-sliced
   `game_logs`/`player_season_bio` (never by trusting `from_year` alone, which is a derived column
   and, on real data, occasionally out of step with actual play history), nulls `from_year >= S`, and
   caps `to_year` at S-1. `build_history` no longer re-sanitises on top of this — an earlier version
   of this module discovered the leak and worked around it locally before the fix moved into the
   shared contract; the `sanitize=`/`sanitize_players=` flags here and on
   `assert_projector_ignores_future` are kept only for backward compatibility and are no-ops today.
2. **Behavioural.** `assert_projector_ignores_future(projector, tables, season)` runs the projector,
   then runs it again while every row of season >= S (and the future-derived parts of `players`) is
   **scrambled in place in the caller's own DataFrames**, then restores them and verifies by content
   hash. Output must be identical. In-place matters: a projector that captured a reference to the full
   tables sees the scrambled data and is caught (`OracleProjector` in the tests is; an honest projector
   passes). A non-deterministic projector is rejected with a different error first, so noise cannot be
   mistaken for leakage.
3. **Sanity.** A too-good score is a smell. The oracle projector scores about 1.0 by construction.

Not caught by the behavioural check (state plainly): a projector that reads files from disk itself,
holds a *copy* of the tables made before the check, or captured a single column object rather than
the frame; and leakage in features that the projector itself derives from data outside `History`
(for example a contracts table that is not as-of-dated). Those need code review and as-of-date
tables; the check is a tripwire, not a proof.

## Benchmarks

* **`naive_last_season`** (`benchmarks.NaiveLastSeason`): each player projected at exactly his last
  season's per-game line and GP; ranking = last season's fantasy points; bands from last season's
  game-level quantiles. Local implementation for tests; the CLI prefers the registry's entry.
* **ADP / consensus** (`benchmarks.load_adp`, `AdpBenchmark`): input is a CSV or Parquet with columns
  `season` (e.g. `2023-24`), `source` (must exist in `player_id_map`, e.g. `espn`), `source_id`,
  `adp` (lower = earlier), optional `name`; ADP must be the pre-draft value. Ids map to `player_id`
  through `player_id_map` (optionally `min_confidence`); unmapped and duplicate rows are counted and
  reported, the lowest ADP wins a duplicate. A missing or malformed file raises `BacktestError` with
  the expected schema; nothing is downloaded. ADP is a ranking, so `AdpBenchmark.rank_only = True`:
  only Spearman, top-K hit/capture and NDCG are scored, error and calibration metrics are NaN.

## Output

`reports/<run-id>/` (default id `<model>_<first>_<last>_<data-hash8>`; not committed, see
`reports/.gitignore`): `report.md`, `metrics_by_season.csv`, `players.parquet`, `run.json`
(config + metadata), `misses.csv`, `ablation.csv`, and PNGs `metrics_by_season`, `scatter_total_fp`,
`calibration`, `ablation`. The report contains: run metadata and reproduce-command, headline metrics
with CIs, per-season table, evaluation-universe/coverage table, benchmark comparison with paired
lifts, predicted-vs-actual scatter, calibration, ablation, biggest misses, and a notes section.
Synthetic runs carry a banner on the first line.

`BacktestResult.save/load` round-trip a run. `run.json` metadata: `data_hash` (order-insensitive
sha256 of the input tables), `created_utc`, harness version, package versions, projector class and
`fingerprint`, leak-check results, per-season seconds.

## Adding a projector

Implement the contract's protocol:

```python
class MyProjector:
    name = "my_model"                          # used in reports and as the ablation label
    fingerprint = "my_model/params-v3"         # optional; required only for --cache-dir; change it
                                               # whenever code or parameters change
    def project(self, history: History) -> pd.DataFrame:
        ...                                    # a `projections` frame for history.target_season
```

Rules: use only `history` (never load tables yourself); be deterministic (seed everything); return one
row per player with `proj_total_fp == proj_fppg * proj_gp` and `proj_fppg` under the league scoring
(`benchmarks.stats_to_projection` builds a valid frame from per-game stat projections and GP);
bands are optional (NaN) but if present must be game-level p10 <= p50 <= p90. Register it in
`src.models.registry`, then:

```python
assert_projector_ignores_future(MyProjector(), tables, "2022-23")   # in the model's own test suite
```

Optional class attribute `rank_only = True` declares a ranking-only projector.

## Determinism and reproducibility

No randomness is consumed by the harness itself; rows are processed in `player_id` order; results do
not depend on input or projection row order (tested). The bootstrap is seeded. `--cache-dir` caches
projections per (projector, fingerprint, history content hash, season) so appending a season, or
rerunning one, recomputes only what changed; cached frames are re-validated on read.

## Known limitations

* **Preseason information gap.** The backtest projects from data through the end of season S-1. A live
  draft also knows offseason trades, signings, contract news and preseason injuries. Where a feature
  cannot be reconstructed as-of-date it must be excluded, and the backtest therefore understates what a
  fully informed live model could do; conversely any such feature that is *not* point-in-time would
  overstate it. Injury/roster/contract layers are only as honest as their as-of-date construction.
* **Small samples.** About 10 evaluation seasons and ~400-500 players each; top-12 hit rate is
  coarse (steps of 1/12). Do not read decimals into it.
* **Regime drift.** Rules, pace and three-point volume shift across 2015-2025; averaging seasons
  hides trends. Look at the per-season table.
* **Model-selection leakage.** Choosing hyperparameters by looking at these same seasons' scores
  makes the reported numbers optimistic. Tuning must be nested inside the walk-forward (train on
  seasons before S only) or be declared as tuned on the evaluation seasons.
* **Optimistic CIs** (see Uncertainty) and multiple comparisons in the ablation.
* **Season-total scoring, not H2H.** Metrics evaluate season-long value; they do not simulate weekly
  head-to-head matchups or in-season management. Variance (floor/ceiling) is judged only through
  calibration.
* **Position and roster construction** are not part of these metrics; VORP weighting uses a single
  replacement level.
* Coverage misses are scored as absent, not as a default projection: a model that quietly gives every
  rookie a mediocre line would score better here than one that skips them, which is the intended
  incentive.

## Breakout evaluation (`python -m src.backtest.breakouts`)

The whole-league metrics above cannot answer whether an offseason signal finds young players about to break out (most players never
appear in Summer League or preseason, so a layer that matters only for them barely moves league-wide numbers).
`python -m src.backtest.breakouts --model baseline_offseason` projects every season walk-forward with the base and the layered model,
then scores subgroup FPPG accuracy (paired bootstrap), and breakout detection (AUC, precision@K against each season's base rate,
bootstrap over seasons) for young players, young under-the-radar players and rookies, on two pre-declared targets: a *breakout* (beat
the baseline by 4 FPPG and 25% over 20+ games) and a *useful* breakout (also finished in the rostered top 169). It writes a report,
the evaluation frame, and with `--save-calibration` the probability calibration the watchlist uses. `--preseason-fraction 0.5` keeps
only the first half of each preseason (a draft held mid-preseason). Definitions and results: ADR 0012.

## Contract-terms evaluation (ADR 0019)

`python -m src.backtest.contract_terms_eval --leak-check --n-boot 1000 --out reports/ --run-id contract_terms_eval` runs `baseline`, `baseline_contract`, `baseline_contract_terms` (gated) and the same layer with the gate forced open
(a diagnostic), and writes `report.md`: league-wide paired lift, the same restricted to players by contract state at the start of the season (known deal, final year, new deal, any/no Wikipedia event; rank metrics within the subset),
the gate history per target season, and the coverage-conditional effect size (baseline residual by contract state with a player-clustered bootstrap, pooled and by era). `player_contracts` travels as `History.extras`
(its `season` column is the season the event follows), so the structural slice and `--leak-check` cover it. The ordinary `--ablate baseline,baseline_contract,baseline_contract_terms` also works.

## Return-from-injury study (ADR 0021)

`python -m src.backtest.return_report --seasons 2016-17:2025-26 --board <draft_board.csv> --out reports/return_<date>` asks whether players
who missed one long block and then played healthily (Tatum in 2025-26: 62 of 82 games missed, 16 of the last 20 played) are under-projected
the next season, in games played and total fantasy points. Cohort and controls are fixed in ADR 0021 before any outcome was computed
(`CohortConfig` defaults); features come from `History.until(tables, s)`, the outcome from season `s`. It reports signed error
(actual minus projected), MAE and the share above projection for the cohort, for nearest-neighbour matched controls (same season, similar
age, minutes and prior fraction played, spread-out absences) and their paired difference, with a 2000-resample bootstrap stratified by
season; an 18-cell sensitivity grid, a leave-one-season-out table, a projected-GP calibration view, the same cohort under
`baseline_injury` and `baseline_offseason_debut`, and an informational table of the players currently in the cohort. It changes no model
or board output. Result: relative to matched controls the cohort is projected too low (+10.7 GP, +251 total FP), but the cohort's own bias is
+1.0 GP [-4.1, +5.8] and +62 FP [-80, +196], so its projection is about right in absolute terms.

## Return-health feature ablation (ADR 0022)

`python -m src.backtest --ablate baseline,baseline_return --seasons 2016-17:2025-26 --leak-check --n-boot 500 --sig-metrics mae_total_fp,mae_gp,spearman_total_fp,top50_hit --out reports/return_ablation`
is the pre-registered comparison; `--sig-metrics` (new, comma list, default unchanged) chooses the metrics of the paired lifts so that MAE of games
played gets an interval beside total FP. `python -m src.backtest.return_feature_eval --out reports/return_feature_<date>` adds the secondary
analyses: the same paired lift restricted to ADR 0021's cohort (and its >= 60% block subgroup), cohort membership computed point in time, and a
player's (default Tatum) projected GP and total FP before and after. Result: no lift, not recommended.

## Methodology checks (ADR 0030)

`python -m src.backtest --model baseline --benchmark naive_last_season,baseline_nopos --adp-file <processed>/adp.parquet --method-checks`
appends a section with (a) games-played calibration by risk group (rookie, returner, part-season, age 33+, prime), (b) leave-one-season-out
ranges of the headline means, and, with `--adp-file`, (c) the `adp`, `model` and `adp_model` arms on the ADP universe (expanding-window fits on
earlier seasons only, first season dropped) with top-12/24/50/100 hit and capture, season-paired differences vs ADP and total-FP error by ADP
rank band. Code: `src/backtest/method_checks.py`. Results and their reading are in ADR 0030; limits in [limitations.md](limitations.md).
`BaselineConfig.tune_with_predicted_mpg` (default True) and `use_position_priors` (registry `baseline_nopos`) are the switches behind the
before/after and ablation numbers there.
