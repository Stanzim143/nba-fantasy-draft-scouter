# ADR 0022: 'Returned and healthy' availability feature

**Status:** accepted, 2026-09-26: **registered, not recommended** (the pre-registration below was committed before any ablation output was computed; results, verdict and limitations follow it)
**Code:** `src/features/return_health.py`, `src/models/return_baseline.py`, registry entries, `src/backtest/return_feature_eval.py` (secondary analyses), `--sig-metrics` on `python -m src.backtest`.
**Evidence:** real walk-forward ablation over 2016-17..2025-26, `reports/return_ablation/`, `reports/return_ablation_injury/`, `reports/return_feature_2026-09-26/` (gitignored; the commands at the end of the pre-registration reproduce it).

## Context

The availability model (ADR 0003) sees only season-level fractions of the schedule played. A player who missed a long lead block and then
played nearly every team game after returning looks like a chronically absent one: Jayson Tatum 2025-26, first game 2026-03-06, 16 of the
20 team games after, season fraction 16 / 82 = 0.20. ADR 0021 measured what this costs. In absolute terms the baseline is roughly right for
such players on average (cohort n = 60, GP bias +1.0 [-4.1, +5.8], total FP +62 [-80, +196]; all CIs include zero), but relative to matched
spread-absence controls they are projected too low (+10.7 GP [+3.5, +17.7]) because the controls are over-projected. A hint (one of 18
cells, not significant) says a long-block subgroup (>= 60%) may be under-projected by about 8 GP. That study changed no model.

This ADR builds the feature that lets the availability model express "long lead block, then healthy tail" and evaluates it honestly, in
the way ADR 0006 evaluated the injury layer. The quantity that matters for the draft is **expected total fantasy points**
(`proj_fppg x proj_gp`; the board ranks by VORP on total FP), never FPPG alone, so every result is judged on total-FP error, games-played
(GP) error and rank quality on total FP. A prior expectation, stated before the run: the cohort is about 1% of projected rows, ADR 0021
found no large absolute bias in it, so a whole-universe lift will be small, and the adoption rule below is demanding on purpose.

## Decisions

* **D1. Features from `History` only.** Pure functions of `game_logs`, `team_games` and `players` (all sliced to seasons before the target),
  hence point-in-time like `src.features.injury`. Recency-weighted with the availability model's own `n_lags` (3) and `decay` (0.6); no
  separate search.
* **D2. Wiring through the existing hook.** `BaselineProjector._build_injury_features` returns the feature builder; `AvailabilityModel`'s
  `extra` matrix carries the columns. `src/models/availability.py` and `src/models/baseline.py` are **not modified**, so plain `baseline`
  is bit-identical by construction (and by a golden-value regression test, `tests/models/test_model_return.py`).
* **D3. Registered variants.** `baseline_return` (return columns only), `baseline_injury_return` (ADR 0006's four columns then the return
  columns in one `extra` matrix, via `ConcatFeatures`), and `baseline_return_offseason_debut` (the board's stack: debutants +
  Summer League / preseason, on top of the return base instead of `baseline`, built the same way as `baseline_offseason_debut`). No
  existing registry entry, default or board output changes.
* **D4. No NaN.** Missing values take the conventions below so the availability model keeps every training row (unlike the injury layer,
  whose NaN rows are dropped from the fit) and the fit and predict widths always agree.

## Feature definitions (fixed before evaluation)

Per player-season, on the player's *primary* team's schedule (the team he played most games for; `L` = that team's games in the season,
the same denominator as `src.features.injury` and `src.backtest.return_report`):

* `first_idx` = team games missed before the first appearance (the *lead block*); `lead_frac = first_idx / L`.
* `tail_len = L - first_idx` (team games from the first appearance to season end); `tail_gp` = games played (all after the first appearance).
* Raw tail health `tail_gp / tail_len`. **Shrunk** tail health `tail_health_s = (tail_gp + K * H0) / (tail_len + K)` with `K = 10` games and
  `H0 = 0.75`, so a 4-game tail with 4 played reads as 0.82, not 1.0.
* `lead_season` = `lead_frac >= 0.10` (a block worth speaking of).
* `return_flag` = ADR 0021's primary-cohort definition on the raw tail: `lead_frac >= 0.25`, `tail_len >= 15`, raw tail health `>= 0.75`.
* **No lead block** (`lead_frac = 0`, `lead_season = 0`, `return_flag = 0`) if the season is not usable: the player was not a veteran that
  season (no earlier season in the history's game logs and `players.from_year >= s`, so a debut or a mid-season signing is not "absence") or
  he played for more than one team (a trade makes the earlier team's games look missed, the ADR 0006 / 0021 limitation; the injury
  layer keeps such rows, this one zeroes them). Unlike ADR 0021 there is no minutes filter (the model already has projected minutes) and no
  cohort membership requirement: the feature is defined for every player.

The seven columns for a target season `s` (`w_k = 0.6^k` for the season `k + 1` years before `s`, zero for seasons the player did not appear
in; sums over the 3 lags):

| # | column | definition | value when undefined |
|---|---|---|---|
| 0 | `lead_bar` | `sum(w * lead_frac) / sum(w)` | 0 |
| 1 | `tail_health_bar` | `sum(w * lead_season * tail_health_s) / sum(w * lead_season)` | 0.75 (no lead season) |
| 2 | `tail_len_bar` | `sum(w * lead_season * tail_len / L) / sum(w)` | 0 |
| 3 | `return_bar` | `sum(w * return_flag) / sum(w)` (the recency-weighted interaction flag) | 0 |
| 4 | `lead_last` | last season's `lead_frac` | 0 (did not appear) |
| 5 | `tail_health_last` | last season's `tail_health_s` if it was a lead season | 0.75 |
| 6 | `return_last` | last season's `return_flag` | 0 |

Traded players: zero lead columns (above). No prior season, or no history at all: the "value when undefined" column, never NaN.

## Pre-registered evaluation

**Comparison.** Real walk-forward, every target season 2016-17..2025-26 (10 seasons), `baseline` vs `baseline_return` (the primary
comparison) and `baseline_injury` vs `baseline_injury_return` (reported with the same rule, labelled as the combination; it cannot replace the
primary comparison). Paired bootstrap over players, stratified by season, 500 resamples, seed 0, 95% percentile intervals, over players
both variants projected; positive lift = better (the harness's own `ablation` machinery; `--leak-check` on the last season).

**Primary metrics** (on the whole projected universe): **MAE of total FP** (the primary one), MAE of GP, Spearman rank correlation on total
FP, top-50 hit rate on total FP (the harness definitions, ADR 0004).

**Adoption rule (fixed).** `baseline_return` is *recommended for the board* only if **all** hold:

1. the 95% CI of the MAE-total-FP lift excludes zero in favour of `baseline_return`;
2. the 95% CI of the MAE-GP lift excludes zero in favour;
3. the top-50 hit lift is not significantly worse (its CI does not lie entirely below zero);
4. it wins at least 6 of 10 seasons on MAE total FP **and** at least 6 of 10 on MAE GP (point lift, season by season);
5. the leak check passes.

Otherwise it is *registered, not recommended* (as ADRs 0010, 0011 and 0013). A significant but tiny lift that meets every clause is still
reported with its size in relative terms; the rule decides "recommended", the size is read beside it. The same rule applied to
`baseline_injury_return` over `baseline_injury` decides whether the combination is worth recommending over the injury model, and is
reported separately.

**Secondary analyses** (reported whole; they cannot change the adoption decision):

* Lift (MAE total FP, MAE GP, and signed bias) restricted to ADR 0021's primary cohort and to its >= 60% lead-block subgroup, cohort
  membership computed point in time (`History.until`), pooled bootstrap resampled within season, 2000 resamples. Small `n`; descriptive.
* Tatum's projected GP and total FP before and after, from the history through 2025-26 (target 2026-27), for `baseline` vs `baseline_return`
  and for the board stack `baseline_offseason_debut` vs `baseline_return_offseason_debut`.

**Exploratory variants.** None are pre-registered. Any variant tried after seeing results (other shrink constants, block thresholds, lags,
last-season-only columns alone) is labelled exploratory in the results section and cannot drive the decision; the definitions above are not
tuned on the results.

**Reproduce (fixed command).**

```
python -m src.backtest --ablate baseline,baseline_return --seasons 2016-17:2025-26 --leak-check --n-boot 500 \
    --sig-metrics mae_total_fp,mae_gp,spearman_total_fp,top50_hit --out reports/return_ablation
python -m src.backtest --ablate baseline_injury,baseline_injury_return --seasons 2016-17:2025-26 --leak-check --n-boot 500 \
    --sig-metrics mae_total_fp,mae_gp,spearman_total_fp,top50_hit --out reports/return_ablation_injury
python -m src.backtest.return_feature_eval --seasons 2016-17:2025-26 --out reports/return_feature_2026-09-26
```

## Results (real walk-forward, 2016-17 through 2025-26; run once, definitions unchanged since the pre-registration)

Ran on 2026-09-26 with the fixed commands; `--leak-check` passed (`{'2025-26': 'passed'}`) for both ablations, and the leak tests in
`tests/models/test_model_return.py` pass for both variants.

**Primary comparison: `baseline_return` vs `baseline`** (paired bootstrap over players, stratified by season, 500 resamples; positive =
`baseline_return` better):

| metric | `baseline` | `baseline_return` | lift [95% CI] | seasons won | verdict |
|---|---:|---:|---|---:|---|
| MAE, total FP (primary) | 422.59 | 423.06 | -0.47 [-1.31, +0.29] | 2/10 | no significant change |
| MAE, games played | 19.219 | 19.256 | -0.037 [-0.085, +0.011] | 3/10 | no significant change |
| Spearman, total FP | 0.7836 | 0.7824 | -0.0012 [-0.0023, -0.0000] | 3/10 | hurts (CI upper end at zero) |
| top-50 hit, total FP | 0.656 | 0.654 | -0.0020 [-0.0120, +0.0060] | 0/10 | no significant change |

The feature makes the whole-universe projection marginally *worse*: MAE of total FP +0.11% and MAE of games +0.19% (relative), both
inside noise, with the rank correlation slightly lower. It wins 2 and 3 of 10 seasons on the two MAE metrics.

**Adoption rule.** Clause 1 (MAE total FP CI excludes zero in favour): **fails** (CI [-1.31, +0.29]). Clause 2 (MAE GP): **fails**
([-0.085, +0.011]). Clause 3 (top-50 not significantly worse): holds. Clause 4 (wins >= 6 of 10 on both MAE metrics): **fails** (2 and 3).
Clause 5 (leak check): holds. **Verdict: registered, not recommended**; nothing about the board changes.

**Combination: `baseline_injury_return` vs `baseline_injury`** (same rule, reported separately): MAE total FP -0.84 [-1.67, -0.06] (2/10),
MAE GP -0.061 [-0.110, -0.015] (2/10), Spearman -0.0020 [-0.0031, -0.0008] (2/10), top-50 +0.0020 [-0.0120, +0.0060] (1/10). Here the
return columns significantly *hurt* on three metrics; also not recommended.

**Secondary: where the feature acts** (lift = base error minus variant error on the same rows, pooled bootstrap within season, 2000
resamples; signed bias = actual minus projected, positive = under-projected; `reports/return_feature_2026-09-26/`):

| subset (n) | metric | MAE base | MAE variant | lift [95% CI] | bias base -> variant |
|---|---|---:|---:|---|---|
| all projected (7,000) | MAE total FP | 420.6 | 421.1 | -0.48 [-1.26, +0.21] | -133.6 -> -133.3 |
| all projected | MAE GP | 19.12 | 19.16 | -0.04 [-0.08, +0.01] | -10.03 -> -10.05 |
| ADR 0021 primary cohort (60) | MAE total FP | 525.5 | 540.7 | -15.1 [-54.2, +23.4] | +62 -> -107 |
| primary cohort | MAE GP | 20.28 | 20.00 | +0.28 [-1.73, +2.24] | +1.0 -> -7.8 |
| cohort, block >= 60% (21) | MAE total FP | 431.1 | 419.8 | +11.2 [-53.8, +79.1] | +210 -> +42 |
| cohort, block >= 60% | MAE GP | 19.37 | 17.39 | +1.98 [-1.46, +5.27] | +8.0 -> -0.6 |

The feature moves the cohort in the intended direction: it raises their projected games by about 8.8 GP on average (the cohort's bias
goes from +1.0 to -7.8 GP). But ADR 0021 had already shown that group was projected about right, so this is an **over-correction**: the
cohort's total-FP MAE gets worse (-15 [-54, +23], inconclusive) and its bias becomes a 7.8-game over-projection. In the >= 60% block
subgroup (n = 21) it removes the hinted under-projection (+8.0 -> -0.6 GP) and the point lifts are positive (+2.0 GP, +11 FP), but the
intervals include zero by a wide margin. On the combination the pattern is the same (cohort MAE GP +0.05 [-1.81, +1.92], total FP -15.5
[-51.5, +20.2]; block >= 60%: +2.0 GP [-1.3, +5.2], +16 FP [-44, +81]).

**Tatum before and after** (history through 2025-26, target 2026-27; FPPG is unchanged because the feature only touches availability):

| model | proj GP | FPPG | proj total FP |
|---|---:|---:|---:|
| `baseline` -> `baseline_return` | 50.07 -> 57.71 | 41.01 | 2,053 -> 2,366 |
| `baseline_injury` -> `baseline_injury_return` | 49.65 -> 57.65 | 41.01 | 2,036 -> 2,364 |
| `baseline_offseason_debut` (board) -> `baseline_return_offseason_debut` | 50.07 -> 57.71 | 41.01 | 2,053 -> 2,366 |

The feature adds 7.6 games (+313 total FP) to Tatum. ADR 0021's top-of-CI raw correction for the cohort was +5.8 GP; this is more, and
the variant over-projects the cohort as a whole by 7.8 GP. It is therefore not evidence that Tatum should move that far, and the board
keeps its `baseline_offseason_debut` row (50.1 GP, 2,053 total FP). `baseline_return_offseason_debut` is available but not the default.

**Exploratory variants: none were run.** The definitions were not adjusted after seeing the results.

## Verdict

**Registered, not recommended.** The 'returned and healthy' columns do not improve the availability model on the criteria that matter
for a total-fantasy-points draft: MAE of total FP and of games played move by about +0.1% and +0.2% in the wrong direction (intervals
include zero), the rank correlation on total FP is slightly lower (-0.0012, CI touching zero), and the season-by-season wins are 2 to 3 of
10. Combined with the injury layer it is significantly worse on three metrics. Where it changes projections (players in ADR 0021's cohort
and the >= 60% subgroup) it over-corrects the cohort on average, which ADR 0021 said was already about right in absolute terms; the one
subgroup where it looks helpful (n = 21) has intervals that include zero. ADR 0021's reading stands: the baseline's discount for such
players is about right in absolute terms and too high only relative to matched spread-absence players. One possible reading of the two ADRs
together (not a tested mechanism): encoding the block as a regressor lifts the cohort, whereas the relative gap came from the controls
being over-projected.

## Limitations

* **One specification.** Seven recency-weighted columns with fixed constants (`K = 10`, `H0 = 0.75`, `LEAD_MIN = 0.10`, ADR 0021 flag
  thresholds) and no search; a different shrink, threshold or a single flag column might do better or worse. The columns are correlated
  (`return_bar` / `return_last`, `lead_bar` / `lead_last`), which the availability model's ridge tolerates but does not sort out.
* **Dilution.** The cohort is under 1% of projected rows (60 of 7,000 in the 10-season universe), so a real cohort-level gain would show
  in the whole-universe MAE as a fraction of a percent; the adoption rule looks at the whole universe for that reason, and the cohort
  results are secondary and low-powered (n = 60 and 21).
* **Non-injury lead blocks.** "Missed the first games" includes contract holdouts, waived-and-re-signed players and G League time that
  `game_logs` cannot tell from injuries (ADR 0021); the feature has no minutes or role filter beyond veteran / single-team, so part of the
  signal it learns is of that mixed kind.
* **Team attribution and traded players.** Multi-team seasons are zeroed (no lead block), so a genuine return by a player who was also
  traded is not recognised; the primary team's schedule is used throughout, as in ADRs 0006 and 0021.
* **Availability is conditional on playing.** `mu` is fit on rows where the player appeared in the target season (ADR 0003), so
  retirements and season-ending injuries before opening night are outside `proj_gp`; the model's over-projection of everyone (about
  -10 GP, -134 total FP, both variants) is untouched.
* **Bootstrap over players, not seasons;** several rows per player across seasons, and two related comparisons that reuse the same data,
  make the intervals optimistic. The season-win counts are the check on consistency.
* **Leak check on the last season only,** as the harness does; the walk-forward itself slices every season with `History.until`.

## Consequences

* `baseline_return`, `baseline_injury_return` and `baseline_return_offseason_debut` are registered and tested; no existing model, default,
  contract or board output changed (`baseline` is bit-identical; `availability.py` and `baseline.py` are untouched).
* `python -m src.backtest` gained `--sig-metrics` (comma list of the metrics for the ablation lifts; the default is unchanged), and
  `src/backtest/return_feature_eval.py` re-runs the secondary analyses.
* Open, precisely: whether a Tatum-scale (>= 60% lead block) subgroup is under-projected by 5 to 10 games (ADR 0021) still cannot be
  settled with this sample; a board flag sized by judgement, as ADR 0021 suggested, remains the only use of this signal that the data
  does not contradict.
