# ADR 0003: Baseline projection model and value engine

**Status:** accepted, 2026-09-22
**Code:** `src/models/` (`baseline.py`, `naive.py`, `registry.py`, and the component modules),
`src/value/` (`replacement.py`, `vorp.py`, `tiers.py`, `positions.py`, `board.py`)
**How it works:** `docs/modeling.md` (formulas and extension points)

## Context

PLANNING.md sections 3 and 4 ask for a points-league value engine on top of a projection model that
treats minutes, per-minute production and availability separately, starting simple and explainable.
The model has to be replaceable by the later injury / roster / contract layers without touching the
backtest, and it must be provably free of look-ahead. Real NBA data was still being ingested while
this was built, so everything was developed and tested on `make_synthetic_tables()`.

## Decisions

1. **Composition, not one big regression.** A projection is minutes per game x per-minute rates,
   times an availability distribution. Fantasy points come last, from the league config through
   `fantasy_points_frame(..., prefix="proj_")`; scoring never appears inside the stat models.
   Consequence: a scoring change alters `proj_fppg` and nothing else (tested), so one fitted model
   serves any league.
2. **One shrinkage machine for everything.** Every rate is
   `prior + n/(n+kappa) * (own recency-weighted, age-adjusted estimate - prior)`. `kappa` (and the
   recency decay) are *fitted* per statistic by minimising next-season squared error inside the
   history, not hard-coded. Noisy stats (STL, BLK, 3P%) end up with a large `kappa` automatically.
3. **Age curves come from the data.** Delta method on consecutive seasons, weighted by harmonic-mean
   exposure, low-degree polynomial, flat outside the observed age range. No age constants.
4. **Role-appropriate priors.** The prior for a rate is a weighted regression on position group
   (G/F/C) and minutes per game, so a 12-minute bench guard is not shrunk toward a 34-minute
   forward.
5. **Availability is a distribution.** Fractional logistic regression for the mean fraction of the
   schedule played, Beta shape for spread, hard cap at the schedule length. Outputs `proj_gp`,
   `proj_gp_sd`, `proj_gp_p10`, `proj_gp_p90`.
6. **Floor / median / ceiling are calibrated, not assumed normal.** The multipliers on the per-player
   game-level FP spread are quantiles of `(game FP - projection) / projected sd` over every game in
   the history, so about 10% of games land below `fppg_p10` (checked out of sample).
7. **Rookies: slot prior, or an explicit replacement prior.** Learned from the history's own
   rookie seasons by draft pick; if fewer than 20 such rookies exist the model falls back to a
   documented replacement-level prior. All rookies are flagged `is_rookie` and `confidence="low"`.
8. **Who is projected.** Players with a game in either of the last two completed seasons, plus
   players whose `draft_year` equals the target season and who have no history. Everyone else,
   including undrafted debutants and long-absent players, is excluded rather than guessed.
   `players.from_year` / `to_year` are never read (they would leak the future).
9. **Naive benchmark is deliberately dumb.** `NaiveLastSeason` returns last season's line and games
   played as-is and has no row for players without a last season.
10. **Registry.** `get_projector(name)` / `available_projectors()` / `register_projector(...)` in
    `src/models/registry.py` is the stable entry point used by the backtest CLI and the board.
11. **Replacement level from the config.** `R = teams x (starter slots + w x bench slots)`, replacement
    = the first player outside the top R. `w` is derived from projected availability
    (`min(1, starters x (1 - availability) / bench)`), overridable. `teams` comes from
    `config/league.yaml` (13, confirmed live against the real ESPN league on 2026-09-23; was
    provisional at 10 when this ADR was written) so nothing in the code assumes a fixed value;
    tests run both 10 and 13.
12. **Positional scarcity is measured, then applied only if material.** A greedy league-wide slot fill
    gives the best unrostered player per ESPN position; the spread relative to a typical starter's
    VORP is the scarcity index. `positional="auto"` uses positional replacement only above a 10%
    threshold and always reports the verdict. Dataset positions map to ESPN eligibility through a
    documented table (`src/value/positions.py`); unknown positions are UTIL-only.
13. **Tiers are gap-based.** Cliff = gap much larger than the local median gap, subject to minimum
    tier size and a maximum tier count; players at/below replacement form the last tier.
14. **Board.** `build_board(...)` and `python -m src.value.board --season S --model M --out f.csv`;
    ADP gap = `adp - rank` (positive = market lets a player fall relative to the model).

## Alternatives considered

* **Gradient boosting / one model per stat on many features.** Rejected for the baseline: PLANNING.md
  asks for explainable first, and with ~10 seasons of real data the shrinkage-plus-age-curve
  structure is hard to beat while being interpretable. Boosting is the natural tool for the feature
  layers once there are features worth learning from; the ablation harness will tell.
* **Fixed textbook shrinkage constants (Marcel weights 5/4/3, regress 1200 PA).** Simple, but the
  right regression strength differs by statistic and by league; fitting it costs almost nothing.
* **Method-of-moments (Beta-binomial) for kappa.** Estimates the between-player talent variance
  directly but ignores year-to-year drift and role change, which the next-season-error objective
  includes. Both agree in the simulated recovery test; the predictive objective was kept.
* **Modelling fantasy points directly.** One noisy target, scoring baked in, no reuse when the
  scoring changes, no way to explain a projection. Rejected.
* **Normal floor/ceiling (`mean +- 1.28 sd`).** Miscalibrated (about 15% of games under the "10th
  percentile" in the first version) because it ignores projection error and skew. Replaced by the
  calibrated quantile shape.
* **Rank-based tier buckets (top 12, 13-24, ...).** Ignore where the actual value cliffs are.
* **A fixed bench weight.** Replaced by a weight derived from availability so it responds to the
  league's health assumptions, with an override.

## Known weaknesses (honest list)

* **No role context.** Minutes are the largest error source (mean absolute error about 4.3 mpg on
  synthetic data) and depend on teammates arriving or leaving. The roster-context layer is the
  main upside; nothing here fakes it.
* **`proj_gp` is conditional on the player being active.** A retirement, a season-ending injury
  before the opener, or a suspension is invisible to training rows. Injury layer territory.
* **Age curves are survivorship-biased** (only players who stay in the league form pairs) and the
  delta method mixes in regression to the mean; both flatter late-career players.
* **Rookies use draft slot only** (no college or international stats; Summer League and preseason evidence is stacked on top as a separate, gated layer, ADR 0012). Rookie
  inclusion depends on `players.draft_year` being populated for the incoming class; in the synthetic
  data the players table is built from logs, so rookie inclusion there is survivor-biased.
* **Independent components.** Rates are shrunk separately; correlations (usage vs efficiency, minutes
  vs availability beyond the features used) are not modelled jointly.
* **Slight in-sample fitting** of shrinkage constants and the quantile calibration on the same history
  the projection uses (lag features only, so no look-ahead, but no held-out fold either).
* **Positional scarcity uses coarse position strings and a greedy fill**; ESPN's real eligibility
  (games played per position) is richer. The verdict is a diagnostic, not a solved assignment.
* **Preseason changes after the data freeze** (trades, signings, injuries) are not reflected until a
  preseason update step exists.
* **All quantitative claims so far are on synthetic data** (see `docs/modeling.md`). The synthetic
  league has random injuries and ignores real-world structure; real results may differ and will be
  reported by the backtest, including failures.

## Consequences

* Feature layers extend `BaselineProjector` through named seams (design matrix of the prior
  regression, availability feature builder, `History.extras`) and register as separate projectors so
  the ablation table can compare them to `baseline`.
* The synthetic-data benchmark (baseline vs naive) is a regression guard for the modelling code, not
  evidence of real-world skill.
* `docs/modeling.md` is the reference for formulas; changing a decision here means updating both.
