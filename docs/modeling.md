# Modeling guide: projections and the fantasy value engine

Decisions and alternatives live in `docs/adr/0003-projection-model.md`; this page is the reference
for how the code works. Everything below is implemented and tested unless it is explicitly
listed under "Extension points" (which are described, not faked).

## 1. Pipeline

```
History (seasons < target)  ->  panel  ->  fitted components  ->  projections table
                                                                        |
config/league.yaml scoring  ------------------------------------------>  proj_fppg
                                                                        |
config/league.yaml roster + teams -> replacement level -> VORP -> tiers -> board.csv
```

Entry points: `src.models.registry.get_projector("baseline" | "naive_last_season")`,
`build_board(...)` and `python -m src.value.board --season 2026-27 --model baseline --out board.csv`
(`--teams N` overrides the configured team count (13, confirmed live 2026-09-23), `--adp` adds the
ADP columns and accepts either a plain `player_id, adp` CSV or the raw ingested ADP table
(`adp.parquet` from `src.ingest.espn_adp`, mapped to `player_id` automatically via
`player_id_map`), `--synthetic` runs the whole thing on the synthetic league for a demo).

## 2. The panel

`src/models/panel.py` reduces `game_logs` to one row per (player, season): games played `gp`,
totals of every counting stat, minutes per game, game-level FP mean/std, the fraction of the
schedule played `f = gp / L` (L = median team games that season), the age on Oct 1 (from
`players.birthdate`, else the last `player_season_bio` age rolled forward) and a position group
G/F/C/U. Rows are sorted canonically first, so results are bit-identical however the input rows
were ordered. Lags are looked up by (player, season - k); a missing season contributes zero
exposure.

## 3. Rates (minutes, 8 per-minute rates, 3 percentages, volatility ratio)

For a target player and lags k = 1..K (K = 4) with numerator `x_k` and exposure `e_k`:

```
shift_k = C(age_now) - C(age_now - k)                      C = antiderivative of the age curve
est     = sum_k d^(k-1) (x_k + e_k shift_k) / sum_k d^(k-1) e_k
n_eff   = sum_k d^(k-1) e_k
rate    = prior + n_eff / (n_eff + kappa) * (est - prior)
```

| Quantity | numerator / exposure | prior |
|---|---|---|
| minutes per game | minutes / games | league mean |
| FGA, FTA, 3PA, REB, AST, STL, BLK, TOV per minute | stat / minutes | regression on position group and mpg |
| FG%, FT%, 3P% | makes / attempts | same regression, weighted by attempts |
| volatility ratio | games x sd/sd_prior / games | 1 |

* **Recency decay `d` and shrinkage `kappa`** are chosen per quantity by grid search on `d` in
  {0.3, 0.5, 0.7, 0.9} and a fine search on `kappa`, minimising exposure-weighted next-season
  squared error over every (player, season) in the history that has an earlier season.
  On one synthetic league this gave, e.g., kappa around 40 minutes for FGA/min (very persistent) and
  above 4000 minutes for STL/min (mostly noise); the ordering STL, BLK, 3P% > FGA, FG% is a test.
* **Age curves** (`src/models/age.py`): for all pairs of consecutive seasons of a player,
  `delta = rate(t+1) - rate(t)`, regressed on age at t with a degree-2 polynomial, weights
  `e_t e_{t+1} / (e_t + e_{t+1})`. Held flat beyond the 2.5th/97.5th weighted age percentile.
  Fewer than 40 pairs means a flat (zero) curve. On the synthetic league (whose generator grows,
  plateaus and declines skill at known ages) the recovered fantasy-points-per-minute curve
  correlates > 0.9 with the truth and gets the decline-to-growth ratio right (test).
* **Prior regression** (`rates.design`): intercept, guard, center, minutes/10 (forward baseline),
  fitted by weighted least squares with a tiny ridge. Rookie and replacement priors reuse it.
* **Minutes cap** 44 mpg; rates clipped to plausible ranges; `fg3a <= fga`, `fg3m <= fgm`.

Per-game stats are `mpg x rate`. `proj_pts = 2 fgm + fg3m + ftm` (the box-score identity),
so PTS is not modelled separately.

## 4. Availability (`src/models/availability.py`)

`f = GP / L` has mean `mu` from a fractional logistic regression (each row enters twice, label 1
weighted `f` and label 0 weighted `1 - f`) on: recency-weighted `f` over 4 seasons, last season's
`f`, the worst recent `f`, an "absent last season" flag, age, age squared and projected mpg.
Spread: `f ~ Beta(mu nu, (1 - mu) nu)` with `Var(f) = phi mu (1 - mu)`, `phi` fitted from the
residuals (clipped to [0.01, 0.6]). Outputs:

```
proj_gp      = mu * L                      L = max of the last 3 completed season lengths (cap; override in config)
proj_gp_sd   = L sqrt(phi mu (1 - mu))
proj_gp_p10, proj_gp_p90 = L x Beta quantiles
```

`proj_gp <= L` always. `mu` is conditional on the player being active in the target season.
With too few training rows the model falls back to half own recent rate, half league mean.

## 5. Volatility, floor / median / ceiling (`src/models/volatility.py`)

* `sd_prior(fppg) = a + b fppg` from a weighted regression of season game-level FP sd on FPPG.
* A player's ratio `z = sd / sd_prior` is recency-weighted and shrunk toward 1 (same machine).
  Projected game sd: `proj_fppg_sd = z_hat x sd_prior(proj_fppg)`.
* `fppg_p10/p50/p90 = proj_fppg + q x proj_fppg_sd`, where `q` are the 10/50/90% quantiles of
  `(game FP - projection) / projected sd` over all history games projected from their own lags.
  Out of sample on synthetic data about 10 to 12% of games fall below the floor and 10 to 12%
  above the ceiling (test asserts 6 to 14%); before this calibration a normal-shaped band left about
  15% below the floor.

For season-total uncertainty combine `proj_gp_sd` and `proj_fppg_sd`: the availability part alone is
`proj_fppg x proj_gp_sd`; game noise adds roughly `sqrt(proj_gp) x proj_fppg_sd` and is much smaller.
The columns are provided, the combination is left to the consumer.

## 6. Rookies (`src/models/rookies.py`)

> **Availability cap (2026-09-24, ADR 0012 D7).** The games-played fraction is capped at the mean fraction of the history's own
> top-10 picks. The log-linear fit had projected ~78 games for picks 1-3 against ~59 actually played walk-forward; the cap cuts
> that bias from +18.2 to +5.1 games and leaves every other slot unchanged.

Genuine rookie seasons in the history (season start == `draft_year`) are regressed on
`log(pick)`, guard, center and age: minutes, each rate, each percentage and the games fraction
(weighted ridge). The regression is the prior for the target draft class (`draft_year` == target
season start, no games in the history). With fewer than `min_rookie_rows` = 20 history rookies
every rookie instead gets the explicit replacement-level prior: 20th-percentile minutes and the
veteran rate prior at that role. Rookies carry `is_rookie=True`, `confidence="low"`,
`n_hist_seasons=0`. Not projected: undrafted players and drafted players without history who are
not in the target draft class.

## 7. Output columns

The `projections` contract columns, plus (extra columns are allowed by the contract):
`proj_fppg_sd`, `proj_gp_sd`, `proj_gp_p10`, `proj_gp_p90`, `age`, `is_rookie`,
`n_hist_seasons`, `confidence` (`low` / `medium` / `high` from the weight own data gets in the FGA
rate). `proj_total_fp = proj_fppg x proj_gp` exactly.

## 8. Value engine

* **Replacement level** (`src/value/replacement.py`): `R = teams x (starter slots + w bench)`,
  replacement = value of the first player outside the top R (interpolated). Starter slots and
  bench come from `league.roster` in `config/league.yaml`; `teams` is 13 (confirmed live against
  the real ESPN league on 2026-09-23) unless overridden. With the current config: R = 130 + 39 w
  (13 teams; the earlier provisional 10-team config gave 100 + 30 w).
  `w = min(1, starters (1 - availability) / bench)` with availability = mean projected games over
  the top `teams x starters` players / schedule length.
* **VORP** (`src/value/vorp.py`): `vorp = proj_total_fp - repl_total`,
  `vorp_per_game = proj_fppg - repl_fppg` (the same replacement player as `repl_total`, i.e. the player at replacement rank R by total FP, expressed per game; interpolated between neighbours like the total). Negative values are
  kept (below replacement).
* **Positional scarcity**: greedy league-wide slot fill (specific position with the most open
  slots, then G/F, then UTIL, then weighted bench); the best unrostered eligible player per ESPN
  position is that position's replacement level. `scarcity_index = (max - min of the five levels)
  / (mean starter total - global replacement)`. `>= 0.10` is "material". `positional="auto"`
  applies positional replacement (lowest level among a player's eligible positions) only when
  material; the verdict is in `board.attrs["positional"]` and printed by the CLI. On the synthetic
  league (positions uninformative about value) the index is about 0.002 to 0.01, i.e. weak, as the
  flexible slots would suggest; **this must be re-checked on real data**, where centers may differ.
* **Position mapping** (`src/value/positions.py`): `G` -> PG,SG; `F` -> SF,PF; `C` -> C;
  `G-F` -> SG,SF; `F-C` -> PF,C; specific tokens (`PG`...) literal; unknown -> UTIL only.
* **Tiers** (`src/value/tiers.py`): cliff where `gap >= 2.5 x` rolling median gap (window +-15
  ranks) and at least 1% of the value range, tiers of at least 3 players, at most `max_tiers = 12` above-replacement tiers;
  players at or below replacement share one extra last tier (so at most 13 in all). Tier is non-decreasing with rank.
* **Board columns**: `rank, player_id, name, position, proj_fppg, proj_gp, proj_total_fp, vorp,
  vorp_per_game, fppg_p10, fppg_p50, fppg_p90, tier` (+ `age, is_rookie, confidence`), and with an
  ADP frame `adp, adp_gap = adp - rank` (positive: model likes him more than the market).

## 9. Synthetic-data benchmark (regression guard, not evidence of real skill)

Walk-forward on 3 seeds x 3 seasons of a 20-team synthetic league (`tests/models/test_model_walkforward.py`),
players covered by both models who played in the target season:

| model | Spearman (season FP) | top-50 hit rate | top-100 hit rate | MAE FPPG |
|---|---|---|---|---|
| naive_last_season | 0.814 | 0.638 | 0.793 | 4.91 |
| baseline | 0.849 | 0.713 | 0.802 | 4.70 |

The baseline wins Spearman in 9 of 9 cells. Real-data numbers will come from the backtest track.

## 10. Feature layers

### 10a. Injury layer (implemented; ADR 0006)

`BaselineInjuryProjector` (`src/models/injury_baseline.py`, registered as `"baseline_injury"`) is
`BaselineProjector` with one change: its `AvailabilityModel` gets four extra feature columns,
built purely from `History.game_logs`/`History.team_games` by `src/features/injury.py` -- no new
raw data source (games missed is already `team_games` minus `game_logs`, per ADR 0001 rule 5).

* `build_injury_panel(history)`: one row per (player, season) with `missed_frac` (games missed /
  schedule length), `longest_streak_frac` (longest run of consecutive missed *team* games /
  schedule length) and `n_streaks` (count of distinct absence runs). Streaks are computed by
  walking the player's primary team's (the team he played the most games for that season) full
  game-by-game schedule in date order and marking each game present/absent -- this is what
  distinguishes one 20-game injury from twenty 1-game absences, which a season total cannot.
* `InjuryFeatures.build(pids, target_s, age)` turns the panel into four columns fed into
  `AvailabilityModel` (see below): recency-weighted longest-streak fraction, recency-weighted
  streak frequency, a binary "missed >= 15% of the schedule in one streak last season" flag, and
  an age-adjusted absence residual (a player's own recency-weighted absence rate minus a
  population-level `missed_frac ~ age + age^2` curve fit once per history).
* **Wiring**: `AvailabilityModel.fit`/`predict_mean` (`src/models/availability.py`) now accept an
  optional `extra` matrix, appended to the existing 7-column feature matrix (`f_bar, f_last,
  f_min, absent, age, age^2, mpg`) before scaling; passing `extra=None` (what plain
  `BaselineProjector` does) is behaviourally identical to before this change, which the full
  existing test suite plus new leakage/determinism tests confirm. `BaselineProjector` exposes a
  single hook, `_build_injury_features(history, sub)` (default: returns `None`), that
  `BaselineInjuryProjector` overrides; `FittedBaseline._core` passes the extra matrix through when
  present. No other component (minutes, rates, volatility, rookies) changes.
* **Leakage**: the streak panel is rebuilt from `History` on every fit, which is already sliced to
  seasons before the target, so this is leakage-safe by the same structural argument as the rest
  of the panel (`src/models/panel.py`). Checked directly: `tests/models/test_model_injury.py` and
  `tests/models/test_model_injury_real_data.py` run `assert_projector_ignores_future` against it.
* **Known approximation**: streaks are computed against a player's single *primary* team for the
  season. A player traded mid-season shows misattributed absences around the trade date (the other
  team's games count as "missed"). There is no point-in-time roster-tenure table in the current
  contract to do better with; documented in ADR 0006, not fixed here.
* **Ablation result**: see ADR 0006 and the README Results table for the honest number from the
  real walk-forward backtest -- read it there rather than assuming this layer helps just because
  it exists.
* **Return from a long absence** (ADR 0021, analysis only, `python -m src.backtest.return_report`): neither this layer nor the plain
  availability model can say "out for a long block, then healthy since". Tested on ten walk-forward seasons, `baseline` is not
  systematically low for such players in absolute terms (n = 60, +1.0 GP [-4.1, +5.8], +62 total FP [-80, +196]); it is only low relative to
  comparable spread-absence players, whom it over-projects. No model term was added there; ADR 0022 then built one as a feature layer (10g) and it showed no lift.

### 10b. Roster-context layer (implemented; ADR 0010) -- no lift, not recommended

`BaselineRosterProjector` (`src/models/roster_baseline.py`, registered as `"baseline_roster"`) is
`BaselineProjector` with one change: `FittedBaseline._core`'s minutes estimate gets an additive
adjustment fitted from three lagged team-context features, built purely from
`History.game_logs`/`History.team_games` by `src/features/roster.py` -- no new raw data source.

* `build_roster_panel(history)`: one row per (player, season) with `pace` (the player's primary
  team's estimated possessions per game, `POSS ~= FGA - OREB + TOV + 0.44*FTA`), `role_share`
  (player mpg / (team mpg-per-game / 5)) and `pos_crowding` (count of teammates at the same
  position group logging >= 15 mpg for the same team that season).
* `RosterFeatures.fit(...)` fits one weighted-least-squares regression, once per history, of the
  residual between a player's actual historical mpg and the plain (context-free) minutes estimate,
  against recency-weighted lagged `pace`/`role_share`/`pos_crowding`. `RosterFeatures.build(...)`
  turns that into a per-row additive mpg shift, clipped to +-4.0 minutes, falling back to zero
  (no-op) when a row has fewer than 60 usable historical pairs or no lagged context at all.
* **Wiring**: exactly the injury layer's hook pattern (10a) -- one optional `roster_features`
  field on `FittedBaseline`, one `_build_roster_features` hook on `BaselineProjector` (default:
  returns `None`, so plain `baseline` is unaffected), overridden by `BaselineRosterProjector`.
* **Leakage**: same structural argument as `build_injury_panel` -- pure function of an already
  `History.until`-sliced `History`, and `.build()` only ever looks at lag seasons before the
  target. Checked directly: `tests/features/test_roster.py`, `tests/models/test_model_roster.py`
  (`assert_projector_ignores_future`).
* **Ablation result: no lift, and a small real cost on MAE.** Real walk-forward ablation
  (2016-17..2025-26): Spearman total FP and top-50 hit rate both flat (no significant change), MAE
  on total FP *worse* by 2.42 points (95% CI excludes zero, verdict `hurts`). Diagnostics on the
  fitted model rule out an implementation bug (small, sane, never-clipped adjustments). The honest
  explanation: this layer can only see *lagged* team context, never a genuine forward-looking
  arrival/departure signal, because the data contract has no point-in-time preseason roster
  source -- see ADR 0010's Context section for the full argument and README's Results for the
  numbers. **`baseline_roster` is registered and tested but not recommended for use** -- kept as a
  complete, honestly reported negative result, not as a suggested model.

### 10c. Roster-transactions layer (implemented; ADR 0011) -- hurts rank order, not recommended

`BaselineTransactionsProjector` (`src/models/transactions_baseline.py`, registered as
`"baseline_transactions"`) closes 10b's stated gap with a real, dated source: Wikipedia team-season
"Transactions" sections, ingested by `src/ingest/wiki_transactions.py` into `team_transactions.parquet`
(an ad-hoc table outside `History`/`src.contracts.TABLES`, following the `adp` precedent -- every
transaction carries an exact date, letting the feature filter to "before this season's October 1
cutoff" directly, which is tighter than `History.until`'s season-granularity slicing).

* `src/features/transactions.py`'s `TransactionContext` resolves each player-season's team from a
  dated `"in"` transaction (falling back to the previous season's primary team) and computes
  `changed_team` (0/1) and `team_departures_lost` (sum of departing teammates' own prior-season
  `role_share`, ADR 0010's formula reused). Unlike 10b, this is **not lagged** -- it uses each
  season's own transactions directly, the same way a rookie's `draft_year == target` is used
  directly, because the underlying data is genuinely dated before the season, not a proxy from an
  earlier one.
* **Wiring**: an independent, parallel hook to 10b's (`transactions_features` field,
  `_build_transactions_features` hook) -- both can coexist on `FittedBaseline` without interfering,
  though only `BaselineTransactionsProjector` ever sets the new one.
* **Ablation result: hurts, and the mechanism is understood.** Real walk-forward ablation
  (2016-17..2025-26): Spearman total FP gets *worse* (-0.0017, 95% CI excludes zero, verdict `hurts`,
  only 1/10 seasons), top-50 hit rate flat, MAE on total FP worse by 3.22 points (`hurts`) -- a worse
  headline result than 10b's, not a fix. Diagnosed directly against 3,933 real training rows: the
  raw signal is real and correctly signed (team-changers get ~0.75 fewer minutes than their
  historical rate predicts, on average -- a believable "new team, unproven role" effect), but one
  linear coefficient applies that average penalty uniformly, which is wrong for exactly the cases
  that matter most for rank order (a star joining a rebuilding team as its new lead option needs the
  *opposite* adjustment). See ADR 0011 for the full diagnostic write-up. **`baseline_transactions` is
  registered and tested but not recommended for use.**

### 10e. Contract layer (rookie-scale clock, built; not recommended)

`BaselineContractProjector` (`src/models/contract_baseline.py`, registry name `baseline_contract`, ADR 0013) is a
stacked projector in the style of 10b's offseason layer, not a `design()` extension: `src/features/contract.py`
derives a nominal rookie-scale clock from `draft_year`, `draft_round`, `draft_number` and the target season alone
(scale year 1 to 4, option year, contract year = extension-eligible year, post-scale year, second-round early years;
each rule is a labelled CBA assumption, see the module docstring and the ADR). A ridge regression on the base model's
walk-forward FPPG residuals, gated by leave-one-season-out cross-validation, gives a per-player multiplier on the counting
stats (clipped to [0.8, 1.25]); with too little history or no cross-validated gain the projection is exactly the baseline
(`contract_enabled = False`). Real result: no meaningful change (Spearman +0.0004, top-50 0.000, MAE -0.04 n.s.); the
contract-year coefficient is null or negative and the positive ones are young-player effects. **Registered and tested, not recommended.**

### 10f. Contract-terms layer (veteran contract years, built; not recommended)

`BaselineContractTermsProjector` (`src/models/contract_terms_baseline.py`, registry name `baseline_contract_terms`, ADR 0019) reuses 10e's gated ridge on walk-forward residuals, with features from
`src/features/contract_terms.py`: the as-of-opening-night state of each player's latest known contract from `player_contracts` (dated Wikipedia events; `python -m src.ingest.wiki_contracts`), i.e.
known final year, years left, new deal, extension, two-way, minimum, lapsed, plus the rookie-scale clock where no deal is known. Unknown players get all-zero indicators (never "not in a contract year").
Real result: Spearman -0.0003, top-50 -0.004 (flat), MAE of total FP -1.20 (0.3% smaller error), no contract-year premium; **registered and tested, not recommended.** An unvalidated `contract_flag` is shown on the board.

### 10g. Return-health layer (implemented; ADR 0022) -- no lift, not recommended

`BaselineReturnProjector` (`src/models/return_baseline.py`, registry name `baseline_return`; `baseline_injury_return` stacks it on 10a's columns, and
`baseline_return_offseason_debut` puts the board's stack on this base) adds seven columns to the availability model through the same `extra`
hook as 10a, built from `History` alone by `src/features/return_health.py`. A player's lead block (team games missed before his first appearance,
as a share of his primary team's schedule), the shrunk health of the tail after it (`(games played + 10 x 0.75) / (tail games + 10)`) and ADR
0021's "long lead block, healthy tail" flag (block >= 25%, tail >= 15 games, raw health >= 75%) are recency-weighted over the same lags and
decay as the availability model, with last-season-only versions; debut seasons and multi-team seasons count as no lead block, and missing values
take fixed conventions so the matrix is never NaN. `availability.py` and `baseline.py` are unchanged, so `baseline` is bit-identical.
Real result (ten walk-forward seasons, pre-registered rule): lift in MAE of total FP -0.47 [-1.31, +0.29], in MAE of games -0.037
[-0.085, +0.011] (negative = worse), Spearman -0.0012, wins 2 and 3 of 10; it raises the ADR 0021 cohort's projected games by about 8.8 (Tatum 50.1 -> 57.7 GP), which
over-corrects a group the baseline already projected about right. **Registered and tested, not recommended.**

### 10d. Not implemented

* **Contract layer, salary** (hypothesis only): no source has point-in-time salary data (ADR 0005 D4); veteran contract *years* were tested from Wikipedia events (10f, no reliable lift).
  Amounts are stated for only 877 events, too few for a salary feature.
* **Rookie layer** (college / international stats) extends `rookies.rookie_design`.
* **An interaction-aware version of the transactions layer** (10c) -- conditioning the `changed_team`
  adjustment on the destination team's positional depth or record, to distinguish a downgrade move
  from an opportunity move -- is the natural next idea 10c's result points to, deliberately not built
  under the pressure of that result; see ADR 0011's Consequences.
* **Registering a variant**: subclass or parameterise `BaselineProjector(name="baseline_injury",
  ...)` and call `register_projector("baseline_injury", factory)`; the backtest and the board pick
  it up by name.

## 11. Testing map

`tests/models/test_model_*.py` and `tests/value/test_value_b_*.py`: contract validity at every
history length, determinism and input-order invariance, leakage (perturbing everything at or after
the target season, including `players.from_year/to_year`, leaves projections bit-identical, and a
History containing the future is rejected), shrinkage limits and recovery of a known kappa,
recency and age behaviour, age-curve recovery, availability monotonicity and bounds, floor/ceiling
ordering and calibration, rookie and injured-player and retired-player cases, scoring-config
change, replacement level for 10 and 13 teams and for a different slot config, positional scarcity
detection, tier properties, board completeness and ordering, the CLI, and the walk-forward
benchmark. `tests/models/test_model_real_data.py` runs the models on real data when it exists.

## 9. The offseason layer (`baseline_offseason`, ADR 0012)

A stacked residual model on top of any base projector: walk-forward baseline residuals regressed on Summer League and preseason
evidence, applied as one per-player multiplier on every counting stat (so box-score identities and league scoring hold exactly),
and switched off by a cross-validation gate when there is no demonstrated gain. Registered as `baseline_offseason` (both events),
`baseline_summer_league`, `baseline_preseason` (one event each, for the ablation) and `baseline_offseason_rich` /
`baseline_summer_league_rich` (a richer component set that did not help). Method, leakage argument, results and the finding that the
preseason carries the signal while Summer League alone does not: ADR 0012 and [`docs/offseason.md`](offseason.md).
