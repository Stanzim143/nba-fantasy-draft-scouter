# ADR 0010: Roster-context feature layer

**Status:** accepted, 2026-09-23
**Code:** `src/features/roster.py`, `src/models/roster_baseline.py`; small additive changes to
`src/models/baseline.py`, `src/models/registry.py`.
**Evidence:** real walk-forward ablation, `reports/` (gitignored; command below reproduces it),
README's [Results](../project-overview.md#results) table.

## Context

`docs/modeling.md` section 10b documented the roster-context layer only as an extension point:
"teammate arrivals/departures, projected role, pace ... attaches to minutes and usage: the minutes
quantity is a `Spec` in `rates.py`; a context adjustment would multiply/shift `mpg_hat` in
`FittedBaseline._core` before the rates are computed." PLANNING.md's roster row lists: projected
role (No. 1 vs. No. 4 option), usage/minutes redistribution when a star arrives or leaves,
positional depth, team pace, coach/system. This ADR builds the part of that list the data contract
can support honestly, and explains what it cannot.

### The constraint that shapes everything below: no preseason roster snapshot exists

The walk-forward backtest projects season S from a `History` that `History.until` has already
sliced to seasons strictly before S (ADR 0004). The most valuable signals PLANNING.md names --
*"a star arrives or leaves,"* a *preseason* trade, an offseason signing that changes who starts --
are genuinely knowable before season S tips off (free agency and trades are public months ahead of
opening night), but they are knowable only from a point-in-time preseason roster/depth-chart
source, which this project has never ingested (the same gap ADR 0005 cited when it deferred the
contract layer indefinitely: "no source with point-in-time correctness"). `team_games`/`game_logs`
only report what actually happened *during* a season, so any "team X's roster for season S" signal
built from them is either (a) looking at season S itself, which is the leakage the whole contract
exists to prevent, or (b) looking at the *last completed* season before S, which is a real, usable,
leakage-safe signal, but is a *lagged proxy*, not the "who's actually on the roster this year"
signal PLANNING.md describes. **This layer builds (b) only.** Modeling a real arrival/departure
signal is future work gated on ingesting a dated preseason roster source (transactions feed,
depth-chart snapshot); flagged here, not built, for the same reason the contract layer is deferred
in ADR 0005 rather than faked with the wrong-shaped data.

## Decisions

### D1. Three lagged team-context features, derived from tables already ingested

No new raw source, same principle as the injury layer (ADR 0006 D1): `src/features/roster.py`
builds `build_roster_panel(history)`, one row per (player_id, s), purely from
`History.game_logs`/`History.team_games` (both already leakage-safe via `History.until`):

1. **`pace`** -- the player's primary team's estimated possessions per game that season:
   `POSS ~= FGA - OREB + TOV + 0.44*FTA` (the standard estimator; `team_games` itself carries no
   shot-attempt columns, only `pts_for`/`pts_against`, so this is summed from `game_logs` rows
   grouped by `(team_id, game_id)`, then averaged over the team's games that season). Team pace is
   persistent year over year (a coaching system, not a random walk), so last season's pace is a
   reasonable, leakage-safe proxy for next season's.
2. **`role_share`** -- the player's minutes per game divided by his team's total minutes per game
   divided by 5 (`player_mpg / (team_min_per_game / 5)`), i.e. his share of an average rotation
   slot. Distinct from raw `mpg` (already modelled by the `MINUTES` spec): two players with
   identical mpg on a team that plays heavy overtime versus one that never does have different
   role concentration once team pace/total minutes are accounted for.
3. **`pos_crowding`** -- count of *other* players who logged >= `CROWD_MPG_THRESHOLD` (15.0) mpg
   for the same team in the same season at the same position group (G/F/C/U, `src.value.positions`)
   as this player -- a proxy for positional depth/competition for minutes. Computed at
   player-team-season granularity (not player-season), so a traded player's crowding is correctly
   attributed per team he actually played for that season, unlike the necessarily coarser
   player-*season* primary-team attribution `pos_crowding`'s own player uses to pick which team's
   crowding count applies to him (documented approximation, D3).

Deliberately **not** included: an explicit "role rank" ordinal (redundant with `role_share`, which
already orders players continuously) and coach/system identity (no coach table in the contract;
`pace` is the observable proxy for system).

### D2. One fitted linear adjustment to minutes, not a redesign of the rate priors

`modeling.md` sketched two possible attachment points: multiply/shift `mpg_hat` directly, or add
role columns to `rates.design()` so *every* rate's regression prior (all 11 `Spec`s: shots,
rebounds, assists, steals, blocks, turnovers, shooting percentages) can use them. This ADR takes
only the first, narrower option, for the same reason ADR 0006 D3 chose one hook over a parallel
model: `design()` is shared machinery behind every per-minute rate, and touching it to add three
context columns would multiply the blast radius (11 regressions instead of 1) for a benefit that
is speculative until the simpler version is shown to help at all. If the ablation below shows real
lift, extending `design()` is the natural next step and is still exactly what modeling.md describes.

`RosterFeatures.fit(...)` (mirroring `InjuryFeatures.fit`) fits one weighted least-squares
regression, once per `History`, predicting the *residual* between a player's actual mpg in a
historical season and the plain (context-free) `MINUTES` spec's own estimate for that season, as a
function of the three recency-weighted lagged context features above:

```
residual = actual_mpg - mpg_hat(no context)  ~  intercept + pace_bar + role_share_bar + pos_crowding_bar
```

fit by `src.models.shrink.wls`, weighted by the season's actual games played (already-known
exposure of a *historical* row, not the target season -- same pattern `fit_spec`'s own kappa search
already uses for every other quantity). Recency weighting of the three inputs over lag seasons
k=1..n_lags reuses the exact `_recency_weighted` decay helper from `src.features.injury`, with
`cfg.default_decay`, so this needed no new `BaselineConfig` field. Rows with fewer than
`MIN_FIT_ROWS` (60, matching `availability_min_rows`'s order of magnitude) usable pairs, or a
target row with no context history at all (new team, or a player/team combination the panel never
saw), get a **zero adjustment** -- the safe, documented fall-back, identical in spirit to
`InjuryFeatures`'s NaN-safe fallback. The fitted shift is clipped to +-`MPG_SHIFT_CAP` (4.0 minutes,
fixed before the ablation ran, not tuned afterward) so a noisy regression on a short history cannot
swing a bench player into a starter's minutes.

### D3. Wired through one new hook, exactly like the injury layer

`FittedBaseline` gains one optional field, `roster_features` (default `None`). `_core`'s minutes
line becomes:

```python
mpg, _ = predict_rate(self.minutes, frame, None)
if self.roster_features is not None:
    mpg = mpg + self.roster_features.build(frame.pids, frame.target_s)
mpg = np.clip(mpg, 0.0, self.config.mpg_max)
```

`BaselineProjector` gains one hook, `_build_roster_features(history, sub, mpg_est, actual_mpg,
weight)` (default: returns `None`, so plain `baseline` is behaviourally unchanged -- checked by the
full existing test suite passing unchanged). `BaselineRosterProjector`
(`src/models/roster_baseline.py`) overrides it to fit and return a real `RosterFeatures`. Because
the adjusted `mpg` then flows into every rate's `predict_rate(fit, frame, mpg)` call (rates whose
prior depends on `mpg` via `design()`'s `mpg/10` term), a context-adjusted role does gently affect
shot volume and rebounding priors too, without the `design()` matrix itself changing -- the
adjustment happens once, upstream, exactly where modeling.md's "multiply/shift `mpg_hat`" text
pointed.

Rookies (`FittedBaseline._rookies()`) do not go through `_core` and are unaffected, same as the
injury layer -- a rookie has no team-context history to build a lagged signal from anyway.

### D4. Registered as `"baseline_roster"`

`src/models/registry.py` adds `"baseline_roster" -> BaselineRosterProjector`, so
`python -m src.backtest --ablate baseline,baseline_roster ...` and the draft board pick it up by
name with no other wiring, matching ADR 0006 D4's pattern exactly.

### D5. Known approximations, accepted rather than fixed

* **No true arrival/departure signal** -- see Context above; this is the layer's main limitation
  and the reason it is a *lagged* proxy, not the forward-looking signal PLANNING.md originally
  described. Flagged for whichever future ADR adds a point-in-time preseason roster source.
* **Primary-team attribution for `pace`/`role_share`** uses the same single-primary-team-per-season
  rule as the injury layer (ADR 0006's own documented approximation): a player traded mid-season
  gets his post-hoc "team most played for" team's pace/role_share, not a trade-split value.
  `pos_crowding` alone avoids this by computing at player-*team*-season granularity for the
  denominator side (who else played meaningful minutes for team X), but the player's own crowding
  row is still looked up via his primary team. Not fixed, for the same reason ADR 0006 didn't fix
  it: no point-in-time roster-tenure table exists yet.

## Result (real ablation, 2016-17 through 2025-26)

Reproduce: `python -m src.backtest --ablate baseline,baseline_roster --seasons 2016-17:2025-26
--leak-check --out reports/`. Ran on 2026-09-23; report at
`reports/baseline_roster_2016-17_2025-26_b30db614/report.md` (generated reports are gitignored;
numbers below are copied from that run). `--leak-check` passed on the last season.

**Honest read: no lift, and it mildly hurts on error magnitude.** Paired bootstrap over players,
stratified by season (positive = `baseline_roster` better; same methodology as ADR 0006's table):

| metric | lift | 95% CI | p(lift<=0) | seasons won | verdict |
|---|---:|---|---:|---:|---|
| Spearman, total FP | -0.0002 | [-0.0008, +0.0004] | 0.780 | 4/10 | no significant change |
| Top-50 hit rate | -0.0040 | [-0.0140, +0.0060] | 0.756 | 1/10 | no significant change |
| MAE, total FP | -2.4211 | [-2.8971, -1.9736] | 1.000 | 3/10 | **hurts** |

Mean top-12 hit rate also drops (0.542 -> 0.525, not one of the three significance-tested
metrics but worth stating plainly since the README's headline table leads with it). Every other
headline metric (Spearman FPPG, top-100 hit rate, NDCG@100) is essentially unchanged to three
decimal places. This is not a rounding-noise result -- the MAE CI is entirely negative and every
season but three individually shows the same direction -- but the honest description is "no
measurable benefit, and a small, real cost in FP-total accuracy," not "a real but small lift" (the
injury layer's finding, ADR 0006). Rank correlation -- the metric the README's baseline-vs-naive
comparison hangs its headline claim on -- is flat within noise (p=0.78, essentially a coin flip on
direction: 4 of 10 seasons "won").

**This was checked for an implementation bug before being reported as a real finding.** Diagnostics
on a fitted `RosterFeatures` against the full real history (target season 2025-26, n=5,386 panel
rows): the fitted coefficients are sane in sign and magnitude (`beta = [10.27, -0.093, -0.253,
-0.147]` for `[intercept, pace, role_share, pos_crowding]`), the resulting per-player adjustment is
small and centered near zero (mean +0.047 mpg, std 0.38, range [-1.18, +0.92] minutes) and **never
reaches the +-4.0 `MPG_SHIFT_CAP`** -- so the cap is not silently distorting the result, and the
regression is not blowing up on noisy inputs. 7.3% of rows get exactly zero adjustment (no usable
lagged context: a player's first tracked season with a team, or a fresh league in the synthetic
tests). The adjustment is simply too small and too weakly correlated with the true outcome to move
the needle -- consistent with, and predicted by, this ADR's own Context section: `pace`,
`role_share` and `pos_crowding` are *lagged* proxies for a team-context signal whose genuinely
predictive component (an offseason arrival or departure) this layer structurally cannot see.

**What this means in practice.** Per PLANNING.md's own stated policy for feature layers ("kept only
if the ablation shows lift"), `baseline_roster` is **not** a recommended replacement for `baseline`
or `baseline_injury` on the draft board or in any future in-season tool -- there is no evidence it
projects better, and some evidence (MAE) that it is slightly worse. The code, tests and this ADR are
kept, deliberately, as a complete and honestly reported negative result: the hypothesis that a
lagged team-context proxy alone would sharpen minutes projections did not pan out, and the reason
why is documented in Context above, rather than left as an untested extension point that looks
more promising on paper than it is in practice. `"baseline_roster"` remains registered and
reproducible for anyone who wants to verify this finding or build on the same hook with a better
context signal.

## Consequences

* `FittedBaseline` and `BaselineProjector` carry the same small, additive, backward-compatible
  pattern ADR 0006 established; every other consumer of `BaselineProjector` is unaffected
  (`roster_features` defaults to `None`/off).
* **`baseline_roster` is not recommended for use** (draft board, future in-season tools, or as a
  default) -- it is kept registered and tested purely as a reproducible, honestly reported negative
  result. `baseline` (or `baseline_injury`, which does show a real if small lift) remain the
  models to use.
* `docs/modeling.md` section 10 now describes the roster layer as implemented and ablated
  (partially: the lagged pace/role/crowding proxy only, not arrival/departure) with a documented
  no-lift result, with the contract layer the only fully-open extension point left.
  *Note, 2026-09-25:* ADR 0013 built the rookie-scale contract proxy and it showed no lift; that closes the contract row for what is testable with the available data.
* If a future ADR adds a point-in-time preseason roster/transactions source, the true
  arrival/departure signal PLANNING.md described becomes buildable as a second, additive extra
  feature on top of this layer's hook, the same way this layer was added alongside the injury one.
