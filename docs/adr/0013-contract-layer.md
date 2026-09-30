# ADR 0013: Contract layer (rookie-scale clock)

**Status:** accepted, 2026-09-24. Result: **built, ablated on the real ten-season walk-forward, no meaningful change, not recommended.**
**Code:** `src/features/contract.py`, `src/models/contract_baseline.py`; one registry line in `src/models/registry.py`.
**Tests:** `tests/features/test_contract.py`, `tests/models/test_model_contract.py`, `tests/models/test_model_contract_real_data.py` (marked `real_data`, skips cleanly).
**Evidence:** real walk-forward runs in `reports/` (gitignored; commands below), README's [Results](../project-overview.md#results) table.

## Context

PLANNING.md's contract row ("contract-year flag, years remaining, salary; hypothesis only") was the last open ablation row.
ADR 0005 D4 deferred it: Spotrac, RealGM, HoopsHype and Basketball-Reference are rejected as automated sources and no source gives
point-in-time contract status, so a salary or contract-year signal cannot be measured historically. D4 left one door open: "only the
rookie-scale final-year proxy (from draft year) is built if wanted". This ADR builds exactly that and measures it.

### What real data could ever carry contract signals (searched, nothing new scraped)

| Where | What it holds | Usable? |
|---|---|---|
| `players` (`draft_year`, `draft_round`, `draft_number`) | Static draft facts, sanitised by `History` | **Yes**: the only point-in-time input; source of the clock below |
| `player_season_bio`, `game_logs`, `team_games` | Age, team, box scores | No contract content |
| nba_api `commonplayerinfo` raw cache | Draft year/round/number, `SEASON_EXP`, roster status | No salary or contract fields (same draft facts as `players`) |
| ESPN `kona_player_info` raw cache (`raw/espn/players/*`, 2016..2027) | `draftRanksByRankType.*.auctionValue`, `ownership.auctionValueAverage`, ADP, ownership | An auction value is a **market opinion of production**, not a salary, and the files are snapshots taken when fetched (ADR 0005 R8); using it would import the crowd's forecast, which is the ADP benchmark, not a contract feature. Not used |
| ESPN league payloads (`raw/espn_league`) | Rosters, settings, draft detail | Neither "salary" nor "contract" appears in any file |
| Wikipedia team-season wikitext (`raw/wikipedia/team_season`, ADR 0011) | Dated transaction lines; 330 pages contain about 2,900 mentions of "contract", about 620 of them of the form "N-year contract" or "N-year, $X" | **The only dated contract text that exists in the project.** The transactions parser keeps only direction, kind and date, so it is discarded today. Unstructured prose, partial (signings and extensions of some players, no rookie-scale deals, no guaranteed/option structure), and would need its own ingest, ADR and player matching. Recorded as the one credible future route; **not built here** |

Nothing already ingested carries salary, years remaining or guarantee status. Honoring ADR 0005's rejected-source list, no source was added.

## Decisions

### D1. A rookie-scale clock from draft facts only, every rule labelled an assumption

`contract_clock(players, target_season)` uses only `draft_year`, `draft_round`, `draft_number` and the target season, so there is
no channel for leakage: no game log, outcome or post-draft date enters a feature. A draft year after the target season yields no clock.

CBA basis (2023 CBA, in force through 2029-30). **Every line is an assumption about the target player's real contract, which we never see.**

* **First-round pick (assumption: signed the full scale).** 2 guaranteed seasons + 2 team-option seasons. `scale_year = season - draft_year + 1`.
  `scale_year 3` = first option year; `scale_year 4` = final scale season = **contract year** and, by the same clock, the **extension-eligible**
  season (eligibility opens after year 3 and closes just before year 4 tips off); `scale_year 5` = **post-scale** (restricted free agency
  behind him: he re-signed or took the qualifying offer; length and size unobservable).
* **Second-round pick.** No scale; first deals are non-guaranteed minimum contracts of 1 to 4 years. Assumption: only "early years, cheap and
  insecure" (years 1 to 4 after the draft) is marked, with no year-of-contract claim.
* **Undrafted and veterans (more than 5 seasons out).** No draft year for the undrafted, none of the flags for veterans: no clock, all features zero.
  The "veterans play harder in their contract year" hypothesis, the largest part of what PLANNING.md meant, is **untestable** here.
* **Uncertainty we do not model:** draft-and-stash players (the clock starts at signing), first-rounders waived or traded before year 3, declined options,
  extensions signed a year early, sub-scale first-round deals. A `scale_year` is a nominal clock, not a fact about a contract.

Features (`FEATURE_NAMES`): one-hot `r1_y1..r1_y5`, `r2_y1..r2_y4`, and `r1_y4_top` (picks 1 to 10) / `r1_y4_late` (picks 20 to 30) contract-year interactions.
Display columns on the projection: `contract_years_since_draft`, `contract_scale_year`, `contract_flag` (`contract_year`, `option_year`, `post_scale`, `rookie_scale`).

### D2. A stacked residual adjustment, gated by cross-validation (the ADR 0012 pattern)

`BaselineContractProjector` wraps the plain `BaselineProjector`, so paired lift versus `baseline` isolates this layer exactly.
For each earlier season the base projector is re-run on `History.until(s)` (memoised; it depends only on data before `s`) and its `proj_fppg`
residual against the season's actual FPPG (players with at least 10 games, weight = games capped at 60) is regressed on the clock features by ridge WLS.
Only the contrast against players with no clock is applied (the intercept, the base model's average bias, is not), as a per-player multiplier on every
counting stat clipped to [0.8, 1.25], so box-score and fantasy-point identities hold. The layer **switches itself off** (base projection, `contract_enabled = False`)
unless leave-one-season-out cross-validation beats "no adjustment" by 0.05%, or when fewer than 150 training rows, 2 seasons or 40 clock-marked rows exist.

### D3. Registered as `baseline_contract`; graceful fallback

`--ablate baseline,baseline_injury,baseline_roster,baseline_contract` (documented in `docs/backtest.md` since the harness was written) works with no
runner change: the ablation is generic over registry names. Note it is stacked on plain `baseline`, so its "lift over the previous variant" in that four-row
chain is a lift over the roster row, a meaningless comparison; the verdict below is the paired comparison against `baseline` (`--ablate baseline,baseline_contract`).

## Result (real data, 2016-17 to 2025-26, `--leak-check` passed)

```
python -m src.backtest --ablate baseline,baseline_contract --seasons 2016-17:2025-26 --leak-check --n-boot 1000 --out reports/ --run-id contract_pair
python -m src.backtest --ablate baseline,baseline_injury,baseline_roster,baseline_contract --seasons 2016-17:2025-26 --leak-check --n-boot 1000 --out reports/ --run-id contract_chain
```

| Metric (paired vs `baseline`, all 10 seasons) | baseline | + contract | lift, 95% CI | seasons won |
|---|---:|---:|---|---|
| Spearman, total FP | 0.7836 | 0.7840 | +0.0004 [+0.0001, +0.0007] | 3/10 |
| Top-50 hit rate | 0.656 | 0.656 | +0.0000 [-0.0040, +0.0040] | 0/10 |
| Top-100 hit rate | 0.670 | 0.669 | -0.001 | n/a |
| MAE, total FP | 422.59 | 422.62 | -0.036 [-0.244, +0.173] | 1/10 |

**Verdict: no meaningful change.** The one CI that excludes zero (Spearman) is +0.0004, a fraction of the injury layer's +0.0009 and far below the noise between
seasons; it appears because the layer is active in only three of ten seasons, and it is not corroborated by top-50 or MAE.

**When did the gate open?** Enabled in 2022-23, 2024-25 and 2025-26 (cross-validated gain 0.09% to 0.22%, near the 0.05% threshold); off in the other seven
(negative or near-zero gain; 2016-17 and 2017-18 had too little history). In the three active seasons alone (paired, 3 seasons): Spearman +0.0014 [+0.0003, +0.0024], won 3/3;
top-50 0.0000; MAE total FP -0.12 [-0.84, +0.58]; MAE FPPG -0.010 [-0.031, +0.010]. Still not a usable improvement.

**Why (diagnosis, active seasons, players with 20+ games, actual minus projected FPPG, baseline to layer):**

| Group | n | bias base | bias layer | MAE base | MAE layer |
|---|---:|---:|---:|---:|---:|
| contract year (first-rounder, year 4) | 70 | -1.12 | -1.31 | 4.51 | 4.60 |
| option year (year 3) | 83 | -0.00 | -0.25 | 4.65 | 4.67 |
| post-scale (year 5) | 66 | -0.41 | -0.30 | 4.21 | 4.22 |
| years 1 to 2 on scale | 163 | +0.19 | -0.62 | 5.10 | 5.14 |
| second-round early years | 179 | +0.27 | +0.44 | 4.81 | 4.78 |
| everyone else | 752 | -0.08 | -0.08 | 4.43 | 4.43 |

1. **The contract-year hypothesis is not supported, in the rookie slice it can be tested.** The baseline *over*-projects contract-year first-rounders by 1.1 FPPG (they
   underperform their projection, the opposite of "motivated walk year"); the fitted `r1_y4` coefficient is near zero or negative (-0.19 FPPG, -0.58 for top-10 picks, in the 2026-27 fit).
2. **The positive coefficients that exist are on year-2 first-rounders (about +0.8 FPPG) and rookies (+0.6)**, i.e. the baseline's age curve and rookie prior slightly under-project young
   players. That is a real but different effect, already the target of the age model and the ADR 0012 preseason layer, and it is not a contract effect; the draft-slot clock only proxies it,
   and applied out of sample it overcorrects (years 1 to 2 bias +0.19 becomes -0.62, MAE slightly worse).
3. **Coverage is the ceiling.** About 20% of projected players carry any clock, and the remaining 80% (all veterans) are exactly where contract years matter most and where the
   layer is blind. With no observable salary, guarantee or contract length the only testable slice is the small, heavily managed rookie-scale one.

Sanity: the layer is bit-identical to the baseline whenever the gate is off, plain `baseline` rows are unchanged (the walk-forward baseline is the same run), and the leak check passed.

## Recommendation and draft board

**Do not use `baseline_contract`.** It is registered, tested and documented as a completed negative/neutral result. It does not change the 2026-27 draft board in any way that matters:
with the gate on for 2026-27, top-12 membership is identical, top-50 differs by one player and top-100 by two; the largest single move is a projection change of at most about 0.8 FPPG.
The board keeps using `baseline` (plus the ADR 0012 offseason layer where wanted).

## Consequences

* The contract row of the ablation is now real: baseline to +injury (small lift) to +roster (no lift) to +contract (no change). Nothing in the ablation is "pending".
* **Update 2026-09-25 (ADR 0019):** the Wikipedia route below was built and measured. The veteran contract-year hypothesis is now tested on the covered subset (13 to 40% of veterans have a known deal length) and not supported; see [ADR 0019](0019-contract-terms-layer.md). What follows is the state before that.
* The veteran contract-year and salary hypotheses remain **untested**, not refuted. The only credible route is ADR 0011's Wikipedia wikitext, whose contract phrases are dated and
  point-in-time (signings and extensions, about 620 explicit "N-year" phrases across 330 team-seasons); it would need a structured parser, player matching, a contract-year definition
  covering veterans, and its own ADR. Given this result on the clean rookie slice, the prior for a large payoff is low.
* No shared file changed (`pyproject.toml`, `config/league.yaml`, `src/contracts.py`); `baseline_contract` reads only `History.players`.
