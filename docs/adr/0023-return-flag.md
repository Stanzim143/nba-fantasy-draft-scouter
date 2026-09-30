# ADR 0023: Advisory return-from-absence flag on the board (no projection change)

**Status:** accepted, 2026-09-26
**Code:** `src/value/return_flag.py`, wired in `src/value/board.py` (CLI, and so the daily refresh) and `src/app/loader.py`; app in `src/app/state.py`, `src/app/draft_board.py`. Tests: `tests/value/test_return_flag.py`, `tests/app/test_return_flag_app.py`.

## Context

ADR 0021 asked whether the baseline over-discounts a player like Jayson Tatum: one long absence block at the start of last season, then healthy.
It found no significant absolute under-projection for that cohort (n = 60: +1.0 GP, 95% CI -4.1 to +5.8; +62 total FP, CI -80 to +196), a relative gap
only against matched spread-absence players (whom the model over-projects), and a hint of about +8 GP in the >= 60%-block cell (n = 21, one of 18
cells, paired test inconclusive). Its recommendation: do not change the projection; if the board is ever given a return flag it should be
**advisory**, "sized by judgement, not derived", and at most about +5 GP. ADR 0022 then built the same information as a model feature; it over-corrects
(Tatum 50.1 -> 57.7 GP) and is registered, not recommended. What is left is to show the drafter the fact (he missed most of last year and then played
almost every game) beside the projection, the way ADR 0016 shows injury status, preseason absence and roster moves.

The metric that matters for the draft is expected **total** fantasy points (`proj_fppg x proj_gp`, and VORP built on it), not FPPG: a high-FPPG player
who does not play is worthless. Everything here is expressed in games and total FP for that reason, and the app now says so under the board.

## Decisions

* **D1. A flag, never a projection.** `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp`, `vorp_per_game`, `fppg_p*`, `tier`, `rank`, `risk_level`,
  `risk_gp_haircut` and `risk_gp` are not written by this module (enforced by tests, see below). Same stance as ADR 0016 and 0019.
* **D2. The cohort is ADR 0021's primary cohort, computed by its own functions.** `return_profile` calls `return_report.season_profiles`,
  `_veteran_flags` and `in_cohort(PRIMARY)` on last season only (`target - 1`) from the `History` that stops before the target season. Nothing is
  re-implemented: veteran (an earlier NBA season), one team, >= 15 mpg, lead block >= 25% of the team's games, tail >= 15 team games, played >= 75% of
  the tail. A test asserts the flagged set equals `prior_features` + `in_cohort` (synthetic and real data).
* **D3. The advisory upside is a fixed rule, fixed before it was applied to any player:**
  `return_gp_upside_adv = 5` games if the lead block is >= 60% of the team's schedule, else `0`; capped so `proj_gp + upside <= season length`.
  `return_fp_upside_adv = return_gp_upside_adv x proj_fppg`, i.e. the same games in total fantasy points. Why 5 and 60%: ADR 0021's recommendation
  ("about +5, roughly the midpoint of the >= 60% cell's +8.0 and the cohort's +1.0", CI shown) and one of only two cells where the raw bias's CI excluded
  zero (the other, the `mid`-type block >= 40% cell, +5.7 [+0.2, +10.6], n = 13, is a different cell type: a player who was playing, missed one block, and returned; it is not used here). Below 60% the study has nothing to say, so 0 (the player is still flagged and his block and tail are shown). It is a judgement, not an
  estimate: no fit, no interpolation by block length, no tail-health scaling.
* **D4. It is not a projection, and the columns say so.** The advisory numbers are never added to `proj_gp` / `proj_total_fp`, do not enter VORP, and
  do not move a rank. Every row carries `return_validation = advisory_judgement_not_projection` (the ADR 0019 `contract_validation` idea, for CSV
  readers with no console). The readable text names it "a judgement, not in the projection".
* **D5. `risk_flags` gets the plain-English sentence; `risk_level` does not change.** The existing UI already shows `risk_flags`, so the sentence is
  appended there when the overlay exists (ADR 0016 window: August to December of the season year): "returned from long absence (missed the first 62
  of 82 team games), healthy since: played 16 of the last 20 team games; advisory +5 GP (a judgement, not in the projection)". `risk_level`,
  `risk_gp_haircut` and `risk_gp` are untouched: returning healthy is not a risk to discount, and a "watch" level would put these players in the
  risk table for the wrong reason. Outside the overlay window `risk_flags` does not exist and only the `return_*` columns are added.
* **D6. Missing inputs degrade to a note.** No game logs, an unusable history or a board without `proj_gp` / `proj_fppg` yields no flags (or unsized
  advisory games) and a note, in the CLI's printed notes and `board.attrs["risk_notes"]` in the app; a raised `FileNotFoundError` / `KeyError` /
  `ValueError` / `OSError` is caught by the CLI and the loader exactly as for the risk and contract overlays.
* **D7. Reports.** The daily refresh calls `src.value.board.main`, so `reports/daily/draft_board_*.csv` carry the columns. `latest.md` and the nightly
  reports show only the top of each board and never showed the risk overlay, so they are unchanged (matching existing behaviour; no new report
  section).
* **D8. UI reinforcement of the total-FP priority.** The board is already ranked by VORP on total FP and the tables already put `proj_fppg`,
  `proj_gp`, `proj_total_fp` and `vorp` side by side in that order (rank, name, position, tier, then those four), so no column moved and no ranking
  logic changed. What was missing was the statement: `RANK_CAPTION` ("Rank is by projected TOTAL fantasy points above replacement (VORP): FPPG x games.
  A high FPPG player who misses games ranks lower.") is now under the Best available and Full board tables. The default order is the board order
  (VORP rank; `best_available` preserves it); the column headers still sort on click.

## Columns (added after the risk overlay, before the contract flags; blank / NaN / 0 for everyone outside the cohort)

| column | meaning |
|---|---|
| `return_flag` | `returned-healthy` or blank |
| `return_block_pct` | lead block as a percent of his team's games, 0 to 100 |
| `return_tail` | games played / team games since his first appearance, e.g. `16/20` |
| `return_gp_upside_adv` | advisory games (0 or 5), see D3 |
| `return_fp_upside_adv` | advisory games x `proj_fppg`, in total fantasy points |
| `return_validation` | constant `advisory_judgement_not_projection` |

## Wiring and UI

* **Read the CSV by column name, not position.** The new columns are inserted after the risk overlay and before the contract columns, so the positions of the contract columns moved.

* `python -m src.value.board` (real data) and the app's `load_board` call `compute_return_flag` / `attach_return_flag` after the risk overlay and
  before the contract flags, with the board's season length for the cap. Synthetic boards get no flag (like the other overlays).
* App: the board tables (`display_columns`) gain `return_flag` and `return_tail` when the board has them; the Best available and Full board tabs get a
  **Returned-healthy only** checkbox (`filter_board(..., returned_only=True)`); the **Debutants & risk** tab gets a "Returned from a long absence,
  healthy since" table (`return_flag_table`: block, tail, `proj_gp`, `proj_fppg`, `proj_total_fp`, the advisory GP and FP, VORP, ADP) with a caption that
  repeats D3/D4 and the study's null absolute result.

## Result on the real board (2026-09-26, `baseline_offseason_debut`, ADP attached; the list is computed from data, not hardcoded)

Nine players are flagged, the same nine ADR 0021 named. The advisory GP rule gives +5 to five of them.

| board rank | player | block % | tail | proj GP | proj FPPG | proj total FP | advisory GP | advisory FP |
|---|---|---|---|---|---|---|---|---|
| 53 | Jayson Tatum | 75.6 (62/82) | 16/20 | 50.1 | 41.0 | 2053 | +5 | +205 |
| 159 | Scoot Henderson | 62.2 | 30/31 | 51.7 | 27.4 | 1413 | +5 | +137 |
| 251 | De'Anthony Melton | 26.8 | 49/60 | 46.7 | 22.8 | 1065 | 0 | 0 |
| 330 | Grant Williams | 46.3 | 36/44 | 45.0 | 17.6 | 793 | 0 | 0 |
| 332 | Max Strus | 81.7 | 12/15 | 39.6 | 19.9 | 788 | +5 | +100 |
| 349 | Killian Hayes | 70.7 | 23/24 | 37.7 | 20.0 | 752 | +5 | +100 |
| 350 | Micah Potter | 36.6 | 47/52 | 42.5 | 17.6 | 751 | 0 | 0 |
| 358 | Josh Green | 29.3 | 58/58 | 55.2 | 13.2 | 731 | 0 | 0 |
| 464 | Cameron Payne | 65.9 | 22/28 | 37.6 | 14.0 | 528 | +5 | +70 |

Projections unchanged: the board rebuilt on this branch and the same command on unmodified code (main at 058b8d8) have identical values in every
pre-existing column for all 790 players except `risk_flags` (exactly the nine flagged rows gained the return sentence; the other 781 are identical); `rank`,
`proj_total_fp` and the other numeric columns also equal the existing (gitignored) `reports/daily/draft_board_baseline_offseason_debut.csv` of the primary checkout.

## Tests

* Cohort: a Tatum-like player is flagged; a spread-absence player, a debutant (`from_year` = last season), a mid-season-traded player, a player with
  no prior season, an interior-block player, a 20% lead block and a poor tail (53% of 15 games) are not; the 25% / 60% boundaries; agreement with
  `prior_features` + `in_cohort`.
* Advisory rule: 5 GP from 60% (inclusive) up, 0 below, schedule cap, FP = GP x `proj_fppg`.
* Projections unchanged: byte-identical CSV of every projection / rank / vorp / tier / `risk_level` / `risk_gp*` column with and without the flag on a
  hand-built board, on a real `BaselineProjector` + `build_board` pipeline, and through `board.main` with the overlay forced to fail.
* No look-ahead: identical output after `scrambled_future` on the target season and after adding a later season to the tables.
* Degradation: empty history, a board without `proj_*` or `risk_*`, a player the history never saw, a raising overlay (CLI note, loader note).
* App: display columns, the `returned_only` filter, `return_flag_table`, the rank caption, the loader with a failing / working overlay, and an
  `AppTest` smoke run of the whole app on synthetic data. A real-data-gated test recomputes the cohort from the store and checks Tatum's 62/82 and 16/20.

## Limitations

* **The rule is a judgement.** ADR 0021 found no significant absolute under-projection (+1.0 GP, CI -4.1 to +5.8); +5 GP is inside that CI and
  the >= 60% cell is one of 18. Read the advisory FP as "what a modest belief in a bounce would be worth", not as an estimate; a drafter who
  distrusts it loses nothing, because nothing else moved.
* **Non-injury absences are in the cohort** (signed mid-season, waived, G League), the cohort's 12 highest-minute rows were over- rather than
  under-projected (ADR 0021), and it excludes anyone under 15 mpg last season, multi-team players and players with a block that is not at the start
  of the season (the `mid` type), so "not flagged" does not mean "healthy".
* A traded-in player keeps the flag (his block was measured on last year's team); a rookie or a player with no NBA season before last year is never flagged.
* The 15 mpg and single-team filters use last season only; the profile is one season deep.

## How to read it on draft day

`return_flag` / `return_tail` beside the projection say "he missed the first N games of last season and has played almost every game since". Use it
as a tiebreaker or a reason to look at the player, not as a correction: his `proj_total_fp` and rank already are the model's answer, and the study says
that answer is about right on average. The advisory +5 GP / FP shows the size of the most the data could be argued to support; a flagged player with
advisory 0 (block under 60%) has no bounce argument at all. Check the tail: 12/15 is a small sample; 58/58 is a full year of health. Then weigh the
usual things the model cannot see (the injury type, minutes plan, age).
