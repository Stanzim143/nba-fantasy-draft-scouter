# ADR 0029: Advisory "external rankings disagreement" flag (no projection change)

**Status:** accepted, 2026-09-29
**Code:** `src/value/rankings_compare.py` (`DISAGREEMENT_THRESHOLD`, `flag_disagreements`, wired into `build_comparison`), `src/app/state.py`
(`RANKINGS_DISAGREEMENT_TABLE_COLUMNS`, `rankings_disagreement_table`), `src/app/draft_board.py` (the "Debutants & risk" tab's new "External
rankings disagreement" section), `src/app/glossary.py`. Tests: `tests/value/test_rankings_compare.py`, `tests/app/test_rankings_disagreement_app.py`,
`tests/app/test_glossary.py`, `tests/app/test_external_rankings_glossary.py`.

## Context

ADR 0028 built a read-only comparison between this project's own board rank and two manually-exported external rankings (Yahoo, FantasyPros),
producing `rank_delta_yahoo`, `rank_delta_fantasypros` and `max_abs_rank_delta` on its own page (`src/app/pages/external_rankings.py`). That
comparison has zero effect on the actual draft board -- a drafter has to separately open the External rankings page, remember what they saw
there, then apply it by hand while drafting. This ADR closes that loop with the same discipline every other advisory feature in this app already
follows (ADR 0016, 0019, 0023, 0024, 0026, 0027): a clearly-labeled, opt-in, display-only signal on top of the board, never a silent change to
`proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp`, `tier` or `rank`.

**Why advisory and not a projection nudge.** Nothing here has been backtested against an outcome -- there is no "external consensus was right"
ground truth to validate against, only two other opinions. A sharp disagreement is a *fact worth looking at*, not evidence either board is wrong;
treating it as a correction would repeat exactly the mistake ADR 0021/0022 warned about (ADR 0022's return-health feature over-corrected the
projection and was left registered-but-not-recommended for that reason). So this ships the same way ADR 0023/0024 did: a fixed, judgement-based
threshold decided before looking at outcomes, a flag and a sentence, nothing added to any projection column.

## Where the flag lives: session-scoped, like the need-adjusted board (ADR 0026), not baked into `board.py`/the CLI (ADR 0023/0024)

Two existing wiring patterns exist in this codebase:

1. **Baked into `src.value.board`'s CLI `main()`** (ADR 0023 return flag, ADR 0024 load-management flag): computed from `History`, which is
   always available for the real data pipeline, so it can run unattended as part of the daily refresh and land in every `reports/daily/*.csv`.
2. **Computed on the fly in the app, from already-loaded session state** (ADR 0026 "Best available by need"): used when the input isn't part of
   the `History`/projection pipeline at all, or isn't guaranteed present at refresh time.

External rankings are pattern 2, not pattern 1: they are one-off manual exports the user re-drops "whenever they want a refresh" (ADR 0028), most
realistically right before the draft (see the user's own offseason-finding note: preseason data drives the useful signal and gets refreshed right
before the 2026-10-17 NZ-morning draft). Requiring `yahoo_latest.xlsx` / `fantasypros_latest.csv` to exist for the unattended nightly/daily jobs
(ADR 0014/0018) to succeed would be a new hard dependency those jobs never had, for a feature that is inherently "whatever's freshest when the
drafter last checked." So `src.value.board.main()` is untouched by this ADR -- no new CLI flag, no new column in `reports/daily/draft_board_*.csv`.

Instead: `src/app/pages/external_rankings.py`'s "Load comparison" button already builds `comparison` and stores it in
`st.session_state.rankings_cmp` (ADR 0028's own code, unchanged here except that `build_comparison` now always includes the three new columns --
see below). `src/app/draft_board.py`'s "Debutants & risk" tab reads that same `st.session_state.rankings_cmp` (Streamlit multipage apps share one
`session_state` across pages within a browser session) and, if it is loaded and non-empty, shows the flagged players crossed against the live
board. If nothing has been loaded yet, the tab says so and explains how to load it -- exactly the same "fallback with an explicit message" idiom
ADR 0026 established for the need board's empty-roster case, and ADR 0023/0024's "no note flags, table doesn't render" idiom for a board without
the column.

## Decisions

* **D1. A flag, never a projection.** `flag_disagreements()` reads `our_rank`, `yahoo_rank`, `fantasypros_rank`, `rank_delta_yahoo`,
  `rank_delta_fantasypros` and `max_abs_rank_delta` and writes only three new columns (`rankings_disagreement`,
  `rankings_disagreement_direction`, `rankings_disagreement_text`). It never touches `our_rank`/`rank_delta_*`/`max_abs_rank_delta` themselves,
  and `rankings_disagreement_table()` never touches `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp` or `rank` on the board it joins onto (same
  "advisory/derived, never overwrites the projection" discipline as ADR 0016/0019/0023/0024/0026).

* **D2. `build_comparison()` calls `flag_disagreements()` with the default threshold automatically**, so the three columns are always present on
  the comparison frame (`COMPARISON_COLUMNS` grew by exactly those three) rather than requiring every caller to remember a second step. A caller
  that wants a different cutoff calls `flag_disagreements(comparison, threshold=...)` again on the already-built frame (it is idempotent: the
  three columns are simply overwritten).

* **D3. The threshold is a fixed, documented judgement, grounded in the real 2026-09-28 snapshot, not fit to any outcome.**
  `DISAGREEMENT_THRESHOLD = 50` rank spots (either direction). How it was chosen: `max_abs_rank_delta` was computed for the real
  `yahoo_latest.xlsx` / `fantasypros_latest.csv` snapshot against the real `baseline` board for 2026-27 (742 players in the comparison, 648 with
  at least one external rank). Two things stood out:

    * **Unrestricted, the distribution is dominated by deep-bench noise.** Across all 648 rows, median `max_abs_rank_delta` is already 61 and the
      75th percentile is 108 -- both external sources' coverage gets thin and noisy past a few hundred players (a board rank of 680 vs. an
      external rank of 569 is not a meaningful disagreement; neither player will be drafted), so a global percentile over the *whole* universe
      would set an unusably high bar for the players a drafter actually cares about.
    * **Restricted to the realistic draft pool (`our_rank <= 150`, roughly this league's 13 teams x 13 roster spots), the distribution is much
      tighter and more informative:** n = 150, mean 46.6, median 36.5, 75th percentile 66.0, 90th percentile 99.3, max 192.0 (the same cutoff at
      `our_rank <= 100` gives median 25.0, 75th percentile 50.2).

    `DISAGREEMENT_THRESHOLD = 50` sits just above the top-100 pool's own 75th percentile (50.2) and around the top-150 pool's 60th-65th
    percentile -- high enough that it does not fire on every-day rank noise near the top of the board, low enough that it still catches a
    meaningful share of real disagreements rather than only the extreme (90th-percentile-plus) tail. **This is a judgement call, sized by
    looking at the real distribution, not a fitted or backtested number** -- there is no outcome to validate "50 rank spots is the right cutoff"
    against, only the shape of one snapshot. A different threshold (e.g. 40 or 75) would also be defensible; 50 was picked as a round number
    inside the range the data supports. `flag_disagreements(comparison, threshold=...)` exposes the cutoff as a parameter specifically so a
    reader who disagrees with this choice can recompute with their own.

  **Open question, honestly flagged:** this was checked against exactly one snapshot pair (2026-09-28). Whether 50 is still the right order of
  magnitude once fresher pre-draft exports land, or once both sources' coverage is closer to symmetric, is untested; re-running the distribution
  check above against a later snapshot before the draft is a reasonable sanity check, not a requirement to change the constant.

* **D4. Direction and text.** `rankings_disagreement_direction` is `"market_favors"` (an external source ranks the player earlier than we do --
  positive `rank_delta`) or `"we_favor"` (we rank him earlier -- negative `rank_delta`), named for whichever of `rank_delta_yahoo` /
  `rank_delta_fantasypros` has the larger magnitude (a tie, or only one source present, goes to Yahoo -- the wider-coverage source, ~686 vs.
  FantasyPros' ~308). `rankings_disagreement_text` is a one-line sentence naming that driving source and both ranks, e.g. *"fantasypros ranks him
  62 spots earlier (#41) than our board (#103); external consensus is higher on him than we are"*. Both are blank when the player is not flagged.

* **D5. Missing/partial inputs degrade to an empty table, not an error.** `rankings_disagreement_table(board, comparison, drafted_ids)` returns
  an empty frame (right columns, zero rows) when `comparison` is `None`, empty, or missing the disagreement columns -- covering "nothing loaded
  this session yet" and any comparison built before this ADR's columns existed. The app tab shows an explicit info message in that case, the same
  idiom as the need board's empty-roster fallback (ADR 0026) and the risk/return/contract overlays' "column absent -> section not shown" idiom
  (ADR 0016/0019/0023/0024).

* **D6. Reports are untouched.** `src.value.board.main()` (and therefore `reports/daily/draft_board_*.csv`, the daily/nightly automation of ADR
  0014/0018) does not call this code at all -- see "Where the flag lives" above. This is purely a session-scoped app feature.

## Columns

**On the comparison frame** (`src.value.rankings_compare.COMPARISON_COLUMNS`, always present -- `build_comparison` calls `flag_disagreements`
internally):

| column | meaning |
|---|---|
| `rankings_disagreement` | `True` when `max_abs_rank_delta >= DISAGREEMENT_THRESHOLD` (50), else `False` |
| `rankings_disagreement_direction` | `market_favors` \| `we_favor` \| `''` |
| `rankings_disagreement_text` | one-line explanation naming the driving source and both ranks, or `''` |

**On the app's disagreement table** (`src.app.state.RANKINGS_DISAGREEMENT_TABLE_COLUMNS`, joined onto the live board, undrafted players only):
`rank, name, position, proj_fppg, proj_gp, proj_total_fp, vorp, yahoo_rank, fantasypros_rank, rank_delta_yahoo, rank_delta_fantasypros,
max_abs_rank_delta, rankings_disagreement_direction, rankings_disagreement_text, adp` -- sorted by `max_abs_rank_delta` descending (biggest
disagreement first), same ordering idiom as `biggest_disagreements()` on the External rankings page.

## Wiring and UI

* `build_comparison()` -> `flag_disagreements()` automatically (D2); the External rankings page's "All players" / "Biggest disagreements" tables
  pick the three new columns up for free through the existing `_column_config` tooltip wiring, no page code change needed.
* `src/app/state.py`: `rankings_disagreement_table(board, comparison, drafted_ids)` -- pure function, joins the flagged subset of `comparison"
  onto the live board by `player_id`, excludes drafted players (`best_available`), sorts by `max_abs_rank_delta` descending.
* `src/app/draft_board.py`: the "Debutants & risk" tab (`_debutants_risk_tab`) gets a new "External rankings disagreement" section after the
  contract-flags table, reading `st.session_state.rankings_cmp`. Loaded and non-empty -> the table plus a caption naming the threshold and
  restating "advisory only, projection/rank unchanged". Not loaded -> an info message pointing at the External rankings page. No new sidebar
  control, no new filter checkbox on Best available / Full board -- the flag is scoped to this one table, following ADR 0026's precedent of a
  self-contained new section rather than threading a new column through `display_columns()`/`filter_board()` (which are tested tightly against
  the columns `build_board` and its CLI overlays can actually produce; this feature's columns are session-only and never appear there).
* `src/app/glossary.py`: three new entries added to the existing "External rankings comparison (ADR 0028)" group (kept in that group rather than
  a new one, since `test_external_rankings_group_covers_exactly_its_own_new_columns` already asserts that group equals exactly
  `COMPARISON_COLUMNS | UNMATCHED_COLUMNS` minus `player_id`/`name`, and these three columns are part of `COMPARISON_COLUMNS`).

## Tests

* `tests/value/test_rankings_compare.py`: `flag_disagreements` flags at-or-above the threshold and not below (boundary case at exactly
  `DISAGREEMENT_THRESHOLD`), direction labels both ways (`market_favors` / `we_favor`), picks the larger-magnitude source for the text when both
  are present, blank/`False` when a player has no external rank at all, and a custom `threshold=` argument; `build_comparison` includes the three
  columns by default.
* `tests/app/test_rankings_disagreement_app.py`: `rankings_disagreement_table` returns an empty (right-shaped) frame for `None`/empty/missing
  comparison, an empty frame when nothing is flagged, correct join + column set + direction for a flagged player, drafted-player exclusion, and
  sort order (biggest disagreement first) with more than one flagged player.
* `tests/app/test_glossary.py` / `tests/app/test_external_rankings_glossary.py`: extended for the three new columns (glossary coverage,
  `RANKINGS_DISAGREEMENT_TABLE_COLUMNS` added to the parametrized display-column-list coverage check, the `st.dataframe` call-site count bumped
  16 -> 17 for the new section in `_debutants_risk_tab`).

## Limitations

* **Unvalidated threshold**, checked against one snapshot pair only (D3). Not backtested against any outcome, because there is no "who was right"
  ground truth for a pre-draft rank disagreement -- read a flagged player as "look closer here", not "our board is wrong" or "the market is
  wrong".
* **Session-scoped, not persisted.** The flag only exists while the comparison is loaded in the current browser session; it never lands in
  `reports/daily/*.csv` or any exported CSV (see "Where the flag lives" above for why).
* **Direction is a magnitude pick, not a consensus vote.** When Yahoo and FantasyPros disagree with each other as much as either disagrees with
  us, only the larger-magnitude source drives the label and text; the other source's number is still visible in the table's own columns, just
  not named in the sentence.
* **Coverage inherits ADR 0028's own asymmetry.** FantasyPros' ~308-player export means a deep-bench player can only ever be flagged (or not) by
  Yahoo; this is expected, not a bug (see ADR 0028's `on_yahoo`/`on_fantasypros` discussion).
* **Independent of the risk/return/load-management/contract overlays and the need-adjusted board.** A flagged player who is also risk-flagged or
  need-boosted is not specially cross-referenced; check the other tabs/sections for that, same limitation ADR 0026 already documents for itself.

## How to read it on draft day

Open **External rankings** first, load the freshest Yahoo/FantasyPros exports (re-drop and reload right before the draft, per the project's own
offseason-finding note that preseason signal should be refreshed close to draft time), then go to the **Debutants & risk** tab on the main board:
the "External rankings disagreement" section lists every undrafted player where our board and the wider consensus disagree by 50+ spots, sorted
biggest-gap-first. `we_favor` players are ones we are notably higher on than the market -- worth a second look at *why* the model likes them (a
skill-position breakout case the market hasn't priced in yet, or a bug). `market_favors` players are ones the market is notably higher on than we
are -- worth checking whether the model is missing something (role change, a new team) or the market is overrating name recognition/ADP momentum.
Either way, `proj_total_fp`, `vorp` and `rank` are exactly what they were on every other tab; this section adds a reason to look, not a
recalculated number.

## Addendum 2026-09-30: threshold re-checked against the real export files

Recomputed `max_abs_rank_delta` from the user's actual exports (`Yahoo_Fantasy_Basketball_Draft_Analysis_2026-09-28.xlsx`,
`FantasyPros_2026_Draft_ALL_Rankings.csv`; byte-identical to the `*_latest` files in `external_rankings/`) against the 2026-27 `baseline` board
(13 teams). D3's figures reproduce exactly (742 in comparison, 648 with an external rank; all rows median 61 / p75 108; `our_rank <= 150`: n 150,
mean 46.6, median 36.5, p75 66.0, p90 99.3; `our_rank <= 100`: median 25.0, p75 50.25). At `DISAGREEMENT_THRESHOLD = 50` the flag fires on 55 of the
top-150 (37%) and 26 of the top-100 (26%) -- a reviewable list, not every player. No change to the constant; the "open question" above
(fresher pre-draft snapshot) still stands.

## Addendum, 2026-09-30: independent re-verification of D3's real-data numbers

The user supplied the real `yahoo_latest.xlsx` / `fantasypros_latest.csv` snapshot files for a follow-up close-out pass. Their md5sums matched
what was already at `NBA_DATA_DIR/external_rankings/` (dated 2026-09-28) byte-for-byte -- this is the same snapshot D3's numbers were computed
against, not a fresher one, so this addendum is an independent re-run of the same analysis, not a re-check against new data.

Rebuilt the real 2026-27/13-team `baseline` board via `src.app.loader.load_board`, re-parsed both files fresh via
`src.ingest.yahoo_rankings.load_yahoo_rankings` / `src.ingest.fantasypros_rankings.load_fantasypros_rankings`, rebuilt the comparison via
`src.value.rankings_compare.build_comparison`, and recomputed the distribution independently (not by reading D3's numbers back out of the code
comments). Result: **every number in D3 reproduced exactly**:

* 742 rows in the comparison, 648 with a computable `max_abs_rank_delta` (see the clarification below); unrestricted median 61.0, 75th percentile
  108.0 -- exact match.
* `our_rank <= 150`: n = 150, mean 46.6, median 36.5, 75th percentile 66.0, 90th percentile 99.3, max 192.0 -- exact match.
* `our_rank <= 100`: median 25.0, 75th percentile 50.2 -- exact match.

**One wording clarification, not a numerical discrepancy.** D3 says "648 with at least one external rank"; re-deriving that count two ways gives
two slightly different numbers: 650 rows have a non-null `yahoo_rank` or `fantasypros_rank`, while 648 rows have a non-null `max_abs_rank_delta`.
The gap is exactly 2 rows that are on Yahoo and/or FantasyPros but *not* on our own board (no `our_rank`), so no `rank_delta` -- and therefore no
`max_abs_rank_delta` -- can be computed for them even though they do carry an external rank. D3's "648" is the `max_abs_rank_delta`-based count
(the one the rest of D3's own distribution stats are computed over), so the number itself is right; "at least one external rank" is the slightly
imprecise phrase -- a more exact description would be "648 rows with both `our_rank` and at least one external rank." Left as this addendum
rather than editing D3's prose, per this ADR file's own "never rewritten after acceptance" rule (see `docs/adr/README.md`).

**New regression coverage**: `tests/value/test_rankings_compare_real_data.py` (opt-in, skipped when the real history tables or the real snapshot
files are not present -- same pattern as `tests/models/test_model_real_data.py`) rebuilds the real board and comparison from scratch and asserts
the flagged fraction of the `our_rank <= 150` pool stays inside a generous `[0.10, 0.75]` band (measured today: 36.7%), that at least 90% of that
pool has a computable `max_abs_rank_delta`, and that the unrestricted median exceeds the top-150 pool's median. None of these pin the exact
2026-09-28 numbers as a golden file -- the point is to catch a *future* snapshot swap that silently breaks the id matching or shifts the
distribution to somewhere implausible, not to freeze today's numbers forever.

**Conclusion: `DISAGREEMENT_THRESHOLD = 50` needs no change.** The real-data numbers behind it are confirmed accurate on independent
re-derivation, and the new regression test gives a tripwire for a future snapshot that breaks matching or shifts the distribution, without
requiring anyone to re-run this manual analysis before every draft.
