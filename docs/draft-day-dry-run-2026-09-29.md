# Draft-day dry run — 2026-09-29

A rehearsal of live draft-day mode (ADR [0009](adr/0009-streamlit-app.md), ADR
[0025](adr/0025-draft-live-sync.md), ADR [0026](adr/0026-need-adjusted-board.md), ADR
[0027](adr/0027-trade-analyzer-ui.md)) ahead of the real draft (2026-10-17). Goal: surface bugs and
rough edges now, against real and synthetic 2025-26-derived data, rather than live during the actual
draft when a bug is expensive (13 teams, snake, 90 seconds/pick, opponents' picks auto-detected via
ESPN live sync).

## What was actually exercised, and how

Two complementary approaches, both against real code paths (not a re-statement of existing unit
tests):

1. **The app itself, actually run.** `streamlit run src/app/draft_board.py` was started for real
   (`.claude/launch.json` + the Browser tool) and driven through the browser: loaded the synthetic
   demo board (195 players), marked a pick, confirmed the Drafted/Best-available counts updated,
   undid the pick via the Drafted tab's "Undo this pick" button and confirmed the board and Drafted
   tab both reverted, opened the Best-available-by-need tab and confirmed its empty-roster fallback
   message, and toggled "Auto-detect picks from ESPN" on with no league id set and confirmed the
   sidebar's graceful "Enter an ESPN league ID..." prompt (no crash, no swallowed state). Server
   logs (`streamlit run ... 2>&1`) were checked for exceptions throughout: none beyond a benign
   `ConnectionResetError` from killing the process at the end of the session. The real 2026-27 board
   (built from this project's real, fully-ingested 2025-26 history) was also loaded directly via
   `src.app.loader.load_board("2026-27", "baseline", teams=13)` outside Streamlit to time it (see
   Performance below) — a full interactive rehearsal of all ~169 picks through the browser was not
   attempted (169 clicks x several seconds of browser-tool round-trip each is not a good use of
   either the session or the user's time for something the state-machine tests below cover exactly
   as thoroughly); the browser pass instead targeted the paths a UI bug is most likely to hide in
   (rerun/DOM-refresh timing, empty-state messages, sidebar conditionals).

2. **The full 169-pick draft, opponent-pick sync, and the other tabs, driven directly through the
   real Python functions** `draft_board.py` itself calls — `src.app.state.draft_player`/
   `undraft_player`/`best_available`/`filter_board`/`best_by_position`/`need_board_view`,
   `src.app.live_sync.build_detected_picks`/`apply_detected_picks`/`sync_once`,
   `src.app.inseason_view.load`/`roster_choice`/`run_trade` (trade analyzer), and
   `src.value.rankings_compare.build_comparison` (external rankings). This is now a permanent,
   re-runnable regression test: **`tests/app/test_draft_day_dry_run.py`** (11 tests, all passing).
   This is the deeper exercise: it is the only way to genuinely simulate 169 sequential picks,
   an ESPN-shaped duplicate-pick race, and an ESPN auth failure without spending the session driving
   169 individual browser clicks.

Both were run against the deterministic **synthetic demo board** (`load_board(..., synthetic=True)`,
14-team fake history, seed 0) for the full-draft simulation — chosen over the real 740-player board
specifically so the rehearsal is exact and reproducible in CI without depending on what happens to be
in `~/dev-data/nba-fantasy-2026` on a given day — plus one real-data smoke load to confirm the real
2025-26-derived board also builds cleanly and to measure real-world timing.

## Draft lifecycle simulated

- Fresh board load (both real 2026-27/13-team data and the synthetic demo).
- Full 13-team x 13-round snake draft, 169 picks, alternating "me" (team 0) and "opponent" (teams
  1-12), each pick taken through `draft_player` — the same function the manual "Mark drafted" form
  and live-sync's `apply_detected_picks` both call. Verified: no duplicate `player_id` across all 169
  picks, `best_available` shrinks by exactly one player per pick and stays rank-ordered, "me" ends
  with exactly 13 picks (one per round, correct for a snake draft).
- A manual override and undo mid-draft: pick #20 undone and re-marked with the opposite
  `drafted_by`, confirming `undraft_player`'s documented contract (pick numbers are a historical
  record and are not renumbered) and that a stale undo (`undraft_player` on a never-drafted id)
  raises rather than silently no-opping.
- Live-sync detecting and applying an opponent's pick (`build_detected_picks` + `apply_detected_picks`
  against an ESPN-shaped payload).
- **The duplicate-pick race** (ADR 0025's documented edge case): a player marked by the manual form
  first, then the same player arriving in a live-sync poll's detected batch — confirmed it is skipped
  as a `duplicate`, never double-counted, and the original "me"/"opponent" attribution is preserved
  (not silently overwritten by the sync's own classification).
- **The unresolved-player-id case**: an ESPN pick whose `espn_player_id` has no `player_id_map` entry
  — confirmed it comes back in `unresolved`, is never guessed at or silently dropped, and leaves the
  draft state untouched.
- **Graceful degradation**: `sync_once` against a session that raises `ESPNAuthError` on every
  request — confirmed it returns `SyncResult(ok=False, error=...)` rather than raising, and that the
  manual `draft_player` path is completely unaffected (drafted a player immediately afterwards in the
  same test). In the running app this is what drives the sidebar's "Live sync: ERROR ... Manual
  'Mark drafted' still works" caption (`_live_sync_status_caption` in `draft_board.py`) — confirmed by
  reading that function, not just the underlying `sync_once` contract, since `draft_board.py` is
  what actually renders it.
- Best-available-by-need mid-draft: drafted three PG-eligible players for "me", confirmed the
  need-adjusted table still excludes every drafted player, is not in fallback mode (`view.fallback is
  False` once a roster exists), and separately confirmed the empty-roster fallback is exactly
  rank-identical to plain best-available-by-VORP (the documented contract in ADR 0026 and the module
  docstring).
- Suggestions ("best remaining by position") and the Full board tab, three rounds into the draft:
  confirmed both stay consistent with the drafted set.
- Trade analyzer (`src.app.inseason_view.run_trade`), exercised against a **mock-draft-team**
  synthetic roster (`resolve_roster(..., mock_team=1)`) — the same path a recent commit
  (`34bf379`, "Fix active_universe() emptying the mock-draft-team demo...") fixed for exactly this
  synthetic/no-real-roster-overlap case. Ran a one-for-one trade with a partner and confirmed both
  sides' verdicts, confidence buckets and deltas come back populated.
- External rankings compare (`src.value.rankings_compare.build_comparison`), exercised with small
  synthetic Yahoo/FantasyPros-shaped frames (matching `RESOLVED_COLUMNS`) built from the drafted
  board's own top 5 players in reversed rank order, confirming the sign convention documented in the
  module (`rank_delta_yahoo = our_rank - yahoo_rank`) holds in practice, not just by inspection.

## Findings

### Nothing found (checked and clean)

- **State machine correctness under a full 169-pick draft.** No duplicate picks, correct per-team
  pick counts, `best_available`/`filter_board` stay consistent throughout. This is the module the
  app docs themselves call out as carrying "the deepest test coverage in this track, since this is
  the logic a bug in would actually bite during a live draft" (`docs/app.md`) — the rehearsal did not
  find anything to add to that.
- **Live-sync dedup/attribution/degradation** (ADR 0025's three documented edge cases: duplicate
  race, unresolved id, sync failure) all behave exactly as documented, both at the pure-function
  level and read through into how `draft_board.py` surfaces them in the sidebar.
- **Best-available-by-need fallback and mid-draft behavior** (ADR 0026) match the documented
  contract exactly.
- **Trade analyzer against a mock-draft-team roster** (the path the recent `active_universe()` fix
  targeted) works cleanly post-fix: non-empty roster, both sides of a partnered trade score.
- **External rankings sign convention** holds under a real `build_comparison` call, not just by
  reading the docstring.
- **UI reactivity**: marking a pick, undoing it, and toggling live-sync all triggered the expected
  `st.rerun()`/conditional-render behavior with no stale state surviving a rerun (see the one
  transient-DOM note below, which is not an app bug).
- **No exceptions in the Streamlit server log** across the whole interactive session (board load,
  mark/undo, tab switches, sidebar toggle).

### Rough edges (documented, not changed — judgment calls for the user)

1. **Resolved 2026-09-30 — `NeedBoardView.table` (the "Best available by need" tab's displayed table) had
   no `player_id` column.** `player_id` was added to `NEED_TABLE_COLUMNS` (`src/value/need.py`, right after
   `rank`), so the table can be joined back to the board without going through `name`; the drafted-player
   exclusion test now checks by `player_id`. Additive only: no projection, rank or need-score value changed.
2. **Informational — real-data board build time.** Loading the real, fully-ingested 2026-27/13-team
   board (`baseline` model, 740 players) took **~27 seconds** end-to-end outside Streamlit
   (`load_board` -> `build_board` + the risk/return/load-flag/contract-flag overlays). This only runs
   on an explicit "Load / rebuild board" click (ADR 0009's caching design — ordinary reruns like a
   tab switch or a filter change are fast, cached), so it does not eat into the 90-second pick clock
   for anything except the very first load and a deliberate mid-draft rebuild. Still worth knowing
   going in: if the user clicks "Load / rebuild board" mid-pick to pick up a last-second roster
   change, budget ~30 seconds for it, not "instant."
3. **Observational, not a bug — Streamlit's DOM briefly shows the previous rerun's content during a
   transition.** Reading the page immediately after a click (before the rerun settles) sometimes
   returned text from both the old and new state (e.g. "No players marked drafted yet." alongside a
   stale "Undo a pick" section from the pick that was just undone). A short wait/re-read always
   showed the correct, single, settled state. This is normal Streamlit rerun behavior, not a defect
   in this app's code — noted here only because it is worth knowing if the user is ever reading the
   screen right at the instant of a click during the real draft (give it a beat).

### Bugs found and fixed

**None found in application code.** Every discrepancy hit while writing the dry-run test was a
test-authoring mistake on the rehearsal's side, not an app bug — caught and fixed before landing:

- The synthetic demo board's dataset `position` strings are compound (`"G-F"`, `"F-C"`, etc.), not
  the specific ESPN slot letters (`"PG"`) — the test's PG-eligible fixture now goes through
  `src.value.positions.eligible_positions`, the same way the app itself determines eligibility,
  instead of a literal string match.
- `ESPNAuthError` requires a `status` argument (`src/ingest/espn_league.py`); the degradation test's
  fake session now passes one.
- The need-board assertions above (`player_id` not in `NeedBoardView.table`) were adjusted to check
  by name once the table's actual (intentional) column set was confirmed against `src/value/need.py`.
  (Superseded 2026-09-30: the table now carries `player_id` and the test checks by `player_id`; see
  rough edge #1.)

## Performance / timing check (90 seconds/pick)

- **Real-data board build** (`load_board`, 740 players, all overlays): ~27s. Only on explicit
  rebuild; cached otherwise (ADR 0009).
- **Synthetic demo board build**: a few seconds, used throughout the interactive browser pass with
  no perceptible lag between clicking "Load / rebuild board" and the board appearing.
- **Mark drafted / undo / tab switch / live-sync toggle**: each completed in well under a second in
  the running app (Streamlit rerun + the underlying pure functions, all O(number of players) pandas
  operations over a few hundred rows) — no interaction observed came close to threatening the
  90-second pick clock.
- **The 169-pick simulated draft** (direct Python, `tests/app/test_draft_day_dry_run.py`) plus every
  other test in the new file runs in about 20-27 seconds total for all 11 tests combined (most of
  that is the one real board-independent synthetic board fixture build, shared across tests via a
  module-scoped fixture) — comfortably fast enough to also serve as a pre-draft CI/smoke check on
  its own, not just documentation of a one-off rehearsal.

## What remains open for the user's attention before the real draft

- Rough edge #2 above: budget ~30 seconds, not "instant," for a mid-draft "Load / rebuild board"
  click against real data — plan around it rather than clicking it reflexively on a close pick.
- This rehearsal did not (and could not, offline) exercise a real ESPN league's live draft-detail
  endpoint end-to-end — `sync_once`/`apply_detected_picks` were exercised against ESPN-shaped
  synthetic payloads and against injected `ESPNLeagueError`s, which covers the app's own logic
  completely, but a real league's actual JSON shape, auth cookies, and rate-limit behavior on
  2026-10-17 are still first exercised for real on draft day itself (as documented in ADR 0025 — this
  was always the accepted scope boundary, not a gap introduced by this rehearsal).
- The full 169-pick sequence was driven through the browser for a small sample (one mark, one undo)
  rather than all 169 picks — the state-machine correctness of the full sequence is covered
  end-to-end by `tests/app/test_draft_day_dry_run.py` instead (see "What was actually exercised"
  above for why).

## Addendum, 2026-09-30: closing rough edges #1 and #2

Both rough edges above were revisited in a follow-up close-out pass, with real measurements rather than re-stating this report's original
estimates.

**Rough edge #1 (`player_id`-less need table) — closed.** `NEED_TABLE_COLUMNS` (`src/value/need.py`) now includes `player_id` (inserted right
after `rank`, matching `src.value.board.BASE_COLUMNS`'s own ordering), so `NeedBoardView.table` can be joined back onto the live board
programmatically instead of only by name. It stays hidden from the rendered "Best available by need" grid via
`column_config={"player_id": None}` in `src/app/draft_board.py` — present in the data, not shown to the user, the same "id present for joins, not
for reading" idea the main board follows by simply never including `player_id` in `display_columns()`'s own output (this table has no separate
"display" subset of its own, so `column_config` is the mechanism instead). `tests/app/test_draft_day_dry_run.py`'s assertion was flipped from
`"player_id" not in view.table.columns` to `"player_id" in view.table.columns` (the exclusion check itself now uses ids, not just names) and a new
test, `test_best_available_by_need_table_joins_back_onto_live_board_by_player_id`, confirms a caller can actually merge the need table back onto
the live board by id and recover the same rows.

**Rough edge #2 (real-data board build time) — investigated, no code change needed; already correctly cached.** `src/app/draft_board.py` already
wraps `load_board` in `@st.cache_data(show_spinner="Building the board...")` (`_cached_load_board`), keyed on exactly the inputs that determine the
board's content (`season, model, teams, synthetic, data_dir, positional`) — a correct cache key, not one that changes on every rerun — and the
"Load / rebuild board" button already calls `_cached_load_board.clear()` immediately before every explicit rebuild specifically so a click never
serves stale data (`tests/app/test_board_cache_refresh.py` already covered this). A spinner with an explicit message was already present, so a
user mid-rebuild is not looking at a blank screen. Confirmed today, directly:

* A fresh `load_board("2026-27", "baseline", teams=13)` call (740 players, all overlays) took **~4.6–7.8s** measured just now on this machine
  across three separate runs — noticeably faster than this report's original ~27s figure from 2026-09-29, most likely OS/disk-cache warmth
  differences between machines/sessions rather than a code change (nothing in `load_board`/`build_board`/the overlays changed between the two
  measurements). Either number is well inside the 90-second pick clock for a deliberate mid-draft rebuild.
* Calling `_cached_load_board` twice in a row with identical arguments: **first call ~4.9s, second call ~0.0015s**, returning an identical
  board — confirming the cache does what it is supposed to for a same-session repeat (e.g. a tab switch or an accidental double-click), not just
  "does not error." A new test, `test_cached_load_board_reuses_result_for_identical_arguments`
  (`tests/app/test_board_cache_refresh.py`), pins this by counting calls to the underlying `load_board` with a monkeypatched stand-in.

No caching change was made because none was needed — the existing design (cache keyed on the real inputs, explicit `.clear()` before every
rebuild, spinner during the wait) already satisfies "never serve a stale board silently" while making repeat loads fast. The only change here is
documentation and test coverage of behavior that was already correct.

## Re-running this rehearsal

```bash
python -m pytest tests/app/test_draft_day_dry_run.py -v
```

No real ingested data or network access required (synthetic board, fixed seed) — safe to run before
the actual draft, or wire into CI alongside the rest of `./dev test`.
