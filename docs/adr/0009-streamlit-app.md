# ADR 0009: Live draft board app

**Status:** accepted, 2026-09-23
**Code:** `src/app/`  **Usage:** `docs/app.md`

## Context

Phase 4 of the roadmap (`PLANNING.md` section 9) is a live draft board: a tool usable at the
actual draft table, not a demo. Section 12 of `PLANNING.md` already decided the two big questions
— Streamlit for the UI, DuckDB as a query layer over the existing parquet store — so this ADR
covers what was built on top of that decision and, more importantly, what was deliberately left
out of v1.

## Decisions

1. **Thin app, reused pipeline.** The app calls `src.value.board.build_board` and
   `src.models.registry.get_projector` directly — the exact code path `python -m src.value.board`
   uses — rather than re-deriving VORP, tiers, or replacement level. A draft board app has no
   business disagreeing with the CLI about what a player is worth; if the ranking logic changes,
   both change together automatically.

2. **Logic lives in plain functions; Streamlit only renders.** `src/app/loader.py`
   (board loading) and `src/app/state.py` (draft state, filtering, suggestions) import nothing
   from `streamlit`. `src/app/draft_board.py` is the only file that calls `st.*`, and its
   functions are thin wiring: read some widget state, call a plain function, render the result.
   This is what makes the state-management logic — where a bug would actually cost someone a pick
   mid-draft — unit-testable at all (`tests/app/test_state.py`, `tests/app/test_loader.py`); testing
   Streamlit's own rendering is out of scope and not attempted.

3. **`DraftState` is immutable.** `draft_player`/`undraft_player` return a new `DraftState` rather
   than mutating one in place. Two reasons: it matches Streamlit's rerun model cleanly
   (`st.session_state.draft_state = new_state`, no aliasing surprises across reruns), and it makes
   the state functions trivial to test — assert the input is untouched and the output is what you
   expect, no setup/teardown needed.

4. **DuckDB is a real, tested layer, not a decoration.** `src/app/db.py` opens an in-memory
   connection and registers a `VIEW` per contract table over `read_parquet(...)` for whatever
   parquet files exist under `data_dir()`. It is read-only and adds nothing `src/store.py` doesn't
   already guarantee (validation, atomic writes) — it exists purely to add SQL on top of already-
   trusted data. **Nothing in the current UI queries through it**: the board comes from
   `History`/`build_board`, which already read parquet the normal way, and pandas DataFrame
   filtering (`src/app/state.py`) is simpler and fast enough for a board of a few hundred to low
   thousands of players — there is no query here big enough to need SQL pushdown. `db.py` is kept
   as the query layer named in section 12, fully tested on its own
   (`tests/app/test_db.py`, real-data-gated + fixture-based), and is the natural place a future
   reporting view or an ad-hoc "who's still available at X" SQL query would plug in without any
   redesign. Building it and then not force-fitting it into the one query pandas already handles
   fine seemed more honest than routing a trivial filter through SQL for the sake of using the
   tool.

5. **In-memory DuckDB, not a persisted file.** The parquet files are the durable store; DuckDB
   here is a read-only lens over them, rebuilt fresh per connection. There is nothing to keep in
   sync across restarts, so a `:memory:` connection avoids a stray `.duckdb` file with no
   real content to lose.

6. **Draft state persistence: session state + manual JSON export/import, not a database.**
   `st.session_state` covers the common case (the app stays open through the draft). Export/import
   covers the real risk (browser crash, laptop sleeps, someone closes the tab) at near-zero
   complexity — a JSON blob of picks, downloaded and re-uploaded. A server-side persistence layer
   (SQLite file, or a DuckDB table actually written to) would remove the "don't lose the tab"
   step, but adds a stateful file to manage for a single-user tool with an unscheduled draft date;
   revisit if the app grows a second concurrent user.

7. **Suggestions: best-by-position, not an auto-drafter.** For each of PG/SG/SF/PF/C, show the
   top remaining players by VORP eligible there. This deliberately does not attempt: roster-slot
   math (how many bench/UTIL spots are left), multi-pick lookahead, or accounting for how positions
   you already have deep well feel like. All of those are real features of a "should I draft this
   player" recommender and none of them are "a simple next-best-by-position panel", which is what
   was asked for. `my_position_counts` gives a lightweight positional-coverage readout alongside
   the suggestions so the information to make that judgment by eye is there, without the app making
   the call.

8. **No ESPN league sync dependency.** ESPN league-roster ingestion is a separate, parallel track
   under `src/ingest`, not built as of this change. The app was built and tested standalone; it
   does not import from or depend on that module. If its output exists by the time this ADR is
   read, wiring "already rostered" players in is additive
   (feed more `player_id`s into `DraftState` at load time) — see `docs/app.md`'s "What it doesn't
   do" section.

## Consequences

- The app is real (not a placeholder wired to nothing): loading a real season's board, marking
  players drafted, undoing a pick, filtering, and exporting/importing state were all exercised
  against real ingested data in a running instance during this change, not just against synthetic
  fixtures in tests.
- The DuckDB layer being unused by the current UI is a conscious trade documented above, not an
  oversight — flagged here so a future reader doesn't "fix" it by rerouting board filtering through
  SQL for no behavioural gain.
- Because `DraftState` is immutable, the picks list is O(n) to rebuild `drafted_ids`/`my_ids` from
  on every access. Fine at draft-board scale (tens of picks); would need a cached index if this
  state model were reused somewhere with thousands of transactions.
- No auto-drafter and no live opponent sync are real, named gaps for an "in-season tool" future
  phase (`PLANNING.md` roadmap phase 5), not silent omissions.
