# Live draft board app

Every column, flag and tab is explained, with formulas and examples, in [`categories.md`](categories.md).

Phase 4 of the roadmap (`PLANNING.md` section 9): a Streamlit app for running an actual live
snake draft against the value board, built on top of the existing model/value pipeline — it does
not add a new projection or ranking method, only a UI and draft-state layer on top of
`src.value.board.build_board`.

## Run it

```bash
streamlit run src/app/draft_board.py
```

The app loads the repo's `.env` at startup (never overriding variables already set), so `ESPN_LEAGUE_ID` and
`ESPN_TEAM_ID` reach the live-sync sidebar without exporting them first.

Requires `streamlit` and `duckdb`, both already in `pyproject.toml`'s dependencies (see the
Quickstart in `README.md` for environment setup). No extra configuration is needed to try it —
turn on **synthetic demo data** in the sidebar for an instant, no-real-data-needed walkthrough, or
point it at real ingested data (`python -m src.ingest.nba_stats`, see the README) for the real
thing.

**Exercised end-to-end against the real 2026-27 season data on 2026-09-23**: loaded the real
687-player board (season `2026-27`, model `baseline`, 13 teams from `config/league.yaml`), marked
and undid a real pick (`#1 Nikola Jokić`), and confirmed the Drafted tab, best-available counts,
and console/server logs were clean throughout — see `PLANNING.md` section 1a. This closes the
"built, not yet run for the actual draft" gap Phase 1 of the roadmap had been carrying.

## What it does

- **Loads a ranked board** for a chosen season and model, reusing `src.value.board.build_board`
  and `src.models.registry` directly (the same code path as `python -m src.value.board`) — the app
  does not reimplement ranking, VORP, tiers, or replacement level. **"Load / rebuild board" always
  gets current data from disk**: the button clears the board's Streamlit cache before rebuilding,
  so it always reflects the latest ingest/`daily_refresh` output (a new trade, roster move, or ADP
  file) even in a browser tab that has been open since before that refresh ran. The cache itself
  still makes ordinary reruns (switching tabs, using a filter) fast; only an explicit click forces
  a rebuild. A caption above the tabs shows when the board currently shown was loaded.
- **Shows the board** — rank, name, position, tier, projected FPPG/games/total FP, VORP,
  floor/median/ceiling (`fppg_p10/p50/p90`) — in a table you can sort by clicking any column
  header, and filter by position (specific: PG/SG/SF/PF/C; flex: G/F; UTIL), tier, and a
  name search.
- **Tracks draft-in-progress state**: mark any player as drafted by yourself or an opponent, in a
  form (search/filter the board first, then pick from the narrowed list — this is meant to be
  fast enough to use pick-by-pick during a live draft). Drafted players drop out of "best
  available" immediately, with an undo per pick on the **Drafted** tab. State lives in
  `st.session_state` for the session, and persists across normal Streamlit reruns (typing in a
  filter, switching tabs) without being lost.
- **Exports/imports draft state as JSON** (sidebar: "Export draft state" / "Import draft state"),
  so a draft can be paused, the browser closed, and resumed later by re-uploading the file. The
  export includes the season/model it was built against; importing against a different board still
  restores every pick by `player_id`, with a warning if the season doesn't match.
- **Best-available-by-position suggestions**: for each of PG/SG/SF/PF/C, the highest-VORP player
  still available who's eligible there, plus a quick readout of how many eligible players you've
  already drafted at each position. This is a scarcity read, not an auto-drafter — it doesn't try
  to solve your whole roster.
- **A full board view** with the same filters, showing every player (drafted or not) with a
  `drafted_by` column, for browsing the whole board rather than just what's left.
- **Best available by need** (ADR 0026): a second, additive ranking that crosses the undrafted
  board against your own drafted roster's positional gaps (`config/league.yaml` roster shape),
  as `need_score` and `need_adj_vorp` alongside the unmodified `vorp`. A documented, unvalidated
  heuristic (a flat VORP-points bonus scaled by how open a position still is), not a projection —
  with an empty roster it falls back to (is rank-identical to) plain best-available-by-VORP.
- **Shows ADP and the model-vs-market gap** (`adp`, `adp_gap`) next to the projection whenever an ADP file is ingested; a missing or
  unusable ADP file only leaves the columns out (the reason is in `board.attrs["adp_note"]`). The board includes the incoming rookie
  class once `python -m src.ingest.nba_incoming` has run.
- **Breakouts tab** (ADR 0012, `docs/offseason.md`): young under-the-radar players ranked by calibrated chance of a *useful*
  breakout, with team, draft pick, board rank, ADP, and each player's own Summer League / preseason lines. Filters (include priced
  players, include older than 23, minimum uplift), a raw-performance sort that ignores the model, and "hide players already
  drafted" re-apply instantly; the projections behind it load on request (one button; about a minute the first time, then cached) so they never slow a pick. It picks the Summer-League-only model until
  preseason games exist and says so. Not available in synthetic demo mode. Once loaded for a given
  (season, teams), a **Refresh watchlist** button replaces the initial load button so it can be
  brought current on request too (it also clears its own cache first) — otherwise, unlike the
  board, nothing would ever call the cached watchlist function again for that key, ttl or not.
- **Says what the rank is**: a caption under the Best available and Full board tables ("Rank is by projected TOTAL fantasy points above replacement (VORP): FPPG x games. A high FPPG player who
  misses games ranks lower."), with `proj_fppg`, `proj_gp`, `proj_total_fp` and `vorp` side by side in that order. The board order is the VORP rank; nothing in the app re-ranks by FPPG.
- **Return-from-absence flag** (ADR 0023): `return_flag` / `return_tail` columns in the board tables, a **Returned-healthy only** checkbox on the Best available and Full board tabs, and a table of the
  flagged players on the **Debutants & risk** tab with block %, games since, and the advisory games and total FP. Advisory judgement, not a projection: nothing about a projection or a rank changes.
- **Short-absence flag** (ADR 0024): an `lm_flag` column, a **Many short absences only** checkbox on the Best available and Full board tabs, and a table on the **Debutants & risk** tab (`lm_iso_n`, `lm_rest_n`,
  advisory -2 GP and total FP). It marks 6+ games missed in 1-2 game absences last season; the data cannot say whether that was rest or a minor injury. Advisory, not a projection: nothing about a projection or a rank changes.
- **External rankings disagreement flag** (ADR 0029, `docs/categories.md` section 24): once a comparison has been loaded on the separate
  **External rankings** page (Yahoo/FantasyPros vs. our board, ADR 0028), the **Debutants & risk** tab shows an "External rankings
  disagreement" table for undrafted players where our board and the wider consensus disagree by 50+ rank spots (either direction) — a fixed,
  data-grounded threshold, not a fitted one. Session-scoped (not part of the daily/nightly CSVs, since the snapshot files are a manual,
  occasional refresh) and purely advisory: nothing about a projection, VORP or rank changes.
- **Handles missing/bad data as a message, not a crash**: an un-ingested season, a malformed
  season string, an unknown model name, or a season with no history to project from all show a
  plain-English `st.error`/`st.info` instead of a traceback. See `src/app/loader.py`
  (`BoardUnavailable`).
- **Hover tooltips and a Glossary tab**: every column on every table has a `help=` tooltip on its
  header (hover it), and a **Glossary** tab lists every column with its full explanation, formula,
  and source ADR/doc, grouped the way `docs/categories.md` section 20 is. The text is a single
  source of truth in `src/app/glossary.py`, derived from `categories.md` (never re-typed ad hoc
  per column); `categories.md` remains the deep reference.
- **Live ESPN sync of opponents' picks** (ADR 0025, `docs/categories.md` section 21): a sidebar
  "Live draft sync (ESPN)" section polls `src.ingest.espn_league`'s draft detail for newly filled
  picks, maps them onto the board's `player_id` via the `player_id_map` table, and marks them
  drafted automatically — attributed to "me"/"opponent" by ESPN team id ($ESPN_TEAM_ID, or the
  team already saved by `python -m src.ops.nightly --set-team`). A status caption shows
  connected/error and the last-synced time; a network/auth failure, a not-yet-ingested id map, or
  another session already syncing the same league all degrade gracefully and never remove the
  manual "Mark drafted" form, which stays the fallback in every case.
- **No auto-drafter.** The suggestions panel is "best remaining by position", not a roster
  optimizer that plans multiple picks ahead or accounts for roster-slot math (starters vs. bench
  vs. UTIL). See `docs/adr/0009-streamlit-app.md` for why.
- **The draft board itself has no trade/waiver tools or schedule awareness.** They live on the separate in-season page
  (`src/app/pages/inseason.py`, see "In-season page" below and `docs/inseason.md`); the pre-draft daily refresh is ADR 0014 and the in-season nightly job is ADR 0018. Those were later roadmap
  phase 5 items, not this phase.
- **No multi-user/shared draft state.** `st.session_state` is per browser session; two people
  looking at the same running app in two tabs see independent draft states unless they explicitly
  export/import between them. A shared live state would need a persistence layer this v1
  deliberately doesn't add (see the ADR).
- **Auction drafts aren't modelled.** The state model (drafted-by-me / drafted-by-opponent, pick
  order) fits the league's actual format (snake, see `config/league.yaml`), not an auction budget.

## Architecture (short version)

- `src/app/db.py` — a thin, read-only DuckDB query layer: opens an in-memory connection and
  registers a `VIEW` per contract table directly over `read_parquet(...)` for whichever parquet
  files exist under `src.contracts.data_dir()`. It duplicates nothing from `src/store.py`
  (validation, atomic writes) — it's SQL on top of already-validated data. The app's current UI
  doesn't call it directly (the board comes from `build_board`/`History`, which read parquet via
  `src/store.py` as everywhere else in the codebase); it exists as the query layer named in
  `PLANNING.md` section 12 and is exercised directly by its own tests
  (`tests/app/test_db.py`) and available for ad-hoc queries or a future reporting view.
- `src/app/loader.py` — `load_board(season, model, ...)`: a plain function wrapping
  `build_board` + the model registry with error handling. No Streamlit import, fully
  unit-testable (`tests/app/test_loader.py`).
- `src/app/state.py` — `DraftState` and the pure functions that operate on it (`draft_player`,
  `undraft_player`, `best_available`, `filter_board`, `best_by_position`,
  `my_position_counts`, JSON export/import). No Streamlit import, fully unit-testable
  (`tests/app/test_state.py` — the deepest test coverage in this track, since this is the logic a
  bug in would actually bite during a live draft).
- `src/app/draft_board.py` — the Streamlit entrypoint. Deliberately thin: every `st.*` call wires
  UI to the plain functions above; no decision logic lives here.

See `docs/adr/0009-streamlit-app.md` for the reasoning behind these choices.

## In-season page

`src/app/pages/inseason.py` (ADR [0015](adr/0015-inseason-tools.md), guide in [`inseason.md`](inseason.md)) is a Streamlit
multipage page: it appears as "inseason" in the sidebar next to the draft board, which is unchanged. Tabs: **Schedule**
(league-wide games per matchup week, a per-team week table with back-to-backs, off nights and heavy/light flags), **Rest of season**
(the blended projection, searchable), **Trade analyzer** (pick what you give and get; freed and filled roster spots included) and
**Waivers** (best adds over the player replaced, streaming candidates for this or next week, minutes and injury flags).

It is offline-safe and does not slow the draft board: nothing is computed until **Load in-season data** is pressed (a few seconds
to about a minute; then cached for ten minutes), and it reads only the local data directory and ESPN caches, never the network. Pick
whose roster to use in the "Whose roster?" expander: an ESPN team id (needs a synced, drafted league), typed names, or a mock draft
team for demos. Logic is in `src/app/inseason_view.py` (no Streamlit import, `tests/app/test_inseason_view.py`, plus a headless
`AppTest` run of the page on the synthetic league).
