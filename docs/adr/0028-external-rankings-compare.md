# ADR 0028: External rankings comparison (Yahoo, FantasyPros)

**Status:** accepted, 2026-09-28
**Code:** `src/ingest/external_rankings.py`, `src/ingest/yahoo_rankings.py`,
`src/ingest/fantasypros_rankings.py`, `src/value/rankings_compare.py`, `src/app/pages/external_rankings.py`.
**Tests:** `tests/ingest/test_external_rankings.py`, `tests/ingest/test_yahoo_rankings.py`,
`tests/ingest/test_fantasypros_rankings.py`, `tests/value/test_rankings_compare.py`,
`tests/app/test_external_rankings_glossary.py`.

## Context

The user wants to sanity-check this project's own draft board rank against two well-known external
rankings: Yahoo's Fantasy Basketball draft-analysis export and FantasyPros' consensus draft rankings.
Both were supplied as one-off manual exports the user downloaded from each site's own UI, not a live
API:

* `Yahoo_Fantasy_Basketball_Draft_Analysis_2026-09-28.xlsx` -- "Players" sheet, 686 rows: `Yahoo
  Display Order, Player, Team, Eligible Positions, Status, Rank, % Drafted, Preseason ADP, All Drafts
  ADP`. A "Source & Notes" sheet documents the capture (686 unique names, no duplicates, Yahoo's own
  status markers P/Q/O/NA/OFS preserved verbatim). Player names use accented forms ("Nikola Jokić",
  "Luka Dončić").
* `FantasyPros_2026_Draft_ALL_Rankings.csv` -- 308 rows: `RK, "PLAYER NAME", TEAM, BEST, WORST, AVG.,
  STD.DEV, "ECR VS. ADP"`. `TEAM` is always empty; `PLAYER NAME` instead jams name + team + eligible
  positions + an optional trailing status tag into one string, e.g. `"Anthony Davis (WAS - PF,C) OUT"`
  or `"Cam Whitmore (DEN - SF,PF) TWO-WAY"`.

This is **not** the ADP-ingest pattern of ADR 0007 (`src/ingest/espn_adp.py`): there is no live HTTP
endpoint to poll, no rate limit to respect, and no caching HTTP client to build. It is a snapshot the
user re-exports and re-drops by hand whenever they want a refresh. What *does* carry over from ADR
0007 is the matching discipline: reuse `src/ingest/id_map.py`'s normalized-name/alias/fuzzy matching
onto the canonical `player_id`, and never silently guess a match -- report unmatched/ambiguous names
instead of dropping them.

## Decisions

### D1. Snapshot files live under `NBA_DATA_DIR/external_rankings/`, never in the repo or hardcoded to `Downloads`

Per CLAUDE.md's data-directory convention, the two files the user downloaded to `Downloads` are
personal, manually-refreshed exports, not project data. They were copied once, as part of this task,
into the shared data directory:

    <NBA_DATA_DIR>/external_rankings/yahoo_2026-09-28.xlsx      (dated snapshot, kept)
    <NBA_DATA_DIR>/external_rankings/yahoo_latest.xlsx          (convenience copy, what the CLI/app default to)
    <NBA_DATA_DIR>/external_rankings/fantasypros_2026-09-28.csv
    <NBA_DATA_DIR>/external_rankings/fantasypros_latest.csv

Neither file is committed (`NBA_DATA_DIR` already lives outside the repo entirely, per CLAUDE.md's
Layout notes; nothing new needed adding to `.gitignore`). Both parser CLIs and the app page take a
configurable `--path` / sidebar text field, defaulting to the `_latest` name under
`external_rankings/` -- never a hardcoded `Downloads` path. **Refreshing means**: export a new file
from each site, drop it in `external_rankings/` (optionally dated), overwrite (or repoint to) the
`_latest` copy, and reload. There is no scheduled job and no API to poll for this ADR, unlike ADR
0007's ESPN/FantasyPros ADP pull or ADR 0018's nightly in-season job.

### D2. One shared "resolved" contract both parsers produce (`src/ingest/external_rankings.py`)

Rather than have the comparison/UI layer know two different source shapes, both parsers call a
shared `resolve_players()` that turns their own source-specific parse step into one common shape
(`RESOLVED_COLUMNS`):

    source, source_id, source_name_raw, name_parsed, team, positions, status_tag, ext_rank,
    adp, ecr_vs_adp, player_id, match_method, confidence, matched

`adp` and `ecr_vs_adp` are deliberately **two different columns**, not one generic "adp": Yahoo
exports a real average-draft-position value (an overall pick number); FantasyPros' CSV exports no raw
ADP at all, only "ECR VS. ADP", a signed *delta* between FantasyPros' consensus rank and whatever ADP
it was compared against. Conflating the two under one name would silently imply they measure the same
thing. Yahoo rows always have `ecr_vs_adp` null; FantasyPros rows always have `adp` null.

`resolve_players()` never drops a row: every input row survives into the output, matched or not.
`match_method` is one of `exact | normalized | fuzzy | ambiguous | unmatched` -- `id_map.match_players`
itself only ever returns the matched subset and reports ambiguous/unmatched separately;
`resolve_players()` folds both failure kinds back in as their own explicit `match_method` value (not
lumped together) so the UI can tell "genuinely no candidate" from "more than one candidate, could not
prefer either" apart.

### D3. Per-source parsers, built as two independent, parallelizable modules

`src/ingest/yahoo_rankings.py` and `src/ingest/fantasypros_rankings.py` were built independently
(as two separate implementation passes against the shared contract above) since neither depends on
the other's internals -- each just needs to produce the "parsed" shape `resolve_players()` requires.

**Yahoo** (`parse_yahoo_xlsx`): reads the "Players" sheet with `pandas.read_excel(..., header=None)`
and locates the real header row by scanning for the literal string `"Yahoo Display Order"` (rather
than hardcoding "skip 5 rows"), so a future export with a shifted banner still parses. `adp` prefers
`All Drafts ADP`, falling back to `Preseason ADP` when the former is blank (both blank -> null, never
0 -- confirmed against the real file's own "Source & Notes" sheet, which states 494-496 rows have no
draft data yet). Yahoo's own status letters (`P`/`Q`/`O`/`NA`/`OFS`) are passed through verbatim as
`status_tag`, with no invented meaning.

**FantasyPros** (`parse_fantasypros_csv`): a regex pulls name / team / positions / an optional
trailing tag out of the single `PLAYER NAME` field. The tag is captured generically (whatever text
follows the closing parenthesis), not matched against a fixed enum of `OUT`/`DTD`/`TWO-WAY`/`RET` --
so an as-yet-unseen tag like `G-League` still parses instead of raising. A name suffix ("Jr.", "III")
stays part of the captured name (`id_map.normalize_name` strips suffixes downstream; the parser must
not also strip them, to avoid double-handling). `TEAM` is "FA" for free agents, preserved as a real
value (a meaningful "not currently on a roster" signal), never turned into a null. The literal `"-"`
sentinel in `ECR VS. ADP` becomes a real null; a genuinely malformed `PLAYER NAME` (does not match the
pattern at all) raises `FantasyProsParseError` naming the bad value, rather than silently emitting a
row with garbage team/positions -- the same "never guess" discipline as ADR 0007 D4.

Adding `openpyxl` to `pyproject.toml` was necessary and is the **only** shared-file change this task
makes: `pandas.read_excel` needs an engine to read `.xlsx`, and none of the existing dependencies
(pandas/pyarrow/duckdb) bundle one.

### D4. Match-rate result on the real files (2026-09-28 snapshot, matched against the current `players` table)

* Yahoo: 645/686 matched (94.0%: 645 exact, 0 normalized/alias, 0 fuzzy; 0 ambiguous, 41 unmatched).
* FantasyPros: 308/308 matched (100.0%: all exact).

The 41 Yahoo misses are mostly deep-bench/rookie names not yet in the local `players` table (this
project's own NBA-stats ingest has its own coverage boundary, independent of this task) -- an honest
gap, listed in the app's "Unmatched names" tab and the CLI's printed report, never silently dropped
from the raw export nor guessed into a wrong `player_id`.

### D5. Comparison join and the `rank_delta` sign convention (`src/value/rankings_compare.py`)

`build_comparison(our_board, {"yahoo": ..., "fantasypros": ...})` outer-joins on `player_id`, so a
player who is on our board only, an external source only, or all three, is kept (never dropped) with
explicit `on_our_board` / `on_yahoo` / `on_fantasypros` boolean flags making the coverage visible.

**Sign convention** (documented once here, kept consistent across both sources and matching the
existing `adp_gap` convention in `src/value/board.py`):

    rank_delta_<source> = our_rank - <source>_rank

**Positive** means the external source ranks the player *earlier* (a lower, more favorable rank
number) than we do. **Negative** means we rank the player earlier than that source. `max_abs_rank_delta`
takes the larger of the two sources' absolute deltas (ignoring a source that has no rank for that
player, never treating a missing source as zero disagreement), for a single "sort by biggest
disagreement, either direction" column.

**FantasyPros only covering ~308 players is expected, not an anomaly.** Its rows are FantasyPros'
own top slice; most of our board's 700+ projected players (and even much of Yahoo's 686) will
legitimately show a null `fantasypros_rank`. The app's caption and this ADR both say so explicitly,
so a viewer does not read "not in FantasyPros" as a data-quality problem.

Unmatched/ambiguous names (from either source) are reported through a separate `unmatched_report()`,
never silently absorbed into the join (a row with no `player_id` cannot be joined on `player_id` by
definition, so it would otherwise simply vanish without a trace).

### D6. A new Streamlit page, not another `draft_board.py` tab

`src/app/pages/external_rankings.py` follows the same multipage pattern `src/app/pages/inseason.py`
already established (a sibling `pages/` file next to the `draft_board.py` entrypoint, auto-discovered
by Streamlit). A new page rather than a `draft_board.py` tab because this comparison is not
draft-state-specific: it reads no drafted/undrafted state and writes none, its own inputs (two
independently-refreshed file paths) don't fit the draft board's season/model sidebar, and
`draft_board.py` already carries a large, fixed set of tabs (`docs/categories.md` section 19).

The page has three tabs: **All players** (the full comparison table, filterable by name/coverage/min
delta, sortable including by `max_abs_rank_delta`), **Biggest disagreements** (top-N by absolute
delta, either direction or restricted to one source), and **Unmatched names** (the combined
unmatched/ambiguous report across both sources, filterable by source). Every `st.dataframe` call
passes `column_config=` built from `src.app.glossary` (the same `_column_config` convention
`draft_board.py`/`inseason.py` already use), so every shown column carries a real hover tooltip.

### D7. Glossary coverage

A new `"External rankings comparison (ADR 0028)"` group in `src/app/glossary.py` documents every
column the comparison table and the unmatched-names report can show. The unmatched-report's own
per-source `team`/`positions`/`status_tag` fields are named `ext_team`/`ext_positions`/
`ext_status_tag` rather than reusing the board's existing `team` entry (glossary column names are a
single flat namespace across the whole app; reusing `team`'s existing, differently-scoped tooltip
would have been misleading). `tests/app/test_external_rankings_glossary.py` enforces that every
`COMPARISON_COLUMNS`/`UNMATCHED_COLUMNS` value has an entry and that the new glossary group is exactly
that set, mirroring the coverage-enforcing pattern already established in
`tests/app/test_glossary.py`.

## Consequences

* `openpyxl` is now a hard dependency (`pyproject.toml`), justified solely by needing to read the
  Yahoo `.xlsx` export via `pandas.read_excel`.
* Refreshing either source is a fully manual step (re-export, re-drop the file, reload the page) --
  there is deliberately no scheduled job or API client for this ADR, unlike ADR 0007/0018.
* FantasyPros' short list means most players on the app's comparison tables will show a null
  `fantasypros_rank`; this is stated in-app and here, not worked around.
* The Yahoo match rate (94%) is bounded by this project's own `players` table coverage, not by the
  matching approach; growing `id_map.ALIAS_GROUPS` further would only close a small remaining part of
  that gap (see `docs/adr/0007-adp-ingest.md`'s own note that the alias table is grown from real
  unmatched reports, not built once and left alone).
* Out of scope for this task: no attempt to reconcile Yahoo's and FantasyPros' *own* rankings against
  each other (only each against our board), no historical tracking of how the comparison changes
  between snapshots, and no attempt to auto-detect a newer export in `Downloads` (the user places the
  file in `external_rankings/` themselves).
