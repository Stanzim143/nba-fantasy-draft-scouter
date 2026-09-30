# ADR 0007: ADP ingest

**Status:** accepted, 2026-09-23
**Code:** `src/ingest/http_cache.py`, `src/ingest/espn_adp.py`, `src/ingest/id_map.py`.
**Tests:** `tests/ingest/test_http_cache.py`, `tests/ingest/test_id_map.py`, `tests/ingest/test_espn_adp.py`,
`tests/ingest/test_espn_adp_real_data.py` (real-data-gated).
**Supersedes:** implements the plan in ADR 0005 (`docs/adr/0005-data-sources.md`, decisions D1/D3), which verified
the approach but shipped no code. Updates `docs/research/data-sources.md` section 4.8 from "verified feasible" to
"implemented".

## Context

PLANNING.md names the ADP benchmark ("beating naive is realistic; matching ADP is a strong result") as an open
item since the backtest's real run (README Results). ADR 0005 verified that ESPN's public fantasy API returns
real historical ADP for 2015-16 through 2024-25 and the live 2026-27 season, that 2025-26 is wiped (every player
reads exactly 140.0), and that FantasyPros' `adp/overall.php?year=` pages can fill that one gap. It also verified,
as a feasibility test, that normalized-name matching resolves 96-99% of ESPN's ADP-carrying players onto the local
NBA player list. None of this was wired into code; `src/backtest/benchmarks.py`'s `AdpBenchmark`/`load_adp` and the
`player_id_map` contract table (`src/contracts.py`) already existed as the target shape, unimplemented against.

## Decisions

### D1. A new, independent HTTP client, not a subclass of `nba_client.NBAClient`

`src/ingest/http_cache.py` duplicates `NBAClient`'s cache/retry/backoff/offline shape (so the two are recognizable
as siblings) but shares no code and no state with it: ESPN/FantasyPros need no auth, a different (stricter, >= 2s)
rate-limit posture, and a client that can address more than one host and body type (JSON for ESPN, HTML for
FantasyPros) from one process, none of which fits `NBAClient`'s one-`base_url`, `stats.nba.com`-shaped design.
Every response is cached verbatim on disk (`raw_dir("espn")/players/<key>.json`,
`raw_dir("fantasypros")/fantasypros_adp/<key>.html`), so a rerun never re-downloads, `NBA_OFFLINE=1` makes a cache
miss a clear, catchable error instead of a network call, and a corrupt cache file is quarantined (online) or raised
as an error (offline) rather than silently served.

### D2. Wiped-season detection, not silent ingestion of the sentinel

`espn_adp.detect_wiped_season` counts players with a *real* ADP (`0 < adp < 139.9`, ESPN's own "no ADP" sentinel
is 140.0/139.99) and flags the season wiped when fewer than `WIPED_MIN_REAL_ADP` (10) carry one. Every real
season observed (2015-16..2024-25, 2026-27) has 149-414; 2025-26 has exactly 1 stray real value. A wiped season
is excluded from `adp.parquet` entirely rather than passed through with garbage ADP=140.0 rows that would rank
every unranked player identically.

### D3. The 2025-26 gap: FantasyPros fill, same `source="espn"`, keyed on the same ESPN ids

`AdpBenchmark`/`load_adp` (unmodified, per the task) take one `source` string and join `player_id_map` filtered
to it. Rather than invent a second id namespace for FantasyPros (which `load_adp`'s single-`source` CLI usage
would then silently exclude), `espn_adp.fill_gap_season` matches FantasyPros' "ESPN" column (its own cross-check
of ESPN ADP, verified 0.81-1.00 Spearman rho against the real ESPN values in ADR 0005) by name onto the **already
resolved ESPN player ids** from the id map. The 2025-26 rows therefore carry `source="espn"`,
`source_id=<the same ESPN id used in every other season>`, and an informational `adp_source="fantasypros_fill"`
column (`"fantasypros_fill_avg"` if FantasyPros' own ESPN column was blank for that player and the AVG column was
used instead) so the substitution is visible in the file, never silently blended with genuine ESPN values. This
keeps `load_adp`'s default `source="espn"` working unmodified for every season including the filled one.
`--no-gap-fill` skips this and leaves 2025-26 absent entirely.

### D4. `player_id_map` population: normalized-name matching, alias table, year-window tie-break, fuzzy last

`src/ingest/id_map.py` matches once, on the union of every distinct ESPN player id seen across every season
pulled (ESPN ids are stable across seasons per ADR 0005), not once per season:

1. **Exact** on `normalize_name` (accent-stripped, punctuation-stripped, suffix-stripped: `"Dončić"` ==
   `"Doncic"`, `"Jimmy Butler III"` == `"Jimmy Butler"`) against the local `players` table. Ambiguous exact
   matches (two real players, same normalized name) are resolved by a `[from_year-1, to_year+1]` window around
   the ESPN season the name was first seen in, when that uniquely resolves it; otherwise they stay unresolved and
   are reported, never guessed.
2. **Normalized/alias**: a small manual table (`id_map.ALIAS_GROUPS`) of known name-form collisions (legal-name
   changes like Enes Kanter/Enes Freedom, common nickname/full-name pairs like Cam Thomas/Cameron Thomas, Jr./III
   suffix pairs the plain suffix-strip does not cover on its own). Confidence 0.9.
3. **Fuzzy**: `difflib.get_close_matches` at a 0.90 cutoff, only reached when nothing above resolved uniquely, with
   the same year-window disambiguation. Confidence is the actual similarity ratio, so a weak fuzzy hit is visibly
   weaker in `player_id_map.confidence` than a good one.
4. **Unmatched**: logged in the run's JSON report (`processed/espn_adp_report.json`) and printed via the match
   report summary, never dropped silently and never written to `player_id_map` with a guessed `player_id` (the
   contract's `player_id` column is not nullable, so there is nowhere honest to put an unresolved row anyway).

`match_method` in the written table is one of `exact | normalized | fuzzy` (never `manual`, in this run — that
value is reserved for a future hand-curated correction).

### D5. `adp.parquet` is not a `TABLES` contract entry

`src/backtest/benchmarks.py` defines the file's exact required shape (`ADP_REQUIRED = ("season", "source",
"source_id", "adp")`) ad hoc, deliberately outside `src/contracts.py` ("nothing here downloads data"). Per the
task's constraint not to touch `src/contracts.py`'s existing specs, `espn_adp.write_adp` writes the file directly
(atomic temp-file-then-rename, same pattern as `store.write_table`) rather than registering a new `TableSpec`.
Two informational columns ride along beyond the required four: `name` (consumed by `load_adp` if present, else it
falls back to `player_id_map.source_name`) and `adp_source` (`"espn"` or one of the FantasyPros-fill values, D3).
`player_id_map` itself *is* an existing contract table (`src/contracts.py`, already specified before this task),
so it is written through `src.store.write_table`, which validates it.

### D6. Output path

`adp.parquet` -> `<NBA_DATA_DIR>/processed/adp.parquet` (`espn_adp.adp_path()`), alongside the existing contract
tables. `player_id_map.parquet` -> `<NBA_DATA_DIR>/processed/player_id_map.parquet`
(`src.contracts.table_path("player_id_map")`, unchanged). Raw responses cache under
`<NBA_DATA_DIR>/raw/espn/` and `<NBA_DATA_DIR>/raw/fantasypros/`. None of this is committed (`.gitignore` already
excludes `data/` and the shared data dir lives outside the repo entirely).

## CLI

    python -m src.ingest.espn_adp --seasons 2015-16:2026-27 [--offline] [--refresh] [--limit 600] [--no-gap-fill]

Mirrors `nba_stats.py`'s season-range grammar (`A:B`, single season, comma list) and CLI shape (`--offline`,
`--refresh`, `--data-dir`, `--min-interval`, `--max-retries`). `--min-interval` refuses to go below 2.0s (ADR
0005/CLAUDE.md's politeness floor for ESPN, stricter than `nba_stats`'s 0.8s stats.nba.com floor) unless offline.

## Real run result (2026-09-23)

Ingested `2015-16:2026-27` (12 seasons; 2025-26 detected wiped and gap-filled from FantasyPros, 2026-27 pulled
live). `player_id_map`: 1,321/1,591 distinct ESPN player ids matched onto NBA `player_id` (83.0%: 1,306 exact,
3 normalized/alias, 12 fuzzy, 0 left ambiguous, 270 unmatched). `adp.parquet`: 3,050 season-player rows across
12 seasons (153 in 2015-16 up to 360 in 2023-24), of which only 45 (1.5%) failed to map onto `player_id_map`
through `load_adp` — the id-map match rate looks lower (83%) because it is computed over the *full* universe of
distinct players ever seen at any ownership rank, not just the ones that end up in `adp.parquet` with a real ADP;
restricted to that smaller, more fantasy-relevant set the match rate is much higher. 2025-26: 253/260 FantasyPros
rows matched onto the already-resolved ESPN ids (D3); 7 stayed unmatched (name-form mismatches not covered by the
alias table).

The real ADP-vs-naive-vs-baseline comparison (`python -m src.backtest --model baseline --benchmark
naive_last_season --seasons 2016-17:2024-25 --adp-file <adp.parquet> --leak-check --out reports/`, 9 seasons —
2025-26 and 2026-27 excluded per the reasons above) is reported in full in README's
[Results](../project-overview.md#results) section; the short version: ESPN's ADP beats the baseline projector on
top-of-draft precision (top-12 hit rate 0.620 vs 0.556, top-100 hit rate 0.749 vs 0.677, NDCG@100 0.929 vs 0.878),
while the baseline's larger player universe gives it a slightly higher overall Spearman correlation (0.787 vs
0.743) that does not hold up on the paired (same-players-only) comparison (Spearman lift -0.073, `hurts`). MAE
against ADP is not a meaningful comparison (`AdpBenchmark` is rank-only; its score is `-adp`, not a fantasy-point
prediction) and is reported as n/a rather than as a spurious win. Known limitations, honestly:

* **Undrafted / deep-bench players are frequently unmatchable or absent from ESPN's own board entirely.** ESPN's
  `--limit` cuts off the pull at N players sorted by ownership; a player who never had meaningful fantasy relevance
  may simply never appear, which is a coverage gap, not a matching failure. The `--limit 600` default follows
  ADR 0005's finding that raising the limit past the real-ADP range adds only deeper, ADP >= ~130 players.
* **The 2025-26 gap-fill depends on FantasyPros carrying the same player under a resolvable name.** A player who
  debuted or changed name between the two sources' snapshots can still slip through; `fill_gap_season`'s own
  match report (folded into `season_counts["2025-26"]["fantasypros_fill"]` in the run's JSON report) names every
  FantasyPros row that could not be matched.
* **ADP snapshot timing is still undocumented** (ADR 0005 D3/R3): this ingest does not change that; the benchmark
  remains a season-level aggregate, not a guaranteed preseason snapshot, and the README says so.
* **The alias table is necessarily incomplete.** `id_map.ALIAS_GROUPS` covers cases found by researching known
  NBA name changes and common nickname pairs; a real run's `unmatched`/`ambiguous` report is the mechanism for
  growing it, not a one-time exercise.
