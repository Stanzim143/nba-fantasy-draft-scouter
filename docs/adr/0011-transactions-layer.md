# ADR 0011: Roster-transactions feature layer (closing ADR 0010's gap)

**Status:** accepted, 2026-09-23
**Code:** `src/ingest/wiki_transactions.py`, `src/features/transactions.py`,
`src/models/transactions_baseline.py`; small additive changes to `src/models/registry.py`.
**Evidence:** real walk-forward ablation, `reports/` (gitignored; command below reproduces it),
README's [Results](../project-overview.md#results) table.

## Context

ADR 0010 built a roster-context layer from lagged, retrospective proxies (last season's pace, role
share, positional crowding) and found no lift -- honestly attributed to a specific, named gap: no
point-in-time preseason roster/transactions source existed, so the layer could never see a genuine
"this player just changed teams" signal, only an indirect echo of it a season later. This ADR closes
that gap with a real, dated, point-in-time source, and re-tests the hypothesis honestly.

### Source: Wikipedia team-season "Transactions" sections, CC BY-SA 4.0

ADR 0005 D5 flagged Wikipedia team-season pages as a "LATER cross-check" and did not pursue them at
the time. Verified now, directly, against five real pages spanning 2016-17 to 2023-24 (Golden State
Warriors, Memphis Grizzlies, Toronto Raptors, Los Angeles Lakers, Charlotte Hornets -- saved as
`tests/ingest/fixtures/wiki_transactions/` for offline, deterministic tests): every team-season page
in this window carries a `==Transactions==` section with `===Trades===` (each row explicitly dated)
and `===Free agency===` with `====Re-signed====`, `====Additions====`, `====Subtractions====`
subsections (each row identifying one player). This is real, structured, usable data, not a
prototype-only finding.

**License and terms.** Wikipedia text is CC BY-SA 4.0; the MediaWiki API is open for read access
with no authentication, subject to a descriptive User-Agent and reasonable rate limiting (this
project: one request per second, identified UA, `raw` action -- no scripted editing, no bulk mirror).
Per ADR 0005 D6's existing policy, nothing from the source is committed: no raw wikitext, no
rendered page content, only the small set of derived facts (which player moved, to which team, on
which date) is written to the local, gitignored data directory, exactly as every other ingest source
in this project already works. Extracting discrete facts (a transaction's player/team/date) from a
table is not a copyright concern the way republishing Wikipedia's prose would be; nothing here
reproduces article text.

**Genuine format variance -- do not trust column headers.** The exact table shape is not fully
standard across pages: some "Additions"/"Subtractions" tables have an explicit "Signed"/"Date"
column, others use that column position for contract-dollar terms instead (verified directly: the
Warriors 2016-17 page's Additions table is headed "Player | Signed | Former team" but the "Signed"
column actually contains strings like `"2-year contract worth $54.3 million"`, not a date -- a real
editorial inconsistency, not a hypothetical). The reliable date source, present on every row in every
sampled page, is the inline `<ref>{{cite web|...|date=Month Day, Year|...}}</ref>` citation. **The
parser trusts only the citation date, or an explicit dated `Trades` first cell (which is reliably a
full date, `"Month Day, Year"`, in every sample) -- never a "Signed" column's raw text**, and drops
(does not guess) any row where no reliable date can be extracted.

## Decisions

### D1. New ad-hoc table, not a `History`/contract change -- same pattern as `adp`

`team_transactions` follows `adp.parquet`'s precedent (`src/ingest/espn_adp.py`): a standalone
parquet file with its own validation, **not** added to `src/contracts.py`'s `TABLES` registry or
`HISTORY_TABLES`, and not threaded through `History`. Two reasons this is the right call, not a
shortcut:

1. Every transaction row carries an exact date, which lets the feature layer filter to "dated before
   this season's actual opening night" *precisely*, per season -- a stronger leakage guarantee than
   `History.until`'s season-granularity cutoff would give (which only knows "season < target", not a
   calendar date within a season). Building the guard directly on real dates is both simpler and
   tighter than adding a new table to the shared contract's coarser slicing mechanism.
2. `src/contracts.py` is explicitly a coordinated, cross-track shared file (ADR 0001, CLAUDE.md);
   `adp` already established that a benchmark/feature input which isn't consumed through `History`
   does not need to become part of the shared contract. `team_transactions` is read directly by the
   new feature module, the same way `benchmarks.py` reads `adp.parquet` directly.

Columns: `season` (str), `team_id` (int, canonical NBA team id), `player_id` (int, resolved via
`src.ingest.id_map.match_players` -- unmatched players are dropped, never guessed, same policy as
`player_id_map`), `direction` (`"in"` or `"out"`), `source_kind` (`"trade"`, `"addition"`,
`"subtraction"`), `txn_date` (date, **nullable** -- a row with no extractable date is kept for
transparency but excluded from every leakage-safe query, which always filters on `txn_date`).

**The leakage cutoff is always "October 1 of the season's start year," for both training rows and
the real target season, deliberately -- not each season's actual first game date.** A training row's
season *is* inside `History` (its games are already in `History.game_logs`/`team_games`, by
construction, since the row exists in the panel), so its true opening date is available -- but the
real target season being projected (the whole point of this layer) is not: `History.until` excludes
it entirely, so no games exist yet to read an opening date from. Using two different cutoff rules
(exact opening date for training rows, an approximation for the target row) would make the model's
training distribution subtly mismatched from how it is actually used at prediction time. One fixed,
season-start-year rule, used uniformly, avoids that mismatch and is a safe, slightly conservative
approximation in every real case checked (NBA seasons always tip off in October).

### D2. Parsing: row/cell-aware, not a flat link scrape

A naive "grab every `[[...]]` link in the Transactions section" approach was tried by hand against
the Warriors sample and rejected: player and team names both appear as **plain** (non-bold)
wikilinks in the same row in `Additions`/`Subtractions` tables (e.g. `| [[Kevin Durant]] | 2-year
contract worth $54.3 million | [[Oklahoma City Thunder]]` -- both cells are ordinary wikilinks,
distinguishable only by column position, not markup). The parser therefore splits each subsection's
wikitable into `|-`-delimited rows, then `|`-delimited cells, and reads the player from the row's
**first** cell, the counterparty team from the row's **last** cell (`Additions`: former team;
`Subtractions`: new team), and the date from that row's first `<ref ... date=... >` (falling back to
`Trades`' own explicit first-cell date, which needs no citation lookup). A row that cannot be split
into the expected cell shape is skipped and counted, never guessed at.

`Trades` rows are two-sided (`To '''Team A'''<hr />• asset 1<br />• asset 2` / `To '''Team B'''...`);
each `•`-prefixed line within a side is a candidate asset, matched against `players` the same way as
`Additions`/`Subtractions` -- draft-pick and cash-consideration lines simply fail to match a player
and are silently dropped, which is the correct behaviour (they are not arrivals/departures).

### D3. Player matching reuses `src.ingest.id_map.match_players` unchanged

No new matching logic. `match_players(wiki_rows, players, source="wikipedia_transactions",
id_col=..., name_col=..., year_col="season_start")` is called exactly as `espn_adp.py` already calls
it for ESPN names, producing `player_id_map` rows with `source="wikipedia_transactions"` alongside
the existing `"espn"` rows. Ambiguous or unmatched names are reported and dropped, never guessed
(same policy, same report shape).

### D4. The feature: `changed_team` and `team_departures_lost` -- not lagged, dated directly

Unlike ADR 0010's layer (which recency-weights *prior* seasons because it had no dated source), this
layer uses each season's own transactions directly, the same way a rookie's `draft_year == target`
is used directly rather than lagged -- both are facts dated before the season, not facts *about* the
season's outcome.

For a player-season (whether a historical training row or the real target season):

1. **Team assignment.** If the player appears as an `"addition"` or an incoming `"trade"` leg on some
   team T's transactions for that season, dated before that season's schedule start, his season team
   is T. Otherwise his season team is his *previous* season's primary team (continuity). A player
   with neither (no prior season and no recorded transaction -- rare: an unsigned free agent who
   surfaces mid-season, a draftee with a delayed contract) gets no team assignment and the features
   below default to 0, the safe fallback.
2. **`changed_team`** -- 1.0 if the player's season team differs from his previous season's primary
   team (or he had no previous season and a recorded addition -- e.g. a true free-agent pickup with
   no prior-season team on record), else 0.0.
3. **`team_departures_lost`** -- count of players who left the player's season team (`"subtraction"`
   or an outgoing `"trade"` leg) dated before that season's start, weighted by each departing
   player's own *previous*-season mpg share (a bigger name leaving counts for more than a
   fringe-roster departure) -- reuses `role_share`'s definition from ADR 0010's
   `src/features/roster.py` for the departing player's own prior season, so a departure is weighted
   by how central that player actually was, not just counted.

Both features are computed directly from `team_transactions` + the already-available `History`
(for the continuity fallback and `role_share` lookups), independently of ADR 0010's `RosterFeatures`
class (D5 explains why this stays a separate, additive layer rather than a v2 of `baseline_roster`).

### D5. Registered as a new sibling variant, `"baseline_transactions"`, not a rewrite of `baseline_roster`

`baseline_roster` (ADR 0010) is kept exactly as it was ablated and reported -- rewriting it to
quietly add this new signal would make ADR 0010's already-published, already-verified result
misleading (a reader comparing the README's historical numbers against a since-changed model).
`BaselineTransactionsProjector` (`src/models/transactions_baseline.py`) is a new, independent
subclass of plain `BaselineProjector` (not of `BaselineRosterProjector`), overriding the same
`_build_roster_features`-style hook with `changed_team`/`team_departures_lost` in place of ADR
0010's lagged pace/role/crowding trio, wired through the identical minutes-adjustment mechanism
`_core` already has. Registered as `"baseline_transactions"` in `src/models/registry.py`.

## Result (real ablation, 2016-17 through 2025-26)

Reproduce: `python -m src.ingest.wiki_transactions --seasons 2015-16:2025-26` (populates
`team_transactions.parquet` and adds `"wikipedia_transactions"` rows to `player_id_map.parquet` --
ran on 2026-09-23: 330/330 team-season pages fetched, 0 skipped, 5,452 `team_transactions` rows,
2,967/3,493 distinct names matched to a `player_id`, 84.9%, 0 ambiguous), then `python -m
src.backtest --ablate baseline,baseline_transactions --seasons 2016-17:2025-26 --leak-check --out
reports/`. Report at `reports/baseline_transactions_2016-17_2025-26_b30db614/report.md` (generated
reports are gitignored; numbers below are copied from that run). `--leak-check` passed.

**Honest read: this hurts, and it is a more specific, better-understood failure than ADR 0010's.**
Paired bootstrap over players, stratified by season (positive = `baseline_transactions` better):

| metric | lift | 95% CI | p(lift<=0) | seasons won | verdict |
|---|---:|---|---:|---:|---|
| Spearman, total FP | -0.0017 | [-0.0022, -0.0013] | 1.000 | 1/10 | **hurts** |
| Top-50 hit rate | +0.0020 | [-0.0080, +0.0100] | 0.489 | 4/10 | no significant change |
| MAE, total FP | -3.2166 | [-3.7291, -2.7247] | 1.000 | 0/10 | **hurts** |

This is a worse result than ADR 0010's lagged-proxy layer on both metrics that moved: Spearman
*decreases* significantly here (ADR 0010's was flat), and the MAE cost is larger (-3.22 vs -2.42).
The real, dated arrival/departure signal this ADR set out to build does not fix the gap ADR 0010
identified -- it makes the headline numbers worse.

**This was traced to a specific, well-understood mechanism, not an implementation bug.** A fitted
`TransactionFeatures` was inspected directly against real training data (3,933 historical
player-season rows with an earlier season, 585 of them -- 14.9% -- with `changed_team=1`):

* The fitted coefficients are sane in sign and are never clipped: `beta = [intercept +0.129,
  changed_team -0.889, team_departures_lost +0.166]`, and the resulting per-player adjustment on the
  real 2025-26 target season is small (mean +0.15 mpg, std 0.25, range [-0.76, +0.49]) and never
  reaches the +-4.0 `MPG_SHIFT_CAP`.
* The raw (unregressed) signal is real and intuitively sensible, not reversed or nonsensical: the
  weighted mean minutes residual (actual mpg minus the context-free estimate) is **-0.45** for
  players who changed teams that season versus **+0.30** for players who stayed -- team-changers
  really do get somewhat fewer minutes than their historical per-minute rate alone would predict,
  on average, which matches a real basketball intuition (a new team means an unproven, often
  smaller role, at least at first). The raw correlations (-0.045 for `changed_team`, +0.040 for
  `team_departures_lost`) are weak but correctly signed.
* **The failure mode is that a single linear coefficient cannot tell a downgrade from an
  opportunity.** `changed_team` is applied as one uniform penalty to every team-changing player,
  whether he is a bench piece signing a minimum deal (where the average penalty is roughly right)
  or a star joining a rebuilding team as its clear new lead option (where the *opposite* adjustment
  is called for -- exactly the "a star arrives" case PLANNING.md's roster-context row originally
  asked about). Pushing every mover's minutes estimate down by roughly the *average* effect actively
  miscalibrates the tail cases that matter most for rank-order accuracy, which is consistent with
  Spearman getting measurably *worse* here while it was merely flat in ADR 0010 (that layer's
  lagged, continuous pace/role/crowding features had no comparable "one binary flag, one crude
  average" failure mode).

**What this means in practice.** Per PLANNING.md's stated policy for feature layers, `baseline_roster`
(D5) is kept, tested and not recommended (ADR 0010); on this same principle, `baseline_transactions`
is kept, tested and **not recommended** for the draft board or any in-season tool either -- and more
clearly so, since it is a measurable, significant *downgrade* on rank order, not merely a wash. No
further tuning was attempted after seeing this result (adding an interaction term to distinguish
"downgrade" from "opportunity" moves, for instance, is the obvious next idea -- but trying that only
after seeing a negative headline number on the real backtest would be exactly the after-the-fact
tuning-on-eval-data ADR 0004 warns against and ADR 0006 explicitly avoided; it is recorded as a
documented, not-yet-built extension point in Consequences below, not built under the pressure of
this result).

**Match rate note.** 84.9% of distinct Wikipedia transaction names resolved to a `player_id` (2,967
of 3,493), 0 left ambiguous, matching `id_map.py`'s existing "never guess" policy exactly. The
526 unmatched names were spot-checked, not a systematic gap: mostly non-player wikilink artifacts
inside asset lines (e.g. "sign and trade" annotations, correctly not real players) and a handful of
genuine name-spelling mismatches (e.g. "Sviatoslav Mykhailiuk" vs. the roster's "Svi Mykhailiuk") not
covered by the existing alias table -- a real, small residual gap, not something suspected of
driving the headline result (14.9% of the *matched* training population already has `changed_team=1`,
plenty of statistical power for the WLS fit to find whatever linear signal exists).

## Consequences

* A new external data source is added (Wikipedia, CC BY-SA 4.0, read-only, rate-limited, nothing
  committed) -- `docs/adr/0005-data-sources.md` D5 is updated to mark this as built rather than
  deferred. This source (`team_transactions.parquet`, real dated arrivals/departures) is genuinely
  useful data regardless of this ADR's model result and is available for other future uses (e.g. a
  richer future feature, or a simple "who's new" display in the live draft board) without re-running
  the ingest.
* `team_transactions.parquet` and its `player_id_map` rows live in the shared data directory,
  outside the repository and outside `History`, following the `adp` precedent exactly.
* **`baseline_transactions` is not recommended for use** (draft board, future in-season tools, or as
  a default) -- kept registered and tested purely as a reproducible, honestly reported negative
  result, more clearly negative than ADR 0010's. `baseline` (or `baseline_injury`) remain the models
  to use.
* `docs/modeling.md` section 10 is updated to describe this layer as implemented, ablated, and not
  recommended, with the specific mechanism (a single linear coefficient cannot distinguish a
  downgrade move from an opportunity move) documented as the reason, not left as a mystery.
* **Not built, deliberately, given this result:** an interaction-term or team/role-aware version of
  `changed_team` (e.g. conditioning the adjustment on the destination team's positional depth or
  recent record, to distinguish "joining a crowded contender" from "joining a rebuilding team as the
  new lead option") is the natural next idea this result points to -- but building and re-ablating it
  now, right after seeing a negative headline number on the real backtest, would be exactly the
  after-the-fact tuning-on-eval-data ADR 0004 warns against. Recorded here as a scoped, well-motivated
  extension point for a future ADR with its own fresh ablation, not attempted under the pressure of
  this one's result.
