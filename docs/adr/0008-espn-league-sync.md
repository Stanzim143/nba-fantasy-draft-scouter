# ADR 0008: ESPN league sync (our real league)

**Status:** accepted, 2026-09-23
**Code:** `src/ingest/espn_league.py`, `tests/ingest/test_espn_league.py`,
`tests/ingest/espn_league_fakes.py`.
**Builds on:** ADR 0005 (`docs/adr/0005-data-sources.md`, decision D1) and
`docs/research/espn-api-findings.md` section 7, which verified the league-agnostic ESPN endpoints
but left our actual league (id 1234567890) unverified because it needed the league id, which the
user has now provided.

## Context

The project's real league, "Example League" (`config/league.yaml`), was configured by hand from
the ESPN settings page on 2026-09-22, with team count explicitly marked provisional ("10
provisional, ESPN page showed 13, manager says ~10 will play"). ADR 0005 flagged this as risk R7
("league not readable anonymously") and open question ("league id and anonymous readability").
This ADR verifies both against the live league and reconciles every checkable `league.yaml` field
against ESPN's own settings.

## What was verified (2026-09-23, live requests against league id 1234567890)

* **The league is public.** `GET .../seasons/2027/segments/0/leagues/1234567890?view=mSettings`
  returns **HTTP 200** with no `espn_s2`/`SWID` cookies, from an ordinary connection with a
  descriptive User-Agent. `settings.isPublic` is `true`. No cookie-auth code path was built --
  there was nothing to verify it against, per the task's instruction not to build speculative,
  untested cookie paths.
* **ESPN season id 2027 (2026-27) is correct on the first try**; no fallback to 2026 was needed.
  The client still tries the season before the derived one if the derived one 404s (see below),
  in case this ever needs to run before ESPN has created the season's league object.
* **Team count is 13, not 10.** `settings.size == 13`, `len(teams) == 13`,
  `status.isFull == true`, `status.teamsJoined == 13`, and the snake-draft `pickOrder` lists 13
  team ids. This matches the ESPN-page figure the config comment already flagged as the likely
  real number, and directly affects replacement-level and VORP math (`src/value/replacement.py`
  reads `league.teams`). **`config/league.yaml`'s `teams:` field is updated to `13` by this
  change** (see "Config change" below).
* **No draft has happened.** `draftDetail.drafted == false`, `draftDetail.inProgress == false`,
  every team's `roster.entries` is `[]`, and every draft pick has `playerId: -1` (unfilled).
  Consequently the free-agent pool (`view=kona_player_info` with the `FREEAGENT`/`WAIVERS` filter)
  is effectively the entire rosterable player universe -- this is handled explicitly, not as an
  accidental side effect of an empty roster (`draft_completed` is a first-class field in the
  output; the CLI prints "Draft: NOT completed yet" rather than silently reporting zero rostered
  players as if that were suspicious).
* **Scoring matches exactly.** All 11 configured scoring weights in `config/league.yaml` reconcile
  as exact matches against ESPN's live `scoringSettings.scoringItems` (stat id to points, mapped
  through `STAT_ID_TO_KEY`, which was checked against the espn-api package's `STATS_MAP` and the
  real payload). This confirms `config/league.yaml`'s scoring section, which the user already
  marked as confirmed real, and which this task was told never to touch -- it wasn't; this is
  read-only cross-checking.
* **Roster slots match exactly**: `lineupSlotCounts` decodes to PG/SG/SF/PF/C/G/F = 1 each, UTIL =
  3, bench = 3, IR = 1 -- identical to `config/league.yaml`'s `roster.starters`/`bench`/`ir`.
* **Draft settings, trade settings and most schedule settings match** (snake, manual order, 90 s
  per pick, pick trading on, no trade limit, 4 veto votes, 20 regular-season matchups, 8 playoff
  teams, no keepers). Two fields reconcile with a caveat rather than a clean match (both explained
  in the reconciliation `note` field, not silently flagged as bugs):
  - **`acquisition.matchup_limit`**: config says "7 per matchup" (a 7-day period); ESPN's own
    representation is `matchupAcquisitionLimit: 1.0` with `matchupLimitPerScoringPeriod: true`,
    i.e. 1 acquisition per *day*, not per matchup week. These are different units (1/day vs.
    7/week) that happen to be consistent with each other for a 7-day matchup period, but the
    reconciliation reports them as non-equal rather than silently converting one into the other,
    because ESPN's own settings payload never states the conversion is exact.
  - **`trades.deadline`**: config says `2027-03-13`; ESPN's `deadlineDate` epoch (UTC) converts to
    `2027-03-12` or `2027-03-13` depending on timezone rounding (the epoch is a UTC midnight
    boundary that lands on the 12th in UTC but is the 13th in most US timezones). Flagged with a
    note rather than claimed as a hard mismatch.
* **No transactions yet** (`view=mTransactions2` returns no `transactions` key at all, not an
  empty list -- handled explicitly in `parse_transactions`, which returns `[]` either way).

## Decision

1. **A new, separate, thin client** (`ESPNLeagueClient` in `src/ingest/espn_league.py`), not the
   `espn-api` package and not a modification of `src/ingest/nba_client.py`. It follows
   `nba_client.py`'s caching pattern loosely (disk-cached raw JSON under
   `raw_dir("espn_league")/<league_id>_<season_id>/<views>__<hash>.json`, atomic writes, an
   offline mode that raises rather than guesses, a descriptive User-Agent) but is deliberately
   smaller: no cookie support, a hard stop (not a retry) on 401/403, and multiple ESPN `view=`
   parameters merged into one request wherever ESPN's API supports that (verified: `mSettings`,
   `mTeam`, `mRoster` and `mStandings` all come back in a single response).
2. **Four read-only GETs per run**: settings+teams+rosters+standings, draft detail, free agents
   (capped at `--free-agent-limit`, default 200), transactions. All are GET; nothing in this
   module writes to ESPN under any circumstance, matching the read-only-reconnaissance requirement.
3. **A hard stop on 401/403**, never a retry, never a guess: `ESPNAuthError` propagates straight to
   a clear CLI message telling the user to add `ESPN_S2`/`ESPN_SWID` to their own `.env` (already
   templated in `.env.example`) if the league turns out to need them later (e.g. a manager
   tightens privacy). This path is untested against a real private league because ours is public;
   the ADR says so plainly rather than pretending otherwise.
4. **A lightweight, explicitly documented JSON output**, not a `TableSpec` contract table --
   `data_dir()/processed/espn_league/<league_id>_<espn_season_id>.json`. This data does not fit
   the per-player-season shape of `src.contracts`'s tables (it is per-league, with nested
   per-team-roster and per-pick shapes, and a "no draft yet" state that has no season-level
   analogue), so forcing it into `TableSpec` would either lose information or require a schema
   change disproportionate to what Phase 5 needs today. The shape is documented in the module's
   own docstring rather than only in this ADR, since that's what a future consumer will actually
   read.
5. **Real ESPN member names are dropped during parsing.** `members[].displayName` (ESPN's own
   pseudonymous "ESPNFANxxxxxxxxx" string) is kept; `firstName`/`lastName` are not persisted
   anywhere this module writes, even though they were visible in the raw payload (which does stay
   cached locally, ungitignored... no -- see below). This is a judgement call beyond what the task
   strictly required, made because the league's real members are private individuals whose full
   names have no use in this project's projections.
6. **No ESPN-to-NBA player id mapping is built here.** Free agents and rosters are reported with
   ESPN's own `id` (`espn_player_id`). ADR 0005 section 8 already designs the alias-table approach
   for this (rookie/ADP matching); wiring league rosters through `player_id_map` and into
   `src/value/board.py`'s "exclude already-rostered players" behavior is left as follow-up work
   (see Limitations) rather than duplicated or guessed here.

## Config change

**`config/league.yaml`'s `league.teams` field changes from `10` to `13`**, and the inline comment
changes from `# PROVISIONAL: ESPN page shows 13, manager says ~10 will play. Update when final.`
to a comment recording the verification: source, date, and method. This is a user-facing,
project-wide change -- `src/value/replacement.py` derives replacement level from `league.teams`,
so every VORP number downstream of it shifts. It is called out separately and prominently in this
change's commit message and the final report, not buried in the diff. `PLANNING.md` and
`README.md` are **not** touched by this change (they belong to the repo-hygiene track per ADR
0001's ownership table); their stale "10 provisional" / "still shows 13" lines are a known
follow-up (see Limitations).

## Limitations

* **Cookie-auth path is unbuilt and unverified.** If this league (or another one this code is ever
  pointed at) is or becomes private, `espn_s2`/`SWID` support needs to be added and tested against
  a real private league before it can be trusted -- not before it merely "looks right" against
  docs.
* **No player id mapping.** `espn_player_id` values in rosters/free agents/draft picks are not
  joined to the canonical NBA `player_id` used everywhere else in this project. This is real,
  scoped-out duplication of effort that ADR 0005's alias-table design (section 8) and the ADP
  track both anticipate; whichever lands first on `main` should be reused by the other rather than
  each building its own mapping.
* **Draft results and post-draft rosters are untested against real data** because no draft has
  happened in this league yet. `parse_draft`/`parse_roster_entries` are unit-tested against
  realistic fixture shapes (`tests/ingest/espn_league_fakes.py`, built from the `espn-api` package
  source and this ADR's live probes of the *shape* of a filled pick/roster entry), not against an
  actual filled roster from this league. Re-verify once the draft happens.
* **`matchup_limit` and `trades.deadline` reconcile with a caveat, not a hard boolean**, as
  described above; a human should read the `note` field rather than trust `match` alone for those
  two rows.
* **`README.md`/`PLANNING.md` still say "10 (provisional)"** and belong to the repo-hygiene track;
  this ADR's config change makes them stale. Flagged here so the next repo-hygiene pass picks it
  up, not fixed by this change (out of this track's owned paths).
* **Raw ESPN payloads are cached locally** under `raw_dir("espn_league")`
  (`~/dev-data/nba-fantasy-2026/raw/espn_league/` by default), which is outside the repo and
  already covered by the project's existing "never commit raw data" convention -- no new
  `.gitignore` entry was needed. They do contain real member first/last names (ESPN includes them
  in the `members` block even though this module's parsed output does not); this is the same
  local-only, non-committed, personal-use posture ADR 0005 already accepted for ESPN data
  generally, not a new risk.

## Consequences

* `python -m src.value.board` and other VORP/replacement-level consumers now have a *verified*
  real team count (13) to use instead of the provisional 10, once someone wires `--teams` from
  this module's output (not done automatically by this change -- board.py's `--teams` flag already
  supports an override; this change only makes the correct number available and documented).
* The Phase 5 roadmap item "ESPN league sync" (`PLANNING.md`/`README.md` roadmap tables) is now
  partially done: settings/rosters/free-agents/draft/transactions sync exists and is tested; a live
  draft board and nightly refresh are still open.

**Update (2026-09-23, same day):** the `README.md`/`PLANNING.md` "10 (provisional)" follow-up
flagged above was picked up immediately (integrated alongside three other parallel tracks landing
the same day) rather than waiting for a separate repo-hygiene pass -- both files, plus the other
stale "provisional (10)" references this ADR's own search missed in `docs/adr/0003-projection-model.md`,
`docs/backtest.md`, and `docs/modeling.md`, now correctly say 13.

**Addendum (2026-09-30):** the live public ESPN league now reports `settings.size == 14` (13 managers plus
one placeholder team; the league is hoping to find a 14th manager), so `config/league.yaml` is `teams: 14`
and the draft is a 14-slot snake. Replacement level, `BreakoutConfig.rostered_rank` (now derived from the
config instead of a hard-coded 169) and the backtest's default replacement pool all follow the config. The
in-season synthetic test world stays sized for 13 rosters via a pinned test config (`ise_testkit.CFG13`).
