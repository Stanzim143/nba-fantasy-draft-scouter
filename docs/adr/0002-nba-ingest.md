# ADR 0002: NBA stats ingest

**Status:** accepted, 2026-09-22
**Code:** `src/ingest/nba_client.py`, `src/ingest/nba_transform.py`, `src/ingest/nba_stats.py`,
`src/ingest/quality.py`. Report: `docs/data-quality.md`.

## Context

ADR 0001 fixes the contract tables (`game_logs`, `team_games`, `players`, `player_season_bio`) that
every downstream track depends on. This ADR covers how the ingest track fills them from real NBA
data for the 11 backtest seasons, 2015-16 through 2025-26, regular season only.

## Source: stats.nba.com via `nba_api`

**stats.nba.com is an unofficial, undocumented API.** It is not a public developer product, has no
published terms of service or SLA, no authentication, and no rate-limit contract. It is widely used
by the open-source basketball-analytics community (the `nba_api` PyPI package, in the shared venv,
wraps it) but the NBA could change or block it at any time without notice. This is a deliberate,
accepted risk for a hobby project; a production system would need a licensed data source
(Sportradar, Genius Sports, or the NBA's paid partner feeds).

We call the endpoints directly with `requests` rather than through `nba_api`'s endpoint classes,
because we need full control over caching, headers and retry behaviour; `nba_api`'s parameter
dataclasses were used only to *verify* our hand-built parameter dicts against the library's own
defaults (`tests/ingest/test_ingest_pipeline.py::test_*_params_match_nba_api*`).

### Endpoints used and why

| Endpoint | Params of note | Used for | Why this one |
|---|---|---|---|
| `playergamelogs` | `Season`, `SeasonType=Regular Season`, `PerMode=Totals` | `game_logs` (primary) | One call returns every player's every game for a season, with **fractional minutes** (e.g. `36.4516...`), the only source found that reports minutes to the second rather than rounded. |
| `leaguegamelog` (`PlayerOrTeam=T`) | same | `team_games` | One call per season returns every team's every game (two rows per game_id), with home/away encoded in `MATCHUP` and both teams' points. |
| `leaguegamelog` (`PlayerOrTeam=P`) | same | cross-check only | Same player rows as `playergamelogs` but with integer minutes; pulled and diffed against the primary source as a consistency check (`nba_transform.crosscheck_game_logs`), not written to any table. |
| `leaguedashplayerbiostats` | `Season`, `PerMode=Totals` | `player_season_bio.age_at_season_start` fallback, height/weight/draft fallback | One call per season returns every player who appeared that season with an integer `AGE`, height, weight and draft info — no per-player calls needed. |
| `playerindex` | `Historical=1` | `players` (position, height, weight, draft, from/to year) | One call (any season) returns **every player in NBA history** with current attributes; used once per ingest run, not per season. |
| `commonplayerinfo` | `PlayerID` | `players.birthdate` (optional, `--birthdates`) | The only endpoint with an exact birthdate. One call per player — the only per-player endpoint in this pipeline, hence optional and resumable. |

`CommonAllPlayers` was evaluated as an alternative to `PlayerIndex` (both return the full player
list) but `PlayerIndex` includes position/height/weight/draft directly, saving a source.

### Rate limits observed

There is no published or documented rate limit. Empirically, sequential requests roughly 1 second
apart (measured in this session: `min_interval=1.0`, occasional short bursts under 0.5s from
already-warm connections) were served without any 429/403 for a full 11-season pull
(~45 season-level requests) and a birthdate pull of ~1,550 `commonplayerinfo` calls at 1 request/s
(about 26 minutes). No throttling or blocking was encountered in either run. `nba_client.NBAClient`
still implements exponential backoff with jitter and honours `Retry-After` for the case where the
server does start throttling, and refuses to go below **0.8 seconds** between requests
(`nba_stats` CLI floor) regardless of what is asked for.

## `nba_client.NBAClient`: caching, retries, offline mode

* **Disk cache, keyed by endpoint + full parameter set** (`cache_key`: a SHA1 of the canonical
  parameter dict, prefixed with a readable slug). Every response is stored **verbatim** (the exact
  response bytes) under `raw_dir("nba_api")/<endpoint>/<key>.json`, so re-running the ingest never
  re-downloads a season or player already fetched, and the raw cache alone is enough to rebuild
  every processed table byte-for-byte (`test_rerun_is_idempotent_bytes_identical_and_offline`).
* **`NBA_OFFLINE=1` / `offline=True`**: a cache miss raises `NBAOfflineCacheMiss` with the exact
  endpoint, parameters and cache path, instead of touching the network. This is how CI and this
  ADR's reproducibility story work: `./dev test` never hits the network (all HTTP is faked in
  tests), and a backtest can be rebuilt offline from a checked-out `~/dev-data` cache.
* **Atomic writes**: every cache file and the ingest report are written to a temp file in the same
  directory and `os.replace`d into place, so a crash mid-write never leaves a half-written file
  (verified with an injected `os.replace` failure in tests).
* **Politeness**: `min_interval` (default 1.0s, CLI floor 0.8s) between *network* requests only —
  cache hits are free and do not reset the rate-limit clock. Exponential backoff
  (`base ** attempt`, capped at `backoff_max`, `Retry-After` honoured, jittered) on 403/408/425/
  429/500/502/503/504 and on transport-level errors (connection reset, timeout). A real
  Chrome-shaped header set is sent (see `nba_client.DEFAULT_HEADERS`); a bare `User-Agent` with no
  matching `Sec-Ch-Ua`/`Accept`/`Referer` set was observed to get the *connection itself* reset
  (`ConnectionError: Remote end closed connection`) during development — stats.nba.com appears to
  fingerprint the header set, not just check for a UA string being present.
* **Resumable**: `nba_stats._fetch_birthdates` checks the cache (`client.peek`, no network) before
  deciding whether to download, so an interrupted `--birthdates` run picks up exactly where it left
  off on the next invocation (`test_birthdates_are_resumable_after_a_failure`).

## Transform decisions (`nba_transform.py`)

* **Minutes**: `playergamelogs` reports `MIN` as a float (already fractional); `MIN_SEC` (`"MM:SS"`)
  is used as a fallback if `MIN` is entirely null. `parse_minutes` also accepts plain `"MM:SS"`
  strings for robustness, though the primary source never needs it.
* **Did-not-play rows**: any row with minutes null, zero, or negative is dropped — the contract
  defines `game_logs` as games actually played. A handful of real rows (see
  `docs/data-quality.md` §9) are 0:00 stint rows that still carry a stat line (e.g. a free throw
  recorded at the buzzer); these are dropped like any other 0-minute row and the lost points are
  reported, not silently discarded.
* **Non-regular-season game ids**: filtered by game-id prefix (`00` + two-digit season-start year +
  five digits = regular season; `SeasonType=Regular Season` already asks the source for this, so
  the filter is a defence-in-depth check, observed to remove 0 rows in the real pull).
* **Team identity vs. abbreviation**: `team_id` is the only identity ever used to key or join;
  `team_abbr` is stored exactly as the source reported it *for that row*, so a franchise rename
  (NJN→BKN 2012, CHA↔CHH/Bobcats-vs-Hornets naming history, NOH→NOP 2013 — all before this
  project's 2015-16 start, so not actually exercised by the real data, but exercised by
  `test_team_abbreviation_changes_do_not_change_team_id_identity`) never fragments a team's game
  log across two ids.
* **Trades**: a traded player's `game_logs` rows use the team of that specific game (from the
  source); `player_season_bio.team_id` is the team of the player's **last game of the season** by
  `game_date` (ties broken by `game_id`), per ADR 0001 ("team at end of regular season").
* **Bubble (2019-20) and shortened (2020-21) seasons**: nothing in the transform or the ingest CLI
  assumes 82 games. `team_games` is built entirely from what the source returns; `docs/data-quality.md`
  §2 verifies from the real data that 2019-20 has 64-75 games per team (30 teams played to the
  March 11 suspension; 22 of them finished the season in the Orlando bubble) and 2020-21 has
  exactly 72 for every team.
* **Duplicate rows**: identical duplicates are dropped outright; conflicting duplicates on the same
  `(game_id, player_id)` keep the row with the most minutes (a deterministic, order-independent
  tie-break) — not observed in the real pull, but exercised by tests.
* **Box-score identities**: enforced by `contracts.validate_table` on every write; the real pull
  has **zero** violations across 281,138 rows (see `docs/data-quality.md` §4).
* **`age_at_season_start`** ("years, as of Oct 1 of the season's start year", per ADR 0001):
  1. **Exact**, when a birthdate is known: `(Oct 1 - birthdate).days / 365.25`. The only
     imprecision is the 365.25-day-year approximation, bounded at about 0.002 years (leap-year
     effects), verified in `test_exact_age_matches_calendar_age_within_a_hundredth_of_a_year`.
  2. **Fallback**, when no birthdate is known: `leaguedashplayerbiostats.AGE - 0.2447`. `AGE` is
     the player's whole-number age; empirically (verified against 2,892 real player-seasons that
     have both a `commonplayerinfo` birthdate and a bio-stats `AGE`, `docs/data-quality.md` §7) it
     equals `floor(exact age on June 30 of the season's end year)` 99.9% of the time. Since Oct 1
     of the start year is 272 days before June 30 of the end year, and the expected fractional
     part of a floored age is 0.5, `E[age at Oct 1] = AGE + 0.5 - 272/365.25 = AGE - 0.2447`. The
     derivation and the constant live at `nba_transform.AGE_OCT1_OFFSET`. Worst-case error is
     **0.5 year** (the fallback carries no information about the fractional year); the real
     calibration's max observed error is reported in `docs/data-quality.md` §7 and was **well
     under** that bound (see the report's Calibration line).
  3. **Cross-season borrow**: if a player has no bio row in the target season (rare — a player who
     appears in `game_logs` but was omitted from that season's bio-stats pull) but does in another
     season, that season's `AGE` is shifted by the number of seasons apart, same offset and bound.
  4. If none of the three apply, the player-season is **dropped from `player_season_bio` and
     counted** (`no_age_source`) — the contract disallows a null age, so we do not invent one. In
     the real pull, every player-season had a birthdate available (0 dropped this way, once the
     `--birthdates` pass had run) — see `docs/data-quality.md` §7 and §9.
* **Birthdates are opt-in and additive**: without `--birthdates`, `commonplayerinfo` is never
  called; ages use the fallback and are provably still within its bound. With `--birthdates`, only
  players who appear in the seasons being (re-)ingested are fetched, and only if not already
  cached; a birthdate learned on one run is never forgotten on a later run without the flag
  (`nba_stats.merge_players`: a null in a new pull never erases a previously known value).

## Idempotent, incremental merge (`nba_stats.py`)

Each of the four tables is **rebuilt from scratch for the requested seasons only** and merged with
whatever the store already has for other seasons (`merge_seasons` for the season-indexed tables,
`merge_players` — union, non-null wins, existing values never erased by a re-pull's nulls — for the
static `players` table). A table on disk is only rewritten when its content actually differs
(`frames_equal`, dtype-insensitive), so re-running the exact same ingest touches no files at all —
verified end-to-end with byte-identical SHA-256 digests of every processed file *and* the ingest
report (`test_rerun_is_idempotent_bytes_identical_and_offline`). Running seasons one at a time and
running them all at once produce byte-identical tables
(`test_seasons_can_be_added_incrementally_and_match_a_single_full_run`).

A season pull that comes back structurally broken — the source returns nothing (season not yet
played), a player row's `(game_id, team_id)` has no matching `team_games` row (an availability-
computation hazard), or a box-score identity fails — **raises before anything is written**, leaving
the store exactly as it was (`test_existing_tables_survive_a_failed_run`).

## Reproducibility

1. `python -m src.ingest.nba_stats --seasons 2015-16:2025-26` — pulls (or resumes from
   `~/dev-data/nba-fantasy-2026/raw/nba_api/`), transforms, validates and writes the four tables.
2. Add `--birthdates` for exact ages (adds ~1 request per not-yet-seen player, ~1550 for the full
   backtest range, ≈0.8-1.5s apart ⇒ roughly 25-35 minutes on a fresh cache; a few seconds when the
   cache is already warm). Safe to interrupt and re-run.
3. `NBA_OFFLINE=1 python -m src.ingest.nba_stats --seasons ...` rebuilds the tables from the raw
   cache alone with **zero** network requests — this is what makes the backtest reproducible
   without re-hitting an unofficial, unversioned API.
4. `python -m src.ingest.quality` regenerates `docs/data-quality.md` from whatever is currently in
   the store plus `nba_ingest_report.json` (drop counts, cross-check results, age-fallback
   calibration) written by step 1.

## Adding a season

Once a new NBA season starts, `python -m src.ingest.nba_stats --seasons 2026-27 --birthdates` pulls
it and merges it in without touching any other season's data. While a season is in progress, add
`--refresh` to re-download that season's (still-changing) season-level responses instead of trusting
a stale cache entry; per-player birthdates never need `--refresh` (a birthdate does not change).
Then regenerate the quality report.

## Known limitations

* **Unofficial source** (see above): no SLA, could break without notice. Fallback if it does:
  `nba_api`'s CDN-backed endpoints (`stats.nba.com`'s companion `cdn.nba.com`) or a licensed feed;
  the raw JSON cache means a backtest already run does not need to be rebuilt even if the source
  disappears.
  Fallback if the source is later blocked from this network: run the ingest once from an unblocked
  network/proxy to populate the shared raw cache, then everyone else works `NBA_OFFLINE=1`.
* **`players` position/height/weight/draft are point-in-time-of-ingest, not point-in-time-of-season.**
  `PlayerIndex` and `CommonPlayerInfo` return a player's *current* attributes; a player's position
  or listed weight from 2015-16 as it was recorded *then* is not available from these endpoints.
  Only `player_season_bio.age_at_season_start` and `.team_id` are genuinely per-season.
* **Birthdates require an explicit, slow, per-player pull** (`--birthdates`); without it, all ages
  use the documented 0.5-year-bounded fallback. In the real run, `--birthdates` was used for the
  full ingest, so every player-season has an exact age (0 fallback rows; see report §7).
* **The cross-check (`leaguegamelog` P vs `playergamelogs`) finds a small number of stat
  disagreements in recent seasons** (2022-23 onward; see `docs/data-quality.md` §9) — these are
  differences *between two stats.nba.com endpoints themselves* (likely post-hoc official
  corrections applied to one endpoint and not the other), not a bug in this ingest: the primary
  source (`playergamelogs`) satisfies every box-score identity with zero violations across all
  281,138 rows.
* **No injury, transaction, contract, or ADP data** — out of scope for this track (see ADR 0001
  track ownership); this ingest covers `game_logs`, `team_games`, `players`, `player_season_bio`
  only.
* **`player_id_map`** (ADR 0001's ID-mapping table for non-NBA sources) is not produced by this
  ingest; `player_id` here *is* the canonical NBA `PERSON_ID` already, so no mapping is needed for
  this source. A future ESPN/Spotrac ingest would populate `player_id_map`.
