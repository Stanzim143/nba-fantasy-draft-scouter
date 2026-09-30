# ADR 0025: Draft-day live sync — auto-detect opponents' picks from the ESPN league sync

**Status:** accepted, 2026-09-28
**Code:** `src/app/live_sync.py` (pure sync/diff/attribution logic), wired into `src/app/draft_board.py`'s "Live draft sync
(ESPN)" sidebar section. Tests in `tests/app/test_live_sync.py`.

## Context

The draft board (ADR 0009) requires a manual "Mark drafted" click for every pick, including the ~12 picks per round made
by opponents in our 13-team league. During the actual live draft (2026-10-17, see `config/league.yaml`), that is a lot of
manual bookkeeping happening at the same time as reading the board — a missed or mistyped opponent pick desyncs the board
from reality for the rest of the draft. `src/ingest/espn_league.py` already pulls the league's live draft detail
(`mDraftDetail`: `draft.picks[].{overall_pick, team_id, espn_player_id}`), read-only, cached, rate-limited. This ADR wires
that into the board so opponents' picks appear automatically.

## Decision

### Polling, not push

ESPN's fantasy API exposes no webhook/subscription for a live draft; the only option is to ask again. Streamlit itself has
no background timer (it re-runs top-to-bottom on every interaction and has none of its own otherwise), so "live" here means:

* An explicit **Sync now** button always attempts a poll.
* Every rerun the app performs for any other reason (a tab switch, a manual pick, widget interaction) *also* attempts a
  poll if enabled — but gated by `should_poll()`, a floor of `DEFAULT_MIN_POLL_INTERVAL` = 20 seconds since the last
  attempt. This gets the app "auto-refreshing" during normal draft-day use (clicking around while waiting for a pick)
  without adding a JS auto-refresh dependency or a real background thread, both of which fight Streamlit's rerun model.

This is a deliberate scope limit: there is no guarantee of a poll within N seconds if the user leaves the tab completely
idle. Accepted because the user is expected to be actively interacting with the board throughout their own live draft;
the manual control remains the hard fallback regardless (see below).

On top of the app-level floor, `src.ingest.espn_league.ESPNLeagueClient` already rate-limits/backs off individual HTTP
requests (`min_interval`, exponential backoff on 429/5xx) and caches every response to disk — reused unchanged here via
`fetch_draft_detail`. One `ESPNLeagueClient` instance is kept per league id in `st.session_state` across reruns so its
internal "time of last request" state persists between polls, rather than resetting (and losing its rate-limit memory)
on every click.

### One poller at a time: reuse `RunLock`

CLAUDE.md's "only one worker downloads from stats.nba.com at a time" convention exists for a different resource
(stats.nba.com ingest), but the same problem shows up here for ESPN if the same league is open in more than one browser
tab/session during the draft (e.g. a phone and a laptop). Rather than invent a second locking primitive, `poll_lock()`
reuses `src.ops.runlock.RunLock` — already this repo's exclusive, stale-aware, atomically-created (`O_EXCL`) file lock,
used by the nightly job. It is scoped with a much shorter `stale_after` (60s, vs. `RunLock`'s 3-hour default built for a
once-a-day batch job): a live-draft poll is expected roughly every 20 seconds, so a lock that outlives that by more than
3x is assumed to belong to a crashed/closed tab and is reclaimed rather than blocking the rest of the draft. Losing the
lock race is not an error: `sync_once(..., lock_path=...)` catches `LockBusy` and returns `SyncResult(locked_out=True)`,
a benign "someone else just checked, try again next interval" — never surfaced as a connection error.

### Attribution: "me" vs "opponent"

The in-app `DraftState` (ADR 0009, `src/app/state.py`) only ever distinguishes two sides, `ME` and `OPPONENT` — it has no
concept of the other ~11 teams individually. Live sync keeps that model rather than expanding it: `classify_team(team_id,
my_team_id)` attributes a detected pick to `"me"` only when its ESPN `team_id` matches a configured `my_team_id`, and to
`"opponent"` otherwise (including when `my_team_id` is unset — the safer default, since only "me" picks would ever be
mislabelled, not opponent picks). `my_team_id` is resolved the same way the existing in-season tooling already resolves
"which team is mine" (`src.ops.nightly --set-team`, `$ESPN_TEAM_ID`) — `default_team_id()` checks `$ESPN_TEAM_ID` first,
then the `team_id` already saved in `<data_dir>/nightly.json` (or the older `daily_refresh.json`) by a previous
`--set-team` run, so a user who set their team after a previous draft never re-enters it. The sidebar's "My ESPN team ID"
field can still override either at any time.

### Player id resolution: reuse `player_id_map`, never guess

ESPN's draft picks carry only `espn_player_id`; the board is keyed on this project's own `player_id` (canonical NBA id).
Rather than build a second name-matching pipeline, `resolve_player()` looks the id up in the existing `player_id_map`
contract table (`src.contracts`), already populated by `python -m src.ingest.espn_adp`'s id-matching (`src.ingest.id_map`,
exact/normalized/alias/fuzzy). A pick whose id is not in that table yet is **not** guessed at by name or dropped
silently — it comes back as `unresolved`, surfaced in the sidebar with a nudge to re-run `espn_adp` and mark it manually.
This keeps live sync leak-free of a second, less-vetted matching heuristic.

### Applying picks: same function as the manual form, dedup by player id

`apply_detected_picks()` runs every resolved, not-yet-drafted pick through the exact same `draft_player()` the manual
"Mark drafted" form calls (`src/app/state.py`) — no parallel state-mutation path to keep in sync. A detected pick whose
`player_id` is already in `DraftState.drafted_ids` is classified `duplicate` and skipped, regardless of *which* path
(manual click or an earlier poll) marked it first: the manual form's click always wins any race, since it applies
immediately in the same rerun, and the next poll simply sees that player as already drafted and skips it. This makes the
two paths commutative rather than requiring the user to disable one while the other runs.

### Failure handling: always degrade, never disable the manual control

Every `ESPNLeagueError` from `fetch_draft_detail` (auth failure, league not found, offline-cache-miss, retries
exhausted on a 429/5xx run) is caught inside `sync_once()` and turned into `SyncResult(ok=False, error=<message>)`
rather than propagating. `draft_board.py` shows that error text plus a running "N in a row" failure count in the
sidebar's status caption (`_live_sync_status_caption`) — connected / error / not-yet-run, with the last-synced
timestamp — but **never** disables, hides, or otherwise touches the manual "Mark drafted" form on any tab. A user whose
league sync is broken mid-draft (private league needing cookies, an ESPN outage, a stale offline cache) keeps drafting
exactly as before ADR 0025 shipped.

### Config

* `$ESPN_LEAGUE_ID` (already used by `src.ingest.espn_league`) — reused unchanged; also overridable per-session in the
  sidebar.
* `$ESPN_TEAM_ID` (already used by `src.ops.nightly`/`src.inseason`) — reused unchanged for "my team" attribution,
  falling back to the `nightly.json`/`daily_refresh.json` `team_id` setting.
* No new credentials are introduced. `ESPN_S2`/`SWID` (private-league cookies) are untouched — this feature works the
  same as `espn_league.py` itself: public leagues out of the box, private leagues once those are set in the user's own
  `.env` (never committed; `.env.example` documents the shape, not values).

## Consequences

* Opponents' picks appear on the board without a manual click, as long as the poll interval has elapsed and the id map
  has the player.
* A pick for a player not yet in `player_id_map` (a very recent signee, a name-matching gap) still requires a manual
  mark — flagged, not silently missed.
* "Live" is best-effort, bounded below by 20 seconds and by however often the app reruns for another reason; it is not
  a guaranteed real-time feed. Acceptable for a single fantasy draft's pace (default 90 seconds/pick, `config/league.yaml`).
* Multi-league support (attributing more than one `my_team_id`/league at once) is explicitly out of scope; the app is
  built for the one real league in `config/league.yaml`, same as every other module in this repo.
