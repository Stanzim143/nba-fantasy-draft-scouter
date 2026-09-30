# ADR 0017: Live transactions tracker (ESPN feed -> dated ledger -> ranked digest)

**Status:** accepted, 2026-09-25
**Code:** `src/ingest/espn_transactions.py`, `src/features/txn_impact.py`, `src/ops/txn_watch.py`, `src/app/txn_view.py`
(the app's "Transactions" tab in `src/app/draft_board.py`); additive changes to `src/ingest/preseason_refresh.py` (opt-in
`transactions` extra step), `src/ops/daily_refresh.py` (a last `transactions` step and diff hook), `src/ops/daily_report.py`.
**Usage:** [`docs/transactions.md`](../transactions.md). Follows ADR [0011](0011-transactions-layer.md) (which used *retrospective*
Wikipedia transactions as a projection feature) and ADR [0012](0012-offseason-layer.md) (roster snapshots).
**Evidence:** real 2026-09-25 run against the live feed, numbers below; tests under `tests/`.

## Context

The request: stay current on trades, free-agent signings and the like in the run-up to the draft (2026-10-17 (NZ morning; fixed 2026-09-26, ADR 0014)) and through the season.
What existed: dated `roster_snapshots` (who is on which roster each day, ADR 0012) and the daily report's roster diff. That sees *that* a
player changed team a day late and cannot say *how* (trade, signing, waiver claim, two-way conversion), sees nothing about a player who
is cut and unsigned, and sees no coach or front-office move. ADR 0011's Wikipedia source is historical (team-season pages appear after the
season) and, more to the point, was measured as a projection input and did not help.

## Decisions

### D1. Source: ESPN's public transactions feed
`site.api.espn.com/apis/site/v2/sports/basketball/nba/transactions`, the same free, no-auth site-API family as the player feed `espn_adp`
already reads. Verified directly on 2026-09-25: one row per team per day with a free-text `description` (several actions per row),
filterable with `dates=YYYYMMDD-YYYYMMDD`, paged with `page` and `limit`, and it reaches back years (2016 rows were returned, including
"Named ... assistant coaches"). It currently carries the whole 2026 offseason: the Jul 6 trades and signings, Sep waivers, extensions,
head-coach and front-office moves. **Terms:** same accepted stance as ADR 0005 R2 (private, non-commercial, low volume, local only, never
committed); polite (>= 2 s between network requests, cached under `raw/espn/transactions`, only windows touching the last three days are
re-downloaded). Rejected: NBA.com has no comparable public feed; Pro Sports Transactions (ADR 0005) and Spotrac forbid automation.

### D2. Parse conservatively into person-level rows
A description is split into sentences on verbs (so "A.J.", "Jr." and "L.A." do not break it) and read with a small grammar: `signed`,
`resigned`, `extended` (any "contract extension"), `converted` (two-way to NBA), `waived` (including "placed on waivers"), `claimed`, and
`trade_in` / `trade_out` (the Acquired sentence: incoming assets, "from <team>", "in exchange for" / "for", "sent ... to <team>", and
"X from T1 and Y from T2" three-team legs). Staff sentences produce `hired`, `fired`, `resigned_staff`, `extended_staff` with a role
(`head_coach`, `assistant_coach`, `general_manager`, `president_ops`, ...). Picks, cash and "draft considerations" are dropped. A sentence
it cannot read is **counted, never guessed**; a row with nothing readable is kept as `kind='other'` with the full text. A person's name must look like a name (2 to 5 capitalised tokens); text the grammar failed to cut off a name ("... for the remainder of the season", a source-team prefix) is refused, not recorded as a person. The independent verifier found real wrong-move bugs in the first version (a three-team trade with two feed typos read a departing player as arriving, mangled names); they are fixed and pinned by regression tests, and the claim is now the narrower one that the grammar refuses what it cannot cut cleanly, not that it can never be wrong.
`league_transactions` is an ad-hoc table (precedent: `adp`, `team_transactions`), one row per person per action.
The key is `sha1(date, team, kind, person)`: it never depends on the counterparty or contract text the parser infers. A later parse fills a missing `player_id` or `other_team_id` but never overwrites one. Because the person is part of the key, a *better name* is a different key; so `merge_ledger` treats a ledger row whose feed row (same day, team and text) is now parsed differently as superseded: it is dropped and its replacements inherit the earliest `first_seen`, so a parser improvement never duplicates rows and never re-announces old news.

### D3. Append-only ledger with `first_seen`
ESPN dates are days, not times, so "new" cannot be a timestamp on the transaction. The ledger stores `first_seen` / `last_seen` (UTC ingest
times). `python -m src.ops.txn_watch` reads "new since last acknowledged" as `first_seen > watermark`; `--ack` moves the watermark
(`<data>/ops/txn_watch.json`), so a plain run is read-only. First run, before any acknowledgement: the last 7 days by transaction date.

### D4. Matching: active players first, then everyone
`match_players` against `players` rows still playing (within a season of the newest data) plus the newest roster snapshot; only names left
over are tried against the all-time `player_profiles` index. This resolves father and son (Tim Hardaway Jr., Larry Nance Jr., Gary Trent Jr.,
Ron Harper Jr.) to the player active now instead of leaving them ambiguous. The league year of the move is the year tie-break.
Unresolved names keep `player_id` null and are listed in `processed/espn_transactions_report.json`.

### D5. Impact is descriptive, and says so
`txn_impact.events` collapses the two teams' rows of a trade into one move `from -> to`; `annotate` adds board rank, position, VORP and ADP,
and a `context` line: the destination's best-ranked teammates (and "N of the next 8 share his position" when three or more do) and who is
left behind at the origin. Only players the board ranks within the top 180 are listed by default (the other moves are counted:
"N other moves involve players outside ..."). **Nothing here changes a projection or a rank.** ADR 0010 and 0011 tested "changed team" and
"a star arrived or left" against ten real seasons and found no lift (0011 measurably hurt), so this feature does not claim a move is worth
some number of fantasy points; it tells you it happened, how much the board cares about the player, and who else is in the room.

### D6. Where it surfaces
* `python -m src.ops.txn_watch` (CLI: `--refresh`, `--ack`, `--days`, `--since`, `--team`, `--kind`, `--player`, `--all`, `--staff`).
* The daily refresh's last step, `transactions`, after the boards exist so the report ranks against this run's board: a "League
  transactions" section listing what the ledger first saw in that run, plus staff moves (`reports/daily/latest.md`).
* The draft-board app's **Transactions** tab (reads the local ledger, joins the board already loaded; never touches the network).
* `python -m src.ingest.preseason_refresh --with-extras` (the opt-in `transactions` step, 45-day lookback).

## Evidence (real data, 2026-09-25)

* A 120-day ingest: 183 feed rows became 280 person-actions (90 signed, 69 waived, 43 trade_in, 41 resigned, 23 trade_out, 11 extended,
  1 claimed, 1 converted, 1 hire); the first pass left one sentence unreadable (a "Converted ... to a two-way contract" phrasing, since
  added to the grammar, so now none); 211 of 248 distinct names resolved to a `player_id` (85.1%, none ambiguous). The 37 unresolved are summer-camp and Exhibit-10 fringe players who never reached the NBA index (mostly waived days after
  signing); they keep a null `player_id` and are listed in the ingest report. One was a real miss worth fixing: ESPN's feed writes the
  #41-ranked rookie as "Anicet Dybantsa" while the board has "AJ Dybantsa"; the surname is unique in the league, so it is now an
  explicit alias in `id_map.ALIAS_GROUPS` (a reviewable one-line entry, not a fuzzy match).
* **Cross-check against an independent source.** For each player with a resolved move, the ledger's latest move was compared with the NBA
  roster snapshot of 2026-09-25 (a different vendor): 207 of 209 agree (99%): an arrival's destination is the player's current team, a
  waived player is not on the waiving team. The two disagreements are both informative. Trey Alexander has two same-day rows (NOP
  "re-signed" a two-way deal, UTA "signed" one) and the check's tie-break chose the wrong one; the ledger holds both. John Konchar was
  traded to MIN, waived, and is on NYK now, but **the feed has no row for the NYK arrival**: the feed is not complete. So the ledger's
  per-player latest move is right almost always, it is not a substitute for the roster snapshot for "who is where", and the daily report's
  roster diff stays as the safety net for moves the feed misses.

## Limitations (stated, not hidden)

* The feed is free text; a new phrasing can be missed. It shows in the `unparsed_sentences` count of the ingest report and `other` rows, and
  the grammar refuses what it cannot read cleanly (a miss shows in the counts) but an unusual phrasing can still produce a wrong role or detail; the independent reviews found and closed several such cases.
* Some trades appear from one team's side only (ESPN does not always publish both). `events` fills the missing side from the counterparty
  when the description names it; otherwise it prints `?`.
* Dates are days in ESPN's time zone (07:00Z stamps), not the moment a deal was reported. For breaking news the ledger is up to a day
  behind; the daily run twice a day is the cadence, `--refresh` is on demand.
* Injury news is not here (that is ESPN status snapshots, ADR 0016): a signing that follows an injury is visible, the injury is not.
* Not tested as a model input, deliberately (D5).

## How to reproduce

```bash
python -m src.ingest.espn_transactions --days 120
python -m src.ops.txn_watch --days 60
python -m src.ingest.espn_transactions --from 2015-07-01 --to 2026-09-25   # full backfill, monthly windows, cached
```
