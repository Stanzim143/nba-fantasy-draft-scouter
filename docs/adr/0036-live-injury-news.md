# ADR 0036: Live ESPN injury-news feed (`injury_news`), advisory flags only

**Status:** accepted, 2026-10-09
**Code:** `src/ingest/espn_news.py`, `src/features/risk.py` (`news_flags`), `src/value/risk.py`; tests `tests/ingest/test_espn_news.py`.
**Follows:** ADR 0016 D4 (status flags are advisory), ADR 0033 (labelled injury archive), ADR 0005 R2 (source stance).

## Context

`espn_status_snapshots` archives only a status code and a "last news" timestamp, so a board flag could say "ESPN day-to-day" but not
whether that was a sore wrist or a torn ACL. ESPN's public site API (`/apis/site/v2/sports/basketball/nba/injuries`, no cookies, one
request for all teams) carries the injury type, body part, estimated return date and RotoWire's news blurb.

## Decisions

**D1. Pull it live and keep every note.** `python -m src.ingest.espn_news` fetches the feed (always bypassing the HTTP cache), maps
players to NBA ids through `player_id_map`, and merges into `injury_news` keyed by (note id, status). Rows keep `first_seen` and get
the newest `last_seen`, so the table is a history even though ESPN exposes only current notes. `--watch MINUTES` repeats the pull
(minimum 5, one request per pull) and prints only new items; `--show` filters stored news by days, team or player.

**D2. Advisory only.** The newest current note per player becomes an informational `news: ...` entry in `risk_flags`. It changes
neither `risk_level` nor the games haircut, and no projection. There is no history to back-test a news effect on, and the table now
being archived is what makes that possible later; until then a hand-set haircut would be an assumption on an assumption.

**D3. A missing table degrades quietly.** `compute_risk` adds a note and skips the flag when `injury_news` has not been built.

## Consequences

News is only as fresh as the last pull; nothing runs it automatically yet (see `PLANNING.md`). Off-court items other than injuries
(suspensions appear as statuses; legal or personal news does not) are not in this feed.
