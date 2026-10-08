# ADR 0037: Off-court flags and hand-written player notes, advisory only

**Status:** accepted, 2026-10-09
**Code:** `src/features/offcourt.py`, `src/ingest/espn_offcourt.py`, `src/ingest/espn_news.py` (`_pull_offcourt`), `src/features/risk.py` (`build_overlay`), `src/value/risk.py`, `config/player_notes.yaml`; tests `tests/features/test_offcourt.py`.
**Follows:** [0036](0036-live-injury-news.md) (same advisory-flag route), [0033](0033-season-intervals-and-injury-labels.md) (reason-labelled injury reports), [0016](0016-risk-overlay.md) D4.

## Context

Question: should a star's off-court absences (contract or trade dispute, e.g. Giannis 2025-26) change a projection? Findings, 2026-10-09:

* The 2025-26 injury reports list Giannis's absences as injuries only (no Personal Reasons / Not With Team); the archive covers 61 game dates, so a later
  off-court element cannot be ruled out.
* The archive does carry off-court categories (Personal Reasons ~1,300 lines, Not With Team ~1,500, Trade Pending 315, suspensions). ADR 0033 tested
  reason-label features: ~1% lower total-fantasy-point error, no change to top-of-board accuracy, not adopted. The minutes-rebound check (PLANNING.md)
  found no systematic under-projection after short, low-minute seasons.
* A star sitting out over a contract or trade dispute has a handful of examples across ten seasons and no structured source, so any adjustment would
  be untestable.

## Decisions

**D1. No projection, risk-level or games-haircut effect.** Everything here is display text in `risk_flags`, like the news flag (ADR 0036 D2).

**D2. Injury-report flag.** Per player, the distinct game dates listed Out last season (newest season before the board's) for an off-court reason:
personal reasons, not with team, trade pending, suspension (`offcourt.report_kind`; "Not With Team" lines that are really G League reconditioning or
covid isolation are excluded). Shown as `7 games out for personal reasons in 2025-26` when at least `MIN_REPORT_GAMES` (3), so a single routine absence is not flagged.

**D3. General-news flag.** `python -m src.ingest.espn_offcourt` (also run best-effort by every `espn_news` pull, so the scheduled news job archives it)
stores ESPN's general NBA news articles per tagged athlete in `offcourt_news` (key article id + ESPN athlete id, `first_seen` / `last_seen`; the feed
holds only ~50 articles, so history exists only through repeated pulls). Keyword screens on headline + blurb (`NEWS_KINDS`: trade request, holdout,
contract dispute, leave of absence, suspension) run at read time, so better keywords apply to stored history. The newest article per kind within 60 days is
shown as `off-court news: trade request, Sep 12 ("headline...")`; the headline is included so a keyword false positive is visible.
An article is attached only to athletes it is about: all of them when it tags at most 3, else those named in the headline (a 30-player draft-board article
must not flag 30 players). A failed second pull never fails the injury pull.

**D4. Notes file.** `config/player_notes.yaml` (`notes: [{player | player_id, note, date?}]`), shown as `note: ...`. Names match case- and accent-insensitively
against the board; no match or an ambiguous one is reported in the board's risk notes and skipped, never guessed. The shipped file is an empty template.
The file is the record; git history is its archive.

## Consequences

Keyword matching is coarse (it can miss a story phrased differently and can match a retrospective); it is a screen for the reader, not a measurement.
Contract-extension talks flag as "contract dispute" because the feed does not distinguish routine talks from a standoff. After a season or two of
`offcourt_news` plus the injury-report archive there is enough history to test whether any of it belongs in the model; until then it stays advisory.
