# ADR 0034: Departed-minutes and playoff-role flags (descriptive)

**Status:** accepted, 2026-10-07
**Code:** `src/features/risk.py` (`minutes_opened`, `depth_flags`, `playoff_flags`), `src/value/risk.py`,
`src/ingest/nba_playoffs.py`; builds on ADR 0016 (risk overlay), 0010/0011 (roster/transaction context).

## Context

Jared McCain (board rank 286, ADP about 138) exposed two gaps. The projection starts from his own past minutes, and (1) OKC lost
about 70 rotation minutes per game (Dort, Joe, Wiggins) that the model cannot see, since the overlay only flagged **star**
arrivals and departures; (2) his regular-season tables stop at the regular season (contract: playoffs excluded), so a late
playoff run as a starter after injuries ahead of him (27 mpg over his last 5 games versus 17 in the regular season) was invisible.

## Decisions

**D1. Both are display flags in `risk_flags`, not projection inputs.** ADR 0010 and 0011 measured team-context features
(including star arrival/departure) as a minutes adjustment over ten real seasons and found no lift (0011 measurably hurt). A
"departed minutes" feature is the same family, and a playoff closing run is a few games in an injury-driven role. Neither is
validated here, so neither moves `proj_fppg`, `proj_gp`, `risk_level`, `risk_gp` or a rank. Wiring them into minutes would need
a point-in-time backtest and its own ADR (PLANNING.md).

**D2. Depth flag (`minutes opened: OKC lost 71 mpg (...), 8 arrived, net 63`).** Per team, minutes per *team* game last season of
players now on a different roster (or on none) are **out**; those minutes are **in** for the team the player is on now. Shown
when net >= 20 mpg, for players projected between 8 and 30 mpg. Limits: unsigned free agents count as departed; rookies and
two-way players are unmeasured; it is not position-aware (guard/wing minutes versus bigs); returning injured players
(Williams, Mitchell) take minutes back, which is why it is a flag, not an adjustment.

**D3. Playoff flag (`playoffs 2025-26: 27 mpg over his last 5 of 14 games, 16.6 FP/g (regular season 17 mpg)`).** New standalone
table `playoff_logs` (`python -m src.ingest.nba_playoffs --seasons 2025-26`, one stats.nba.com request per season; same
accepted-risk stance as ADR 0005 R1, never committed). Shown for players with >= 5 playoff games, a regular-season average under
28 mpg (starters always play more in the playoffs) and closing-run (last 5 games) minutes >= regular season + 4. The closing run,
not the whole-run average, because a role that grew after an injury is invisible in the average (McCain's whole-run mpg is flat).

## Consequences

The flags appear on the board, watchlist and CLI in the live window with the rest of the risk overlay. Missing inputs degrade to a
note, never an error. Read them as "worth a look at ADP", not as a quantified edge.
