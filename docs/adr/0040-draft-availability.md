# ADR 0040: Draft-availability simulator (P(still there at my next pick))

**Status:** accepted, 2026-10-10
**Code:** `src/value/draft_sim.py`; the **Next pick** tab in `src/app/draft_board.py`; glossary group "Draft availability (ADR 0040)"
in `src/app/glossary.py`; tests in `tests/value/test_draft_sim.py`.
**Follows:** ADR 0007 (ESPN ADP), ADR 0009 (draft board app), ADR 0025 (live sync), ADR 0026 (need-adjusted board, same advisory pattern).

## Context

The board knows the ranking, who is drafted and the user's picks, but not whether a target will survive until the user's next turn.
In a 14-team snake the gap between turns runs from 0 (at the turn) to 26 opponent picks, so "can I wait on him?" is the most common
draft-day question the board could not answer.

What the data allows: the `adp` table holds one number per player (ESPN `averageDraftPosition`). The raw ESPN payload has
`percentOwned`, `percentChange` and `auctionValueAverage` but no spread or standard deviation, and the repo has no record of actual
draft positions (the league has not drafted yet; no historical draft results are ingested). ESPN's 2026-27 ADP covers 178 players and
compresses at the tail: 25 players sit between 139 and 140.

## Decisions

**D1. Monte Carlo of opponents drafting by ADP with noise.** In each simulated draft every available player gets
`score = market_adp + sd * z`, `z ~ N(0, 1)`, and the opponents take the lowest scores in order. A player is available at a target
pick when fewer than `k` players score below him, where `k` is the number of *opponent* picks between the current pick and the target.
The user's own picks in between are assumed not to take him (the question is "if I pass on him"). All targets share the same draws,
so P(after) <= P(next) always. Default 2,000 draws, seed 0: deterministic, about 0.1 s for 300 players and 3,000 draws.

**D2. The noise is an unvalidated heuristic:** `sd = min(25, 1 + 0.2 * adp)` picks (about 1.6 at ADP 3, 7 at ADP 30, 21 at ADP 100).
Nothing in the repo can calibrate it (see Context). It is the documented default until real draft results exist; the one cheap
calibration later is this league's own completed draft (actual pick vs ADP), which live sync already records.

**D3. Market order.** ESPN ADP, ties broken by board rank. A player with no ADP is placed after every player with one (and after the
current pick), in board-rank order. A board with no ADP at all (synthetic demo, or no ADP ingested) uses the board rank as the proxy,
and the tab says so.

**D4. Targets.** If the user is on the clock, the targets are his following two turns; otherwise his next two. Snake pick numbers come
from the slot, `teams` (board override, else `config/league.yaml`) and `rounds` = `roster.size` (13; IR is not drafted). Traded picks
(`pick_trading: true`) are not modelled.

**D5. Slot resolution**, first match wins: a manual entry on the tab; the ESPN league snapshot's `draftSettings.pickOrder`
(`processed/espn_league/<league>_<season id>.json`, written by `python -m src.ingest.espn_league`) with the user's team id
($ESPN_TEAM_ID, the live-sync sidebar, or `nightly --set-team`); the overall number of the user's first recorded pick. Otherwise the tab
asks for the slot. The current pick is "picks marked so far + 1", with a manual override for a missed pick.

**D6. Advisory only.** The tab adds `market_adp`, `p_avail_next`, `p_avail_after` and `vorp_at_risk = max(vorp, 0) * (1 - p_avail_next)`
(a sort key, not an expected loss: it ignores who else would be there). Projections, VORP and rank are untouched; a test asserts the
board is unchanged.

## Consequences

- Validated: the snake arithmetic, determinism, monotonicity and edge cases (tests). Not validated: the noise level, and the
  assumption that opponents draft by ESPN ADP at all (managers draft by need, by ESPN's own default rankings and by news). Treat the
  probabilities as rough, especially past ADP 100, where ESPN's tail compression makes the ordering itself weak.
- With no ADP, the numbers describe how fast *our* board order would empty, which is a weaker question.
- Opponents are independent of roster need; a run on a position is not modelled.
