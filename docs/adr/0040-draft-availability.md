# ADR 0040: Draft-availability simulator (P(still there at my next pick))

**Status:** accepted, 2026-10-10
**Code:** `src/value/draft_sim.py`, `src/value/draft_sim_calibrate.py`; the **Next pick** tab in `src/app/draft_board.py`; glossary group "Draft availability (ADR 0040)"
in `src/app/glossary.py`; tests in `tests/value/test_draft_sim.py`.
**Follows:** ADR 0007 (ESPN ADP), ADR 0009 (draft board app), ADR 0025 (live sync), ADR 0026 (need-adjusted board, same advisory pattern).

## Context

The board knows the ranking, who is drafted and the user's picks, but not whether a target will survive until the user's next turn.
In this 14-team snake (`config/league.yaml`: 13 managers plus a placeholder team) the gap between turns runs from 0 to 26 picks.

The `adp` table holds one number per player (ESPN `averageDraftPosition`, no spread). This league's own earlier drafts are public:
the league's `status.previousSeasons` lists 2022 to 2026, and `fetch_draft_detail` (read-only, cached in the data dir, never
committed) returns all five (10, 10, 16, 12 and 13 teams). ESPN ADP exists for the first four; 2025-26 ADP is FantasyPros gap-fill.

## Decisions

**D1. Monte Carlo of the league drafting by ADP with noise.** Each simulated draft gives every available player
`score = market_adp + sd * z`, `z ~ N(0, 1)`, and in a share `skip` of drafts pushes him past every pick (the league passes on him:
injury news, poor points-league fit). Picks take the lowest scores in order. The target survives when fewer players score below
him than the picks up to the last opponent pick before the target: the user's own picks in between take someone else, not him
(fixes a bias of about +0.03 to +0.06 from counting opponent picks only). Targets share the draws, so P(after) <= P(next).
2,000 draws, seed 0: deterministic, about 0.3 s for 300 players with the band.

**D2. Noise fitted on the league's own drafts:** `sd = min(25, 2 + 0.5 * adp)`, `skip = 0.2`, the best pooled Brier score on a
grid (sd_min 2/6, sd_frac 0.1-0.5, cap 15/25/40, skip 0-0.2) over decision points every 3 picks, gaps of 3 to 26 picks, and the 30
lowest-ADP players still available (`python -m src.value.draft_sim_calibrate`). Leave one season out (Brier, lower is better):

| held out | fit | old 1 + 0.2 x ADP, no skip | constant |
|---|---:|---:|---:|
| 2021-22 | 0.1745 | 0.1749 | 0.2189 |
| 2022-23 | 0.1931 | 0.2232 | 0.2014 |
| 2023-24 | 0.1915 | 0.1967 | 0.2210 |
| 2024-25 | 0.1896 | 0.1952 | 0.2224 |

Held-out calibration is still off at the ends: predicted 0.22 happened 35% of the time, predicted 0.96 happened 90%. Every fitted
season chose `skip = 0.2`, the grid edge, so the true pass rate may be higher: 16 to 34 of each draft's top-N ADP players were
never drafted at all. On the 2025-26 draft (FantasyPros ADP, not fitted) the old default scored better (0.150 vs 0.163).

**D3. Uncertainty band.** `p_next_lo` / `p_next_hi` are the min and max of `p_avail_next` over the fit and three other settings
(`BAND_PARAMS`: the old default, and two near-best fits). Sensitivity, fresh draft, P(available after k picks):

| setting | k 13, ADP 10 | ADP 20 | ADP 30 | k 26, ADP 10 | ADP 20 | ADP 30 |
|---|---:|---:|---:|---:|---:|---:|
| fit (2, 0.5, 25, 0.2) | 0.55 | 0.80 | 0.88 | 0.21 | 0.43 | 0.65 |
| old (1, 0.2, 25, 0) | 0.21 | 0.92 | 0.99 | 0.00 | 0.17 | 0.77 |
| (2, 0.3, 15, 0.2) | 0.37 | 0.79 | 0.94 | 0.19 | 0.30 | 0.59 |
| (6, 0.5, 40, 0.2) | 0.66 | 0.80 | 0.87 | 0.30 | 0.53 | 0.70 |

**D4. Market order.** ESPN ADP, ties broken by board rank (ESPN compresses its tail near 140). A player with no ADP goes after every
player with one, in board order. A board with no ADP (synthetic demo) uses the board rank, and the tab says so.

**D5. Targets and slot.** On the clock, the targets are the user's following two turns, otherwise his next two; snake numbers
from the slot, `teams` (board override, else config: 14) and `rounds = roster.size` (13). The slot comes from, first match wins: a
manual entry on the tab; the ESPN league snapshot's `draftSettings.pickOrder` with the user's team id ($ESPN_TEAM_ID, the live-sync
sidebar, or `nightly --set-team`; on 2026-10-10 this resolved to slot 11); the overall number of the user's first recorded pick.
The current pick is picks marked so far + 1, with a manual override.

**D6. Advisory only.** The tab adds `market_adp`, `p_avail_next`, `p_next_lo`, `p_next_hi`, `p_avail_after` and
`vorp_at_risk = max(vorp, 0) * (1 - p_avail_next)` (a sort key, not an expected loss). Projections, VORP and rank are untouched; a
test asserts the board is unchanged.

## Consequences

- The numbers are rough: the fit improves on a guess modestly in three of four held-out seasons and a lot in one, and stays
  overconfident near 0 and 1. Read the band, and treat a gap of 0.1 between two players as noise.
- Not modelled: the placeholder team, which ESPN autodrafts by its default rank (`draftRanksByRankType`), not by ADP; traded picks;
  need-driven runs on a position. ESPN ADP pools every league type, while this league is H2H points, which is part of why picks drift
  from ADP (median 10 picks earlier than ADP for ADP 48-100 here). The past ADP values are ESPN's season-final averages, not the
  draft-day values. 2022-23's picks are all flagged `autoDraftTypeId` 3 (meaning undocumented); that season fits worst.
- Recalibrate after the 2026-10-17 draft by adding season 2027 to `--seasons` once ADP for it is in `adp.parquet`.

## Addendum 2026-10-13: the skip is gated by ADP (rule fixed before the run)

**Found:** clicking through the app on the real board showed the uniform skip (D2: 20% of drafts pass on a player entirely, applied to
every player) giving Jokic (ADP 1.4) a 21% chance of lasting ten picks, against about 4% without the skip. The 20% was fitted on the
relevant mid-round players and never checked at the top. At the actual first decision (picks 1-10 gone, slot 11, next turn #18) the skip
changes nothing (Flagg 0.61 with and without), so only the pre-draft view of the top players is off.

**Change:** `skip` applies only to players whose market ADP is above `skip_from`, fitted with the other noise parameters (grid:
`skip_from` 0/15/30/45, `skip` 0/0.1/0.2/0.3, `sd_min` 2/6, `sd_frac` 0.3/0.5, `sd_max` 25, the neighbourhood of the first fit). The
random draws are unchanged (`skip_from` only masks them), so existing seeds give the same draws for ungated players.

**Rule (written before the run):** adopt the gated skip with the pooled-best `skip_from` if (a) the leave-one-season-out Brier over
all rows is within 0.001 of the uniform skip's or better, and (b) the Brier on rows with market ADP <= 20 is lower than the uniform
skip's in at least 3 of the 4 held-out seasons. Otherwise keep the uniform skip and label the top-of-draft odds as overstated.
