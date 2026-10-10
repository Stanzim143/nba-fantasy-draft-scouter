# ADR 0044: Take now or wait? (a two-pick lookahead on the draft simulator)

**Status:** accepted, 2026-10-10
**Code:** `src/value/pick_decision.py`; `simulate_order` / `survival_from_order` and `AvailabilityResult.order` in
`src/value/draft_sim.py`; the "Take now or wait?" panel on the **Next pick** tab (`_decision_panel` in `src/app/draft_board.py`);
glossary group "Take now or wait (ADR 0044)"; tests in `tests/value/test_pick_decision.py` and `tests/app/test_next_pick_tab.py`.
**Follows:** ADR 0040 (draft-availability simulator), ADR 0026 (need-adjusted VORP), ADR 0016 (advisory overlays).

## Context

The Next pick tab (ADR 0040) gives P(still there at my next turn) per player, and `vorp_at_risk` as a sort key, but it does not
answer the question the user faces on the clock: who to take now, given who will probably still be there next time.
`vorp_at_risk` ignores who else would be left. The simulator's draws already hold that information.

## Decisions

**D1. Objective: a two-pick lookahead.** On the clock with `k` opponent picks before the user's next turn, for each candidate `c`
with value `v_c`:

- `next_best(c)` = E[max value among players still there at the next turn | `c` taken now], averaged over the simulator's own
  draws. In each draw `c` leaves the pool, the opponents take the `k` lowest-scoring of the rest, and the best value left is what
  the next pick gets (0 if nobody is left).
- `two_pick(c) = v_c + next_best(c)`. **The recommendation maximises it.**
- `wait_cost(c) = two_pick(c) - max over d != c of two_pick(d)`: the expected two-pick value lost by passing on `c` and playing the
  best other plan, which still gets `c` back in the draws where he survives. Positive only for the recommended pick (its margin);
  zero or below for everyone else. The brief's literal "VORP(c) - E[value at my next pick | I pass on c]", with the pool otherwise
  intact, subtracts the same number from everyone and only re-sorts by VORP, so the two-pick form is used instead.
- `pos_next_best(c)`: the same expectation restricted to `c`'s coarse position (first letter of G/F/C), for the plain-English
  reason ("next-best F then is 40 lower").

The horizon is two picks. Later turns, opponents' roster needs and the change in the user's own need after taking `c` are not
modelled. This is not a full-draft optimisation.

**D2. Value.** Plain `vorp` by default; `need_adj_vorp` (ADR 0026) when the panel's box is ticked, held fixed for the next pick.
`playoff_fp` (fantasy-playoff points, ADR 0039) is shown and orders exact ties of
`two_pick` only; it is never added to the objective. Without the column the code is unchanged.

**D3. Same draws, no re-simulation per candidate.** `draft_sim.simulate_order` returns the draws behind `survival_probabilities`
(results unchanged against the previous code) and `availability_table` keeps them on `AvailabilityResult.order`. The panel's `k` is
the opponent picks between this pick and the next turn, because it models the user's pick explicitly (taking `c`); the shown
`p_avail_next` is the simulator's own, which counts the user's pick as a market pick (ADR 0040). Per draw the panel needs only the
best and second-best value from rank `k` on and the best from rank `k + 1` on. Candidates whose value plus the largest possible
next-pick value cannot reach the top five totals are skipped, which is exact. One rerun on a 700-player pool takes about 0.5 s.

**D4. Noise band.** The simulator's noise is fitted but rough (ADR 0040) and it already draws the `BAND_PARAMS` settings for
`p_next_lo`/`p_next_hi`; it keeps those draws (`AvailabilityResult.band_orders`), and the panel recomputes only the argmax on the
band's first (low-noise) and last (high-noise) settings. If the pick changes, the panel turns into a warning that names the
alternatives and says to treat the top options as close.

**D5. Edge cases.** Not on the clock: the panel says so and waits (it does not plan a future turn, because the players taken before
it are unknown). Back-to-back picks at the turn (`k = 0`): the top two tie exactly at `wait_cost` 0 and the panel says waiting
costs nothing. Last pick: `next_best` is 0 and the recommendation is the highest value left. No ADP, no league sync and the
synthetic demo work through the simulator's own fallbacks (board-rank market order, manual slot).

**D6. Advisory only.** Projections, VORP and the board rank are unchanged; tests assert the board is identical after the panel.

## Consequences

- Validated: the expectation on hand-computed pools, a brute-force check of the vectorised form, exact ties at the turn, the
  tiebreak rule and the edge cases (tests). Not validated: whether following the panel drafts better teams. It inherits every
  weakness of the simulator (noise level, opponents drafting by ESPN ADP), and a two-pick horizon can undervalue positional runs
  that play out over several turns.
- Small `wait_cost` margins (a few VORP) are within the noise; the band check flags the clearest cases only.
- The panel depends on the simulator only through `AvailabilityResult.order` / `band_orders` and `opponent_picks`.
