# ADR 0026: "Best available by need" — crossing the VORP board with roster construction

**Status:** accepted, 2026-09-28
**Code:** `src/value/need.py` (the scoring model, pure/no Streamlit), `src/app/state.py` (`need_board_view`, the app-facing orchestration), `src/app/draft_board.py` (the "Best available by need" tab). Tests: `tests/value/test_need.py`, `tests/app/test_state.py`, `tests/app/test_glossary.py`.

## Context

The existing board ranks every player by VORP alone (ADR 0009, section 8 of `docs/categories.md`) and the Suggestions tab (`_suggestions_tab`) shows the top 3 remaining players at each position, but neither weighs a suggestion by how much the drafter still *needs* that position. A drafter with three centers already rostered and zero guards gets the same "best available" list as a drafter with the opposite roster. This ADR adds a new view that crosses the ranked board against the user's own drafted roster and `config/league.yaml`'s roster shape, without touching the existing VORP board, its rank, or its columns.

## What "need" means here, structurally

`config/league.yaml` (`league.roster`) is the source of truth:

```yaml
roster:
  size: 13
  starters: {PG: 1, SG: 1, SF: 1, PF: 1, C: 1, G: 1, F: 1, UTIL: 3}
  bench: 3
  ir: 1
```

Five specific ESPN positions, two flex slots (`G` = PG or SG, `F` = SF or PF), `UTIL` (anyone), a bench (anyone), and IR (excluded — it is not an active lineup slot). `league.format` is `h2h_points`: a **points league**, not categories/roto, so there is no notion of punting a stat category here (see "Explicitly scoped out" below).

## The model

**Capacity.** Each specific position's fractional roster "capacity" is its own starter slot, plus an even share of every flex slot that can reach it (a `G` starter splits into 0.5 PG + 0.5 SG), plus an even share of `UTIL` and of the bench (any of the 5 positions can occupy either). For the league above: PG capacity = 1 (starter) + 0.5 (G flex) + 0.6 (UTIL, 3/5) + 0.6 (bench, 3/5) = 2.7; C capacity = 1 + 0.6 + 0.6 = 2.2 (no flex reaches C).

**Filled.** How many of the user's own drafted players are eligible at each specific position (`src.value.positions.eligible_positions`), counting a multi-position player toward every position he qualifies for.

**Need score**, per specific position: `clamp((capacity - filled) / capacity, 0, 1)` — 0 when the league carries no slot there at all, or the position is already fully covered by drafted players; 1 when none of its capacity is covered yet. A player's own need score is the *best* (highest) need score among the specific positions he is eligible at (he only needs to fill one), or 0 for a player whose dataset position is unparseable (the same "never invent a position" rule as `src.value.positions.is_known`).

**Combined score.** `need_adj_vorp = vorp + DEFAULT_NEED_BONUS(40) * need_score`. Additive, not multiplicative, on purpose: `vorp` can be negative (below replacement), and a multiplicative need bonus would push an already-bad player further negative in a way that reads as "more valuable," which is backwards. The raw `vorp` column is always still present and unmodified; `need_adj_vorp` is a second, clearly-separate column and the second sortable ranking (`need_rank`) the spec calls for, not a replacement for the first.

**Empty-roster fallback.** With nothing drafted, every position's capacity is entirely unfilled, so every known-position player's need score is exactly 1.0 — the bonus becomes the *same constant* added to every player's `vorp`. A constant additive shift never changes an ordering, so the need-adjusted board is then rank-identical to the plain VORP board. This is what "sensibly fall back to plain best-available-by-VORP" means in the code (`NeedResult.fallback`), and the app also says so explicitly (a caption on the tab) rather than relying on the reader noticing the numbers happen to agree.

**Why this heuristic and not an exact lineup optimizer.** Solving "which slot would this exact player occupy, given every other drafted player and every remaining flex choice" is an assignment problem. Folding that into the ranking without a validated backtest would repeat the caution this codebase already takes seriously elsewhere — see ADR 0016's positional-scarcity `auto` mode, which only turns on positional VORP replacement when a data-driven check says it is material, and every advisory flag (ADR 0019, 0023, 0024) that is careful never to imply more precision than the data supports. `need_score` and the 40-FP bonus size are **not backtested or empirically calibrated** — there is no outcome to validate a need-weighting scheme against (need is about roster construction preference, not a projection of future production) — so this ships as a transparent, documented heuristic on a *new, additional* view, with the raw VORP board unchanged and always available.

## Explicitly scoped out

* **Category needs.** `is_categories_league(cfg)` is the single gate for this: it returns `False` for the current `h2h_points` config and `True` for a categories/roto `format`. No category-need scoring is implemented — there is no categories league in this data to build or validate it against, and adding an unvalidated per-category weighting on top of an already-unvalidated positional heuristic would compound the guesswork with no way to check it. If the league ever moves to categories, `is_categories_league` is where that computation would be added.
* **Punt strategies.** Not applicable to a points league (nothing to punt; every counting stat converts to the same fantasy-point currency, ADR 0009/`docs/categories.md` section 2).
* **Exact lineup/assignment optimization.** See "why this heuristic" above — a real assignment solve is a larger, separately-justified project.
* **Bye-week/injury-flag interactions.** The need view reuses the same undrafted board (`best_available`) and the same `filter_board` position/tier/search filters as the other tabs; it does not touch `risk_flags`, `return_flag`, `lm_flag` or `contract_flag`, which remain visible on the other tabs untouched. It does not currently expose those flag columns or the returned/short-absence filter checkboxes on the need tab itself (they were left off to keep the new tab's table narrow; adding them is a small, low-risk follow-up, not a behavior change to this ADR's model).

## Columns and app

`src/value/need.py` adds (only inside the new tab's computation, never onto the stored/exported board): `need_score`, `need_adj_vorp`, `need_rank` (reassigned 1..N over the undrafted board sorted by `need_adj_vorp` descending, ties by `vorp` then `player_id`), and, in a small per-position breakdown table, `need_position`, `need_capacity` and `need_filled`. None of these are written by `src.value.board` or the daily-refresh CSVs — they exist only for this app view, computed on the fly from the already-loaded board plus the in-session draft state, so `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp`, `vorp_per_game`, `tier` and `rank` are never touched (same "advisory/derived, never overwrites the projection" discipline as ADR 0016/0019/0023/0024, even though this isn't framed as an advisory flag but a second ranking).

The app gets a new **"Best available by need"** tab (`src/app/draft_board.py:_need_tab`) between Suggestions and Drafted: Position/Tier/Search filters matching the existing idiom, a table sorted by `need_adj_vorp` (columns: `need_rank, rank, player_id, name, position, proj_fppg, proj_gp, proj_total_fp, vorp, need_score, need_adj_vorp, tier, adp`), and a small positional-need table (`need_position, need_capacity, need_filled, need_score`) so the score is never a black box. With an empty roster the tab shows an explicit info message alongside the (rank-identical) table. Every column has a glossary entry and hover tooltip (`src/app/glossary.py`, group "Best available by need (ADR 0026)"), consistent with commit f06e65a's discipline.

## Tests

`tests/value/test_need.py`: `is_categories_league` (points config false, categories/roto config true, missing format defaults false), `position_capacity` (flex/UTIL/bench spread, zero capacity at a position the league carries no slot for), `position_fill_counts`, `position_need_scores` (open position -> 1.0, partially filled -> strictly between 0 and 1, fully/over-filled -> clamped to 0 not negative, zero-capacity -> 0 not a crash), `player_need_scores` (best of eligible positions, unknown position -> 0), and `compute_need_board` end to end: empty-roster fallback preserves VORP ordering exactly, a non-empty roster lets an open position's player overtake a higher-VORP player at an already-filled position, `vorp` itself is never mutated, the categories flag is surfaced without being computed, and an empty board does not crash. `tests/app/test_state.py` covers `need_board_view`: fallback + ordering through the app-facing function, and that drafted players are excluded from the need table while still counting toward "filled". `tests/app/test_glossary.py` extends the existing "every board/table column must have a glossary entry" checks to `NEED_TABLE_COLUMNS`/`POSITION_NEED_COLUMNS` and the two new `st.dataframe` call sites (14 -> 16), without weakening any existing assertion.

## Limitations

* **Unvalidated bonus size.** `DEFAULT_NEED_BONUS = 40` FP is a chosen constant, not fit to any outcome; a different league or draft strategy might reasonably want it larger or smaller. It is a module-level constant specifically so it is easy to find and change, not because 40 is known to be right.
* **Demand-side approximation, not an assignment.** `position_fill_counts` counts a multi-position drafted player toward every position he is eligible at, which can understate how "used up" his flexibility really is once an actual lineup is assembled.
* **Positions are the existing coarse approximation.** `src.value.positions.eligible_positions` is already documented as an approximation of ESPN's real games-played eligibility (`docs/categories.md` section 3); this ADR inherits that limitation rather than adding a new one.
* **No category/punt support**, by construction (see "Explicitly scoped out").
* **Independent of the risk/return/load-management overlays.** A need-boosted player who is also risk-flagged is not specially called out on this tab; check the other tabs for that.

## Consequences

* A new, additive "Best available by need" tab exists; the original "Best available", "Full board" and "Suggestions" tabs and every column they show are unchanged.
* Roster-construction preference now has a transparent, testable (if unvalidated) home, instead of living only in the drafter's head.
* Open: whether a need-weighted suggestion actually produces better rosters is untested (there is no outcome to backtest against); treat this as a decision aid, the same caveat every advisory feature in this app already carries.
