# ADR 0015: In-season tools (schedule awareness, rest-of-season projection, trade analyzer, waiver finder)

**Status:** accepted, 2026-09-24
**Code:** `src/inseason/` (`weeks`, `schedule`, `ros`, `ros_eval`, `lineup`, `trade`, `signals`, `waivers`, `context`),
`src/app/inseason_view.py`, `src/app/pages/inseason.py`. **Tests:** `tests/inseason/`, `tests/app/test_inseason_view.py`.
**User guide and evidence:** [`docs/inseason.md`](../inseason.md). **Builds on:** ADR 0005 D1 (ESPN, anonymous, cached, low
volume), ADR 0008 (league sync), ADR 0009 (Streamlit app), ADR 0011 D1 (standalone tables). The nightly refresh that keeps
these inputs fresh is ADR 0014 and is not part of this decision.

## Context

Roadmap phase 5 listed four in-season deliverables as not started: schedule awareness, a trade analyzer, a waiver /
free-agent finder and (with them) a rest-of-season projection to feed them. The season starts 2026-10-20, so nothing here could
be run against a live season; everything is built and tested on synthetic data and smoke-tested on the real 2025-26 season "as
if" it were mid-season (only games up to a cutoff date). One finding shaped the schedule design: on 2026-09-24 ESPN still
returned a **placeholder** for the league's `matchupPeriods` (23 periods of one scoring period each, while
`finalScoringPeriod` was 167), so the league's real weeks are not available yet.

## Decisions

### D1. `schedule_games` is a standalone table (ADR 0011 D1 pattern), one cached ESPN request

One row per game: `season, game_id, scoring_period, game_date, home/away team id and NBA abbreviation, source, time_tbd`.
Not in `src/contracts.py`, not in `History`. A future game in it is not leakage (a schedule is known in advance); it holds no
results. Live season: ESPN `proTeamSchedules_wl` through `CachedHttpClient` (one GET, cached, offline-safe), dates from the
scoring-period day counter (an 8 pm ET tip is the next UTC day), ESPN pro-team ids mapped to NBA ids by a checked table.
Finished seasons: rebuilt from `team_games` (used by the backtest and the smoke run; five 2025-26 games carry no home flag and
get a deterministic home). The 2026-27 pull is 1,200 games, exactly 80 per team; the two missing per team are the NBA Cup games
ESPN has not resolved, and the weekly table reports each week's league mean so the shortfall is visible.

### D2. Matchup weeks: ESPN's periods when real, otherwise a derived calendar, and it says which

`src/inseason/weeks.py` rejects the placeholder and otherwise uses ESPN. The fallback derives Monday-Sunday weeks from opening
night, merges the All-Star week into the week before it, and cuts to the league's period count (20 regular + 3 playoff). For
2026-27 that gives 23 periods ending 2027-04-04, which matches the league's own `finalScoringPeriod` 167 exactly; it is still an
approximation (ESPN may merge a different week) and every output names the source (`espn` / `derived`).

### D3. Rest-of-season projection = per-player shrinkage of the preseason projection toward season-to-date

Three separate shrinkage averages, `posterior = (k*prior + n*sample)/(k + n)`: per-minute production (n = minutes), minutes per
game (n = games), availability (n = team games). Games remaining come from the schedule. Pseudo-counts are chosen on the train
seasons of a walk-forward evaluation (2016-17..2020-21) and reported on 2021-22..2025-26 (`python -m src.inseason.ros_eval`).
Everything read from the current season is filtered to `game_date <= as_of`; the prior is projected from
`History.until(season)`. Leakage tests perturb everything after `as_of` and require identical output. **Result** (details in
`docs/inseason.md`): on the out-of-sample seasons the blend beats preseason-only in every season and every metric, and beats
season-to-date-only in every season on ROS-total MAE and per-game error but only marginally on rank correlation (+0.008
Spearman). The honest summary: in-season, season-to-date data is a much better predictor than the preseason projection, and the
blend adds a real but modest amount on top of it; the optimum sits at the edge of the grid for two of three pseudo-counts (the
sample dominates minutes and availability quickly).

### D4. Roster value is a slot-aware matching, and both sides of a trade are first made a full roster

`src/inseason/lineup.py`: maximum-weight matching of players to the league's ten starting slots with the real eligibility rules
(G = PG/SG, F = SF/PF, UTIL = anyone; empty slots allowed), plus a bench worth `w` times its ROS total with `w` from the same
derivation the value engine uses for replacement level, and IR players valued like the bench without using a roster spot. The
trade analyzer values `roster - give + get` against `roster` after fitting each to capacity: a freed spot is filled by the best
available free agent (greedy by marginal lineup value; a generic replacement-level player when the pool is unknown), an overfull
roster drops its least valuable player (named). So a 2-for-1 is not credited with a phantom zero and a 1-for-2 is charged for
the drop. This is an expected-volume model (daily lineups and game-by-game injuries are not simulated), the same logic as the
draft board's VORP; the result carries a stated tolerance (1.5% of roster value) inside which the verdict is "roughly even".

### D5. Waiver finder: ROS value over the drop, flags for a human, streaming by games in the week

Free agents are every NBA-active player no league team has rostered (before the draft: everyone). Each candidate is scored by the
best add/drop against your lineup with D4's value function. Flags are rising minutes / usage (last 5 games vs earlier, with a z
test), an absent rotation teammate whose minutes he is likely to inherit (game-log absence, or ESPN `OUT` from the cached
snapshot), and his own injury status. The redistribution rule is headroom-weighted (36 minus current minutes) with a same-position
boost and blended with observed with/without minutes once the absent player has missed 3+ games; weighting by current minutes was
tried first and is anti-correlated with reality (Spearman -0.13 on 10 seasons), so it was rejected. **These are alerts, not model
inputs**: ADR 0010 and 0011 found roster-context features do not lift the model, and nothing here feeds the projection.
Streaming ranks free agents by expected FP in the coming matchup week (games left x availability x FPPG), with back-to-backs and
off-night games; an ESPN `OUT` player is excluded from streaming.

### D6. Surfaces: CLIs under `python -m src.inseason.<x>` and a Streamlit page that loads on demand

`schedule`, `ros`, `ros_eval`, `signals --eval`, `trade`, `waivers`. The app gains `src/app/pages/inseason.py` (Streamlit
multipage; `draft_board.py` is untouched); nothing is computed until "Load in-season data" is pressed, so the draft board loads
exactly as fast as before, and the page reads only local data and ESPN caches (offline-safe). Logic lives in
`src/app/inseason_view.py`, tested without Streamlit, plus a headless `AppTest` run of the page.

## Limitations and unresolved

* Real league rosters are untested against real data (the draft has not happened); the ESPN-team path is covered by fixtures
  and the mock-draft path is used for demos. Re-verify after the draft, including that `player_id_map` maps every rostered player.
* Matchup weeks are derived until ESPN publishes real periods; re-run `python -m src.ingest.espn_league` (ADR 0008) after the
  season starts so the real `matchupPeriods` are cached and picked up automatically.
* The ROS pseudo-counts reached the grid edge for games and availability; a finer or wider grid was not explored.
* The replacement-level and bench-weight machinery assumes a full 13-team league of active rosters; a league with
  many idle rosters would have a stronger free-agent pool than replacement level implies.
* Injury status is a snapshot and only as fresh as the last ESPN pull.

## Consequences

Phase 5 gets its trade, waiver, schedule and ROS deliverables (the nightly refresh and league sync remain ADR 0014 / 0008). A new
standalone parquet table `schedule_games` exists; no shared file (`pyproject.toml`, `config/league.yaml`, `src/contracts.py`)
changed. `scipy` is used directly (it is already installed as a dependency of scikit-learn).
