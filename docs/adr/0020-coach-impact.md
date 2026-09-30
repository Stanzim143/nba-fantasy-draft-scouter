# ADR 0020: Coach impact: does the system follow the coach, and who benefits?

**Status:** accepted, 2026-09-25
**Code:** `src/ingest/wiki_coaches.py`, `src/features/coach.py`, `src/models/coach_baseline.py` (`baseline_coach`),
`src/backtest/coach_report.py`, `src/value/coaches.py`, `src/app/coach_view.py` (the app's "Coaches" tab); additive changes to
`src/models/baseline.py` (a `_build_coach_features` hook and `coach_features` field, mirroring the transactions hook),
`src/models/registry.py`, `src/ingest/preseason_refresh.py` and `src/ops/daily_refresh.py` (a `coaches` step).
**Usage and numbers:** [`docs/coaches.md`](../coaches.md). Follows ADR [0010](0010-roster-context-layer.md) (which listed "coach / system" as
roster context and never built it) and ADR [0017](0017-transactions-tracker.md) (staff moves from the ESPN feed).
**Evidence:** real 2016-17..2025-26 data; commands at the end (`reports/` is gitignored).

## Context

The request: account for how coaches run teams (some play their starters 40 minutes, some push young players, some rest people) and
find out, historically, which systems and kinds of players benefited. PLANNING listed "coach / system" under roster context but ADR
0010 built only lagged team pace and role share.

## Decisions

### D1. Head coach per team-season from Wikipedia, opening coach first
Team-season infoboxes (already cached for ADR 0011; CC BY-SA, read-only, polite) list the head coach and, on a mid-season change,
every coach in order with a tag. `team_coaches` has one row per coach per team-season with `is_opening` (the first listed: what a
preseason decision can know) and `n_coaches`. All 330 team-seasons parsed; one (2015-16 Cleveland) has no infobox field and a
documented override. 31 seasons had a mid-season change; those are excluded wherever a style must be attributed to one coach.
The live season has no team page: "List of current NBA head coaches" (one request, re-downloaded each run, refuses to write unless
all 30 teams parse) supplies the 2026-27 coaches and start dates. It already shows this offseason's hires that ESPN's transactions
feed misses for Chicago and Orlando (six new head coaches: Chicago, Dallas, Milwaukee, New Orleans, Orlando, Portland).

### D2. What "style" means: five measurable axes from box scores
Per team-season: `star` (top minute-getter's minutes), `top5` (mean minutes of the top five), `depth10` (players over 10 minutes),
`pace` (possessions per game), `three` (three-point share of shots), `young` (minute share of players 23 and under), `old` (31+).
League-relative (each season's league mean removed). Team-seasons under 20 games are dropped.

### D3. The finding: style follows the coach for minutes concentration, depth and shot mix, but not for pace or youth
* Persistence (203 same-coach and 48 new-coach team-season pairs): style correlates 0.59 to 0.70 year to year when the coach stays and
  only 0.14 to 0.61 when he changes (pace 0.61 vs 0.14, star minutes 0.66 vs 0.34, top-five minutes 0.59 vs 0.37, depth 0.59 vs 0.35).
* Transfer (28 team-seasons where a coach with earlier head-coaching seasons takes a new team): this season's team style regressed on
  the coach's earlier style and the team's own last-season style; the coefficient on the coach's style (1 = takes on his style fully),
  with a 95% bootstrap interval and a permutation p-value: top-five minutes +0.43 [0.02, 0.70] (p 0.004), rotation depth +0.42
  [0.03, 0.78] (p 0.017), three-point share +0.32 [0.13, 0.63] (p 0.018), star minutes +0.58 [-0.10, 0.88] (p 0.001: probably, not
  certainly), **pace +0.07 [-0.28, 0.45] (p 0.35), youth share -0.08 [-0.57, 0.33] and veteran share +0.10 [-0.48, 0.55] do not follow.**
  An independent review caught that the first version of this test regressed the style *change* on (coach style minus last style),
  which puts last season's style on both sides and inflates every slope through mean reversion (pace looked like +0.42 with an
  interval excluding 0; it is 0.07). The pace persistence contrast above (0.61 vs 0.14) is real but is not evidence that a coach
  carries a pace. So "some coaches play starters 40 minutes" is real and portable; "some
  coaches develop young players" is not established: how much a team plays its youth is mostly the roster.

### D4. Who benefits: lasting coach effects on kinds of players are small and mostly not distinguishable from zero
For each player-season, actual minus the baseline's own walk-forward projection (minutes, fantasy points per game, games); predictor =
the same coach's earlier mean residual for the same archetype (young 23-, veteran 31+, star = top two on the team by last-season
minutes, bench), shrunk, any team, seasons before the row only. 12 tests: veteran minutes +0.45 [0.04, 0.85], veteran FPPG +0.52
[0.15, 0.84] and bench minutes +0.25 [0.01, 0.46] clear zero (about one expected by chance); young players, stars and games played do
not (young minutes -0.26 [-0.68, +0.14]). Suggestive of "coaches who trust veterans keep doing it", not strong enough to act on,
and it comes from the same ten seasons that suggested it.

### D5. The projection layer was built and did not help: `baseline_coach`, not recommended
Additive minutes adjustment for players whose team has a new opening coach (not the one it finished last season with): fitted by
weighted least squares from the coach's expected style shift (top-five and star minutes, rotation depth, damped by his number of
earlier seasons) times the player's last-season minutes weight, plus a "new coach" flag; the intercept is fitted but never applied;
capped at 4 minutes; a first-time head coach has no shift and only the flag. Point-in-time: a coach's prior uses only earlier seasons,
the target season contributes only its opening coach (a perturbation test pins that future coach rows change nothing). Team of a player
before the season is resolved as in ADR 0011 (dated arrival, else last season's team); for the live season the newest roster snapshot.
Real ten-season walk-forward (`--ablate baseline,baseline_coach`): Spearman -0.0005 [-0.0012, +0.0001], won 5/10 seasons; top-50 hit
+0.002 [-0.010, +0.008]; MAE (total FP) -0.41 [-0.91, +0.12]: **no significant change**. Same verdict as ADR 0010, 0011 and 0013.
Likely reasons, stated as hypotheses: the baseline's lagged minutes already carry the previous coach's system for the 90% of
players on teams whose coach did not change, and about 28 informative coaching changes in ten seasons is too little to fit five
coefficients. The layer is registered and kept, not used on the board.

### D5b. What the board does instead
Descriptive, like the ADR 0016 risk overlay: `python -m src.value.coaches` and the app's **Coaches** tab list each new head coach,
who he replaces, his career style versus the league and the expected style shift in plain words ("tighter rotation (top five +1.6
min)"), and the ranked players (board top 150) under a new coach. Nothing changes a projection.

### D6. Staff (assistants, front office): not analysable with the data available
ESPN's feed records assistant appointments irregularly (76 in eleven years; none in 2020, one in 2025), Wikipedia infoboxes leave the
assistant field empty, and no compliant complete source exists (Basketball-Reference, Spotrac and similar forbid automation, ADR 0005).
Head-coach, general-manager and executive changes are in the ledger (ADR 0017, `python -m src.ops.txn_watch --staff`) and shown, not
modelled. This is a data limit, stated here rather than filled with a guess.

## Limitations

* The opening coach is credited for the whole season; the 31 mid-season changes are excluded from persistence and transfer but
  counted for the projection layer's training rows as the opening coach's. 2015-16 Golden State lists Kerr first although Luke Walton
  coached the first 43 games (39-4).
* Style is estimated from box scores and confounded with roster quality and injuries; the transfer test is the safeguard (the roster a
  coach inherits is not his).
* Coach identity is by normalised name; a coach listed under two spellings would split. None found among 75 opening coaches.
* 2026-27 coaches are Wikipedia's list as of the fetch; `python -m src.ingest.wiki_coaches --current 2026-27` refreshes it.

## How to reproduce

```bash
python -m src.ingest.wiki_coaches --seasons 2015-16:2025-26 --current 2026-27
python -m src.backtest --ablate baseline,baseline_coach --seasons 2016-17:2025-26 --n-boot 500 --out reports/coach_ablation
python -m src.backtest.coach_report --players reports/real_.../players.parquet     # persistence, transfer, archetypes, profiles
python -m src.value.coaches --board-csv reports/daily/draft_board_baseline_offseason_debut.csv
```
