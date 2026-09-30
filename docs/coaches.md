# Coaches

How coaches run teams, what changes when a team gets a new one, and the 2026-27 head coaches. Design and evidence: ADR
[0020](adr/0020-coach-impact.md). **Descriptive**: `baseline_coach`, the same information as a minutes adjustment, gave no significant
change over ten real seasons and is not used on the board.

## What to look at before the draft

```bash
python -m src.value.coaches --board-csv reports/daily/draft_board_baseline_offseason_debut.csv
```

Six teams have a new head coach for 2026-27 (Chicago: Tiago Splitter; Dallas: Dusty May; Milwaukee: Taylor Jenkins; New Orleans: Jamahl
Mosley; Orlando: Sean Sweeney; Portland: Micah Nori). For each the tool prints who he replaces, his career style against the league and
what the change implies, then the ranked players under him (37 of the board's top 150 on 2026-09-25). Four of the six are first-time
head coaches, so there is no style to import: for Chicago, Dallas, Orlando and Portland the honest statement is "new, unknown".
Milwaukee (Jenkins: more spread minutes, deeper rotation than Rivers's team) and New Orleans (Mosley: more spread minutes) have a history to read. The app's **Coaches** tab shows the same table. `--all` lists every team.

## What we found (real 2015-16..2025-26 team-seasons; walk-forward 2016-17..2025-26)

* **Some coaches concentrate minutes and it travels.** Team style correlates 0.59 to 0.70 year to year with the same coach and only
  0.14 to 0.61 after a change (pace 0.61 vs 0.14; star minutes 0.66 vs 0.34). When a coach with head-coaching history takes a new team,
  the team takes on part of his earlier style, beyond the style it already had: top-five minutes +0.43, rotation depth +0.42 and
  three-point share +0.32 (bootstrap interval excludes 0 and permutation p < 0.05), star minutes +0.58 (permutation p 0.001 but the
  interval reaches -0.10: probably). **Pace does not clearly follow** (+0.07): pace is not shown to travel with the coach (the
  same-coach vs new-coach persistence gap, 0.61 vs 0.14, hints at some coach effect that this small sample cannot isolate). Only 28 informative coaching changes, so the intervals are wide. Coach profiles (league-relative career means; positive = more than average) are in
  `reports/coach_*/coach_profiles.csv`, e.g. Thibodeau star minutes +2.9 and top-five +2.4 (heavy starters), Popovich and Budenholzer
  about -1.7 and -1.6 (spread minutes), Nurse +2.1.
* **"Develops young players" is not a portable coach trait.** The youth-minute share does not clearly follow a coach to a new team
  (-0.08, interval -0.57 to +0.33); it is mostly the roster.
* **Who benefits, beyond the baseline.** Twelve coach-by-archetype tests: veteran minutes (+0.45), veteran fantasy points per game
  (+0.52) and bench minutes (+0.25) carry over from a coach's earlier seasons; young players, stars and games played do not. With
  twelve tests one false positive is expected, so read it as a lean, not a rule: coaches who leaned on veterans tended to again.
* **As a projection input: nothing measurable.** `baseline_coach` vs `baseline`: Spearman -0.0005 (5/10 seasons won), top-50 hit +0.002,
  MAE -0.41 FP, all "no significant change". The baseline's minutes already reflect last year's system, and about 28 informative coaching
  changes in ten seasons is too few to fit a correction.
* **Assistants and front office cannot be analysed** with compliant data (ESPN's feed records assistant hires irregularly). Head-coach,
  GM and executive moves are tracked: `python -m src.ops.txn_watch --staff`.

## Refresh and reproduce

| What | Command |
|---|---|
| Coaches of completed seasons (cached; no new requests after the transactions ingest) | `python -m src.ingest.wiki_coaches --seasons 2015-16:2025-26` |
| The live season (one request; also a step of the daily refresh) | `python -m src.ingest.wiki_coaches --current 2026-27` |
| Full analysis tables | `python -m src.backtest.coach_report --players <saved players.parquet>` |
| The ablation | `python -m src.backtest --ablate baseline,baseline_coach --seasons 2016-17:2025-26` |

Caveats: the opening coach is credited for a whole season (31 team-seasons changed coach mid-year and are excluded from the style
tests); style is measured from box scores and is confounded with roster quality, which is why the transfer test (a coach moving to a
roster that is not his) is the one to trust.
