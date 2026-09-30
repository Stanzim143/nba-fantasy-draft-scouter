# ADR 0024: Load management: does an absence-shape proxy predict next-season games beyond the baseline?

**Status:** accepted, 2026-09-26 (the pre-registration below was written before any outcome was computed; results, decisions and limitations follow it)
**Code:** `src/features/load_management.py` (the proxies), `src/backtest/load_report.py` (the study), tests in `tests/features/test_load_management.py` and `tests/backtest/test_load_report.py`.

## Context

"Load management" (a healthy player sitting a game for rest, most visibly the second night of a back-to-back, or late in a season
with nothing at stake) lowers games played. The 2023-24 Player Participation Policy and the 65-game award threshold were meant to
curb it. The question for the draft board is whether a player's *rest-like* absences in the last season say something about next
season's games (and total fantasy points, `proj_fppg x proj_gp`) that the current projection does not already contain.

## What data exists, and what does not

Exists (all local, nothing is downloaded): `game_logs` (every game a player appeared in, with minutes, 2015-16..2025-26),
`team_games` (every team game with its date, so back-to-backs are derivable), `players` (birthdate, from_year), `player_season_bio`
(age), `offseason_logs` (preseason / Summer League box scores), `espn_status_snapshots` (three days, 2026-09-24..26, current status
only), `schedule_games` (2026-27 only, dates, no history). The existing injury layer (ADR 0006) already turns the games-missed
shape into `longest_streak_frac`, `n_streaks`, `age_residual`; the risk overlay (ADR 0016) adds ESPN status and preseason absence.

**Not available, and the whole difficulty:** any *reason* for a missed game. There is no DNP-rest / DNP-injury / suspension /
personal label (the NBA injury-report PDFs start 2018-12-19 and are not ingested, ADR 0005; ESPN's status feed is a snapshot, not a
history, and has no "rest" reason). A game the player did not appear in is indistinguishable from an injury, a coach's decision
or a trade artefact. So load management can only be a **proxy**: the shape of the absences. The proxy is contaminated by minor
injuries by construction (a one-game absence on a back-to-back is what a sore ankle also looks like), which is why the study asks
whether it predicts anything, rather than assuming it means rest.

## Pre-registered design (fixed before any outcome was inspected)

Only the games-played histogram near 65 was looked at while writing this (no projection, error or outcome): it rises steadily
through 60-72 with no cliff at 65 (counts at 64 / 65 / 66 / 67 / 68 for players averaging >= 28 mpg, excluding 2019-20 and 2020-21:
22 / 27 / 28 / 34 / 40). The 65-game threshold is therefore carried only as a descriptive marker.

**Protocol.** Same as ADR 0021: walk-forward over target seasons 2016-17..2025-26, projection and every feature from
`History.until(tables, s)` (seasons `< s`), season `s` supplies only the outcome. Models under test: `baseline` (primary) and
`baseline_injury` (secondary, which already has the streak features). Outcomes are signed *actual minus projected* GP and total FP
(zero-game targets count 0, the harness rule), positive = under-projected.

**Proxies (season `p = s-1`, primary team's schedule in date order; `src/features/load_management.py`).**

* *interior run*: consecutive missed team games starting after game 1 and ending before the last game.
* `iso_n`: games missed inside interior runs of length <= 2, rescaled to an 82-game schedule (`iso_n82`) because 2019-20 and
  2020-21 were shorter. **Primary proxy.**
* `rest_n`: interior runs of length exactly 1 that are the second night of a back-to-back (previous team game the day before), also
  per 82 (`rest_n82`). Secondary.
* `iso_frac = iso_n / games missed` (0 with no absence). Secondary.
* `near65`: `65 <= GP <= 67` in a season of >= 80 team games. Secondary, descriptive.
* `hm_vet`: age >= 31 and >= 30 mpg at `p` (an age x minutes cell). Secondary.

**Universe.** Projected players who appeared in `p`, veteran (a season before `p`), single team in `p`, `>= 20` mpg and `>= 41` GP
in `p` (rest needs a player healthy enough to be rested; below half a season the absences are mostly injury).

**Primary test.** Per target season fixed effects plus controls, OLS: `err = season + b * z(iso_n82) + g1 f + g2 age + g3 age^2 +
g4 mpg + g5 proj_gp/target_games`, `f` = prior GP fraction, `z` standardised over the pooled universe, so `b` is the effect of one
standard deviation more isolated absences. Outcome `err_fp` (total FP; the metric that matters) and `err_gp`. 95% percentile
bootstrap, 2000 resamples, rows resampled within season (the coefficient refit each time). The other proxies are each fitted in the
same model in place of `iso_n82` and reported as they fall (no selection). Descriptive: mean error by `iso_n` bin (0-1, 2-3, 4-5,
6+). Reported, not tested: the effect before and after 2023-24 (participation policy).

**Out-of-sample test.** For each target season from 2019-20 on, fit the same regression (no season effect, unknowable in advance)
on the errors of all earlier target seasons (`err_gp` on the primary proxy and the controls), correct `proj_gp` by the prediction
(clipped to `[0, target games]`), convert to FP with `proj_fppg`, and compare absolute error to the uncorrected baseline. The paired
difference (baseline minus corrected absolute error, positive = the correction helps) gets the same stratified bootstrap.

**Decision rule (fixed).** The proxy is *incremental* only if ALL hold: (1) the primary `b` on `err_fp` has a bootstrap CI excluding
zero; (2) the season-by-season `b` on `err_fp` has the pooled sign in at least 7 of the 10 target seasons; (3) the out-of-sample
total-FP MAE improvement has a CI above zero. Incremental: register a not-default projector variant and validate it against the
baseline in the ablation, never change the default. (1) and (2) but not (3): the proxy is a *descriptive association* and only an
advisory, clearly labelled flag is justified. (1) failing: null, no projection change and no flag that implies a forecast.

## Result (real walk-forward, 2016-17 through 2025-26, projector `baseline`)

Reproduce: see the last section (about two minutes). 5,386 projected player-seasons; the pre-registered universe (veteran, single team, >= 20 mpg
and >= 41 GP the season before) has **n = 1,830**, 173 to 193 a season. Over that universe the baseline over-projects by 4.4 GP and 77 total FP
(MAE 14.1 GP, 520 total FP), so everything below is read against that uniform over-projection, which the controls and season effects absorb.

**Primary test** (`iso_n82` per +1 SD, about 2.9 more games missed in short absences; controls: prior GP fraction, age, age^2, mpg, projected
fraction; season fixed effects; bootstrap by season):

| outcome (actual - projected) | per +1 SD [95% CI] |
|---|---|
| games played | **-1.4 [-2.4, -0.4]** |
| total fantasy points | **-53 [-88, -20]** (rule (1): CI excludes zero) |

Rule (2): the season-by-season total-FP coefficient has the pooled sign in **9 of 10** target seasons (-63, -88, -62, -30, -5, -66, -4, **+19**,
-112, -74 FP for 2016-17 to 2025-26). Rule (3): the out-of-sample correction (targets 2019-20 to 2025-26, n = 1,263) **raised** MAE: +0.8 GP
[+0.5, +1.1] worse and +14 total FP [+6, +22] worse (reported in the tool as improvement -0.8 [-1.1, -0.5] GP and -14 [-22, -6] FP). **Verdict under the
fixed rule: descriptive association** (1 and 2 pass, 3 fails). It is not incremental, so no projection changes and no projector variant is registered.

With `baseline_injury` (whose availability model already has the streak features) as the model under test the picture is the same and only
slightly smaller: -1.1 GP [-2.1, -0.1], -47 total FP [-83, -13] per SD, the sign in 7 of 10 seasons, the out-of-sample correction again worse
(-13 FP [-21, -5] MAE improvement), verdict descriptive association. So the streak features do not explain it away.

**Every other proxy, as it fell (each in its own regression):**

| proxy | mean (sd) | GP per +1 SD [95% CI] | total FP per +1 SD [95% CI] | seasons with pooled sign |
|---|---|---|---|---|
| `iso_n82` (primary) | 3.1 (2.9) | -1.4 [-2.4, -0.4] | -53 [-88, -20] | 9/10 |
| `rest_n82` (one-game absence on the 2nd night of a back-to-back) | 0.7 (1.2) | -0.4 [-1.3, +0.5] | -3 [-36, +30] | 6/10 |
| `iso_frac` | 0.35 (0.35) | -0.8 [-1.8, +0.1] | -26 [-60, +7] | 5/10 |
| `near65` (65 to 67 GP) | 7% of rows | +0.0 [-0.9, +0.8] | +4 [-26, +32] | 5/10 |
| `hm_vet` (age >= 31 and >= 30 mpg) | 9% of rows | +0.7 [-0.4, +1.8] | +26 [-14, +67] | 6/10 |

Only the primary proxy is distinguishable from zero. The one that is closest to what "load management" means, a healthy player missing the
second night of a back-to-back, carries **no** signal (-3 FP [-36, +30]), and neither does the 65-game marker. Error by number of isolated
absences last season (raw counts, `iso_n`): 0-1 (n = 657): -3.4 GP, -41 FP; 2-3 (556): -5.2 GP, -92 FP; 4-5 (307): -3.5 GP, -64 FP; **6+ (310):
-6.1 GP [-8.3, -4.1], -140 total FP [-221, -66]**. By era: targets before 2023-24 -1.4 GP [-2.6, -0.3], -52 FP [-92, -15]; 2023-24 to 2025-26 (the
participation-policy era, n = 548) -1.1 GP [-2.9, +0.6], -49 FP [-118, +14]: same size, wider, not distinguishable from zero on its own.
The GP histogram shows no cliff at 65 (players averaging >= 28 mpg: 22 / 27 / 28 / 34 / 40 at 64 / 65 / 66 / 67 / 68 GP).

**Post-hoc diagnostics** (added after the out-of-sample test came back negative; not part of the verdict). (a) The pre-registered out-of-sample
test scores the whole correction (intercept and controls included) against the uncorrected baseline, and errors are skewed by players who never
play, so a mean-fitted correction can lose on MAE while helping the mean. Isolating the proxy (the same walk-forward fit with and without it)
gives -0.0 GP [-0.1, +0.1] and -3 total FP [-6, -0] of MAE gain: the proxy adds nothing out of sample beyond the controls (and, at the FP margin,
very slightly hurts). (b) A player with 6+ isolated absences (n = 310, 17% of the universe) lands **-2.6 GP [-5.1, -0.2] and -98 total FP [-183,
-13]** further under the projection than an otherwise similar player with fewer. The flag threshold and the advisory size below come from this
pre-registered top bin and this figure.

## Verdict

1. **Not incremental; the projection is unchanged.** Correcting `proj_gp` with the proxy does not
   lower out-of-sample error (it raises it slightly), its effect is small (about 1.4 GP, about 10% of the FP MAE per SD), and it is not present in every season (one
   positive year, one near-zero pair in 2020-21 and 2022-23).
2. **What the association is.** More short interior absences, not fewer, go with a *lower* next season than projected, in both models and in 9
   of 10 seasons. That is what fragility looks like (minor knocks recurring), the same direction as ADR 0021's spread-absence controls, which the
   baseline over-projects. It is not what a rest story predicts: pure rest (`rest_n82`) has no effect and the 65-game marker has none. The data
   therefore say "frequent short absences predict a somewhat worse availability than the season total suggests", and say nothing about
   *why*. Calling the proxy "load management" would be a label the data cannot support.
3. **Advisory only.** Because (1) and (2) of the decision rule hold, a small descriptive flag is allowed: `lm_flag` for a veteran with 6 or more
   games missed in 1-2 game interior absences, an advisory of **-2 GP** (about the -2.6 estimate, rounded toward zero because its CI ends at -0.2)
   and that times `proj_fppg` in total FP, labelled "cause unknown, rest or minor injury" and "a judgement, not in the projection". On the 2026-27
   board 54 of 185 universe players carry it (29%, more than the 17% of the study rows, because 2025-26 was a high-absence season; the rule counts
   games, not a percentile), with 14 isolated absences at the top (Gui Santos, Terance Mann), Klay Thompson (13), Michael Porter Jr. (13), Al
   Horford (12) and Jaylen Brown (10) among them.

## Decisions

* **D1. Analysis plus advisory flag; no projection change.** `proj_gp`, `proj_total_fp`, `vorp`, `rank`, `risk_level`, `risk_gp_haircut`,
  `risk_gp` are read for nothing except sizing the advisory games and are never written (byte-identity tests, as ADR 0023).
* **D2. Pre-registered proxies, universe, primary test and rule** (above); the other proxies and every diagnostic are reported whole.
* **D3. Post-hoc items are labelled.** The isolating out-of-sample comparison, the 6+ contrast and the era split were added after seeing results
  and do not enter the verdict; the flag threshold is the pre-registered top `iso_n` bin, not a tuned cut-off.
* **D4. Point-in-time.** The proxies use only the season before the target (`prior_load_features`); the out-of-sample fit for each target uses only
  earlier target seasons. Tests scramble every row of season `>= s` and separately extend the tables; the features and flags are identical.
* **D5. No new data source.** Nothing is downloaded; a labelled rest signal would change this (see below).

## Columns and app

`python -m src.value.board` (and the daily refresh CSVs, and the app loader) add: `lm_flag` (`short-absences` | blank), `lm_iso_n`, `lm_rest_n`,
`lm_gp_risk_adv` (0 or -2), `lm_fp_risk_adv` (that times `proj_fppg`), `lm_validation` (constant `advisory_descriptive_not_projection`), and the
sentence in `risk_flags`. The app gets a **Many short absences only** checkbox on the Best available and Full board tabs, the `lm_flag` column
in the board tables and a table on the Debutants & risk tab. Code: `src/value/load_flag.py`, `src/app/{state,loader,draft_board}.py`,
`src/value/board.py`.

## Tests

`tests/features/test_load_management.py` (interior runs, the <= 2 rule, the second-night definition from real dates, multi-team, no look-ahead),
`tests/backtest/test_load_report.py` (universe filters, a planted effect is recovered and a null is not, the decision rule as a conjunction, the
walk-forward correction trains only on earlier seasons, the report renders), `tests/value/test_load_flag.py` (threshold at 5 vs 6, runs of exactly 2,
3-game runs and lead blocks not counted, traded and debutant seasons excluded, the -2 rule and its floor, byte-identical projections through
`build_board` and `board.main`, point-in-time, degradation), `tests/app/test_load_flag_app.py`.

## Limitations

* **No reason label.** The proxy mixes rest, minor injury, coach decisions and illness. The back-to-back part is the closest thing to rest and it
  is null, but a one-game back-to-back absence can also be a sore ankle; a player rested *and* injured is uncountable.
* **Team schedule attribution** as in ADR 0006 (a mid-season trade misattributes games; the universe is single-team, so traded players are never flagged).
* **Universe** is rotation veterans (>= 20 mpg, >= 41 GP); the study says nothing about bench players or players who missed half a season.
* **Ten target seasons, 1,830 rows**; the bootstrap is over players within season, so season-level shocks are only visible in the per-season table
  (one positive year of ten). 2019-20 and 2020-21 are shortened and rescaled by 82 / L, an approximation.
* **The advisory is a judgement**, inside the CI of a post-hoc estimate; `hm_vet` (age x minutes) and `near65` are single cells of a five-proxy set.
* **The flag rate moves with the league.** Counts, not percentiles: 29% of the 2026-27 universe is flagged against 17% in the study rows.
* **2023-24 onward** (participation policy, 65-game rule) is only three targets; the effect there is the same size but not significant alone.

## What data would settle it

A per-game inactive *reason* (rest / injury / suspension / personal) with dates: ESPN's or the NBA's game-day inactive lists carry "Rest" and are
published in real time but are not archived in any source ingested here (ADR 0005); the NBA injury-report PDFs list "Injury/Illness" versus
"Rest" but start 2018-12-19 and were not ingested. With two or three seasons of such labels the study would run again on labelled rest absences
instead of a proxy, with the same protocol. The 2026-27 season itself can accumulate the ESPN status snapshots (`python -m src.ingest.espn_status`) to build a history of
current status going forward, though that feed is an injury status, not a rest label.

## Reproduce

```
python -m src.backtest.load_report --seasons 2016-17:2025-26 --out reports/load_2026-09-26
python -m pytest tests/features/test_load_management.py tests/backtest/test_load_report.py tests/value/test_load_flag.py tests/app/test_load_flag_app.py
```

`load_report.md`, `proxies.csv`, `bins.csv` and the row-level `study_frame.parquet` are written to `--out` (gitignored). `--compare` selects the
other projectors (default `baseline_injury`). Only local data is read.

## Consequences

* A reusable, leakage-tested proxy study exists and can be re-run in season or with labelled data.
* The board gains a clearly labelled advisory flag; its numbers, and every projection, VORP and rank, are unchanged.
* Open: whether *labelled* rest absences predict anything is untestable with the current data.
