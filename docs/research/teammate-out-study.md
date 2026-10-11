# Teammate-out usage boost: study and pre-registered evaluation

Question: when a usual rotation teammate is out for a game, do the remaining players gain enough minutes and fantasy
points, predictably enough, to improve the in-season short-horizon projection (waiver streaming, start/sit)?
Pre-season departed minutes are a different effect, already covered by `baseline_depth` (ADR 0035).

Code: `src/inseason/teammate_out.py` (the adjustment) and `python -m src.inseason.teammate_out_eval` (study + evaluation).

## Pre-registration (written 2026-10-10, committed before the evaluation was run)

**Candidate.** The short-horizon projection is the ROS per-game projection `ros_fppg` (ADR 0015 D3, default parameters),
built as of the day before the game. The adjustment adds `beta * gain_fppg` where `gain_fppg` is the existing
redistribution rule (`signals.beneficiaries`: headroom-weighted, same-position boost, blended with observed with/without
minutes once the absentee has missed 3+ games), summed over absent teammates, multiplied by the share of the beneficiary's
games so far in which that teammate played (the with/without part of the baseline is already in the season average), capped
at 36 total minutes, and converted to FP with the player's ROS FP per minute. `beta >= 0` is a single scalar fitted by least
squares (through the origin) on the fit seasons only.

**Absence known before the game.** Rotation = `signals.absent_players` definition (season-to-date 18+ mpg over 8+ games, as of
the day before).
* Primary (R): rotation teammates listed `Out` in the NBA official 5 PM injury report dated the game day (ADR 0033 table
  `injury_reports`; the live analogue is the ESPN `OUT` status the nightly job already pulls). Dates without a report are skipped.
* Secondary (G): rotation teammates absent from their team's most recent game(s) before the game day (game logs only).

**Seasons.** Fit 2016-17..2019-20 (R: 2018-19..2019-20, reports start December 2018). Evaluate 2020-21..2025-26 (6
seasons; 2025-26 reports end 2025-12-22). Nothing is tuned on the evaluation seasons.

**Target and metric.** Next-game FP of every player who plays that game and has 5+ games this season before it. Metric: MAE
over the player-games whose adjustment is non-zero ("affected"); lift = MAE(baseline) - MAE(baseline + adjustment). 95% CI from
the repo's `bootstrap_groups` (2,000 replicates, seed 0), seasons as groups, resampling (team, date) clusters within season.

**Adoption rule (decided on R).** Adopt the adjustment as a projection input if, on the evaluation seasons, the lift is
positive, its 95% CI excludes zero, **and** it is positive in at least 5 of the 6 evaluation seasons. G is applied to
game-log-only absences only if G passes the same rule on its own. Reported but not decisive: next-week mean FP (as of each
Sunday), the bias of the affected residuals, overall MAE.

**Advisory fallback.** If the rule fails but the fitted beta is positive and the evaluation-season mean residual of the affected
players (actual minus baseline) is positive with a CI excluding zero, the effect is real but not useful as an input: keep it as
the existing advisory flag and record that. Otherwise record a null result.

## Results (run 2026-10-10, 2016-17..2025-26, real data)

### Study: the effect is real and grows with the missing minutes

Player-games with a 5+ appearance baseline (own last 10 games); "missing" = trailing mpg of the team's usual rotation
players (6+ of the team's last 10 games at 15+ mpg) absent from this game. Change vs the player's own baseline:

| missing teammate minutes | player-games | minutes | FP | same, no blowouts / B2B / last 10 games: minutes | FP |
|---|---|---|---|---|---|
| none | 86,876 | -1.04 | -0.94 | -0.68 | -0.81 |
| 15-30 | 53,640 | -0.15 | -0.22 | +0.03 | -0.18 |
| 30-50 | 41,766 | +0.68 | +0.78 | +0.85 | +0.83 |
| 50+ | 42,801 | +3.01 | +3.20 | +2.78 | +2.97 |

OLS fitted on 2016-17..2019-20: the gain is mostly in the interaction "missing minutes x the player's headroom (36 minus his
own mpg)" (+0.13 minutes per missing minute at full headroom) plus +0.03 per missing minute in the same position group;
other-group minutes on their own add nothing. Blowouts cost 1.4 minutes, back-to-backs and late season change little, so
the controls do not explain the effect away. Out of sample (2020-21..2025-26) the missing-teammate terms cut the MAE of the
minutes change from 5.29 to 5.10 on games with a teammate missing; of the FP change only from 10.17 to 10.15. Minutes are
predictable, single-game FP is mostly noise.

Confounders, noted not modelled: blowouts and tanking (controlled above, small), back-to-backs (small), coach rest
decisions, and the absence itself being correlated with a team's lost season. The descriptive study uses the realised absence
(who did not play); the evaluation below uses only what was known before tip-off.

### Evaluation (pre-registered)

| | R: 5 PM report OUT | G: game-log absence |
|---|---|---|
| beta (fit seasons) | 0.727 (2018-19..2019-20) | 0.565 (2016-17..2019-20) |
| affected eval player-games | 67,023 (55% of all) | 85,847 (62%) |
| mean residual actual - baseline, affected | +2.08 [+1.97, +2.20] | +1.73 [+1.63, +1.84] |
| mean residual, unaffected | -0.62 | -0.45 |
| **next-game MAE lift (affected)** | **+0.052 FP [+0.025, +0.076]** | -0.014 [-0.034, +0.008] |
| seasons positive | **5/6** (2025-26 -0.045, a partial season of reports) | 3/6 |
| next-week mean FP lift (secondary) | +0.062 [+0.001, +0.131], 4/6 | +0.055 [+0.003, +0.110], 5/6 |
| the rule unscaled (beta = 1) | -0.106, 0/6 | -0.368, 0/6 |
| **pre-registered rule** | **PASS** | FAIL |

Per season (R): 2020-21 +0.037, 2021-22 +0.091, 2022-23 +0.016, 2023-24 +0.053, 2024-25 +0.156, 2025-26 -0.045.
Dose-response on the evaluation seasons (R, quintiles of the raw adjustment): mean residual +0.24, +0.71, +1.21, +2.50,
+5.98 FP against raw adjustments 0.53, 1.32, 2.30, 3.95, 8.99, a slope close to the fitted beta, so the scaled adjustment is
calibrated and not a generic bias correction (unaffected players sit slightly below their baseline).

**Verdict.** Real effect, statistically clear and calibrated, but practically small for a single game: the MAE gain is 0.05
FP on a baseline MAE of 10.0 (about 0.5%), because one game's FP is noisy. It passes the pre-registered rule for
status-driven absences (R) and is adopted there with beta = 0.727 (ADR 0042), for the next game only: the week horizon was
positive in only 4/6 seasons, below the rule. Game-log-only absence (G) fails: a player who
missed the last game often returns, so G stays the existing advisory `inherits_from` flag. The rule's raw magnitude
over-predicts the per-game FP gain (beta < 1) even though the earlier event study found it under-predicts total redistributed
minutes: the FP-per-minute of the extra minutes and absentees' returns both pull it down.

Reproduce: `python -m src.inseason.teammate_out_eval --seasons 2016-17:2025-26 --out DIR` (about 45 minutes, run it alone on
an 8 GB machine), then `--analyze --out DIR`.
