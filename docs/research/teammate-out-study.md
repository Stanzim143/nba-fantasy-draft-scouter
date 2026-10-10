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

## Results

(filled in after the run)
