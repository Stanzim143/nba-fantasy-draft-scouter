# ADR 0042: Teammate-out boost in the short-horizon projection (status-driven absences only)

Status: accepted, 2026-10-10

## Context

ADR 0015 D5 flags who inherits an absent rotation teammate's minutes (`signals.beneficiaries`) and states that nothing there
feeds a projection. ADR 0035 covers minutes departed in the *offseason*. The open question was whether the *game-level*
effect, a teammate out tonight, is predictable enough to change in-season numbers: streaming and start/sit.

## Decision

1. **Evidence first, pre-registered.** `docs/research/teammate-out-study.md` holds the rule (committed before the run) and the
   results. Baseline = ROS `ros_fppg` as of the day before; adjustment = the existing redistribution rule's minute gain,
   scaled by the share of games the absentee has played (the season average already holds the others), capped at 36 minutes,
   times ROS FP per minute, times a fitted `beta`. Fit 2016-17..2019-20 (reports from 2018-19), evaluate 2020-21..2025-26.
2. **Adopted for status-driven absences.** With rotation teammates listed `OUT` on the NBA 5 PM injury report, the adjustment
   cuts next-game FP MAE on affected player-games by 0.052 FP [0.025, 0.076], positive in 5/6 evaluation seasons, and is
   calibrated (residuals rise with the adjustment at the fitted slope). `beta = 0.727` (`teammate_out.BETA_STATUS`).
3. **Not adopted for game-log-only absence** (missed the last game, no status): -0.014 [-0.034, +0.008], 3/6. That stays the
   advisory `inherits_from` flag of ADR 0015.
4. **Where it applies** (`teammate_out.status_boost`, live statuses = the ESPN `OUT` status the nightly job and the waiver
   finder already load, only when the snapshot is within 3 days of `as_of`):
   * waiver streaming for **this** matchup week: `week_exp_fp = expected games x (ros_fppg + out_boost_fppg)`, for the
     candidate and for the drop; next week is unchanged (statuses that far ahead were not tested);
   * nightly lineup hints (ADR 0018): the week and slate expected FP use the same boosted per-game value;
   * a new `out_boost_fppg` column in both, and "teammate-out +FP/g" in the nightly streaming table.
   ROS totals, ROS ranks, best adds and the trade analyzer are unchanged: the boost is short-horizon only.

## Consequences

* The effect is real but small per game (0.5% of single-game MAE); it changes close streaming and start/sit calls, not
  rankings. Its size is about 2.5 FP per game for an affected player, up to 6-9 for the biggest openings.
* ESPN `OUT` is assumed to match the NBA report's `Out` used in the evaluation; the 2025-26 report season is partial (to
  2025-12-22) and was the one losing evaluation season. Recheck after a full 2026-27 season of nightly snapshots with
  `python -m src.inseason.teammate_out_eval`.
* The redistribution rule itself is unchanged; `beta < 1` means its raw FP magnitude is too high per game.
* Leakage: `tests/inseason/test_teammate_out.py` wrecks every game on or after the target date and asserts the baseline,
  the absences and the adjustment do not move.
