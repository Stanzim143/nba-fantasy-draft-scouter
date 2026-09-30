# ADR 0016: Offseason gaps: stash and undrafted debutants, origin proxies, draft-day risk overlay

**Status:** accepted, 2026-09-24
**Code:** `src/ingest/nba_profiles.py`, `src/ingest/espn_status.py`, `src/features/debutants.py`, `src/features/risk.py`,
`src/models/debutants.py`, `src/models/rookie_origin.py`, `src/value/risk.py`, `src/backtest/debutants.py`,
`src/backtest/rookie_origin.py`, `src/backtest/preseason_availability.py`; additive changes to `src/models/baseline.py`
(`prior_rows`, `rookie_adjust`), `src/models/registry.py`, `src/value/board.py`, `src/value/breakouts.py`,
`src/ingest/preseason_refresh.py` (opt-in extra steps), `src/app/` (loader, state, the "Debutants & risk" tab).
**Usage:** [`docs/offseason.md`](../offseason.md). Follows ADR [0012](0012-offseason-layer.md), which listed these as open.
**Evidence:** real walk-forward runs, commands at the end (`reports/` is gitignored).

## Context

ADR 0012 shipped the preseason roster update and left four gaps: (1) five rostered debutants drafted in earlier years
(Sorber, Marković, Toohey, Diop, Biberovic) were named on the board's notes but not projected; (2) undrafted signees,
two-way and Exhibit-10 players (49 on the 2026-09-24 roster snapshot) were absent; (3) nothing used what players did before
the NBA; (4) nothing flagged preseason injuries or team-context changes. The draft is 2026-10-17 (NZ morning; fixed 2026-09-26, ADR 0014).

## Decisions

### D1. Debutants: the draft-slot prior, evaluated, flagged low confidence (`baseline_debut`, `baseline_offseason_debut`)

A **debutant** has no NBA game before the target season and is not in that season's draft class. `stash` = drafted earlier with a
real pick; `undrafted` = everything else. `DebutantBaselineProjector` subclasses the baseline: veterans and the current class are
**exactly** its rows (tested to 1e-12) under `baseline_debut`; it only appends debutant rows with `projection_class`, `p_play`, `confidence = "low"`.
`baseline_offseason_debut` stacks the ADR 0012 layer on top, so Summer League and preseason evidence adjust debutants too. This is not exact for veterans: debutant rows enter the offseason layer's training residuals. Measured 2026-09-25 on the 739 players shared with the plain boards: `baseline_offseason_debut` vs `baseline_offseason` differs for 186, max |dFPPG| 1.07, 1 above 1 FPPG; vs `baseline` also 186 differ, max 3.24, 23 above 1 FPPG (the offseason layer's own adjustment, which is intended, plus the training-residual effect).

* Candidates, live: the newest `roster_snapshots` day of the target season. Candidates, backtest: the players who appeared in
  the season's **preseason** games (the camp roster: same Exhibit-10 and two-way population, public before opening night).
* **Lines**: the rookie slot prior (`FittedBaseline.prior_rows`, a pure refactor of the rookie path) at the player's own pick,
  age and position group. **Games**: `p_play * share`. Undrafted: minutes multiplier, games share and a small logistic
  `p_play` (preseason games share and minutes, Summer League minutes) fitted on earlier seasons' camp participants. Stash:
  the prior as it is; only an international-development multiplier, shrunk toward 1.
* **`STASH_P_PLAY = 0.90` is an assumption, not an estimate.** The index only has a draft record for people who eventually
  played, so a backtest can only recognise a stash as "drafted earlier" if he debuts: every historical stash analogue played
  (28 of 28 in the scored seasons). The true rate for a rostered stash is not identifiable from history. Rostered draftees
  overwhelmingly play at least some games, but 0.90 is a judgement.

### D2. Leakage rules for candidates

Draft year and pick, country, size and birthdate are facts fixed before the season. **Not used**: whether a person has an
index row at all (only future NBA players do), and `from_year >= season` (a future-debut marker). `from_year` before the
game-log window is used only to *exclude* someone who played earlier. Everything fitted uses earlier seasons only; a
future-invariance test (bend-only-the-future, `assert_projector_ignores_future`) covers the projector.

### D3. Origin information (gap 3): what is legitimate, what is not

* **Used (stats.nba.com, same accepted-risk stance as ADR 0005 R1):** `playerindex` COUNTRY and COLLEGE (already downloaded and
  cached; **no new request**), giving `origin` (`usa`/`intl`) and `prev_org_type` (`college`/`other`: a club, prep school or none;
  "college" = an organisation at least three USA-born players list, a documented heuristic). `draftcombinestats` (12 requests):
  measured height, wingspan, reach, jumps, agility, ~57% attendance. New tables `player_profiles` and `draft_combine`
  (`python -m src.ingest.nba_profiles`).
* **Not built, with reasons.** NCAA / college production and international-league production: (i) **Sports-Reference /
  Basketball-Reference**: terms forbid automated access that hurts performance and building a competing database, 20 requests/minute
  with a day's lockout, robots disallow gamelog/splits (ADR 0005 rejected it; college and international pages share the network
  and terms). (ii) **Spotrac, HoopsHype, RealGM, Pro Sports Transactions**: rejected in ADR 0005 for robots/terms. (iii) **ESPN**
  college / league payloads: the accepted R2 covers our own league and the NBA player universe at low volume; extending it to
  ~600 historical draftees would need a name-matching layer to NBA ids for players ESPN's NBA feed does not list, and ESPN's terms
  forbid scripted access. That was not accepted for this volume. (iv) **Wikipedia / Wikidata** are CC-licensed and API-friendly but
  carry college and league per-season statistics only as free text in infoboxes, not a machine-readable table with dates
  (there is no field-level structure to parse point-in-time-safely). (v) **Barttorvik**: its robots.txt (read 2026-09-24) disallows the JSON data files and database pages and blocks ClaudeBot, anthropic-ai and
  most other named crawlers with `Disallow: /`; **stats.ncaa.org** answered 403 to a plain fetch of its robots.txt (bot-blocked). Neither is a
  source we may automate. **Euroleague / national leagues**: no terms granting automated bulk access were found, and no single league covers
  every place a stash plays.
  Conclusion: **no compliant, machine-readable, historically complete production source exists for the 10-season backtest**.
  Country, previous-organisation type, draft age, size and combine measurements are the strongest legitimate proxies and are used.
* **Result** (`python -m src.backtest.rookie_origin`, 470 rookies 2018-19..2025-26, direct-target regressions with vs without):
  origin flags lower total-FP MAE by 10.7 (95% CI 8.3 to 13.2, 8 of 8 seasons), combine by 9.5 (6.8 to 12.3), both by 15.7 (11.6 to
  19.8); FP/game MAE 6.12 -> 5.91. About 3.5% of rookie error. Intervals resample rookies independently (they share seasons), so they
  understate uncertainty. **Whole-league**, through the shipped multiplier form (`baseline_origin`, harness): total-FP MAE
  422.6 -> 422.2 (paired +0.41, CI +0.05 to +0.78, "improves" but trivial), Spearman total FP 0.784 -> 0.782 (paired -0.0015, "hurts",
  CI excludes 0 by a hair). Rookies are about a tenth of the board; the gain is real for them and invisible in the league.
  `baseline_origin` is registered, **not recommended as a default**; the debutant models leave the rookie prior unchanged.
  For debutants themselves, the one origin effect the evidence supports is international development (correlation of the stash residual
  with `intl` -0.30, n=28): stashes from abroad produced less than their slot implied (8.7 vs 11.8 FP/game for international vs USA stashes in an exploratory look at 36 debutants; the shipped multiplier is shrunk toward 1).

### D4. Draft-day risk overlay (gap 4): flags, not projections

`src/features/risk.py` / `src/value/risk.py` attach `risk_level` (`""`/`watch`/`high`), `risk_flags`, `risk_gp_haircut` and `risk_gp` to the
board and watchlist for the live season. **`proj_gp` and every projection are unchanged.**

* **ESPN `injuryStatus`** (600-player snapshot; 12 OUT, 56 DAY_TO_DAY on 2026-09-24): archived daily by `espn_status`
  (`espn_status_snapshots`, a dated append-only table; ESPN keeps no history, so this is the only way to get a point-in-time record).
  Not backtestable. The per-status haircut (OUT 20%, DAY_TO_DAY 3%, SUSPENDED 10%; see the amendment below) is an explicit **assumption**, and a status whose last news is
  older than 45 days is shown but not discounted. The NBA injury-report PDFs (ADR 0005 D2) were probed for preseason dates in 2024 and
  2025: **absent** until opening week (first present dates 2024-10-22, 2025-10-20; every earlier date tried answered 403), so they cannot
  inform a draft on 2026-10-17 (NZ morning; fixed 2026-09-26, ADR 0014) and were not ingested.
* **Preseason absence** (a rotation veteran on a roster whose team played 3+ preseason games and who played none, or fewer than half): shown
  as a warning. `src.backtest.preseason_availability` tried to size it on 10 seasons and **cannot**: the history has no roster snapshots, so
  an injured veteran and one no longer under contract look identical (78% of the flagged group played no regular-season game at all, mostly
  people who left the league). The measured -24 games residual is that artefact, so `PRESEASON_HAIRCUT = 0`: it warns, it does not discount.
  Live it is shown only for players on the current roster snapshot. `roster_snapshots` accumulates from 2026-09-24, so next year it can be tested.
* **Roster context** (new team, star arrived at or left his team): from the roster snapshot against last season's final team. ADR 0010 and 0011
  measured roster and transaction context as inputs and found no lift, so these flags are descriptive.

> **Amendment, 2026-09-27 (documentation only; no code change).** D4 originally listed only OUT and DAY_TO_DAY. The code (`ASSUMED_STATUS_HAIRCUT` in `src/features/risk.py`) has always handled a third
> ESPN status, `SUSPENDED`: it is flagged `ESPN suspended`, gets `risk_level` `watch` (the same level as DAY_TO_DAY; only OUT is `high`) and a **10%** games haircut in `risk_gp`. Like the other two it is an
> assumption, not an estimate, and a suspension whose last news is older than 45 days is shown as stale and not discounted. Pinned by `tests/features/test_risk.py`.

### D5. Interface changes, all additive

`preseason_refresh` keeps its three steps, order and default; `--with-extras` (or `--only`) adds `profiles` and `status` after them. It also stops
crashing on a Windows console when a player name has an accent (the exact failure seen on 2026-09-24: `Marković` under cp1252). The board CLI
and app carry `projection_class`, `p_play`, `risk_*` when present. The daily automation (ADR 0014) now runs the `profiles` and `status` steps after `adp` (commit 1b04928), and its default boards
add `baseline_offseason_debut`, the board that carries the debutants and undrafted signees.

## Results (real data, 2026-09-24)

| Question | Answer |
|---|---|
| Debutant treatment vs omitting them (`python -m src.backtest.debutants`, 853 camp participants, 8 seasons) | All: model MAE 93.5 vs omit 70.7 vs prior 225.1 vs class mean 108.2. Zeros dominate (mean actual 71 FP; 65% never play). The model beats the raw prior by 131.6 FP (CI 124 to 139) and the class mean by 14.7 (10 to 20); it does **not** beat "predict 0" on MAE for undrafted players (+27.9, CI 24 to 32) because zero is the right answer for two thirds of them. It is unbiased (+0.7 FP) where omission is biased by -71, and ranks them (Spearman 0.37; omit has no ranking). |
| Undrafted play probability | AUC 0.689; mean predicted 0.303 vs actual 0.331 (825 camp participants). |
| Stash (28 who debuted) | Model MAE 303 vs omit 430 (paired -126, CI -319 to +21: not distinguishable, n=28), Spearman 0.65 vs 0.09 for the class mean; the plain slot prior is as good as anything fitted (paired +0.6, CI -22 to +28). FP/game 11.8 predicted vs 10.9 actual. |
| League-wide harness, `baseline_debut` vs `baseline`, 10 seasons | Paired lift on shared players exactly 0. Mean over the larger universe: MAE total FP 380.0 vs 422.6 (more players scored, coverage misses fall from ~57 to ~17 a season), MAE games 17.8 vs 19.2, Spearman total FP 0.773 vs 0.784 (the added low-value rows enlarge the universe; top-K hit rates unchanged). |
| League-wide harness, `baseline_offseason_debut` vs `baseline_offseason` | Because debutant rows also enter the offseason layer's training residuals, the shared players' numbers move slightly: paired Spearman total FP +0.0020 (CI +0.0017 to +0.0023, 8/10 seasons, `improves`), paired MAE total FP -1.37 (CI -1.55 to -1.20, `hurts`), top-50/100 hit rates unchanged. Over the whole scored universe: MAE total FP 372.2 vs 413.1, games MAE 17.8 vs 19.2, Spearman 0.771 vs 0.781 (larger universe). Net: no reason to prefer it for the 690 veterans, every reason to use it for coverage of debutants; `baseline_offseason` stays the model whose numbers earlier results were measured on. |
| Origin proxies on rookies | See D3: about 3.5% lower rookie error, no league-wide effect. |
| Preseason-absence and ESPN-status haircuts | Not validated; shown as flags only (D4). |

## Consequences

* At the time of writing (2026-09-24) the board had about 790 players for 2026-27 (roughly 690 veterans, 52 rookies, 5 stashes, about 50 undrafted; counts move with each roster refresh). Every named debutant is projected and flagged low confidence.
* Undrafted signees carry small totals by design (median expectation about 1 in 3 to play, few games); they rank far down the board, as the evidence says they should.
* The stash projections are the slot prior (Sorber, a 2025 first-round pick, is a real 735 FP projection on a 40-game season; Diop, Biberovic, Marković, Toohey 250 to 390). They are the least certain rows on the board; `p_play = 0.90` is a judgement.
* Not closed: college and international production (no compliant source), any historical validation of injury or preseason-absence effects.

## Reproduce

```
python -m src.ingest.nba_profiles                       # player_profiles, draft_combine (65 requests once, cached)
python -m src.backtest.debutants --out reports/debutants
python -m src.backtest.rookie_origin --out reports/rookie_origin
python -m src.backtest.preseason_availability --out reports/preseason_availability
python -m src.backtest --model baseline_debut --benchmark baseline --seasons 2016-17:2025-26 --out reports/harness --run-id debut
python -m src.backtest --model baseline_origin --benchmark baseline --seasons 2016-17:2025-26 --out reports/harness --run-id origin
python -m src.backtest --model baseline_offseason_debut --benchmark baseline_offseason --seasons 2016-17:2025-26 --out reports/harness --run-id offseason_debut
python -m src.value.board --season 2026-27 --model baseline_offseason_debut --adp ~/dev-data/nba-fantasy-2026/processed/adp.parquet --out reports/board_debut.csv
```
