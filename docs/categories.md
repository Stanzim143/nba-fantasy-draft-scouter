# Every category, label and column in the app: what it means and how it is computed

A reference for the draft board, its flags, the watchlist, the Transactions and Coaches tabs and the in-season page. Every formula and constant
below was read from the code (`file:line`), not from the ADR prose; where the two disagree the disagreement is listed in
[section 17](#17-code-versus-documentation-discrepancies-found-while-writing-this). Worked examples use the real board built on **2026-09-26**
(`baseline_offseason_debut`, season 2026-27, 13 teams, 790 rows; `reports/daily/draft_board_baseline_offseason_debut.csv` in the primary
checkout) unless another date is stated. Line numbers are those of the current tree (including the ADR 0024 short-absence flag in `src/app/*` and `src/value/board.py`); they drift with edits, the function names do not.

## Contents

1. [The big picture: what is projected, what is only flagged](#1-the-big-picture)
2. [League settings and fantasy scoring](#2-league-settings-and-fantasy-scoring)
3. [Positions and eligibility](#3-positions-and-eligibility)
4. [Projection columns: `proj_fppg`, `proj_gp`, `proj_total_fp`](#4-projection-columns)
5. [The floor / median / ceiling band: `fppg_p10`, `fppg_p50`, `fppg_p90`](#5-the-floor--median--ceiling-band)
6. [Who gets which projection: `projection_class`, `is_rookie`, `confidence`, `p_play`](#6-who-gets-which-projection)
7. [The offseason layer: `offseason_adj`, `uplift`](#7-the-offseason-layer)
8. [Value: replacement level, `vorp`, `vorp_per_game`, `rank`, positional scarcity](#8-value-replacement-level-vorp-rank)
9. [Tiers](#9-tiers)
10. [ADP and `adp_gap`](#10-adp-and-adp_gap)
11. [Risk overlay: `risk_level`, `risk_flags`, `risk_gp_haircut`, `risk_gp`](#11-risk-overlay)
12. [Returned-healthy flag (ADR 0023)](#12-returned-healthy-flag)
13. [Contract flags (ADR 0019)](#13-contract-flags)
14. [Breakouts tab (watchlist)](#14-breakouts-tab-the-watchlist)
15. [Transactions and Coaches tabs](#15-transactions-and-coaches-tabs)
16. [In-season page](#16-in-season-page)
17. [Code versus documentation discrepancies](#17-code-versus-documentation-discrepancies-found-while-writing-this)
18. [Load management: where it fits](#18-load-management-where-it-fits)
19. [Where each tab and page is defined](#19-tabs-and-pages)
20. [Glossary of every column](#20-glossary-of-every-column)
21. [Draft-day live sync (ADR 0025)](#21-draft-day-live-sync-adr-0025)
22. [Best available by need (ADR 0026)](#22-best-available-by-need)
23. [External rankings comparison (ADR 0028)](#23-external-rankings-comparison)
24. [External rankings disagreement flag (ADR 0029)](#24-external-rankings-disagreement-flag)

---

## 1. The big picture

```
history (games before the season)  ->  projection  ->  value (VORP)  ->  rank, tier   ->  the board
                                       proj_fppg x proj_gp = proj_total_fp
overlays that never change a projection or a rank:  risk (ADR 0016)  return flag (ADR 0023)  short-absence flag (ADR 0024)  contract flags (ADR 0019)  ADP gap
```

Two kinds of column live on the board. **Modelled** columns (`proj_*`, `vorp`, `rank`, `tier`, the p10/p50/p90 band) come from a projection that was
walk-forward backtested. **Overlay** columns (`risk_*`, `return_*`, `lm_*`, `contract_*`, `adp*`) are shown beside the projection; each was either tested and
found not to help as a model input, or could not be tested, so none of them moves a number (ADR 0010, 0011, 0016, 0019, 0020, 0023, 0024). Every overlay
row says so: `return_validation = advisory_judgement_not_projection`, `lm_validation = advisory_descriptive_not_projection`, `contract_validation = unvalidated_display_only`.

The rank is by projected **total** fantasy points above replacement, never FPPG alone (`src/app/state.py:214`).

## 2. League settings and fantasy scoring

Source: `config/league.yaml` (ESPN league "Example League", H2H points, 14 teams per the live ESPN league on 2026-09-30 (13 managers + 1 placeholder; was 13 on 2026-09-23), ADR 0008).

| Setting | Value | `league.yaml` |
|---|---|---|
| Teams | 13 | line 8 |
| Starting slots | PG 1, SG 1, SF 1, PF 1, C 1, G 1 (PG or SG), F 1 (SF or PF), UTIL 3 = **10** | lines 26-34 |
| Bench / IR | 3 / 1 (IR is extra, not counted in the 13) | lines 35-36 |
| Roster size | 13 | line 25 |
| Draft | snake, 120 s a pick, 2026-10-17 morning NZ | `draft:` block |

**Points formula** (weights `config/league.yaml:58-69`, applied by `fantasy_points`, `src/value/points.py:9`, and its DataFrame twin
`fantasy_points_frame`, `src/value/frame.py:11`; column mapping `STAT_COLUMN_MAP`, `src/contracts.py:62`):

```
FP = 1*PTS + 1*REB + 2*AST + 4*STL + 4*BLK - 2*TO + 2*FGM - 1*FGA + 1*FTM - 1*FTA + 1*3PM
```

No double-double, triple-double or threshold bonuses exist. Consequences that explain most of the rankings:

- a made two = 2 (PTS) + 2 (FGM) - 1 (FGA) = **+3**; a made three = 3 + 2 + 1 (3PM) - 1 = **+5**; a missed shot = **-1**;
- a made free throw = 1 + 1 - 1 = **+1**; a missed free throw = **-1**;
- steals and blocks are worth 4 each, an assist 2, a turnover -2, so defensive stats and playmaking weigh far more than in points-only scoring.

Worked example, Nikola Jokic's 2026-27 per-game projection (`proj_*` columns, all per game):

| Stat | Projected | Weight | FP |
|---|---|---|---|
| PTS | 23.954 | 1 | 23.954 |
| REB | 11.343 | 1 | 11.343 |
| AST | 9.101 | 2 | 18.202 |
| STL | 1.303 | 4 | 5.212 |
| BLK | 0.712 | 4 | 2.848 |
| TO | 3.031 | -2 | -6.062 |
| FGM | 8.783 | 2 | 17.567 |
| FGA | 15.461 | -1 | -15.461 |
| FTM | 4.867 | 1 | 4.867 |
| FTA | 6.002 | -1 | -6.002 |
| 3PM | 1.521 | 1 | 1.521 |
| **`proj_fppg`** | | | **57.988** |

Limitation: the scoring keys are read from the config, so the same code scores any league; `fantasy_points` raises rather than scoring a missing stat as 0
(`points.py:11-13`).

## 3. Positions and eligibility

The dataset carries one coarse position string per player (`G`, `F`, `C`, `G-F`, `F-C`, ...), not ESPN's per-position games-played eligibility, which the
project does not have. `src/value/positions.py` maps it to ESPN slots; the mapping is an approximation and says so (module docstring, lines 1-45).

| Dataset position | Specific positions he can fill (`eligible_positions`, line 64) | Example on the board |
|---|---|---|
| `G` | PG, SG | Shai Gilgeous-Alexander |
| `F` | SF, PF | Brandon Ingram |
| `C` | C | Nikola Jokic |
| `G-F` / `F-G` | SG, SF (`_SEAMS`, line 48) | Luka Doncic (`F-G`) |
| `F-C` / `C-F` | PF, C (`_SEAMS`, line 48) | Victor Wembanyama (`F-C`) |
| specific token (`PG`, `SF`, ...) | itself | |
| any other combination | union of the tokens | `PG-SG` gives PG, SG |
| missing / unparseable | none, UTIL only (`is_known` is False) | never invented |

**Slots** (`eligible_slots`, line 80): the specific positions, plus `G` if he can fill PG or SG, plus `F` if he can fill SF or PF (`FLEX_SLOTS`, line 39), plus
`UTIL` for everybody. Doncic (`F-G` -> SG, SF) can fill SG, SF, G, F and UTIL.

**Position group** (`position_group`, line 91): `G`, `F`, `C` or `U` from the *first* token (`F-C` is a forward, `C-F` a center). It is a modelling covariate only
(the rate priors and the rookie prior), never used for the lineup.

Where it is used: the position filter on the board tabs (`filter_board`, `src/app/state.py:96`, choices `All, PG, SG, SF, PF, C, G, F, UTIL`,
`draft_board.py:68`), the Suggestions tab (specific positions only, `best_by_position`, `state.py:127`), the positional replacement level (section 8) and
the in-season lineup solver (section 16).

Limitation: the flex slots mean position is a weak constraint in this league; ESPN's real eligibility (games at a position last year, 20+ games) can make a
player eligible somewhere this table does not.

## 4. Projection columns

Built by `BaselineProjector` (`src/models/baseline.py`, ADR 0003, `docs/modeling.md`). Everything estimable from the history is estimated (age curves,
shrinkage strength, recency decay, dispersion); the scoring is applied only at the end. Constants live in `BaselineConfig` (`src/models/config.py:14-45`).

### 4.1 Per-minute production and minutes

For a player and a target season, lags `k = 1..K` seasons back (`K = n_lags = 4`, `config.py:15`) each give a numerator `x_k` and an exposure `e_k` (minutes for
per-minute rates, shot attempts for shooting percentages, games for minutes per game). With decay `d` and the fitted age curve's expected change `shift_k`
between the player's age in lag `k` and now (`src/models/rates.py:134-147`):

```
est   = sum_k d^(k-1) * (x_k + e_k * shift_k)  /  sum_k d^(k-1) * e_k          recency-weighted, age-adjusted
n_eff = sum_k d^(k-1) * e_k
rate  = prior + n_eff / (n_eff + kappa) * (est - prior)                          empirical-Bayes shrinkage (shrink.py:16)
```

- Eleven rate quantities are modelled (`SPECS`, `rates.py:42`): `fga, fta, fg3a, reb, ast, stl, blk, tov` per minute, and `fg_pct, ft_pct, fg3_pct`; plus minutes per game
  (`MINUTES`, line 55, constant prior) and the volatility ratio (`VOLATILITY`, line 56).
- `d` is chosen per quantity from `{0.3, 0.5, 0.7, 0.9}` (`decay_grid`, `config.py:16`) and `kappa` by a fine search, both by minimising next-season squared error
  inside the history (`fit_spec`, `rates.py:242`, `shrink.py:28`). Fall-backs when the history is too thin: decay 0.6, kappa in `default_kappa` (`config.py:32-37`).
  Values fitted on the 2026-27 history on 2026-09-26: minutes decay 0.3, kappa 9.9; FGA decay 0.3, kappa 290.7; steals decay 0.7, kappa 425.0; blocks 0.5 / 337.4; FT% 0.7 / 61.7.
- `prior` is a weighted regression of the rate on an intercept, guard, center and minutes per game (forward is the baseline; `design`, `rates.py:61`), i.e. a
  "role-appropriate mean".
- Age curve: quadratic in `(age - 27)` fitted on consecutive-season pairs, its yearly change held at the edge value outside the 2.5%-97.5% age range of the data, and used only with 40+ pairs
  (`src/models/age.py:21-84`). Age is measured on 1 Oct of the season start (`panel.py`, `AgeLookup`).
- Projected minutes are clipped to `[0, 44]` (`mpg_max`, `config.py:28`, applied at `baseline.py:93`).

The per-game stat block respects box-score identities (`_stat_block`, `baseline.py:181-196`): `fga = fga_rate * mpg`, three-point attempts never exceed
attempts, `fgm = max(fga * fg_pct, fg3m)`, `pts = 2*fgm + fg3m + ftm`. `proj_fppg` is the league scoring applied to that block (`baseline.py:218`); `proj_pts`,
`proj_reb`, ... `proj_fg3m` and `proj_mpg` are the components.

### 4.2 Games played: `proj_gp`

Availability is a distribution, not a point (`src/models/availability.py`). With `f = GP / L` the fraction of the schedule played and `L` the season length
(`proj_gp = mu * L`, `baseline.py:212`; `L = target_season_games`: the largest of the last three completed season lengths so a shortened season does not lower
the cap, `panel.py:46`, 82 for 2026-27):

```
mu = logistic( w . [ f_bar, f_last, f_min, absent_last_year, age_c, age_c^2, mpg_c ] )      availability.py:41-52, fractional logistic
     f_bar  recency-weighted (decay 0.6) share of the schedule played over the last seasons
     f_last last season's share;  f_min worst recent season (an injury-proneness proxy);  absent = 1 if he played no game last season
     age_c = (age - 27) / 5;  mpg_c = (mpg - 20) / 10
mu is clipped to [0.02, 0.985]  (MU_CLIP, line 36)
Var(f) = phi * mu * (1 - mu),  f ~ Beta(mu*nu, (1-mu)*nu),  nu = 1/phi - 1;  phi clipped to [0.01, 0.6], fall-back 0.12   (lines 37-38, 127-133)
proj_gp_sd = L * sqrt(phi * mu * (1 - mu));   proj_gp_p10 / proj_gp_p90 = L * Beta quantiles
```

`phi` is fitted (0.269 on the 2026-27 history). Worked example, Jokic: `mu = 63.258 / 82 = 0.7715`, `proj_gp_sd = 82 * sqrt(0.269 * 0.7715 * 0.2285) = 17.86`,
`proj_gp_p10 = 35.8`, `proj_gp_p90 = 81.1`. His 63 games reflects a recency-weighted history of missed games and age 31.6, not a claim that he will sit 19.

Limitation (stated in the module docstring): `mu` is *conditional on the player being active in the target season*; a retirement or a season-ending injury
before opening night is not in `proj_gp` of the plain `baseline`. That risk belongs to the risk overlay (section 11), which is advisory.

**The hurdle stage (ADR 0031; `baseline_hurdle*` models, `src/models/appearance.py`).** The conditional model above is only half of the story, so the hurdle models add the
missing first stage: `proj_p_appear`, the probability that a veteran plays at least one game. It is a logistic model over `[f_bar, f_last, f_min, absent, age, age^2, mpg]`
plus how many of the last four seasons he played and how long ago the last one was (and, in the ADP / labelled-injury models, those extra columns), trained on every
historical (eligible player, season) row (eligible = played in one of the two previous seasons, the same rule the target season uses) labelled by whether the player has
any game that season. Then `proj_gp = proj_p_appear * mu * L`, games played is `0` with probability `1 - proj_p_appear` and `L * Beta(mu*nu, (1-mu)*nu)` otherwise, and
`proj_gp_sd` / `proj_gp_p10` / `proj_gp_p90` are the moments and quantiles of that mixture (`gp_mixture`; a player with more than a 10% chance of not playing has a zero floor).
Rookies and debutants keep their draft-slot prior (`proj_p_appear = 1`). Real ten-season result (backtest, walk-forward): returners' games-played bias `+15.3` to `+1.5`, total-FP
bias `+134` to `+14`, MAE of total FP `422.5` to `367.5`, Spearman `0.784` to `0.810` (10 of 10 seasons); P(appear) is calibrated out of sample (Brier `0.113` vs `0.209` for the base rate).

### 4.3 Total: `proj_total_fp`

```
proj_total_fp = proj_fppg * proj_gp                     baseline.py:219
```

Jokic: `57.988 * 63.258 = 3668.2`. This is the number the rank is built on (section 8), so a 57-FPPG player who is expected to miss a quarter of the season can rank
below a 50-FPPG player who plays 78 games.

### 4.4 Season-total band: `proj_total_fp_p10`, `proj_total_fp_p50`, `proj_total_fp_p90`

`fppg_p10/p90` (section 5) are quantiles of one *game*. A season total is `T = Fbar * G` (season-mean FPPG over the games played, times games played), and its spread has
three sources, all modelled (ADR 0033, `src/models/season_interval.py`): games played `G` from the hurdle mixture of 4.2; talent misprojection, a fitted relative spread `tau`
(fitted on the same out-of-fold projections, ADR 0033 addendum) of `(actual season FPPG - projected) / projected` by seasons of history (0, 1, 2, 3+; game noise `sd_game^2 / gp` is subtracted so it is not counted twice; floor 2%); and the
game noise of the average itself, `sd_game / sqrt(G)`. The columns are the 10th / 50th / 90th percentiles of a deterministic simulation of `Fbar * G` (1000 draws per player, a fixed
seed per player id, so a projection never changes when another player is added). `proj_total_fp` is the *mean* and sits above the median for anyone with a real chance of missing the
year. `Fbar` and `G` are treated as independent. Calibrated out of sample by risk group in the backtest report ('Season-total interval coverage'): overall 85% of realised totals fall
inside the nominal 80% band (5.5% below p10, 9.2% above p90); prime players 82%, returners 90% (conservative), rookies 68% (the rookie path has no appearance stage, so the band is
too narrow for them).

## 5. The floor / median / ceiling band

`fppg_p10`, `fppg_p50`, `fppg_p90` are the 10th, 50th and 90th percentiles of a **single game's** fantasy points (not of a season average), on a per-game
basis (`src/models/volatility.py`, `baseline.py:220-222`):

```
sd_prior(x) = max(a + b * x, 1.0)              a, b: weighted regression of a player-season's game-level FP sd on its mean FPPG   (volatility.py:40-41, SD_FLOOR line 29)
z_hat       = the player's own sd / sd_prior, recency-weighted and shrunk toward 1     (VOLATILITY spec, kappa fitted; 156 on 2026-27)
sd          = z_hat * sd_prior(proj_fppg)                                              = proj_fppg_sd
p_q         = proj_fppg + q_q * sd,   q = (q10, q50, q90)                              (floor_median_ceiling, volatility.py:76)
```

The multipliers `q` carry the *shape* of the distribution (right skew, thin left tail) and are calibrated in `FittedBaseline.calibrate_quantiles`
(`baseline.py:149-171`) as the pooled quantiles of `(game FP - projection) / projected sd` over every historical player-season projected from earlier lags
only (requires 500+ games, else the pooled per-season fall-back). Those projections are out-of-fold: the last three training seasons are each projected by
a model refit on the history before them (ADR 0033 addendum), not by the model fit on the same rows. Because the residual is taken around the **projection** and not the realised season mean,
the band contains projection error as well as game-to-game noise, so about 10% of games really do land below `fppg_p10`.

Values on 2026-09-26: `a = 7.982`, `b = 0.1810`, `q = (-1.2885, -0.0701, +1.5423)`.

Worked example, Jokic: `sd_prior(57.988) = 7.982 + 0.1810 * 57.988 = 18.48`; `z_hat = 0.925`, so `sd = 17.10` (`proj_fppg_sd`);

```
p10 = 57.988 + (-1.2885)(17.10) = 35.95      p50 = 57.988 + (-0.0701)(17.10) = 56.79      p90 = 57.988 + (1.5423)(17.10) = 84.36
```

Reading it: `p50 < proj_fppg` because game scores are right-skewed (the mean sits above the median); a wide `p10..p90` (Jokic 36 to 84) marks a player whose
individual games swing a lot, which matters for H2H weekly matchups more than for the season total. The offseason layer (section 7) multiplies the band and `sd`
by the same factor it applies to the stats (`offseason_baseline.py:142-144`).

Limitation: the band is per game. It says nothing about games missed (that is `proj_gp_p10/p90`, not on the board CSV).

## 6. Who gets which projection

| `projection_class` | Who | How projected | `confidence` | Where |
|---|---|---|---|---|
| `veteran` | played in one of the last `active_seasons = 2` completed seasons (`baseline.py:68`, `config.py:25`) | own history, section 4 | from data weight | all models |
| `rookie` | drafted in the target year, no games in the history (`baseline.py:74`) | draft-slot prior | `low` | all models |
| `stash` | drafted in an earlier year with a real pick, now arriving (`features/debutants.py:81-82`) | slot prior at a discounted pick, `p_play` share | `low` | `*_debut` models only |
| `undrafted` | never drafted / unknown to the draft records, in the NBA picture, no NBA game | slot prior at pick 61, small games share | `low` | `*_debut` models only |

`is_rookie` is True only for the current draft class. `confidence` for veterans is set from the FGA data weight `w = n_eff / (n_eff + kappa_fga)`
(`baseline.py:98`): **`low` below 0.5, `medium` below 0.8, `high` otherwise** (`baseline.py:231`); rookies and debutants have `w = 0`, so `low`.
On 2026-09-26: 317 high, 183 medium, 290 low.

**Draft-slot prior** (`src/models/rookies.py`, ADR 0003): from the history's own rookie seasons (a season whose start year equals the player's draft year), minutes,
each rate, each shooting percentage and the games share `f` are regressed (weighted ridge, `RIDGE = 1e-3`) on
`intercept, log(pick) - mean, guard, center, age - mean` (`rookie_design`, line 54). `pick_effective` (line 43) uses the overall pick, else a round midpoint
`(round - 1) * 30 + 15.5`, else `UNDRAFTED_PICK = 61` (line 32). The games share is capped at the mean share of the history's top-10 picks
(`TOP_PICK_PLATEAU`, line 34; `f_cap`, line 118) because availability plateaus at the top slots (walk-forward +18 games over-projection without the cap, +5
with it). Today `f_cap = 0.7996`, so the top rookies all show the same `proj_gp = 0.7996 * 82 = 65.56` (AJ Dybantsa, Darryn Peterson). With fewer than
`min_rookie_rows = 20` historical rookies the model falls back to a replacement-level prior (20th-percentile minutes, `baseline.py:328`).

**Debutants** (`src/models/debutants.py`, ADR 0016): the same prior at an *effective* pick that decays toward undrafted,
`61 - (61 - pick) * decay^years_since_draft` (line 116-120, `decay` chosen from `{1.0, 0.85, 0.70, 0.50}` only if it cuts training error 10%). Games are
`proj_gp = p_play * share * 82` (`predict_rows`, line 244), with `mu_f` clipped to `[0.005, 0.985]`:

- `p_play` for a stash is the **assumed constant `STASH_P_PLAY = 0.90`** (line 37): history cannot identify it (every historical stash that could be recognised played);
- `p_play` for an undrafted signee is a small logistic fit on preseason games share, preseason minutes and Summer League minutes (`EVIDENCE`, line 42), clipped to
  `[0.02, 0.97]` (line 41); the class base rate is used with no evidence.
- share: a stash plays the prior's share, an undrafted signee the fitted class share (default 0.20, `gp_share`, line 103).

Worked example: Thomas Sorber (stash) `p_play = 0.90`, share `0.543`, `proj_gp = 0.90 * 0.543 * 82 = 40.1`, `proj_fppg = 18.3`, `confidence = low`. On 2026-09-26 the
board has 52 rookies, 5 stashes and 46 undrafted signees.

Limitation: rookie and debutant lines are priors, not evidence of who the player is; the board is least trustworthy for exactly the rows the market
argues about most.

## 7. The offseason layer

`baseline_offseason` (and `baseline_offseason_debut`, the model the daily refresh writes) wraps the baseline and multiplies each player's projected counting
stats by a factor learned from how Summer League (SL) and preseason (pre) production explained the baseline's past misses (ADR 0012,
`src/models/offseason_baseline.py`, `src/features/offseason.py`). Games played, minutes and the volatility ratio are untouched.

1. **Evidence per player** (`_event_rows`, `offseason.py:106`): games, minutes, minutes per game, `fp36 = FP * 36 / minutes`, and `z`, the `fp36` z-score against that
   event's cohort (players with at least `MIN_COHORT_MINUTES = 40`, minutes-weighted mean and sd, clipped to `+-Z_CLIP = 3.0`, `_cohort_z`, line 147; `sl_z`, `pre_z`).
2. **Design** (`design`, line 218): for each event, `z * m / (m + K)` (minutes `m`, `K` chosen from `{60, 120, 240, 480}`), that times a `youth_weight`
   (1 at age <= 20 falling linearly to 0 at 26+, line 212), that times `is_rookie`, and the mpg deviation from 25 (SL) or 20 (pre).
3. **Fit**: ridge regression of `actual FPPG - baseline FPPG` on the design, over ten earlier walk-forward seasons, ridge chosen from `{3, 30, 300, 3000}` by leave-one-season-out
   CV. **Honesty gate**: the adjustment is switched off unless CV improves on "no adjustment" by at least `MIN_CV_GAIN = 0.2%` (line 55).
4. **Apply**: `factor = clip((proj_fppg + adj) / proj_fppg, 0.6, 1.6)` (`factor_from_adjustment`, line 338; `FACTOR_LO/HI`, line 56); every `proj_*` stat, `fppg_p10/p50/p90`
   and `proj_fppg_sd` are multiplied by it. `offseason_adj = new proj_fppg - base proj_fppg` (FPPG, `offseason_baseline.py:138`).

`uplift` on the watchlist is `offseason_adj`. The ADR 0012 result is that the **preseason** carries the signal and Summer League alone is close to noise; the board's
adjustment is therefore small until preseason games exist (none had been played on 2026-09-26).

## 8. Value: replacement level, VORP, rank

`src/value/replacement.py`, `vorp.py`, `board.py` (ADR 0003, `docs/modeling.md`).

**Rostered pool and replacement rank** (lines 75-87):

```
availability = mean(proj_gp of the top teams * starter_slots players by proj_total_fp) / season_games                      line 138
w            = min(1, starter_slots * (1 - availability) / bench_slots)                                                    derive_bench_weight, line 75
R            = teams * (starter_slots + w * bench_slots)                                                                   replacement_rank, line 83
```

`w` in [0, 1] is the share of bench spots that behave like starters because starters are hurt or resting; `w = 0` if the league has no bench. IR does not count.
**Replacement value** is the (R+1)-th best player's total (linear interpolation when R is fractional, `value_at_rank`, line 90). If fewer than R players exist, the
minimum. That is who you get for free from waivers.

Worked example (2026-09-26 board): top `13 * 10 = 130` players average 62.19 games, `availability = 0.7584`; `w = min(1, 10 * 0.2416 / 3) = 0.8052`;
`R = 13 * (10 + 0.8052 * 3) = 161.40`. Interpolating between the 162nd and 163rd best totals (1401.15 and 1396.54) gives a **replacement season total of 1399.3 FP**.

**VORP** (`compute_vorp`, `vorp.py:43`; lines 70-71):

```
vorp          = proj_total_fp - repl_total       (what you draft on)
vorp_per_game = proj_fppg     - repl.per_game    (rate value; ignores health; always league-wide; repl.per_game is the FPPG of the same replacement player)
```

Jokic: `vorp = 3668.19 - 1399.29 = 2268.90`; `vorp_per_game = 57.99 - 24.30 = 33.69`. Negative values are kept (below replacement): Kelly Oubre Jr. (rank 169)
has `vorp = -24.7`. `repl.per_game = 24.305` is the FPPG of the replacement player by total, interpolated the same way as the total (Noah Clowney 23.505 and Daniel Gafford 25.489, weights about 0.6 / 0.4;
`value_at_rank(fppg, R, by=total)`, `replacement.py`); see [section 17](#17-code-versus-documentation-discrepancies-found-while-writing-this) item 1.

**Rank** (`board.py:89-91`): sort by `vorp` descending, ties by `proj_total_fp` descending, then `player_id` ascending; `rank = 1..N`. Every projected player appears.

**Positional scarcity** (`positional_replacement`, `replacement.py:197-213`; `positional` mode `auto` by default, `vorp.py:64`): the league's slots are filled greedily from the best player
down (specific position with the most open slots, then G/F flex, then UTIL, then the weighted bench, `_fill_slots`, line 149); each of PG/SG/SF/PF/C then has a level =
the best *unrostered* eligible player's total. `scarcity_index = (max level - min level) / (mean of the top teams*starters totals - global level)`;
**material if `>= DEFAULT_SCARCITY_THRESHOLD = 0.10`** (line 45). Only when material does each player's replacement become the lowest level among his eligible positions
(`positional_replacement_per_player`, line 216). On 2026-09-26: levels PG 1395.8, SG 1395.8, SF 1388.7, PF 1401.2, C 1401.2, spread 12.5, typical starter 706.1,
`index = 0.0176`, **not material**, so every player uses the single league-wide 1399.3. The verdict is always stored in `board.attrs["positional"]`, never silent.

Why (ADR 0003): ESPN's flex slots (G, F, 3 UTIL) mostly erase positional scarcity; the check confirms rather than assumes it.

Limitations: the greedy fill is a heuristic (optimal assignment is a matching problem), its result only feeds this diagnostic; the model ranks by *expected volume*
(games x rate), not by weekly H2H matchup variance.

## 9. Tiers

`assign_tiers`, `src/value/tiers.py:26-`, applied to `vorp` with `floor = 0.0` (`board.py:92`). Tiers group players by value **cliffs**, not by rank buckets.

1. Sort by VORP descending; `gap_i = v_i - v_{i+1}` (line 45).
2. `scale_i` = rolling median of the gaps over a centred window of `2 * window + 1 = 31` gaps (`window = 15`, min 3 periods; line 46), so the yardstick adapts to density.
3. `i` is a cliff candidate when `gap_i >= 2.5 * scale_i` (`gap_factor`) **and** `gap_i >= 1%` of the value range (`min_gap_frac = 0.01`) **and** `gap_i > 0` (line 49).
4. Candidates are accepted strongest first (largest `gap/scale`), skipping any that would make a tier smaller than `min_tier_size = 3`, until `max_tiers - 1 = 11` cliffs are used (so at most `max_tiers = 12` tiers above replacement).
5. Every player with `vorp <= 0` (at or below replacement) forms one final "replaceable" tier (line 62), **on top of** the cap, so a board has at most `max_tiers + 1 = 13` tiers.

Tier 1 is best; tiers never decrease with rank (property-tested). The best-available and full-board tabs filter on exact tier value.

Worked example (2026-09-26): 162 players have `vorp > 0`, span `= 2267.0`, so the 1% floor is 22.7. After SGA (rank 4, VORP 1967.3) comes Maxey (1770.7): gap 196.7 vs
local `scale = 33.2`, ratio 5.9 -> cliff, tier 1 ends at rank 4. After Austin Reaves (rank 42, 775.1) comes Alex Sarr (751.6): gap 23.5 >= 22.7, ratio 3.24 >= 2.5 -> tier 6
ends at 42. Result: tiers 1 to 11 for the positive-VORP players (4, 8, 16, 4, 7, 3, 10, 21, 24, 35, 30 players) and **tier 12: 628 players at or below replacement**.

Limitation: a tier is a statement about *this projection's* value gaps; with noisy projections a "cliff" between two players 20 VORP apart is mostly noise. Use
tiers to decide when to wait on a position group, not to rank within a tier.

## 10. ADP and `adp_gap`

`adp` is ESPN's average draft position (overall pick number, 1 = first pick, fractional) from the ESPN players feed (`src/ingest/espn_adp.py`, ADR 0007), mapped to
`player_id`; the earliest (lowest) ADP wins if two rows map to one player. It is optional: without an ingested file the columns are absent and `board.attrs["adp_note"]`
says why (`loader.py:40-56`, `_load_adp`).

```
adp_gap = adp - rank                            board.py:100
```

**Positive: the model ranks him earlier than the market drafts him (a value pick).** **Negative: the market pays more than the model** (a reach, or a player the model
distrusts). Examples: Tyrese Maxey `adp 12.63`, `rank 5`, `adp_gap = +7.63`; Jayson Tatum `adp 9.68`, `rank 53`, `adp_gap = -43.32` (the market treats him as a top-10 pick, the model,
which expects 50.1 games after his long 2025-26 absence, does not; section 12); Jabari Smith Jr. `+82.9`.

**ADP as a model input and the board blend (ADR 0032).** The market prices injury news faster than game logs can, so `baseline_adp` / `baseline_hurdle_adp` feed ADP (`[covered, listed,
ln ADP - 4]`, `src/features/adp.py`; the target season's own ADP is preseason information and is used, later seasons never) into the availability and appearance models and, through a
fitted ridge, into minutes per game. Separately, ADR 0030 found ADP beats the model at the top of the board while the model beats ADP on magnitude and depth, so the board can carry
`blend_total_fp`, `blend_vorp`, `blend_rank`, `blend_tier`: for ADP-listed players a regression of past season totals on `ln ADP`, `ln ADP^2` and the model's own total (fit on earlier
completed seasons only, stored as `adp_blend.json`, `python -m src.value.adp_blend fit`), for everyone else the model total, then the ordinary VORP machinery on those totals. `rank`, `vorp` and
every model column are unchanged; `blend_rank` is an alternative ordering to read beside `rank` and `adp`.

Limitations, important:
- **Coverage.** ESPN's feed is capped (the ingest pulls the top 600 by ownership but the "no ADP" sentinel is 140.0, ADR 0007 D2). On 2026-09-26 only 202 ADP rows exist, 198 matched to the
  790-row board. Everyone else has a blank `adp`.
- **The tail is not comparable.** ADPs near 139.x are the deep end of the feed; a player with `adp 139.9` and `rank 743` shows `adp_gap = -603`. Read `adp_gap` only where both numbers are
  inside the first 140 picks.
- **Different scales.** `rank` is a VORP order over all 790 players; `adp` is an average pick number. A rank-60 player and an ADP-60 player are not interchangeable
  (draft position also depends on team needs).
- The ADP itself is the market's belief today, updated twice a day by the daily refresh (`docs/offseason.md`); ESPN's 2025-26 ADP was wiped and filled from FantasyPros.

## 11. Risk overlay

Columns: `risk_level` (`""`, `watch`, `high`), `risk_flags` (readable text), `risk_gp_haircut` (share, 0 to 0.20), `risk_gp` (`proj_gp` after the haircut). Built by
`compute_risk` (`src/value/risk.py:35`) from `src/features/risk.py` (ADR 0016 D4). It exists only in the live window, from **August of the season's start year to the end of that year**
(`season_is_live`, `value/risk.py:102`), and only for real (non-synthetic) boards. **`proj_gp`, `proj_total_fp`, `vorp` and `rank` are never changed.**

Three flag sources (constants at `features/risk.py:27-39`):

| Source | Flag text | Trigger | Level | Haircut |
|---|---|---|---|---|
| ESPN injury status (newest `espn_status_snapshots` day, `status_flags`, line 72) | `ESPN out` / `ESPN day-to-day` / `ESPN suspended` `(last news Nd ago)` | status `OUT`, `DAY_TO_DAY`, `SUSPENDED` | `OUT` -> `high`; other two -> `watch` | `ASSUMED_STATUS_HAIRCUT = OUT 0.20, DAY_TO_DAY 0.03, SUSPENDED 0.10` (line 35) |
| Same, stale | `..., stale` | last news older than `STALE_NEWS_DAYS = 45` | none | none (text only) |
| Preseason absence (`preseason_flags`, line 55) | `played 0 of N preseason games` / `played k of N ...` | a veteran (`n_hist >= 1`) with `proj_mpg >= ROTATION_MPG = 15`, on a team with `>= MIN_TEAM_GAMES = 3` preseason games, who played none (`dnp_all`) or under `PARTIAL_SHARE = 0.5` of them (`partial`); only players on the current roster snapshot | `watch` | **`PRESEASON_HAIRCUT = 0.0` for both** (line 39) |
| Team change (`context_flags`, line 93) | `new team (from XXX)` | last season's final team differs from the roster snapshot | none | none |
| Star context | `star arrived: A, B` / `star left: C` | a top-`STAR_RANK = 60` board player (line 31) arrived at / left his current team since last season | none | none |

```
risk_level      = max over the triggered sources of (0 / 1 watch / 2 high)                   build_overlay, lines 139-172
risk_gp_haircut = max over the triggered haircuts                                            (max, never a sum)
risk_gp         = proj_gp * (1 - risk_gp_haircut)                                            line 174
```

Worked examples: Shaedon Sharpe, `ESPN out (last news 32d ago)`: `proj_gp = 59.84`, haircut 0.20, `risk_gp = 47.88`, level `high`. Giannis Antetokounmpo, `ESPN day-to-day
(last news 23d ago); new team (from MIL)`: `51.45 * 0.97 = 49.91`, level `watch` (only the day-to-day part discounts; the team change is text). On 2026-09-26: 5 `high`, 20 `watch`, none from preseason (no preseason games yet).

Why the haircuts are what they are (ADR 0016 D4): ESPN keeps no status history and the NBA injury-report PDFs do not exist before opening week, so **the status haircuts are assumptions**, not
estimates. The preseason-absence flag was measured (`python -m src.backtest.preseason_availability`) and could not be separated from "not under contract" (78% of "absent" veterans played no
game at all), so it warns without discounting. Roster and transaction context was tested as a projection input and found no lift (ADR 0010, 0011), so team and star flags are descriptive.

Limitations: a single ESPN snapshot; no reason for an absence; retirees and unsigned free agents are invisible in a roster snapshot and are not flagged.

## 12. Returned-healthy flag

ADR 0023, `src/value/return_flag.py`. An advisory flag for a player like Jayson Tatum: out for the first part of last season, then healthy. The cohort is exactly
ADR 0021's primary cohort, computed by that study's own functions (`PRIMARY`, `return_report.py:68`; `in_cohort`, line 171): on **last season only**,

- a veteran (`_veteran_flags`, line 133) on **one team** (`n_teams == 1`), with `mpg >= 15` (`in_universe`, line 162);
- missed the first `>= 25%` of his team's games (the lead block: `first_idx >= 0.25 * L`);
- then had `>= 15` team games left and played `>= 75%` of them (`gp / (L - first_idx) >= 0.75`).

| Column | Meaning | Formula |
|---|---|---|
| `return_flag` | `returned-healthy` or blank | cohort membership (`return_flag.py:125`) |
| `return_block_pct` | lead block as a percent of team games | `100 * first_idx / L`, one decimal (lines 81, 126) |
| `return_tail` | games played / team games since his first appearance | `gp / (L - first_idx)`, e.g. `16/20` (line 129) |
| `return_gp_upside_adv` | advisory games: 0 or 5 | `5` if block share `>= 0.60` (`UPSIDE_MIN_BLOCK`, line 48) else 0, capped so `proj_gp + upside <= season_games` (line 52-57) |
| `return_fp_upside_adv` | the same games in total FP | `return_gp_upside_adv * proj_fppg` (line 131) |
| `return_validation` | constant `advisory_judgement_not_projection` | line 45 |

The readable sentence ("returned from long absence (missed the first 62 of 82 team games), healthy since: ...") is appended to `risk_flags` (line 151-154); `risk_level`,
`risk_gp_haircut` and `risk_gp` are never modified. The sibling ADR 0024 flag follows the same pattern (`lm_flag = short-absences`, `lm_iso_n`, `lm_rest_n`, `lm_gp_risk_adv` = 0 or -2 games, `lm_fp_risk_adv`, `lm_validation = advisory_descriptive_not_projection`; `src/value/load_flag.py`) and is likewise never added to a projection or rank; see section 18. In the app: the `return_flag` and `return_tail` columns after `vorp`, a **Returned-healthy only** checkbox on the Best available and
Full board tabs (`draft_board.py:183-188`), and a table on the Debutants & risk tab (`draft_board.py:376-384`, `state.py:217-226`).

Worked example: Tatum missed the first 62 of 82 games and played 16 of the last 20: `return_block_pct = 100 * 62 / 82 = 75.6`, `return_tail = 16/20`, block >= 60% so
`return_gp_upside_adv = 5`, `return_fp_upside_adv = 5 * 41.01 = 205.0`. His `proj_gp = 50.07`, `proj_total_fp = 2053.3` and rank 53 are **unchanged**. Nine players are flagged on 2026-09-26; five get
the +5 (Tatum, Scoot Henderson, Max Strus, Killian Hayes, Cameron Payne).

Why advisory (ADR 0021, 0022, 0023): the pre-registered study (n = 60 over ten seasons) found no significant absolute under-projection for this group (+1.0 GP, 95% CI -4.1 to +5.8); the
only cell with a raw hint was the `>= 60%` block (n = 21, about +8 GP) and it was one of 18 cells. The ADR 0022 model feature over-corrected (Tatum 50.1 -> 57.7 GP) and is registered, not recommended. So the +5 is
a fixed judgement, "sized by judgement, not derived", written before it was applied to any player. A player not on the list may still have missed time.

## 13. Contract flags

ADR 0019, `src/value/contract_flags.py` + `src/features/contract_terms.py`. **Unvalidated and display only**: a ten-season walk-forward found no reliable lift from contract
year, so nothing here changes a projection or a rank; a blank means *not known to be*, never *known not to be* (Wikipedia coverage is partial and biased to notable signings).

Columns and states (`contract_terms.py:88-165`, `contract_flags.py:34-40`):

- `contract_status`: `known` (an active deal with a stated length, `years_remaining >= 1`), `no_length` (a signing whose length the page omits), `lapsed` (the latest known deal ended or was waived
  / bought out, so he is on a deal we never saw), `unknown` (no event). `contract_years_left` is NaN unless `known`: `end - season_start + 1` (line 118).
- `contract_basis`: `wiki` (any contract event), `rookie_scale` (nominal first-round clock, only when the status is not `known`, `basis`, line 161), or blank.
- `contract_flag`, by precedence (`flag`, line 151; a later assignment overwrites): `contract year` (known, 1 year left) > `rookie final year (nominal)` > `extension` > `new deal` (a deal dated
  1 Jul - 30 Sep of the season's start year, `cov == s`, line 115) > `rookie option year (nominal)`.
- Nominal rookie clock (`contract_clock`, `features/contract.py:83`): first-rounders only; `years_since_draft = season_start - draft_year + 1`, `SCALE_YEARS = 4` (line 46); year 3 is the option
  year (`is_option_year`, line 107), year 4 the final year. A first-rounder with a known Wikipedia deal is never given the nominal flag.
- Length arithmetic (module docstring): a deal dated 1 Jul or later covers the season starting that year; a `N`-year deal covers `start .. start + N - 1`; an extension adds `N` seasons after the
  running contract's end; ten-day contracts never set the status.

Examples: Michael Porter Jr. `contract year`, `years_left 1`, `known`, `wiki`; Victor Wembanyama `rookie final year (nominal)`, `unknown`, `rookie_scale`. On 2026-09-26: 83 new deal, 30 rookie final,
30 rookie option, 14 contract year, 1 extension; 35 players have a known length and 376 are `unknown`.

Limitation: a `new deal` flag is far more common than a `contract year` flag simply because recent signings are the best-covered events.

## 14. Breakouts tab (the watchlist)

`src/value/breakouts.py` (ADR 0012, `docs/offseason.md`). A different question from the board: **which young players does the offseason evidence move the most, and which of those is the market not paying for?**

Definitions (identical to the backtest, `BreakoutConfig`, `src/backtest/breakouts.py:50-57`):

| Term | Definition | Constant |
|---|---|---|
| young | `age <= 23.0` | `young_age` |
| under the radar | no ADP, or `adp_rank > 100` where `adp_rank` is the player's rank among ADP holders (`breakouts.py:199, 204`) | `radar_rank = 100` |
| breakout | actual FPPG beat the *base* projection by `>= 4.0` FPPG **and** `>= 25%`, while playing `>= 20` games (`breakout_flag`, line 65) | `abs_uplift`, `rel_uplift`, `min_gp` |
| useful breakout | a breakout that also finished in the rostered pool (top 169 by season total, `13 x 13`) | `rostered_rank = 169` |
| `uplift` | `offseason_adj`: layered `proj_fppg` minus base `proj_fppg` (FPPG); only players with `uplift > min_uplift (0)` are listed (`select_watchlist`, line 253) | |

Probabilities are a logistic calibration fitted on the backtest's out-of-sample scores (`BreakoutCalibration.probability`, line 93):

```
P(outcome) = sigmoid( intercept + coef_score * uplift + coef_rookie * is_rookie )         outcome in {breakout, useful}
```

`baseline_summer_league` calibration (`processed/breakout_calibration_baseline_summer_league.json`, n = 1010): breakout `-1.5358 + 0.1378*uplift + 0.5184*rookie` (base rate 0.214); useful `-2.0783 +
0.0900*uplift + 0.2877*rookie` (base rate 0.125). **Both are conditional on playing 20+ games** (see `proj_gp`).

Worked example, Brayden Burries (rookie, SL 4 games, `uplift = 4.741`): `breakout_prob = sigmoid(-1.5358 + 0.1378*4.741 + 0.5184) = sigmoid(-0.364) = 0.410`; `useful_prob =
sigmoid(-2.0783 + 0.0900*4.741 + 0.2877) = sigmoid(-1.364) = 0.204`. `evidence` reads `SL 4g 27mpg z+2.8` (games, minutes per game, `z` = per-36 production versus that event's cohort).

Model choice (`choose_model`, line 142): `baseline_summer_league` until any preseason game of the season exists, then `baseline_offseason` (the app's tab and the daily report both say which).

Columns (`WATCH_COLUMNS`, line 239; app view `WATCH_DISPLAY`, `draft_board.py:57`): `watch_rank`, `name`, `team`, `position`, `age`, `is_rookie`, `draft_pick`, `board_rank` (the board's rank), `adp`, `adp_gap`,
`base_fppg`, `layer_fppg`, `uplift`, `proj_gp`, `proj_total_fp`, `vorp`, `useful_prob`, `breakout_prob`, `changed_team`, `sl_z`, `pre_z`, `evidence`, plus `risk_flags` and `contract_flag`. Sort options:
`useful_prob` (default), `breakout_prob`, `uplift`, or raw `sl_z` / `pre_z` (no model). Filters: include priced players, include older than 23, minimum uplift, hide drafted.

Evidence (ADR 0012, ten seasons): the top ten flagged young under-the-radar players became a **useful** breakout about **24%** of the time against a **10.3%** base rate (+13.7 pp, 95% interval +5.8
to +22.1); on the plain breakout target `35%` against `20.6%` (the `35% vs 21%` in the `breakouts.py` docstring is this second target). The preseason carries the signal (rookies AUC 0.66); Summer League
alone is close to noise (AUC 0.52-0.57). An edge, not a lock.

## 15. Transactions and Coaches tabs

### Transactions (ADR 0017, `src/features/txn_impact.py`, `src/app/txn_view.py`)

Descriptive only; "a player changed team" and "a star arrived or left" were tested as projection inputs and did not help (ADR 0010, 0011: 0011 measurably hurt). Source: ESPN's public
transactions feed, refreshed by `python -m src.ops.txn_watch --refresh` (also in the daily refresh); the tab never touches the network.

- **One row per player move** (`events`, `txn_impact.py:38`): a trade appears in the ledger once per team (`trade_in`, `trade_out`) and collapses into one `trade` row `from_abbr -> to_abbr`; `signed`,
  `resigned`, `extended`, `converted` (two-way to NBA), `claimed` are arrivals (`ARRIVAL_KINDS`, line 29, `to_abbr` only); `waived` is a departure (`from_abbr` only). Staff kinds (`hired`, `fired`,
  `resigned_staff`, `extended_staff`) go to the coach table; unparsed text is stored as `other` and never guessed.
- **`rank`, `adp`, `position`, `vorp`** are joined from the *loaded* board (`annotate`, line 78), so they match the rest of the app.
- **`relevant`** = `rank <= RELEVANT_RANK = 180` (line 26, roughly the top 13 rounds of a 13-team draft); the tab hides the rest unless "Include players outside the board's top 180" is ticked or a player
  name is typed.
- **`context`** (relevant rows only): the destination team's three best-ranked teammates, and "N of the next 8 best teammates share his position" when `N >= 3` (`CROWD_TOP = 8`, line 28;
  position compared on the first token, `_primary_pos`, line 74), plus who is left behind at the origin team.
- Filters: last N days (default 14), team, kind (`KIND_CHOICES`, `txn_view.py:16`), player substring. Table columns `DISPLAY_COLUMNS` (`txn_view.py:15`).
- Second table: coach and front-office moves for the last `max(N, 120)` days.

Example: `2026-09-22 extended Ausar Thompson G-F rank 77 adp 104.48 -> DET: DET best teammates: Cade Cunningham (#6), Jalen Duren (#16), John Collins (#135); 4 of the next 8 best teammates share his position`.
Limitation: the feed is known to be incomplete; 99% of per-player latest moves agree with the NBA roster snapshot (ADR 0017).

### Coaches (ADR 0020, `src/value/coaches.py`, `src/features/coach.py`, `src/app/coach_view.py`)

Descriptive only: `baseline_coach` (the same information as a minutes adjustment, capped at +-4 mpg) showed no significant lift on ten seasons and is registered, not recommended.

Style axes per team-season (`team_style`, `features/coach.py:45-84`), each also stored league-relative (`*_x` = style minus that season's league mean); a team-season needs `MIN_GAMES = 20` games:

| Axis | Definition |
|---|---|
| `star` | minutes of the team's top minute-getter, averaged over games |
| `top5` | mean minutes of the five biggest minute-getters |
| `depth10` | players per game with 10+ minutes (`LEAGUE_MIN_MPG = 10`) |
| `pace` | `(FGA + 0.44*FTA - OREB + TOV) / games` (possessions per game) |
| `three` | `FG3A / FGA` |
| `young` / `old` | share of team minutes by players aged `<= 23` (`YOUNG_AGE`) / `>= 31` (`OLD_AGE`) |

Table columns (`TABLE_COLUMNS`, `coach_view.py:13`): `team, coach, since, new_to_team, previous_coach` (marked `(interim)` / `(acting)` where he was), `prior_seasons` (his earlier head-coaching seasons in the
table), `career_*` (his own earlier single-coach seasons' league-relative style: `star`, `top5`, `depth10`, `pace`, `old`, `young`). `new_to_team` means the opening coach differs from the one the team finished with
last season (`is_new`, `coach.py:152`). The `shift_*` columns (kept in the table, used in the summary sentences, not in the displayed table) are the **expected style change**:

```
shift_c = n / (n + PRIOR_SEASONS_K) * ( career_c - last_c )      PRIOR_SEASONS_K = 1.0  (coach.py:41, 157-169)    n = his prior seasons; 0 if the coach is unchanged or a first-timer
```

Worked example, Taylor Jenkins (MIL, replacing Doc Rivers): `n = 5`, `career_top5 = -1.819`, MIL last season `+0.043`, `shift_top5 = 5/6 * (-1.819 - 0.043) = -1.552`: "more spread minutes (top five
-1.6 min)". `describe` (`coaches.py:76`) prints a sentence when `shift_top5` exceeds +-0.5 minute or `shift_depth10` exceeds +-0.15 players; a first-time head coach gets "no head-coaching history to import".
The ranked-players table lists board ranks `<= 150` on a team with a new head coach (`affected_players`, `coaches.py:67`).

What the data shows (ADR 0020): style follows the coach for top-five minutes (+0.43 [+0.02, +0.70]), rotation depth (+0.42 [+0.03, +0.78]) and three-point share (+0.32 [+0.13, +0.63]); star minutes probably;
**not** pace, youth share or veteran share, so "develops young players" is not a portable coach trait here. Pace is displayed only for context.

## 16. In-season page

`src/app/pages/inseason.py` (ADR 0015, 0018, `docs/inseason.md`). Nothing is computed until **Load in-season data** is pressed; it reads only the local data directory, never the network. Sidebar: season,
`as_of` (games dated on or before it count), preseason model (default `baseline`), synthetic toggle, data directory, ESPN league id.

### Rest of season (`src/inseason/ros.py`)

The preseason projection is shrunk toward this season's production. For each of three quantities `posterior = (k * prior + n * sample) / (k + n)` (`_shrink`, line 67), with pseudo-counts chosen by walk-forward
evidence on 2016-17..2020-21 (`RosParams`, lines 54-64):

| Quantity | `n` | `k` | Constant |
|---|---|---|---|
| per-minute production `fp_per_min` | minutes played to date | `minutes_pseudo = 300` | line 57 |
| minutes per game | games played | `games_pseudo = 3` | line 58 |
| availability (chance of playing a team game) | team games played | `avail_pseudo = 5` | line 59 |

```
ros_fppg     = fp_per_min * mpg                                          line 194
ros_games    = clip(avail, 0, 1) * team_games_left                       line 197   (games after as_of from the schedule, else season length - team games played)
ros_total_fp = ros_fppg * ros_games                                      line 209
w_sample     = minutes / (300 + minutes)                                 line 210   share of the answer that is this season's data
ros_vorp     = same VORP machinery as the board, on the ROS values, season_games = games left     add_value, line 219; ros_rank by ros_vorp
```

Players with no preseason projection (a call-up, a two-way signing) get a deliberately weak prior `0.45 FP/min, 12 mpg, 0.55 availability` (`weak_*`, lines 62-64) and `has_prior = False`.

Worked example (a real past date, Jokic 2025-26 as of 2025-12-01, 20 games, 34.84 mpg, 70.95 FPPG so far; preseason 59.71 FPPG, 33.14 mpg, availability 0.808; 62 team games left):
`w_sample = 696.8/(300+696.8) = 0.699`; `mpg = (3*33.14 + 20*34.84)/23 = 34.62`; `fp_per_min = (300*1.8016 + 696.8*2.0364)/996.8 = 1.9657`; `ros_fppg = 68.05`;
`avail = (5*0.808 + 20*1.0)/25 = 0.9615`; `ros_games = 0.9615*62 = 59.6`; `ros_total_fp = 4056.9`; replacement level 1196.7 -> `ros_vorp = 2860.3`.

Columns (`ROS_COLUMNS`, `inseason_view.py:23`): `ros_rank, name, position, team, gp, fppg_to_date, prior_fppg, ros_fppg, avail, ros_games, ros_total_fp, ros_vorp, w_sample`. Evidence (`docs/inseason.md`): the blend beats
preseason-only in 5/5 held-out seasons but only marginally beats season-to-date-only on rank.

### Schedule (`src/inseason/schedule.py`, `weeks.py`)

Per matchup week (real ESPN periods when published, otherwise derived Monday-Sunday with All-Star week merged; the page names its source): a team's `games`, `games_left` (strictly after `as_of`), `b2b` (second night of
consecutive days, `flag_back_to_backs`, line 251), `off_night_games`, `heavy`, `light`.

```
per7      = games * 7 / days_in_week
heavy     = per7 >= 3.75      (>= 4 games in 7 days; a 6-day week 1 with 4 games counts)     HEAVY_PER_7, line 76
light     = per7 <= 2.5       (<= 2 games in 7 days)                                          LIGHT_PER_7, line 77
off-night game = a game on a league-wide slate no larger than floor(35th percentile of slate sizes)     OFF_NIGHT_QUANTILE = 0.35, lines 78, 246
```

Week 1 of 2026-27 (2026-10-20..25, 6 days): `PHI` 4 games = 4.67 per 7 -> heavy; a 2-game team = 2.33 per 7 -> light. Summary columns: `mean_games, min_games, max_games, heavy_teams, light_teams, b2b_total`.

### Trade analyzer (`src/inseason/trade.py`, `lineup.py`)

A roster's value is the slot-aware rest-of-season total (`lineup_from_arrays`, `lineup.py:75`):

```
value = sum of ROS totals of the best assignment of players to the 10 starting slots + w * sum of ROS totals of the best bench (3) + IR
```

The assignment is a maximum-weight matching (`scipy.optimize.linear_sum_assignment`, line 91) with eligibility from `eligible_slots`; an empty slot is a dummy zero (a roster without a center leaves C empty). `w` is the bench weight
of section 8 recomputed from the ROS frame (`bench_weight_for`, line 57). Both rosters are first made a full roster (`fit_to_capacity`, line 135): a freed spot is filled by the best free agent (greedy by marginal value) or a generic
replacement, an overfull roster drops the least valuable player. `delta = value(after) - value(before)` (`trade.py:101`); verdict "roughly even" inside `+-1.5%` of the roster's value (`DEFAULT_TOLERANCE = 0.015`,
line 47), else "favours you" / "favours the other side". Not exemplified with numbers here: it needs a real roster, which exists only after the draft (logic is tested on synthetic and as-if-mid-season 2025-26 data).

**App UI** (ADR 0027, `src/app/pages/inseason.py`'s Trade analyzer tab, wired through `src/app/inseason_view.py`'s `run_trade`): pick players to give/get from multiselects sourced off the chosen roster and every other
rostered/free player, and optionally a partner team label (`RosterChoice.others`, mirroring the CLI's `--partner`) to score their side of the same trade with `evaluate_trade` run a second time, give/get swapped.
`delta` is shown as an `st.metric` (Streamlit's own sign coloring) plus a five-way **confidence bucket** — `clear win` / `lean your way` / `roughly even` / `lean other way` / `clear loss` — from
`confidence_bucket(delta, tolerance)` (`trade.py`, a pure function, unit-tested): inside the tolerance band is "roughly even" (the same rule `verdict` already uses); up to `CONFIDENCE_CLEAR_MULT = 3.0` times the
band is a "lean"; beyond that is a "clear" win/loss. It is a magnitude read on the existing tolerance band, not a new statistic, shown with the matching `st.success/info/warning/error` banner (the app has no
bespoke CSS/badge styling to reuse instead). Below it: **advisory notes** (`inseason_view.trade_caveats`) for the traded players — current ESPN injury status (`OUT`/`DAY_TO_DAY`/`SUSPENDED`/...) from
`ctx.injuries`, plus the same `return_flag`/`lm_flag` advisory text the draft board shows (ADR 0021/0023, ADR 0024), recomputed for just the traded players off the rest-of-season frame
(`ros_games`/`ros_fppg` standing in for `proj_gp`/`proj_fppg` so the advisory games size against what is left of this season). Every note says explicitly that it is advisory and never changes the projection;
the list (plus `ctx.notes`) is empty, never an error, when the underlying data is missing.

### Waivers (`src/inseason/waivers.py`, `signals.py`)

- **Best adds**: for each free agent (every NBA-active player no league team rostered; top `DEFAULT_CANDIDATES = 80` by ROS total plus up to 40 flagged), the best drop is chosen by marginal lineup value:
  `gain_ros_fp = value(roster + FA - drop) - value(roster)` (`waivers.py:138`). Columns `ADD_COLS` (line 200).
- **Streaming**: free agents (not ESPN `OUT`, line 181) ranked by `week_gain_fp = week_exp_fp - drop's week_exp_fp`, where `week_exp_fp = games_in_week * availability * ros_fppg` (`expected_week_games`, `schedule.py:321`). Columns
  `STREAM_COLS` (line 202): `week_team_games, week_exp_games, week_b2b, week_off_night, ...`.
- **Flags** (`signals.py`, constants lines 42-51): `rising_minutes` = last `WINDOW = 5` games vs earlier games (needs `MIN_BASE_GAMES = 8`): `min_delta >= 3.0` and `min_z >= 1.5`; `rising_usage` = usage per 36
  `(FGA + 0.44*FTA + TOV)/MIN*36` up `>= 2.0` and `z >= 1.5`, where `z = delta / (max(baseline sd, 3) / sqrt(n_recent))` (lines 78-82). `inherits_from` / `exp_gain_mpg`: an absent rotation player (`>= 18` mpg over 8+
  games, whose team has played a game since his last appearance, or ESPN `OUT`; `long_term` = missed 10+ in a row) frees his minutes, allocated to teammates in proportion to headroom `(36 - mpg)` times 1.5 for the same position group
  (`allocate_minutes`, line 131), blended with observed with/without minutes once he has missed 3+ games with weight `n_out / (n_out + 4)` (`beneficiaries`, line 149; only gains above 0.25 mpg are kept).
  `injury` = his own ESPN status.
- The nightly job (`src/ops/nightly.py`, ADR 0018) runs the same analysis and writes `reports/nightly/latest.md`.

Limitations: expected-volume only (no daily-lineup or game-by-game injury simulation); the ROS projection is thin against real rosters until the season is played.

## 17. Code versus documentation discrepancies found while writing this

None changes a rank on today's board; they are recorded so the docs can be corrected (items 1 to 3 are resolved).

1. **Resolved (2026-09-27): `vorp_per_game` replacement was not "the same player".** `replacement.py`, `vorp.py:6` and `docs/modeling.md` say the per-game variant subtracts the same replacement player's FPPG.
   The code used to hand `value_at_rank` an array that it re-sorted descending, so it returned the R-th best FPPG *in the league*, not the FPPG of the R-th best player by total (25.155 versus 24.305 on the
   2026-09-26 board, a 0.85 FPPG offset). **Fixed in code**: `replacement_level` now calls `value_at_rank(fppg, R, by=total)`. On the real 2026-27 board only `vorp_per_game` changed (every other column, rank and
   tier byte-identical; a constant +0.507 in this build, replacement 25.059 -> 24.552). The rank uses `vorp` (total-based), so no rank was affected. Tests: `tests/value/test_value_b_replacement.py`.
2. **Resolved (2026-09-27): tier count.** The docs said at most 12 tiers; `max_tiers` (default 12) caps only the *above-replacement* tiers (up to 11 cliffs) and the replaceable tier (`tiers.py:62`) is added on top, so a
   board can show 13. The docs and docstrings now say "at most `max_tiers` above-replacement tiers plus one replaceable tier"; pinned by `test_max_tiers_caps_above_replacement_tiers_and_the_replaceable_tier_is_extra`.
   The 2026-09-26 board had 12 (11 + replaceable); the rebuilt board has 11.
3. **Resolved (2026-09-27): `SUSPENDED` haircut was undocumented.** `features/risk.py:35` gives a suspended player level `watch` and a 10% haircut (OUT 20%, DAY_TO_DAY 3%, SUSPENDED 10%, all assumptions). ADR 0016 D4
   (amendment note) and `docs/offseason.md` now state it; pinned by `test_suspended_status_is_a_watch_flag_with_a_ten_percent_haircut`.
4. **Returned-healthy tooltip is incomplete.** The checkbox help (`draft_board.py:187`) omits the `>= 15` remaining-games condition that the Debutants & risk caption (line 379), the code (`min_tail = 15`,
   `return_report.py:55`) and ADR 0023 all state.
5. **Two breakout hit rates, both correct.** `breakouts.py:11-14` quotes 35% vs 21% (plain breakout), the app caption (`draft_board.py:287-291`) and `docs/offseason.md` quote 24% vs 10% (useful breakout).
   Same ten-season backtest, different target (ADR 0012 lines 174 and 182). Worth a sentence in the module docstring.
6. **Signals docstring.** `signals.py` module docstring says `min_z` scales the minute change "by the baseline's game-to-game minute spread (floored at 3 minutes)"; the code also divides by
   `sqrt(n_recent)` (line 78), so it is a standard-error z, not a spread z.
7. **`docs/app.md` has no dedicated Transactions or Coaches entry.** Its "What it does" list mentions the Debutants & risk tab only as the home of the return-flag table; the Transactions and Coaches tabs are described in `docs/transactions.md` and `docs/coaches.md`.

## 18. Load management: where it fits

Load management (a healthy player sitting a game, most visibly the second night of a back-to-back) was evaluated in [ADR 0024](adr/0024-load-management.md). **No data labels why a game was missed** (no rest / injury /
suspension reason anywhere; the injury-report PDFs start 2018-12-19 and are not ingested), so it can only be proxied by the shape of last season's absences. Pre-registered, walk-forward over ten seasons, universe
n = 1,830 rotation veterans (single team, `>= 20` mpg, `>= 41` GP the season before):

- **Primary proxy** `iso_n82`, games missed in interior runs of `<= 2` consecutive games (per 82): **-1.4 GP [-2.4, -0.4] and -53 total FP [-88, -20] per +1 SD** versus the baseline (`baseline_injury`: -1.1 GP, -47 FP), the same sign in 9 of 10
  seasons. More short absences go with a *lower* season than projected, the opposite of a "rested star bounces back" story.
- **The rest-specific proxy shows nothing**: a one-game absence on the second night of a back-to-back, `rest_n82`, is -3 total FP [-36, +30]; the 65-game marker is +4 [-26, +32]; age >= 31 and >= 30 mpg is +26 [-14, +67].
- **Not incremental**: correcting `proj_gp` with the proxy did not lower out-of-sample MAE (total FP +14 [+6, +22] *worse*), so the verdict under the fixed rule is "descriptive association". Projections, VORP and ranks are unchanged.
- **Advisory flag shipped** (style of sections 11-13): `lm_flag` = `short-absences` for a veteran with 6+ such games (post-hoc: -2.6 GP [-5.1, -0.2], -98 FP [-183, -13] versus similar players), advisory **-2 GP** and that times
  `proj_fppg` in FP, cause labelled unknown. 54 of 185 universe players carry it on the 2026-27 board (29%, a high-absence season). Code: `src/value/load_flag.py`, `src/features/load_management.py`, study `src/backtest/load_report.py`.
- Needs a labelled rest feed (game-day inactive reasons) to say anything about load management proper.

## 19. Tabs and pages

**Draft board** (`streamlit run src/app/draft_board.py`, `draft_board.py:467`). Sidebar: Season, Model (any registered projector, default `baseline`), Teams override (0 = config), Positional scarcity
(`auto`/`on`/`off`), synthetic demo, data directory, Load / rebuild board, draft-state reset / export / import (JSON), and **Live draft sync (ESPN)** (section 21). **Load / rebuild board always fetches current data from disk**: the click
handler calls `_cached_load_board.clear()` before invoking the cached loader, so a rebuild is never served the previous click's `st.cache_data` result even though that cache has no ttl (unlike the
watchlist's ttl=600 and the coach table's ttl=3600) — the cache still speeds up ordinary reruns (tab switches, widget interactions) between clicks. A caption above the tabs shows when the
currently-shown board was (re)loaded (`st.session_state.board_loaded_at`).

| Tab | Function | Shows |
|---|---|---|
| Best available | `_best_available_tab`, line 199 | the board minus drafted players, filters (position, tier, name, returned-healthy only, many-short-absences only), a "Mark a player drafted" form (me / opponent). Columns `display_columns` (`state.py:245`): `rank, name, position, tier, proj_fppg, proj_gp, proj_total_fp`, then `adp, adp_gap`, `vorp`, `return_flag, return_tail, lm_flag`, `fppg_p10/p50/p90`, and when present `projection_class, p_play, risk_level, risk_flags, contract_flag` |
| Suggestions | line 221 | for each of PG/SG/SF/PF/C the top 3 undrafted players eligible there (`rank, name, proj_fppg, vorp, tier`) and how many of your picks are eligible at each position. A scarcity read, not an auto-drafter |
| Drafted | line 242 | the pick log (`pick_no, player_id, name, position, drafted_by`) with undo |
| Best available by need | `_need_tab` (ADR 0026) | the undrafted board crossed with your drafted roster's positional gaps, ranked by `need_adj_vorp` (`need_rank, rank, name, position, proj_fppg, proj_gp, proj_total_fp, vorp, need_score, need_adj_vorp, tier, adp`), plus a per-position `need_position, need_capacity, need_filled, need_score` breakdown. Falls back to (is rank-identical to) plain VORP order with an empty roster — see section 22 |
| Full board | line 263 | every player, drafted or not, with a `drafted_by` column; same filters as Best available, including the **Many short absences only** checkbox (`_short_absences_checkbox`, line 191, shown only when the board has `lm_flag`) |
| Breakouts | line 285 | the watchlist (section 14), loaded on request (about a minute the first time). Once loaded for a (season, teams) key, a **Refresh watchlist** button (re-clears and re-calls the ttl=600 cache) replaces the initial load button, since nothing else would call the cached function again for that key |
| Debutants & risk | line 348 | stash / undrafted debutants (`p_play`), players with a `risk_level` (`proj_gp` vs `risk_gp`), the returned-healthy table (section 12), the **Many short absences last season** table (`lm_iso_n`, `lm_rest_n`, advisory games; `load_flag_table`, `state.py:233`; section 18), contract-flag lists by flag (section 13), and the **External rankings disagreement** table when a comparison is loaded this session (`rankings_disagreement_table`; section 24) |
| Transactions | line 408 | section 15 |
| Coaches | line 443 | section 15 |

**In-season page** (`src/app/pages/inseason.py:152`): tabs Schedule, Rest of season, Trade analyzer, Waivers (section 16).

**External rankings page** (`src/app/pages/external_rankings.py`): tabs All players, Biggest disagreements, Unmatched names (section 23).

## 20. Glossary of every column

Board columns first (all on `reports/daily/draft_board_*.csv` and in the app), then the other tables. The app itself has a
**Glossary** tab (and a `help=` hover tooltip on every table column) with the same facts, shorter, grouped the same way as
below; its text is a single source of truth in `src/app/glossary.py`, derived from this document rather than re-typed
ad hoc — this document remains the deep reference.

### Board: identity and projection

| Column | Meaning | Section |
|---|---|---|
| `rank` | position in the VORP ordering (1 = best) | 8 |
| `player_id`, `name`, `position`, `age` | NBA id, name, coarse dataset position, age on 1 Oct of the season | 3 |
| `proj_fppg` | projected fantasy points per game played (a mean) | 4 |
| `proj_gp` | expected games played (mean of the availability distribution) | 4.2 |
| `proj_total_fp` | `proj_fppg * proj_gp` | 4.3 |
| `vorp` | `proj_total_fp` minus replacement total | 8 |
| `vorp_per_game` | `proj_fppg` minus replacement FPPG | 8 |
| `fppg_p10`, `fppg_p50`, `fppg_p90` | floor, median, ceiling of a single game's FP | 5 |
| `proj_p_appear` | chance a veteran plays at least one game (hurdle models only; 1.0 for rookies and debutants) | 4.2 |
| `proj_total_fp_p10`, `proj_total_fp_p50`, `proj_total_fp_p90` | floor, median, ceiling of the season total (hurdle models only) | 4.4 |
| `tier` | value-cliff group, 1 best, last = at or below replacement | 9 |
| `is_rookie` | current draft class | 6 |
| `confidence` | `low` / `medium` / `high` from the data weight | 6 |
| `projection_class` | `veteran` / `rookie` / `stash` / `undrafted` | 6 |
| `p_play` | chance a debutant plays at all (stash fixed 0.90) | 6 |

Projection-table extras (not on the board CSV): `proj_mpg`, `proj_pts, proj_reb, proj_ast, proj_stl, proj_blk, proj_tov, proj_fgm, proj_fga, proj_ftm, proj_fta, proj_fg3m` (per-game stat block),
`proj_fppg_sd` (sd used for the band), `proj_gp_sd`, `proj_gp_p10`, `proj_gp_p90` (games-played moments; a mixture with a point mass at zero in the hurdle models), `n_hist_seasons` (seasons of history behind the projection), `offseason_adj`, `offseason_factor`, `offseason_enabled`,
`sl_gp, sl_mpg, sl_fp36, sl_z, pre_gp, pre_mpg, pre_fp36, pre_z` (Summer League and preseason lines), `model`, `season`.

### Board: market and overlays

| Column | Meaning | Section |
|---|---|---|
| `adp` | ESPN average draft position | 10 |
| `adp_gap` | `adp - rank`; positive = model likes him more than the market | 10 |
| `blend_total_fp`, `blend_vorp`, `blend_rank`, `blend_tier` | the ADP-anchored season total and the VORP, rank and tier computed on it (only when an ADP blend is fit) | 10 |
| `risk_level` | `""`, `watch`, `high` | 11 |
| `risk_flags` | readable flag text (ESPN status, preseason games, new team, star arrived / left, and the return sentence) | 11, 12 |
| `risk_gp_haircut` | advisory share of games at risk, 0 to 0.20 | 11 |
| `risk_gp` | `proj_gp * (1 - risk_gp_haircut)` | 11 |
| `return_flag` | `returned-healthy` or blank | 12 |
| `return_block_pct` | lead block, percent of team games | 12 |
| `return_tail` | games played / team games since first appearance | 12 |
| `return_gp_upside_adv` | advisory 0 or 5 games, not in `proj_gp` | 12 |
| `return_fp_upside_adv` | advisory games times `proj_fppg` | 12 |
| `return_validation` | constant `advisory_judgement_not_projection` | 12 |
| `lm_flag` | `short-absences` or blank: 6+ games missed last season in 1-2 game interior absences (veteran, single team, `>= 20` mpg, `>= 41` GP); cause unknown | 18 |
| `lm_iso_n` | those games, last season (NaN outside the universe) | 18 |
| `lm_rest_n` | of them, one-game absences on the second night of a back-to-back (information only) | 18 |
| `lm_gp_risk_adv` | advisory 0 or -2 games, not in `proj_gp` | 18 |
| `lm_fp_risk_adv` | advisory games times `proj_fppg` | 18 |
| `lm_validation` | constant `advisory_descriptive_not_projection` | 18 |
| `contract_flag` | `contract year`, `new deal`, `extension`, `rookie final year (nominal)`, `rookie option year (nominal)`, blank | 13 |
| `contract_years_left` | seasons left including this one, known deals only | 13 |
| `contract_status` | `known` / `no_length` / `lapsed` / `unknown` | 13 |
| `contract_basis` | `wiki` / `rookie_scale` / blank | 13 |
| `contract_validation` | constant `unvalidated_display_only` | 13 |

### Breakouts tab

`watch_rank`, `team`, `draft_pick` (overall pick), `board_rank`, `base_fppg`, `layer_fppg`, `uplift`, `useful_prob`, `breakout_prob`, `changed_team` (roster-snapshot team differs from last season's ending team),
`sl_z` / `pre_z` (per-36 z-score vs cohort), `evidence` (text), plus the board columns above (section 14). Internal: `young`, `under_radar`, `adp_rank`, `contract_validation`.

### Transactions tab

`txn_date`, `kind` (`trade, signed, resigned, extended, converted, claimed, waived`), `person`, `position`, `rank`, `adp`, `from_abbr`, `to_abbr`, `detail` (e.g. `two_way`), `context`. Staff table: `txn_date`, `team_abbr`, `kind`
(`hired, fired, resigned_staff, extended_staff`), `person`, `role`. Internal: `relevant`, `first_seen`, `description`, `txn_key`.

### Coaches tab

`team, coach, since, new_to_team, previous_coach, prior_seasons, career_star, career_top5, career_depth10, career_pace, career_old, career_young`; second table `rank, name, position, team, coach, previous_coach,
first_time_head_coach`. Hidden but computed: `last_*` (team style last season), `shift_*` (expected change), `first_time_head_coach`.

### In-season page

| Column | Meaning |
|---|---|
| `ros_rank`, `ros_vorp` | order and VORP over the rest of the season |
| `gp`, `fppg_to_date` | games and FPPG this season through `as_of` |
| `prior_fppg` | preseason projection (`prior_mpg`, `prior_avail` for its components) |
| `ros_fppg`, `ros_mpg` | blended per-game production and minutes |
| `avail` | blended probability of playing a team game |
| `team_games_left`, `ros_games`, `ros_total_fp` | games left, expected games (`avail * left`), expected total |
| `w_sample` | `minutes / (300 + minutes)`, weight on this season's data |
| `has_prior` | False for players with no preseason projection |
| `repl_total` | replacement total used for `ros_vorp` |
| Schedule | `week, kind, start, end, n_days, games, games_left, b2b, off_night_games, per7, heavy, light, mean_games, min_games, max_games, heavy_teams, light_teams, b2b_total` |
| Waivers | `gain_ros_fp, drop, min_delta, usg_delta, rising_minutes, rising_usage, inherits_from, exp_gain_mpg, injury`; streaming: `week_team_games, week_exp_games, week_b2b, week_off_night, week_exp_fp, week_gain_fp` |
| Trade | `delta` (ROS FP gained or lost), `verdict`, `tolerance`, `before` / `after` roster value, `bench_weight`, `dropped`, `empty_slots` (see also the Trade analyzer app columns below) |

### Trade analyzer app columns (ADR 0027)

The columns shown in the Trade analyzer tab's own tables (`src/app/pages/inseason.py`), distinct names from the CLI's plain wording above so each one has its own glossary/tooltip entry:

| Column | Meaning |
|---|---|
| `trade_side` | `give` or `get`: which side of the trade this row is |
| `trade_player` | Player name on that side |
| `trade_ros_total_fp` | That player's rest-of-season total (`ros_total_fp`) |
| `trade_verdict` | `roughly even` / `favours you` / `favours the other side` (`TradeResult.verdict`) |
| `trade_confidence` | The five-way confidence bucket label (see section 16) |
| `trade_delta` | Rest-of-season FP gained (+) or lost (-) by the trade, all effects included |
| `trade_tolerance` | The `+-1.5%`-of-roster-value band `verdict`/the confidence bucket are read against |
| `trade_before_value` / `trade_after_value` | Roster value before and after the trade (slot-aware, section 16) |
| `trade_dropped` | Players force-dropped because the roster is overfull after the trade, or `none` |
| `trade_empty_slots` | Starting slots left empty after the trade, or `none` |
| `trade_partner` | The optional partner team label whose side is also scored (mirrors the CLI's `--partner`) |

Advisory notes/caveats shown below these tables (`inseason_view.trade_caveats`) are free text, not a column, so they carry no tooltip; each line names its own source (ESPN status, return-flag, lm_flag).

### Best available by need tab (ADR 0026)

`need_rank` (position in the `need_adj_vorp` ordering), `need_score` (0 to 1, this player's best-fit position's openness on your roster), `need_adj_vorp` (`vorp` plus a bonus scaled by `need_score`),
`need_position`, `need_capacity` / `need_filled` (per-position breakdown table only). None of these are written by `src.value.board` or the daily-refresh CSVs — computed on the fly for this tab. See section 22.

### Statistics abbreviations

`FP` fantasy points; `FPPG` FP per game; `GP` games played; `mpg` minutes per game; `VORP` value over replacement player; `ADP` average draft position; `SL` Summer League; `pre` preseason; `ROS` rest of season;
`b2b` back-to-back; `z` a standard score; `p10/p50/p90` the 10th/50th/90th percentiles.

## 21. Draft-day live sync (ADR 0025)

Auto-detects opponents' picks from the ESPN league sync (`src.ingest.espn_league`) during a live draft, instead of requiring
a manual "Mark drafted" click for every one. Logic lives in `src/app/live_sync.py` (plain, unit-tested, no Streamlit import);
`draft_board.py`'s **Live draft sync (ESPN)** sidebar section owns the `st.session_state` wiring and the status indicator.

**Mechanism.** Streamlit has no background timer, so "live" means polling ESPN's `mDraftDetail` view again on an explicit
**Sync now** click, or opportunistically on any rerun the app already performs (a tab switch, a manual pick) while enabled —
never faster than `DEFAULT_MIN_POLL_INTERVAL` (20s, `should_poll`), on top of `ESPNLeagueClient`'s own per-request rate
limit/backoff. Every poll is wrapped in `src.ops.runlock.RunLock` (the same file lock `src.ops.nightly` uses) keyed by
league id, so at most one session/browser tab hits ESPN for a given league at a time (`poll_lock`); a lock older than 60s is
reclaimed as stale rather than blocking the rest of a live draft.

**Diff and attribution.** `diff_new_picks` keeps only picks with a filled `espn_player_id` not already seen this session;
`classify_team` attributes a pick to `"me"` only when its ESPN `team_id` matches `$ESPN_TEAM_ID` (or the team id already
saved by `python -m src.ops.nightly --set-team`, `default_team_id`), else `"opponent"`. `resolve_player` maps ESPN's player
id onto this project's own `player_id` through the `player_id_map` contract table (`python -m src.ingest.espn_adp`); a pick
with no map entry yet is returned as **unresolved**, not silently dropped, so the user can mark it by hand.

**Applying picks and avoiding double-counting.** `apply_detected_picks` runs every resolved pick through the exact same
`draft_player` function the manual "Mark drafted" form uses. A pick whose `player_id` is already in the draft state (most
often because the user's own manual click logged it first) is classified `duplicate` and skipped — never double-counted,
regardless of which path saw it first.

**Failure handling.** Every `ESPNLeagueError` (auth, not-found, offline-cache-miss, retries exhausted) is caught inside
`sync_once` and turned into `SyncResult(ok=False, error=...)`; the sidebar shows it in the status caption
(`_live_sync_status_caption`) alongside the last-synced time, and the manual "Mark drafted" control is never disabled by
any of this. See `docs/adr/0025-draft-live-sync.md` for the full design writeup and rejected alternatives.

## 22. Best available by need

[ADR 0026](adr/0026-need-adjusted-board.md) crosses the undrafted board against the user's own drafted roster and `config/league.yaml`'s roster shape (starters, flex `G`/`F`, `UTIL`, bench — IR excluded).
It never changes `vorp`, `rank`, `tier` or any projection column; it adds a second, additive ranking on a new tab.

- **Capacity.** Each specific position (PG/SG/SF/PF/C) gets a fractional roster "capacity": its own starter slot, plus an even share of every flex slot that reaches it, plus an even share of `UTIL`
  and the bench (anyone can fill either). Example, this league's roster: PG = 1 (starter) + 0.5 (G flex) + 0.6 (UTIL) + 0.6 (bench) = 2.7; C = 1 + 0.6 + 0.6 = 2.2 (no flex reaches C).
- **Filled.** How many of the user's drafted players are eligible at each specific position (`src.value.positions.eligible_positions`); a multi-position player counts toward every position he qualifies for.
- **Need score**, per position: `clamp((capacity - filled) / capacity, 0, 1)`; 0 if the league has no slot there or it is already fully covered, 1 if nothing is drafted there yet. A player's own need
  score is the best of his eligible positions' scores (0 for an unparseable position).
- **`need_adj_vorp = vorp + 40 * need_score`** — additive (not multiplicative, since `vorp` can be negative), a documented, **unvalidated** heuristic bonus, not a projection input.
- **Empty-roster fallback**: with nothing drafted every position scores 1.0, so the bonus is the same constant for every player and the ordering is unchanged from plain VORP — the tab says so explicitly.
- **Categories/punt leagues**: not applicable. `config/league.yaml` is `h2h_points`; `is_categories_league()` is the single gate that would enable a category-need computation for a categories/roto
  league, and returns `False` here. No category-need scoring is implemented.
- Code: `src/value/need.py` (model), `src/app/state.py::need_board_view` (orchestration), `src/app/draft_board.py::_need_tab` (UI). Tests: `tests/value/test_need.py`, `tests/app/test_state.py`.

## 23. External rankings comparison

[ADR 0028](adr/0028-external-rankings-compare.md) compares this project's own board rank against two manually-exported snapshots: Yahoo's Fantasy
Basketball draft-analysis workbook and FantasyPros' consensus draft rankings CSV. Both are one-off downloads the user refreshes by hand (no live
API, unlike [ADR 0007](adr/0007-adp-ingest.md)'s ESPN/FantasyPros ADP pull) — see the ADR for where the raw files live
(`<NBA_DATA_DIR>/external_rankings/`, never committed) and how to refresh them.

- **Parsing.** `src.ingest.yahoo_rankings.parse_yahoo_xlsx` reads the "Players" sheet, locating the header row by content rather than a hardcoded
  row-skip; ADP prefers "All Drafts ADP", falling back to "Preseason ADP". `src.ingest.fantasypros_rankings.parse_fantasypros_csv` regex-parses
  name/team/positions/an optional trailing status tag (`OUT`/`DTD`/`TWO-WAY`/`RET`/anything else FantasyPros adds later) out of the single
  `PLAYER NAME` field, and turns the `"-"` "no ADP data" sentinel in `ECR VS. ADP` into a real null.
- **Matching.** Both parsers resolve names onto `player_id` through `src.ingest.external_rankings.resolve_players`, which wraps
  `src.ingest.id_map.match_players` (section reused unmodified from ADR 0007's ingest infrastructure). Every input row survives into the
  resolved frame, matched or not; `match_method` is `exact | normalized | fuzzy | ambiguous | unmatched`. Real-data match rate (2026-09-28
  snapshot): Yahoo 645/686 (94.0%), FantasyPros 308/308 (100%).
- **Comparison.** `src.value.rankings_compare.build_comparison` outer-joins our board against both sources on `player_id`. Sign convention:
  `rank_delta_<source> = our_rank - <source>_rank` — **positive** means the external source ranks the player earlier (more favorably) than we
  do, matching the existing `adp_gap` convention (section 10). `on_our_board`/`on_yahoo`/`on_fantasypros` flags make coverage gaps visible;
  FantasyPros only covering ~300 players is expected (its own published range), not an anomaly. `unmatched_report` surfaces every
  unmatched/ambiguous name from either source, never silently dropped.
- **UI.** A new page, `src/app/pages/external_rankings.py` (not a `draft_board.py` tab — this comparison is not draft-state-specific): tabs
  All players, Biggest disagreements, Unmatched names. Every table column has a glossary/tooltip entry (`src/app/glossary.py`, group "External
  rankings comparison (ADR 0028)").
- Code: `src/ingest/external_rankings.py`, `src/ingest/yahoo_rankings.py`, `src/ingest/fantasypros_rankings.py`, `src/value/rankings_compare.py`,
  `src/app/pages/external_rankings.py`. Tests: `tests/ingest/test_external_rankings.py`, `tests/ingest/test_yahoo_rankings.py`,
  `tests/ingest/test_fantasypros_rankings.py`, `tests/value/test_rankings_compare.py`, `tests/app/test_external_rankings_glossary.py`.

## 24. External rankings disagreement flag

[ADR 0029](adr/0029-rankings-disagreement-flag.md) closes the loop on section 23: an advisory, opt-in flag that crosses section 23's rank
comparison against the live draft board, so a sharp our-board-vs-consensus disagreement is visible while drafting, not only on the separate
External rankings page. Same discipline as sections 11-13/18/22: never changes `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp`, `tier` or `rank`.

- **Flag.** `src.value.rankings_compare.flag_disagreements` (called automatically by `build_comparison`, section 23) adds
  `rankings_disagreement` (`True` when `max_abs_rank_delta >= DISAGREEMENT_THRESHOLD`), `rankings_disagreement_direction`
  (`market_favors` when an external source ranks the player earlier than we do, `we_favor` when we rank him earlier — named for whichever of
  `rank_delta_yahoo`/`rank_delta_fantasypros` has the larger magnitude), and `rankings_disagreement_text` (a one-line sentence naming the
  driving source and both ranks).
- **Threshold: `DISAGREEMENT_THRESHOLD = 50` rank spots, either direction** — a fixed judgement grounded in the real 2026-09-28 snapshot, not a
  fitted number. Unrestricted, `max_abs_rank_delta` across all 648 comparable players has median 61 and p75 108 (dominated by deep-bench noise
  where coverage is thin); restricted to the realistic draft pool (`our_rank <= 150`), median is 36.5 and p75 is 66.0 (`our_rank <= 100`: median
  25.0, p75 50.2). 50 sits just above the top-100 pool's own 75th percentile — high enough not to fire on ordinary rank noise near the top of
  the board, low enough to still catch a meaningful share of real disagreements. See the ADR for the full distribution and the open caveat
  (checked against one snapshot only).
- **Session-scoped, not baked into the CLI or `reports/daily/*.csv`** — unlike sections 12/18/13's overlays (computed from `History`, always
  available for the unattended daily/nightly jobs), the external rankings snapshot is a manual, occasional refresh (section 23), so this reuses
  section 22's app-only wiring pattern instead: the app reads whatever `st.session_state.rankings_cmp` the External rankings page already built
  this session.
- **UI.** No new page or sidebar control. The **Debutants & risk** tab (section 19) gets a new "External rankings disagreement" table after the
  contract-flag lists: undrafted players carrying the flag, sorted by `max_abs_rank_delta` descending, with a caption naming the threshold and
  restating "advisory only". If nothing has been loaded on the External rankings page yet, an info message says so.
- Code: `src/value/rankings_compare.py` (`DISAGREEMENT_THRESHOLD`, `flag_disagreements`), `src/app/state.py`
  (`rankings_disagreement_table`), `src/app/draft_board.py` (`_debutants_risk_tab`). Tests: `tests/value/test_rankings_compare.py`,
  `tests/app/test_rankings_disagreement_app.py`.
