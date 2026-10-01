# NBA Fantasy Draft Scouter — Planning Doc

**Status:** Brainstorm → plan (v2)
**League format:** ESPN, Head-to-Head **Points**
**Goal:** Rank players by projected fantasy output for our league's scoring, prove the model with a walk-forward backtest, then extend it into a live in-season tool.

---

## 1. Concept

A projection engine that ranks players **using only data available before a season**, then scores itself against what actually happened. The backtest is the portfolio centerpiece. The same engine later powers live tools (draft board, trades, waivers).

```
Ingest → Feature store → Projection model → Fantasy points engine → Rankings → Backtest + Live app
```

---

## 1a. Status as of 2026-09-28

Built in parallel across five tracks (each in its own git worktree, integrated onto `main` via `./dev integrate`), plus several rounds of independent-verifier-driven fixes (two real-data leakage-guard crashes from the same pandas-nullable-dtype hazard, a Windows console encoding crash, a markdown-rendering bug in backtest reports found while writing up the first real results below, and stale docs — including this line, more than once; treat any specific test count here as a snapshot, not a promise, and prefer `./dev test`'s own output). 1,964 tests collected as of this writing (2026-09-29, `./dev test --collect-only`, up from 1,415 on 2026-09-25 as ADRs 0025-0028 landed); run `./dev test` for the current pass/fail count.

**Done:**
- **Shared data contract** (`src/contracts.py`, `src/store.py`): table schemas, atomic parquet IO, the `History`/`Projector` leakage-guard primitives. ADR 0001.
- **Synthetic league** (`src/synthetic.py`): deterministic, contract-valid fake NBA data used to build and test everything below before real data existed.
- **Fantasy points engine** (`src/value/points.py`, `frame.py`, `league.py`): scalar and vectorized, driven entirely by `config/league.yaml`.
- **Real NBA data ingest** (`src/ingest/`): 11 seasons, 2015-16 through 2025-26, pulled from `nba_api` into `~/dev-data/nba-fantasy-2026`. 281,138 game-log rows, 26,418 team-game rows, 1,554 players, 5,968 player-seasons. Zero box-score identity violations; 25/25 known-fact sanity checks passed. ADR 0002; `docs/data-quality.md` has the full report, including documented anomalies (COVID-shortened 2019-20 and 2020-21 seasons, minor source-side score mismatches).
- **Projection model + value engine** (`src/models/`, `src/value/vorp.py`, `replacement.py`, `tiers.py`, `positions.py`, `board.py`): baseline projector (minutes, per-minute rates with empirical-Bayes shrinkage, data-derived age curves, availability distribution, floor/median/ceiling), naive-last-season benchmark, a rookie prior by draft slot, replacement level and VORP driven by league config (tested at both 10 and 13 teams), gap-based tiers, and a draft-board CLI (`python -m src.value.board`). ADR 0003; `docs/modeling.md`.
- **Walk-forward backtest harness** (`src/backtest/`): per-season actuals, Spearman/top-K/NDCG/MAE/RMSE/VORP-weighted-error metrics with bootstrap CIs, the leakage guards described below, ablation scaffolding, miss analysis, markdown+chart reporting, and a CLI (`python -m src.backtest`). ADR 0004; `docs/backtest.md`.
- **Real backtest results, 2016-17 through 2025-26** (see README's [Results](docs/project-overview.md#results) for the full honest write-up; report at `reports/real_2026-09-22/`, not committed — regenerate with the command in the README). Headline: the baseline projector beats naive-last-season on rank correlation (Spearman total FP 0.784 vs 0.763, +0.009, 95% CI excludes 0, won 8/10 seasons) and top-of-draft precision (top-12 hit rate 0.542 vs 0.442), but *not* on error magnitude (MAE total FP 422.6 vs 426.6, paired-bootstrap verdict `hurts`). Floor/ceiling calibration is good (78.4% of games inside the projected p10-p90 band vs 80% nominal). ~70% of miss error is availability-driven (injuries/DNPs) — the clearest signal for what to build next.
- **Leakage protection**, hardened twice: the backtest track's own perturbation tests caught a real hole (the static `players` table's `from_year`/`to_year`/`draft_year` columns revealed future debuts, retirements, and draft classes to anything reading `History` directly, not just the backtest harness). Fixed at the source in `History.until` itself, robustly — a player is kept if drafted by the season's draft or if they actually appear in the season-sliced game logs, never by trusting the (occasionally noisy) `from_year` field alone.
- **Source research** (`docs/research/`, ADR 0005): verified findings on ESPN's public player/ADP/schedule endpoints (no cookies needed for our public league), historical ADP coverage 2015-16 to 2026-27 (with a documented gap and a fallback), NBA injury-report PDF availability from 2018-12-19, and rejected/deferred sources (contracts, Pro Sports Transactions, balldontlie) with reasons.
- **Repo tooling**: `./dev` (worktree create/remove/list, test, locked integrate, lint, ci) hardened with real integration tests; GitHub Actions CI (Ubuntu + Windows) and dependabot; `README.md`, `docs/architecture.md`, ADR index, `CONTRIBUTING.md`.

**Open / not yet built:**
- **Draft date and time SET** (date 2026-09-26, clock time confirmed 2026-10-02): 2026-10-17 07:00 NZDT = 2026-10-16 18:00 UTC = 20:00 CEST (`config/league.yaml` `draft.*`; ADR 0014 notes 2026-09-26 and 2026-10-02).
- **In-season nightly refresh of rest-of-season rankings and league alerts** (roadmap phase 5): BUILT 2026-09-25 (ADR 0018, `python -m src.ops.nightly`, task "NBA Fantasy 2026 Nightly In-Season", daily 09:30 local 2026-10-20 to 2027-04-06). Exercised live only before the season (all steps ran on 2026-09-25; the games step correctly skipped, no team configured); the incremental game ingest against the real stats.nba.com, the team artifacts on real rosters and the scheduled firing are unproven until opening night. Older text follows. The *pre-draft* daily refresh is done (ADR 0014, below), but it does not run the in-season pieces: no scheduled job invokes `ros`, `trade`, `waivers` or the schedule pull nightly, and no league-alert step (roster or waiver changes) exists; nothing can run live until the season starts; the rest-of-season projection, trade analyzer, waiver finder and schedule awareness are built (ADR 0015, below) but cannot run on a live season until it starts.

**Resolved:**
- **Hurdle availability, ADP as an input, ADP blend, season-total intervals, labelled injury data** (ADR 0031, 0032, 0033), 2026-10-01: the six model-improvement items
  from the post-ADR-0030 review. (1) `baseline_hurdle*` models P(appear) and projects the unconditional games played (walk-forward: Spearman 0.784 to 0.810, MAE of
  total FP 422.5 to 367.5, returners' games bias +15.3 to +1.5). (2) and (3) ADP feeds the availability, appearance and minutes models (`baseline_hurdle_adp*`) and
  the board carries an ADP-anchored second ordering (`blend_*`, `python -m src.value.adp_blend fit`); the best measured model and the app's default on real data is the stack
  `baseline_hurdle_adp_offseason_debut`. (4) the hurdle models emit a calibrated-by-risk-group season-total band (`proj_total_fp_p10/p50/p90`). (5) the NBA injury-report
  archive, its reason-text features and a nightly `injuries` step exist (`src.ingest.nba_injury_reports`, `src.features.injury_labels`); see ADR 0033 for what the ablation found.
  (6) point-in-time positions: closed by evidence, not built (the `baseline_nopos` ablation shows the position priors carry no measurable benefit).
  Still open from this batch: rookies' and debutants' own chance of not appearing (drafted-but-never-played rookies are not in `players`), so the rookie season-total band is too
  narrow; the in-season live run of the `injuries` nightly step (first real night 2026-10-20); refit `adp_blend.json` right before the draft after the final ADP refresh.
- **External rankings comparison** (ADR 0028), 2026-09-28: manually-exported Yahoo and FantasyPros draft rankings, matched onto `player_id` (`src/ingest/yahoo_rankings.py`, `fantasypros_rankings.py`, `src/value/rankings_compare.py`) and shown against our own board rank on an app page (`src/app/pages/external_rankings.py`). Snapshot files live outside the repo under `NBA_DATA_DIR`; refreshed by hand whenever the user re-exports them, not a live feed.
- **Trade analyzer confidence UI** (ADR 0027), 2026-09-27: the in-season app's Trade analyzer tab adds a five-way magnitude-bucketed confidence read (`clear win` to `clear loss`) on the existing `+-1.5%` tolerance band, per-player advisory caveats (ESPN injury status, return/load-management flags), and an optional partner-side score. A read on the existing statistic, not a new one; no projection change.
- **"Best available by need"** (ADR 0026), 2026-09-27: a second, additive ranking (`need_score`/`need_adj_vorp`) that crosses the undrafted board against the user's own drafted roster's positional gaps from `config/league.yaml`. Documented, unvalidated heuristic; falls back to plain VORP order with an empty roster; no change to `vorp` itself.
- **Draft-day live sync** (ADR 0025), 2026-09-28: `src/app/live_sync.py` polls the ESPN league sync (`src/ingest/espn_league.py`'s draft detail) for opponents' newly-filled picks and marks them drafted automatically, attributed by ESPN team id, gated by a floor of 20s between polls (`should_poll`) and a `RunLock` so only one session polls a given league at a time. The manual "Mark drafted" form remains the permanent fallback on any network/auth failure or unsynced league.
- **Coach and staff impact** (ADR 0020), 2026-09-25: head coach per team-season (Wikipedia infoboxes plus the current-coaches list for 2026-27: six new head coaches). Style follows the coach for top-five minutes, rotation depth and three-point share (coefficients +0.32 to +0.43; star minutes probably); pace and youth development do not travel with a coach. A `baseline_coach` minutes layer, ablated on ten real seasons, gave no significant change (not recommended); coach facts are descriptive (`python -m src.value.coaches`, app tab, daily-report coach-change line). Assistants and front office are not analysable with compliant data. See `docs/coaches.md`.
- **Live transactions tracker** (ADR 0017), 2026-09-25: ESPN's public transactions feed into a dated, append-only `league_transactions` ledger (trades, signings, extensions, waivers, coach and front-office moves), `python -m src.ops.txn_watch`, a daily-report section and an app tab, ranked against the board. Descriptive only (ADR 0010/0011 found no projection lift from team-change signals). Cross-check: 207 of 209 per-player latest moves agree with the NBA roster snapshot; the feed is not complete (one arrival has no row), so the roster diff remains the safety net. See `docs/transactions.md`.
- **UI:** Streamlit. **Storage:** DuckDB as a query layer over the existing parquet store, not a replacement for it. See section 12.
- **NBA.com's fantasy-use restriction and scripted ESPN access** (both flagged by ADR 0005): the user has accepted both — data stays private, non-commercial, and uncommitted, and ESPN access stays low-volume and local-only.
- **Team count**: confirmed **13** against the real live ESPN league (id 1234567890, public, no cookies needed) on 2026-09-23 — was provisional at 10. `config/league.yaml` and every VORP/replacement-level computation now use the real number. See ADR 0008.
- **ADP benchmark table** (`adp`): done. `python -m src.ingest.espn_adp` (ADR 0007) populates `adp` and `player_id_map`; the "ADP / consensus" results row is real (README Results). Headline: ESPN's ADP beats the baseline projector at the top of the draft board (top-12/50/100 hit rate, NDCG), matching PLANNING's own framing that matching ADP is the hard bar.
- **Injury feature layer** (ADR 0006): done — a real, statistically genuine but small lift over the baseline (README Results has the numbers).
- **Return-from-injury study** (ADR 0021), 2026-09-26: analysis only, no model change. For players with one long absence block and a healthy tail (Tatum's profile), the baseline is about right in absolute terms (n = 60: +1.0 GP [-4.1, +5.8], +62 total FP [-80, +196]) and low only relative to matched spread-absence controls, whom it over-projects; a >= 60%-block subgroup hints at about +8 GP (n = 21) but is one of 18 cells and not established.
- **Return-health availability feature** (ADR 0022), 2026-09-26: the 'long lead block, then healthy' columns (`baseline_return`, `baseline_injury_return`, `baseline_return_offseason_debut`) were built and ablated under a pre-registered rule; no lift (MAE total FP -0.47 [-1.31, +0.29], MAE GP -0.037 [-0.085, +0.011], wins 2 and 3 of 10), and it over-corrects the ADR 0021 cohort (Tatum 50.1 -> 57.7 GP). Registered, not recommended; the board is unchanged.
- **Advisory return flag** (ADR 0023), 2026-09-26: the board shows `return_flag` / `return_tail` (Tatum and eight others) with a fixed advisory judgement of +5 GP (x `proj_fppg` in total FP) for a lead block >= 60%, per ADR 0021's recommendation. Display only: projections, VORP, ranks and `risk_*` are byte-identical; the app now states that the rank is by total fantasy points, not FPPG.
- **Load-management study and advisory flag** (ADR 0024), 2026-09-26: no data labels why a game was missed, so load management was proxied by the shape of last season's absences and tested walk-forward over ten seasons (n = 1,830 rotation veterans). Games missed in 1-2 game absences predict a *lower* next season than the baseline projects (-1.4 GP [-2.4, -0.4], -53 total FP [-88, -20] per SD, 9 of 10 seasons), but not the back-to-back "rest" proxy (-3 FP [-36, +30]) or the 65-game marker, and correcting projections with it does not lower out-of-sample error. Descriptive association, not incremental: projections unchanged; the board gets an advisory `lm_flag` (6+ such games) with a fixed -2 GP judgement, cause unknown. Needs a labelled rest feed to go further.
- **External rankings disagreement flag** (ADR 0029), 2026-09-29: closes the loop on ADR 0028's read-only Yahoo/FantasyPros comparison, crossing it onto the live draft board. `flag_disagreements` marks players where our board and the external consensus disagree by 50+ rank spots either direction — the threshold checked against the real 2026-09-28 snapshot restricted to the realistic draft pool (`our_rank <= 150`: median |rank_delta| 36.5, p75 66.0), not fit to any outcome (there is no "who was right" ground truth to validate against). Session-scoped, like the need-adjusted board (ADR 0026), not baked into `src.value.board`'s CLI or the daily/nightly CSVs, since the snapshot files are a manual, occasional refresh. Surfaced as a new table on the **Debutants & risk** tab when a comparison is loaded; `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp`, `tier` and `rank` are unchanged.
- **Roster-context feature layer** (ADR 0010): done — ablated on the real walk-forward backtest, honest result is **no lift, mildly hurts MAE, not recommended for use** (README Results has the numbers and the reasoning). Kept as a complete, tested, documented negative result.
- **Contract layer** (ADR 0013): done 2026-09-24 as far as the data allows. No source has dated salary or contract data (ADR 0005 D4), so only the rookie-scale clock (from draft year and pick: scale year, option year, contract year / extension-eligible year, post-scale year, second-round early years) was built and ablated (`baseline_contract`, `python -m src.backtest --ablate baseline,baseline_contract`). Real ten-season result: **no meaningful change** (Spearman +0.0004, 3/10 seasons; top-50 0.000; MAE -0.04 n.s.), not recommended, no change to the draft board; veteran contract years and salary remain untestable. The ablation table has no pending row.
- **Contract-terms layer** (ADR 0019): done 2026-09-25 — veteran contract years parsed from the Wikipedia wikitext (`player_contracts`, 2,653 events, 953 with a stated length, partial coverage) and tested on the ten-season walk-forward: no reliable lift on rank order (Spearman -0.0003, top-50 -0.004), MAE -0.3%, contract-year premium not supported; not recommended; shown only as an unvalidated display flag.
- **Roster-transactions feature layer** (ADR 0011): done — a real, dated arrival/departure signal (330 Wikipedia team-season pages ingested, 5,452 transactions, 84.9% player match rate) purpose-built to close ADR 0010's identified gap. Honest result is **worse than the roster layer**: measurably hurts rank order (Spearman `hurts`, MAE `hurts` by more than ADR 0010), diagnosed down to a specific, understood mechanism (a single linear coefficient can't distinguish a downgrade move from an opportunity move) rather than left a mystery. Not recommended for use. Contract layer remains the only fully open ablation row.
- **Live draft board** (roadmap phase 4): done — `streamlit run src/app/draft_board.py`; see `docs/app.md`, ADR 0009.
- **Preseason roster update** (risk table row; ADR 0012): done 2026-09-24. The incoming draft class (52 rostered rookies, real slots and birthdates) is now in `players`, so the 2026-27 board has 739 players instead of 687 and includes every rookie; dated `roster_snapshots` record who is on which roster each day and can be diffed until the draft; `python -m src.ingest.preseason_refresh` runs roster, Summer League/preseason and ADP in dependency order in about 15 seconds. Five rostered debutants drafted in earlier years (Sorber, Marković, Toohey, Diop, Biberovic) were named but not projected; the offseason gaps track (ADR 0016, below) now projects them in the `baseline_debut` / `baseline_offseason_debut` models.
- **Offseason gaps** (ADR 0016), 2026-09-24. **(1) Stash debutants: closed.** Sorber, Marković, Toohey, Diop and Biberovic are projected (draft-slot prior at their own pick, low confidence) by `baseline_debut` and `baseline_offseason_debut`; a walk-forward analogue (28 stash players who debuted 1+ years after their draft, 2018-19..2025-26, drawn from preseason camp rosters) scores the treatment. The stash play probability (0.90) is an assumption history cannot identify. **(2) Undrafted signees / two-way / Exhibit-10: closed** (about 50 on the board as of 2026-09-25, tiny totals by design; play-probability AUC 0.69 on 825 camp participants, 65% of them never play). The plain `baseline*` boards are unchanged (pinned to 1e-12), so they still omit these players. **(3) College and international production: open, precisely.** No compliant, machine-readable, historically complete source exists (Sports-Reference, Spotrac, HoopsHype, RealGM and Barttorvik forbid or block automation; stats.ncaa.org is bot-blocked; ESPN and Wikipedia are not usable for this at 10-season scale); country, previous-organisation type, draft age, size and combine measurements are ingested as proxies (`player_profiles`, `draft_combine`) and lower rookie error about 3.5% (real, 7-8 of 8 seasons) but do not move the whole-league metrics, so `baseline_origin` is registered and not recommended. **(4) Preseason injuries and team context: partially closed.** ESPN injury status, preseason absence, new team and star arrival/departure are flags on the board, watchlist and app (`risk_level`, `risk_flags`, `risk_gp`) with **no change to any projection**; the NBA injury-report PDFs do not exist before opening week (probed 2024 and 2025), ESPN status has no history (now archived daily in `espn_status_snapshots`), and the preseason-absence backtest is confounded (no historical rosters), so none of the haircuts is validated. See `docs/offseason.md`.
- **Summer League and preseason layer, and breakout watchlist** (ADR 0012): done, honest mixed result. 11 summers (no 2020) and 11 preseasons ingested (39,836 player-games; real-data quirks handled and pinned by tests). `baseline_offseason` improves per-game and VORP-weighted accuracy, and finds young under-the-radar breakouts at about 2.3 times the base rate (top ten: 24% useful breakout vs 10%). **The signal is the preseason; Summer League alone is close to noise**, which contradicts the premise the feature was requested on and is reported as such. Also fixed: the rookie availability prior over-projected picks 1-3 by 18 games (capped). See `docs/offseason.md`.
- **ESPN league sync** (part of roadmap phase 5): done for read-only settings/roster/free-agent sync against our real league; see ADR 0008. Trade/waiver tools and schedule awareness are built (ADR 0015); the in-season nightly refresh is built (ADR 0018, task installed; see the first Open item above) but has only been exercised before the season, so its first live run is opening night 2026-10-20 (the incremental game ingest against the real stats.nba.com, the team artifacts on real rosters and the scheduled firing are unproven until then).
- **In-season tools** (roadmap phase 5; ADR 0015): built 2026-09-24, tested on synthetic data and smoke-run on the real 2025-26 season as if mid-season (leakage-safe `--as-of`). `python -m src.inseason.schedule` (ESPN pro schedule, one cached request, `schedule_games` table, games per team per matchup week, back-to-backs, off nights, heavy/light, playoff weeks; matchup weeks are *derived* because ESPN still returned a placeholder for the league's `matchupPeriods`, and the derivation reproduces the league's `finalScoringPeriod` 167 exactly), `ros` (preseason projection shrunk toward season-to-date; out of sample on 2021-22..2025-26 it beats preseason-only in 5/5 seasons on every metric and season-to-date-only in 5/5 seasons on MAE but only +0.008 Spearman), `trade` (slot-aware matching, freed/filled roster spots), `waivers` (ROS gain over the drop, rising-minutes/usage and injury-beneficiary flags, streaming by games in the week) and an app page. Untested against real league rosters until the draft happens. See `docs/inseason.md`.
- **Daily pre-draft refresh automation** (part of roadmap phase 5; ADR 0014): done 2026-09-24. `python -m src.ops.daily_refresh` runs roster, Summer League/preseason, ADP, a dated ESPN snapshot archive, the league sync, the breakout watchlist and draft-board CSVs, then writes a diffed report (`reports/daily/latest.md`); it is registered with Windows Task Scheduler (`./dev schedule install`, twice daily, catch-up after sleep, self-expiring after the draft window) on this machine. GitHub Actions cron was rejected (stats.nba.com blocks datacenter IPs; ESPN terms; data is never committed). It runs only while the computer is on and the user logged in.
- **Draft board run for real** (roadmap phase 1): done on 2026-09-23. `python -m src.value.board --season 2026-27 --model baseline` produces a real, complete 687-player board (Jokić/Wembanyama/Dončić/SGA at the top, matching the model's own logic — no surprises). The CLI's `--adp` flag was extended to accept the raw ingested ADP table (`adp.parquet`, `season/source/source_id/adp`) directly, mapped to `player_id` via `player_id_map` automatically, instead of requiring a manual join first — real run: 197 of 211 raw ESPN ADP rows for 2026-27 mapped (an initial version of this stat mixed in counts from other seasons in the same file, caught in independent verification and fixed; see the multi-season regression test in `tests/value/test_value_b_board.py`). **Correction (2026-09-24):** an earlier version of this note called the 14 unmapped rows "expected for fringe players". That was wrong: they were the entire 2026 rookie class, missing from the `players` table (and so from the board). Fixed by the preseason roster update below; unmapped 2026-27 rows are now 2, both non-NBA internationals. The Streamlit app (`streamlit run src/app/draft_board.py`) was exercised live against the same real data: loaded the 687-player board, marked and undid a real pick, checked the Drafted tab and best-available counts updated correctly, with clean browser console and server logs throughout. See `docs/app.md`.

---

## 2. League Settings (config-driven)

Everything league-specific lives in `config/league.yaml` so the engine is reusable. Values below were copied from the ESPN settings page on 2026-09-22.

| Setting | Value |
|---|---|
| Teams / format | **13** (confirmed live against the real ESPN league, 2026-09-23), H2H Points, weekly matchups, 20 regular-season weeks, 8 playoff teams |
| Draft | Snake, 90 s/pick, order set manually, pick trading allowed, **2026-10-17 morning NZ time** (set 2026-09-26; start 2026-10-16 18:00 UTC = 07:00 NZDT, confirmed 2026-10-02 — see section 1a and `config/league.yaml` `draft.*`) |
| Starters (10) | PG, SG, SF, PF, C, G, F, UTIL x3 |
| Bench / IR | 3 bench + 1 IR (13 rostered) |
| Lineups | Daily, lock at each player's game time |
| Acquisitions | Waivers (1 day), no season cap, **7 per matchup** |
| Trades | No limit, deadline 2027-03-13 |
| Keepers | None |

**Scoring:** PTS 1, REB 1, AST 2, STL 4, BLK 4, TO -2, FGM 2, FGA -1, FTM 1, FTA -1, 3PM 1. No DD/TD or bonuses.

---

## 3. Fantasy Value Engine (H2H points)

Points leagues make the value model simpler than 9-cat but change what matters:

- **Fantasy points per game (FPPG)** = Σ (projected stat × scoring weight).
- **Projected season value** = FPPG × projected games played.
- **Value over replacement (VORP):** replacement level is derived from league size and roster slots (how deep the league goes, how many UTIL/bench spots). ESPN's flexible slots usually mean weak positional scarcity, so verify this from data rather than assume.
- **No category balance or punting.** Raw volume wins. Minutes, usage, and heavily weighted stats (STL/BLK if weighted high, turnovers if penalized) drive value.
- **Variance matters for H2H.** Track game-to-game FP standard deviation, floor, median, and ceiling. A consistent 40 FPPG player and a volatile one are not equal in weekly matchups.
- **Tiers:** group players by projected value gaps. Tier cliffs matter more than exact rank at the draft table.

**Output per player:** projected FPPG, projected games, projected total FP, VORP, floor/median/ceiling, tier, ADP gap (value vs. draft cost).

---

## 4. Projection Model

Model components separately, then combine:

1. **Minutes:** the biggest driver of points-league output.
2. **Usage / role:** shot volume, assists, rebound share.
3. **Per-minute production:** efficiency and stat rates, regressed to the mean for outlier seasons.
4. **Availability (games played):** modeled as a distribution, not a point estimate.

### Feature layers
| Layer | Features |
|---|---|
| Baseline | Multi-season rolling stats, age curves, regression to the mean |
| Injury | Games missed per season, days since last injury, injury type, recurrence, typical recovery time by injury type |
| Roster context | Projected role (No. 1 vs. No. 4 option), usage/minutes redistribution when a star arrives or leaves (with/without splits), positional depth, team pace, coach/system (**coach/system built and ablated, ADR 0020: no significant lift**) |
| Contract | Contract-year flag, years remaining, salary. **Hypothesis; keep it only if the backtest shows lift. Veteran contract years measured in ADR 0019 (Wikipedia events, partial coverage): no lift on rank order, not recommended; unvalidated display flag only.** **Built for the rookie-scale slice only (ADR 0013): no lift, not recommended**; salary and veteran contract years have no point-in-time source |
| Rookies / returners | Separate path using draft position and college/international stats |

Start with simple, explainable models (regularized regression, gradient boosting) before anything fancy.

---

## 5. Backtesting

- **Walk-forward:** for season S, train only on data through S-1, rank players, compare with actual season-S fantasy points under our scoring. Repeat across ~10 seasons (2015-16 through 2025-26).
- **Metrics:** Spearman rank correlation, top-50/top-100 hit rate, error in FPPG and total FP, VORP-weighted error.
- **Benchmarks:** (1) naive last-season ranking, (2) ESPN ADP/consensus if obtainable. Beating naive is realistic; matching ADP is a strong result.
- **Ablation table:** baseline → +injury → +roster → +contract, showing the lift of each layer.
- **Leakage guard:** every feature is computed "as of date" (contracts, injury status, roster). Add tests that fail if future data leaks in.
- **Miss analysis:** write up the biggest misses and why (injury, trade, role change).

---

## 6. Live / In-Season Features

- **Draft mode:** live board that updates as picks are made, with best-available and VORP-based suggestions (format depends on snake vs. auction).
- **Nightly refresh:** scheduled job updates data, projections, and rest-of-season rankings. The pre-draft data refresh is built as a Windows Task Scheduler job on the user's machine (ADR 0014; GitHub Actions cron was rejected there); the rest-of-season projection and the other in-season tools now run in the unattended nightly job (ADR 0018), untested on a live season.
- **Trade analyzer:** rest-of-season projected FP in vs. out, adjusted for roster slots. **Built** (ADR 0015, `python -m src.inseason.trade`).
- **Waiver / free-agent finder:** rising minutes or usage, injury-replacement beneficiaries. **Built** (ADR 0015, `python -m src.inseason.waivers`).
- **Schedule awareness (H2H weekly):** games per week and playoff-week schedules for streaming and start/sit decisions. **Built** (ADR 0015, `python -m src.inseason.schedule`).
- **Injury alerts:** who gains minutes when a teammate goes down.
- **ESPN integration:** pull our league's rosters, settings, and free agents.

---

## 7. Data Sources (verify in Phase 0)

| Need | Candidates |
|---|---|
| Stats / game logs | `nba_api` (unofficial), balldontlie |
| Injuries / transactions | Official NBA injury report, ESPN, Pro Sports Transactions |
| Contracts | Spotrac, HoopsHype |
| League data | ESPN fantasy API via the community `espn-api` Python package (unofficial) |
| ADP | ESPN (if accessible), other public ADP sources |

**Notes**
- Check terms of service and rate limits before automating any scraping. This matters for a public repo.
- Private ESPN leagues require `espn_s2` and `SWID` cookies. Keep them in `.env`, gitignored, never committed.
- Cache raw pulls locally so backtests are reproducible offline.

---

## 8. Tech Stack

- **Language / data:** Python, pandas, parquet store (`src/store.py`) with DuckDB as the SQL query layer over it (see section 12)
- **Modeling:** scikit-learn, LightGBM
- **App:** Streamlit, reading directly from the parquet store / DuckDB (see section 12)
- **Tooling:** pytest, GitHub Actions CI, YAML config, ADRs in `/docs`

### Proposed repo layout
```
/
├── README.md
├── PLANNING.md
├── config/league.yaml
├── data/                # raw + processed (gitignored where large)
├── src/
│   ├── ingest/
│   ├── features/
│   ├── models/
│   ├── value/           # fantasy points engine
│   ├── backtest/
│   └── app/
├── notebooks/           # exploration only
├── tests/
└── docs/adr/
```

---

## 9. Roadmap

| Phase | Deliverable | Done when | Status |
|---|---|---|---|
| **0. Spike** | Confirm sources, pull ~10 seasons of data, explore quality (gaps, ID matching, missing injuries) | Data-quality report + chosen sources | **Done** — `docs/data-quality.md`, ADR 0002, ADR 0005 |
| **1. MVP** | Baseline projections + fantasy points engine + first ranked board using our scoring | Ranked list usable for our draft | **Done** — `python -m src.value.board` run end to end against the real 2026-27 season on 2026-09-23 (687-player board, real ADP joined); see section 1a |
| **2. Backtest** | Walk-forward harness, benchmarks, results report | Reproducible backtest with naive benchmark | **Done** — real 2016-17..2025-26 run reported in README ([Results](docs/project-overview.md#results)); ADP benchmark ingested and run too (2016-17..2024-25, ADR 0007) |
| **3. Feature layers** | Injury, roster context, contract, each with ablation | Lift measured per layer | **Injury, roster context, roster transactions and Summer League / preseason done** (ADR 0012: preseason carries a real signal, Summer League alone does not) (ADR 0006: real small lift; ADR 0010: no lift; ADR 0011: hurts rank order, worse than 0010 — both not recommended) — see README [Results](docs/project-overview.md#results). Contract done as far as data allows (ADR 0013: rookie-scale clock only, no meaningful change, not recommended) |
| **4. Draft app** | Live draft board with suggestions | Works against a mock draft | **Done** — `streamlit run src/app/draft_board.py`; ranked/filterable board, draft-in-progress state with JSON export/import, best-available-by-position suggestions; exercised live against real 2026-27 board data (load, mark/undo picks, filters, suggestions) — see `docs/app.md`, ADR 0009 |
| **5. In-season** | Nightly refresh, trade/waiver tools, alerts, ESPN league sync | Runs unattended for a full week | **Partial**: the pre-draft data refresh runs unattended twice a day (ADR 0014); trade analyzer, waiver finder, schedule awareness and the rest-of-season projection are built and tested as-if-mid-season (ADR 0015, `docs/inseason.md`), not yet run on a live season; the unattended in-season nightly job is built and scheduled (ADR 0018); its live-API ingest was proven on 2026-09-26 by replaying 24 real nights of 2025-26 against stats.nba.com (idempotent, equal to the canonical pull row for row) and its trigger time is derived (09:30 NZDT), but no real in-season night has run and the user's team id is unset until after the draft (`--set-team`); "runs unattended for a full week" is NOT yet demonstrated and can only be after 2026-10-20 |
| **6. Polish** | README with charts, methodology, demo, docs | Repo is presentable | **Mostly done** (README, architecture doc, ADRs, real results section); charts are report PNGs, not embedded in the repo yet |

**Timing:** the season tips off soon, so prioritize Phase 1 to get a usable ranking before our draft. Do Phase 2+ after, in season.

---

## 10. Portfolio Checklist

- [x] README leads with backtest results, including honest failures — real 2016-17..2025-26 numbers, including that MAE is *worse* than naive, not just where the model wins
- [x] One command reruns any season's backtest — `python -m src.backtest --model baseline --seasons A:B`
- [x] Ablation table and miss analysis published — miss analysis is done, injury and roster-context ablation rows are real (see Results); the contract row is real too (ADR 0013: rookie-scale clock, no meaningful change), so no ablation row is open
- [x] Architecture diagram + methodology doc — `docs/architecture.md`, `docs/modeling.md`, `docs/backtest.md`
- [x] CI running tests, including leakage tests — `.github/workflows/ci.yml`
- [x] No secrets or licensed data committed — `.gitignore` excludes `.env` and `data/`; real data lives outside the repo in `~/dev-data/nba-fantasy-2026`

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Injuries are noisy | Model availability as a distribution; show floor/ceiling |
| Player/team ID mismatches across sources | Build a canonical ID map early; test it |
| Preseason trades and role changes after data freeze | Add a preseason roster/minutes update step. **Built** (ADR 0012): `python -m src.ingest.preseason_refresh`; re-run it right before the draft |
| Rostered debutants and undrafted signees unprojected; preseason injuries unflagged | **Built** (ADR 0016): `baseline_offseason_debut` projects them (low confidence) and the board carries `risk_level` / `risk_flags` (advisory, unvalidated haircuts; projections unchanged). Still open: college / international production (no compliant source), validation of injury effects |
| Overfitting to a few seasons | Many seasons, simple models first, held-out evaluation |
| Data source breaks or changes terms | Cache raw pulls; keep ingestion modular |
| No point-in-time contract or salary source (ADR 0005 D4) | Only the draft-derived rookie-scale clock is built (ADR 0013, no lift). Veteran contract years were then tested via the Wikipedia route (ADR 0019): no reliable lift. Previously: the one credible route is the contract phrases in ADR 0011's Wikipedia wikitext, not built |

---

## 12. Open Decisions

- [x] Exact ESPN scoring settings (see section 2)
- [x] Number of teams and roster slots: **14 teams** (13 managers + 1 placeholder; the live ESPN league reported 14 on 2026-09-30, it was 13 on 2026-09-23 and provisional at 10 before; the league is looking for a 14th manager), 10 starters + 3 bench + 1 IR
- [x] Draft type: snake, date **2026-10-17 morning NZ time** (set 2026-09-26; clock time 07:00 NZDT confirmed 2026-10-02)
- [x] Acquisition limits: 7 per matchup, no season cap
- [x] Keeper or dynasty rules: none
- [x] UI choice: **Streamlit**. Plugs directly into the existing pandas/DuckDB stack with no separate API layer, gets a usable draft-day tool built fast against an unscheduled draft date. Next.js would look more polished for the portfolio but costs much more build time for what is, day to day, a single-user tool — revisit post-draft if the polish is worth it then.
- [x] Storage choice: **DuckDB as a query layer over the existing parquet files**, not a replacement for them. The parquet store (`src/store.py`) is already tested, atomic, and git-friendly (real data lives outside the repo); DuckDB reads parquet natively with no ETL, so it adds SQL for reports and the live app without migrating anything. Postgres would add server/migration overhead this single-user, largely-batch workload doesn't need.

- [x] Methodology follow-ups (ADR 0030, docs/limitations.md): zero-game seasons modelled (hurdle, ADR 0031), calibrated season-total interval (ADR 0033); a dated position source is not needed (ADR 0033, closed by evidence).
- [ ] Rookies' and debutants' chance of not appearing at all (needs drafted-but-never-played rookies, which the `players` table does not hold).
