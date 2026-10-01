# Project overview, status and full results

> Long-form status and results. For the short intro and quickstart see the [README](../README.md).


A projection engine for an **ESPN head-to-head points** fantasy basketball league, plus a
**walk-forward backtest** that scores the engine against what actually happened.

The engine ranks players using only information available before a season starts, converts projected
stat lines into fantasy points under this league's exact scoring, and turns those into a draft board
(value over replacement, tiers, floor and ceiling). The backtest replays roughly ten past seasons
(2015-16 to 2025-26) and measures how well each model's preseason ranking predicted the season that
followed, against naive and market benchmarks. The same engine is meant to power live draft, trade and
waiver tools later.

> **Honest status:** the pipeline is built end to end and runs on real data. The first real
> walk-forward result is in — the baseline model beats naive on where it ranks players, but not
> on the size of its errors, and ~70% of its miss error is availability-driven (injuries/DNPs).
> ESPN ADP is now ingested too, and it beats our baseline at the top of the draft board (top-12/50/100
> hit rate, NDCG) — a humbling, not-yet-matched benchmark. Three ablation layers have now been tried
> to close that gap: injury (a real, small lift), lagged roster context (no lift, mildly hurts), and
> a real, dated roster-transactions signal purpose-built to fix the roster layer's identified gap
> (measurably hurts rank order, and we traced exactly why — see Results). None closes the gap; all
> three are honestly reported, not recommended where they don't help. See [Results](#results) for the
> numbers and the honest read, not a summary that flatters them.

## Status

As of 2026-09-28.

| Area | State | What exists |
|---|---|---|
| Data contract (ADR 0001) | Done | Table specs, season helpers, `History` leakage guard, `Projector` protocol in `src/contracts.py` |
| Parquet store | Done | Atomic, validated table read/write in `src/store.py` |
| Synthetic league | Done | Contract-valid league with a known generating process in `src/synthetic.py` (for tests only) |
| Fantasy points engine | Done | Scalar and vectorized points, league config loader (`src/value/`) |
| Repo tooling and CI | Done | `./dev` worktree/integrate tooling with tests, GitHub Actions CI, docs |
| NBA data ingestion | Done | `python -m src.ingest.nba_stats`; 11 seasons (2015-16..2025-26) pulled and validated — see `docs/data-quality.md` |
| Projection models | Done | Baseline projector, naive-last-season benchmark, rookie prior, model registry (`src/models/`) — see `docs/modeling.md` |
| Value board | Done | `python -m src.value.board` (VORP, tiers, positional scarcity, floor/median/ceiling) |
| Walk-forward backtest | **Done — real results reported** | `python -m src.backtest`; see [Results](#results): baseline beats naive on rank (Spearman, top-12), not on error magnitude (MAE) |
| Data-source research | Done | ADRs 0002 to 0005 (ingest, model, backtest, data sources) |
| Injury, roster, transactions and contract feature layers | **Done — real ablations reported** | Injury: real, small lift (ADR 0006). Roster: no lift, mildly hurts MAE, not recommended (ADR 0010). Contract (rookie-scale clock): no meaningful change, not recommended (ADR 0013). Transactions: real dated arrival/departure signal, hurts rank order more than the roster layer, not recommended (ADR 0011). |
| Preseason roster update | **Done** | `python -m src.ingest.preseason_refresh`: the 2026 draft class is now in `players` (the board had no rookies), dated roster snapshots, Summer League + preseason box scores, ADP re-mapped (unmapped 2026-27 rows 14 to 2); see [`docs/offseason.md`](offseason.md), ADR [0012](adr/0012-offseason-layer.md) |
| Summer League / preseason layer and breakout watchlist | **Done — real ablation reported** | `baseline_offseason`, `python -m src.value.breakouts`, Breakouts tab in the app. **Preseason carries the signal; Summer League alone is close to noise** (ADR 0012, [Results](#results)) |
| Live draft app | **Done** | `streamlit run src/app/draft_board.py` — ranked board, draft-in-progress state with export/import, best-available-by-position suggestions; see [`docs/app.md`](app.md) and ADR [0009](adr/0009-streamlit-app.md) |
| Draft-day live sync | **Done** | A sidebar "Live draft sync (ESPN)" section polls the ESPN league sync for opponents' picks and marks them drafted automatically (attributed by ESPN team id), with the manual "Mark drafted" form kept as the permanent fallback; see `src/app/live_sync.py`, [`docs/app.md`](app.md), ADR [0025](adr/0025-draft-live-sync.md) |
| Best available by need | **Done, unvalidated heuristic** | An additive `need_score`/`need_adj_vorp` ranking that crosses the undrafted board with your own roster's positional gaps (`config/league.yaml` roster shape); falls back to plain VORP order with an empty roster. Not a projection change. See `src/value/need.py`, ADR [0026](adr/0026-need-adjusted-board.md) |
| Offseason gaps: debutants, origin proxies, risk flags | **Done, two parts honestly open** | `baseline_offseason_debut` projects the 5 stash debutants and about 50 undrafted signees (low confidence); origin/combine proxies (real but small rookie gain); ESPN status / preseason absence / team-change flags that never change a projection. No compliant college or international production source; injury effects unvalidated. ADR [0016](adr/0016-offseason-gaps.md) |
| Daily pre-draft refresh automation | **Done** | `python -m src.ops.daily_refresh`, scheduled with `./dev schedule install` (Windows Task Scheduler, twice daily); diffed report in `reports/daily/latest.md`, dated ESPN snapshot archive, plus the ADR 0016 profiles and status-archive steps; see [`docs/offseason.md`](offseason.md), ADR [0014](adr/0014-daily-refresh-automation.md) |
| Live transactions tracker | **Done** | `python -m src.ops.txn_watch --refresh --ack`: trades, signings, extensions, waivers and coach / front-office moves from ESPN's public feed into a dated append-only ledger, ranked against the board (descriptive only; projections unchanged). Daily-report section and an app tab; 99% of per-player latest moves agree with the NBA roster snapshot, and the feed is known to be incomplete. See [`docs/transactions.md`](transactions.md), ADR [0017](adr/0017-transactions-tracker.md) |
| In-season nightly job | **Built; run live only before the season (2026-09-25), the games ingest and team artifacts are unproven until opening night** | `python -m src.ops.nightly`, scheduled with `./dev schedule install --job nightly` (daily 09:30 local, 2026-10-20 to 2027-04-06): incremental game-log ingest, ESPN league sync, rest-of-season projection, waiver/streaming/alert/lineup artifacts and a diffed report in `reports/nightly/latest.md`; see [`docs/inseason.md`](inseason.md), ADR [0018](adr/0018-nightly-inseason-job.md) |
| Coach and staff impact | **Done, honest mostly-negative result** | Head coaches for 2015-16..2026-27; a coach's style (top-five minutes, rotation depth, three-point share; star minutes probably) follows him to a new team, pace and youth development do not; a `baseline_coach` minutes layer showed no significant change (not recommended); `python -m src.value.coaches` and an app tab list the six new 2026-27 head coaches and the ranked players under them. Assistants and front office cannot be analysed with compliant data. See [`docs/coaches.md`](coaches.md), ADR [0020](adr/0020-coach-impact.md) |
| Return-from-injury study | **Done; pre-registered relative test says "too harsh", the absolute correction is inconclusive (about zero)** | `python -m src.backtest.return_report`: does the baseline over-discount a player who missed a long block and then played healthily (Tatum's position)? Pre-registered cohort, n = 60 over ten walk-forward seasons: the model is about right in absolute terms (+1.0 GP [-4.1, +5.8], +62 total FP [-80, +196]); it looks low only relative to matched spread-absence players, whom it over-projects (+10.7 GP, +251 FP). A >= 60%-block subgroup (n = 21) hints at +8 GP but is one cell of 18. No model or board change. See ADR [0021](adr/0021-return-from-injury-study.md) |
| Return-health availability feature | **Done, registered and not recommended** | `baseline_return` / `baseline_injury_return` / `baseline_return_offseason_debut`: lead-block share, tail health and ADR 0021's return flag as extra availability columns (`src/features/return_health.py`). Pre-registered ten-season ablation vs `baseline`: MAE of total FP -0.47 [-1.31, +0.29] (worse, n.s.), MAE of games -0.037 [-0.085, +0.011], Spearman -0.0012, wins 2 and 3 of 10; it lifts the Tatum-type cohort by about 8.8 GP, over-correcting a group the baseline had about right (Tatum would go 50.1 -> 57.7 GP, +313 total FP; this number is NOT recommended for use: the feature over-corrects the study cohort, and the board keeps 50.1). Board unchanged. See ADR [0022](adr/0022-return-health-feature.md) |
| Advisory return flag | **Done; display-only** | `return_flag`, `return_block_pct`, `return_tail`, `return_gp_upside_adv`, `return_fp_upside_adv` on the board CSVs and in the app (Returned-healthy filter, Debutants & risk table), plus a sentence in `risk_flags`. ADR 0021's cohort (Tatum and eight others on the 2026-27 board), with a fixed advisory judgement of +5 GP (x `proj_fppg` in total FP) for a lead block >= 60%, else 0. **Not a projection**: every projection, VORP, rank, `risk_level` and `risk_gp` value is byte-identical with and without it (Tatum stays rank 53, 50.1 GP). The app now states that the rank is by total fantasy points (FPPG x games). See ADR [0023](adr/0023-return-flag.md) |
| Load-management study and advisory flag | **Done; honest weak result, display-only** | `python -m src.backtest.load_report`: no data says why a game was missed, so load management is proxied by the shape of last season's absences (ADR 0024, pre-registered, ten walk-forward seasons, n = 1,830). Games missed in short 1-2 game absences go with a lower next season than projected (-1.4 GP [-2.4, -0.4], -53 total FP [-88, -20] per SD; 9 of 10 seasons) but the back-to-back rest proxy and the 65-game marker show nothing, and adjusting projections with it does not lower out-of-sample error, so it is a descriptive association, not an incremental feature. The board and CSVs gain an advisory `lm_flag`, `lm_iso_n`, `lm_rest_n`, `lm_gp_risk_adv` (-2 GP for 6+ short absences), `lm_fp_risk_adv`; every projection, VORP and rank is byte-identical with and without it. See ADR [0024](adr/0024-load-management.md) |
| In-season tools (schedule, rest-of-season projection, trades, waivers) | **Built and tested on synthetic and as-if-mid-season 2025-26 data; season not started** | `python -m src.inseason.{schedule,ros,trade,waivers}` and an app page: games per team per matchup week (derived weeks until ESPN publishes them), a rest-of-season projection that beats preseason-only out of sample in 5/5 test seasons (but only marginally beats season-to-date-only on rank), slot-aware trade analyzer, waiver finder with streaming and minutes flags; see [`docs/inseason.md`](inseason.md), ADR [0015](adr/0015-inseason-tools.md). Untested against real league rosters until the draft |
| Trade analyzer confidence UI | **Done** | The app's Trade analyzer tab adds a magnitude-bucketed confidence read (`clear win`/`lean your way`/`roughly even`/`lean other way`/`clear loss`) on the existing tolerance band, per-player advisory caveats (ESPN status, return/load-management flags), and an optional partner-side score; see `docs/categories.md` section 16, ADR [0027](adr/0027-trade-analyzer-ui.md) |
| External rankings comparison | **Done** | Manually-exported Yahoo and FantasyPros draft rankings, matched onto `player_id` and compared against our own board rank (`rank_delta`) on an app page; snapshot-refreshed by hand, not a live feed. See `src/value/rankings_compare.py`, `src/app/pages/external_rankings.py`, ADR [0028](adr/0028-external-rankings-compare.md) |

## League configuration

Everything league-specific lives in [`config/league.yaml`](../config/league.yaml) so the engine is
reusable. Values were copied from the ESPN settings page on 2026-09-22 and must be re-verified before
the draft.

| Setting | Value |
|---|---|
| Teams / format | 13 (confirmed live against the real ESPN league, 2026-09-23), H2H points, weekly matchups, 20 regular-season weeks, 8 playoff teams |
| Draft | Snake, 90 s per pick, order set manually, pick trading allowed, **2026-10-17 morning NZ time** (earliest plausible start 2026-10-16 18:00 UTC; exact clock time TBD — `config/league.yaml` `draft.*`) |
| Starters (10) | PG, SG, SF, PF, C, G, F, UTIL x3 |
| Bench / IR | 3 bench + 1 IR (13 rostered) |
| Lineups | Daily, locking at each player's game time |
| Acquisitions | Waivers (1 day), no season cap, 7 per matchup |
| Trades | No limit, deadline 2027-03-13 |
| Keepers | None |

Scoring weights (no double-double, triple-double or threshold bonuses):

| PTS | REB | AST | STL | BLK | TO | FGM | FGA | FTM | FTA | 3PM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 4 | 4 | -2 | 2 | -1 | 1 | -1 | 1 |

## Quickstart

Requires Python 3.12 or 3.13 and, for the `./dev` tooling, bash (Git Bash on Windows).

```bash
# 1. Environment (the project is not pip-installable as a package; install its dependency list)
python -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
python -c "import tomllib; p = tomllib.load(open('pyproject.toml','rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > .venv/requirements.txt
pip install -r .venv/requirements.txt

# 2. Run the tests (offline; synthetic data only)
NBA_VENV="$PWD/.venv" ./dev test

# 3. Pull real data into the shared data directory (unofficial API; be polite about rate limits)
python -m src.ingest.nba_stats --seasons 2015-16:2025-26

# 3b. Before the draft: bring rosters, rookies, Summer League/preseason and ADP current (see docs/offseason.md)
python -m src.ingest.preseason_refresh
python -m src.value.breakouts --season 2026-27 --out reports/watchlist.csv

# 4. Build the draft board from a projector (add --adp to join ADP straight from src.ingest.espn_adp's output)
python -m src.value.board --season 2026-27 --model baseline --out board.csv
python -m src.value.board --season 2026-27 --model baseline --out board.csv \
  --adp ~/dev-data/nba-fantasy-2026/processed/adp.parquet

# 5. Run the walk-forward backtest
python -m src.backtest --model baseline --seasons 2016-17:2025-26 --out reports/

# 6. Launch the live draft board (try it with no real data via the sidebar's synthetic demo mode)
streamlit run src/app/draft_board.py
```

Full flags for each are in `--help` (`python -m src.ingest.nba_stats --help`, etc.) and in the ingest,
model and backtest ADRs. Data is written to `~/dev-data/nba-fantasy-2026` by default (override with
`NBA_DATA_DIR`), which is outside the repository on purpose. Tests marked `network` are excluded by
default; run them explicitly with `./dev test -m network`.

## Quick start for a shared copy

```bash
git clone <repo-url> && cd <repo>
./dev setup                                    # creates ./.venv, installs dependencies (Git Bash on Windows)
export NBA_VENV="$PWD/.venv"
./dev data unpack path/to/nba-data-pack.zip    # if you were sent one; otherwise ingest (Quickstart step 3)
./dev test
streamlit run src/app/draft_board.py
```

The data pack is produced with `./dev data pack` and sent privately (never committed): it holds the
processed tables only, without the ESPN league snapshot, backups or the manually exported (licensed)
Yahoo/FantasyPros rankings. The board's external-rankings page stays empty until you export your own.
Live league sync needs your own `.env` (copy `.env.example`); nothing in the repo contains credentials.

## Architecture

```
Ingest -> Contract tables / History -> Projectors -> Fantasy points engine -> Value board -> Backtest + Live app
```

Ingest, features, models, value engine and backtest are built independently against one shared data
contract, and models only ever see a `History` object that structurally excludes the season being
projected. The full pipeline diagram, table-level data flow, the leakage-guard design, module map and
the extension points for injury, roster and contract layers are in
[`docs/architecture.md`](architecture.md). Decisions are recorded in the
[ADR index](adr/README.md). A reference for every category, flag and column the app shows, with formulas and worked examples, is in
[`docs/categories.md`](categories.md).

## Methodology

This is the intended methodology; the pieces marked in progress above are what is being implemented.
The reasoning and open questions are in [`PLANNING.md`](../PLANNING.md).

- **Fantasy value (H2H points).** Fantasy points per game are the sum of projected stats times the
  league's scoring weights. Projected season value is that times projected games played. Value over
  replacement is measured against a replacement level derived from league size and roster slots,
  verified from data rather than assumed. There is no category balance or punting in a points league, so
  minutes, usage and the heavily weighted stats (steals, blocks, turnovers) drive value.
- **Variance.** Weekly matchups reward consistency differently from season totals, so the engine tracks
  floor, median and ceiling, and groups players into tiers by value gaps.
- **Projection model.** Minutes, usage and role, per-minute production (regressed to the mean for outlier
  seasons) and availability (a distribution, not a point estimate) are modelled separately and combined.
  Simple, explainable models come first.
- **Feature layers.** Baseline, then injury history, roster context and contract, each added behind a
  flag and kept only if the ablation shows lift. Contract was an explicit hypothesis: only its rookie-scale slice is testable, and it shows no lift (ADR 0013).
- **Backtest.** Walk-forward: for season S, train only on data through S-1, rank players, compare with
  actual season-S fantasy points under this league's scoring. Metrics: Spearman rank correlation,
  top-50 and top-100 hit rate, error in FPPG and total fantasy points, and VORP-weighted error.
  Benchmarks: naive last-season ranking, and ESPN ADP/consensus if obtainable. Reported with an
  ablation table and a miss analysis (injury, trade, role change).
- **Leakage guard.** Every feature is computed as of a date before the target season, enforced by
  `History.until` and covered by tests that run in CI. No reported result may use synthetic data.

## Results

First real walk-forward run, 2016-17 through 2025-26 (10 target seasons, real ingested data,
`--leak-check` passed). Reproduce: `python -m src.backtest --model baseline --benchmark
naive_last_season --seasons 2016-17:2025-26 --leak-check --out reports/`. Full report with
per-season breakdowns, calibration and miss analysis: `reports/real_2026-09-22/baseline_2016-17_2025-26_b30db614/report.md`
(generated reports are gitignored; the numbers below are copied from that run on 2026-09-22).

| Model | Spearman, total FP | Top-12 hit rate | Top-50 hit rate | Top-100 hit rate | MAE, total FP | Status |
|---|---:|---:|---:|---:|---:|---|
| Naive: last season's ranking | 0.763 | 0.442 | 0.634 | 0.633 | 426.6 | run |
| ESPN ADP / consensus \* | 0.743 | 0.620 | 0.740 | 0.749 | n/a (rank-only) | run — see note \* below |
| Baseline projector (before the rookie availability cap) | 0.784 | 0.542 | 0.654 | 0.667 | 422.6 | run |
| Baseline projector (current, with the cap) | 0.784 | 0.542 | 0.656 | 0.670 | 422.6 | run, 2026-09-24 |
| `baseline_hurdle` (appearance stage, ADR [0031](adr/0031-hurdle-availability.md)) | 0.810 | 0.550 | 0.654 | n/a | 367.5 | run, 2026-10-01 |
| `baseline_hurdle_adp_offseason_debut` (board stack: hurdle + ADP as input, ADR [0032](adr/0032-adp-as-input-and-blend.md)) | 0.819 | 0.533 | 0.674 | 0.687 | 316.3 | run, 2026-10-01 (vs `baseline_offseason_debut` 0.771 / 0.542 / 0.658 / 0.683 / 372) |
| + injury layer | 0.785 | 0.542 | 0.652 | 0.672 | 421.4 | run |
| + roster layer | 0.783 | 0.525 | 0.650 | 0.667 | 425.0 | run — **no lift, not recommended** |
| + transactions layer | 0.782 | 0.558 | 0.656 | 0.671 | 425.8 | run — **hurts rank order, not recommended** |
| + Summer League only | 0.786 | 0.542 | 0.656 | 0.670 | 422.1 | run — no meaningful change |
| + Summer League + preseason | 0.781 | 0.542 | 0.660 | 0.682 | 413.1 | run — **better error size, slightly noisier total-FP rank order; strong for the young/rookie question** (see below) |
| + contract layer (rookie-scale clock) | 0.784 | 0.542 | 0.656 | 0.669 | 422.6 | run — **no meaningful change, not recommended** (ADR 0013) |
| + contract-terms layer (veteran contract years, Wikipedia) | 0.783 | 0.517 | 0.652 | 0.671 | 421.4 | run — **no lift on rank order (Spearman -0.0003, top-50 -0.004); MAE -1.2 (0.3%); not recommended** (ADR 0019) |

*Which baseline each row is measured against.* The injury, roster and transactions rows were run before the rookie availability cap (top-50 0.654, top-100 0.667); the Summer League rows and the current baseline include it (0.656, 0.670). The cap moves only the top-50 and top-100 hit rates, by 0.002 to 0.003; Spearman, top-12 and MAE are identical. Compare each layer to the baseline row from its own era, not across the two.

**Honest read, not a spin.** The baseline projector beats the naive last-season ranking on where it
ranks players (Spearman total FP +0.009, 95% CI [+0.002, +0.016], won 8 of 10 seasons — paired-bootstrap
verdict `improves`) and clearly on top-of-draft precision (top-12 hit rate +0.10). It does **not**
clearly improve on the size of the error: on the **paired comparison** (players both models actually
projected, which is the fair test — see below), baseline's MAE on total fantasy points is about 27.6
points *worse* than naive's (95% CI [-36.0, -21.0], verdict `hurts`), and top-50 hit rate shows no
significant change. So: better at ordering players, not better at predicting the number.

The table above shows baseline's *unconditioned* mean MAE (422.6) as slightly *lower* than naive's
(426.6) — that is not a typo and it is not the number to trust: naive projects far fewer players than
baseline (it has no answer for anyone without a full prior season, so rookies and returners are simply
absent from its universe), so the two means are averaged over different, non-comparable player sets.
The paired-bootstrap comparison above restricts to players *both* models actually projected, which is
the fair like-for-like test, and that is where baseline loses on MAE.

**Where the error comes from.** Availability — projecting games played for a player who then gets hurt
or DNPs, or the reverse — accounts for ~70% of the total absolute error in the biggest misses (rookie
debuts are next at ~15%). This is exactly what section 4's roadmap calls out: an explicit injury feature
layer is the highest-value next step, not more tuning of the baseline's per-minute rates.

**The injury layer: a real but small lift, not a fix.** `+ injury layer` (`baseline_injury`, ADR 0006)
adds absence-streak length/frequency, recency-weighted and age-adjusted absence rate, and a "coming off
a long absence" flag to the availability model, built entirely from `game_logs`/`team_games` already
ingested (no new data source). Paired bootstrap over players vs. plain `baseline`: MAE on games played
improves 19.24 → 19.14 (95% CI [+0.075, +0.131], won 8/10 seasons, verdict `improves`), MAE on total FP
improves 422.6 → 421.4 (95% CI [+0.76, +1.74], `improves`), Spearman total FP is essentially flat
(+0.0009, 95% CI [+0.0002, +0.0018], `improves` but only 5/10 seasons won), and top-50 hit rate shows no
significant change. Every CI that moved excludes zero — this is a real, consistent, non-tuned effect —
but the *size* of the lift is small (about 0.5% relative on games-played MAE) and availability is still
~70% of the biggest misses after adding it. See ADR 0006 for the full honest write-up.

**The roster layer: no lift, and it mildly hurts.** `+ roster layer` (`baseline_roster`, ADR 0010)
adds a fitted, recency-weighted adjustment to projected minutes from three lagged team-context
signals (team pace, the player's share of team minutes, and positional crowding on his team) —
built entirely from `game_logs`/`team_games` already ingested, no new data source, same pattern as
the injury layer. Paired bootstrap over players vs. plain `baseline`: Spearman total FP is flat
(-0.0002, 95% CI [-0.0008, +0.0004], no significant change, 4/10 seasons), top-50 hit rate is flat
(-0.0040, 95% CI [-0.0140, +0.0060], no significant change), and MAE on total FP gets **worse**
(-2.4211, 95% CI [-2.8971, -1.9736], verdict `hurts`, only 3/10 seasons won). Diagnostics on a real
fit (5,386 player-seasons) rule out an implementation bug — the fitted adjustment is small (mean
+0.05 mpg, std 0.38, max magnitude 1.2 minutes) and never reaches the ±4.0-minute cap, so this is a
genuinely weak signal, not a clipped or exploding one. The likely reason, documented in ADR 0010's
own design discussion before the ablation ran: this layer can only see *lagged* team context (last
season's pace/role/crowding), never a true forward-looking "a star just arrived or left this
offseason" signal, because the data contract has no point-in-time preseason roster source — exactly
the kind of signal PLANNING.md's roster row actually wanted. **`baseline_roster` is not recommended
for the draft board or any in-season tool.** The code, tests and ADR are kept as a complete,
honestly reported negative result — see ADR 0010 for the full write-up, including what a real fix
would require.

**The transactions layer: a real, dated fix attempt that made things worse — and we know why.**
`+ transactions layer` (`baseline_transactions`, ADR 0011) was built specifically to close the gap
ADR 0010 identified: it ingests every team's actual, dated offseason moves (330 Wikipedia team-season
pages, 2016-17 through 2025-26, 5,452 transactions, 84.9% of distinct names matched to a real player)
and feeds a genuine forward-looking "did this player change teams, and how many of his new team's
rotation players just left" signal — not a lagged proxy. Paired bootstrap over players vs. plain
`baseline`: Spearman total FP gets **worse** (-0.0017, 95% CI [-0.0022, -0.0013], verdict `hurts`,
only 1/10 seasons), top-50 hit rate is flat (+0.0020, no significant change), and MAE on total FP
gets worse by more than the roster layer did (-3.2166, 95% CI [-3.7291, -2.7247], `hurts`). This is a
*worse* headline result than ADR 0010's, not a fix. Diagnosed directly against 3,933 real training
rows before writing this up (not just assumed): the underlying signal is real and correctly signed —
players who changed teams that season really do get about 0.75 fewer minutes than their historical
per-minute rate alone would predict, on average (a believable "new team, unproven role" effect) — but
a single linear coefficient applies that *average* penalty to every mover alike, which is exactly
wrong for the cases that matter most: a star signing with a rebuilding team as its new lead option
needs the opposite adjustment, and this model can't tell the two cases apart. **`baseline_transactions`
is not recommended for use**, more clearly so than the roster layer — see ADR 0011 for the full
diagnostic write-up and the scoped (not built) extension point this result points to.

**The offseason layer: preseason works, Summer League mostly does not.** `+ Summer League + preseason`
(`baseline_offseason`, ADR 0012) stacks a fitted adjustment on the baseline from how each player's Summer League and
preseason lines (production per 36 minutes against that summer's cohort, and minutes given) explained the baseline's own past
misses; it switches itself off unless leave-one-season-out cross-validation shows a gain. Whole league, paired vs baseline:
MAE on total FP improves 422.6 to 413.1 (95% CI [+7.9, +10.9], 8/10 seasons), FPPG MAE 4.67 to 4.51, VORP-weighted MAE 669.7 to
646.5, top-100 hit rate 0.670 to 0.682; Spearman on total FP is -0.0028 (CI [-0.0047, -0.0009], `hurts`, 3/10 seasons). That drop
is entirely among the 41% of projected players who barely played (near-replacement ordering noise): among players who played 20+
games the layer *improves* Spearman, 0.737 to 0.749. **The question it was built for**, whether young under-the-radar players
about to break out can be found: yes, modestly. The top ten flagged became a *useful* breakout (a breakout that also finished in
the rostered top 169) 24% of the time against a 10% base rate (+13.7 percentage points, 95% CI [+5.8, +22.1]; AUC 0.62 [0.56,
0.68]); rookies AUC 0.65. **But the signal is the preseason, not Summer League:** Summer League alone scores AUC 0.52 [0.46, 0.58]
with a top-ten lift of +4.7pp [-0.9, +10.7], indistinguishable from chance, and a richer component feature set did not change
that. So the watchlist is materially better once preseason games exist (the draft, 2026-10-17 in the morning NZ time = the evening of 2026-10-16 in central Europe, will have most of them); with
only the first half of the preseason the lift falls to +8.7pp. Also found and fixed on the way: the 2026 rookie class was missing
from `players` entirely (so from the board and from ADP mapping), and the rookie availability prior over-projected picks 1-3 by
18 games (now capped, +5). Everything is in ADR 0012 and [`docs/offseason.md`](offseason.md).

**Offseason gaps: debutants, origin, risk flags (ADR 0016).** *Debutants:* the five earlier-year debutants (Sorber, Marković, Toohey, Diop, Biberovic) and the
about 50 undrafted signees are now projected by `baseline_debut` / `baseline_offseason_debut` (low confidence; veterans and rookies are the baseline's numbers exactly under `baseline_debut`, under `baseline_offseason_debut` they differ from `baseline_offseason` for 186 of 739 shared players, max 1.07 FPPG, 1 above 1 FPPG, and from `baseline` for the same 186, max 3.24 FPPG, 23 above 1 FPPG; measured 2026-09-25).
Walk-forward over 853 camp participants with no earlier NBA game (2018-19..2025-26, players who never play scored as zero): total-FP MAE 93.5 for the model vs 225.1 for the raw rookie
prior, 108.2 for a class mean, and 70.7 for simply predicting zero; the model is unbiased (+0.7 FP) where omission is -71, but **does not beat predicting zero on error size for
undrafted players** (65% never play). Stash players (28 debuted): the model ranks them well (Spearman 0.65) and the plain slot prior is as good as anything fitted (n too small to
separate). The stash play-probability (0.90) is an assumption. *Origin proxies:* country / previous-organisation type / combine measurements lower rookie total-FP MAE
445.2 to 429.5 (95% CI 11.6 to 19.8 lower, 7 of 8 seasons) but change the whole-league board by nothing measurable (paired MAE +0.4 FP, Spearman -0.0015 `hurts`), so it is not the
default. No compliant college/international production source exists (ADR 0016 D3). *Risk flags:* ESPN status, preseason absence, new team and star moves are shown, never applied; the preseason-absence
backtest is confounded (no historical rosters: 78% of "absent" veterans never played, mostly not under contract) and the NBA injury PDFs do not exist before opening week.
Reproduce with the commands in ADR 0016.

**Calibration is good.** The floor/ceiling bands hold up: 78.4% of individual games fall inside the
projected [p10, p90] range (nominal 80%), 10.9%/10.7% fall below/above it (nominal 10% each).

\* **ESPN ADP note.** The ADP row is a *separate* run, `python -m src.ingest.espn_adp --seasons
2015-16:2026-27` then `python -m src.backtest --model baseline --benchmark naive_last_season
--seasons 2016-17:2024-25 --adp-file <adp.parquet> --leak-check --out reports/` (ADR 0007), covering
**9 seasons (2016-17 to 2024-25)**, not the 10 above: 2025-26 is excluded because ESPN's ADP for that
season is wiped (every player reads the 140.0 "no ADP" sentinel; ADR 0007 detects and drops it rather
than filling the gap into this headline row — a FantasyPros fallback exists but is not used for this
comparison to keep the "espn" numbers honestly ESPN's own), and 2026-27 has no actuals yet. So this
row's baseline/naive figures (not shown here) also differ slightly from the 10-season rows above; the
full 9-season comparison (baseline vs naive vs ADP) is in `docs/adr/0007-adp-ingest.md` and the
regenerated report under `reports/`.
`AdpBenchmark` is rank-only by design (`src/backtest/benchmarks.py`): its "prediction" is `-adp`, an
ordinal score, not a fantasy-point total, so MAE against it is not meaningful and is reported as n/a,
never as a spurious "win". **Honest read:** ESPN's crowd-sourced ADP clearly beats our baseline model
at identifying the right players *near the top of the draft* — top-12 hit rate 0.620 vs baseline's
0.556, top-100 hit rate 0.749 vs 0.677, NDCG@100 0.929 vs 0.878, all on the paired (same-player)
comparison too (baseline vs adp top50_hit lift -0.071, 95% CI excludes 0, 0/9 seasons won). Our model's
one edge is that it covers a far larger population (541-766 projected players per season vs ADP's
149-352, since ESPN only assigns ADP to players who were actually drafted somewhere), and on the full,
harder population its overall Spearman rank correlation (0.787) is a little ahead of ADP's (0.743) —
but restricted to the same players ADP also covers, ADP wins there too (paired spearman lift -0.073,
`hurts`). So: **ADP is a strong benchmark and, at the top of the board where it matters most for a
snake draft, our baseline does not yet match it.** This is exactly the outcome PLANNING.md called
realistic ("beating naive is realistic; matching ADP is a strong result") — reported as it landed, not
softened.

**Match rate.** `player_id_map` resolved 1,321 of 1,591 distinct ESPN player ids seen across
2015-16..2026-27 (83.0%: 1,306 exact name matches, 3 via the manual alias table, 12 fuzzy, 0 left
ambiguous, 270 unmatched — mostly deep bench/international players ESPN's own board barely ranks).
Restricted to players who actually carry a real ADP value (the ones that matter for this benchmark),
only 1.5% (45 of 3,050 season-player rows) failed to map. Full numbers: `docs/adr/0007-adp-ingest.md`.

**The contract layer: built to the extent the data allows, and it does nothing.** No source in the project carries dated
salary or contract data (ADR 0005 D4), so the only point-in-time contract signal is the rookie-scale clock derived from draft
year and pick: scale year 1 to 4, option year, contract year (also the extension-eligible year), post-scale year, second-round early
years. `+ contract layer` (`baseline_contract`, ADR 0013) stacks a cross-validation-gated adjustment on the baseline. Paired
bootstrap vs plain `baseline` over 10 seasons: Spearman total FP +0.0004 (95% CI [+0.0001, +0.0007], only 3/10 seasons, because the gate opened
in three seasons and stayed off in seven), top-50 hit rate +0.0000, MAE on total FP -0.04 (CI [-0.24, +0.17]): **no meaningful change**. Why:
the "contract-year" hypothesis points the wrong way where it can be tested (the baseline over-projects year-4 first-rounders by 1.1 FPPG, not
under-projects them), the positive coefficients belong to young year-2 players and rookies (an age-curve effect, not a contract effect),
and the clock covers only about a fifth of players, none of them the veterans whose contract years the hypothesis was really about (untestable
without salary data). It does not change the 2026-27 board (top-12 identical, top-50 differs by one player). **Not recommended.** **Results pending:** none; every ablation row
is measured now. The four-row chain command (`--ablate baseline,baseline_injury,baseline_roster,baseline_contract`) runs, but
`baseline_contract` is stacked on plain `baseline`, so its lift over `baseline_roster` there is not meaningful; use the pair
`--ablate baseline,baseline_contract`.

**The veteran contract-terms layer (ADR 0019): built from Wikipedia, tested, no reliable lift.** The team-season wikitext already cached for ADR 0011 carries dated contract
phrases; `python -m src.ingest.wiki_contracts` parses them into `player_contracts` (2,653 contract events over 360 pages, 953 with a stated length, 91.5% of names matched;
Wikipedia coverage is partial and biased toward notable signings: a known deal length exists for 38 to 40% of scored veterans in 2016-19 and 13 to 16% since, any event for 44 to 66%). Features
as of opening night (final year of a known deal, years left, new deal, extension, two-way, minimum; unknown stays unknown) feed `baseline_contract_terms`. Paired vs `baseline` over ten seasons: Spearman -0.0003
[-0.0007, +0.0001], top-50 -0.004 [-0.008, +0.008], MAE of total FP -1.18 [-1.53, -0.84] (a 0.3% error-size gain). **No lift on ranking; not recommended.** The contract-year premium is not there: known final-year veterans
beat their projection by -0.4 FPPG [-0.9, +0.2] (-0.9 [-1.7, -0.2] against players with 2+ years left); the one consistent effect is that fresh signees underperform it by 0.8 FPPG, a mover effect like ADR 0011's. The board is unchanged;
a `contract_flag` column is shown on the board, watchlist and app as **unvalidated, display only** (blank means not known to be, never known not to be). Contract facts derive from Wikipedia (CC BY-SA 4.0).

## Data sources and legal notes

- **NBA stats** come from [`nba_api`](https://github.com/swar/nba_api), an unofficial client for
  stats.nba.com. It is not endorsed by the NBA and the endpoints can change or rate-limit without notice.
  Ingestion is cached and throttled, and only one worker pulls at a time.
- **ESPN league data** (Phase 5) uses the community `espn-api` package. Private leagues need the
  `espn_s2` and `SWID` cookies, which live in a local, gitignored `.env`
  (template: [`.env.example`](../.env.example)) and are never committed.
- **No licensed or raw third-party data is committed to this repository.** Raw pulls and processed
  tables live in the shared data directory outside the repo, so anyone reproducing the results fetches
  their own copy under the sources' terms. Source terms and choices are recorded in the data-sources ADR
  (in progress; see the [ADR index](adr/README.md)).
- This project is for personal fantasy-league analysis and portfolio purposes. It is not affiliated
  with the NBA, ESPN, or any data provider, and nothing here is betting or financial advice.

## Parallel worktree workflow

Work is done by several people or agents at once, so the repo enforces isolation:

```bash
./dev worktree create my-task     # branch task/my-task + worktree outside the repo
cd ~/dev-worktrees/nba-fantasy-2026/my-task
# ...edit, test (./dev test), lint (./dev lint), commit explicit paths...
./dev integrate                   # lock, rebase, verify, fast-forward, verify, unlock
cd - && ./dev worktree remove my-task
```

Never edit tracked files in the primary checkout, never use `git add -A` in a shared checkout, and
never discard work you did not create. Integration is serialized behind a lock and stops on a semantic
conflict instead of guessing. Full rules: [`CLAUDE.md`](../CLAUDE.md); a shorter guide:
[`CONTRIBUTING.md`](../CONTRIBUTING.md); who owns which paths: the "Track ownership" table in
[ADR 0001](adr/0001-data-contract.md). Continuous integration
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the full suite, leakage tests included, on
Linux and Windows, plus a lint job.

## Roadmap

From [`PLANNING.md`](../PLANNING.md). The season tips off soon, so Phase 1 comes first.

| Phase | Deliverable | Done when |
|---|---|---|
| 0. Spike | Confirm sources, pull about 10 seasons of data, explore quality (gaps, ID matching, missing injuries) | Data-quality report and chosen sources |
| 1. MVP | Baseline projections, fantasy points engine, first ranked board using this league's scoring | Ranked list usable for the draft |
| 2. Backtest | Walk-forward harness, benchmarks, results report | Reproducible backtest with a naive benchmark |
| 3. Feature layers | Injury, roster context, contract, each with an ablation | Lift measured per layer (all measured; contract: no change) |
| 4. Draft app | Live draft board with suggestions | Works against a mock draft |
| 5. In-season | Nightly refresh, trade and waiver tools, alerts, ESPN league sync | Runs unattended for a full week |
| 6. Polish | README with charts, methodology, demo, docs | Repository is presentable |

Portfolio checklist (tracked in `PLANNING.md`): results led by honest failures, one command reruns any
season's backtest, ablation table and miss analysis published, architecture diagram and methodology
doc, CI running tests including leakage tests, no secrets or licensed data committed.

## License

No license has been chosen yet, so all rights are reserved by default. Choose one before inviting
outside contributions.
