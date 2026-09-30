# ADR 0019: Veteran contract-terms layer (Wikipedia contract events)

**Status:** accepted, 2026-09-25. Result: **built, ablated on the real ten-season walk-forward, no reliable lift on rank order, not recommended; shown only as an unvalidated display flag.**
**Code:** `src/ingest/wiki_contracts.py`, `src/features/contract_terms.py`, `src/models/contract_terms_baseline.py`, `src/backtest/contract_terms_eval.py`,
`src/value/contract_flags.py`; small additive changes to `src/features/contract.py` (`names`, `min_gain`, `adjustment_from_design`), `src/models/registry.py`,
`src/backtest/runner.py` (loads the table into `History.extras`), `src/value/board.py`, `src/value/breakouts.py`, `src/app/{loader,state,draft_board}.py` (display column only).
**Tests:** `tests/ingest/test_wiki_contracts.py`, `tests/features/test_contract_terms.py`, `tests/models/test_model_contract_terms.py`,
`tests/models/test_model_contract_terms_real_data.py` (marked `real_data`, skips cleanly), `tests/backtest/test_contract_terms_eval.py`.
**Evidence:** `reports/contract_terms_eval/` and `reports/contract_terms_triplet/` (gitignored; commands below), README's [Results](../project-overview.md#results).

## Context

ADR 0013 tested only the rookie-scale slice (about a fifth of players) and recorded the one credible route to the veterans: the dated
"N-year contract" phrases in the Wikipedia team-season wikitext already cached for ADR 0011, which the transactions parser discards. This ADR
builds that parser, a leakage-safe feature set, a gated projector, and evaluates it honestly. No new source was added and stats.nba.com was not
touched. The only network use was fetching the 30 `2026-27` team pages (they exist and carry the 2026 offseason) through the existing polite cached client.

## Decisions

### D1. A new standalone table, `player_contracts` (ADR 0011 D1 pattern)

`python -m src.ingest.wiki_contracts --seasons 2015-16:2026-27` parses the `Re-signed` / `Additions` / `Subtractions` / `Contract extensions`
(and `Signings`, `Waivings`) tables of every page into one row per dated contract event, with its own schema check (`validate_player_contracts`).
`team_transactions` and its numbers are untouched. Columns: `season` (leak-guard tag), `page_season`, `start_year`, `team_id`, `player_id`, `event`
(sign, resign, extension, claimed, waived, bought_out, expired, retired, option_*, traded, other), `contract_type` (standard, two_way, ten_day, exhibit10,
minimum, rookie), `years`, `days`, `amount_usd`, `signed_date`, `date_source`, flags (`is_two_way`, `is_ten_day`, `is_exhibit10`, `is_minimum`,
`is_extension`, `is_rookie`, `non_guaranteed`, `has_option`, `multiyear_unspecified`), `section`, `terms_source` (table cell, or only the cited article's title).
Only derived facts are stored; no Wikipedia text (CC BY-SA 4.0, same posture as ADR 0011: read-only, one request per second, descriptive User-Agent,
nothing from the source committed; the small fixtures in `tests/ingest/test_wiki_contracts.py` are hand-written). The shared `player_id_map` is not modified.

**Leak-guard tag.** `season` is the season the event *follows*: an event dated 1 Oct of year Y or later follows season Y, one dated Jan to Sep follows Y-1.
`History.until(T)` (drops `season >= T`) therefore keeps exactly the events dated before 1 Oct of T's start year (ADR 0011's cutoff), structurally, and the
runner passes the table as `History.extras` so `--leak-check` scrambles it too. `start_year` is the first season a deal covers (league year turns 1 Jul).
The schema check refuses a table whose tag disagrees with its date. An undated row is kept but tagged with its page season (invisible to that season's target).

### D2. Parser: a rowspan-aware grid read by content, not by header

The five real page shapes differ (2015-16 `||` inline rows; the 2016+ "Signed" column holding contract text; 2019+ Date / Player / Terms tables sharing a date or a
"Two-way contract" cell over several rows with `rowspan`; `{{sortname|First|Last}}` player cells on the newest pages; plain-text names; commented-out tables).
Each table becomes a full grid (rowspan carried down, inline cells split), then the row is read by content: player = first non-team, non-contract link
(or sortname, or plain name), date = first date-looking cell text (a `{{dts|...}}` template counts; else the row's first citation `date=`, else `access-date`; see "Date provenance"),
terms = regexes over the row. A departure row that states a contract ("4-year contract ... New team: Dallas") is a signing by the *new* team and is deduplicated
against that team's own arrival row (within seven days). When a table states nothing, the cited article's title is read as a labelled fallback
(`terms_source = title`). Player matching reuses `id_map.match_players` unchanged.

### D3. Features (as of the start of the target season; unknown is unknown)

`terms_features(contracts, season, player_ids, clock)` walks each player's events chronologically and returns a status: `known` (active deal with stated length),
`no_length` (signing known, length not stated), `lapsed` (latest deal ended: expired, or waived/bought out/retired after it), `unknown`. From `known`:
`years_remaining`, `contract_year` (final season), plus `new_deal` (signed 1 Jul to 30 Sep), `extension`, `two_way`, `minimum`. Length arithmetic is a set of labelled
assumptions (see the module docstring), e.g. an extension adds its years after the running deal's end, or after the signing season when no earlier deal is known.
Missing is never "not in a contract year": every indicator is 0 for unknown players (the fitted contrast is "state vs unknown") and the display columns say NaN /
`unknown`. The ADR 0013 rookie-scale clock is added (`rookie_cy`, `rookie_opt`) only for first-rounders whose status is not `known`. Rows tagged at or after the
target are ignored even if the caller passes the unsliced table.

### D4. Projector `baseline_contract_terms` (stacked, gated)

Same mechanics as ADR 0013 (ridge WLS of the base model's walk-forward FPPG residuals on the features, leave-one-season-out gate at 0.05%, contrast against unknown,
multiplier clipped to [0.8, 1.25], fallback to the exact baseline when there is no table, too little history or no gain). Registered without changing any existing
model's numbers (`baseline_contract`, re-run as a check, reproduces ADR 0013 to the digit).

### D5. Advisory flags on the board, watchlist and app: unvalidated

`src/value/contract_flags.py` adds `contract_flag` (contract year, new deal, extension, nominal rookie final / option year), `contract_years_left`, `contract_status`,
`contract_basis` to the board CSV, the app's board tables and its "Debutants & risk" tab, and the watchlist. **Display only: no projection or rank changes**, labelled
unvalidated in the app because the evaluation below did not support using it. Every board and watchlist CSV also carries a constant column `contract_validation` = `unvalidated_display_only`
(added at the end, no existing column renamed or moved), so a CSV consumer, who never sees the console note, can see the flag is not a validated signal.

## Coverage (exact, from `wiki_contracts_report.json` and `coverage_by_season`)

360 pages (`2015-16` to `2026-27`, 0 skipped). 4,325 table rows parsed into events; 2,653 are contract-creating events (sign / re-sign / extension), of which **953 (36%) state a length**
and 877 an amount; 3,633 rows carry a date (84% of the 4,325 parsed rows; the report's `rows_dated`). "N-year contract/deal" phrases in the page wikitext: 1,153 (958 inside the parsed tables), matching ADR 0013's "about 620" only in
order of magnitude (that count was the narrower "N-year contract" form on the pre-2026 pages). Player match: 2,775 of 3,032 distinct (name, season) names matched (91.5%: 2,733 exact,
4 alias, 38 fuzzy, 0 ambiguous; the rest are mostly G League, college and overseas names); 4,005 rows matched, 3,940 written after deduplication. (An earlier draft of this ADR,
the README and `docs/data-quality.md` said "3,054 names, 90.8%"; that was wrong, the report file says 91.5%. The pre-fix parse had 2,774/3,031; the date fixes below moved one row's tag.)

Length is stated far more often on the early pages (`2015-16` 178 of 215 events, `2016-17` 196 of 253) than later (`2019-20` 70 of 260, `2022-23` 32 of 201, `2025-26` 27 of 144): newer pages list
the date and player and leave the terms to the cited article. Coverage of the players the backtest scores (veterans with 10+ games and an earlier season), state at the start of the season:

| season | veterans | known deal length | length not stated | lapsed | unknown | any event | known final year | new deal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016-17 | 365 | 143 (39%) | 8 | 10 | 204 | 161 (44%) | 38 | 99 |
| 2018-19 | 391 | 155 (40%) | 91 | 13 | 132 | 259 (66%) | 69 | 107 |
| 2020-21 | 410 | 64 (16%) | 120 | 59 | 167 | 243 (59%) | 41 | 2 |
| 2022-23 | 407 | 58 (14%) | 157 | 33 | 159 | 248 (61%) | 23 | 84 |
| 2025-26 | 432 | 55 (13%) | 147 | 42 | 188 | 244 (56%) | 22 | 67 |

So a *known* current contract exists for 38 to 40% of veterans in the early seasons and 13 to 16% since 2019-20; any event for 44 to 66%. Coverage is partial, biased toward notable signings
(two-way, minimum and 10-day deals are listed but rarely with terms; deals signed before July 2015 are invisible), and not stationary, which limits what any pooled model can learn.
`2020-21` has only 2 fresh deals because the COVID offseason ran November to December 2020 and the 1 Oct tag (conservatively) hides those signings from that season's target.
For the 2026-27 board (739 players): 35 have a known deal length, 414 any Wikipedia contract event dated before opening night (83 fresh deals, 14 known contract years, 1 extension),
plus 60 nominal rookie-scale flags.

## Parse quality (spot check)

30 random parsed rows (seed 4242, final parser) compared with the source text: 1 clear error (a sub-header row "In-Season Additions" read as a player; it cannot match a `player_id` and is dropped,
and the parser now rejects such words), 2 rows whose date comes from a citation `access-date` (waived events; date provenance weaker, see "Date provenance"). Earlier random samples
found and fixed: `10 days` not recognised, "Two 10-day contracts / 2-year contract" classed as ten-day, "Second 10-day contract expired" classed as a signing, `{{sortname}}` player cells,
commented-out tables, plain-text player names before a linked foreign club, a link to "Name (basketball)" mistaken for a non-player page. Known residual: a cell such as
"10-day contract worth $50,752 / 3-year contract worth 3.4 million" takes the first dollar figure. 

### Date provenance (verifier fixes, 2026-09-25) and what re-running the evaluation changed

Before/after of the whole ten-season evaluation with the `dts`, citation-window and yearless-roll fixes (one date-source change moved 20 undated rows to dated, a handful of tags moved a season; 3,941 -> 3,940 rows):
gated `baseline_contract_terms` vs `baseline` Spearman -0.0003 [-0.0007, +0.0001] 1/10 (unchanged), top-50 -0.0040 [-0.0080, +0.0080] (unchanged), MAE total FP +1.20 [+0.86, +1.55] -> +1.18 [+0.84, +1.53],
MAE FPPG +0.014 [+0.005, +0.024] (unchanged); gate forced open Spearman -0.0003 -> -0.0004, MAE +2.19 -> +2.10; levels 0.783 / 0.652 / 421.4 unchanged; leak check passed; the gate opens in the same seasons
(2022-23 to 2025-26). The change is tiny and the verdict is the same (the tables above are the after numbers).


* **`{{dts|...}}` cells** (`{{dts|July 8, 2016}}`, `{{Dts|2026|June|23}}`, ISO and numeric-month forms) are now read as the row's own date; `wiki_transactions._strip_markup` (ADR 0011, untouched)
  deletes all templates, so such rows used to be undated (e.g. a 2016-17 Wizards row). 20 rows gained a cell date.
* **Citation `date=` / `access-date` fallbacks** (used only when the row has no date of its own; 865 + 35 rows after the strict window below) come from the article that reports the event, so they are the event's date or *later*
  (a July 2019 signing whose cited article is dated 2019-10-24 was tagged from October). A later tag only *hides* the event for longer; it never leaks it. To stop a background citation that predates
  the event from ever supplying an *earlier* date, a citation or access date must fall inside the page's own window, on or after 1 June of the page's start year (`_plausible(..., strict=True)`; the first version of this fix accepted the looser window that starts a year earlier, so an August-of-the-previous-year citation was still read as the event date, which is the leaking direction; 8 real rows had such dates and are now undated, tagged with their page season). A citation date is therefore an upper
  bound on the event date, not the date; `signed_date` on those rows should not be read as exact (`date_source` says which).
* **Yearless dates** ("July 9") are read in the page's start year for Jun-Dec and the next year for Jan-May. The earlier ADR text said a wrong reading "can only look older, never hide one"; that is the
  *leaking* direction (an older date makes an event visible to an earlier target), not a safe one. Pages also list the summer *after* their season, so a yearless June/July row there was read a year too early.
  Audit: 863 parsed rows carry a yearless date. The first version of `roll_yearless_dates` moved 4 of them (Raptors 2024-25: Gueye, Trent Jr., Freeman-Liberty, Vezenkov, a June or July date each
  moved to 2025) and **all 4 were wrong**: that table opens with "April 17" (Jontay Porter's April 2024 ban, listed on the 2024-25 page and read as 2025-04-17), which made the later June/July 2024
  rows look "a year behind". Each of those rows' own citation carries a 2024 date. Rule as it now stands (`roll_yearless_dates`): within one table, a yearless date more than 120 days earlier than the
  latest earlier cell date of the table moves to the next year (the table has already reached January-April, so a following "June 30" would be the summer after), **unless the row's own citation is dated
  more than 30 days before the rolled date** (a citation is dated on or after the event, so it contradicts the roll); and a yearless row whose page-year reading is later than its own citation by more than
  30 days cannot anchor the running maximum for later rows. Validation on the real pages: **0 of the 863 rows now move**; the rule fires on none of the real data, i.e. it is a guard that only acts
  when a table shows a year-wrap *and* the row's citation agrees (tested on synthetic tables, `tests/ingest/test_wiki_contracts.py`, including a Raptors-shaped table). It is not evidence that any
  real row needed rolling. 3 real rows have a page-year reading later than their own citation (Porter 2025-04-17 vs a 2024-04-17 citation; two 2019-20 rows dated July 2019 with July 2018 citations):
  they keep the later reading, which only hides the event for longer (the safe direction). Every path of the rule moves a date later, never earlier. **Residual, stated plainly:** a yearless
  June/July row with no later-dated row before it in its table keeps the page-year reading, and if it really were the following summer it would carry a tag one season too early (visible to an earlier
  target). Of the 30 rows dated 23-30 June, none is shown by the text to belong to the following year, and for most the event itself (a re-sign on 30 June, the draft-night signing) is a known
  offseason-start event of that page's year; the ten-season evaluation is insensitive to this (see the before/after).

**Verifier fixes 3 (2026-09-25), before/after** (offline re-parse from the cached pages, then the paired backtest `--ablate baseline,baseline_contract_terms`, ten seasons, `--leak-check` passed):
rows_dated 3,642 -> 3,633 (9 parsed rows, 8 of the written rows, whose only date was a citation or access date from before 1 June of the page year became undated; 3,940 written rows unchanged); the 3 Raptors rows that
matched a player return to their 2024 dates. Backtest: gated `baseline_contract_terms` vs `baseline` Spearman -0.0003 [-0.0007, +0.0001] 1/10, top-50 -0.0040 [-0.0080, +0.0080], MAE total FP
+1.18 [+0.85, +1.53]; levels 0.783 / 0.652 / 421.4: all identical to the previous run at the printed precision. The verdict (no reliable lift, display flag only) is unchanged.

## Result (real data, 2016-17 to 2025-26, `--leak-check` passed)

```
python -m src.ingest.wiki_contracts --seasons 2015-16:2026-27
python -m src.backtest --ablate baseline,baseline_contract,baseline_contract_terms --seasons 2016-17:2025-26 --leak-check --n-boot 1000 --out reports/ --run-id contract_terms_triplet
python -m src.backtest.contract_terms_eval --leak-check --n-boot 1000 --out reports/ --run-id contract_terms_eval
```

Paired vs plain `baseline` (bootstrap over players, stratified by season, 1,000 replicates; positive = better):

| Variant | Spearman, total FP | top-50 hit | MAE, total FP | MAE, FPPG |
|---|---|---|---|---|
| `baseline_contract` (ADR 0013, reproduced) | +0.0004 [+0.0001, +0.0007], 3/10 | +0.0000 [-0.0040, +0.0040] | -0.04 [-0.24, +0.17] | -0.003 [-0.009, +0.003] |
| `baseline_contract_terms` (gated) | -0.0003 [-0.0007, +0.0001], 1/10 | -0.0040 [-0.0080, +0.0080] | **+1.18 [+0.84, +1.53]**, 4/10 | **+0.014 [+0.005, +0.024]** |
| same, gate forced open (diagnostic) | -0.0004 [-0.0010, +0.0001], 3/10 | -0.0040 [-0.0100, +0.0080] | **+2.10 [+1.66, +2.53]**, 8/10 | **+0.025 [+0.014, +0.038]** |

Levels: baseline 0.784 / 0.656 / 422.6; gated terms 0.783 / 0.652 / 421.4. The gate opened in 2022-23 to 2025-26 (cross-validated gain +0.13%, +0.40%, +0.55%, +0.45%) and stayed off before
(gain -0.28% to -0.10%, 2016-17 and 2017-18 too little history). **Verdict: no lift on rank order** (the draft's currency): Spearman and top-50 are flat to slightly negative, and among players with
any Wikipedia event Spearman *hurts* (-0.0016 [-0.0024, -0.0008], 0/10 seasons). The layer does reduce the error *size*, by 0.3% of total-FP MAE (0.5% forced), a real but small calibration gain,
concentrated in players with a fresh deal (MAE FPPG +0.060 [+0.016, +0.104] for "signed this offseason", forced +0.135 [+0.085, +0.186]) and absent where nothing is known (no lift, and forced it slightly hurts).

Restricted to players with known contract status (rank metrics within the subset; all "no significant change" unless noted): known deal Spearman -0.0008 [-0.0046, +0.0027]; known final year
-0.0009 [-0.0088, +0.0061] (MAE FPPG +0.046 [-0.035, +0.121]); known 2+ years left +0.0014 [-0.0033, +0.0069].

**Coverage-conditional effect size** (the baseline's own residual, actual minus projected FPPG, 20+ games, season-demeaned, player-clustered CI; pooled 2016-17 to 2025-26):

| Group | n | mean residual [95% CI] |
|---|---:|---|
| known final year | 357 | -0.39 [-0.94, +0.18] |
| known, 2+ years left | 560 | +0.54 [-0.01, +1.12] |
| length not stated | 1,008 | -0.24 [-0.60, +0.13] |
| lapsed | 296 | -0.32 [-1.10, +0.50] |
| nominal rookie clock only | 442 | -0.02 [-0.57, +0.51] |
| nothing known | 1,476 | +0.13 [-0.16, +0.45] |
| contrast: final year minus 2+ years left | | **-0.94 [-1.69, -0.22]** |
| contrast: final year minus nothing known | | -0.49 [-1.13, +0.14] |
| contrast: signed this offseason minus not | 783 | **-0.78 [-1.17, -0.37]** |

## Diagnosis

1. **The contract-year hypothesis is not supported, in either direction that would help.** Veterans in a known final year do *not* beat their projection: the point estimate is -0.4 FPPG (CI includes 0),
   and -0.9 [-1.7, -0.2] against players with 2+ years left. By era it is -0.88 [-1.61, -0.13] in 2016-19 (the years with the best coverage) and +0.18 [-0.64, +1.03] in 2020-25, an unstable sign. This agrees with
   ADR 0013's rookie slice (year-4 first-rounders under-perform their projection). The fitted `cy` coefficient in the 2026-27 fit is -0.08 FPPG.
2. **What the data does carry is a "new deal" effect, not a contract-year effect.** Players who just signed underperform the baseline by about 0.8 FPPG [-1.2, -0.4], consistently in both eras
   (-0.80 and -0.78). It is the same population as ADR 0011's `changed_team` effect (a new team, a new role) and is a bias in the projection of movers and re-signers, worth about 0.3% of MAE. It does not
   re-order players enough to help Spearman or the top 50.
3. **Coverage is the ceiling, and it is not stationary.** Only 13 to 16% of veterans have a known deal length after 2019-20, against 38 to 40% before, because newer pages stop stating terms. The gate opens only late
   (2022-23 on) and on the strength of the new-deal, extension and minimum columns, not of the final-year flag. The hypothesis is *measured on the covered, notable subset*, not settled for all veterans.
4. **Parse error is not the explanation**: the sampled parse error rate is about 3%, terms come from the table cell for 97% of stated lengths, and the leak check passes. A parse noise of that size could not turn a real,
   large contract-year premium into the estimates above.

## Recommendation and draft board

**Do not use `baseline_contract_terms` for the board.** It stays registered and tested as a completed null-on-ranking result (it would move the 2026-27 top-50 by one player and the top-100 by two; the largest single move is a projection change of 2.9 FPPG).
The board keeps `baseline` (plus the ADR 0012 offseason layer where wanted). **The advisory flags are shown, labelled unvalidated**: the 2026-27 board (739 players) gains `contract_flag` for 158 players (83 new deal, 30 nominal rookie final year, 30 nominal option year, 14 contract year,
1 extension) with 35 known deal lengths (the strict citation window left the Joe Ingles 2-year signing, whose only date was a 2026-05-19 citation, undated; 2026-27 board counts as regenerated 2026-09-26); **no rank or projection changed**. The app and CLI say so next to the columns.

## Consequences

* ADR 0013's untested-veteran hypothesis is now **tested on the covered subset** and not supported. It is not refuted for all veterans (coverage), but the prior for a large payoff is now low.
* `player_contracts.parquet` and `wiki_contracts_report.json` live in the shared data directory (outside the repository and `History`); regenerate with the command above (offline, from the cached pages).
  No scheduled job regenerates it, deliberately: an offline re-parse of the same cached pages with the same parser code gives the same table, so it changes only when the parser changes or new pages are fetched. Refresh it by hand (`python -m src.ingest.wiki_contracts --seasons 2015-16:2026-27 --offline`, back the old file up first) after any change to `wiki_contracts.py`; fetching new pages (drop `--offline`, or `--refresh` to re-fetch) is live traffic to Wikipedia at one request per second and stays a manual, deliberate step. The table only feeds the display-only flags, so a stale copy cannot change a projection or a rank. The shared copy was regenerated from cache on 2026-09-25 after these fixes (old file kept under `backups_pre_verifier_fixes_3/`).
* Finding for ADR 0011 (not fixed here, to keep its numbers intact): its transactions parser takes the first wikilink of a row and so misses players written as `{{sortname|First|Last}}` (38 of about 3,700
  Additions/Subtractions rows, in 2015-16, 2019-20 and 2024-25) and reads a former-team link as the player. A follow-up could fix it and re-run ADR 0011.
* No shared file changed (`pyproject.toml`, `config/league.yaml`, `src/contracts.py`).
* Not built, deliberately: a "new deal" mover model conditioned on team context (the same after-the-fact-tuning caution as ADR 0011); salary-level features (amounts are stated for only 877 events).
* Attribution: contract facts are derived from Wikipedia (CC BY-SA 4.0); no article text is stored or committed, consistent with ADR 0011. Cite "Wikipedia team-season pages, CC BY-SA 4.0" when publishing derived output.
