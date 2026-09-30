# ADR 0005: External data sources, terms compliance and risks

**Status:** accepted for design, 2026-09-22; items marked *needs user sign-off* are provisional until answered.
**Evidence:** `docs/research/data-sources.md`, `docs/research/espn-api-findings.md`, `docs/research/snippets/`.
**Scope:** everything except NBA game logs / box scores (`nba_api`, owned by the ingest track; see risk R1).

## Context

The projection engine needs ADP for the backtest benchmark and the draft board, injury history and status,
schedule and league data for the live tool, and possibly contracts. Every candidate was tested or its terms
read on 2026-09-22. No source is both openly licensed and complete; the choices below trade coverage against
legal exposure explicitly.

## Decisions

### D1. ESPN is used, anonymously, at low volume, local only *(needs user sign-off, R2)*

* **Player universe** (`.../games/fba/seasons/{end_year}/players?view=kona_player_info`, top-level
  `X-Fantasy-Filter`): ADP, ownership, auction value, ESPN projections, injury flag, eligible slots. Verified.
  ESPN season id is the **end year** (2027 = 2026-27).
* **Schedule** (`.../seasons/2027?view=proTeamSchedules_wl`): verified, 80 of 82 games per team until NBA Cup group play ends.
* **League sync**: own thin `requests` client (5 GETs). `espn-api` 0.46.0 stays a reference only: it is eager
  (several requests per `League()`), and pins `urllib3<=2.2.3`. **Not verified on our league** (needs the league id).
* Nightly **snapshot** of the player universe from now on; that archive is our only real point-in-time ADP/injury source.
* Compute fantasy points from raw stat ids with our own scoring. ESPN's `appliedTotal` matches our scoring only for the
  request form without `filterStatsForTopScoringPeriodIds`.
* Never commit raw ESPN payloads, ADP tables, or league data. No cookies unless anonymous access to our league fails; then
  only in `.env`.

### D2. Injuries: official NBA report primary, box-score-derived availability as the backbone *(no sign-off needed)*

* **Primary:** NBA official injury report PDFs (`ak-static.cms.nba.com/referee/injury/...`), existing from 2018-12-19,
  parsed with `pdftotext -table` (prototype verified on 2019, 2021, 2023, 2025 files). Layout changed three times. Only the
  last report per game day is needed; slot names must be probed. Injury-layer ablation runs on 2019-20 onward.
* **Fallback/backbone:** availability spells from `team_games` minus `game_logs`, for all seasons from 2015-16. No injury type.
* **Rejected:** Pro Sports Transactions (Cloudflare bot challenge, terms unreadable), Kaggle copies of it (unknown licence,
  derived from a restricted source), Basketball-Reference (terms/robots), balldontlie (paid, current-only).

### D3. ADP benchmark: ESPN API + FantasyPros, local only, labelled honestly

* ESPN: real ADP for 2015-16 to 2024-25 and 2026-27; **2025-26 is wiped** (all 140.0). FantasyPros `?year=` pages: 2015-16 to
  2025-26, second consensus and gap filler. Wayback captures (dated) are used to spot-check snapshot timing.
* Snapshot time of past-season ADP is undocumented, so the README calls it a season-level aggregate and does not claim a
  strict preseason benchmark. The headline benchmarks are the ADP-free ones (previous-season rank, minutes-based).
* Never use `percentOwned/percentStarted` of a past season (end-of-season state, leaks).
* Cross-source check already done: FantasyPros' ESPN column vs ESPN API ADP, Spearman 1.00 (2015-16, 2016-17) to 0.81 (2020-21).

### D4. Contracts stay an unproven hypothesis layer *(no sign-off needed)*

Spotrac (ToS forbids robots/scrapers), RealGM (human click-by-click only), HoopsHype (robots block Claude agents, terms not
located) and Basketball-Reference (terms, robots) are rejected as automated sources. No source provides point-in-time contract
status, so lift cannot be measured historically. Only the rookie-scale final-year proxy (from draft year) is built if wanted. **Built and measured in ADR 0013: no meaningful change.**

### D5. Transactions and preseason info

**Built (ADR 0011):** Wikipedia team-season "Transactions" sections (CC BY-SA 4.0), read via the
MediaWiki API's raw wikitext, rate-limited and cached like every other source here. Not used the way
originally sketched here (a "later cross-check" against ESPN snapshots) -- it turned out to be a
sufficient primary source on its own for a real, dated arrival/departure signal (`src/ingest/
wiki_transactions.py`, `team_transactions.parquet`), covering the full 2015-16..2025-26 backtest
window directly, which a nightly-snapshot-only approach could never do retroactively. See ADR 0011
for the full design and the honest (negative) modeling result. Our own daily ESPN snapshots and the
opening-night-roster-from-box-scores idea remain unbuilt; summer-league/preseason minutes and depth
charts: still no legal machine-readable source found, not pursued.

**Also built (ADR 0019):** the dated contract phrases in the same wikitext (`src/ingest/wiki_contracts.py`, `player_contracts.parquet`); no new source, same CC BY-SA 4.0 posture. Result: no reliable lift.

**Update 2026-09-24 (ADR 0012):** Summer League and preseason *box scores* turned out to be available from the same stats.nba.com
source as the game logs and are now ingested (`src/ingest/nba_offseason.py`, same R1 stance). Depth charts and minutes projections
remain unsourced.

### D6. Repository hygiene

Nothing from an external source is committed: no raw pulls, no parsed player-level tables from ESPN, FantasyPros, NBA injury
PDFs or NBA stats, no league data, no credentials. Commit code, synthetic fixtures, and aggregate results. The `.gitignore`
additions, CI guard and README wording are in `data-sources.md` section 6 and are applied by the repo-hygiene track.

## Terms compliance notes (quoted or paraphrased, all read 2026-09-22)

| Source | What the terms/robots say | Our stance |
|---|---|---|
| ESPN / Disney ([disneytermsofuse.com](https://disneytermsofuse.com/english/), updated May 24, 2024, "Usage Rules") | Forbids accessing or extracting the products "using a robot, spider, script, or other automated means", including "data mining or web scraping"; license is "personal, noncommercial use only". No robots file on the API host. | Accepted risk R2: personal, non-commercial, few requests, identified UA, no redistribution. |
| NBA.com ([nba.com/termsofuse](https://www.nba.com/termsofuse), updated July 13, 2026, "NBA Statistics") | Attribution required; only "legitimate news reporting or private, non-commercial purposes"; "may not be used in connection with any fantasy game"; no comprehensive-statistics database. | Risk R1, see below. Injury PDFs: personal use, no redistribution. `ak-static` robots allow `/referee/injury/`, `Crawl-Delay: 1`. |
| Basketball-Reference (terms May 19, 2023; data-use page) | No automation that adversely affects the site; no competing database; no generative-AI training; 20 req/min limit with up to a day's lockout (bot page 403'd to me, read via search summary); robots: Crawl-delay 3, Disallow gamelog/splits/lineups/shooting. | Not used for bulk. |
| Spotrac ([spotrac.com/service](https://www.spotrac.com/service)) | No "data mining, robots, or similar data gathering and extraction tools", no scraper/spider/robot. Robots crawl-delay 5. | Not used. |
| RealGM (terms via search summary, page blocked to scripts) | Use only "click-by-click ... by a human being". | Not used. |
| HoopsHype | robots block Claude/anthropic-ai agents; terms not located. | Not used. |
| FantasyPros ([legal](https://www.fantasypros.com/about/legal/)) | One personal copy; no republishing. robots: crawl-delay 5, `/nba/ranker/` disallowed (ADP pages are not). | Local, personal, 5 s spacing, not committed. |
| Pro Sports Transactions | Cloudflare challenge; "all rights reserved" reported by a client library. | Not used. |
| balldontlie | API key required (401 verified); free 5 req/min; terms not read. | Not used. |
| Wikipedia | CC BY-SA 4.0. | Attribute and share-alike if ever used. |

## Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | **NBA.com "fantasy game" clause** applies to NBA statistics (the ingest track's primary source) and to a public fantasy repo | Real, unassessed by counsel / high for a public repo | Minimum posture: private, non-commercial, attribution, no data committed. *Needs user sign-off*: keep model outputs private too, or request permission. Ingest track should read the NBA terms with this in mind. |
| R2 | ESPN terms forbid scripted access; the API is undocumented and can change (2025-26 ADP already wiped) | Medium / medium | Few requests, own archive, disclose, keep ESPN behind one client module, degrade gracefully (fallback benchmarks). *Needs user sign-off.* |
| R3 | ADP snapshot timing undocumented; benchmark may include in-season information | High / medium | Label as season aggregate; ADP-free benchmarks are the headline; Wayback spot checks; snapshot our own ADP from now on. |
| R4 | ESPN ids to NBA `player_id` matching (97-99% by exact name; rookies and name changes fail) | Certain / low | Alias table, team/position/age tie-break, tests; ambiguous rows stay unmapped rather than guessed. |
| R5 | Injury PDF parser breaks on layout changes; slot enumeration unknown; NBA edge rejects some User-Agents | Medium / medium | Content-based parser, gap report per season, `Mozilla/5.0 (compatible; ...)` agent, per-file cache, ablation only where coverage is proven. |
| R6 | NBA Cup: 2 games per team missing from the schedule until about Dec 10 | Certain / low | Flag provisional weeks; refresh after Cup group play. |
| R7 | League not readable anonymously (403/401 reports exist) | Unknown / medium | Ask for league id; test first; if it fails, cookies in `.env` or manual export. |
| R8 | Leakage through ESPN fields (`percentOwned`, `draftRanks`, team labels on FantasyPros pages are as of today) | High if careless / high | Column allow-list per source; only ADP rank + eligible slots (as of draft) enter features; `History` stays the gate. |

## Alternatives considered

* **Scrape Basketball-Reference / Spotrac for everything:** rejected by their terms; and B-R is not a licence-clean stand-in for NBA stats.
* **Paid API (balldontlie GOAT, $39.99/month) for contracts/injuries:** rejected for now; would still not provide point-in-time history.
* **Skip ESPN:** possible, loses league sync, ADP benchmark and the schedule; kept as the fallback if the user rejects R2.

## Consequences

* The ingest track builds, in order: nightly ESPN snapshot + id map, `adp`, availability spells, league sync, schedule, then
  the injury-report archive (`data-sources.md` section 7). New tables (`adp`, `espn_player_snapshots`, `injury_reports`,
  `injury_spells`, `schedule_games`) require an ADR 0001 update via a dedicated contract task.
* Backtest claims are limited accordingly: injury type from 2019-20, ADP comparison with the stated caveat, no contract
  claim.
* Open questions for the user: league id and anonymous readability; sign-off on R1/R2; final team count and draft date.
