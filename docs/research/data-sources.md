# External data sources: research, evidence and verdicts

**Date:** 2026-09-22. **Track:** source research. **Companion files:** `espn-api-findings.md` (ESPN detail),
`snippets/*.py` (prototypes), `docs/adr/0005-data-sources.md` (decisions).

Evidence legend used everywhere:

* **V** = verified by me on 2026-09-22 by running a request or a parser.
* **R** = read in official documentation, source code or a web page (link given), not executed by me.
* **U** = unverified / could not be tested (reason given).

Politeness during testing: at most one request per 2 s per host (5 s for FantasyPros, which asks for a crawl
delay of 5), descriptive User-Agent, no cookies, no logins, no CAPTCHA bypass, nothing against `stats.nba.com`.
I did not fetch pages from Basketball-Reference, Spotrac, HoopsHype, RealGM or Pro Sports Transactions for data
(RealGM and Pro Sports Transactions sit behind bot challenges; I stopped there). I only read their robots.txt and terms.

## 1. Executive summary

| Need | Recommendation | Verdict | Confidence |
|---|---|---|---|
| ESPN player universe (ADP, ownership, auction, ESPN projections, injury flag, eligible slots, ids) | Direct anonymous GET, snapshot daily from today | **USE** (with accepted ToS risk) | V |
| ESPN league sync (settings, rosters, free agents, draft, transactions) | Own thin `requests` client; needs league id; espn-api as reference | **USE** (untested on our league) | R/U |
| ESPN pro schedule + weekly games per team | `proTeamSchedules_wl`, plus league `matchupPeriods` | **USE** | V |
| Injuries, primary | NBA official injury report PDFs, 2018-12-19 to today, parse with `pdftotext -table` | **USE** | V |
| Injuries, fallback / backbone | Availability spells derived from `team_games` minus `game_logs` (2015-16 on, no external source) | **USE** | design |
| Injury type before Dec 2018 | No legally clean source found | **AVOID** (accept the gap) | R |
| Historical ADP benchmark | ESPN API (2015-16 to 2024-25, 2026-27) + FantasyPros consensus pages (2015-16 to 2025-26) as second opinion and gap filler; Wayback captures for true point-in-time spot checks | **IMPLEMENTED** — `python -m src.ingest.espn_adp` (ADR 0007) | V |
| Fallback benchmark if ADP is disputed | Previous-season total FP rank; ADP-free "minutes x per-minute" consensus | **USE** | design |
| Contracts | Only a rookie-scale proxy from draft year; no scrapeable, point-in-time source | **LATER** (hypothesis layer, likely drop) | V/R |
| Transactions / preseason moves | Own daily ESPN snapshots + opening-night roster from box scores; Wikipedia (CC BY-SA) as a dated, machine-readable cross-check | **USE (snapshots) / LATER (Wikipedia)** | V |
| Summer League / preseason **box scores** | stats.nba.com `leaguegamelog` (`LeagueID=15`; `SeasonType="Pre Season"`), same source and R1 stance as the game logs | **USE** (ADR 0012, built) | V |
| Depth charts / projected minutes | Nothing legal and machine-readable found | **LATER / AVOID** | R |
| Basketball-Reference | Robots + terms restrict automation | **AVOID** for bulk; manual lookups fine | R |
| Spotrac, RealGM, HoopsHype | Terms forbid automated collection (RealGM: human click-by-click only) | **AVOID** | R |
| Pro Sports Transactions | Cloudflare bot challenge on the one path I tried (`robots.txt`); terms not readable | **AVOID** | V/U |
| balldontlie | Free tier: teams, players, games only; injuries need paid tier and are current-only | **LATER** (not needed) | V/R |

## 2. The finding that matters most: NBA.com Terms of Use vs a fantasy tool

The NBA.com Terms of Use (fetched 2026-09-22, **Last Updated July 13, 2026**,
[nba.com/termsofuse](https://www.nba.com/termsofuse), section "NBA STATISTICS") state that by using NBA
Statistics you agree that they:

* need "prominent attribution to NBA.com";
* may be used only "for legitimate news reporting or private, non-commercial purposes";
* "may not be used in connection with any fantasy game or other commercial product or service";
* "may not be used in connection with any website, product, or service that features a database ... of
  comprehensive, regularly updated statistics".

In my reading that covers statistics the NBA publishes through its services, which plausibly includes what
`nba_api` and the `stats.nba.com` endpoints deliver (another track owns that source; I did not test it, and the
terms page does not name those hosts). Read literally, a
public repository whose purpose is fantasy projections sits on the wrong side of the "fantasy game" clause,
regardless of the fact that we are non-commercial. I am not a lawyer and this is not legal advice, but the
project should decide deliberately instead of drifting:

1. **Minimum posture (recommended regardless):** private, non-commercial use; attribute NBA.com; never commit
   or publish raw NBA statistics or a game-log database; publish code and aggregate backtest metrics, not data.
2. **Stronger posture (user decision):** keep model outputs private too (do not publish per-player rankings
   derived from NBA statistics), or ask the NBA for written permission.
3. Basketball-Reference is *not* a cleaner replacement: its terms forbid automation that hurts the site, forbid
   building a competing database, forbid training generative-AI models on its data, and its robots.txt
   disallows game logs (section 4.5).

The same terms give the official injury PDFs no special licence. They are "material of the Services": personal,
non-commercial download only, no redistribution. See section 4.3.

## 3. Comparison table

| Source | Provides | Access | Auth | Rate limit | ToS / robots (URL) | History | Verdict |
|---|---|---|---|---|---|---|---|
| ESPN player universe `.../games/fba/seasons/{y}/players?view=kona_player_info` | ADP, ownership %, auction value, ESPN projections, injury flag, eligible slots, ESPN ids, per-game logs | GET + `X-Fantasy-Filter` header | none (V) | undocumented; I used 1 req / 2 s | Disney ToU forbids scripts/scraping ([disneytermsofuse.com](https://disneytermsofuse.com/english/), R); no robots file on API host | 2010-2027 answers; real ADP 2015-16..2024-25 and 2026-27 (V) | USE, low volume, local only |
| ESPN league endpoints | settings, rosters, FA, draft, transactions, matchup periods | GET on `.../segments/0/leagues/{id}` | none if league public (R/U); cookies if private | undocumented | same | league lifetime | USE (test needs league id) |
| ESPN `proTeamSchedules_wl` | every NBA game, day-indexed | GET `.../seasons/2027?view=proTeamSchedules_wl` | none (V) | undocumented | same | current season | USE |
| NBA official injury report PDFs `ak-static.cms.nba.com/referee/injury/...pdf` | game-day status (Out/Doubtful/Questionable/Probable/Available) + reason text, per team, timestamped | direct file GET | none (V) | none stated; be gentle | NBA.com ToU (personal, non-commercial, no redistribution; R) | files exist from 2018-12-19 (V by HEAD probes) | USE |
| ESPN injury flag | current status only | in player universe | none | - | as ESPN | none | snapshot only |
| Basketball-Reference | stats, injuries page (current), contracts | HTML | none | 20 req/min then a session "jail" up to a day (R via search summary; bot-traffic page returned 403 to me) | terms last updated May 19, 2023: no automation that hurts performance, no competing DB, no generative-AI training (V, read from the page); `robots.txt`: Crawl-delay 3, Disallow `*/gamelog/`, `*/splits/`, `*/lineups/`, `*/shooting/` (V) | deep | AVOID bulk |
| Pro Sports Transactions | injuries/IL, trades, signings history | HTML | none | - | Cloudflare managed challenge on `robots.txt` (V; pages not tried); site terms not readable | since 1900s claimed (R) | AVOID |
| balldontlie | teams/players/games (free), injuries (ALL-STAR $9.99/mo), contracts/standings/box scores (GOAT $39.99/mo) | REST | API key (401 without, V) | free 5 req/min, ALL-STAR 60, GOAT 600 (R) | terms at app.balldontlie.io (not read) | games since 1946; injury docs describe current entries only (R) | LATER |
| Spotrac | contracts, cap | HTML | none | robots `Crawl-delay: 5` (V) | ToS forbids "data mining, robots, or similar" and scrapers (V, [spotrac.com/service](https://www.spotrac.com/service)) | multi-year | AVOID |
| HoopsHype | salaries, contracts | HTML | none | - | robots disallow Claude/anthropic-ai agents (V); site terms not located | multi-year | AVOID |
| RealGM | contracts, transactions, ADP-type pages | HTML | none | 403 to scripts (V) | ToU: use "by click-by-click selections directly by a human being" (R via search summary of [terms](https://basketball.realgm.com/info/terms-of-use)) | deep | AVOID |
| FantasyPros ADP `/nba/adp/overall.php?year=` | consensus ADP (Yahoo, ESPN, CBS or NFBC, AVG) | HTML table | none | robots `Crawl-delay: 5` (V) | ToU: single personal copy, no republishing (V, [fantasypros.com/about/legal](https://www.fantasypros.com/about/legal/)) | 2015-16..2026-27 (V, 8 of 12 seasons fetched) | USE local only |
| Wayback Machine captures of FantasyPros / ESPN pages | dated snapshots | `web.archive.org/web/<ts>id_/<url>` and CDX API | none | polite use | original site terms still apply | FantasyPros ADP page captured monthly from Oct 2014 (V) | USE for spot checks |
| Hashtag Basketball ADP | current ADP across Yahoo/Fantrax/ESPN | HTML | none | robots only blocks Semrush (V) | terms not read | history unknown | LATER |
| Wikipedia team-season pages | trades, signings, waivers with sources; page revision history gives as-of | MediaWiki API | none | polite | CC BY-SA 4.0 | back to past seasons | LATER |
| Kaggle injury datasets | scraped Pro Sports Transactions / official reports | download | Kaggle login | - | licences not visible to me (U); derived from restricted sources | 2010-2025 claimed | AVOID |

## 4. Per-source detail

### 4.1 ESPN league data (question 1a)

**What it provides:** league settings (scoring, slots, matchup periods), teams and rosters, standings, free
agents with ownership, draft picks, transactions, box scores. The community package `espn-api` (0.46.0, MIT,
installed in the venv) wraps exactly these `lm-api-reads.fantasy.espn.com/apis/v3/games/fba/...` calls; I read its
basketball module (`league.py`, `player.py`, `constant.py`, `requests/espn_requests.py`).

**Access without cookies (R, U):** ESPN's own support article says a league set to "Public" is viewable by
non-members via a link (League Office, Standings, Box Scores, Team Pages; never the manager list or message
board) ([ESPN support](https://support.espn.com/hc/en-us/articles/360000991871-Making-a-Private-League-Viewable-to-the-Public)).
The espn-api code sends cookies only when both are supplied, and raises `ESPNAccessDenied` on HTTP 401.
Issue #547 (June 2024) reports 403 for public leagues, unresolved. **I could not test our league** (no league id).
Requests that are documented as needing cookies even for public leagues by the community: the activity feed
(`kona_league_communication`). Private-league data always needs `espn_s2` and `SWID`.

**Verdict: USE**, via a thin own client (5 GETs), keep espn-api as reference. See `espn-api-findings.md` section 7.

**Integration sketch:** a local-only `league_snapshot` (not in the public contract): `league_settings.json`
(verify the scoring dict equals `config/league.yaml`, alert on drift, resolve the "13 vs 10 teams" question from
`settings.size`), `rosters`, `free_agents` (ESPN id -> `player_id_map` -> our projections), `draft_picks`,
`transactions`. Never written to the repo.

### 4.2 ESPN player universe (question 1b)

Fully documented in `espn-api-findings.md`. Short answer: yes, anonymously, one request returns ADP,
ownership %, auction value, ESPN projected stat lines (raw stat ids), injury status and eligible slots for the whole
pool, for 2026-27 and, with the same request, for older seasons. **V.** Caveats: ADP for 2025-26 has been wiped,
the ownership numbers of past seasons are end-of-season (leaky), the ADP snapshot time of past seasons is
undocumented, and it is against the Disney ToU to automate access.

**Integration sketch (proposed, needs a contract change through a dedicated task):**
`espn_player_snapshots(snapshot_ts, espn_id, player_id, pro_team_id, injury_status, percent_owned, adp,
auction_value, eligible_slots, proj_gp, proj_mpg, proj_stat_* )` written nightly to `data_dir()/raw/espn/`
and to `processed/`. Start the nightly job **now**: it is the only way to get a true point-in-time archive of the
current season's ADP and injury flags (draft is unscheduled, ADP is moving daily).

### 4.3 NBA official injury report (question 2)

* **What/where (V):** one PDF per report time at
  `https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>[_<MM>]<AM|PM>.pdf`.
  Contents: every team's game, every player listed with `Current Status` in {Out, Doubtful, Questionable,
  Probable, Available} and a `Reason` such as `Injury/Illness - Right Knee; Sprain`, `G League - Two-Way`,
  `Not With Team`. Each report is timestamped ("Injury Report: 12/15/25 05:30 PM"), so rows are **as-of safe**.
* **Depth (V, by HEAD probes; absent keys answer 403, present answer 200):** nothing at 2016-11-10, 2016-12-05,
  2017-01-10, 2017-11-10, 2017-12-05, 2018-01-10, 2018-03-14, 2018-11-10, 2018-12-05, 2018-12-12. Present from
  **2018-12-19** and at every later date I sampled (2019-01-10, 2019-03-14, 2019-10-25, 2019-11/12, 2020-01/02,
  2020-08-05, 2020-12-26, 2021-01-10, 2021-12-01, 2022-12-01, 2023-12-01, 2025-12-15). Official page:
  [official.nba.com/nba-injury-report-2025-26-season](https://official.nba.com/nba-injury-report-2025-26-season/)
  (links to the 2025-26 and 2022-23 seasons; the report list itself is script-rendered, so file names must be
  constructed). The `nbainjuries` package (MIT, [GitHub](https://github.com/mxufc29/nbainjuries)) says
  data "in this format" exist since 2021-22, uses `tabula-py` (needs Java), warns that data are missing for preseason
  games, All-Star break and postseason gaps (R). My probes show older files exist back to Dec 2018 but in layouts that package
  does not target.
* **File-name slots (V partly):** `05PM` (5:30 PM ET) and `01PM` exist on every sampled 2019-2021 date; 2021-12-01 also
  has 08AM, 01PM, 03PM, 06PM; 2025-26 uses quarter-hour names such as `06_15PM`. The full slot list per day is **not
  known** (U); enumerating means probing candidate names. Plan on "last report of the day before tip-off" per game
  day, and verify coverage with a gap report.
* **Layouts (V):** (a) 2018-19 to about 2020-21: Category, Reason, Current Status, Previous Status; (b) 2021-22 to 2024-25:
  Current Status, Reason; (c) 2025-26: same columns, reason text `Injury/Illness - <body part>; <detail>`.
* **Parse feasibility (V):** `pdftotext -table` (xpdf 4.06, present on this machine; no pip install) gives one row per
  player; a content-based parser (`snippets/nba_injury_report.py`) matched an independent status-token count exactly on the
  two files I hand-checked (55/55 and 100/100) and produced no unknown statuses on four samples (2019, 2021, 2023, 2025).
  `-layout` interleaves wrapped lines and must not be used. Full-archive parse rate is U.
* **Connection quirk (V):** `ak-static.cms.nba.com` and `cdn.nba.com` **reset the TCP connection** when the User-Agent contains an
  e-mail address in parentheses, and `cdn.nba.com` answers 403 to the schedule JSON even with a browser-like agent.
  A `Mozilla/5.0 (compatible; <project>/<version>; <purpose>)` agent worked for the PDFs.
* **Terms:** NBA.com ToU (R, quoted in section 2): personal, non-commercial download; no redistribution. The official injury
  page footer: "No portion of NBA.com may be duplicated, redistributed or manipulated in any form." `ak-static.cms.nba.com/robots.txt` (V) disallows `/wp-admin/`, `/wp-content/uploads/`, `/wp-content/plugins/` and any URL with a query string, sets `Crawl-Delay: 1`, and does **not** disallow `/referee/injury/`; the same file is served by `official.nba.com`. Keep to one request per 1-2 s.
* **What it does not give:** injury *duration* or expected return, and nothing for the offseason (the offseason
  news lives elsewhere; the ESPN flag is the only live offseason signal I found).
* **Verdict: USE as the primary injury-type source, from 2018-12-19.** Backtest seasons before 2018-19 have **no**
  injury-type data; the injury layer's ablation must be run on 2019-20 onward (full seasons) and stated as such.

**Integration sketch:** `injury_reports(report_ts, game_date, matchup, team_abbr, player_name, player_id, status, category,
body_part, reason_text)`; key (report_ts, game_date, player_id). Player names come as "Last, First" with suffixes
("Moore Jr., Wendell"); map to `player_id` via the ingest track's name + team + date tie-break (same problems as 8 in
`espn-api-findings.md`). Feature use: `games_missed_by_body_part`, `days_since_last_listed`, `recurrence_count`,
computed only from reports with `report_ts < season_cutoff` (History guard).

**Fallback (design, no external source):** derive availability spells from `team_games` minus `game_logs` (already in
the contract): spell length, spell count, days since a spell, recurrence of long spells, for every season from 2015-16.
This drops injury *type* but keeps 100% coverage and no legal exposure. The other candidates (Pro Sports Transactions,
Kaggle copies of it) are restricted or of unknown licence; treat their pre-2018 injury text as unavailable rather than
forcing it.

### 4.4 Other injury sources (question 2, evidence)

* **ESPN:** current flag only (section 4.2 above).
* **Pro Sports Transactions:** `robots.txt` answers a Cloudflare managed challenge ("Just a moment...",
  HTTP 403) to a scripted client (V). Bypassing bot challenges is out of bounds for this project; the maintainers of
  `pro_sports_transactions` note that direct requests are typically blocked and that "all rights reserved" apply (R,
  [PyPI](https://pypi.org/project/pro_sports_transactions/)). AVOID.
* **Basketball-Reference:** its injuries page lists the current situation only; the site's terms and robots
  are in section 4.5.
* **balldontlie:** `GET /v1/player_injuries` (cursor pagination, filters by team/player id) returns current entries with
  `status`, `return_date`, `description`; ALL-STAR tier ($9.99/month) and up; the docs do not describe history. The API
  answers 401 without a key (V). Not needed given the official report.
* **Kaggle datasets** ("NBA Injury Stats 1951-2023", "2016-2025 NBA injury data", others): the pages did not expose a
  licence to my fetch (U) and at least one is described as scraped from Pro Sports Transactions. Also a Substack archive
  (Oct 2021 - Jun 2024) says it is no longer maintained. AVOID as a committed dependency.

### 4.5 Contracts / salary (question 3)

| Source | Terms and access (evidence) | Point-in-time? |
|---|---|---|
| Spotrac | ToS prohibits "any automated use of the system ... data mining, robots, or similar data gathering and extraction tools" and "spider, robot, ... scraper" ([spotrac.com/service](https://www.spotrac.com/service), V); robots `Crawl-delay: 5` | site shows current tables; per-year pages exist (R), but retroactive edits (buyouts, extensions) are not versioned |
| HoopsHype | robots disallow Claude/anthropic-ai agents (V); its terms page was not locatable at the URLs I tried (U) | current tables |
| RealGM | terms require "click-by-click selections directly by a human being" (R); scripts get 403 (V) | current tables |
| Basketball-Reference | `/contracts/` pages; terms May 19, 2023: no automated access that "adversely impacts site performance", no competing database, no generative-AI training use (V from the page); bot policy 20 req/min (R, page 403'd to me); robots disallow `*/gamelog/`, `*/splits/` | current tables |
| balldontlie contracts | GOAT tier only, $39.99/month; `/v1/contracts/players`, `/teams`, `/aggregates` (R); depth unknown (U) | unknown |

**Point-in-time is the real blocker.** For a leakage-free "contract year" flag at preseason S you need the contract status
**as known on that date**. None of the sites versions its tables; a scraped 2026 page will show later extensions and
buyouts. Wayback captures could restore point-in-time for a handful of pages but multiply the terms problems.

**Verdict: LATER; keep as the "hypothesis" layer in PLANNING.md and be prepared to drop it.**
Cheap, legal, leak-free proxy that can be backtested today: **rookie-scale final year** =
`season_start - draft_year == 3` for first-round picks (computed from `players.draft_year`, `draft_round`). Everything else
(extensions, max-contract flags) would need a dated signing log; without history the layer cannot show lift in a
walk-forward test, so no claim should be made about it. If the user still wants it for the live season only, enter
contract-year flags for the top ~150 players by hand into a local YAML.

### 4.6 Transactions, roster moves, preseason news (question 4)

* **What is machine-readable, free and as-of safe:**
  1. **Our own nightly ESPN universe snapshots** (`proTeamId`, `injuryStatus`, `lastNewsDate`, ADP): as-of by construction,
     from 2026-09-22 forward. **V** for the payload, design for the archive.
  2. **Opening-night roster from box scores** for the backtest: a fantasy draft happens a few days before opening night,
     and by then every preseason trade/signing is public. So "team of each player in his first regular-season game (or the
     team on the opening-night roster)" is a fair as-of-draft feature for season S. Player-level moves *after* opening night
     are correctly excluded. Depends on ingest's `game_logs`. Slight caveat: it also reveals who is not on any roster.
  3. **League transactions** from our own ESPN league (waivers, adds, drops) for the live tool; see 4.1.
  4. **Wikipedia** team-season pages have Trades / Free agency / Re-signed / Additions / Subtractions sections (V for the
     2026-27 76ers page: sections 11-16 exist) and the MediaWiki API exposes revision timestamps (V), so an as-of
     reconstruction is possible. CC BY-SA 4.0 (share-alike + attribution applies to any derived text; facts themselves are
     free). Hand-editable, unofficial, quality varies: use as a cross-check, not as the backbone. LATER.
  5. **NBA.com transactions page** (`nba.com/players/transactions`) loads its data by script (the HTML holds no table);
     the feed behind it belongs to the `stats.nba.com` family that another track owns, so I did not touch it. Candidate for
     the ingest track to evaluate, subject to the NBA.com ToU.
* **Not usable:** Pro Sports Transactions (bot-blocked), RealGM (human-only), Spotrac transactions (ToS), `site.api.espn.com`
  transactions (403 in my test).

### 4.7 Schedule for weekly H2H (question 5)

* **ESPN pro schedule (V):** see `espn-api-findings.md` section 5 and `snippets/espn_pro_schedule.py`. 1,200 games,
  80 per team, days indexed by `scoringPeriodId` (1 = 2026-10-20). The missing 2 games per team are decided after NBA Cup
  group play and land Dec 4-10, so weekly counts for Dec are provisional until ESPN adds them.
* **Official:** the league announcement gives dates (opening night Oct 20, final day Apr 11, All-Star Feb 19-21, Cup Oct
  30 - Dec 11) ([NBA.com](https://www.nba.com/news/2026-27-nba-regular-season-schedule), V). The static JSON
  `cdn.nba.com/static/json/staticData/scheduleLeagueV2.json` answered 403 to me (U). No downloadable file
  format is offered on the announcement page (only nba.com/schedule and an iCal service at nba.ecal.com, R).
* **Games per matchup week:** take the league's own `scheduleSettings.matchupPeriods` (mSettings) and count each team's games
  in those scoring periods. Do not assume Monday-Sunday; week 1 begins on a Tuesday. Playoff rounds are weeks 21-23 in our
  config; ESPN's regular season here ends after 20 matchup periods, so the calendar-to-matchup mapping must come from ESPN.
* **Backtest:** weekly schedules for past seasons come from `team_games` (actual games), no external source needed.

**Integration sketch:** `schedule_games(season, espn_game_id, scoring_period, game_date, home_team_id, away_team_id,
is_provisional)` and a helper `games_in_matchup(team_id, matchup_period)`. Refresh nightly until the Cup games are added.

### 4.8 Historical ADP / consensus rankings (question 6)

**Status: implemented**, 2026-09-23. See `docs/adr/0007-adp-ingest.md` for the design decisions, real match-rate
numbers and known limitations, and `src/ingest/espn_adp.py` / `src/ingest/id_map.py` / `src/ingest/http_cache.py`
for the code (`python -m src.ingest.espn_adp --seasons 2015-16:2026-27`). The research below is the verified plan
this shipped against; it is left as-is as the record of what was checked before building.

Candidates checked:

| Source | Coverage | Point-in-time? | Legal | Verdict |
|---|---|---|---|---|
| ESPN API (`ownership.averageDraftPosition`) | 2015-16..2024-25, 2026-27 (V); 2025-26 wiped | season-level aggregate; snapshot time undocumented | Disney ToU forbids scripts (R) | USE local only, benchmark #1 |
| FantasyPros `?year=Y` | 2015-16..2025-26 (fetched 2015, 2016, 2018, 2020, 2022, 2024, 2025, plus the bare 2026-27 page; others expected); 203-266 players per season; Yahoo/ESPN/(CBS or NFBCK)/AVG | rebuilt later (the live 2015 page differs from the Oct 2015 capture); player team/status shown are today's | ToU: single personal copy, no republishing; robots crawl-delay 5 (V) | USE local only, benchmark #2 and fills 2025-26 |
| Wayback captures of the same pages | monthly since Oct 2014 (V CDX); the Oct 2015 capture opens with per-source dates (Yahoo 10/04/2015, ESPN 10/05, CBS 9/29) | **yes, dated** | original site terms apply; Internet Archive access itself is open | USE to validate the two above on a few seasons |
| Yahoo ADP page | claimed to exist ([Yahoo](https://basketball.fantasysports.yahoo.com/nba/draftanalysis)); not fetched | current only (R) | Yahoo ToS not read | LATER |
| Hashtag Basketball | current ADP across Yahoo/Fantrax/ESPN (R) | history U | terms not read | LATER |
| NFBC | paid; appears as a FantasyPros column in 2015-16 (V) | - | - | not needed |
| Kaggle / GitHub datasets | search found scrapers and injury sets, **no historical NBA ADP dataset with a visible licence** | - | - | none |

**Honest conclusion:** there is no clean, openly licensed, point-in-time historical ADP dataset. What exists is
public data that both sites' terms restrict for automated collection; using it for a personal, non-committed benchmark
is a small, disclosed risk. Because snapshot timing is undocumented, the benchmark must be labelled as such.

**Fallback benchmark set** (make these the headline, ADP the bonus):
1. `naive_last_season`: rank by previous-season total fantasy points (or FPPG x GP), the plan's benchmark 1.
2. `naive_minutes`: previous-season minutes x per-minute production regressed to the mean (no external data).
3. `espn_adp` and `fp_consensus` where available (2015-16..2024-25 for ESPN; 2015-16..2025-26 for FantasyPros).

Suggested README wording: "*ADP benchmark: average draft position from ESPN (via its unofficial fantasy API) and
FantasyPros consensus pages, collected once for personal research. The values are season-level aggregates whose
snapshot time the sources do not document, so they can include a little in-season information; treat the comparison with
ADP as indicative. The underlying tables are not redistributed. No claim is made that the model beats ADP in every
season; see the results table for per-season numbers.*"

**Integration sketch:** `adp(season, player_id, source, adp, rank, n_drafts_or_sources, snapshot_kind, as_of)`, key
(season, source, player_id); `snapshot_kind` in {season_aggregate, dated_capture, live}. Convert ADP to within-season
rank before comparing; drop ADP = 0 and >= 139.9 as "no ADP"; players without ADP get rank = N + 1 (documented).

### 4.9 Preseason / summer-league minutes and depth charts (question 7)

* Summer League and preseason box scores are published on NBA.com and its stats services (subject to the NBA.com ToU
  clause in section 2). ESPN's per-game stat rows (`statSplitTypeId 5`) I saw were regular-season games only (U for
  preseason).
* Depth-chart-style data for the NBA: I found no free, machine-readable and license-clean source. Commercial fantasy
  sites publish projected minutes behind paywalls or terms that restrict copying (FantasyPros ToU, RealGM, Rotowire
  pages not evaluated).
* The market already prices this information: ESPN ADP and `percentStarted` move with preseason news. So the daily ESPN snapshot
  (section 4.2) captures it indirectly at no extra legal cost.
* **Verdict for depth charts and projected minutes: LATER / AVOID.** Revisit only if the backtest shows big misses from role changes
  and a licensed source exists.
* **Update 2026-09-24: the box scores are the useful part, and they are usable.** Verified against the live API: Summer League
  (`LeagueID=15`, 11 summers, 2015 to 2026 with none in 2020) and preseason (`SeasonType="Pre Season"`, every year through 2025)
  return full player box scores. Ingested by `src/ingest/nba_offseason.py`; findings and the evidence they support are in
  ADR 0012 (notably: `playergamelogs` has placeholder ids for most pre-2024 summers, and the preseason carries the signal while
  Summer League alone is weak).

## 5. Proposed contract additions (do not edit `src/contracts.py` from this track)

Each is a candidate for a dedicated small task with an ADR 0001 update.

| Table | Key | Producer | Feeds |
|---|---|---|---|
| `adp` | season, source, player_id | ingest (ESPN, FantasyPros) | backtest benchmark, draft board ADP gap |
| `espn_player_snapshots` | snapshot_ts, espn_id | ingest (nightly) | live tool, point-in-time archive |
| `injury_reports` | report_ts, game_date, player_id | ingest (official PDFs) | injury feature layer |
| `injury_spells` (derived) | season, player_id, spell_start_game | features | availability features (all seasons) |
| `schedule_games` | season, espn_game_id | ingest (ESPN) | weekly-games tool |
| `player_id_map` rows | source in {espn, nba_injury_name} | ingest | everything (contract already lists `espn`) |

## 6. Public-repo hygiene (question 8)

| Source | Commit raw? | Commit derived? |
|---|---|---|
| ESPN payloads (universe, league, schedule) | **Never** | Only aggregates (e.g. Spearman vs ADP per season). No per-player ADP or ownership tables. |
| FantasyPros / Yahoo / Wayback ADP pages | **Never** | Same as ESPN |
| NBA official injury PDFs and parsed rows | **Never** (ToU: no redistribution) | Aggregated injury-feature importance charts are fine; no player-level tables |
| NBA stats (`nba_api`) | **Never** | Model outputs: see section 2; recommended default is not to commit player-level tables |
| balldontlie, Spotrac, etc. (if ever used) | **Never** | aggregates only |
| Wikipedia | Text is CC BY-SA 4.0: would need attribution and share-alike; avoid copying prose | facts + source revision ids only |
| Our league (rosters, owners, ids, cookies) | **Never** | none |
| Synthetic data (`make_synthetic_tables`) | yes | yes |

**`.gitignore` additions (text only, another track owns the file):**

```gitignore
# secrets and cookies (ESPN espn_s2 / SWID live only in .env)
.env
.env.*
!.env.example
*.cookies
espn_s2*
swid*
# any data that may ever come from an external source
data/
!data/.gitkeep
*.parquet
*.duckdb
*.pdf
*.har
# prototype output
docs/research/snippets/out/
docs/research/snippets/*.csv
docs/research/snippets/*.json
```

Add a CI/pre-commit check that fails on (a) files larger than 1 MB, (b) the strings `espn_s2=`, `SWID=`, `swid=`, (c)
any tracked `.pdf`/`.parquet`/`.csv` outside `tests/fixtures`. Test fixtures must be synthetic or hand-typed, never
excerpts of a licensed source.

**README wording (draft):**

> *Data and terms.* This project uses publicly reachable data for personal, non-commercial research. Player statistics
> come from NBA.com/NBA Stats and are attributed to NBA.com; ADP benchmarks come from ESPN and FantasyPros pages; league
> data comes from my own ESPN league. None of this data is redistributed in this repository, and no credentials are stored
> in it: everything is fetched at run time and cached locally (`~/dev-data`, outside the repo). ESPN's fantasy API is
> undocumented and its terms restrict automated access, so requests are few, spaced and identified. If a rights holder
> objects, the corresponding fetcher will be removed. Results in the README are aggregates computed from this data.

## 7. What to build next (ingest track) - order, effort, risk

| # | Item | Effort | Risk | Why now |
|---|---|---|---|---|
| 1 | **Nightly ESPN universe snapshot** (`espn_player_snapshots`) + id mapping to `player_id_map` | S (1-2 days) | Low technical; ESPN ToS accepted risk | The draft is unscheduled and ADP moves daily. Start the archive today; it also gives injury flags and eligible slots. |
| 2 | **`adp` table**: ESPN API 2015-16..2024-25 + 2026-27, FantasyPros 2015-16..2025-26 (local only) with the cleaning rules above; add Wayback spot checks for 2-3 seasons | M (2-3 days) | Medium: id matching (about 97-99%), snapshot-time caveat | Needed for the benchmark; low effort because requests are few. |
| 3 | **Availability spells** from `team_games` minus `game_logs` | S | Low | Injury layer for all 11 seasons with no external data. |
| 4 | **ESPN league sync** (settings check vs `league.yaml`, rosters, FA, draft, transactions, `matchupPeriods`) | S-M | Medium: public-league anonymous access unproven; may need cookies | Blocked on the league id (open question). Also settles teams = 10 or 13. |
| 5 | **Schedule** (`schedule_games`, games-in-matchup helper) | S | Low | Data is verified; do it after the league id gives the real matchup periods. Refresh after Dec 10. |
| 6 | **Official injury report archive + parser** -> `injury_reports` | L (4-6 days) | Medium-high: three layouts, slot enumeration unknown, name mapping, NBA ToU | Gives injury type from 2018-12-19; only justified after item 3 shows the layer has promise. |
| 7 | Contracts | - | - | LATER; only the rookie-scale proxy (free, from `players`). |
| 8 | Wikipedia transactions, preseason minutes | - | - | LATER. |

## 8. Open questions for the user

1. **ESPN league id** (from the league URL, `leagueId=`), and is the league readable when you are logged out? Without
   it the league-sync claims stay unverified. (No cookies needed to answer.)
2. **Accept the ESPN/Disney ToS risk** for low-volume, local-only automated access (ADR 0005 recommends yes) or
   restrict ESPN to manual CSV exports from the browser?
3. **NBA.com "fantasy game" clause:** do you accept the minimum posture in section 2 (private, non-commercial,
   attribution, no data committed), or do you want model outputs kept private / permission requested?
4. Final **team count** (10 or 13) and the draft date; both change replacement-level and the ADP snapshot timing.
5. Is a paid API acceptable for a month (balldontlie GOAT: contracts, standings) if contracts are pursued? My advice: no.
