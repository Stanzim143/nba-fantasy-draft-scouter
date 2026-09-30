# ESPN Fantasy API: verified findings

**Date of all tests: 2026-09-22.** Tester: source-research track. Everything marked **VERIFIED** was
executed by me against the live endpoint on that date, without cookies, from an ordinary home connection
(about 60 requests in total, at least 2 s apart, User-Agent naming this project). Everything marked
**READ** comes from package source code or web pages. Everything marked **UNVERIFIED** could not be
tested (usually because it needs our league id). Reproducible code: `docs/research/snippets/espn_*.py`.

## 0. Bottom line

| Question | Answer | Status |
|---|---|---|
| League-agnostic player universe without cookies (ADP, ownership %, auction value, ESPN projections, injury status, eligible slots)? | Yes, one GET, anonymous | VERIFIED |
| Historical ADP for 2015-16 ... 2024-25? | Yes, all 10 seasons return real values. 2025-26 ADP has been wiped (all 140.0). | VERIFIED |
| Is that historical ADP point-in-time (preseason)? | Probably an all-season average of ESPN drafts; timing not documented. Usable as a benchmark with a caveat. | PARTLY (see section 4) |
| Pro-team schedule for 2026-27 (dates, opponents) without cookies? | Yes, 1,200 of 1,230 games (80 of 82 per team, NBA Cup games TBD) | VERIFIED |
| Our league's settings, rosters, free agents, draft, transactions without cookies? | Should work for a league set to public; not tested because I do not have the league id | UNVERIFIED |
| What requires `espn_s2` + `SWID`? | Private leagues (HTTP 401 per espn-api code); some community reports of 401/403 even for public leagues | READ |
| ESPN terms of use | Prohibit scripted access and data collection. This API is undocumented. See section 9. | READ (quoted) |

## 1. Season year convention (VERIFIED)

ESPN `seasonId` is the **end year** of the season: `2027` = 2026-27, `2026` = 2025-26, `2016` = 2015-16.
Evidence: the 2027 payload holds projections for the coming season plus 2026 game logs; the 2016 payload
ranks Anthony Davis first (his 2015-16 hype year), and Curry/Harden/LeBron follow.
`contracts.season_str(start_year)` -> ESPN id is `start_year + 1`.

## 2. The player universe request (VERIFIED)

Minimal working request (returns 3 players, about 5 KB):

```bash
curl -s -A "nba-fantasy-2026-research/0.1 (personal research)" \
  -H 'X-Fantasy-Filter: {"limit":3,"sortPercOwned":{"sortPriority":1,"sortAsc":false},"filterStatsForTopScoringPeriodIds":{"value":1,"additionalValue":["002027","102027"]}}' \
  'https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027/players?view=kona_player_info'
```

Python equivalent: `python docs/research/snippets/espn_players_adp.py 2027 --limit 500 --out adp_2027.csv`.

Gotchas found the hard way:

* **The filter must be top-level** on this games-level URL. `{"players": {...}}` (the wrapper that
  `espn-api` uses on *league* URLs) is silently ignored here and returns the whole universe:
  3,184 players, 24 MB.
* `filterStatsForTopScoringPeriodIds` with `additionalValue: ["00<season>", "10<season>"]` shrinks the
  payload roughly 30x (about 2.5 KB per player instead of about 45 KB). Without it every player carries
  about 80 per-game stat entries. (`value: 0` returns HTTP 400; use 1.)
* `view=players_wl` is tiny (name, ids, eligibleSlots, `ownership.percentOwned` only). It has **no** ADP.
* `view=kona_player_info` is the one with ADP/auction/injury. `mDraftDetail` and no view return `[{},{},{}]`.
* `view=kona_playercard` (136 KB for 3 players) also carries `draftRanksByRankType` but adds nothing we need.
* No documented rate limit. I stayed at one request per 2 s; the full-universe 24 MB pull was done once.
* `lm-api-reads.fantasy.espn.com` accepted our descriptive User-Agent, but `site.api.espn.com`
  (the player-news endpoint used by espn-api) answered **403 Access Denied** to me. `robots.txt` on the
  `lm-api-reads` host also returns 403 (there is no robots file for the API host); `www.espn.com/robots.txt`
  is for the website.

### Sample of the real fields (top of the 2026-27 board; Wembanyama, fetched 2026-09-22)

```json
{"id": 5104157, "fullName": "Victor Wembanyama", "proTeamId": 24, "defaultPositionId": 5,
 "eligibleSlots": [4, 9, 10, 11, 12, 13], "active": true, "injured": false, "injuryStatus": "ACTIVE",
 "lastNewsDate": 1788209495000,
 "ownership": {"averageDraftPosition": 3.06, "averageDraftPositionPercentChange": -0.03,
               "auctionValueAverage": 75.11, "percentOwned": 99.92, "percentStarted": 99.47,
               "percentChange": 0.0, "leagueType": 0, "date": 1790029854816},
 "draftRanksByRankType": {"STANDARD": {"rank": 4, "auctionValue": 60, "published": false},
                          "ROTO": {"rank": 2, "auctionValue": 65, "published": false}},
 "stats": [{"id": "102027", "seasonId": 2027, "statSourceId": 1, "statSplitTypeId": 0,
            "stats": {"0": 1764.0, "1": 235.0, "2": 74.0, "3": 235.0, "6": 817.0, "11": 194.0,
                      "13": 616.0, "14": 1226.0, "15": 386.0, "16": 469.0, "17": 146.0,
                      "28": 31.9, "40": 2137.3, "42": 67.0}}]}
```

Top of the 2026-27 ADP board that day: Jokic 1.66, Gilgeous-Alexander 2.92, Wembanyama 3.06,
Doncic 4.05, Antetokounmpo 5.70.

### Field dictionary (2026-27 universe, 3,184 players)

| Field | Meaning | Coverage / notes |
|---|---|---|
| `id` | ESPN player id (NOT the NBA person id) | all; map through `player_id_map` (section 8) |
| `fullName`, `firstName`, `lastName`, `jersey` | identity | all (`jersey` 1,808) |
| `proTeamId` | ESPN pro team id; 0 = free agent | see team-id table below |
| `defaultPositionId` | 1 PG, 2 SG, 3 SF, 4 PF, 5 C | -1/0 for inactive/unassigned |
| `eligibleSlots` | lineup-slot ids the player may occupy | 2,694 players; rules in section 6 |
| `active` | in the current pool | 1,095 true |
| `injuryStatus` | `ACTIVE`, `DAY_TO_DAY`, `OUT` (813 players carry one; 715 ACTIVE, 85 DTD, 13 OUT) | snapshot only |
| `injured` | boolean | 13 true |
| `lastNewsDate` | epoch ms of the latest news item | 681 players |
| `ownership.averageDraftPosition` | ADP; **140.0 is the "no ADP" sentinel** (also 139.99) | 222 players with a real value (< 139.9) |
| `ownership.auctionValueAverage` | average auction price (dollars, $200 budget) | 0.0 when unranked |
| `ownership.percentOwned`, `percentStarted` | % of ESPN leagues rostering / starting | **snapshot at fetch time, for past seasons the end-of-season state (leaky)** |
| `ownership.date` | epoch ms of the ownership snapshot | populated for 2027 only; null for all past seasons |
| `draftRanksByRankType.STANDARD/ROTO` | ESPN's own rank + auction value; `published:false` | 411 players; present 2023+ only |
| `stats[]` | see stat-id table | see below |

ESPN pro-team ids (espn-api `constant.py` and the live schedule payload agree on all 30 ids; VERIFIED):
1 ATL, 2 BOS, 3 NOP, 4 CHI, 5 CLE, 6 DAL, 7 DEN, 8 DET, 9 GSW, 10 HOU, 11 IND, 12 LAC, 13 LAL, 14 MIA,
15 MIL, 16 MIN, 17 BKN, 18 NYK, 19 ORL, 20 PHI, 21 PHX, 22 POR, 23 SAC, 24 SAS, 25 OKC, 26 UTA, 27 WAS,
28 TOR, 29 MEM, 30 CHA (NBA abbreviations). ESPN's own `abbrev` strings differ from NBA's for six teams in the schedule payload (NY, SA, UTAH,
WSH, NO, GS) and espn-api spells two more differently (PHL, PHO). Key on the numeric id, never on the abbreviation.

### Stats block

`stats[]` entries are keyed by `id = <statSplitTypeId-ish prefix><season>` and carry:

| `statSourceId` | `statSplitTypeId` | id pattern | Meaning |
|---|---|---|---|
| 0 | 0 | `00<season>` | **actual** season totals (present for past seasons: e.g. 2025: Jokic FP 4,879) |
| 1 | 0 | `10<season>` | **ESPN projection** for the season |
| 0 | 1 / 2 / 3 | `01/02/03<season>` | last 7 / 15 / 30 days |
| 0 | 5 | `05<espnGameId>` | **one game** (with `scoringPeriodId` = day index); Jokic had 65 populated game rows for 2025-26 in the 2027 payload |

Stat ids (matches espn-api `STATS_MAP`; I checked 0, 1, 2, 3, 6, 11, 13-17, 28, 40, 42 against real
numbers): 0 PTS, 1 BLK, 2 STL, 3 AST, 4 OREB, 5 DREB, 6 REB, 9 PF, 11 TO, 13 FGM, 14 FGA, 15 FTM, 16 FTA,
17 3PM, 18 3PA, 28 MPG, 40 MIN (total), 41 GS, 42 GP. Note: there is no DD/TD in our scoring.

**Fantasy-point check (VERIFIED).** Applying `config/league.yaml` scoring to the raw projection stats of
Wembanyama, Gilgeous-Alexander, Jokic, Edwards and Doncic (2026-27) reproduces the payload's
`appliedTotal` **exactly** (3968, 4320, 4838, 3402, 4123) when the request has no
`filterStatsForTopScoringPeriodIds`. So our league uses ESPN's default points scoring. When
`filterStatsForTopScoringPeriodIds` is present the payload's `appliedTotal` changes (Wembanyama 2883.0)
and must not be trusted. **Always compute FP from the raw stat ids with our own scoring.**

ESPN historical **projections** (`10<season>`): present for 2018 ... 2027 (about 260-390 players per
season) but **not frozen preseason values**: 2022-23 shows Jokic at 43 projected games, which is a
rest-of-season projection. Do not use them as a benchmark without checking provenance (UNVERIFIED);
2026-27 projections are fine as one more feature/benchmark for the draft board.

## 3. Injury information from ESPN (VERIFIED)

Only a **current snapshot**: `injuryStatus` in {ACTIVE, DAY_TO_DAY, OUT} and `injured`. No body part, no
expected return, no history. The `expectedReturnDate` that espn-api reads exists only on the league
endpoints (READ) and the news endpoint (`site.api.espn.com`, 403 to me). Verdict: not a source for the
injury feature layer; useful only as a current-status snapshot for the live tool.

## 4. Historical ADP: what you get and how far you can trust it

Pull: `python espn_players_adp.py <seasonId> --limit 500` (limit 500 sorted by percent owned; raising it to 900
for 2022-23 added 31 more players with an ADP, all deeper than ADP 133, so nothing in the draftable range was cut).

| seasonId | Season | Players with real ADP (0 < ADP < 140) | Best ADP | Real ADP <= 100 | Note |
|---|---|---|---|---|---|
| 2016 | 2015-16 | 153 | 1.4 (A. Davis) | 97 | one decimal |
| 2017 | 2016-17 | 149 | 1.9 (Westbrook) | 96 | one decimal |
| 2018 | 2017-18 | 147 | 1.9 (Antetokounmpo) | 101 | one decimal |
| 2019 | 2018-19 | 341 | 1.53 | 83 | fractional (draft-count weighted) |
| 2020 | 2019-20 | 358 | 1.29 | 89 | 3 rows with ADP = 0.0 (invalid) |
| 2021 | 2020-21 | 373 | 2.06 | 83 | 7 rows with ADP = 0.0 (invalid) |
| 2022 | 2021-22 | 384 | 2.05 | 85 | |
| 2023 | 2022-23 | 313 | 2.72 | 59 | sparse at the top (36 players below 50) |
| 2024 | 2023-24 | 405 | 1.91 | 80 | |
| 2025 | 2024-25 | 414 | 1.83 | 81 | |
| **2026** | **2025-26** | **1** | 127.04 | 0 | **wiped**: every star reads 140.0 and auction 0.0 |
| 2027 | 2026-27 | 309 | 1.66 | 87 | live, moving daily |

Rules for using it: drop `adp == 0` and `adp >= 139.9`; convert to a **rank** (ADP values are averages and
have gaps); never use `percentOwned/percentStarted` for past seasons (they are end-of-season and leak: A. Davis
2015-16 has ADP 1.4 but 56% ownership because he was hurt).

Timing evidence (why I call it "probably an all-season average"):

* 2015-16 ADP ordering (Davis, Curry, Harden, Durant, LeBron) matches the well-known preseason consensus, while
  `percentOwned` is clearly end-of-season. So the ADP field is not end-of-season state.
* But the payload has `date: null` for past seasons and nothing states the snapshot time. ESPN ADP is an average
  over every ESPN draft of that season, so late drafts and redrafts are included.
* Cross-source check: FantasyPros' "ESPN" column vs this API's ADP on matched players, Spearman rho
  1.00 (2015-16, 2016-17), 0.97 (2018-19), 0.81 (2020-21), 0.84 (2022-23), 0.92 (2024-25). Perfect agreement
  in the early seasons, drift later: the two sources do not always freeze at the same moment.
* A Wayback Machine capture of FantasyPros dated 2015-10-07 lists Curry first (avg 1.7) and Davis second, with
  per-source dates (Yahoo 10/04/2015, ESPN 10/05/2015, CBS 9/29/2015), while FantasyPros' live `?year=2015` page
  now shows Davis first with a different source set (NFBCK instead of CBS). So the live "year" pages are rebuilt
  later and are **not** identical to what was published pre-draft. True point-in-time ADP exists only in
  archives such as the Wayback Machine (see `data-sources.md`, ADP section).

Consequence for the backtest: ESPN ADP is a *reasonable but imperfect* benchmark. Report it as "ESPN average
draft position (season-level aggregate, snapshot time not documented)" and say so in the README.

## 5. Schedule (VERIFIED)

```bash
curl -s -A "nba-fantasy-2026-research/0.1" \
  'https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027?view=proTeamSchedules_wl'
```

Returns 423 KB. `settings.proTeams[*].proGamesByScoringPeriod` maps `scoringPeriodId` -> list of games
`{id, date(ms UTC), homeProTeamId, awayProTeamId, scoringPeriodId, startTimeTBD, statsOfficial, validForLocking}`.

* `scoringPeriodId` is a day counter: 1 = 2026-10-20 (opening night), 174 = 2027-04-11 (last regular-season day).
* 1,200 unique games, exactly 80 per team. The NBA published dates/opponents for 80 of each team's 82 games; the
  other two (played Dec 4-10) depend on NBA Cup group play
  ([nba.com announcement](https://www.nba.com/news/2026-27-nba-regular-season-schedule): opening night Oct 20,
  final day Apr 11, All-Star Feb 19-21, Cup group play Oct 30 - Nov 27, knockouts Dec 4-11). Weeks near Dec 7 are
  therefore incomplete until ESPN adds those games (week of 2026-12-07 shows 0.93 games per team).
* Games per team per Mon-Sun week (`espn_pro_schedule.py`): usually 3-4 mean, min 2, max 4-5; 1.93 in the Cup
  knockout week of 11-30; about 2.0 in All-Star week.
* ESPN matchup weeks: the league's own `settings.scheduleSettings.matchupPeriods` (`{matchupPeriodId:
  [scoringPeriodIds]}`, from espn-api source `base_settings.py`, READ) is the authoritative grouping. Needs the
  league id (`view=mSettings`). Do **not** hard-code Monday-Sunday for the league; week 1 starts on a Tuesday and
  the 20 regular-season weeks + 3 playoff rounds do not fit 25 calendar weeks without ESPN merging/skipping weeks.
* The official NBA static schedule (`cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`) answered
  **403** to my request on 2026-09-22 (the CDN rejects some clients); not verified as a source.

## 6. Position ids and lineup slots (VERIFIED on 2,694 players)

`docs/research/snippets/espn_position_map.py` holds the table and the derivation rules.

| Slot id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Name | PG | SG | SF | PF | C | G | F | SG/SF | G/F | PF/C | F/C | UTIL | BE | IR |

Rules that hold for every one of the 2,694 players carrying `eligibleSlots`:
G (5) iff PG or SG; F (6) iff SF or PF; SG/SF (7) iff SG or SF; G/F (8) iff any non-C; PF/C (9) iff PF or C;
F/C (10) iff SF, PF or C; UTIL/BE/IR always. `defaultPositionId` = base slot id + 1 (1 PG ... 5 C).

For our slots (PG SG SF PF C G F UTIL x3): a **pure C is not eligible for F** and a pure PG/SG is eligible for G.
About 10% of players carry two base positions (622 SG-only, 548 PF-only, 438 C-only, 437 SF-only, 422 PG-only,
74 PG/SG, 59 SF/PF, 58 SG/SF, 36 PF/C in the 2027 universe). ESPN's rule for *earning* a new eligibility is
not in the payload (UNVERIFIED).

## 7. League endpoints (READ; UNVERIFIED without the league id)

From `espn_api` 0.46.0 source (installed in the venv):

| Data | Request (league url = `.../games/fba/seasons/{season}/segments/0/leagues/{league_id}`) | Cookies? |
|---|---|---|
| Settings, teams, rosters, standings | `?view=mTeam&view=mRoster&view=mMatchup&view=mSettings&view=mStandings` | none for a public league (READ) |
| Draft results | `?view=mDraftDetail` (`draftDetail.picks[]`: teamId, playerId, roundId, roundPickNumber, keeper) | same |
| Free agents | `?view=kona_player_info&scoringPeriodId=N` + header `X-Fantasy-Filter: {"players":{"filterStatus":{"value":["FREEAGENT","WAIVERS"]},"limit":50,...}}` (note the `players` wrapper here) | same |
| Transactions | `?view=mTransactions2&scoringPeriodId=N` + filter `{"transactions":{"filterType":{"value":["FREEAGENT","WAIVER","WAIVER_ERROR"]}}}` | same |
| Recent activity feed | `/communication/?view=kona_league_communication` (only >= 2019) | community reports say it wants cookies |
| Box scores | `?view=mMatchupScore&view=mScoreboard&scoringPeriodId=N` | same as roster |
| Matchup periods | in `mSettings` -> `scheduleSettings.matchupPeriods` | same |

Error semantics I observed: a non-existent league id returns **404** `GENERAL_NOT_FOUND` (tested ids 12345,
123456789, 987654321 for 2027 and the `leagueHistory` form for 2016). espn-api documents **401** for a private league
and retries with the alternate `leagueHistory` URL for seasons before 2018. Public-league access is claimed by the
ESPN support article "Making a Private League Viewable to the Public"
([link](https://support.espn.com/hc/en-us/articles/360000991871-Making-a-Private-League-Viewable-to-the-Public)):
non-members can view League Office, Standings, Box Scores and Team Pages via a shared link; the manager list and
message board are never public. espn-api issue #547 (June 2024) reports HTTP 403 for public leagues; unresolved.
**So our league's public flag makes cookies unnecessary in theory; only a test with the league id proves it.**
Open question for the user: the ESPN league id (the number after `leagueId=` in the league URL), and
whether it is already viewable when logged out. Never paste `espn_s2`/`SWID` into chat or the repo; if cookies
turn out to be required they go into `.env` (gitignored).

### The `espn-api` package (READ + VERIFIED install)

* Version 0.46.0 installed, MIT license, PyPI metadata pins `urllib3<=2.2.3` and `requests<3`. The urllib3 cap can
  conflict with future dependency upgrades.
* `League(...)` constructor issues several requests eagerly (active-player list, pro schedule, teams, draft), so a
  single call is heavy. It supports 2019+ only for free agents, box scores and activity.
* Recommendation: write a **thin `requests` client** for the four or five GETs we need (universe, schedule, league
  settings/rosters, transactions, draft) and keep espn-api as a reference. The payloads are plain JSON.

## 8. Mapping ESPN ids to NBA `player_id` (VERIFIED as a feasibility test)

Normalised exact-name match (accents stripped, punctuation removed, suffixes kept) of ESPN players with a real ADP
against the local `nba_api.stats.static` player list (offline; no request to stats.nba.com):

| Season | ESPN players with ADP | unique match | ambiguous | no match |
|---|---|---|---|---|
| 2015-16 | 153 | 148 | 1 | 4 |
| 2018-19 | 341 | 330 | 1 | 10 |
| 2021-22 | 384 | 379 | 1 | 4 |
| 2024-25 | 414 | 411 | 2 | 1 |
| 2026-27 | 309 | 269 | 1 | 39 |

Misses are name-form differences (`Jimmy Butler` vs `Jimmy Butler III`, `Enes Kanter` vs `Enes Freedom`,
`Louis Williams` vs `Lou Williams`, `Wang Zhelin`) and, for 2026-27, **rookies not yet in the static list** (39). So
`player_id_map` needs (a) a small manual alias table, (b) team + position + age tie-breaks for ambiguous names,
(c) rookies added when the ingest track sees their NBA ids. ESPN ids are stable across seasons.

## 9. Terms and politeness

* Disney/ESPN Terms of Use (United States, last updated May 24, 2024,
  [disneytermsofuse.com](https://disneytermsofuse.com/english/), section 3 "Usage Rules"): users may not "access,
  monitor, copy or extract the Disney Products using a robot, spider, script, or other automated means ... data
  mining or web scraping or otherwise compiling ... any collection of data", and the license is "personal,
  noncommercial use only". **Automated use of this API is therefore against the written terms even though it is
  technically open.** It is also unofficial and can change or vanish without notice (the 2025-26 ADP already did).
* Our practical stance: (1) personal, non-commercial use; (2) low volume (a nightly universe pull is one GET);
  (3) descriptive User-Agent; (4) cache raw pulls; (5) **never commit raw ESPN payloads or ADP tables**;
  (6) keep league/private data local; (7) disclose the stance in the README.
* This is a judgement call, not legal advice; the alternative is to not use ESPN at all, which removes the league
  sync and the ADP benchmark. Recorded in ADR 0005 as an accepted risk needing the user's sign-off.

## 10. What I did not test

* Anything on our league (needs league id; public-league anonymous access, transactions, draft results, rosters).
* Whether `kona_player_info` ADP for the *current* season keeps moving through draft week (its `date` field says
  it is live; snapshot daily from now on to build a real point-in-time archive).
* Long-term stability: none of these endpoints is documented by ESPN.
