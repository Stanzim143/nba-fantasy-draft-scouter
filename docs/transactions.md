# Transaction tracker

Stay current on trades, signings, extensions, waivers and coach / front-office moves, ranked against the draft board. Design and evidence:
ADR [0017](adr/0017-transactions-tracker.md). It is **descriptive**: it does not change a projection or a rank (ADR 0010 and 0011 found that
"changed team" and "star arrived or left" do not improve the projections).

## The one command you will use

```bash
python -m src.ops.txn_watch --refresh --ack
```

Pulls the last 14 days from ESPN, prints what the ledger has *not shown you yet* (first run: the last 7 days), and marks it read. Leave
`--ack` off to look without marking anything read. The board it ranks against is built live (about 45 s); pass
`--board-csv reports/daily/draft_board_baseline_offseason_debut.csv` to reuse the one the daily refresh already wrote.

```text
Sep 22  EXTENDED  Ausar Thompson (G-F, #77): to DET [rookie scale+extension]
            DET best teammates: Cade Cunningham (#6), Jalen Duren (#16), John Collins (#135); 4 of the next 8 best teammates share his position
Aug 20  TRADE     Peyton Watson (G, #127): DEN -> CLE
            CLE best teammates: Evan Mobley (#15), Donovan Mitchell (#19), James Harden (#38) | left behind at DEN: Nikola Jokić (#1), Jamal Murray (#13), DeMar DeRozan (#104)
(89 other moves involve players outside the board's top 180 or unranked)
```

Other views: `--days 30 --team MIA`, `--player Giannis` (a name search shows unranked players too), `--kind trade`, `--all` (every move,
including camp signings and cuts), `--staff` (head coach, assistant coach and front-office hires, firings, resignations, extensions).

## Where else it shows up

* **Daily refresh** (`python -m src.ops.daily_refresh`, twice a day): the last step, `transactions`, adds the new rows to the ledger and the
  report `reports/daily/latest.md` gains a **League transactions** section: what the ledger first saw in that run, ranked against that
  run's board, and any staff moves. If the board could not be built that run the section says so and lists moves unranked.
* **App**: the draft-board app has a **Transactions** tab (last N days, teams, kinds, player search, coach moves). It reads the local ledger
  and the board already loaded; it never makes a request. It says how old the ledger is.
* **Ledger by hand**: `python -m src.ingest.espn_transactions --days 120` (or `--from 2015-07-01 --to 2026-09-25` for history, monthly
  windows, cached). `python -m src.ingest.preseason_refresh --with-extras` runs the same step with a 45-day lookback.

## What the ledger is

`<data dir>/processed/league_transactions.parquet`, append-only, one row per person per action (`kind`: `signed`, `resigned`, `extended`,
`converted`, `waived`, `claimed`, `trade_in`, `trade_out`, and for staff `hired`, `fired`, `resigned_staff`, `extended_staff`;
`other` for a row nothing could be read from). `first_seen` is when *we* first saw a row, which is what "new" means; ESPN's own dates are
days. `processed/espn_transactions_report.json` has the last ingest's parse counts and every name that did not resolve to a player.

## How far to trust it

* Per-player latest move agreed with the independent NBA roster snapshot for 207 of 209 players (99%) on 2026-09-25. The two misses show
  the real limitation: **ESPN's feed is not complete** (one player's arrival on his current team has no row at all), and same-day rows from
  two teams can conflict. The daily report's roster diff remains the safety net for who is where.
* Names outside the NBA index (summer-camp and Exhibit-10 fringe players) keep a blank `player_id`; they are listed in the report and never
  guessed.
* A sentence it cannot read is counted, not guessed (`parse.unparsed_sentences` in the ingest report), and a name that does not look
  like a name is refused. The grammar was hardened after an independent review found wrong-move cases, but free text can still surprise it. If the
  unparsed count climbs after a phrasing change on ESPN's side, extend the grammar in `src/ingest/espn_transactions.py` and re-run over the
  cached feed: rows whose feed text is now read differently replace the old ones and keep their `first_seen`.
* The first run's last 60 days all count as new (`first_seen` is the ingest time); older rows are history (`first_seen` is their date), so a
  move the feed publishes more than 60 days late is never flagged new.
* It reports the news a day late at worst (the feed is dated, not timed); run `--refresh` for the newest.
