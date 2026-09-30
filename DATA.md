# Data sources and terms

**This repository contains no data and never will.** It ships scripts that rebuild the dataset on your
own machine, from the original sources, under those sources' terms. Do not commit or redistribute what
they download (`.gitignore` blocks the data directory; `./dev data pack` is for a *private* handoff only).

| Data | Source | How you get it | Rights basis / caveat |
|---|---|---|---|
| Game logs, players, schedule, rosters | stats.nba.com via [`nba_api`](https://github.com/swar/nba_api) | `./dev bootstrap` (`src.ingest.nba_stats`) | NBA terms: private, non-commercial use only; restrictions on fantasy use and database extraction. Unofficial endpoints; may change or rate-limit. |
| ADP, player universe, league settings | ESPN public endpoints | `./dev bootstrap` (`src.ingest.espn_adp`, `espn_league`) | Disney/ESPN terms; no open data licence; automated extraction is restricted. Private leagues need your own cookies in `.env`. |
| Coaches, contracts, transactions | Wikipedia | `src.ingest.wiki_*` (optional) | CC BY-SA 4.0 / GFDL (Wikidata is CC0). Attribute if you republish. |
| Yahoo rankings | Manual export | Export yourself; see `docs/research/data-sources.md` | Proprietary; personal/internal use only. |
| FantasyPros rankings | Manual export | Export yourself | Proprietary; personal use; redistribution restricted. |

Practical rules: run it for yourself, keep the data local, be polite to the servers (the ingest is
throttled and cached; only one worker should pull at a time), and don't publish per-player rankings
derived from licensed rankings. Not affiliated with the NBA, ESPN, Yahoo or FantasyPros; nothing here is
betting or financial advice. This is a summary, not legal advice; the analysis behind it is in
[`docs/research/data-sources.md`](docs/research/data-sources.md) and
[ADR 0005](docs/adr/0005-data-sources.md).
