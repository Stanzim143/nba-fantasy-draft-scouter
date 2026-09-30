# NBA Fantasy Draft Scouter

[![CI](https://github.com/Stanzim143/nba-fantasy-draft-scouter/actions/workflows/ci.yml/badge.svg)](https://github.com/Stanzim143/nba-fantasy-draft-scouter/actions/workflows/ci.yml)

A projection engine, draft board and walk-forward backtest for an **ESPN head-to-head points** fantasy
basketball league. It projects each player's season from information available before it starts, values
players over replacement under the league's scoring, and scores its own forecasts against what happened
over ten past seasons.

> **Honest status.** An interpretable projection system with evidence of value over a naive
> last-season baseline on *rank order*, not on error size. **ESPN ADP beats it at the top of the draft
> board**, and most of its miss error is availability (injuries). Several extra feature layers were tried
> and mostly did not help; the negative results are reported, not hidden. Known methodology limits are in
> [`docs/limitations.md`](docs/limitations.md). Numbers: [`docs/project-overview.md`](docs/project-overview.md#results).

## Quickstart (about 5 minutes, no data needed)

Needs Python 3.12 or 3.13 and bash (Git Bash on Windows; macOS/Linux terminal as is).

```bash
git clone <repo-url> && cd <repo>
./dev setup        # creates ./.venv and installs pinned dependencies
./dev test         # offline tests, synthetic data only
./dev app          # opens the draft board; tick "Use synthetic demo data" in the sidebar
```

No bash (plain PowerShell or cmd)? Use pip instead:

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # offline tests
nba-fantasy app                                     # also: nba-fantasy bootstrap
```

A `.devcontainer` is included (Codespaces works). Hosting notes: [`docs/deploy.md`](docs/deploy.md).

**Stuck?** `./dev setup` needs Python 3.12 or 3.13 (`python --version`); a newer one fails on pinned
dependencies. If `./dev` says "command not found" on Windows, run it from Git Bash or as `bash ./dev ...`.

## Use your real data

The repo ships **no data** (source terms forbid redistribution, see [`DATA.md`](DATA.md)). Rebuild it
from the public sources with one command; it is resumable and throttled:

```bash
./dev bootstrap --seasons 2023-24:2025-26   # quick trial window (a few minutes)
./dev bootstrap                             # full history, 2015-16 onward (long, be polite)
./dev app
```

Data goes to `~/dev-data/nba-fantasy-2026` (override with `NBA_DATA_DIR`), outside the repo. Optional:
copy `.env.example` to `.env` for live league sync; export Yahoo/FantasyPros rankings by hand.
Set your own league's scoring, roster and draft in [`config/league.yaml`](config/league.yaml).

## Common commands

| Task | Command |
|---|---|
| Rebuild data and board | `./dev bootstrap` (`--help` for flags) |
| Backtest a model | `python -m src.backtest --model baseline --seasons 2016-17:2025-26 --out reports/` |
| Draft board CSV | `python -m src.value.board --season 2026-27 --model baseline --out board.csv` |
| Tests / lint / both | `./dev test`, `./dev lint`, `./dev ci` |

## Where to read next

- [`docs/project-overview.md`](docs/project-overview.md): full status table, methodology, results
- [`docs/architecture.md`](docs/architecture.md): pipeline, data contract, leakage guard
- [`docs/limitations.md`](docs/limitations.md): what the evaluation can and cannot claim
- [`docs/adr/README.md`](docs/adr/README.md): design decisions; [`CONTRIBUTING.md`](CONTRIBUTING.md): how to contribute
- [`PLANNING.md`](PLANNING.md): open work; [`CLAUDE.md`](CLAUDE.md): operating rules for AI agents and contributors

## License

MIT for the code and docs ([`LICENSE`](LICENSE)). Third-party data is not included or licensed; see
[`DATA.md`](DATA.md).
