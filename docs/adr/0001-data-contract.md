# ADR 0001: Shared data contract

**Status:** accepted, 2026-09-22
**Code:** `src/contracts.py`, `src/store.py`, `src/synthetic.py`

## Context

Ingest, features, models, value engine, backtest and app are built in parallel by
independent workers, each in its own git worktree. Without a fixed interface they would
invent incompatible table shapes, and the incompatibility would only surface at merge time.

## Decision

1. **One contract module** (`src/contracts.py`) defines every shared table (`TableSpec`),
   the season helpers, the `History` leakage guard and the `Projector` protocol.
   Producers call `validate_table` (via `store.write_table`); consumers may rely on it.
2. **Tables are Parquet files** under `data_dir()/processed/<table>.parquet`, written
   atomically. `data_dir()` defaults to `~/dev-data/nba-fantasy-2026` (override with
   `NBA_DATA_DIR`). It sits outside the repo and OneDrive so all worktrees share one copy.
   Raw pulls live under `data_dir()/raw/<source>/` so backtests can be rebuilt offline.
3. **Canonical identity is the NBA `PERSON_ID`** (`player_id`). Every other source is
   mapped through `player_id_map`.
4. **Regular season only.** ESPN H2H points ignores playoffs.
5. **No row for a game not played.** Games missed = `team_games` rows for the player's
   team minus the player's `game_logs` rows. (Roster tenure is handled by features.)
6. **Leakage is prevented structurally.** Models never receive the raw tables: they get a
   `History`, built by `History.until(tables, target_season)`, which drops every
   season >= the target. `History.assert_no_future()` is the check the leakage tests call.
7. **`Projector` protocol:** `name: str` and `project(history) -> projections table`.
   The backtest harness, the draft board and the ablation runner accept any `Projector`.
8. **Scoring keys vs columns:** league config uses `3PM`/`TO`; columns are `fg3m`/`tov`.
   `STAT_COLUMN_MAP` is the single translation.

## Tables

| Table | Key | Producer | Notes |
|---|---|---|---|
| `game_logs` | game_id, player_id | ingest | box-score identities are validated (pts = 2*fgm + fg3m + ftm etc.) |
| `team_games` | game_id, team_id | ingest | every team game, used for availability |
| `players` | player_id | ingest | static attributes; nullable birthdate/position/draft |
| `player_season_bio` | season, player_id | ingest | `age_at_season_start` as of Oct 1 |
| `player_id_map` | source, source_id | ingest (ids) | canonical map; confidence + match method |
| `projections` | season, player_id, model | models | per-game stat projections, FPPG, GP, floor/ceiling |

## Synthetic data

`make_synthetic_tables()` generates a contract-valid league with a known generating process
(age curves, roles, injuries, rookies, retirements). Model/value/backtest work is developed and
unit-tested against it before real data exists. **No reported result may use synthetic data.**

## Track ownership (parallel work)

| Track | Owns | Must not touch |
|---|---|---|
| Ingest | `src/ingest/nba_*`, `tests/ingest/` | model/value/backtest code |
| Model + value | `src/models/`, `src/value/` (except `points.py`, `league.py`, which stay stable) | ingest, backtest |
| Backtest | `src/backtest/`, `tests/backtest/` | model code |
| Source research | `docs/adr/0002+`, `docs/research/` | code |
| Repo hygiene | `.github/`, `README.md`, `CLAUDE.md` | src |

Shared files (`pyproject.toml`, `config/league.yaml`, `src/contracts.py`) change only through a
dedicated small task that states the reason. Only one worker downloads from stats.nba.com at a
time, and only one installs into the shared virtualenv.

## Consequences

* Real data must be transformed to fit these schemas; anything the source cannot provide is
  nullable and documented, not silently invented.
* A schema change is an ADR update plus a coordinated change, deliberately a little expensive.
