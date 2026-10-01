# Contributing

Thanks for helping. This repo is built by several people and agents working at the same time, so
the workflow is stricter than usual. The authoritative rules are in [`CLAUDE.md`](CLAUDE.md); this page is
the practical version.

> Outside contributor? A normal fork, branch and pull request is fine; run `./dev ci` first. The worktree
> and `./dev integrate` rules below are the maintainers' multi-agent workflow. Never commit data (see
> [`DATA.md`](DATA.md)).

## Setup

```bash
./dev setup      # creates ./.venv with pinned dependencies; ./dev then uses it automatically
./dev test       # offline tests
./dev app        # draft board; use the synthetic demo mode with no data (localhost only: it has no
                 # authentication, so do not pass --server.address 0.0.0.0)
```

Real data is never distributed (see [`DATA.md`](DATA.md)); rebuild it with `./dev bootstrap`.

The project maintainers share one virtualenv at `~/.venvs/nba-fantasy-2026`; `./dev` uses it by default and
falls back to the `python` on your PATH. `./dev` is a bash script (Git Bash on Windows).

## Workflow

1. **Never edit tracked files in the primary checkout.** Create a worktree:
   `./dev worktree create <task>` (branch `task/<task>`, outside the repo), then `cd` into it.
2. Make small, coherent commits. Stage explicit paths (never `git add -A` in a shared checkout) and
   inspect `git diff` and `git diff --cached` first.
3. Verify with `./dev ci` (tests plus lint).
4. Integrate with `./dev integrate` from the worktree. It takes the repo-wide lock, rebases onto the
   integration branch, re-runs the tests, fast-forwards, and re-runs the tests on the combined state.
   It aborts cleanly on a rebase conflict. Mechanical conflicts you can resolve yourself; a semantic
   conflict (incompatible APIs or schema meanings) means stop and report.
5. Remove the worktree from the primary checkout: `./dev worktree remove <task>`.

Never discard, reset, clean or overwrite work you did not create, and never delete another worker's
`integration.lock`: a stale one is reported by `./dev integrate`, and a person verifies and removes it.

## Who owns what

Tracks own fixed paths (ingest, model + value, backtest, source research, repo hygiene); see the
"Track ownership" table in [ADR 0001](docs/adr/0001-data-contract.md). Shared files
(`pyproject.toml` dependencies, `config/league.yaml`, `src/contracts.py`) change only through a
dedicated small task that states the reason. A schema change is an ADR update plus a coordinated
change, deliberately a little expensive. Only one worker downloads from stats.nba.com at a time, and
only one installs into the shared virtualenv.

## Tests

```bash
./dev test                       # whole suite (network tests excluded)
./dev test tests/test_points.py  # anything after `test` goes to pytest
./dev test -m network            # opt in to live-service tests
./dev test -m "not slow and not network"   # skip slow tests (a -m you pass replaces the default, so repeat "not network")
```

- Markers (registered in `pyproject.toml`): `slow`, `network` (excluded by default) and `real_data`
  (needs ingested data; must skip cleanly when it is absent).
- Tests must not need the internet, credentials or a populated data directory by default. Use
  `make_synthetic_tables()` from `src/synthetic.py` for fixtures, and a temporary `NBA_DATA_DIR`.
- **Synthetic data is for testing code, never for reporting results.**
- Anything that builds features or projections needs a leakage test: build a `History` for season S and
  assert nothing from S or later reached the model (`History.assert_no_future()`).
- Test files in `tests/` have unique basenames (there are no `__init__.py` files there).
  Repo-tooling tests are named `test_repo_*.py`.

## Lint

`./dev lint` runs `ruff check` using the settings in `pyproject.toml`. If ruff is not installed it says so
and falls back to a syntax-only check; it never installs anything. CI installs a pinned ruff
(`.github/requirements-lint.txt`) and requires it (`DEV_REQUIRE_LINT=1`). To lint locally, install ruff
into your venv yourself.

`./dev schedule install|status|remove|run-now` manages the Windows Task Scheduler entries: `--job daily` (the default for every action except `status`, which shows both jobs unless `--job` is given) is the pre-draft
refresh ([ADR 0014](docs/adr/0014-daily-refresh-automation.md)), `--job nightly` the in-season job
([ADR 0018](docs/adr/0018-nightly-inseason-job.md)); it is operations (`src/ops/schedule.py`), not part of the worktree workflow.

## `./dev` environment variables

| Variable | Meaning | Default |
|---|---|---|
| `NBA_WORKTREE_ROOT` | Where task worktrees live | `~/dev-worktrees/nba-fantasy-2026` |
| `NBA_VENV` | Virtualenv used for Python | `~/.venvs/nba-fantasy-2026` |
| `NBA_DATA_DIR` | Shared data directory (read by the Python code) | `~/dev-data/nba-fantasy-2026` |
| `DEV_TEST_CMD` | Shell command that replaces pytest in `test`/`ci`/`integrate` (extra args arrive as `"$@"`) | unset |
| `DEV_RUFF` | Path to a ruff executable, or `none` to disable detection | auto-detect |
| `DEV_REQUIRE_LINT` | `1` makes a missing ruff an error | unset |
| `DEV_LOCK_TRIES` / `DEV_LOCK_SLEEP` | Attempts and seconds between attempts when waiting for the integrate lock | `360` / `5` (30 min) |

## Data and secrets

- Never commit `.env`, cookies (`espn_s2`, `SWID`), raw pulls or licensed data. Copy
  [`.env.example`](.env.example) to `.env` (gitignored) for local secrets. CI installs with
  `constraints.txt` and pins actions by commit SHA; `./dev setup` warns loudly if it must fall back to unpinned deps.
- Data lives outside the repository in `NBA_DATA_DIR`. Respect the terms and rate limits of every source
  before automating a pull; see [ADR 0005](docs/adr/0005-data-sources.md) for the sources, terms and
  accepted/rejected decisions.

## Documentation

- Keep the status table in [`README.md`](README.md) truthful: never report a result that was not
  produced from real data, and never leave a placeholder that looks like a number.
- Structural changes update [`docs/architecture.md`](docs/architecture.md); decisions get an ADR (see the
  [ADR index](docs/adr/README.md)).
- Relative links in the Markdown docs are checked by `tests/test_repo_docs_links.py`.

## Style

`.editorconfig` sets UTF-8, LF endings, 4-space Python indent and 2-space YAML and JSON. Line length is 110.
`dev` and shell scripts must keep LF endings (`.gitattributes` enforces this); CRLF breaks bash.
