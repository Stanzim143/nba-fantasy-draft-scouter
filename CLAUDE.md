# Agent rules for this repository

These rules apply to every agent and human working in this repo. Follow them always. This file is
the canonical operating guide for AI-assisted development here — read it at the start of every
session; it should be short enough for that and complete enough that a fresh session needs nothing
else to start safely.

## Forks and outside contributors

The worktree, `./dev integrate` and verifier rules below are the maintainer's multi-agent workflow. On a
fork, a normal branch and pull request is fine: run `./dev ci` before opening it and keep to the scope,
data-rights (`DATA.md`) and never-commit-data rules, which apply to everyone.

## Fresh-session workflow

For a normal new task, do the equivalent of:

1. Read this file.
2. Check `git status`.
3. Inspect recent relevant git history when useful (not the whole log).
4. Read only the ADR/docs/files relevant to the requested task.
5. Inspect relevant code/tests.
6. Work.

The repository — git history, ADRs, `PLANNING.md`, docs, code, tests — is durable project memory.
Do not reread the entire repository or reconstruct every historical decision unless the task
genuinely requires it. If the user references previous work, reconstruct it from git/docs first
rather than assuming a prior conversation must be recovered. When repository state and old
conversation context disagree, trust the repository unless there is strong evidence it is itself
stale or wrong.

## Context efficiency

Optimize for the minimum context needed to make the correct change.

- Don't keep investigating once there is sufficient evidence to act; don't reread the same files,
  rerun equivalent searches, or re-prove conclusions already supported by direct evidence and
  passing tests.
- Prefer targeted inspection over broad repository sweeps.
- Don't do unrelated cleanup just because it was noticed while working on something else — see
  "Scope discipline" below for what to do with it instead.
- Don't ask open-ended questions like "anything else?" or start an improvement sweep once the
  requested work is done. Once it is correct, tested, documented as necessary, integrated, and
  clean: stop.

## Scope discipline and completeness

Complete the requested scope thoroughly — implementation, tests, relevant documentation,
migration/schema changes, failure handling, integration, and final verification, as applicable.
Prefer a permanent fix over a workaround when the permanent fix is reasonably in scope. Don't
knowingly leave a directly-caused defect, broken test, dangling migration, missing required
documentation, or incomplete integration behind.

Completeness does not mean expanding the task indefinitely. When an adjacent issue turns up:

1. Fix it immediately only if it's caused by the requested change, required for correctness, or so
   tightly coupled that leaving it would make the implementation itself incomplete.
2. Otherwise, record it succinctly in the appropriate tracking location (usually `PLANNING.md`) if
   it's genuinely worth tracking.
3. Don't launch additional agents or side projects for unrelated improvements unless explicitly
   asked.

Don't invent cleanup work to demonstrate thoroughness.

## Search before building

Before adding a new abstraction, helper, component, command, table, convention, or dependency:
search for an existing implementation or established repository pattern, and prefer extending it
over creating a duplicate mechanism. For nontrivial changes, understand the relevant code path
before modifying it. This means a targeted search, not an exhaustive one — use judgment.

## Testing strategy

Use the cheapest test that gives meaningful confidence first:

1. targeted tests for the changed behavior;
2. related subsystem/integration tests when appropriate;
3. the full `./dev ci` only when the workflow requires it (see "Serialized self-integration") or
   before integrating a substantial change.

Don't rerun the full suite after every small edit when targeted tests already give sufficient
feedback. Tests should verify behavior and real boundaries, not just raise the test count. Don't
weaken an existing test to make a change pass unless that test is demonstrably wrong.

## Subagent policy

Subagents are a precision tool, not the default execution mechanism. Use one when:

- multiple genuinely independent workstreams can run in parallel;
- a problem benefits from a clean, isolated investigation context;
- a substantial codebase search can be delegated without duplicating the primary agent's own work;
- independent verification materially reduces regression risk (see below);
- the change touches high-risk logic: architecture, migrations, persistence, auth/security,
  deployment, critical calculations, complex algorithms, or major cross-cutting behavior.

Do not spawn a subagent for a simple search, a small doc change, an obvious single-file edit, a
routine test update, a mechanical refactor, or anything the primary agent can finish and verify
cheaply itself — and not to re-verify work that passing tests already directly prove.

Avoid recursive agent trees: a subagent should not spawn its own subagents without a strong,
specific reason. Prefer one focused agent over several duplicating the same investigation.

## Parallel work

Parallelize only genuinely independent work — not tasks likely to touch the same files, that
depend on each other's findings, or that need sequential design decisions. When running more than
one agent: give each a bounded scope and clear ownership, avoid duplicate investigations, integrate
deliberately, and close out finished agents rather than leaving monitors or watches running. Don't
leave background agents running once their useful work is done.

Tracks (ingest, model + value, backtest, source research, repo hygiene) each own a fixed set of
paths and must not touch the others'. The ownership table is in
[`docs/adr/0001-data-contract.md`](docs/adr/0001-data-contract.md) under "Track ownership"; shared
files (`pyproject.toml` dependencies, `config/league.yaml`, `src/contracts.py`) change only through a
dedicated small task that states the reason. Only one worker downloads from stats.nba.com at a time.

## Independent verification policy

Not every change needs a verifier. For substantial or high-risk work (see "Subagent policy"), use
one fresh-context independent verifier after implementation is complete.

Give the verifier: the task/spec or explicit acceptance criteria, the resulting diff or clearly
identified changed files, relevant repository instructions, and enough context to test the behavior
itself. Do not hand it a narrative explaining why the implementation is supposedly correct —
independence is the point, and a fresh-context reviewer avoids the confirmation bias and inherited
context of one that watched the work happen. It should inspect the diff, resulting behavior, and
tests against the acceptance criteria (not reread the whole implementation conversation) and return
either:

- **PASS**, or
- **FAIL** — with the concrete issue, the affected file/location, reproduction or evidence, and why
  it violates the acceptance criteria.

On a material FAIL: the primary agent fixes it, runs the relevant tests, and re-verifies just the
failed area — don't restart an entire new verification cycle for unrelated aspects that already
passed unless the fix could plausibly affect them. Stop once acceptance criteria pass. Use more than
one independent verifier only when the work spans multiple genuinely high-risk domains that one
reviewer can't credibly cover alone.

## Documentation policy

Documentation should describe durable project truth: architecture decisions, important data
contracts, setup requirements, operational behavior, deployment expectations, new commands or
workflows, intentionally-accepted known limitations, important external data assumptions. Update it
when an implementation changes something a future developer or agent needs to know. Don't create
documentation to narrate a routine edit, and keep task-specific implementation chatter out of
permanent docs.

## ADR policy

Use an ADR for a meaningful architectural or durable design decision, not every implementation
detail. Before writing a new one, check whether an existing ADR already governs the issue and
update or supersede it if that better represents the decision history — don't let ADRs proliferate
for trivial changes. See `docs/adr/README.md` for the numbering/index convention. If a real-data
run later validates or disproves an ADR's stated assumption, don't rewrite the ADR (accepted ADRs
are never rewritten) — append a short, dated addendum recording what was checked and what was
found.

## Git/worktree discipline

Follow the worktree and integration rules below for all substantial changes: start from a known
clean state, isolate work where required, make logically coherent commits, keep unrelated changes
out, and integrate only after required tests/verification pass. Before declaring a task complete,
confirm the intended branch/commit, `git status`, that no unintended worktrees/locks/background
agents remain, and that the required changes are actually integrated — check git state at useful
transition points and at completion, not continuously throughout the task.

### Worktree isolation

Implementation agents MUST work in a dedicated task worktree.

Before modifying any tracked file:

1. Determine whether the current checkout is already a dedicated worktree for
   this task.
2. If it is not, create a task branch and worktree from the current integration
   HEAD.
3. Change into that worktree before editing.
4. Perform all implementation, verification, and commits there.

Do not modify tracked files in the primary integration checkout.

The only exception is when the user explicitly instructs you to work in the
current checkout.

If a worktree cannot be created, stop and report the reason rather than
silently falling back to the integration checkout.

**If you are not already in a task worktree, run: ./dev worktree create <task-name>**

### Worktree safety

Multiple humans or agents may operate on the repository concurrently.

- Never discard, reset, clean, overwrite, or revert changes you did not create.
- Never use destructive Git commands on another task's work.
- Do not reuse another agent's worktree.
- Do not modify another worktree merely to obtain a clean status.
- Do not use `git stash` as a way to hide someone else's work.
- Stage explicit task-owned paths or hunks.
- Avoid `git add -A` or equivalent broad staging in a dirty checkout.
- Before committing, inspect both the working diff and staged diff.
- If unrelated changes prevent verification, report that rather than modifying
  them.

### Serialized self-integration

When work was performed on a dedicated task branch/worktree, the implementation
agent normally integrates its own completed work.

A separate integration agent is not required for ordinary conflict-free work.

Integration is serialized even though implementation may be parallel.

Only one worker may update the integration branch at a time.

Use the repository-provided integration command if one exists.

Otherwise use an exclusive repository-wide lock under the common Git directory,
for example with `flock`, so cooperating agents cannot race while updating the
integration branch.

Conceptually:

```text
acquire integration lock

    identify current integration HEAD
    update task branch onto that baseline
    resolve safe mechanical conflicts if necessary
    run relevant verification on the updated task branch
    fast-forward/integrate into the integration worktree
    run verification required for the combined state
    confirm integration succeeded

release integration lock
```

Do not hold the integration lock while doing ordinary implementation work.

The lock protects only the short integration phase.

#### Integration rules

Before integration:

- the implementation must already be committed;
- the task worktree must contain no accidental uncommitted changes;
- the integration worktree must not contain unexplained dirty changes;
- identify the current integration branch rather than assuming an outdated
  baseline.

While integrating:

- preserve all already-integrated work;
- never resolve a conflict by discarding another task's change;
- never weaken tests or contracts to make branches combine;
- rerun relevant verification after updating against the latest integration
  branch;
- prefer fast-forward integration after updating the task branch when practical;
- preserve coherent task commits unless repository convention says otherwise.

Straightforward mechanical conflicts may be resolved by the worker when the
intended combined behavior is clear.

Examples include:

```text
independent import additions
non-overlapping manifest entries
mechanical generated-file updates
adjacent documentation edits
```

If integration exposes an actual semantic conflict, stop rather than guess.

Examples include:

```text
two incompatible public API changes
different schema meanings
conflicting IR semantics
different persistence behavior
incompatible architectural decisions
tests whose correct expectation depends on choosing one task over another
```

Report that conflict clearly for review.

After successful integration:

- confirm the integration branch contains the task commit(s);
- run `git status --short`;
- remove the task worktree/branch when safe and no longer needed;
- do not leave an abandoned integration/rebase operation behind.

## Completion rule

A task is complete when: the requested acceptance criteria are satisfied; directly-related defects
found during implementation are resolved; appropriate tests pass; required documentation is
current; required independent verification has passed, if applicable; changes are integrated per
this workflow; and repository state is clean, or any intentional remaining state is explicitly
explained. At that point, stop — don't go looking for another task, don't run an extra general
hygiene sweep unless requested, and don't turn "anything else?" into a new scope-expansion cycle.

## Session-boundary rule

One conversation should generally correspond to one coherent unit of work or a closely-related
batch. Stay in the same conversation while debugging the same issue, implementing and testing the
same feature, fixing failures produced by that implementation, or finishing a tightly related
verification loop. Prefer a fresh session after a substantial feature is finished and integrated, an
ADR-sized piece of work is complete, work moves to a materially different subsystem/domain, most of
the current conversation describes already-completed work, or context has been compacted repeatedly
and the repository now holds the durable state that matters. Don't start a new session mid-implementation
just to save tokens if that would lose useful working context — keep working context while it's
relevant; once the repository holds the completed truth, prefer a clean context for the next
substantial task. If a task is stopped mid-flight to continue in a fresh session, leave the
worktree/branch and any commits already made exactly as they are — the next session picks up from
that repository state, not from a re-told narrative.

When the same substantial task is still ongoing but the conversation has grown large, prefer
compacting over pushing through with a bloated context, and prefer re-deriving state from the
repository (git status/log, the task's own worktree, its commits so far) over asking to have it
re-explained.

## Maintaining CLAUDE.md

Update this file when a task reveals a durable lesson that would materially improve how future
agents work in this repo — for example: a canonical command or workflow was established, a
recurring source of mistakes was found, an important architectural constraint became clear, a
repo-specific testing requirement was learned, a required sequencing rule was discovered, a
reliable source of truth for some kind of project information was established, a project-specific
convention keeps mattering, or a previous instruction here turned out to be obsolete or misleading.

Do not update this file for transient task status, one-off debugging notes, temporary branch or
worktree names, current commit hashes, individual feature implementation details already captured
in code/ADR/docs, facts unlikely to matter to a future task, or a verbose account of what happened
in a session — that kind of detail belongs in an ADR, `PLANNING.md`, or feature docs, not here (see
"Documentation policy" above for which).

When you do update this file: edit the existing relevant rule instead of appending a duplicate,
remove or correct anything now stale, keep it concise, prefer durable principles over anecdotes,
and check for duplicated rules, contradictions, and stale instructions while you're in here. The
goal is a high-signal operating manual, not a project diary — don't touch this file just to have
touched it.

## Repo tooling (`./dev`)

The repository-provided command is `./dev` (bash; run it from Git Bash or via `bash ./dev`).

| Command | What it does |
|---|---|
| `./dev worktree create <task>` | Branch `task/<task>` + worktree from the integration HEAD; prints the path |
| `./dev worktree list` | Show worktrees |
| `./dev worktree remove <task>` | Remove a worktree and branch, only if fully integrated and clean |
| `./dev test` | Run pytest with the project's Python |
| `./dev lint` | `ruff check`; if ruff is not installed it says so and falls back to a syntax-only check (never installs anything). `DEV_REQUIRE_LINT=1` makes a missing ruff an error |
| `./dev ci` | `./dev test` + `./dev lint`, reporting both |
| `./dev setup` | Fresh clone: create `./.venv` and install dependencies (then `export NBA_VENV="$PWD/.venv"`) |
| `./dev bootstrap [args]` | Rebuild the dataset from public sources (nba_stats -> preseason_refresh/ADP) and write a board; resumable, `--seasons` limits history |
| `./dev app` | Launch the Streamlit draft board (synthetic demo mode needs no data) |
| `./dev data pack\|unpack` | Zip / restore `<data dir>/processed` (parquet + json; excludes the ESPN league snapshot, backups and licensed rankings) to hand the project to someone PRIVATELY (never publish it, see DATA.md); send the zip privately, never commit it |
| `./dev integrate` | Run from a task worktree: lock, rebase onto integration, verify, fast-forward, verify, unlock |
| `./dev schedule install/status/remove/run-now [--job daily\|nightly]` (`status` without `--job` shows both jobs; other actions default to `daily`) | Manage the per-user Windows Task Scheduler entries: `daily` = pre-draft refresh (ADR 0014), `nightly` = in-season job (ADR 0018); operations, not part of the worktree workflow |

Layout notes:

- The primary checkout lives at `~/dev/NBA Fantasy 2026` (any path outside a synced folder), **outside OneDrive**: OneDrive
  syncing `.git` caused permission errors and risked corruption. Never move it back into a synced folder.
- Task worktrees are also outside OneDrive, under `$NBA_WORKTREE_ROOT`
  (default `~/dev-worktrees/nba-fantasy-2026/<task>`).
- The shared data directory is `~/dev-data/nba-fantasy-2026` (override with `NBA_DATA_DIR`); all
  worktrees read and write the same copy. It is never committed.
- The shared virtualenv lives outside the repo at `~/.venvs/nba-fantasy-2026` (override with `NBA_VENV`).
  Only one worker installs into it at a time.
- Integration branch is whatever the primary checkout has checked out (currently `main`).
- `DEV_TEST_CMD` replaces the pytest command used by `test`/`ci`/`integrate` (used by the `./dev`
  tests so they do not recurse into the suite). A stale `integration.lock` is reported, never
  deleted automatically — verify no live process holds it (its `owner` file names the PID) before a
  person removes it by hand. See `CONTRIBUTING.md` for the full list of `DEV_*` variables.

## Where things live

- CLAUDE.md (this file): how to work in this repo — durable operating rules, canonical commands,
  workflow. Not project status.
- ADRs (`docs/adr/`): architectural/design decisions and why. Never rewritten after acceptance; a
  later ADR supersedes an earlier one, and a validated/disproven assumption gets a dated addendum.
- `PLANNING.md`: genuinely open work, future work, unresolved project-level items.
- Feature/domain docs (`docs/*.md`): durable product behavior, subsystem contracts, operational
  instructions.
- Git history: what changed and when.
- Tests: executable behavioral expectations.

Don't duplicate the same fact across all of these without a strong reason — put it in the one place
that's the source of truth for that kind of knowledge, and link to it from elsewhere if needed.

## Project notes

- Product plan: `PLANNING.md`. League settings: `config/league.yaml`.
- Architecture: `docs/architecture.md`. Decisions: `docs/adr/README.md`. Contributing: `CONTRIBUTING.md`.
- Never commit `.env`, cookies, or licensed/raw data (see `.gitignore`).
