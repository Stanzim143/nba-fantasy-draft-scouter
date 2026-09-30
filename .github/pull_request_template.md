## What and why

<!-- One or two sentences: what changes, and the reason. Link the ADR or issue if there is one. -->

## Checklist

- [ ] Work was done on a task branch/worktree and integrated with `./dev integrate` (see CONTRIBUTING.md)
- [ ] `./dev ci` passes locally (tests + lint)
- [ ] New behaviour has tests; leakage-sensitive code has a leakage test
- [ ] No secrets, cookies, `.env`, raw or licensed data are committed
- [ ] Shared files (`pyproject.toml` dependencies, `config/league.yaml`, `src/contracts.py`) are untouched, or the reason is stated above and the ADR is updated
- [ ] Docs updated (README status table, `docs/architecture.md`, ADR index) if behaviour or structure changed
- [ ] Any reported number comes from real data, never from the synthetic league

## Results (if this changes model output or the backtest)

<!-- Paste the before/after metrics and the command that reproduces them. -->
