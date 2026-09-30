# Architecture decision records

An ADR captures one decision: the context, what was decided, and the consequences. They are short,
numbered in order and never rewritten after acceptance; a later ADR supersedes an earlier one and says
so. Shared interfaces (`src/contracts.py`, `config/league.yaml`) change only together with an ADR.

## Index

Status as of 2026-09-29.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-data-contract.md) | Shared data contract | Accepted |
| [0002](0002-nba-ingest.md) | NBA ingestion (`src/ingest`) | Accepted |
| [0003](0003-projection-model.md) | Projection model design | Accepted |
| [0004](0004-backtest-methodology.md) | Backtest design | Accepted |
| [0005](0005-data-sources.md) | Data sources and terms of use | Accepted |
| [0006](0006-injury-layer.md) | Injury/availability feature layer | Accepted |
| [0007](0007-adp-ingest.md) | ADP ingest (`src/ingest/espn_adp.py`) | Accepted |
| [0008](0008-espn-league-sync.md) | ESPN league sync (our real league) | Accepted |
| [0009](0009-streamlit-app.md) | Live draft board app | Accepted |
| [0010](0010-roster-context-layer.md) | Roster-context feature layer | Accepted |
| [0011](0011-transactions-layer.md) | Roster-transactions feature layer | Accepted |
| [0012](0012-offseason-layer.md) | Preseason roster update, Summer League / preseason layer, breakout watchlist | Accepted |
| [0013](0013-contract-layer.md) | Contract layer (rookie-scale clock; no lift) | Accepted |
| [0014](0014-daily-refresh-automation.md) | Daily refresh automation and Windows Task Scheduler | Accepted |
| [0015](0015-inseason-tools.md) | In-season tools: schedule awareness, rest-of-season projection, trade analyzer, waiver finder | Accepted |
| [0016](0016-offseason-gaps.md) | Offseason gaps: stash and undrafted debutants, origin proxies, draft-day risk overlay | Accepted |
| [0017](0017-transactions-tracker.md) | Live transactions tracker: ESPN feed, dated ledger, ranked digest | Accepted |
| [0018](0018-nightly-inseason-job.md) | The in-season nightly job: incremental ingest, league sync, ROS, waiver/lineup artifacts, season-window gate, second scheduled task | Accepted |
| [0019](0019-contract-terms-layer.md) | Veteran contract-terms layer (Wikipedia contract events; no reliable lift, unvalidated display flag) | Accepted |
| [0020](0020-coach-impact.md) | Coach impact: does the system follow the coach, and who benefits? | Accepted |
| [0021](0021-return-from-injury-study.md) | Return-from-injury study: does the availability model over-discount a long absence with a healthy tail? | Accepted |
| [0022](0022-return-health-feature.md) | 'Returned and healthy' availability feature: lead block and tail health for the availability model: registered, not recommended | Accepted |
| [0023](0023-return-flag.md) | Advisory return-from-absence flag on the board: fixed +5 GP / total-FP judgement for a >= 60% block, no projection change | Accepted |
| [0024](0024-load-management.md) | Load management: a proxy (no rest label exists) is a descriptive association, not incremental; advisory short-absences flag, no projection change | Accepted |
| [0025](0025-draft-live-sync.md) | Draft-day live sync: auto-detect opponents' picks from the ESPN league sync, polling with a stale-aware lock, manual "Mark drafted" as the permanent fallback | Accepted |
| [0026](0026-need-adjusted-board.md) | "Best available by need": crossing the VORP board with roster construction (positional need), an unvalidated heuristic, new tab, no projection change | Accepted |
| [0027](0027-trade-analyzer-ui.md) | Trade analyzer confidence/notes surfaced in the app: a magnitude-bucketed confidence read on the existing tolerance band, per-player advisory caveats, and a partner-side picker | Accepted |
| [0028](0028-external-rankings-compare.md) | External rankings comparison: manually-exported Yahoo and FantasyPros snapshots parsed, matched onto player_id, and compared against our own board rank, with a documented rank_delta sign convention | Accepted |
| [0029](0029-rankings-disagreement-flag.md) | Advisory "external rankings disagreement" flag: a fixed, data-grounded 50-rank-spot threshold on ADR 0028's rank_delta, surfaced session-scoped on the draft board's Debutants & risk tab, no projection change | Accepted |
| [0030](0030-methodology-critique-fixes.md) | Methodology critique: rate-shrinkage tuning on predicted (not actual) MPG, ADP + model arm and per-tier comparison vs ADP, risk-group availability calibration, position-prior ablation, disclosed limits | Accepted |

## Writing a new ADR

1. Take the next free number and copy the shape of [0001](0001-data-contract.md): title, status and
   date, context, decision, consequences (plus tables or code pointers where they help).
2. Keep it to the decision. Long research belongs in `docs/research/`, referenced from the ADR.
3. Status is one of: proposed, accepted, superseded by NNNN.
4. Add it to the index above in the same change.
