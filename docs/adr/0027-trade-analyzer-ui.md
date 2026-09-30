# ADR 0027: Trade analyzer confidence/notes surfaced in the app

**Status:** accepted, 2026-09-28
**Code:** `src/inseason/trade.py` (`confidence_bucket`, `confidence_label`), `src/app/inseason_view.py` (`run_trade`, `trade_caveats`,
`_flag_overlay_for`, `TRADE_*_COLUMNS`), `src/app/pages/inseason.py` (Trade analyzer tab), `src/app/glossary.py` (Trade analyzer group).
Tests: `tests/inseason/test_lineup_trade.py`, `tests/app/test_inseason_view.py`, `tests/app/test_glossary.py`.

## Context

ADR 0015 shipped `src/inseason/trade.py`'s `evaluate_trade`/`analyze` (a CLI and a library function) and a minimal Streamlit tab
(`_trade_tab` in `src/app/pages/inseason.py`) that already called `run_trade` and showed the FP delta and a give/get table. Two things
were missing before this ADR:

1. `TradeResult` has no literal "confidence score" or "notes" field — only `verdict` (a three-way string from `delta` vs `tolerance`),
   `dropped`/`added` (forced roster moves) and `ctx.notes` (context-level caveats such as a missing schedule). The verdict alone is not a
   legible confidence read: "favours you" says nothing about *how much* to trust that over the tolerance band a 1.5%-imprecise ROS
   projection already admits to.
2. Nothing surfaced a traded player's own advisory flags — current ESPN injury status, or the same `return_flag`/`lm_flag` advisory
   text the draft board shows (ADR 0021/0023, ADR 0024) — at the point where a trade decision is actually made.
3. The tab had no partner picker, even though `analyze()`/the CLI's `--partner` already supported scoring the other side of the trade.

## Decisions

* **D1. Confidence is a magnitude bucket on the existing tolerance band, not a new statistic.** `confidence_bucket(delta, tolerance)`
  (`trade.py`, pure and unit-tested) reads `|delta| / tolerance`: `<= 1` is `roughly_even` (the same rule `verdict` already uses),
  `<= CONFIDENCE_CLEAR_MULT` (3.0) is a `lean_your_way`/`lean_other_way`, beyond that is `clear_win`/`clear_loss`. Inventing a
  probability or a second model here would overstate precision the ROS projection does not have (the same reasoning ADR 0021/0023/0024
  use for their advisory games); a magnitude read on the band already used for `verdict` does not. NaN-safe: a non-finite `delta` reads
  as no signal (`roughly_even`); a non-finite or non-positive `tolerance` falls back to sign-only bucketing rather than raising.
* **D2. Display reuses the app's existing idiom, not new CSS.** The app has no bespoke tier/badge styling anywhere (the draft board's
  own `tier` column is a plain filterable integer); the closest existing "visually legible tier" idiom is `st.metric`'s own delta
  coloring and the sidebar's `st.success`/`st.error` feedback. The confidence bucket is shown as an `st.metric` help string plus a
  matching `st.success`/`st.info`/`st.warning`/`st.error` banner (`_CONFIDENCE_BANNER` in `inseason.py`) — reusing components already
  in the codebase rather than adding a new visual system for one tab.
* **D3. Per-player advisory notes reuse the board's own overlays, adapted to the rest-of-season frame.** `trade_caveats(ctx, ids)`
  (`inseason_view.py`) builds, for exactly the traded player ids:
  * current ESPN injury status from `ctx.injuries` (already loaded by `InSeasonContext`) when it is anything other than `ACTIVE`
    (`OUT`, `DAY_TO_DAY`, `SUSPENDED`, ...) — the same vocabulary `src/features/risk.py`'s status haircuts use;
  * the same `return_flag`/`lm_flag` advisory text ADR 0021/0023 and ADR 0024 compute for the draft board, via `_flag_overlay_for`,
    which builds `History.until(ctx.tables, ctx.season)` (exactly as `src/value/board.py` does) and a tiny adapter frame with
    `proj_gp = ros_games` and `proj_fppg = ros_fppg` for just the traded players — so the advisory games are sized against what is
    left of *this* season rather than the full schedule, while the cohort test itself (last season's absence pattern) is untouched.
  Every line states plainly that it is advisory and does not change the projection, matching the framing `load_flag.py`/`return_flag.py`
  already use. The function never raises: a missing/short history or unrecognised player degrades to fewer notes, same as the board's
  own overlays. `ctx.notes` (context-level caveats, e.g. a missing schedule) are shown alongside these per-player notes.
* **D4. The partner side is exposed through the existing `analyze(..., partner=...)`, not reimplemented.** The tab adds a selectbox of
  `RosterChoice.others` labels; picking one calls `run_trade(..., partner=label)`, which mirrors the CLI's `--partner` and scores the
  other side with `evaluate_trade` run a second time, give/get swapped. Both sides get their own confidence bucket and notes.
* **D5. Every new displayed field gets its own glossary entry (ADR-independent convention from `docs/categories.md` section 20 /
  `src/app/glossary.py`).** New keys are prefixed `trade_*` (`trade_side`, `trade_player`, `trade_ros_total_fp`, `trade_verdict`,
  `trade_confidence`, `trade_delta`, `trade_tolerance`, `trade_before_value`, `trade_after_value`, `trade_dropped`,
  `trade_empty_slots`, `trade_partner`) rather than reusing bare words like `verdict`/`delta`/`tolerance`, so they do not collide with
  any future board column of the same plain name and so every one is independently documented. `TRADE_DISPLAY_COLUMNS`
  (`inseason_view.py`) is the single list `tests/app/test_glossary.py` checks against the glossary's "Trade analyzer" group, the same
  discipline already used for `BOARD_COLUMNS`, `WATCH_COLUMNS`, etc. The tab's own `_column_config` helper (a copy of
  `draft_board.py`'s, since `glossary.py` deliberately has no Streamlit import) attaches the `help=` tooltip to every `st.dataframe` the
  same way `draft_board.py` already does; a source-level test pins that every `st.dataframe(` call in `inseason.py` passes
  `column_config=`.

## Consequences

* The trade analyzer is reachable through normal navigation (it was already a tab; nothing new needed wiring in `state.py`/`loader.py`,
  since `src/app/pages/inseason.py` is Streamlit's own multipage mechanism off `src/app/draft_board.py`).
* `confidence_bucket`/`confidence_label` are plain functions with no Streamlit or pandas dependency, usable from the CLI in future
  without pulling in the app.
* Recomputing `return_flag`/`lm_flag` per trade (rather than once per board) costs a `History.until` slice plus two small overlay calls
  on 1-4 players; this is cheap compared to `evaluate_trade` itself, which already runs a lineup optimisation per side.
* Explicitly out of scope: retrofitting `column_config`/glossary tooltips onto the in-season page's pre-existing Schedule/Rest-of-season
  tabs (`ROS_COLUMNS`, the schedule tables) — those columns predate this ADR and were never wired to the glossary; this ADR only
  guarantees coverage for the Trade analyzer tab it adds to. A future task could close that gap the same way `f06e65a` did for the
  draft board.
