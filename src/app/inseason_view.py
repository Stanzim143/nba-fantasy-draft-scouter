"""Plain, Streamlit-free functions behind the in-season page (``src/app/pages/inseason.py``).

Kept separate so the page stays a thin ``st.*`` wiring layer and everything that decides what is shown is
unit-testable (``tests/app/test_inseason_view.py``). Nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inseason.context import (
    ContextUnavailable, InSeasonContext, RosterChoice, free_agent_ids, load_context, resolve_roster,
)
from src.inseason.lineup import NameError_, bench_weight_for, resolve_names
from src.inseason.schedule import ABBR_OF_TEAM_ID
from src.inseason.trade import analyze, confidence_bucket, confidence_label, describe
from src.inseason.waivers import ADD_COLS, STREAM_COLS, WaiverReport, find_waivers

__all__ = ["ContextUnavailable", "NameError_", "load", "week_table", "weeks_summary", "ros_table", "player_options",
           "roster_choice", "run_trade", "run_waivers", "ADD_COLS", "STREAM_COLS", "trade_caveats",
           "TRADE_LINES_COLUMNS", "TRADE_SUMMARY_COLUMNS", "TRADE_DISPLAY_COLUMNS"]

ROS_COLUMNS = ["ros_rank", "name", "position", "team", "gp", "fppg_to_date", "prior_fppg", "ros_fppg", "avail",
               "ros_games", "ros_total_fp", "ros_vorp", "w_sample"]

#: Columns of the per-player "give/get" table the Trade analyzer tab shows (``src/app/pages/inseason.py``).
TRADE_LINES_COLUMNS = ["trade_side", "trade_player", "trade_ros_total_fp"]
#: Columns of the one-row verdict/confidence summary table the Trade analyzer tab shows for each side.
TRADE_SUMMARY_COLUMNS = ["trade_verdict", "trade_confidence", "trade_before_value", "trade_after_value",
                        "trade_tolerance", "trade_dropped", "trade_empty_slots"]
#: Every glossary-backed column/field the Trade analyzer tab can display, including the partner picker's
#: own help text (``trade_partner``); used by ``tests/app/test_glossary.py`` to pin full coverage.
TRADE_DISPLAY_COLUMNS = TRADE_LINES_COLUMNS + TRADE_SUMMARY_COLUMNS + ["trade_delta", "trade_partner"]


def load(season: str | None, as_of, *, model: str = "baseline", synthetic: bool = False,
         data_dir: str | None = None, league_id: int | None = None) -> InSeasonContext:
    """Build the in-season context, or raise ``ContextUnavailable`` with a message safe to show verbatim."""
    return load_context(season or None, as_of or None, model=model, synthetic=synthetic,
                        data_dir=Path(data_dir) if data_dir else None, league_id=league_id)


def _team_label(tid) -> str:
    return ABBR_OF_TEAM_ID.get(int(tid), str(int(tid))) if pd.notna(tid) else ""


def weeks_summary(ctx: InSeasonContext) -> pd.DataFrame:
    """League-wide by matchup week: length, mean/min/max games, heavy and light team counts, back-to-backs."""
    if ctx.weekly is None or ctx.weekly.empty:
        return pd.DataFrame()
    return ctx.weekly.groupby(["week", "kind", "start", "end", "n_days"]).agg(
        mean_games=("games", "mean"), min_games=("games", "min"), max_games=("games", "max"),
        heavy_teams=("heavy", "sum"), light_teams=("light", "sum"), b2b_total=("b2b", "sum")).reset_index()


def week_table(ctx: InSeasonContext, week: int) -> pd.DataFrame:
    """One matchup week by team: games, games left after ``as_of``, back-to-backs, off-night games, flags."""
    if ctx.weekly is None:
        return pd.DataFrame()
    w = ctx.weekly[ctx.weekly["week"] == week].sort_values(["games", "off_night_games"], ascending=False)
    return w[["abbr", "games", "games_left", "b2b", "off_night_games", "heavy", "light"]].reset_index(drop=True)


def ros_table(ctx: InSeasonContext, *, search: str = "", position: str = "All", top: int = 100) -> pd.DataFrame:
    r = ctx.ros.copy()
    r["team"] = r["team_id"].map(_team_label)
    if search:
        r = r[r["name"].fillna("").str.contains(search, case=False, regex=False)]
    if position != "All":
        from src.value.positions import eligible_slots

        r = r[[position in eligible_slots(p) for p in r["position"]]]
    return r[[c for c in ROS_COLUMNS if c in r.columns]].head(top).reset_index(drop=True)


def player_options(ctx: InSeasonContext) -> dict[int, str]:
    """player_id -> 'Name (POS)' for pickers, best rest-of-season value first."""
    r = ctx.ros.drop_duplicates("player_id")
    return {int(p): f"{n} ({pos or '?'})" for p, n, pos in zip(r["player_id"], r["name"], r["position"])}


def roster_choice(ctx: InSeasonContext, *, team_id: int | None = None, names: str | None = None,
                  mock_team: int | None = None) -> RosterChoice:
    return resolve_roster(ctx, team_id=team_id, roster_names=names or None, mock_team=mock_team)


def _flag_overlay_for(ctx: InSeasonContext, ids: list[int]) -> pd.DataFrame:
    """The same advisory ``return_flag``/``lm_flag`` overlays the draft board shows (ADR 0021/0023,
    ADR 0024), computed for exactly ``ids`` off the rest-of-season frame rather than a full board.

    ``proj_gp``/``proj_fppg`` are the columns those overlay functions expect; here they are filled from
    the rest-of-season frame (``ros_games``, ``ros_fppg``) so the advisory games are sized against what
    is left of *this* season, not the full schedule. The cohort test itself (last season's absence
    pattern) is untouched. Never raises: a missing/short history degrades to an empty overlay (no
    flags), same as the board's own overlays do.
    """
    cols = ["player_id", "lm_flag", "lm_text", "return_flag", "return_text"]
    idx = ctx.ros.drop_duplicates("player_id").set_index("player_id")
    keep = [i for i in ids if i in idx.index]
    if not keep:
        return pd.DataFrame(columns=cols)
    sub = idx.loc[keep]
    board = pd.DataFrame({"player_id": keep, "proj_gp": sub["ros_games"].to_numpy(float),
                          "proj_fppg": sub["ros_fppg"].to_numpy(float)})
    try:
        from src.contracts import History
        from src.value.load_flag import compute_load_flag
        from src.value.return_flag import compute_return_flag

        history = History.until(ctx.tables, ctx.season)
        season_games = float(sub["team_games_left"].median()) if "team_games_left" in sub.columns else 82.0
        lf, _ = compute_load_flag(board, history, season_games=season_games)
        rf, _ = compute_return_flag(board, history, season_games=season_games)
    except (KeyError, ValueError):
        return pd.DataFrame({"player_id": keep, "lm_flag": "", "lm_text": "", "return_flag": "", "return_text": ""})
    out = board[["player_id"]].copy()
    lfi, rfi = lf.set_index("player_id"), rf.set_index("player_id")
    out["lm_flag"] = lfi["lm_flag"].reindex(keep).fillna("").to_numpy()
    out["lm_text"] = lfi["lm_text"].reindex(keep).fillna("").to_numpy()
    out["return_flag"] = rfi["return_flag"].reindex(keep).fillna("").to_numpy()
    out["return_text"] = rfi["return_text"].reindex(keep).fillna("").to_numpy()
    return out


def trade_caveats(ctx: InSeasonContext, ids: list[int]) -> list[str]:
    """Per-player advisory caveats for the traded players named in ``ids``: current ESPN injury status
    (``SUSPENDED``/``OUT``/``DAY_TO_DAY``/...) and the same advisory return-from-absence / short-absences
    flags the draft board shows. Every line is explicit that it is advisory, never a projection change
    (the same framing ``load_flag.py``/``return_flag.py`` use) and the list is empty, never an error, when
    the underlying data is missing (NaN-safe by construction: only non-empty/non-NaN flags are appended)."""
    if not ids:
        return []
    names = ctx.ros.drop_duplicates("player_id").set_index("player_id")["name"]
    out: list[str] = []
    if ctx.injuries is not None:
        for pid in ids:
            status = ctx.injuries.get(pid) if hasattr(ctx.injuries, "get") else None
            if isinstance(status, str) and status and status != "ACTIVE":
                out.append(f"{names.get(pid, pid)}: ESPN status {status} (advisory, current at load time; "
                           "not reflected in the rest-of-season projection)")
    overlay = _flag_overlay_for(ctx, ids)
    if len(overlay):
        ov = overlay.set_index("player_id")
        for pid in ids:
            if pid not in ov.index:
                continue
            row = ov.loc[pid]
            if row.get("return_text"):
                out.append(f"{names.get(pid, pid)}: {row['return_text']}")
            if row.get("lm_text"):
                out.append(f"{names.get(pid, pid)}: {row['lm_text']}")
    return out


def run_trade(ctx: InSeasonContext, choice: RosterChoice, give_ids: list[int], get_ids: list[int], *,
             partner: str | None = None) -> dict:
    """Trade analysis as display-ready pieces: ``text`` (the CLI's wording), ``delta``, ``verdict``,
    a display-ready ``confidence`` bucket/label (see ``src.inseason.trade.confidence_bucket``), ``notes``
    (context-level caveats plus per-player advisory flags on the traded players), ``lines`` and, when
    ``partner`` names a known other roster, a mirrored ``partner`` dict scoring their side of the same
    trade (as the CLI's ``--partner`` does)."""
    out = analyze(ctx, choice, give_ids, get_ids, partner=partner)
    res = out["mine"]
    names = ctx.ros.drop_duplicates("player_id").set_index("player_id")["name"]
    ros_idx = ctx.ros.drop_duplicates("player_id").set_index("player_id")
    lines = pd.DataFrame({
        "side": ["give"] * len(give_ids) + ["get"] * len(get_ids),
        "player": [names.get(p, p) for p in give_ids + get_ids],
        "ros_total_fp": [float(ros_idx.at[p, "ros_total_fp"]) for p in give_ids + get_ids]})
    bucket = confidence_bucket(res.delta, res.tolerance)
    result = {"text": describe(ctx, res, "You"), "delta": res.delta, "verdict": res.verdict, "before": res.before.value,
              "after": res.after.value, "tolerance": res.tolerance, "lines": lines,
              "dropped": [names.get(p, p) for p in res.dropped], "empty_slots": res.after.empty_slots,
              "bench_weight": out["bench_weight"], "confidence_bucket": bucket, "confidence_label": confidence_label(bucket),
              "notes": list(ctx.notes) + trade_caveats(ctx, list(give_ids) + list(get_ids)), "partner": None,
              "partner_label": out["partner_label"]}
    if out["partner"] is not None:
        pres = out["partner"]
        pbucket = confidence_bucket(pres.delta, pres.tolerance)
        result["partner"] = {"text": describe(ctx, pres, out["partner_label"]), "delta": pres.delta, "verdict": pres.verdict,
                             "before": pres.before.value, "after": pres.after.value, "tolerance": pres.tolerance,
                             "dropped": [names.get(p, p) for p in pres.dropped], "empty_slots": pres.after.empty_slots,
                             "confidence_bucket": pbucket, "confidence_label": confidence_label(pbucket)}
    return result


def run_waivers(ctx: InSeasonContext, choice: RosterChoice, *, top: int = 20, next_week: bool = False) -> WaiverReport:
    return find_waivers(ctx, choice, top=top, next_week=next_week)


def free_agent_count(ctx: InSeasonContext, choice: RosterChoice) -> int:
    return len(free_agent_ids(ctx, choice.rostered))


def bench_weight(ctx: InSeasonContext) -> float:
    from src.value.replacement import league_shape

    return bench_weight_for(ctx.ros, league_shape(ctx.cfg))


def resolve_player_names(ctx: InSeasonContext, text: str) -> list[int]:
    return resolve_names(ctx.ros, text)
