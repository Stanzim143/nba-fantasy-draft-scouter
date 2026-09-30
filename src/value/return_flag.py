"""Advisory "returned from a long absence, healthy since" flag for a board (ADR 0023). Projections and ranks never change.

    from src.value.return_flag import compute_return_flag, attach_return_flag
    overlay, notes = compute_return_flag(board, history, season_games=82.0)
    board = attach_return_flag(board, overlay)

The cohort is exactly ADR 0021's primary cohort, computed by that module's own point-in-time functions
(:func:`src.backtest.return_report.season_profiles` / :func:`~src.backtest.return_report.in_cohort`; nothing is re-implemented here)
on last season only, from a ``History`` that stops before the target season: a veteran on a single team, at least 15 mpg, who
missed the first 25% or more of his team's games (the *lead block*), then had at least 15 team games left and played at least 75% of
them (the *healthy tail*). Jayson Tatum 2025-26: out for the first 62 of 82 team games, then 16 of the last 20.

**This is a display flag and an advisory judgement, not a projection.** ADR 0021 found no significant absolute under-projection for the
cohort (+1.0 GP, 95% CI -4.1 to +5.8) and recommended that any flag be advisory, "sized by judgement, not derived", at most about +5 GP.
ADR 0022 built the availability feature and it over-corrected, so the model is untouched. The advisory upside is therefore a fixed rule
written down before the numbers were looked at (:func:`advisory_gp_upside`): 0 games for a lead block below 60% of the schedule
(:data:`UPSIDE_MIN_BLOCK`), :data:`UPSIDE_GP` (5) games for a block of 60% or more, never past the schedule length. The board's
``proj_gp``, ``proj_total_fp``, ``vorp``, ``rank``, ``risk_level`` and ``risk_gp`` are read for nothing except sizing the advisory games (``proj_gp``, the schedule cap) and converting them
into FP (``upside_gp * proj_fppg``), and are never written.

Columns added by :func:`attach_return_flag` (blank / NaN for everyone not in the cohort):

``return_flag``           ``returned-healthy`` | ``''``
``return_block_pct``      lead block as a percent of his team's games (0 to 100)
``return_tail``           games played over team games since his first appearance, e.g. ``16/20``
``return_gp_upside_adv``  advisory games, 0 or 5 (see above); NOT added to ``proj_gp`` anywhere
``return_fp_upside_adv``  ``return_gp_upside_adv * proj_fppg``: the same advisory games expressed in total fantasy points
``return_validation``     constant ``advisory_judgement_not_projection`` on every row, so a CSV consumer (no console) sees it

If the board has a ``risk_flags`` column (the draft-day overlay, ADR 0016) the readable flag string is appended to it so the existing
UI surfaces it; ``risk_level`` / ``risk_gp_haircut`` / ``risk_gp`` are not touched.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.backtest.return_report import PRIMARY, in_cohort, season_profiles, veteran_flags
from src.contracts import History, season_start

RETURN_COLUMNS = ["return_flag", "return_block_pct", "return_tail", "return_gp_upside_adv", "return_fp_upside_adv"]
VALIDATION_COLUMN = "return_validation"
VALIDATION_LABEL = "advisory_judgement_not_projection"
FLAG_LABEL = "returned-healthy"

UPSIDE_MIN_BLOCK = 0.60     # advisory games only for a lead block of at least this share of the schedule (ADR 0021: the >= 60% cell)
UPSIDE_GP = 5.0             # ... and never more than this (ADR 0021 recommendation: "about +5")


def advisory_gp_upside(block_share: np.ndarray, proj_gp: np.ndarray, season_games: float = 82.0) -> np.ndarray:
    """The advisory games rule: ``UPSIDE_GP`` where the lead block is >= ``UPSIDE_MIN_BLOCK`` of the schedule, else 0; capped so
    ``proj_gp + upside`` never exceeds ``season_games``. A judgement (ADR 0021 / 0023), not a fitted number."""
    share = np.asarray(block_share, float)
    room = np.clip(season_games - np.nan_to_num(np.asarray(proj_gp, float), nan=season_games), 0.0, None)   # unknown proj_gp: no advisory games
    return np.where(np.nan_to_num(share, nan=0.0) >= UPSIDE_MIN_BLOCK, np.minimum(UPSIDE_GP, room), 0.0)


def return_profile(history: History) -> pd.DataFrame:
    """Last season's cohort members from ``history`` alone: ``player_id, L, gp, first_idx, block_share, tail_len``.

    Only season ``target - 1`` is profiled (the games of every earlier season are used only for the veteran test), so this is the
    same set ``prior_features`` + ``in_cohort`` give for the primary cohort, at a fraction of the cost.
    """
    cols = ["player_id", "L", "gp", "first_idx", "block_share", "tail_len"]
    if history.game_logs.empty or history.team_games.empty:
        return pd.DataFrame(columns=cols)
    prior = season_start(history.target_season) - 1
    seasons = lambda df: df["season"].map(season_start)  # noqa: E731
    last = dataclasses.replace(history, game_logs=history.game_logs[seasons(history.game_logs) == prior],
                               team_games=history.team_games[seasons(history.team_games) == prior])
    prof = season_profiles(last)
    if prof.empty:
        return pd.DataFrame(columns=cols)
    prof = prof[prof["s"] == prior].copy()
    prof["veteran"] = veteran_flags(history, prof["player_id"].to_numpy(), prof["s"].to_numpy())
    prof["f"] = prof["gp"] / prof["L"]
    prof["season"] = history.target_season
    prof = prof[in_cohort(prof, PRIMARY)].copy()
    prof["block_share"] = prof["first_idx"] / prof["L"]
    prof["tail_len"] = prof["L"] - prof["first_idx"]
    return prof[cols].reset_index(drop=True)


def flag_text(L: int, first_idx: int, gp: int, tail_len: int, upside: float) -> str:
    s = f"returned from long absence (missed the first {first_idx} of {L} team games), healthy since: played {gp} of the last {tail_len} team games"
    if upside > 0:
        s += f"; advisory +{upside:g} GP (a judgement, not in the projection)"
    return s


def compute_return_flag(board: pd.DataFrame, history: History, *, season_games: float = 82.0) -> tuple[pd.DataFrame, list[str]]:
    """The overlay (``player_id`` + :data:`RETURN_COLUMNS` + ``return_text``) for every player of ``board`` and notes on what was missing.

    Reads ``board`` only for ``player_id``, ``proj_gp`` and ``proj_fppg`` (the FP conversion); never fails on missing inputs: an empty
    history or a board without projections gives no flags and a note.
    """
    notes: list[str] = []
    ids = board["player_id"].to_numpy("int64")
    n = len(ids)
    out = pd.DataFrame({"player_id": ids, "return_flag": "", "return_block_pct": np.nan, "return_tail": "",
                        "return_gp_upside_adv": 0.0, "return_fp_upside_adv": 0.0, "return_text": ""})
    try:
        prof = return_profile(history)
    except (KeyError, ValueError) as exc:
        notes.append(f"return flag unavailable ({exc}): no players flagged")
        return out, notes
    if history.game_logs.empty or history.team_games.empty:
        notes.append("return flag: no game logs or team games before the target season, so no players are flagged")
        return out, notes
    p = prof.drop_duplicates("player_id").set_index("player_id").reindex(ids)
    hit = p["L"].notna().to_numpy()
    if "proj_gp" in board.columns and "proj_fppg" in board.columns:
        proj_gp = board["proj_gp"].to_numpy(float)
        fppg = board["proj_fppg"].to_numpy(float)
    else:
        proj_gp = np.zeros(n)
        fppg = np.zeros(n)
        notes.append("return flag: the board has no proj_gp / proj_fppg, so the advisory games are not sized")
    share = p["block_share"].to_numpy(float)
    up = np.where(hit, advisory_gp_upside(share, proj_gp, season_games), 0.0)
    if "proj_gp" not in board.columns:
        up = np.zeros(n)
    out.loc[hit, "return_flag"] = FLAG_LABEL
    out["return_block_pct"] = np.where(hit, np.round(100.0 * share, 1), np.nan)
    gp_i = p["gp"].fillna(0).to_numpy(int)
    tail_i = p["tail_len"].fillna(0).to_numpy(int)
    out["return_tail"] = np.where(hit, [f"{a}/{b}" for a, b in zip(gp_i, tail_i)], "")
    out["return_gp_upside_adv"] = up
    out["return_fp_upside_adv"] = np.where(np.isfinite(fppg), up * np.nan_to_num(fppg), 0.0)
    L_i, f_i = p["L"].fillna(0).to_numpy(int), p["first_idx"].fillna(0).to_numpy(int)
    out["return_text"] = [flag_text(ln, f, g, t, u) if h else "" for h, ln, f, g, t, u in zip(hit, L_i, f_i, gp_i, tail_i, up)]
    notes.append(f"return flag (advisory judgement, not a projection): {int(hit.sum())} of {n} board players missed the first 25%+ of last "
                 f"season and then played 75%+ of at least 15 remaining team games (ADR 0021 cohort; ADR 0023); "
                 f"{int((up > 0).sum())} get the advisory +{UPSIDE_GP:g} GP (block >= {UPSIDE_MIN_BLOCK:.0%})")
    return out, notes


def attach_return_flag(board: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """``board`` with the overlay columns added (row order and every existing column unchanged except ``risk_flags``, which gains the
    readable text when the board has one). ``risk_level``, ``risk_gp_haircut`` and ``risk_gp`` are never modified."""
    o = overlay.drop_duplicates("player_id").set_index("player_id").reindex(board["player_id"].to_numpy("int64"))
    out = board.copy()
    out["return_flag"] = o["return_flag"].fillna("").to_numpy()
    out["return_block_pct"] = o["return_block_pct"].to_numpy(float)
    out["return_tail"] = o["return_tail"].fillna("").to_numpy()
    out["return_gp_upside_adv"] = o["return_gp_upside_adv"].fillna(0.0).to_numpy(float)
    out["return_fp_upside_adv"] = o["return_fp_upside_adv"].fillna(0.0).to_numpy(float)
    out[VALIDATION_COLUMN] = VALIDATION_LABEL
    if "risk_flags" in out.columns:
        text = o["return_text"].fillna("").to_numpy(object)
        cur = out["risk_flags"].fillna("").to_numpy(object)
        out["risk_flags"] = [c if not t else (f"{c}; {t}" if c else t) for c, t in zip(cur, text)]
    out.attrs = dict(board.attrs)
    return out
