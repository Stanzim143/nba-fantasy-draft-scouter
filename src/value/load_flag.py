"""Advisory "many short absences last season" flag for a board (ADR 0024). Projections and ranks never change.

    from src.value.load_flag import compute_load_flag, attach_load_flag
    overlay, notes = compute_load_flag(board, history, season_games=82.0)
    board = attach_load_flag(board, overlay)

**This is a descriptive proxy, not a load-management detector.** No data says why a game was missed (ADR 0024), so the flag cannot
tell rest from a minor injury. What ADR 0024 measured on ten walk-forward seasons: a rotation veteran (single team, >= 20 mpg and
>= 41 GP the season before) with :data:`FLAG_MIN_ISO` (6) or more games missed in short (<= 2 game) interior absences lands further
below the baseline projection than a similar player with fewer: -2.6 GP [-5.1, -0.2], -98 total FP [-183, -13] (n = 310 of 1,830), and
the association has the same sign in 9 of 10 seasons. It did not survive the pre-registered out-of-sample test (correcting the projection
with it did not lower MAE), and the back-to-back one-game "rest" proxy on its own showed nothing. So the flag is advisory, sized by a
fixed rule written down before it was computed: :data:`ADVISORY_GP` (2) games *fewer* for a flagged player, never below zero, in FP
``advisory_gp * proj_fppg``. The board's ``proj_gp``, ``proj_total_fp``, ``vorp``, ``rank``, ``risk_level`` and ``risk_gp`` are read
for nothing except sizing the advisory games and are never written.

Columns added by :func:`attach_load_flag` (blank / NaN / 0 for everyone not flagged):

``lm_flag``          ``short-absences`` | ``''``
``lm_iso_n``         games missed last season in interior runs of <= 2 games (NaN for players outside the study universe)
``lm_rest_n``        of those, one-game absences on the second night of a back-to-back (information only)
``lm_gp_risk_adv``   advisory games, 0 or -2; NOT subtracted from ``proj_gp`` anywhere
``lm_fp_risk_adv``   ``lm_gp_risk_adv * proj_fppg``, the same advisory games as total fantasy points
``lm_validation``    constant ``advisory_descriptive_not_projection`` on every row

If the board has a ``risk_flags`` column the readable sentence is appended to it; ``risk_level`` / ``risk_gp_haircut`` / ``risk_gp``
are not touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import History
from src.features.load_management import prior_load_features

LM_COLUMNS = ["lm_flag", "lm_iso_n", "lm_rest_n", "lm_gp_risk_adv", "lm_fp_risk_adv"]
VALIDATION_COLUMN = "lm_validation"
VALIDATION_LABEL = "advisory_descriptive_not_projection"
FLAG_LABEL = "short-absences"

MIN_MPG, MIN_GP = 20.0, 41       # the ADR 0024 universe
FLAG_MIN_ISO = 6                 # the top pre-registered iso_n bin
ADVISORY_GP = 2.0                # fixed rule: about the study's -2.6 GP, rounded toward zero (its CI ends at -0.2)


def load_profile(history: History) -> pd.DataFrame:
    """Last season's universe members from ``history`` alone: ``player_id, L, gp, iso_n, rest_n``."""
    cols = ["player_id", "L", "gp", "iso_n", "rest_n"]
    if history.game_logs.empty or history.team_games.empty:
        return pd.DataFrame(columns=cols)
    p = prior_load_features(history)
    if p.empty:
        return pd.DataFrame(columns=cols)
    p = p[p["veteran"].astype(bool) & (p["n_teams"] == 1) & (p["mpg"] >= MIN_MPG) & (p["gp"] >= MIN_GP)]
    return p[cols].reset_index(drop=True)


def advisory_gp_risk(iso_n: np.ndarray, proj_gp: np.ndarray) -> np.ndarray:
    """``-ADVISORY_GP`` where ``iso_n >= FLAG_MIN_ISO`` (never taking ``proj_gp`` below zero), else 0. A judgement, not a fitted number."""
    n = np.nan_to_num(np.asarray(iso_n, float), nan=0.0)
    room = np.clip(np.nan_to_num(np.asarray(proj_gp, float), nan=0.0), 0.0, None)
    return np.where(n >= FLAG_MIN_ISO, -np.minimum(ADVISORY_GP, room), 0.0)


def flag_text(iso_n: int, rest_n: int, gp: int, L: int, adv: float) -> str:
    s = (f"many short absences last season: missed {iso_n} games in 1-2 game absences (played {gp} of {L}; {rest_n} were one-game "
         f"back-to-back absences); cause unknown, rest or minor injury")
    if adv < 0:
        s += f"; advisory {adv:g} GP (a judgement, not in the projection)"
    return s


def compute_load_flag(board: pd.DataFrame, history: History, *, season_games: float = 82.0) -> tuple[pd.DataFrame, list[str]]:  # noqa: ARG001
    """The overlay (``player_id`` + :data:`LM_COLUMNS` + ``lm_text``) for every board player and notes on what was missing.

    Reads ``board`` only for ``player_id``, ``proj_gp`` and ``proj_fppg``; never fails on missing inputs.
    """
    notes: list[str] = []
    ids = board["player_id"].to_numpy("int64")
    n = len(ids)
    out = pd.DataFrame({"player_id": ids, "lm_flag": "", "lm_iso_n": np.nan, "lm_rest_n": np.nan, "lm_gp_risk_adv": 0.0,
                        "lm_fp_risk_adv": 0.0, "lm_text": ""})
    try:
        prof = load_profile(history)
    except (KeyError, ValueError) as exc:
        notes.append(f"short-absence flag unavailable ({exc}): no players flagged")
        return out, notes
    if history.game_logs.empty or history.team_games.empty:
        notes.append("short-absence flag: no game logs or team games before the target season, so no players are flagged")
        return out, notes
    p = prof.drop_duplicates("player_id").set_index("player_id").reindex(ids)
    known = p["L"].notna().to_numpy()
    iso = p["iso_n"].to_numpy(float)
    hit = known & (np.nan_to_num(iso, nan=0.0) >= FLAG_MIN_ISO)
    if "proj_gp" in board.columns and "proj_fppg" in board.columns:
        proj_gp, fppg = board["proj_gp"].to_numpy(float), board["proj_fppg"].to_numpy(float)
        adv = np.where(hit, advisory_gp_risk(iso, proj_gp), 0.0)
    else:
        fppg, adv = np.zeros(n), np.zeros(n)
        notes.append("short-absence flag: the board has no proj_gp / proj_fppg, so the advisory games are not sized")
    out["lm_flag"] = np.where(hit, FLAG_LABEL, "")
    out["lm_iso_n"] = np.where(known, iso, np.nan)
    out["lm_rest_n"] = np.where(known, p["rest_n"].to_numpy(float), np.nan)
    out["lm_gp_risk_adv"] = adv
    out["lm_fp_risk_adv"] = np.where(np.isfinite(fppg), adv * np.nan_to_num(fppg), 0.0)
    gp_i, L_i = p["gp"].fillna(0).to_numpy(int), p["L"].fillna(0).to_numpy(int)
    iso_i, rest_i = np.nan_to_num(iso).astype(int), p["rest_n"].fillna(0).to_numpy(int)
    out["lm_text"] = [flag_text(a, r, g, ln, u) if h else "" for h, a, r, g, ln, u in zip(hit, iso_i, rest_i, gp_i, L_i, adv)]
    notes.append(f"short-absence flag (advisory, descriptive, cause unknown; not a projection): {int(hit.sum())} of {n} board players missed "
                 f"{FLAG_MIN_ISO}+ games in 1-2 game absences last season (ADR 0024); {int((adv < 0).sum())} get the advisory -{ADVISORY_GP:g} GP")
    return out, notes


def attach_load_flag(board: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """``board`` with the overlay columns added (row order and every existing column unchanged except ``risk_flags``, which gains the
    readable text when the board has one). ``risk_level``, ``risk_gp_haircut`` and ``risk_gp`` are never modified."""
    o = overlay.drop_duplicates("player_id").set_index("player_id").reindex(board["player_id"].to_numpy("int64"))
    out = board.copy()
    out["lm_flag"] = o["lm_flag"].fillna("").to_numpy()
    out["lm_iso_n"] = o["lm_iso_n"].to_numpy(float)
    out["lm_rest_n"] = o["lm_rest_n"].to_numpy(float)
    out["lm_gp_risk_adv"] = o["lm_gp_risk_adv"].fillna(0.0).to_numpy(float)
    out["lm_fp_risk_adv"] = o["lm_fp_risk_adv"].fillna(0.0).to_numpy(float)
    out[VALIDATION_COLUMN] = VALIDATION_LABEL
    if "risk_flags" in out.columns:
        text = o["lm_text"].fillna("").to_numpy(object)
        cur = out["risk_flags"].fillna("").to_numpy(object)
        out["risk_flags"] = [c if not t else (f"{c}; {t}" if c else t) for c, t in zip(cur, text)]
    out.attrs = dict(board.attrs)
    return out
