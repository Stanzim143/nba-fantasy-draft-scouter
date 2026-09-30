"""Who is a "debutant" and what do we know about him before the season (ADR 0016).

A *debutant* is a player who is in the NBA picture for season ``S`` but has no NBA game log before ``S`` and is not
a member of ``S``'s own draft class (that class has its draft-slot prior in ``src.models.rookies``):

* ``stash``: drafted in an earlier year (with a real pick) and only now arriving: overseas stashes (Marković, Toohey,
  Biberovic, Diop) and players who lost their rookie year to injury (Sorber).
* ``undrafted``: never drafted (undrafted signees, two-way and Exhibit-10 players) or unknown to the draft records.

Two sources of "is in the picture", one function:

* live: the newest ``roster_snapshots`` day (who is under contract now);
* backtest: the players who appeared in the season's **preseason** games (the camp roster). A backtest has no
  historical roster snapshots, and camp participation is the honest point-in-time analogue: it is public before the
  regular season and it includes the same Exhibit-10 and two-way players.

Leakage rules. Draft year and pick, country, size and birthdate are facts fixed before the season, so they are used
as recorded. What is **not** used: whether a person has an index/profile row at all, and ``from_year >= S`` (both
reveal a future debut). ``from_year`` before the game-log window is used, only to *exclude*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.models.rookies import UNDRAFTED_PICK, pick_effective, to_float
from src.value.positions import position_group

STASH = "stash"
UNDRAFTED = "undrafted"
CLASSES = (STASH, UNDRAFTED)

EVENT_COLUMNS = ["sl_gp", "sl_minutes", "sl_mpg", "sl_z", "pre_gp", "pre_minutes", "pre_mpg", "pre_gp_share", "pre_z"]


def age_on_oct1(birthdate: pd.Series, start_year: int) -> np.ndarray:
    bd = pd.to_datetime(birthdate, errors="coerce")
    return ((pd.Timestamp(f"{start_year}-10-01") - bd).dt.days / 365.25).to_numpy(dtype="float64")


def camp_ids(events: pd.DataFrame, season: str) -> np.ndarray:
    """Players with at least one preseason game of ``season`` (``event_player_table`` output)."""
    e = events[events["event_season"] == season]
    return e.loc[e["pre_gp"].fillna(0) > 0, "player_id"].to_numpy("int64")


def find_candidates(target_season: str, *, prior_player_ids, profiles: pd.DataFrame, source_ids,
                    names: pd.Series | None = None, players: pd.DataFrame | None = None,
                    events: pd.DataFrame | None = None, window_start: int | None = None) -> pd.DataFrame:
    """Debutant candidates among ``source_ids`` (roster snapshot ids, or the preseason camp for a backtest).

    ``prior_player_ids``: everyone with an NBA game log before ``target_season`` (never a debutant).
    ``profiles``: ``player_profiles``; ``players``: a ``players`` table used only to fill missing birthdates;
    ``events``: ``event_player_table`` rows of ``target_season`` (evidence columns are merged in when given).
    ``window_start``: first season (start year) of the game-log history; a recorded ``from_year`` before it means he
    played before the logs begin, so he is not a debutant. (``from_year`` is *not* used otherwise: it is the first
    season on a roster, which for a stash who spent a year injured is already the season before his first game.)
    Returns one row per candidate with class, slot, age, size, origin and evidence columns.
    """
    start = season_start(target_season)
    ids = pd.unique(np.asarray(list(source_ids), dtype="int64"))
    ids = np.setdiff1d(ids, np.asarray(list(prior_player_ids), dtype="int64"))
    prof = profiles.drop_duplicates("player_id").set_index("player_id")
    df = pd.DataFrame({"player_id": ids})
    for col in ("player_name", "birthdate", "country", "origin", "prev_org_type", "position", "height_in",
                "weight_lb", "draft_year", "draft_round", "draft_number", "from_year"):
        df[col] = prof[col].reindex(ids).to_numpy() if col in prof.columns else np.nan
    if players is not None and len(players):
        pl = players.drop_duplicates("player_id").set_index("player_id")
        df["birthdate"] = pd.to_datetime(df["birthdate"]).where(df["birthdate"].notna(),
                                                                pd.to_datetime(pl["birthdate"].reindex(ids).to_numpy()))
    if names is not None:
        nm = names.reindex(ids).to_numpy()
        df["player_name"] = np.where(pd.isna(df["player_name"]), nm, df["player_name"])
    fy, dy = to_float(df["from_year"]), to_float(df["draft_year"])
    old = np.isfinite(fy) & (fy < (window_start if window_start is not None else start))
    df = df[~old & ~(np.isfinite(dy) & (dy == start))].reset_index(drop=True)
    fy, dy = to_float(df["from_year"]), to_float(df["draft_year"])
    pick = to_float(df["draft_number"])
    real_pick = np.isfinite(dy) & (dy < start) & np.isfinite(pick) & (pick >= 1)
    df["klass"] = np.where(real_pick, STASH, UNDRAFTED)
    df["pick"] = np.where(real_pick, pick_effective(df["draft_number"], df["draft_round"]), UNDRAFTED_PICK)
    df["years_since_draft"] = np.where(real_pick, start - dy, np.nan)
    df["age"] = age_on_oct1(df["birthdate"], start)
    df["group"] = [position_group(p) for p in df["position"]]
    df["intl"] = (df["origin"] == "intl").astype(float)
    df["intl_known"] = df["origin"].notna()
    ev = events[events["event_season"] == target_season] if events is not None and len(events) else None
    for c in EVENT_COLUMNS:
        df[c] = (ev.drop_duplicates("player_id").set_index("player_id")[c].reindex(df["player_id"]).to_numpy()
                 if ev is not None and c in ev.columns else np.nan)
    return df.reset_index(drop=True)
