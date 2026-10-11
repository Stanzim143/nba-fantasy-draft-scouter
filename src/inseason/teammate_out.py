"""Teammate-out adjustment: the per-game FP a player is expected to gain because rotation teammates are out.

Builds on ``signals`` (ADR 0015 D5): who counts as an absent rotation player and how his minutes are shared out
(``beneficiaries``) are unchanged. This module turns those minute gains into a per-game FP adjustment on top of the ROS
per-game projection, and adds an absence source driven by an injury status (ESPN ``OUT`` live, the NBA 5 PM injury
report in the evaluation). Evidence and verdict: ``docs/research/teammate-out-study.md``.

Leakage: every function reads only the game logs and team games it is given; callers pass rows dated on or before the
as-of day (the day before the game). A status is a pre-game snapshot the caller vouches for.
``tests/inseason/test_teammate_out.py`` perturbs everything dated on or after the target date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inseason.signals import MIN_BASE_GAMES, MINUTES_CAP, ROTATION_MPG, _chronological, beneficiaries

ADJ_COLS = ["player_id", "team_id", "gain_mpg", "adj_fppg", "out_players"]


def status_absences(gl: pd.DataFrame, statuses: pd.Series, *, min_mpg: float = ROTATION_MPG,
                    min_games: int = MIN_BASE_GAMES) -> pd.DataFrame:
    """Rotation players (season-to-date ``min_mpg``+ over ``min_games``+ games in ``gl``) whose status is ``OUT``.

    Same columns as ``signals.absent_players``; ``streak`` is unknown here (0)."""
    cols = ["player_id", "team_id", "mpg", "gp", "streak", "source", "long_term"]
    if gl.empty or statuses is None or not len(statuses):
        return pd.DataFrame(columns=cols)
    gl = _chronological(gl)
    per = gl.groupby("player_id").agg(mpg=("min", "mean"), gp=("game_id", "size"), team_id=("team_id", "last"))
    rot = per[(per["mpg"] >= min_mpg) & (per["gp"] >= min_games)]
    out_ids = [int(p) for p, s in statuses.items() if str(s).upper() == "OUT" and p in rot.index]
    r = rot.loc[out_ids]
    return pd.DataFrame({"player_id": out_ids, "team_id": r["team_id"].astype("int64").to_numpy(),
                         "mpg": r["mpg"].to_numpy(float), "gp": r["gp"].astype(int).to_numpy(), "streak": 0,
                         "source": "status OUT", "long_term": False}, columns=cols)


def share_present(gl: pd.DataFrame, tg: pd.DataFrame, pid: int, team_id: int) -> float:
    """Share of ``team_id``'s games since ``pid`` first played for it in which he played (this season, to date)."""
    mine = gl[(gl["player_id"] == pid) & (gl["team_id"] == team_id)]
    if mine.empty:
        return 0.0
    first = pd.Timestamp(mine["game_date"].min())
    n_team = int((tg["team_id"].eq(team_id) & (pd.to_datetime(tg["game_date"]) >= first)).sum())
    return float(min(1.0, mine["game_id"].nunique() / n_team)) if n_team else 0.0


def teammate_out_adjustment(gl: pd.DataFrame, tg: pd.DataFrame, absent: pd.DataFrame, positions: pd.Series,
                            ros: pd.DataFrame, *, beta: float = 1.0) -> pd.DataFrame:
    """Per-player FP-per-game adjustment for the absent teammates in ``absent``.

    ``gl``/``tg``: this season's logs and team games up to the as-of day. ``ros``: ``build_ros`` output (``player_id``,
    ``ros_fppg``, ``ros_mpg``). For each beneficiary the rule's minute gain from each absentee is scaled by the share of
    games that absentee has played (the season average already holds the games he missed), summed, capped at
    ``MINUTES_CAP`` total minutes, converted with the ROS FP per minute and multiplied by ``beta``.
    """
    if absent is None or absent.empty or gl.empty:
        return pd.DataFrame(columns=ADJ_COLS)
    ben = beneficiaries(gl, tg, absent, positions)
    if ben.empty:
        return pd.DataFrame(columns=ADJ_COLS)
    present = {(int(a), int(t)): share_present(gl, tg, int(a), int(t))
               for a, t in ben[["out_player_id", "team_id"]].drop_duplicates().itertuples(index=False)}
    ben = ben.assign(gain=[g * present[(int(a), int(t))] for g, a, t in
                           ben[["gain_mpg", "out_player_id", "team_id"]].itertuples(index=False)])
    agg = ben.groupby("player_id").agg(team_id=("team_id", "first"), gain_mpg=("gain", "sum"),
                                       out_players=("out_player_id", lambda s: tuple(sorted(int(x) for x in s))))
    r = ros.drop_duplicates("player_id").set_index("player_id")
    mpg = r["ros_mpg"].reindex(agg.index).astype(float)
    fppg = r["ros_fppg"].reindex(agg.index).astype(float)
    room = (MINUTES_CAP - mpg).clip(lower=0.0).fillna(0.0)
    agg["gain_mpg"] = np.minimum(agg["gain_mpg"], room)
    with np.errstate(divide="ignore", invalid="ignore"):
        fppm = np.where(mpg > 0, fppg / mpg, np.nan)
    agg["adj_fppg"] = beta * agg["gain_mpg"] * np.nan_to_num(fppm, nan=0.0)
    agg = agg[agg["adj_fppg"] > 0]
    return agg.reset_index()[ADJ_COLS]


# Fitted on the 2018-19..2019-20 NBA 5 PM injury reports (status OUT), evaluated on 2020-21..2025-26 (ADR 0042).
BETA_STATUS = 0.727
MAX_STATUS_AGE = pd.Timedelta(days=3)


def snapshot_is_fresh(injuries: pd.Series | None, as_of) -> bool:
    """True when the status snapshot was fetched within ``MAX_STATUS_AGE`` of ``as_of`` (either side). The fetch time is
    ``injuries.attrs["fetched_at"]`` (``context.load_injuries`` sets it from the cache file); unknown age is not fresh."""
    if injuries is None or as_of is None:
        return False
    fetched = injuries.attrs.get("fetched_at")
    if fetched is None or pd.isna(fetched):
        return False
    return abs(pd.Timestamp(fetched) - pd.Timestamp(as_of)) <= MAX_STATUS_AGE


def status_boost(gl: pd.DataFrame, tg: pd.DataFrame, injuries: pd.Series | None, positions: pd.Series,
                 ros: pd.DataFrame, *, as_of, beta: float = BETA_STATUS) -> pd.Series:
    """player_id -> FP to add to a player's **next game** because rotation teammates are ``OUT``.

    Next game only: the week horizon did not meet the pre-registered rule (ADR 0042). Status-driven absences only: the
    game-log-only variant failed and stays an advisory flag (``signals.beneficiaries``). Empty without a status snapshot
    fetched within 3 days of ``as_of`` (a stale or undated snapshot would boost teammates of players already back)."""
    if injuries is None or not len(injuries) or gl.empty or not snapshot_is_fresh(injuries, as_of):
        return pd.Series(dtype=float)
    adj = teammate_out_adjustment(gl, tg, status_absences(gl, injuries), positions, ros, beta=beta)
    return adj.set_index("player_id")["adj_fppg"] if len(adj) else pd.Series(dtype=float)
