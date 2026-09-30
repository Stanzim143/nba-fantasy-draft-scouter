"""Draft-day risk overlay: preseason absence, ESPN injury status and roster-context changes (ADR 0016, gap 4).

The overlay is a set of **flags and an advisory games-at-risk estimate** beside the projection; it does not change
``proj_gp`` or any validated number. What is validated and what is not:

* ``preseason_flags``: a rotation veteran on a roster whose team played preseason games and who played none (or few) of
  them. **Unvalidated**: 11 preseasons of box scores exist, but the history has no roster snapshots, so it cannot separate
  an injured or resting player from one who is not under contract (``src.backtest.preseason_availability`` shows the
  confound: 78% of "absent" veterans played no game at all). ``PRESEASON_HAIRCUT`` is therefore 0: the flag warns, it
  does not discount. Live, it is shown only for players on the current roster snapshot.
* ``status_flags``: ESPN's current ``injuryStatus``. Not backtestable (ESPN keeps no history; the NBA's own injury-report
  PDFs do not exist before opening week, so there is no preseason archive anywhere). The haircut per status is an
  explicit **assumption** (:data:`ASSUMED_STATUS_HAIRCUT`), shown as such, never fed into a projection.
* ``context_flags``: a new team, and stars who arrived at or left his team since last season, from the roster snapshot
  against last season's final team. ADR 0010 and 0011 measured roster and transaction context as a projection input and
  found no lift, so this is descriptive: it tells you what changed, not what it is worth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.offseason import PRESEASON

MIN_TEAM_GAMES = 3            # a team must have played this many preseason games for absence to mean anything
ROTATION_MPG = 15.0           # "rotation veteran": projected minutes at or above this
PARTIAL_SHARE = 0.5           # played fewer than this share of his team's preseason games = partial
STALE_NEWS_DAYS = 45          # an ESPN status whose last news is older than this is shown but not counted as active risk
STAR_RANK = 60                # "star" for arrival/departure flags: board rank at or better than this

# Advisory haircuts (share of the projected games), NOT estimates from data. They exist so a board column can show
# "games at risk" at all; the walk-forward evidence for preseason absence is in ``PRESEASON_HAIRCUT``.
ASSUMED_STATUS_HAIRCUT = {"OUT": 0.20, "DAY_TO_DAY": 0.03, "SUSPENDED": 0.10}

# Games-share haircut for a rotation veteran who played none of his team's preseason games. Set from the walk-forward
# evaluation (``python -m src.backtest.preseason_availability``); 0.0 because that evaluation could not isolate an effect.
PRESEASON_HAIRCUT: dict[str, float] = {"dnp_all": 0.0, "partial": 0.0}


def preseason_usage(logs: pd.DataFrame, team_games: pd.DataFrame, season: str, *, as_of=None) -> tuple[pd.Series, pd.Series]:
    """(games played by player, games played by team) in ``season``'s preseason, optionally only through ``as_of``.

    ``logs`` / ``team_games`` are the offseason tables. The team series is indexed by ``team_id``.
    """
    lg = logs[(logs["event_season"] == season) & (logs["context"] == PRESEASON)]
    tg = team_games[(team_games["event_season"] == season) & (team_games["context"] == PRESEASON)]
    if as_of is not None:
        lg = lg[lg["game_date"] <= pd.Timestamp(as_of)]
        tg = tg[tg["game_date"] <= pd.Timestamp(as_of)]
    return lg.groupby("player_id").size(), tg.groupby("team_id").size()


def preseason_flags(player_ids, team_of: pd.Series, proj_mpg: np.ndarray, n_hist: np.ndarray, player_games: pd.Series,
                    team_games: pd.Series) -> pd.DataFrame:
    """Per player: ``pre_gp``, ``team_pre_games``, ``pre_share`` and ``pre_flag`` in ``{"", "dnp_all", "partial"}``.

    Only veterans (``n_hist >= 1``) projected at ``ROTATION_MPG`` or more are ever flagged. ``team_of`` maps player_id to
    the team he is on (live) or was on last season (backtest); a player who appeared for any team counts as appearing.
    """
    pids = np.asarray(player_ids, dtype="int64")
    g = player_games.reindex(pids).fillna(0).to_numpy(float)
    tid = team_of.reindex(pids).to_numpy()
    n_t = team_games.reindex(tid).to_numpy(float)
    share = np.where(n_t > 0, g / np.where(n_t > 0, n_t, 1.0), np.nan)
    eligible = (np.asarray(n_hist) >= 1) & (np.asarray(proj_mpg, float) >= ROTATION_MPG) & (n_t >= MIN_TEAM_GAMES)
    flag = np.where(eligible & (g == 0), "dnp_all", np.where(eligible & (share < PARTIAL_SHARE), "partial", ""))
    return pd.DataFrame({"player_id": pids, "pre_gp": g, "team_pre_games": n_t, "pre_share": share, "pre_flag": flag})


def status_flags(board_ids, status: pd.DataFrame | None, *, today) -> pd.DataFrame:
    """ESPN ``injury_status`` per player, with the age of the last news and whether it is stale (see ``STALE_NEWS_DAYS``)."""
    ids = np.asarray(board_ids, dtype="int64")
    out = pd.DataFrame({"player_id": ids, "espn_status": "", "status_news_days": np.nan})
    if status is None or status.empty:
        return out
    known = status.dropna(subset=["player_id"]).drop_duplicates("player_id")
    s = known.set_index(known["player_id"].astype("int64"))
    st = s["injury_status"].reindex(ids)
    out["espn_status"] = st.fillna("").to_numpy()
    news = pd.to_datetime(s["last_news_date"]).reindex(ids)
    out["status_news_days"] = ((pd.Timestamp(today) - news).dt.days).to_numpy(dtype="float64")
    return out


@dataclass
class ContextResult:
    flags: pd.DataFrame       # one row per rostered player
    star_moves: pd.DataFrame  # the stars who changed team: player_id, from_team, to_team


def context_flags(roster: pd.DataFrame, last_team: pd.Series, star_ids: set[int], names: pd.Series) -> ContextResult:
    """New team and star arrival / departure per player.

    ``roster``: newest ``roster_snapshots`` day (``player_id, team_id, team_abbr``); ``last_team``: player_id -> team_id at
    the end of last season; ``star_ids``: players counted as stars (board rank within ``STAR_RANK``); ``names``: id -> name.
    A star **arrival** at a team is a star on the roster now whose last team was different; a **departure** is a star
    whose last team was this one and who is now on another roster. Players who left the rosters entirely (retired,
    unsigned free agents) are not visible in a roster snapshot and are not counted.
    """
    r = roster.drop_duplicates("player_id").set_index("player_id")
    now = r["team_id"]
    abbr = r.drop_duplicates("team_id").set_index("team_id")["team_abbr"]
    prior = last_team.reindex(r.index)
    moved = prior.notna() & (prior != now)
    arr = pd.DataFrame({"player_id": r.index[moved & r.index.isin(star_ids)]})
    arr["to_team"] = now.reindex(arr["player_id"]).to_numpy()
    arr["from_team"] = prior.reindex(arr["player_id"]).to_numpy()
    flags = pd.DataFrame({"player_id": r.index.to_numpy(), "team": r["team_abbr"].to_numpy(),
                          "new_team": moved.to_numpy(),
                          "from_team": prior.map(abbr).to_numpy()})
    star_in = arr.groupby("to_team")["player_id"].apply(list) if len(arr) else pd.Series(dtype=object)
    star_out = arr.groupby("from_team")["player_id"].apply(list) if len(arr) else pd.Series(dtype=object)

    def label(ids, pid, side):
        others = [names.get(i, str(i)) for i in ids if i != pid]
        return (f"star {side}: " + ", ".join(others)) if others else ""

    arrivals, departures = [], []
    for pid, tid in zip(r.index, now):
        a = star_in.get(tid, [])
        d = star_out.get(tid, [])
        arrivals.append(label(a, pid, "arrived"))
        departures.append(label(d, pid, "left"))
    flags["star_arrivals"], flags["star_departures"] = arrivals, departures
    return ContextResult(flags, arr)


def build_overlay(board: pd.DataFrame, *, preseason: pd.DataFrame | None = None, status: pd.DataFrame | None = None,
                  context: pd.DataFrame | None = None, today=None) -> pd.DataFrame:
    """``board`` plus ``risk_flags`` (readable), ``risk_level`` (``""`` / ``watch`` / ``high``), ``risk_gp_haircut``
    (advisory share of projected games) and ``risk_gp`` (``proj_gp`` after that haircut). Projections are untouched."""
    today = today if today is not None else pd.Timestamp.today().normalize()
    out = board.copy()
    ids = out["player_id"].to_numpy("int64")
    flags: list[list[str]] = [[] for _ in ids]
    haircut = np.zeros(len(ids))
    level = np.zeros(len(ids), dtype=int)
    if preseason is not None and len(preseason):
        p = preseason.set_index("player_id").reindex(ids)
        for i, f in enumerate(p["pre_flag"].fillna("").to_numpy()):
            if f == "dnp_all":
                flags[i].append(f"played 0 of {int(p['team_pre_games'].iloc[i])} preseason games")
                haircut[i] = max(haircut[i], PRESEASON_HAIRCUT["dnp_all"])
                level[i] = max(level[i], 1)
            elif f == "partial":
                flags[i].append(f"played {int(p['pre_gp'].iloc[i])} of {int(p['team_pre_games'].iloc[i])} preseason games")
                haircut[i] = max(haircut[i], PRESEASON_HAIRCUT["partial"])
                level[i] = max(level[i], 1)
    if status is not None and len(status):
        s = status.set_index("player_id").reindex(ids)
        for i, (st, days) in enumerate(zip(s["espn_status"].fillna("").to_numpy(), s["status_news_days"].to_numpy(float))):
            if st in ("OUT", "DAY_TO_DAY", "SUSPENDED"):
                stale = np.isfinite(days) and days > STALE_NEWS_DAYS
                flags[i].append(f"ESPN {st.replace('_', '-').lower()}" + (f" (last news {int(days)}d ago)" if np.isfinite(days) else "")
                                + (", stale" if stale else ""))
                if not stale:
                    haircut[i] = max(haircut[i], ASSUMED_STATUS_HAIRCUT[st])
                    level[i] = max(level[i], 2 if st == "OUT" else 1)
    if context is not None and len(context):
        c = context.set_index("player_id").reindex(ids)
        for i, (nt, fr, sa, sd) in enumerate(zip(c["new_team"].fillna(False).to_numpy(bool), c["from_team"].to_numpy(),
                                                 c["star_arrivals"].fillna("").to_numpy(), c["star_departures"].fillna("").to_numpy())):
            if nt:
                flags[i].append(f"new team (from {fr})" if isinstance(fr, str) else "new team")
            if sa:
                flags[i].append(sa)
            if sd:
                flags[i].append(sd)
    out["risk_flags"] = ["; ".join(f) for f in flags]
    out["risk_level"] = np.array(["", "watch", "high"])[level]
    out["risk_gp_haircut"] = haircut
    out["risk_gp"] = out["proj_gp"].to_numpy(float) * (1.0 - haircut) if "proj_gp" in out.columns else np.nan
    return out
