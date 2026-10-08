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
* ``minutes_opened`` / ``depth_flags``: rotation minutes that left a team since last season (traded, signed elsewhere or off every
  roster) net of the minutes that arrived, for the players still there. Descriptive, same status as ``context_flags``.
* ``playoff_flags``: a role-player's closing playoff role that was much larger than in the regular season (the
  regular-season tables exclude playoffs by contract). Descriptive and a small sample by nature.
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
MIN_OPENED_NET = 20.0         # net team minutes per game opened (departed - arrived) before the depth flag shows
DEPART_NAME_MPG = 8.0         # a departed player is named in the flag when he averaged this many minutes per team game
PLAYOFF_MIN_GAMES = 5         # fewer playoff games than this is too thin to mention
PLAYOFF_RECENT_GAMES = 5      # the closing run: the player's last this-many playoff games
PLAYOFF_MPG_UPLIFT = 4.0      # closing playoff mpg above the regular season by at least this much -> flag
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


def news_flags(board_ids, news: pd.DataFrame | None) -> pd.DataFrame:
    """``news_flag`` text per player from the live ESPN injury feed (``espn_news.latest_per_player``): the injury and
    estimated return date. Informational only: no history exists to back-test a news effect, so it moves neither
    ``risk_level`` nor the games haircut."""
    ids = np.asarray(board_ids, dtype="int64")
    out = pd.DataFrame({"player_id": ids, "news_flag": ""})
    if news is None or news.empty:
        return out
    n = news.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
    n = n.set_index(n["player_id"].astype("int64")).reindex(ids)
    text = []
    for _, r in n.iterrows():
        if pd.isna(r.get("note_id")):
            text.append("")
            continue
        what = " ".join(x for x in (r["injury_type"] or r["detail"], f"({r['body_part']})" if r["body_part"] else "") if x)
        back = f", est. return {r['return_date']:%b %d}" if pd.notna(r["return_date"]) else ""
        text.append(f"news: {r['status'].lower()}" + (f", {what.lower()}" if what else "") + back)
    out["news_flag"] = text
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


def minutes_opened(logs: pd.DataFrame, team_games: pd.DataFrame, roster: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """Per team: ``out_mpg``, ``in_mpg``, ``net_open`` and ``departed`` (readable names) from last season's minutes.

    ``logs`` / ``team_games`` are the regular-season tables for the completed season (the last season of the history);
    ``roster`` is the newest snapshot. A player's minutes with a team are ``sum(min) / team games`` (minutes per *team* game, so a
    midseason trade is split correctly). Minutes with a team the player is not on now count as **out** for that team and, if he is
    on a roster, **in** for his new one. Players on no roster (retired, unsigned) count as out, so a free agent who signs later
    overstates the departures; rookies and two-way fill-ins are unmeasured and not counted as in.
    """
    cols = ["team_id", "out_mpg", "in_mpg", "net_open", "departed"]
    if logs.empty or team_games.empty:
        return pd.DataFrame(columns=cols)
    season = logs["season"].max()
    lg, tg = logs[logs["season"] == season], team_games[team_games["season"] == season]
    n_games = tg.groupby("team_id").size()
    mins = lg.groupby(["player_id", "team_id"])["min"].sum().rename("min_sum").reset_index()
    mins["mpg"] = mins["min_sum"] / mins["team_id"].map(n_games)
    now = roster.drop_duplicates("player_id").set_index("player_id")["team_id"]
    mins["now"] = mins["player_id"].map(now)
    off = mins[mins["now"].isna() | (mins["now"] != mins["team_id"])]
    out = off.groupby("team_id")["mpg"].sum()
    inn = off[off["now"].notna()].groupby("now")["mpg"].sum()
    named = {}
    for tid, g in off[off["mpg"] >= DEPART_NAME_MPG].sort_values("mpg", ascending=False).groupby("team_id"):
        named[tid] = ", ".join(f"{names.get(i, str(i))} {m:.0f}" for i, m in zip(g["player_id"].head(4), g["mpg"].head(4)))
    teams = pd.Index(sorted(set(out.index) | set(inn.index)), name="team_id")
    res = pd.DataFrame({"out_mpg": out.reindex(teams).fillna(0.0), "in_mpg": inn.reindex(teams).fillna(0.0)})
    res["net_open"] = res["out_mpg"] - res["in_mpg"]
    res["departed"] = [named.get(t, "") for t in teams]
    return res.reset_index()[cols]


def depth_flags(roster: pd.DataFrame, opened: pd.DataFrame, proj_mpg: pd.Series) -> pd.DataFrame:
    """``player_id, depth_flag``: text for players on a team that lost ``MIN_OPENED_NET``+ net minutes (see :func:`minutes_opened`)."""
    r = roster.drop_duplicates("player_id").set_index("player_id")
    o = opened.set_index("team_id") if len(opened) else pd.DataFrame(columns=["out_mpg", "in_mpg", "net_open", "departed"])
    rows = []
    for pid, tid, abbr in zip(r.index, r["team_id"], r["team_abbr"]):
        if tid not in o.index:
            continue
        row, m = o.loc[tid], proj_mpg.get(pid, np.nan)
        if row["net_open"] >= MIN_OPENED_NET and np.isfinite(m):
            rows.append((int(pid), f"minutes opened: {abbr} lost {row['out_mpg']:.0f} mpg ({row['departed']}), "
                                   f"{row['in_mpg']:.0f} arrived, net {row['net_open']:.0f}"))
    return pd.DataFrame(rows, columns=["player_id", "depth_flag"])


def playoff_flags(playoffs: pd.DataFrame, logs: pd.DataFrame, scoring: dict | None = None) -> pd.DataFrame:
    """``player_id, playoff_flag`` for role players whose closing playoff role was much bigger than their regular-season one.

    ``playoffs`` is the ``playoff_logs`` table (``src.ingest.nba_playoffs``); only its newest season is read, against the same
    season of ``logs``. The signal is the player's last ``PLAYOFF_RECENT_GAMES`` playoff games (a whole-run average hides a role
    that grew late, usually after an injury ahead of him), needing ``PLAYOFF_MIN_GAMES`` games in all and closing mpg >=
    regular-season mpg + ``PLAYOFF_MPG_UPLIFT``, for every player (no role cap: the text carries both minutes figures). ``scoring`` (stat key -> weight) adds fantasy points per game to the text.
    """
    cols = ["player_id", "playoff_flag"]
    if playoffs.empty or logs.empty:
        return pd.DataFrame(columns=cols)
    season = playoffs["season"].max()
    po = playoffs[playoffs["season"] == season].sort_values(["game_date", "game_id"])
    reg_mpg = logs[logs["season"] == season].groupby("player_id")["min"].mean()
    if scoring:
        from src.value.frame import fantasy_points_frame
        po = po.assign(_fp=fantasy_points_frame(po, scoring))
    rows = []
    for pid, g in po.groupby("player_id"):
        if len(g) < PLAYOFF_MIN_GAMES:
            continue
        rm = float(reg_mpg.get(pid, 0.0))
        tail = g.tail(PLAYOFF_RECENT_GAMES)
        m = float(tail["min"].mean())
        if m < rm + PLAYOFF_MPG_UPLIFT:
            continue
        extra = f", {tail['_fp'].mean():.1f} FP/g" if "_fp" in tail else ""
        rows.append((int(pid), f"playoffs {season}: {m:.0f} mpg over his last {len(tail)} of {len(g)} games{extra} "
                               f"(regular season {rm:.0f} mpg)"))
    return pd.DataFrame(rows, columns=cols)


def build_overlay(board: pd.DataFrame, *, preseason: pd.DataFrame | None = None, status: pd.DataFrame | None = None,
                  context: pd.DataFrame | None = None, depth: pd.DataFrame | None = None,
                  playoffs: pd.DataFrame | None = None, news: pd.DataFrame | None = None, today=None) -> pd.DataFrame:
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
    for frame, col in ((news, "news_flag"), (depth, "depth_flag"), (playoffs, "playoff_flag")):
        if frame is not None and len(frame):
            text = frame.drop_duplicates("player_id").set_index("player_id")[col].reindex(ids).fillna("").to_numpy()
            for i, t in enumerate(text):
                if t:
                    flags[i].append(t)   # informational: neither changes risk_level nor the games haircut
    out["risk_flags"] = ["; ".join(f) for f in flags]
    out["risk_level"] = np.array(["", "watch", "high"])[level]
    out["risk_gp_haircut"] = haircut
    out["risk_gp"] = out["proj_gp"].to_numpy(float) * (1.0 - haircut) if "proj_gp" in out.columns else np.nan
    return out
