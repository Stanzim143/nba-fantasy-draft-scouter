"""The in-season artifacts the nightly job produces from the local data (ADR 0018). Read-only, no network.

Everything is computed from :func:`src.inseason.context.load_context` as of the last *completed* game day, so the tools of
ADR 0015 (rest-of-season projection, waiver finder, signals, schedule) are reused, not re-implemented:

* ``ros``       the rest-of-season board (top of it goes to CSV, ranks go to the diff state);
* ``waivers``   best adds and streaming candidates for the user's team, this week and next week;
* ``alerts``    rising-minutes players and injury beneficiaries anywhere in the league, tagged mine / free agent / rostered;
* ``week``      games per team for this and next matchup week, and who plays on the next slate;
* ``lineup``    a start/sit view of the user's roster for this week and for the next game day.

Each artifact is built in isolation: one raising does not stop the others; the caller gets the errors as data.
Without a configured team (or before the draft) the team-specific artifacts are skipped with a stated reason and the
league-wide ones still run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inseason.context import (
    ContextUnavailable, InSeasonContext, RosterChoice, free_agent_ids, load_context, mapped_teams, resolve_roster,
)
from src.inseason.lineup import lineup_value, roster_frame
from src.inseason.schedule import ABBR_OF_TEAM_ID, team_game_days
from src.inseason.signals import absent_players, beneficiaries, trend_signals
from src.inseason.waivers import _season_logs, find_waivers, pick_week
from src.value.replacement import league_shape

LOGGER = logging.getLogger("nightly")
TOP_ROS_ROWS = 300
STATE_ROS_RANKS = 60


@dataclass
class Artifacts:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)          # small, JSON-able, diffed run over run
    meta: dict[str, Any] = field(default_factory=dict)           # facts the report prints
    errors: dict[str, str] = field(default_factory=dict)         # artifact -> error text
    notes: list[str] = field(default_factory=list)


def _abbr(team_id) -> str:
    try:
        return ABBR_OF_TEAM_ID.get(int(team_id), str(team_id)) if pd.notna(team_id) else ""
    except (TypeError, ValueError):
        return str(team_id)


def eastern_offset_hours(utc_dt) -> int:
    """US Eastern offset from UTC (-5 standard, -4 daylight) for a UTC datetime; no tz database needed (Windows has none in
    this venv). DST runs from the second Sunday of March 02:00 local to the first Sunday of November 02:00 local."""
    from datetime import datetime, timedelta

    d = utc_dt.astimezone(timezone.utc).replace(tzinfo=None)

    def nth_sunday(year, month, n):
        first = datetime(year, month, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    start = nth_sunday(d.year, 3, 2) + timedelta(hours=2 + 5)       # 02:00 EST expressed in UTC
    end = nth_sunday(d.year, 11, 1) + timedelta(hours=2 + 4)        # 02:00 EDT expressed in UTC
    return -4 if start <= d < end else -5


def completed_through(now) -> date:
    """The latest US game date whose games are all final at ``now`` (an aware datetime).

    NBA games are dated in US Eastern time and the last of a night ends about 03:30 ET on the next calendar day, so
    the newest fully finished date is ``(now_ET - 3.5 h).date() - 1 day``.
    """
    from datetime import timedelta

    et = now.astimezone(timezone.utc) + timedelta(hours=eastern_offset_hours(now))
    return (et - timedelta(hours=3, minutes=30)).date() - timedelta(days=1)


def build_context(season: str, through: date, slate: date, *, data_dir: Path, league_id: int | None,
                  use_injuries: bool | None = None, model: str = "baseline") -> InSeasonContext:
    ctx = load_context(season, pd.Timestamp(through), model=model, data_dir=data_dir, league_id=league_id,
                       use_injuries=use_injuries)
    ctx.week_anchor = pd.Timestamp(slate)
    return ctx


# --------------------------------------------------------------------------- roster choice

def choose_roster(ctx: InSeasonContext, *, team_id: int | None, mock_team: int | None) -> tuple[RosterChoice | None, str | None]:
    """The user's roster, or ``(None, reason)`` when it cannot be known (no team id, draft not held, unmapped ids)."""
    if mock_team is not None:
        return resolve_roster(ctx, mock_team=mock_team), None
    if team_id is None:
        return None, "no team id configured (ESPN_TEAM_ID, --team-id, or team_id in nightly.json / daily_refresh.json)"
    if not ctx.league:
        return None, "no synced league data (the league step has not produced processed/espn_league/<id>_<year>.json)"
    try:
        return resolve_roster(ctx, team_id=team_id), None
    except ContextUnavailable as exc:
        return None, str(exc)


def ownership(choice: RosterChoice | None, ctx: InSeasonContext) -> dict[int, str]:
    """player_id -> ``mine`` | ``other`` (rostered by another team)."""
    out: dict[int, str] = {}
    teams, _ = mapped_teams(ctx.league, ctx.id_map)
    for t in teams.values():
        for pid, _slot in t["players"]:
            out[pid] = "other"
    if choice is not None:
        for pid in choice.my_ids:
            out[pid] = "mine"
    return out


# --------------------------------------------------------------------------- artifacts

def make_ros(ctx: InSeasonContext) -> pd.DataFrame:
    cols = ["player_id", "name", "position", "team_id", "gp", "mpg", "fppg_to_date", "prior_fppg", "ros_fppg", "avail",
            "team_games_left", "ros_games", "ros_total_fp"]
    r = ctx.ros.drop_duplicates("player_id").sort_values("ros_total_fp", ascending=False)
    r = r[[c for c in cols + ["ros_vorp"] if c in r.columns]].head(TOP_ROS_ROWS).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    r.insert(4, "team", [_abbr(t) for t in r["team_id"]])
    return r


def make_alerts(ctx: InSeasonContext, own: dict[int, str], fa_ids: set[int], *, top: int = 15) -> dict[str, pd.DataFrame]:
    idx = ctx.ros.drop_duplicates("player_id").set_index("player_id")
    gl, tg = _season_logs(ctx)
    tag = lambda pid: "mine" if own.get(pid) == "mine" else ("other team" if own.get(pid) == "other" else ("free agent" if pid in fa_ids else "?"))  # noqa: E731
    out: dict[str, pd.DataFrame] = {}
    if gl.empty:
        return {"rising": pd.DataFrame(), "beneficiaries": pd.DataFrame(), "absent": pd.DataFrame()}
    trends = trend_signals(gl)
    if len(trends):
        rise = trends[(trends["rising_minutes"] | trends["rising_usage"]) & (trends["recent_mpg"] >= 15)].copy()
        rise = rise[rise["player_id"].isin(idx.index)]
        rise["name"] = rise["player_id"].map(idx["name"])
        rise["team"] = [_abbr(idx.at[p, "team_id"]) for p in rise["player_id"]]
        rise["ros_fppg"] = rise["player_id"].map(idx["ros_fppg"])
        rise["status"] = [tag(p) for p in rise["player_id"]]
        rise["injury"] = [("" if ctx.injuries is None else ctx.injuries.get(p, "")) for p in rise["player_id"]]
        out["rising"] = rise.sort_values("min_delta", ascending=False).head(top * 2).reset_index(drop=True)
    else:
        out["rising"] = pd.DataFrame()
    absent = absent_players(gl, tg, injuries=ctx.injuries)
    positions = idx["position"]
    fpm = gl.groupby("player_id")["fp"].sum() / gl.groupby("player_id")["min"].sum()
    ben = beneficiaries(gl, tg, absent, positions, fp_per_min=fpm) if len(absent) else pd.DataFrame()
    if len(ben):
        ben = ben[ben["player_id"].isin(idx.index)].copy()
        ben = ben.sort_values("gain_mpg", ascending=False).drop_duplicates("player_id")
        ben["name"] = ben["player_id"].map(idx["name"])
        ben["team"] = [_abbr(t) for t in ben["team_id"]]
        ben["out_name"] = ben["out_player_id"].map(idx["name"])
        ab = absent.set_index("player_id")
        ben["out_streak"] = ben["out_player_id"].map(ab["streak"])
        ben["long_term"] = ben["out_player_id"].map(ab["long_term"])
        ben["status"] = [tag(p) for p in ben["player_id"]]
        ben["injury"] = [("" if ctx.injuries is None else ctx.injuries.get(p, "")) for p in ben["player_id"]]
        out["beneficiaries"] = ben.sort_values("gain_fppg", ascending=False).head(top * 2).reset_index(drop=True)
    else:
        out["beneficiaries"] = pd.DataFrame()
    if len(absent):
        a = absent.copy()
        a["name"] = a["player_id"].map(idx["name"])
        a["team"] = [_abbr(t) for t in a["team_id"]]
        a["status"] = [tag(p) for p in a["player_id"]]
        out["absent"] = a.sort_values("mpg", ascending=False).head(top * 2).reset_index(drop=True)
    else:
        out["absent"] = pd.DataFrame()
    return out


def make_week(ctx: InSeasonContext, slate: date, mine_teams: set[int]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """This/next week's games per team and the next slate. Requires the schedule and calendar."""
    if ctx.weekly is None or ctx.calendar is None or ctx.schedule is None:
        raise ContextUnavailable("no schedule_games for this season (the schedule step has not stored it)")
    cur = ctx.calendar.current_week(slate)
    nxt = ctx.calendar.next_week(slate)
    frames: dict[str, pd.DataFrame] = {}
    meta: dict[str, Any] = {"calendar_source": ctx.calendar.source, "slate": slate.isoformat()}
    for label, wk in (("this_week", cur), ("next_week", nxt)):
        if wk is None:
            continue
        t = ctx.weekly[ctx.weekly["week"] == wk.week].copy()
        t["mine"] = t["team_id"].isin(mine_teams)
        frames[label] = t.sort_values(["games_left" if label == "this_week" else "games", "off_night_games"],
                                      ascending=False).reset_index(drop=True)
        meta[label] = {"week": wk.week, "start": wk.start.isoformat(), "end": wk.end.isoformat(), "kind": wk.kind,
                       "mean_games": float(t["games"].mean())}
    tgd = team_game_days(ctx.schedule)
    day = pd.Timestamp(slate)
    today = tgd[tgd["game_date"].dt.normalize() == day]
    meta["slate_games"] = int(today["game_id"].nunique())
    meta["slate_teams"] = sorted(_abbr(t) for t in today["team_id"].unique())
    if meta["slate_games"] == 0:
        later = ctx.schedule[pd.to_datetime(ctx.schedule["game_date"]) > day]
        meta["next_game_day"] = None if later.empty else pd.Timestamp(later["game_date"].min()).date().isoformat()
    meta["_slate_team_ids"] = sorted(int(t) for t in today["team_id"].unique())
    return frames, meta


def make_lineup(ctx: InSeasonContext, choice: RosterChoice, slate: date, slate_team_ids: set[int],
                streams: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    shape = league_shape(ctx.cfg)
    n_ir = int(ctx.cfg["league"]["roster"].get("ir", 1))
    idx = ctx.ros.drop_duplicates("player_id").set_index("player_id").copy()
    week = pick_week(ctx)
    if week is None or ctx.weekly is None:
        raise ContextUnavailable("no matchup week for this date (season over, or no schedule)")
    wk = ctx.weekly[ctx.weekly["week"] == week.week].set_index("team_id")
    idx["wk_games_left"] = idx["team_id"].map(wk["games_left"]).fillna(0.0)
    idx["b2b"] = idx["team_id"].map(wk["b2b"]).fillna(0).astype(int)
    idx["week_fp"] = idx["wk_games_left"] * idx["avail"] * idx["ros_fppg"]
    inj = ctx.injuries if ctx.injuries is not None else pd.Series(dtype=object)
    idx["injury"] = [inj.get(p, "") for p in idx.index]
    plays = idx["team_id"].isin(slate_team_ids)
    idx["today_fp"] = np.where(plays & idx["injury"].ne("OUT"), idx["avail"] * idx["ros_fppg"], 0.0)

    ids = [p for p in choice.my_ids if p in idx.index]
    week_lineup = lineup_value(roster_frame(idx, ids, "week_fp"), shape, 0.0, ir_ids=choice.ir_ids, n_ir=n_ir)
    day_lineup = lineup_value(roster_frame(idx, ids, "today_fp"), shape, 0.0, ir_ids=choice.ir_ids, n_ir=n_ir)
    slot_of = {pid: slot for slot, pid in week_lineup.starters.items() if pid is not None}
    today_slot = {pid: slot for slot, pid in day_lineup.starters.items() if pid is not None and idx.at[pid, "today_fp"] > 0}
    bench_with_game = [p for p in ids if p not in slot_of and p in today_slot and p not in week_lineup.ir]
    bench_with_game.sort(key=lambda p: -float(idx.at[p, "today_fp"]))
    sitters = [p for p in ids if p in slot_of and int(idx.at[p, "team_id"]) not in slate_team_ids and idx.at[p, "wk_games_left"] > 0]
    swap_in = dict(zip(sorted(sitters, key=lambda p: float(idx.at[p, "week_fp"])), bench_with_game))
    rows = []
    for pid in ids:
        r = idx.loc[pid]
        where = slot_of.get(pid) or ("IR" if pid in week_lineup.ir else "BENCH")
        hints = []
        if r["injury"] == "OUT":
            hints.append("OUT: do not start; IR spot or replace" if where != "IR" else "OUT and on IR: fine")
        elif r["injury"] not in ("", "ACTIVE"):
            hints.append(f"{r['injury']}: check status before the lock")
        if where not in ("BENCH", "IR") and r["wk_games_left"] == 0:
            hints.append("no games left this week")
        if where == "BENCH" and r["wk_games_left"] > 0 and pid in today_slot:
            hints.append(f"start on {slate:%a} (plays; slot {today_slot[pid]})")
        if pid in swap_in:
            hints.append(f"no game on {slate:%a}: swap in {idx.at[swap_in[pid], 'name']}")
        rows.append({"player_id": pid, "name": r["name"], "position": r["position"], "team": _abbr(r["team_id"]),
                     "slot_this_week": where, "games_left_week": float(r["wk_games_left"]), "b2b": int(r["b2b"]),
                     "plays_today": bool(int(r["team_id"]) in slate_team_ids), "today_slot": today_slot.get(pid, ""),
                     "injury": r["injury"], "ros_fppg": float(r["ros_fppg"]), "week_exp_fp": float(r["week_fp"]),
                     "hint": "; ".join(hints)})
    table = pd.DataFrame(rows)
    swaps = []
    if streams is not None and len(streams):
        for _, s in streams[streams["week_gain_fp"] >= 3.0].head(3).iterrows():
            swaps.append({"add": s["name"], "drop": s["drop"], "week_gain_fp": float(s["week_gain_fp"]),
                          "games": float(s["week_exp_games"])})
    meta = {"week": week.week, "week_expected_fp": float(week_lineup.value), "empty_slots": week_lineup.empty_slots,
            "today_players": int(len(today_slot)), "today_expected_fp": float(day_lineup.value),
            "slate": slate.isoformat(), "swaps": swaps}
    return table, meta


def top_free_agents(ctx: InSeasonContext, rostered: set[int], n: int = 20) -> pd.DataFrame:
    ids = set(free_agent_ids(ctx, rostered))
    r = ctx.ros.drop_duplicates("player_id")
    r = r[r["player_id"].isin(ids)].sort_values("ros_total_fp", ascending=False).head(n)
    return pd.DataFrame({"name": r["name"].to_numpy(), "position": r["position"].to_numpy(),
                         "team": [_abbr(t) for t in r["team_id"]], "ros_fppg": r["ros_fppg"].to_numpy(),
                         "ros_total_fp": r["ros_total_fp"].to_numpy()})


# --------------------------------------------------------------------------- driver

def build_artifacts(ctx: InSeasonContext, slate: date, *, team_id: int | None, mock_team: int | None = None,
                    top: int = 15, espn_fa_ids: set[int] | None = None) -> Artifacts:
    art = Artifacts()
    art.notes.extend(ctx.notes)
    art.meta.update({"season": ctx.season, "as_of": ctx.as_of.date().isoformat(), "slate": slate.isoformat(),
                     "games_played_players": int((ctx.ros["gp"] > 0).sum()),
                     "league_synced": bool(ctx.league), "injuries_loaded": ctx.injuries is not None})

    def isolated(name: str, fn):
        try:
            return fn()
        except Exception as exc:                                    # noqa: BLE001 - one artifact never stops the others
            LOGGER.exception("artifact %s failed", name)
            art.errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    ros = isolated("ros", lambda: make_ros(ctx))
    if ros is not None:
        art.frames["ros"] = ros
        art.state["ros_rank"] = {int(r.player_id): {"rank": int(r.rank), "name": r.name} for r in ros.head(STATE_ROS_RANKS).itertuples()}

    choice, why = (None, None)
    try:
        choice, why = choose_roster(ctx, team_id=team_id, mock_team=mock_team)
    except Exception as exc:                                        # noqa: BLE001
        art.errors["roster"] = f"{type(exc).__name__}: {exc}"
    if choice is None and "roster" not in art.errors:
        art.meta["team_skipped"] = why
        art.notes.append(f"team-specific artifacts skipped: {why}")
    own = ownership(choice, ctx)
    rostered = choice.rostered if choice is not None else set(own)
    art.meta["team_label"] = choice.label if choice is not None else None
    art.meta["my_players"] = len(choice.my_ids) if choice is not None else 0
    art.state["my_roster"] = sorted(int(p) for p in choice.my_ids) if choice is not None else None
    if mock_team is not None and choice is not None:
        art.meta["mock"] = True

    fa_ids = set(free_agent_ids(ctx, rostered))
    if not own:
        art.notes.append("no league rosters are known yet (draft not held or league not synced): every active player counts as a free agent")

    alerts = isolated("alerts", lambda: make_alerts(ctx, own, fa_ids, top=top))
    if alerts is not None:
        art.frames.update({f"alerts_{k}": v for k, v in alerts.items()})
        rising = alerts.get("rising", pd.DataFrame())
        art.state["rising_minutes"] = ({int(r.player_id): {"name": r.name, "min_delta": round(float(r.min_delta), 1)}
                                        for r in rising.itertuples() if r.rising_minutes} if len(rising) else {})
        ben = alerts.get("beneficiaries", pd.DataFrame())
        art.state["beneficiaries"] = ({int(r.player_id): {"name": r.name, "out": r.out_name, "gain_mpg": round(float(r.gain_mpg), 1)}
                                       for r in ben.head(top).itertuples()} if len(ben) else {})

    mine_teams = {int(ctx.ros.drop_duplicates("player_id").set_index("player_id").at[p, "team_id"])
                  for p in (choice.my_ids if choice is not None else []) if p in set(ctx.ros["player_id"])}
    week = isolated("week", lambda: make_week(ctx, slate, mine_teams))
    slate_ids: set[int] = set()
    if week is not None:
        frames, meta = week
        slate_ids = set(meta.pop("_slate_team_ids", []))
        art.frames.update({f"week_{k}": v for k, v in frames.items()})
        art.meta["week"] = meta
        art.state["calendar_source"] = meta.get("calendar_source")

    streams_now = None
    if choice is not None:
        wv = isolated("waivers", lambda: find_waivers(ctx, choice, top=top))
        if wv is not None:
            adds = wv.adds.copy()
            if espn_fa_ids is not None and len(adds):
                adds["in_espn_fa_list"] = adds["player_id"].isin(espn_fa_ids)
            art.frames["waivers_adds"], art.frames["waivers_streams"] = adds, wv.streams
            streams_now = wv.streams
            art.meta["waiver_notes"] = wv.notes
            art.meta["stream_week"] = wv.week_label
            art.state["adds_top"] = {int(r.player_id): {"name": r.name, "gain": round(float(r.gain_ros_fp), 1), "drop": r.drop}
                                     for r in adds.head(10).itertuples() if r.gain_ros_fp > 0}
        wn = isolated("waivers_next", lambda: find_waivers(ctx, choice, top=top, next_week=True))
        if wn is not None:
            art.frames["waivers_streams_next"] = wn.streams
            art.meta["stream_week_next"] = wn.week_label
        lu = isolated("lineup", lambda: make_lineup(ctx, choice, slate, slate_ids, streams_now))
        if lu is not None:
            art.frames["lineup"], art.meta["lineup"] = lu
    else:
        fa = isolated("free_agents", lambda: top_free_agents(ctx, rostered))
        if fa is not None:
            art.frames["free_agents_top"] = fa
    return art
