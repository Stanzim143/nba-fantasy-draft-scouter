"""Rest-of-season (ROS) projection: the preseason projection shrunk toward season-to-date production.

    python -m src.inseason.ros --season 2026-27 --as-of 2026-12-01 [--model baseline] [--top 30]

Method (ADR 0015 D2; evidence in ``docs/inseason.md`` and ``python -m src.inseason.ros_eval``)
------------------------------------------------------------------------------------------------
For every player, three quantities are each a conjugate-style shrinkage average of the *prior* (the
preseason projection, built only from seasons before this one) and the *sample* (this season's games up
to and including ``as_of``), with the sample's weight growing with exposure::

    w = n / (n + k)          posterior = (k * prior + n * sample) / (k + n)

* **Per-minute production** ``fp_per_min``: ``n`` = minutes played to date, ``k`` = ``minutes_pseudo``.
* **Minutes per game** ``mpg``: ``n`` = games played, ``k`` = ``games_pseudo`` (a role change shows in
  minutes quickly, per-minute rates stabilise slowly, hence two separate weights).
* **Availability** (probability of playing a given team game): ``n`` = team games played to date,
  sample = games played / team games, ``k`` = ``avail_pseudo``. A player who has missed the whole season
  so far is pulled down accordingly; a healthy one is pulled toward 1.

``ros_fppg = fp_per_min * mpg`` and ``ros_games = availability * team games remaining`` (games remaining
after ``as_of`` from the schedule when given, else season length minus team games played), so
``ros_total_fp = ros_fppg * ros_games``. The pseudo-counts are chosen by walk-forward evidence on
seasons before the ones they are reported on (see ``ros_eval``); the defaults below are those values.
Players with no preseason projection (a call-up, a two-way signing) get a deliberately weak prior at the
level of a bench player, flagged ``has_prior = False``.

``mode`` exists so the backtest can compare ``"prior"`` (preseason only), ``"sample"`` (season-to-date
only), ``"blend"`` (this method) and ``"blend_fppg"`` (shrink FPPG directly).

Leakage: only game log and team-game rows with ``game_date <= as_of`` of the target season are read, and
the prior is projected from ``History.until(tables, season)``. ``tests/inseason/test_ros.py`` perturbs
everything dated after ``as_of`` and asserts the output is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import HISTORY_TABLES, History
from src.value.frame import fantasy_points_frame
from src.value.league import load_league
from src.value.replacement import league_shape
from src.value.vorp import compute_vorp

MODES = ("blend", "blend_fppg", "prior", "sample")


@dataclass(frozen=True)
class RosParams:
    # Chosen on the train seasons 2016-17..2020-21 of ``ros_eval`` (lowest mean ROS-total MAE); the grid edge
    # was reached for games_pseudo and avail_pseudo: the sample dominates minutes and availability.
    minutes_pseudo: float = 300.0    # pseudo-minutes behind the preseason per-minute rate
    games_pseudo: float = 3.0        # pseudo-games behind the preseason minutes per game
    avail_pseudo: float = 5.0        # pseudo team-games behind the preseason availability
    fppg_pseudo: float = 5.0         # pseudo-games for the direct-FPPG variant
    mode: str = "blend"
    weak_fppm: float = 0.45          # prior for players without a projection: FP per minute ...
    weak_mpg: float = 12.0           # ... minutes per game ...
    weak_avail: float = 0.55         # ... and availability


def _shrink(prior, sample, n, k):
    n = np.asarray(n, float)
    prior = np.asarray(prior, float)
    sample = np.where(n > 0, np.asarray(sample, float), prior)
    return (k * prior + n * sample) / (k + n)


def season_to_date(tables, season: str, as_of, scoring) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This season's game logs and team games with ``game_date <= as_of`` (the only in-season data read)."""
    cut = pd.Timestamp(as_of)
    gl = tables["game_logs"]
    gl = gl[(gl["season"] == season) & (pd.to_datetime(gl["game_date"]) <= cut)].copy()
    gl["fp"] = fantasy_points_frame(gl, scoring) if len(gl) else pd.Series(dtype="float64")
    tg = tables["team_games"]
    tg = tg[(tg["season"] == season) & (pd.to_datetime(tg["game_date"]) <= cut)]
    return gl, tg


def player_to_date(gl: pd.DataFrame) -> pd.DataFrame:
    """Per-player totals to date: gp, min, fp, latest team."""
    if gl.empty:
        return pd.DataFrame(columns=["player_id", "gp", "min", "fp", "team_id"]).set_index("player_id")
    last = gl.sort_values(["game_date", "game_id"], kind="mergesort").groupby("player_id").tail(1)
    agg = gl.groupby("player_id").agg(gp=("game_id", "size"), min=("min", "sum"), fp=("fp", "sum"))
    agg["team_id"] = last.set_index("player_id")["team_id"]
    return agg


def build_ros(tables, season: str, as_of, *, prior: pd.DataFrame | None = None, model: str = "baseline",
              cfg: dict | None = None, schedule: pd.DataFrame | None = None, params: RosParams | None = None,
              teams: pd.Series | None = None, season_games: float | None = None,
              with_value: bool = True, league_teams: int | None = None) -> pd.DataFrame:
    """Rest-of-season projection for every projected player as of ``as_of`` (inclusive).

    ``tables`` needs ``game_logs``, ``team_games``, ``players``, ``player_season_bio``. ``prior`` may be a
    precomputed preseason projection (one model, this season) to avoid re-projecting; it must have been
    built without this season's data. ``teams`` (player_id -> team_id) places players who have not played
    yet (preseason); a player's latest game-log team wins once he has played. ``schedule`` is a
    ``schedule_games`` frame used for games remaining.
    """
    params = params or RosParams()
    if params.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cfg = cfg or load_league()
    scoring = cfg["scoring"]
    as_of_ts = pd.Timestamp(as_of)

    if prior is None:
        from src.models.registry import get_projector

        hist = History.until({k: tables[k] for k in tables if k in HISTORY_TABLES}, season)
        hist.assert_no_future()
        prior = get_projector(model).project(hist)
    prior = prior[prior["season"] == season]
    if prior.empty:
        raise ValueError(f"no preseason projection for {season}")
    if season_games is None:
        from src.models.panel import season_lengths, target_season_games

        hist_games = History.until({k: tables[k] for k in tables if k in HISTORY_TABLES}, season)
        season_games = float(target_season_games(season_lengths(hist_games, None))) if len(hist_games.team_games) else 82.0

    gl, tg = season_to_date(tables, season, as_of_ts, scoring)
    ptd = player_to_date(gl)
    team_gp = tg.groupby("team_id").size()

    ids = np.union1d(prior["player_id"].to_numpy(), ptd.index.to_numpy()).astype("int64")
    df = pd.DataFrame({"player_id": ids}).set_index("player_id")
    p = prior.drop_duplicates("player_id").set_index("player_id")
    df["name"] = p["player_name"].reindex(ids)
    if "player_name" in gl.columns and len(gl):
        nm = gl.drop_duplicates("player_id").set_index("player_id")["player_name"]
        df["name"] = df["name"].fillna(nm.reindex(ids))
    pos = tables["players"].drop_duplicates("player_id").set_index("player_id")["position"]
    df["position"] = pos.reindex(ids)
    df["has_prior"] = pd.Series(ids).isin(p.index).to_numpy()

    season_games = float(season_games)
    prior_mpg = p["proj_mpg"].reindex(ids).to_numpy(float)
    prior_fppg = p["proj_fppg"].reindex(ids).to_numpy(float)
    prior_gp = p["proj_gp"].reindex(ids).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_fppm = np.where(prior_mpg > 0, prior_fppg / prior_mpg, np.nan)
    has = df["has_prior"].to_numpy()
    prior_mpg = np.where(has, prior_mpg, params.weak_mpg)
    prior_fppm = np.where(has & np.isfinite(prior_fppm), prior_fppm, params.weak_fppm)
    prior_fppg = np.where(has, prior_fppg, prior_fppm * prior_mpg)
    prior_avail = np.where(has, np.clip(prior_gp / season_games, 0, 1), params.weak_avail)

    gp = ptd["gp"].reindex(ids).fillna(0).to_numpy(float)
    minutes = ptd["min"].reindex(ids).fillna(0).to_numpy(float)
    fp = ptd["fp"].reindex(ids).fillna(0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_fppm = np.where(minutes > 0, fp / minutes, np.nan)
        s_mpg = np.where(gp > 0, minutes / gp, np.nan)
        s_fppg = np.where(gp > 0, fp / gp, np.nan)

    team = ptd["team_id"].reindex(ids)
    if teams is not None:
        team = team.fillna(teams.reindex(ids))
    df["team_id"] = team.to_numpy()
    tgp = pd.Series(df["team_id"].to_numpy()).map(team_gp).to_numpy(float)
    if schedule is not None:
        sch = schedule[schedule["season"] == season]
        from src.inseason.schedule import games_remaining

        left_by_team = games_remaining(sch, as_of_ts)
        # a team with no game after as_of has 0 left (not unknown): reindex over every scheduled team so only
        # players with an unknown team id reach the median fallback below
        all_teams = pd.unique(pd.concat([sch["home_team_id"], sch["away_team_id"]]).dropna())
        left_by_team = left_by_team.reindex(all_teams.astype(left_by_team.index.dtype, copy=False), fill_value=0)
        left = pd.Series(df["team_id"].to_numpy()).map(left_by_team).to_numpy(float)
    else:
        left = np.maximum(season_games - tgp, 0.0)
    fallback_left = float(np.nanmedian(left)) if np.isfinite(left).any() else season_games
    left = np.where(np.isfinite(left), left, fallback_left)
    tgp = np.where(np.isfinite(tgp), tgp, np.nanmax(team_gp) if len(team_gp) else 0.0)
    s_avail = np.where(tgp > 0, np.minimum(gp / np.maximum(tgp, 1), 1.0), np.nan)

    mode = params.mode
    if mode == "prior":
        fppg, mpg, avail = prior_fppg, prior_mpg, prior_avail
    elif mode == "sample":
        fppg = np.where(np.isfinite(s_fppg), s_fppg, prior_fppg)
        mpg = np.where(np.isfinite(s_mpg), s_mpg, prior_mpg)
        avail = np.where(np.isfinite(s_avail), s_avail, prior_avail)
    else:
        mpg = _shrink(prior_mpg, s_mpg, gp, params.games_pseudo)
        avail = _shrink(prior_avail, s_avail, np.where(np.isfinite(tgp), tgp, 0), params.avail_pseudo)
        if mode == "blend":
            fppm = _shrink(prior_fppm, s_fppm, minutes, params.minutes_pseudo)
            fppg = fppm * mpg
        else:
            fppg = _shrink(prior_fppg, s_fppg, gp, params.fppg_pseudo)
    ros_games = np.clip(avail, 0, 1) * left
    df["gp"] = gp.astype(int)
    df["mpg"] = s_mpg
    df["fppg_to_date"] = s_fppg
    df["prior_fppg"] = prior_fppg
    df["prior_mpg"] = prior_mpg
    df["prior_avail"] = prior_avail
    df["avail"] = np.clip(avail, 0, 1)
    df["ros_fppg"] = fppg
    df["ros_mpg"] = mpg
    df["team_games_left"] = left
    df["ros_games"] = ros_games
    df["ros_total_fp"] = fppg * ros_games
    df["w_sample"] = np.where(minutes > 0, minutes / (params.minutes_pseudo + minutes), 0.0)
    df["as_of"] = as_of_ts.normalize()
    df = df.reset_index()
    if with_value:
        df = add_value(df, cfg, season_games_left=float(np.median(left)) if len(left) else season_games,
                       teams=league_teams)
    return df


def add_value(ros: pd.DataFrame, cfg: dict, *, season_games_left: float, teams: int | None = None) -> pd.DataFrame:
    """VORP over the rest of the season using the draft board's replacement machinery.

    The value engine's inputs are ``proj_fppg``, ``proj_gp`` and ``proj_total_fp``; here they are the ROS
    values, and ``season_games`` is the games left, so replacement level and the bench weight are derived
    for the rest of the season exactly as they are for the preseason board.
    """
    shape = league_shape(cfg, teams=teams)
    frame = pd.DataFrame({"proj_fppg": ros["ros_fppg"], "proj_gp": ros["ros_games"],
                          "proj_total_fp": ros["ros_total_fp"], "position": ros["position"]}, index=ros.index)
    has_pos = frame["position"].notna().any()
    res = compute_vorp(frame if has_pos else frame.drop(columns="position"), shape,
                       season_games=max(season_games_left, 1.0), positional="auto" if has_pos else "off")
    out = ros.copy()
    out["ros_vorp"] = res.frame["vorp"].to_numpy()
    out["repl_total"] = res.frame["repl_total"].to_numpy()
    out.attrs["replacement"] = res.replacement.as_dict()
    out = out.sort_values(["ros_vorp", "ros_total_fp", "player_id"], ascending=[False, False, True],
                          kind="mergesort").reset_index(drop=True)
    out["ros_rank"] = np.arange(1, len(out) + 1)
    out.attrs["replacement"] = res.replacement.as_dict()
    return out


def with_params(params: RosParams, **kw) -> RosParams:
    return replace(params, **kw)


# --------------------------------------------------------------------------- CLI

def _safe_print(text: str, *, file=None) -> None:
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), file=stream)


DISPLAY = ["ros_rank", "name", "position", "gp", "fppg_to_date", "prior_fppg", "ros_fppg", "avail", "ros_games",
           "ros_total_fp", "ros_vorp", "w_sample"]


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.ros", description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", default=None, help="default: config/league.yaml")
    ap.add_argument("--as-of", default=None, help="inclusive cutoff date (default: today)")
    ap.add_argument("--model", default="baseline", help="preseason projector (src.models.registry)")
    ap.add_argument("--mode", choices=MODES, default="blend")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None, help="CSV to write (all players)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic league (demo/testing only)")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    from src.inseason.context import ContextUnavailable, load_context

    try:
        ctx = load_context(args.season, args.as_of, model=args.model, mode=args.mode, data_dir=args.data_dir,
                           synthetic=args.synthetic)
    except ContextUnavailable as exc:
        _safe_print(f"error: {exc}", file=sys.stderr)
        return 2
    ros = ctx.ros
    rep = ros.attrs.get("replacement", {})
    _safe_print(f"ROS projection, {ctx.season} as of {ctx.as_of.date()} ({args.model}, mode {args.mode}): "
                f"{len(ros)} players; season-to-date games from {int((ros['gp'] > 0).sum())} players; "
                f"replacement ROS total {rep.get('total', float('nan')):.0f} FP")
    for note in ctx.notes:
        _safe_print(f"note: {note}")
    _safe_print(ros.head(args.top)[DISPLAY].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ros.to_csv(args.out, index=False, encoding="utf-8")
        _safe_print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
