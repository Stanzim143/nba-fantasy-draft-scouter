"""Does late-season availability, in the fantasy-playoff weeks, fall below the model's rate predictably? (ADR 0039, part B)

    python -m src.backtest.playoff_availability [--seasons 2016-17:2025-26] [--cache-dir DIR] [--out reports/playoff_availability]

For every season the league's playoff weeks are found with the same fallback the live tools use
(:func:`src.inseason.weeks.calendar_from_settings` with no ESPN payload, over that season's ``team_games`` schedule and the league's
period counts). The universe is rotation veterans (>= 41 GP and >= 20 mpg the season before), projected walk-forward by the board
model. The baseline expectation of playoff-week games is ADR 0039 part A's: ``team_po * clip(proj_gp / season_games, 0, 1)``. Two
pre-registered candidates add a predicted late-season share gap ``G_hat`` fitted on earlier seasons only: (1) the training mean, (2)
an OLS on pre-season features (team's previous-season win%, age, own previous-season gap, star on a bad team, previous-season mpg and
minutes). The decision rule is in the ADR and is applied by :func:`decide`.

Every feature comes from ``History.until(target)``; the label of a training season uses only that season's games, and training rows
are strictly earlier seasons (:func:`training_rows`). ``tests/backtest/test_playoff_availability.py`` checks both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd

from src.backtest import metrics as M
from src.contracts import HISTORY_TABLES, History, season_start
from src.value.league import load_league

DEFAULT_SEASONS = "2016-17:2025-26"
BOARD_MODEL = "baseline_hurdle_adp_depth_rookie_offseason_debut"
MIN_MPG, MIN_GP = 20.0, 41
MIN_MEAN_PLAYOFF_TEAM_GAMES = 8.0       # a season whose derived playoff weeks average fewer team games is not scored
BAD_TEAM_WINPCT = 0.40
STAR_QUANTILE = 0.75
N_BOOT = 2000
MIN_WIN_SHARE = 0.70
FEATURES = ["team_winpct", "age", "prior_gap", "star_bad", "prior_mpg", "prior_min_k"]


# --------------------------------------------------------------------------- calendar per season

def season_windows(team_games: pd.DataFrame, season: str, cfg: Mapping | None = None) -> dict | None:
    """Regular and playoff week date ranges plus team games in each, for one season's realised schedule (``None``: no games)."""
    from src.inseason.schedule import ScheduleError, from_team_games, games_per_day
    from src.inseason.weeks import calendar_from_settings

    cfg = cfg or load_league()
    try:
        sched = from_team_games(team_games, season)
    except ScheduleError:
        return None
    if sched.empty:
        return None
    day_one = pd.to_datetime(sched["game_date"]).min()
    cal = calendar_from_settings(None, day_one, cfg=cfg, games_per_day=games_per_day(sched))
    reg = [w for w in cal.weeks if w.kind == "regular"]
    po = [w for w in cal.weeks if w.kind == "playoff"]
    if not reg or not po:
        return None
    tgd = pd.concat([pd.DataFrame({"team_id": sched["home_team_id"], "d": pd.to_datetime(sched["game_date"])}),
                     pd.DataFrame({"team_id": sched["away_team_id"], "d": pd.to_datetime(sched["game_date"])})])
    rs, re_ = pd.Timestamp(reg[0].start), pd.Timestamp(reg[-1].end)
    ps, pe = pd.Timestamp(po[0].start), pd.Timestamp(po[-1].end)
    teams = sorted(set(tgd["team_id"]))
    team_reg = tgd[(tgd["d"] >= rs) & (tgd["d"] <= re_)].groupby("team_id").size().reindex(teams, fill_value=0).astype(float)
    team_po = tgd[(tgd["d"] >= ps) & (tgd["d"] <= pe)].groupby("team_id").size().reindex(teams, fill_value=0).astype(float)
    return {"season": season, "reg_start": rs, "reg_end": re_, "po_start": ps, "po_end": pe, "po_weeks": (po[0].week, po[-1].week),
            "team_reg": team_reg, "team_po": team_po, "mean_team_po": float(team_po.mean()),
            "usable": bool(team_po.mean() >= MIN_MEAN_PLAYOFF_TEAM_GAMES)}


def late_season_table(game_logs: pd.DataFrame, win: dict, scoring: Mapping[str, float]) -> pd.DataFrame:
    """Per player who played in ``win['season']``: games and FP in the regular / playoff weeks and the share gap ``G``.

    ``gap`` is NaN for players who played for more than one team that season (no single team denominator).
    """
    from src.backtest.actuals import game_fp

    s = win["season"]
    rows = game_logs[game_logs["season"] == s]
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "team_id", "n_teams", "gp_reg", "gp_po", "fp_po", "gap"])
    gfp = game_fp(game_logs, s, scoring)
    gfp["team_id"] = rows["team_id"].to_numpy()
    d = pd.to_datetime(gfp["game_date"])
    gfp["reg"] = (d >= win["reg_start"]) & (d <= win["reg_end"])
    gfp["po"] = (d >= win["po_start"]) & (d <= win["po_end"])
    g = gfp.groupby("player_id")
    out = pd.DataFrame({"team_id": g["team_id"].last(), "n_teams": g["team_id"].nunique(),
                        "gp_reg": g["reg"].sum().astype(float), "gp_po": g["po"].sum().astype(float),
                        "fp_po": gfp[gfp["po"]].groupby("player_id")["fp"].sum()}).fillna({"fp_po": 0.0})
    treg = out["team_id"].map(win["team_reg"]).to_numpy(float)
    tpo = out["team_id"].map(win["team_po"]).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where((treg > 0) & (tpo > 0), out["gp_po"] / tpo - out["gp_reg"] / treg, np.nan)
    out["gap"] = np.where(out["n_teams"].to_numpy() == 1, gap, np.nan)
    return out.reset_index()


# --------------------------------------------------------------------------- pre-season features

def team_win_pct(team_games: pd.DataFrame, season: str) -> pd.Series:
    tg = team_games[team_games["season"] == season]
    if tg.empty:
        return pd.Series(dtype=float)
    return (tg["pts_for"] > tg["pts_against"]).groupby(tg["team_id"]).mean()


def prior_features(history: History, scoring: Mapping[str, float], cfg: Mapping | None = None) -> pd.DataFrame:
    """Universe rows and features for ``history.target_season``, from ``history`` alone (seasons strictly before the target).

    Columns: ``player_id, team_id`` (last season's team), ``prior_gp, prior_mpg, prior_min_k, prior_fppg, team_winpct, age,
    prior_gap, star, bad_team, star_bad``.
    """
    from src.backtest.actuals import season_actuals
    from src.value.breakouts import last_season_team

    cols = ["player_id", "team_id", "prior_gp", "prior_mpg", "prior_min_k", "prior_fppg", "team_winpct", "age", "prior_gap",
            "star", "bad_team", "star_bad"]
    gl = history.game_logs
    if gl.empty:
        return pd.DataFrame(columns=cols)
    prev = max(gl["season"].unique(), key=season_start)
    if season_start(prev) != season_start(history.target_season) - 1:
        return pd.DataFrame(columns=cols)
    act = season_actuals(gl, prev, scoring)
    u = act[(act["gp"] >= MIN_GP) & (act["mpg"] >= MIN_MPG)].copy()
    if u.empty:
        return pd.DataFrame(columns=cols)
    team = last_season_team(history.player_season_bio)
    u["team_id"] = u["player_id"].map(team).fillna(u["team_id"]).astype("int64")
    u["prior_gp"], u["prior_mpg"], u["prior_min_k"], u["prior_fppg"] = u["gp"], u["mpg"], u["minutes"] / 1000.0, u["fppg"]
    u["team_winpct"] = u["team_id"].map(team_win_pct(history.team_games, prev)).fillna(0.5)
    bio = history.player_season_bio
    b = bio[bio["season"] == prev].drop_duplicates("player_id").set_index("player_id")["age_at_season_start"]
    u["age"] = (u["player_id"].map(b) + 1.0).fillna(27.0)
    win = season_windows(history.team_games, prev, cfg)
    if win is not None and win["usable"]:
        lt = late_season_table(gl, win, scoring).set_index("player_id")["gap"]
        u["prior_gap"] = u["player_id"].map(lt).fillna(0.0)
    else:
        u["prior_gap"] = 0.0
    u["star"] = (u["prior_fppg"] >= u["prior_fppg"].quantile(STAR_QUANTILE)).astype(float)
    u["bad_team"] = (u["team_winpct"] < BAD_TEAM_WINPCT).astype(float)
    u["star_bad"] = u["star"] * u["bad_team"]
    return u[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- frame

def season_games_for(team_games: pd.DataFrame, season: str) -> float:
    """The board's schedule length for ``season`` (``target_season_games`` over earlier seasons), as the board computes it."""
    from src.models.panel import season_lengths, target_season_games

    tg = team_games[team_games["season"].map(season_start) < season_start(season)]
    if tg.empty:
        return 82.0
    return float(target_season_games(season_lengths(SimpleNamespace(team_games=tg), None)))


def build_frame(tables: Mapping[str, pd.DataFrame], seasons: list[str], projections: pd.DataFrame, *,
                cfg: Mapping | None = None, log=print) -> tuple[pd.DataFrame, list[dict]]:
    """One row per universe player and season, with features, label, baseline expectation and realised playoff-week results.

    ``projections``: ``season, player_id, proj_gp, proj_fppg`` of the board model (walk-forward). Seasons before the first
    projected one still contribute training rows (they need no projection).
    """
    cfg = cfg or load_league()
    scoring = cfg["scoring"]
    gl, tg = tables["game_logs"], tables["team_games"]
    have = sorted(set(tg["season"]), key=season_start)                # the schedule decides which seasons have rows
    first = season_start(min(gl["season"], key=season_start)) + 1      # features need one earlier season of games
    row_seasons = [s for s in have if first <= season_start(s) <= season_start(seasons[-1])]
    frames, meta = [], []
    for S in row_seasons:
        win = season_windows(tg, S, cfg)
        if win is None:
            continue
        h = History.until(tables, S)
        f = prior_features(h, scoring, cfg)
        lt = late_season_table(gl, win, scoring).set_index("player_id")
        f["season"], f["s"] = S, season_start(S)
        f["usable"] = win["usable"]
        f["team_po"] = f["team_id"].map(win["team_po"]).fillna(0.0)
        f["gp_po"] = f["player_id"].map(lt["gp_po"]).fillna(0.0)
        f["fp_po"] = f["player_id"].map(lt["fp_po"]).fillna(0.0)
        played = f["player_id"].isin(lt.index)
        f["gap"] = np.where(played, f["player_id"].map(lt["gap"]), 0.0)     # no game at all: both shares 0
        f["season_games"] = season_games_for(tg, S)
        p = projections[projections["season"] == S].drop_duplicates("player_id").set_index("player_id")
        f["proj_gp"] = f["player_id"].map(p["proj_gp"]) if len(p) else np.nan
        f["proj_fppg"] = f["player_id"].map(p["proj_fppg"]) if len(p) else np.nan
        f["avail"] = np.clip(f["proj_gp"] / f["season_games"], 0.0, 1.0)
        f["e0"] = f["team_po"] * f["avail"]
        frames.append(f)
        meta.append({"season": S, "po_start": str(win["po_start"].date()), "po_end": str(win["po_end"].date()),
                     "po_weeks": list(win["po_weeks"]), "mean_team_po": round(win["mean_team_po"], 2), "usable": win["usable"],
                     "n_universe": int(len(f)), "n_projected": int(f["proj_gp"].notna().sum())})
        log(f"  {S}: playoff weeks {win['po_start'].date()}..{win['po_end'].date()}, mean team games {win['mean_team_po']:.1f}"
            f"{'' if win['usable'] else ' (not scored)'}; {len(f)} rotation veterans")
    return pd.concat(frames, ignore_index=True), meta


def training_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows a model for ``target`` may learn from: usable seasons strictly before it, with a defined label."""
    t = frame[(frame["s"] < season_start(target)) & frame["usable"] & frame["gap"].notna()]
    return t.dropna(subset=FEATURES)


def _ols(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = train[FEATURES].to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Z, train["gap"].to_numpy(float), rcond=None)
    return beta, mu, sd


def predict(frame: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Adds ``g_mean`` / ``g_ols`` (walk-forward predicted gap) and ``e1_mean`` / ``e1_ols`` to the target seasons' rows."""
    out = []
    for S in seasons:
        rows = frame[frame["season"] == S].copy()
        if rows.empty:
            continue
        tr = training_rows(frame, S)
        if len(tr) >= 30:
            rows["g_mean"] = float(tr["gap"].mean())
            beta, mu, sd = _ols(tr)
            Z = np.column_stack([np.ones(len(rows)), (rows[FEATURES].to_numpy(float) - mu) / sd])
            rows["g_ols"] = Z @ beta
            rows["n_train"] = len(tr)
        else:
            rows["g_mean"] = rows["g_ols"] = np.nan
            rows["n_train"] = len(tr)
        for k in ("mean", "ols"):
            rows[f"e1_{k}"] = rows["team_po"] * np.clip(rows["avail"] + rows[f"g_{k}"], 0.0, 1.0)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- scoring and decision

def scored(pred: pd.DataFrame) -> pd.DataFrame:
    """Rows that count: usable season, projected, with a trained candidate."""
    return pred[pred["usable"] & pred["proj_gp"].notna() & pred["g_mean"].notna()].reset_index(drop=True)


def _lift(d: pd.DataFrame, base: str, cand: str, actual: str, kind: str, scale: str | None = None) -> dict:
    """Paired, season-stratified bootstrap of the error reduction ``err(base) - err(cand)`` (positive = candidate better)."""
    f = (lambda x: x ** 2) if kind == "mse" else np.abs
    groups, per = [], {}
    for S, g in d.groupby("season", sort=True):
        k = g[scale].to_numpy(float) if scale else 1.0
        a = g[actual].to_numpy(float)
        e0, e1 = f(g[base].to_numpy(float) * k - a), f(g[cand].to_numpy(float) * k - a)
        groups.append((e0, e1))
        per[S] = float(e0.mean() - e1.mean())
    b = M.bootstrap_groups(groups, lambda gr: float(np.mean(gr[0] - gr[1])) if len(gr[0]) else float("nan"),
                           n_boot=N_BOOT, seed=0)
    wins = sum(v > 0 for v in per.values())
    return {"lift": b.estimate, "lo": b.lo, "hi": b.hi, "wins": int(wins), "n_seasons": len(per), "per_season": per}


def evaluate(pred: pd.DataFrame) -> dict:
    d = scored(pred)
    res: dict = {"n_rows": int(len(d)), "seasons": sorted(d["season"].unique().tolist(), key=season_start)}
    for k in ("mean", "ols"):
        res[k] = {
            "games_mse": _lift(d, "e0", f"e1_{k}", "gp_po", "mse"),
            "games_mae": _lift(d, "e0", f"e1_{k}", "gp_po", "mae"),
            "fp_mse": _lift(d, "e0", f"e1_{k}", "fp_po", "mse", scale="proj_fppg"),
            "fp_mae": _lift(d, "e0", f"e1_{k}", "fp_po", "mae", scale="proj_fppg"),
        }
    res["ols_over_mean_games_mse"] = _lift(d, "e1_mean", "e1_ols", "gp_po", "mse")
    res["descriptive"] = {
        "mean_actual_gp_po": float(d["gp_po"].mean()), "mean_e0": float(d["e0"].mean()),
        "mean_resid_e0": float((d["gp_po"] - d["e0"]).mean()), "mean_gap": float(d["gap"].mean()),
        "resid_by_season": {k: float(v) for k, v in (d["gp_po"] - d["e0"]).groupby(d["season"]).mean().items()},
    }
    tr = training_rows(pred, "2100-01")
    if len(tr) >= 30:
        beta, _mu, _sd = _ols(tr)
        res["ols_all_rows_std_coef"] = {"intercept": float(beta[0]), **{f: float(c) for f, c in zip(FEATURES, beta[1:])}}
    res["decision"] = decide(res)
    return res


def passes(c: dict) -> bool:
    """ADR 0039 D2: MSE-reduction CI above zero, MSE lower in >= 70% of scored seasons, MAE CI not entirely worse."""
    mse, mae = c["games_mse"], c["games_mae"]
    return bool(mse["lo"] > 0 and mse["wins"] >= np.ceil(MIN_WIN_SHARE * mse["n_seasons"]) and not mae["hi"] < 0)


def decide(res: dict) -> str:
    p_ols, p_mean = passes(res["ols"]), passes(res["mean"])
    if p_ols and not (p_mean and not res["ols_over_mean_games_mse"]["lo"] > 0):
        return "adopt_ols"
    if p_mean:
        return "adopt_mean"
    return "reject"


def render(res: dict, meta: list[dict]) -> str:
    L = ["# Late-season (fantasy-playoff week) availability study (ADR 0039)", "",
         f"Scored seasons: {', '.join(res['seasons'])}; {res['n_rows']} player-seasons. Decision: **{res['decision']}**.", "",
         "| season | playoff weeks | mean team games | scored | universe |", "|---|---|---:|---|---:|"]
    for m in meta:
        L.append(f"| {m['season']} | {m['po_start']} .. {m['po_end']} | {m['mean_team_po']:.1f} | {'yes' if m['usable'] else 'no'} | {m['n_universe']} |")
    L += ["", "| candidate | metric | lift (positive = better) | 95% CI | seasons won |", "|---|---|---:|---|---:|"]
    for k in ("mean", "ols"):
        for m in ("games_mse", "games_mae", "fp_mse", "fp_mae"):
            r = res[k][m]
            L.append(f"| {k} | {m} | {r['lift']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['wins']}/{r['n_seasons']} |")
    r = res["ols_over_mean_games_mse"]
    L.append(f"| ols over mean | games_mse | {r['lift']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['wins']}/{r['n_seasons']} |")
    dsc = res["descriptive"]
    L += ["", f"Mean playoff-week games {dsc['mean_actual_gp_po']:.2f} against the baseline's {dsc['mean_e0']:.2f} "
          f"(residual {dsc['mean_resid_e0']:+.2f}); mean share gap {dsc['mean_gap']:+.3f}.",
          "Residual by season: " + ", ".join(f"{k} {v:+.2f}" for k, v in dsc["resid_by_season"].items())]
    if "ols_all_rows_std_coef" in res:
        L.append("OLS on all scored-season rows (standardised): " + ", ".join(f"{k} {v:+.4f}" for k, v in res["ols_all_rows_std_coef"].items()))
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI

def board_projections(tables: Mapping[str, pd.DataFrame], seasons: list[str], cache_dir: Path | None) -> pd.DataFrame:
    from src.backtest.harness import walk_forward
    from src.backtest.runner import resolve_projector

    # The board model has no ``fingerprint``, so the harness cache cannot be used; ``cache_dir`` keeps this run's projections in one
    # file instead (delete it after any model change).
    path = Path(cache_dir) / f"{BOARD_MODEL}__{seasons[0]}_{seasons[-1]}.parquet" if cache_dir else None
    if path is not None and path.exists():
        return pd.read_parquet(path)
    res = walk_forward(tables, resolve_projector(BOARD_MODEL), seasons)
    p = res.players
    out = p.loc[p["projected"], ["season", "player_id", "proj_gp", "proj_fppg"]].reset_index(drop=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def main(argv=None) -> int:
    from src.backtest.harness import expand_seasons
    from src.backtest.runner import load_run_tables

    ap = argparse.ArgumentParser(prog="python -m src.backtest.playoff_availability", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--cache-dir", type=Path, default=None, help="cache the board model's walk-forward projections here")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    seasons = expand_seasons(args.seasons)
    tables = load_run_tables(SimpleNamespace(synthetic=False))
    proj = board_projections(tables, seasons, args.cache_dir)
    frame, meta = build_frame({k: tables[k] for k in HISTORY_TABLES}, seasons, proj)
    pred = predict(frame, seasons)
    res = evaluate(pred)
    text = render(res, meta)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "playoff_availability.md").write_text(text, encoding="utf-8")
        (args.out / "summary.json").write_text(json.dumps({"meta": meta, **res}, indent=1, default=float), encoding="utf-8")
        pred.to_csv(args.out / "frame.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
