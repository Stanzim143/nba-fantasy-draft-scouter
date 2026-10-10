"""Teammate-out study and pre-registered evaluation (``docs/research/teammate-out-study.md``).

    python -m src.inseason.teammate_out_eval --seasons 2016-17:2025-26 --out DIR      # build per-season panels
    python -m src.inseason.teammate_out_eval --analyze --out DIR                       # fit on early seasons, evaluate later

Panels are written one season at a time (``forecast_<season>.parquet``, ``study_<season>.parquet``) so a run can be split
across processes and resumed; ``--analyze`` only reads them.

Study panel (descriptive, per player-game): the player's minutes and FP against his own last 10 appearances, and the
trailing minutes / usage of the team's *usual* rotation players missing from this game. "Usual" is decided from the
team's previous 10 games only (6+ appearances, 15+ mpg in them). Confounder columns: blowout (final margin 20+), second
night of a back-to-back, the team's last 10 games of the season.

Forecast panel (the pre-registered test): for every game day ``d`` the ROS projection is built as of ``d - 1``; the
teammate-out adjustment (``teammate_out.teammate_out_adjustment``, beta = 1) is computed for two absence sources known
before tip-off: the NBA 5 PM injury report dated ``d`` (R) and game-log absence as of ``d - 1`` (G). The target is the
FP each player scored on ``d``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import bootstrap_groups
from src.contracts import HISTORY_TABLES, History, season_start, seasons_between
from src.inseason.ros import build_ros
from src.inseason.schedule import from_team_games
from src.inseason.signals import absent_players
from src.inseason.teammate_out import status_absences, teammate_out_adjustment
from src.value.frame import fantasy_points_frame
from src.value.league import load_league
from src.value.positions import position_group

FIT = ("2016-17", "2017-18", "2018-19", "2019-20")
EVAL = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
MIN_OWN_GAMES = 5
WINDOW, MIN_APP, USUAL_MPG = 10, 6, 15.0
N_BOOT, SEED = 2000, 0


# --------------------------------------------------------------------------- study panel

def study_panel(gl: pd.DataFrame, tg: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """One row per player-game with a 5+ appearance baseline. ``gl`` has ``fp``; ``groups``: player_id -> G/F/C/U."""
    gl = gl.assign(usg=gl["fga"] + 0.44 * gl["fta"] + gl["tov"])
    out = []
    for team, tgt in tg.groupby("team_id"):
        games = tgt.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
        gids = games["game_id"].to_numpy()
        t = gl[gl["team_id"] == team]
        if t.empty:
            continue
        piv = {c: t.pivot_table(index="game_id", columns="player_id", values=c, aggfunc="sum").reindex(gids)
               for c in ("min", "fp", "usg")}
        m = piv["min"]
        app = m.notna().astype(float)
        r_app = app.rolling(WINDOW, min_periods=WINDOW).sum().shift(1)
        trail_mpg = m.fillna(0).rolling(WINDOW, min_periods=WINDOW).sum().shift(1) / r_app
        trail_usg = piv["usg"].fillna(0).rolling(WINDOW, min_periods=WINDOW).sum().shift(1) / r_app
        usual = (r_app >= MIN_APP) & (trail_mpg >= USUAL_MPG)
        missing = usual & m.isna()
        miss_min = trail_mpg.where(missing)
        team_usg = trail_usg.where(usual).sum(axis=1)
        miss_usg_share = trail_usg.where(missing).sum(axis=1) / team_usg.replace(0, np.nan)
        col_group = groups.reindex(m.columns).fillna("U").to_numpy()
        miss_by_group = {g: miss_min.loc[:, col_group == g].sum(axis=1) for g in ("G", "F", "C", "U")}
        miss_total = miss_min.sum(axis=1)
        dates = pd.to_datetime(games["game_date"])
        b2b = (dates.diff().dt.days == 1).to_numpy()
        margin = (games["pts_for"] - games["pts_against"]).abs().to_numpy()
        late = np.arange(len(games)) >= len(games) - 10
        for j, pid in enumerate(m.columns):
            s_min = m[pid].dropna()
            if len(s_min) <= MIN_OWN_GAMES:
                continue
            s_fp = piv["fp"][pid].reindex(s_min.index)
            base_min = s_min.rolling(WINDOW, min_periods=MIN_OWN_GAMES).mean().shift(1)
            base_fp = s_fp.rolling(WINDOW, min_periods=MIN_OWN_GAMES).mean().shift(1)
            ok = base_min.notna()
            if not ok.any():
                continue
            idx = pd.Index(gids).get_indexer(s_min.index[ok])
            g = col_group[j]
            same = miss_by_group[g].to_numpy()[idx] if g != "U" else np.zeros(len(idx))
            out.append(pd.DataFrame({
                "team_id": team, "player_id": pid, "game_id": s_min.index[ok], "game_date": dates.to_numpy()[idx],
                "min": s_min[ok].to_numpy(), "fp": s_fp[ok].to_numpy(), "base_min": base_min[ok].to_numpy(),
                "base_fp": base_fp[ok].to_numpy(), "miss_min": miss_total.to_numpy()[idx], "miss_same": same,
                "miss_usg_share": miss_usg_share.fillna(0).to_numpy()[idx], "blowout": margin[idx] >= 20,
                "b2b": b2b[idx], "late": late[idx]}))
    if not out:
        return pd.DataFrame()
    p = pd.concat(out, ignore_index=True)
    p["d_min"] = p["min"] - p["base_min"]
    p["d_fp"] = p["fp"] - p["base_fp"]
    p["miss_other"] = p["miss_min"] - p["miss_same"]
    return p


# --------------------------------------------------------------------------- forecast panel

def _report_status(reports: pd.DataFrame | None, season: str) -> dict:
    """game date -> Series(player_id -> 'OUT' / other) from the 5 PM NBA injury report."""
    if reports is None or reports.empty:
        return {}
    r = reports[(reports["season"] == season) & reports["player_id"].notna()]
    r = r.sort_values("report_ts").drop_duplicates(["game_date", "player_id"], keep="last")
    return {pd.Timestamp(d).normalize(): pd.Series(g["status"].str.upper().to_numpy(), index=g["player_id"].astype("int64"))
            for d, g in r.groupby("game_date")}


def forecast_rows(s_tables: dict, season: str, day: pd.Timestamp, *, prior, cfg, schedule, season_games, positions,
                  statuses: pd.Series | None) -> pd.DataFrame:
    """Rows for every player who played on ``day``: actual FP, the baseline and the adjustment of each source.

    Only rows dated ``<= day - 1`` feed the projection and the absences; ``day``'s rows are read for the target only.
    """
    as_of = day - pd.Timedelta(days=1)
    gl_all = s_tables["game_logs"]
    dts = pd.to_datetime(gl_all["game_date"])
    gl = gl_all[dts <= as_of]
    tgd = pd.to_datetime(s_tables["team_games"]["game_date"])
    tg = s_tables["team_games"][tgd <= as_of]
    target = gl_all[dts == day]
    if gl.empty or target.empty:
        return pd.DataFrame()
    ros = build_ros(s_tables, season, as_of, prior=prior, cfg=cfg, schedule=schedule, season_games=season_games,
                    with_value=False)
    teams_today = set(s_tables["team_games"].loc[tgd == day, "team_id"])
    gp = gl.groupby("player_id").size()
    t = target[target["player_id"].map(gp).fillna(0) >= MIN_OWN_GAMES]
    if t.empty:
        return pd.DataFrame()
    r = ros.drop_duplicates("player_id").set_index("player_id")
    out = pd.DataFrame({"season": season, "date": day, "team_id": t["team_id"].to_numpy(),
                        "player_id": t["player_id"].to_numpy(), "fp": t["fp"].to_numpy(),
                        "ros_fppg": r["ros_fppg"].reindex(t["player_id"]).to_numpy(float),
                        "ros_mpg": r["ros_mpg"].reindex(t["player_id"]).to_numpy(float),
                        "has_report": statuses is not None})
    sources = {"G": absent_players(gl, tg)}
    if statuses is not None:
        sources["R"] = status_absences(gl, statuses)
    for name, ab in sources.items():
        ab = ab[ab["team_id"].isin(teams_today)] if len(ab) else ab
        adj = teammate_out_adjustment(gl, tg, ab, positions, ros, beta=1.0) if len(ab) else pd.DataFrame()
        a = adj.set_index("player_id")["adj_fppg"] if len(adj) else pd.Series(dtype=float)
        # the adjustment is for the team the player is on today (a traded player's old team does not count)
        same_team = adj.set_index("player_id")["team_id"].reindex(out["player_id"]).to_numpy() == out["team_id"].to_numpy() \
            if len(adj) else np.zeros(len(out), bool)
        out[f"x_{name}"] = np.where(same_team, a.reindex(out["player_id"]).fillna(0.0).to_numpy(), 0.0)
        out[f"n_out_{name}"] = len(ab)
    if "x_R" not in out:
        out["x_R"] = np.nan
    return out


def build_season(tables: dict, season: str, out_dir: Path, *, reports=None, cfg=None, model="baseline",
                 log=lambda s: None) -> None:
    from src.models.panel import season_lengths, target_season_games
    from src.models.registry import get_projector

    cfg = cfg or load_league()
    scoring = cfg["scoring"]
    hist = History.until(tables, season)
    hist.assert_no_future()
    prior = get_projector(model).project(hist)
    season_games = float(target_season_games(season_lengths(hist, None)))
    del hist
    gl = tables["game_logs"][tables["game_logs"]["season"] == season].copy()
    gl["fp"] = fantasy_points_frame(gl, scoring)
    gl["game_date"] = pd.to_datetime(gl["game_date"])
    tg = tables["team_games"][tables["team_games"]["season"] == season].copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    s_tables = {"game_logs": gl, "team_games": tg, "players": tables["players"],
                "player_season_bio": tables["player_season_bio"]}
    positions = tables["players"].drop_duplicates("player_id").set_index("player_id")["position"]
    groups = pd.Series([position_group(p) for p in positions], index=positions.index)
    study_panel(gl, tg, groups).to_parquet(out_dir / f"study_{season}.parquet", index=False)
    log(f"{season}: study panel written")
    schedule = from_team_games(tables["team_games"], season)
    status = _report_status(reports, season)
    rows = []
    for day in sorted(tg["game_date"].dt.normalize().unique()):
        day = pd.Timestamp(day)
        rows.append(forecast_rows(s_tables, season, day, prior=prior, cfg=cfg, schedule=schedule,
                                  season_games=season_games, positions=positions, statuses=status.get(day)))
    fc = pd.concat([r for r in rows if len(r)], ignore_index=True)
    fc.to_parquet(out_dir / f"forecast_{season}.parquet", index=False)
    log(f"{season}: forecast panel {len(fc)} rows, affected G {(fc['x_G'] > 0).sum()}, R {(fc['x_R'] > 0).sum()}")


# --------------------------------------------------------------------------- analysis

def fit_beta(fc: pd.DataFrame, col: str) -> float:
    a = fc[fc[col] > 0]
    x, r = a[col].to_numpy(float), (a["fp"] - a["ros_fppg"]).to_numpy(float)
    return max(0.0, float((x * r).sum() / (x * x).sum())) if len(a) else 0.0


def lift(fc: pd.DataFrame, col: str, beta: float, seasons, *, n_boot=N_BOOT, seed=SEED) -> dict:
    """MAE(baseline) - MAE(baseline + beta x) over affected player-games; cluster bootstrap over (team, date)."""
    groups, per_season, n = [], {}, 0
    for s in seasons:
        a = fc[(fc["season"] == s) & (fc[col] > 0)]
        if a.empty:
            continue
        d = np.abs(a["fp"] - a["ros_fppg"]) - np.abs(a["fp"] - a["ros_fppg"] - beta * a[col])
        cl = a.assign(d=d.to_numpy()).groupby(["team_id", "date"])["d"].agg(["sum", "size"])
        groups.append((cl["sum"].to_numpy(float), cl["size"].to_numpy(float)))
        per_season[s] = float(d.mean())
        n += len(a)
    if not groups:
        return {"lift": float("nan"), "lo": float("nan"), "hi": float("nan"), "wins": 0, "seasons": 0, "n": 0,
                "per_season": {}}
    b = bootstrap_groups(groups, lambda g: float(g[0].sum() / g[1].sum()), n_boot=n_boot, seed=seed)
    wins = sum(v > 0 for v in per_season.values())
    return {"lift": b.estimate, "lo": b.lo, "hi": b.hi, "wins": wins, "seasons": len(per_season), "n": n,
            "per_season": per_season}


def residual_ci(fc: pd.DataFrame, col: str, seasons) -> dict:
    """Mean of actual - baseline over affected player-games (is there a positive effect at all?)."""
    groups = []
    for s in seasons:
        a = fc[(fc["season"] == s) & (fc[col] > 0)]
        if len(a):
            cl = a.assign(r=a["fp"] - a["ros_fppg"]).groupby(["team_id", "date"])["r"].agg(["sum", "size"])
            groups.append((cl["sum"].to_numpy(float), cl["size"].to_numpy(float)))
    if not groups:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    b = bootstrap_groups(groups, lambda g: float(g[0].sum() / g[1].sum()), n_boot=N_BOOT, seed=SEED)
    return {"mean": b.estimate, "lo": b.lo, "hi": b.hi}


def week_frame(fc: pd.DataFrame) -> pd.DataFrame:
    """Secondary target: as of each Sunday, mean FP over the player's games Monday..Sunday; the adjustment is Monday's."""
    m = fc[fc["date"].dt.weekday == 0][["season", "date", "team_id", "player_id", "ros_fppg", "x_G", "x_R"]]
    f = fc[["player_id", "date", "fp"]]
    j = m.merge(f.rename(columns={"date": "d2"}), on="player_id")
    j = j[(j["d2"] >= j["date"]) & (j["d2"] < j["date"] + pd.Timedelta(days=7))]
    return j.groupby(["season", "date", "team_id", "player_id"], as_index=False).agg(
        fp=("fp", "mean"), ros_fppg=("ros_fppg", "first"), x_G=("x_G", "first"), x_R=("x_R", "first"))


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]


def study_summary(st: pd.DataFrame) -> list[str]:
    lines = []
    fit, ev = st[st["season"].isin(FIT)], st[st["season"].isin(EVAL)]
    bins = pd.cut(st["miss_min"], [-0.1, 0.1, 15, 30, 50, 500], labels=["0", "0-15", "15-30", "30-50", "50+"])
    tab = st.assign(bin=bins, same=st["miss_same"] > 0).groupby(["bin"], observed=True).agg(
        n=("d_min", "size"), d_min=("d_min", "mean"), d_fp=("d_fp", "mean"))
    lines.append("Mean change vs own last-10 baseline, by trailing minutes of usual rotation teammates missing (all seasons):")
    lines.append(tab.to_string(float_format=lambda v: f"{v:.2f}"))
    clean = st[~st["blowout"] & ~st["b2b"] & ~st["late"]]
    tc = clean.assign(bin=pd.cut(clean["miss_min"], [-0.1, 0.1, 15, 30, 50, 500],
                                 labels=["0", "0-15", "15-30", "30-50", "50+"])).groupby("bin", observed=True).agg(
        n=("d_min", "size"), d_min=("d_min", "mean"), d_fp=("d_fp", "mean"))
    lines.append("\nSame, excluding blowouts, back-to-back second nights and the last 10 team games:")
    lines.append(tc.to_string(float_format=lambda v: f"{v:.2f}"))
    feats_full = ["miss_same", "miss_other", "head_x_miss", "base_min", "blowout", "b2b", "late"]
    feats_ctrl = ["base_min", "blowout", "b2b", "late"]

    def X(d, cols):
        d = d.assign(head_x_miss=d["miss_min"] * (36 - d["base_min"]).clip(lower=0) / 36)
        return d[cols].astype(float).to_numpy()
    for target in ("d_min", "d_fp"):
        bf = _ols(X(fit, feats_full), fit[target].to_numpy())
        bc = _ols(X(fit, feats_ctrl), fit[target].to_numpy())
        pf = np.column_stack([np.ones(len(ev)), X(ev, feats_full)]) @ bf
        pc = np.column_stack([np.ones(len(ev)), X(ev, feats_ctrl)]) @ bc
        y = ev[target].to_numpy()
        aff = (ev["miss_min"] > 0).to_numpy()
        coefs = ", ".join(f"{c} {v:+.3f}" for c, v in zip(["const"] + feats_full, bf))
        lines.append(f"\nOLS {target} fitted on {FIT[0]}..{FIT[-1]}: {coefs}")
        lines.append(f"  eval {EVAL[0]}..{EVAL[-1]} MAE: controls only {np.abs(y - pc).mean():.3f}, + missing teammates "
                     f"{np.abs(y - pf).mean():.3f}; on games with a teammate missing {np.abs(y - pc)[aff].mean():.3f} -> "
                     f"{np.abs(y - pf)[aff].mean():.3f} (n={int(aff.sum())})")
    return lines


def analyze(out_dir: Path) -> list[str]:
    fc = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("forecast_*.parquet"))], ignore_index=True)
    st = pd.concat([pd.read_parquet(p).assign(season=p.stem.split("_", 1)[1])
                    for p in sorted(out_dir.glob("study_*.parquet"))], ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"])
    lines = study_summary(st)
    wk = week_frame(fc)
    for col, label in (("x_R", "R (5 PM injury report OUT)"), ("x_G", "G (game-log absence)")):
        d = fc[fc[col].notna()]
        beta = fit_beta(d[d["season"].isin(FIT)], col)
        lines.append(f"\n== {label}: beta fitted on {sorted(set(d['season']) & set(FIT))} = {beta:.3f}")
        aff_fit = d[d["season"].isin(FIT) & (d[col] > 0)]
        lines.append(f"  fit: affected player-games {len(aff_fit)}, mean raw adjustment {aff_fit[col].mean():.2f} FP, "
                     f"mean residual {(aff_fit['fp'] - aff_fit['ros_fppg']).mean():+.2f}")
        res = lift(d, col, beta, EVAL)
        rc = residual_ci(d, col, EVAL)
        aff = d[d["season"].isin(EVAL) & (d[col] > 0)]
        lines.append(f"  EVAL next-game: affected {res['n']} player-games ({res['n'] / max(1, d['season'].isin(EVAL).sum()):.1%} "
                     f"of all), MAE lift {res['lift']:+.4f} FP [{res['lo']:+.4f}, {res['hi']:+.4f}], positive in "
                     f"{res['wins']}/{res['seasons']} seasons")
        lines.append("  per season: " + ", ".join(f"{s} {v:+.3f}" for s, v in res["per_season"].items()))
        lines.append(f"  mean residual actual-baseline (affected) {rc['mean']:+.3f} [{rc['lo']:+.3f}, {rc['hi']:+.3f}]; mean "
                     f"adjustment {beta * aff[col].mean():.3f}; baseline MAE {np.abs(aff['fp'] - aff['ros_fppg']).mean():.3f}")
        raw = lift(d, col, 1.0, EVAL)
        lines.append(f"  (beta = 1, the rule as is: lift {raw['lift']:+.4f} [{raw['lo']:+.4f}, {raw['hi']:+.4f}], "
                     f"{raw['wins']}/{raw['seasons']})")
        w = wk[wk[col].notna()]
        wl = lift(w.assign(date=w["date"]), col, beta, EVAL)
        lines.append(f"  EVAL next-week mean FP (secondary): lift {wl['lift']:+.4f} [{wl['lo']:+.4f}, {wl['hi']:+.4f}], "
                     f"{wl['wins']}/{wl['seasons']} seasons, n={wl['n']}")
        passed = res["lift"] > 0 and res["lo"] > 0 and res["wins"] >= 5
        lines.append(f"  PRE-REGISTERED RULE: {'PASS' if passed else 'FAIL'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.teammate_out_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        print("\n".join(analyze(args.out)))
        return 0
    from src.store import load_tables

    a, _, b = args.seasons.partition(":")
    seasons = seasons_between(season_start(a), season_start(b or a))
    tables = dict(load_tables(HISTORY_TABLES))
    try:
        from src.ingest.nba_injury_reports import read_injury_reports

        reports = read_injury_reports()
    except FileNotFoundError:
        reports = None
    for s in seasons:
        if args.skip_existing and (args.out / f"forecast_{s}.parquet").exists():
            continue
        build_season(tables, s, args.out, reports=reports, log=lambda m: print(m, file=sys.stderr, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
