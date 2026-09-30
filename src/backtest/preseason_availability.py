"""Does preseason absence predict a thin regular season? A walk-forward test that CANNOT answer it (ADR 0016, gap 4).

    python -m src.backtest.preseason_availability [--seasons 2016-17:2025-26] [--out reports/preseason_availability]

The question behind "flag preseason injuries": a rotation veteran whose team played preseason games and who played none
(``dnp_all``) or fewer than half (``partial``) of them. For each target season the baseline projects from earlier data
only; every rotation veteran is flagged from the season's own preseason box scores (the draft happens after most of
them); the outcome is his regular-season games minus ``proj_gp``, zero games included. The haircut a flag would earn is
fitted on **earlier seasons only** and applied to the flagged rows; the paired MAE change in games and in total fantasy
points, against the unadjusted baseline, is what is reported.

**Why the result is not usable.** The history has no roster snapshots, so "rotation veteran" cannot be conditioned on
"is under contract for this season". A veteran who retired, signed abroad or is an unsigned free agent looks exactly like
an injured one in a box score (no preseason game, no regular-season game): about 78% of the ``dnp_all`` group played no
game at all, evidently mostly players who were never on a roster. The measured haircut is therefore an artefact and is
**not** applied anywhere (``PRESEASON_HAIRCUT`` stays 0). Live, the flag is only ever shown for players on the current
roster snapshot, where the confound is gone, but no history exists to size its effect; ``roster_snapshots`` accumulates daily
from 2026-09-24, so next year it can be tested properly. The script is kept so that test is one command.

Limits, stated because they decide how far the result travels: preseason absence mixes injury with rest and roster
churn (nothing separates them in a box score); the flag is measured with the whole preseason, while a draft on
Oct 17 or 18 sees most but not all of it; the team used is the one he ended last season on (a traded player's new
team's game count is not known here), and any appearance for any team counts as appearing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import HISTORY_TABLES, History, season_start
from src.features.risk import preseason_flags, preseason_usage
from src.models.baseline import BaselineProjector
from src.value.league import load_league

DEFAULT_SEASONS = "2016-17:2025-26"
BOOT = 2000


def load_inputs(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    from src.ingest import nba_offseason
    from src.store import load_tables

    t = dict(load_tables(HISTORY_TABLES, base=data_dir))
    t["offseason_logs"] = nba_offseason.read_offseason_logs(data_dir)
    t["offseason_team_games"] = nba_offseason.read_offseason_team_games(data_dir)
    return t


def evaluate(tables: dict[str, pd.DataFrame], seasons: list[str], *, log=print) -> pd.DataFrame:
    from src.value.breakouts import last_season_team

    scoring = load_league()["scoring"]
    gl = tables["game_logs"]
    actual = gl.groupby([gl["season"].map(season_start), "player_id"]).size().rename("gp")
    frames = []
    for S in seasons:
        sy = season_start(S)
        h = History.until(tables, S)
        proj = BaselineProjector(scoring).project(h)
        pl_games, tm_games = preseason_usage(tables["offseason_logs"], tables["offseason_team_games"], S)
        team = last_season_team(h.player_season_bio)
        f = preseason_flags(proj["player_id"], team, proj["proj_mpg"].to_numpy(), proj["n_hist_seasons"].to_numpy(), pl_games, tm_games)
        d = proj[["player_id", "proj_gp", "proj_fppg", "proj_mpg", "n_hist_seasons"]].merge(f, on="player_id")
        d["actual_gp"] = actual.xs(sy, level=0).reindex(d["player_id"]).fillna(0).to_numpy() if sy in actual.index.get_level_values(0) else 0.0
        d["season"] = S
        d["s"] = sy
        frames.append(d)
        log(f"  {S}: {int((d['pre_flag'] == 'dnp_all').sum())} dnp_all, {int((d['pre_flag'] == 'partial').sum())} partial of {int((d['proj_mpg'] >= 15).sum())} rotation veterans")
    return pd.concat(frames, ignore_index=True)


def _boot(x: np.ndarray) -> list[float]:
    if len(x) < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarise(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["resid_gp"] = d["actual_gp"] - d["proj_gp"]
    out: dict = {"seasons": sorted(d["season"].unique()), "n_rotation_veterans": int((d["proj_mpg"] >= 15).sum())}
    unflagged = d[(d["pre_flag"] == "") & (d["proj_mpg"] >= 15) & (d["team_pre_games"] >= 3)]
    out["unflagged"] = {"n": int(len(unflagged)), "mean_resid_gp": float(unflagged["resid_gp"].mean())}
    for flag in ("dnp_all", "partial"):
        sub = d[d["pre_flag"] == flag]
        if sub.empty:
            continue
        # walk-forward haircut: mean residual of the same flag in strictly earlier seasons, as a share of projected games
        adj = np.zeros(len(sub))
        for i, s in enumerate(sub["s"]):
            prev = d[(d["pre_flag"] == flag) & (d["s"] < s)]
            adj[i] = float(prev["resid_gp"].mean()) if len(prev) >= 8 else 0.0
        gp_adj = np.clip(sub["proj_gp"].to_numpy(float) + adj, 0, 82)
        e0 = np.abs(sub["actual_gp"].to_numpy(float) - sub["proj_gp"].to_numpy(float))
        e1 = np.abs(sub["actual_gp"].to_numpy(float) - gp_adj)
        t0 = np.abs(sub["actual_gp"] * 0 + (sub["proj_gp"] - sub["actual_gp"]) * sub["proj_fppg"])
        t1 = np.abs((gp_adj - sub["actual_gp"].to_numpy(float)) * sub["proj_fppg"].to_numpy(float))
        r = sub["resid_gp"].to_numpy(float)
        out[flag] = {
            "n": int(len(sub)), "mean_proj_gp": float(sub["proj_gp"].mean()), "mean_actual_gp": float(sub["actual_gp"].mean()),
            "mean_resid_gp": float(r.mean()), "resid_ci95": _boot(r),
            "share_played_zero": float((sub["actual_gp"] == 0).mean()),
            "resid_gp_by_season": {k: float(v) for k, v in sub.groupby("season")["resid_gp"].mean().items()},
            "walk_forward": {"mae_gp_baseline": float(e0.mean()), "mae_gp_adjusted": float(e1.mean()),
                             "mae_gp_change": float((e1 - e0).mean()), "mae_gp_change_ci95": _boot(e1 - e0),
                             "mae_total_fp_change": float((t1 - t0).mean()), "mae_total_fp_change_ci95": _boot((t1 - t0).to_numpy(float))},
        }
    return out


def render(s: dict) -> str:
    L = ["# Preseason absence and the regular season", "",
         "> **Not a validation.** The history cannot tell an injured veteran from one who is not under contract (no historical roster "
         "snapshots), so the group below is dominated by players who left the league. No haircut is derived from it.", "",
         f"Seasons {s['seasons'][0]} to {s['seasons'][-1]}.",
         f"Unflagged rotation veterans (team played 3+ preseason games): n={s['unflagged']['n']}, mean actual-minus-projected games {s['unflagged']['mean_resid_gp']:+.1f}.", ""]
    for flag in ("dnp_all", "partial"):
        r = s.get(flag)
        if not r:
            continue
        w = r["walk_forward"]
        L += [f"## {flag} (n={r['n']})", "",
              f"* projected {r['mean_proj_gp']:.1f} games, played {r['mean_actual_gp']:.1f}; residual {r['mean_resid_gp']:+.1f} games "
              f"(95% CI {r['resid_ci95'][0]:+.1f} to {r['resid_ci95'][1]:+.1f}); {r['share_played_zero']:.0%} played no game at all.",
              f"* walk-forward haircut: games MAE {w['mae_gp_baseline']:.2f} -> {w['mae_gp_adjusted']:.2f} "
              f"({w['mae_gp_change']:+.2f}, 95% CI {w['mae_gp_change_ci95'][0]:+.2f} to {w['mae_gp_change_ci95'][1]:+.2f}); "
              f"total FP MAE change {w['mae_total_fp_change']:+.1f} ({w['mae_total_fp_change_ci95'][0]:+.1f} to {w['mae_total_fp_change_ci95'][1]:+.1f}).",
              "* residual by season: " + ", ".join(f"{k} {v:+.1f}" for k, v in r["resid_gp_by_season"].items()), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    from src.backtest.harness import expand_seasons

    ap = argparse.ArgumentParser(prog="python -m src.backtest.preseason_availability", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    d = evaluate(load_inputs(args.data_dir), expand_seasons(args.seasons))
    s = summarise(d)
    text = render(s)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "preseason_availability.md").write_text(text, encoding="utf-8")
        (args.out / "summary.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
        d.to_csv(args.out / "frame.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
