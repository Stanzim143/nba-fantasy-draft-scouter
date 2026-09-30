"""Does origin information add lift to the rookie prior? A walk-forward test (ADR 0016, gap 3).

    python -m src.backtest.rookie_origin [--seasons 2018-19:2025-26] [--out reports/rookie_origin]

The rookie prior uses draft slot, position group and age. Three legitimate, machine-readable, point-in-time-safe
proxies for "what he did before the NBA" exist in the sources the project already uses:

* ``intl``        born outside the USA (``playerindex`` COUNTRY);
* ``non_college`` the organisation the index lists is not a college (a pro club, a prep school, none);
* ``combine``     draft-combine measurements (``draftcombinestats``): wingspan minus height and max vertical, with an
  attendance indicator (about 60% attend; a non-attendee gets the attendees' training mean, so the indicator carries
  only the absence).

Variants, all fitted on **earlier seasons' rookies** and scored on the target season's rookies who are in ``players``
(same evaluation universe as the main backtest: a rookie who never plays scores zero total fantasy points):

* ``base``    the same regression as the shipped prior, refitted on the three direct targets so that the comparison
              isolates the extra features, not the implementation;
* ``origin``  ``base`` + ``intl`` + ``non_college``;
* ``combine`` ``base`` + the combine features;
* ``both``    everything.

Targets: minutes per game (weighted by games), games-played fraction, and fantasy points per game (weighted by games);
predicted total = FP/game x fraction x schedule. Paired bootstrap confidence intervals on the per-rookie absolute-error
difference against ``base`` (rookies are resampled independently, which understates the uncertainty: they share seasons).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.contracts import HISTORY_TABLES, History, season_start
from src.models.baseline import BaselineProjector
from src.models.rookie_origin import RIDGE, VARIANTS, predict_variants
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

DEFAULT_SEASONS = "2018-19:2025-26"
BOOT = 2000


def load_inputs(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    from src.ingest import nba_profiles
    from src.store import load_tables

    t = dict(load_tables(HISTORY_TABLES, base=data_dir))
    t["player_profiles"] = nba_profiles.read_profiles(data_dir)
    try:
        t["draft_combine"] = nba_profiles.read_combine(data_dir)
    except FileNotFoundError:
        t["draft_combine"] = pd.DataFrame(columns=nba_profiles.COMBINE_COLUMNS)
    return t


def evaluate(tables: dict[str, pd.DataFrame], seasons: list[str], *, ridge: float = RIDGE, log=print) -> pd.DataFrame:
    scoring = load_league()["scoring"]
    gl = tables["game_logs"]
    fp = fantasy_points_frame(gl, scoring).to_numpy()
    act = pd.DataFrame({"s": gl["season"].map(season_start).to_numpy(), "player_id": gl["player_id"].to_numpy(), "fp": fp,
                        "min": gl["min"].to_numpy(float)}).groupby(["s", "player_id"]).agg(
        gp=("fp", "size"), tot=("fp", "sum"), mpg=("min", "mean")).reset_index()
    frames = []
    for S in seasons:
        sy = season_start(S)
        h = History.until(tables, S)
        fitted = BaselineProjector(scoring).fit(h)
        out = predict_variants(fitted, h, tables["player_profiles"], tables["draft_combine"], scoring, ridge=ridge)
        if out is None:
            log(f"  {S}: too few training rookies, skipped")
            continue
        L = float(fitted.season_games)
        out["season"] = S
        for v in VARIANTS:
            out[f"{v}_total"] = out[f"{v}_fppg"] * out[f"{v}_f"] * L
        a = act[act["s"] == sy].set_index("player_id")
        rids = out["player_id"].to_numpy()
        out["actual_gp"] = a["gp"].reindex(rids).fillna(0.0).to_numpy()
        out["actual_total"] = a["tot"].reindex(rids).fillna(0.0).to_numpy()
        out["actual_fppg"] = np.where(out["actual_gp"] > 0, out["actual_total"] / out["actual_gp"].where(out["actual_gp"] > 0, 1.0), np.nan)
        frames.append(out)
        log(f"  {S}: {len(out)} rookies")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ci(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    m = diff[rng.integers(0, len(diff), size=(BOOT, len(diff)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarise(d: pd.DataFrame) -> dict:
    y = d["actual_total"].to_numpy(float)
    played = (d["actual_gp"] > 0).to_numpy()
    out: dict = {"n": int(len(d)), "n_played": int(played.sum()), "seasons": sorted(d["season"].unique()),
                 "n_intl": int(d["intl"].sum()), "combine_coverage": float(d["has_combine"].mean())}
    for v in VARIANTS:
        p = d[f"{v}_total"].to_numpy(float)
        fe = np.abs(d.loc[played, f"{v}_fppg"].to_numpy(float) - d.loc[played, "actual_fppg"].to_numpy(float))
        r = {"mae_total": float(np.abs(p - y).mean()), "spearman_total": float(spearmanr(p, y)[0]), "fppg_mae_played": float(fe.mean())}
        if v != "base":
            diff = np.abs(p - y) - np.abs(d["base_total"].to_numpy(float) - y)
            r["mae_total_minus_base"] = {"mean": float(diff.mean()), "ci95": _ci(diff)}
            fb = np.abs(d.loc[played, "base_fppg"].to_numpy(float) - d.loc[played, "actual_fppg"].to_numpy(float))
            r["fppg_mae_minus_base"] = {"mean": float((fe - fb).mean()), "ci95": _ci(fe - fb)}
            by = pd.DataFrame({"s": d["season"], "d": diff}).groupby("s")["d"].mean()
            r["seasons_improved"] = int((by < 0).sum())
        out[v] = r
    out["n_seasons"] = int(d["season"].nunique())
    return out


def render(s: dict) -> str:
    L = ["# Rookie prior: does origin information add lift?", "",
         f"{s['n']} rookies ({s['n_played']} played, {s['n_intl']} international, {s['combine_coverage']:.0%} with combine data), "
         f"{s['seasons'][0]} to {s['seasons'][-1]}, walk-forward.", "",
         "| variant | MAE total FP | Spearman total | MAE FP/game (played) | total MAE minus base (95% CI) | FP/game MAE minus base (95% CI) | seasons improved |",
         "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = s[v]
        a, b = r.get("mae_total_minus_base"), r.get("fppg_mae_minus_base")
        L.append(f"| {v} | {r['mae_total']:.1f} | {r['spearman_total']:.3f} | {r['fppg_mae_played']:.2f} | "
                 + (f"{a['mean']:+.1f} ({a['ci95'][0]:+.1f}, {a['ci95'][1]:+.1f})" if a else "-") + " | "
                 + (f"{b['mean']:+.2f} ({b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f})" if b else "-") + " | "
                 + (f"{r['seasons_improved']} of {s['n_seasons']}" if a else "-") + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from src.backtest.harness import expand_seasons

    ap = argparse.ArgumentParser(prog="python -m src.backtest.rookie_origin", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    d = evaluate(load_inputs(args.data_dir), expand_seasons(args.seasons))
    if d.empty:
        print("no rookies")
        return 1
    s = summarise(d)
    text = render(s)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rookie_origin.md").write_text(text, encoding="utf-8")
        (args.out / "summary.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
        d.to_csv(args.out / "frame.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
