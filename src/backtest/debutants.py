"""Walk-forward evaluation of the debutant treatment (ADR 0016).

    python -m src.backtest.debutants [--seasons 2018-19:2025-26] [--out reports/debutants] [--data-dir ...]

For each target season ``S`` the candidate set is every **camp participant** (preseason games of ``S``) with no earlier
NBA game and not in ``S``'s draft class, split into ``stash`` (drafted earlier) and ``undrafted``. Everything the
treatment learns comes from earlier seasons' camp participants and their realised seasons; ``S``'s own outcomes are
only used to score it. Players who never play score zero games and zero fantasy points (they are not dropped).

Treatments compared, all on the same rows:

* ``omit``      what the board did before: no projection (predicts 0 total fantasy points).
* ``prior``     the rookie draft-slot prior at the raw pick, every player assumed to play the prior's games share.
* ``prior_p``   ``prior`` multiplied by the model's play probability (isolates what the fitted multipliers add).
* ``class_mean`` the mean total fantasy points of the class in earlier seasons (no player information).
* ``model``     the shipped treatment (:class:`src.models.debutants.DebutantBaselineProjector`).

Read the results with the caveats printed in the report: stash membership is only recognisable for players with an
index record, so every stash analogue plays and the stash play-probability is an assumption (see the module docstring
of ``src.models.debutants``); the undrafted play-probability, which *is* identifiable from the camp roster, is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.contracts import HISTORY_TABLES, History, season_start, season_str
from src.features.debutants import STASH, UNDRAFTED
from src.models.debutants import DebutantBaselineProjector, season_actuals
from src.value.league import load_league

DEFAULT_SEASONS = "2018-19:2025-26"
BOOT = 2000


def _seasons(spec: str) -> list[str]:
    from src.backtest.harness import expand_seasons

    return expand_seasons(spec)


def load_inputs(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    from src.ingest import nba_offseason, nba_profiles
    from src.store import load_tables

    t = dict(load_tables(HISTORY_TABLES, base=data_dir))
    t["offseason_logs"] = nba_offseason.read_offseason_logs(data_dir)
    t["offseason_team_games"] = nba_offseason.read_offseason_team_games(data_dir)
    t["player_profiles"] = nba_profiles.read_profiles(data_dir)
    return t


def evaluate(tables: dict[str, pd.DataFrame], seasons: list[str], *, log=print) -> pd.DataFrame:
    """One row per (season, candidate) with each treatment's predicted total fantasy points and the outcome."""
    scoring = load_league()["scoring"]
    act = season_actuals(tables["game_logs"], scoring).set_index(["s", "player_id"])
    frames = []
    for S in seasons:
        sy = season_start(S)
        h = History.until(tables, S)
        proj = DebutantBaselineProjector(scoring)
        fitted = proj.fit(h)
        extra = proj._debutant_rows(h, fitted)
        cand, prm = proj.last_candidates, proj.last_params
        if extra is None or cand is None or cand.empty:
            log(f"  {S}: no candidates")
            continue
        L = float(fitted.season_games)
        m = extra.set_index("player_id")
        d = cand.copy()
        d["model_total"] = m["proj_total_fp"].reindex(d["player_id"]).to_numpy()
        d["model_gp"] = m["proj_gp"].reindex(d["player_id"]).to_numpy()
        d["model_fppg"] = m["proj_fppg"].reindex(d["player_id"]).to_numpy()
        d["model_mpg"] = m["proj_mpg"].reindex(d["player_id"]).to_numpy()
        d["p_play"] = m["p_play"].reindex(d["player_id"]).to_numpy()
        raw = fitted.prior_rows(d["player_id"].to_numpy("int64"), d["pick"].to_numpy(float), d["group"].to_numpy(),
                                d["age"].to_numpy(float))
        d["prior_fppg"] = fitted._fppg(raw)
        d["prior_mpg"] = raw["proj_mpg"].to_numpy()
        d["prior_total"] = d["prior_fppg"] * raw["mu_f"].to_numpy() * L
        d["prior_p_total"] = d["prior_total"] * d["p_play"]
        # class means from earlier seasons only
        from src.models.debutants import build_analogs

        tr = build_analogs(h.game_logs, h.extras["offseason_logs"], h.extras.get("offseason_team_games"),
                           h.extras["player_profiles"], scoring,
                           [season_str(y) for y in range(2016, sy)], players=h.players)
        cm = tr.groupby("klass")["tot"].mean() if len(tr) else pd.Series(dtype=float)
        d["class_mean_total"] = d["klass"].map(cm).fillna(0.0)
        d["omit_total"] = 0.0
        a = act.reindex(pd.MultiIndex.from_arrays([np.full(len(d), sy), d["player_id"]]))
        d["actual_gp"] = a["gp"].fillna(0.0).to_numpy()
        d["actual_total"] = a["tot"].fillna(0.0).to_numpy()
        d["actual_mpg"] = a["mpg"].to_numpy()
        d["actual_fppg"] = a["fppg"].to_numpy()
        d["season"] = S
        d["decay"], d["stash_intl_scale"] = prm.decay, prm.stash_intl_scale
        d["train_debutants"] = prm.n_train.get(STASH, 0) + prm.n_train.get(UNDRAFTED, 0)
        frames.append(d)
        log(f"  {S}: {len(d)} candidates ({(d['klass'] == STASH).sum()} stash), {int((d['actual_gp'] > 0).sum())} played")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


TREATMENTS = {"omit": "omit_total", "prior": "prior_total", "prior_p": "prior_p_total", "class_mean": "class_mean_total", "model": "model_total"}


def _boot_ci(diff: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if len(diff) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(d: pd.DataFrame) -> dict:
    """Metrics per class: total-FP MAE / RMSE / Spearman for every treatment, paired MAE difference of the model, and
    level accuracy (minutes, FP per game) among those who played."""
    out: dict = {"n_candidates": int(len(d)), "seasons": sorted(d["season"].unique())}
    for name, sub in (("all", d), (STASH, d[d["klass"] == STASH]), (UNDRAFTED, d[d["klass"] == UNDRAFTED])):
        if sub.empty:
            continue
        y = sub["actual_total"].to_numpy(float)
        res: dict = {"n": int(len(sub)), "n_played": int((sub["actual_gp"] > 0).sum()),
                     "mean_actual_total": float(y.mean()), "mean_actual_gp": float(sub["actual_gp"].mean())}
        for t, col in TREATMENTS.items():
            p = sub[col].to_numpy(float)
            r = {"mae": float(np.abs(p - y).mean()), "rmse": float(np.sqrt(((p - y) ** 2).mean())), "bias": float((p - y).mean())}
            r["spearman"] = float(spearmanr(p, y)[0]) if np.ptp(p) > 0 else float("nan")
            res[t] = r
        for base in ("omit", "prior", "prior_p", "class_mean"):
            diff = np.abs(sub["model_total"].to_numpy(float) - y) - np.abs(sub[TREATMENTS[base]].to_numpy(float) - y)
            lo, hi = _boot_ci(diff)
            res[f"model_minus_{base}_mae"] = {"mean": float(diff.mean()), "ci95": [lo, hi]}
        pl = sub[sub["actual_gp"] > 0]
        if len(pl):
            res["played"] = {
                "mpg_mae_model": float((pl["model_mpg"] - pl["actual_mpg"]).abs().mean()),
                "mpg_mae_prior": float((pl["prior_mpg"] - pl["actual_mpg"]).abs().mean()),
                "fppg_mae_model": float((pl["model_fppg"] - pl["actual_fppg"]).abs().mean()),
                "fppg_mae_prior": float((pl["prior_fppg"] - pl["actual_fppg"]).abs().mean()),
                "fppg_spearman_model": float(spearmanr(pl["model_fppg"], pl["actual_fppg"])[0]) if len(pl) > 4 else float("nan"),
                "mean_actual_fppg": float(pl["actual_fppg"].mean()), "mean_model_fppg": float(pl["model_fppg"].mean()),
            }
        if name == UNDRAFTED:
            from sklearn.metrics import roc_auc_score

            yy = (sub["actual_gp"] > 0).to_numpy()
            if yy.min() != yy.max():
                res["p_play_auc"] = float(roc_auc_score(yy, sub["p_play"]))
            res["p_play_mean_pred"], res["p_play_actual"] = float(sub["p_play"].mean()), float(yy.mean())
        out[name] = res
    # does a pick / years-since-draft / origin signal exist in the stash residuals?
    st = d[(d["klass"] == STASH) & (d["actual_gp"] > 0)]
    if len(st) >= 8:
        resid = (st["actual_fppg"] - st["prior_fppg"]).to_numpy(float)
        out["stash_residual_correlations"] = {
            c: float(pd.Series(resid).corr(pd.to_numeric(st[c], errors="coerce").reset_index(drop=True)))
            for c in ("years_since_draft", "age", "intl", "pick")}
    return out


def render(summary: dict) -> str:
    L = ["# Debutant treatment: walk-forward evaluation", "",
         f"Seasons {summary['seasons'][0]} to {summary['seasons'][-1]}, {summary['n_candidates']} camp participants with no earlier NBA game.", ""]
    for k in ("all", STASH, UNDRAFTED):
        r = summary.get(k)
        if not r:
            continue
        L += [f"## {k} (n={r['n']}, played={r['n_played']}, mean actual total FP {r['mean_actual_total']:.0f}, mean games {r['mean_actual_gp']:.1f})", "",
              "| treatment | MAE total FP | RMSE | bias | Spearman |", "|---|---|---|---|---|"]
        for t in TREATMENTS:
            m = r[t]
            L.append(f"| {t} | {m['mae']:.1f} | {m['rmse']:.1f} | {m['bias']:+.1f} | {m['spearman']:.3f} |")
        for base in ("omit", "prior", "prior_p", "class_mean"):
            c = r[f"model_minus_{base}_mae"]
            L.append(f"\nmodel minus {base}, paired MAE: {c['mean']:+.1f} (95% CI {c['ci95'][0]:+.1f} to {c['ci95'][1]:+.1f})")
        if "played" in r:
            p = r["played"]
            L += ["", f"Among those who played: minutes MAE model {p['mpg_mae_model']:.2f} vs prior {p['mpg_mae_prior']:.2f}; "
                  f"FP/game MAE model {p['fppg_mae_model']:.2f} vs prior {p['fppg_mae_prior']:.2f}; "
                  f"FP/game Spearman {p['fppg_spearman_model']:.2f}; mean predicted {p['mean_model_fppg']:.1f} vs actual {p['mean_actual_fppg']:.1f}."]
        if "p_play_auc" in r:
            L.append(f"\nPlay probability: AUC {r['p_play_auc']:.3f}, mean predicted {r['p_play_mean_pred']:.3f} vs actual {r['p_play_actual']:.3f}.")
        L.append("")
    if "stash_residual_correlations" in summary:
        L += ["## Stash: does anything explain the residual of the raw prior?", "",
              "Correlation of (actual FP/game minus prior FP/game) with: " + ", ".join(
                  f"{k} {v:+.2f}" for k, v in summary["stash_residual_correlations"].items()), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.debutants", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=DEFAULT_SEASONS)
    ap.add_argument("--out", type=Path, default=None, help="directory for debutants.md, summary.json and frame.csv")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    d = evaluate(load_inputs(args.data_dir), _seasons(args.seasons))
    if d.empty:
        print("no candidates")
        return 1
    s = summarise(d)
    text = render(s)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "debutants.md").write_text(text, encoding="utf-8")
        (args.out / "summary.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
        d.to_csv(args.out / "frame.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
