"""ADP + model blend for the draft board (ADR 0032).

ADR 0030 found ADP beats the model at the top of the draft board while the model beats ADP on magnitude and depth, and that a
regression of season total FP on *both* (``adp_model``) keeps ADP's ranking at the top and cuts total-FP error. This module
turns that arm into a board feature without changing any model column:

    blend_total_fp = b0 + b1*log(ADP) + b2*log(ADP)^2 + b3*model_total_fp + b4*(model missing)        ADP-listed players
                   = model_total_fp                                                                   everyone else

The coefficients are fit on *earlier completed seasons only* (walk-forward projections of the same model against what
happened), stored as JSON next to the data (``adp_blend.json``) and applied to the live board. ``blend_vorp`` /
``blend_rank`` / ``blend_tier`` then come from the ordinary VORP machinery run on the blended totals, so they are directly
comparable with ``vorp`` / ``rank`` / ``tier``; floor, ceiling and games played stay the model's own.

CLI (fits from the real history, about a minute): ``python -m src.value.adp_blend fit --model baseline_hurdle_adp_offseason_debut``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

BLEND_FILE = "adp_blend.json"
BLEND_COLUMNS = ("blend_total_fp", "blend_vorp", "blend_rank", "blend_tier")
MIN_ROWS = 150          # listed player-seasons needed to fit the regression


def _design(adp, model, model_fill: float) -> np.ndarray:
    la = np.log(np.maximum(np.asarray(adp, float), 1.0))
    m = np.asarray(model, float)
    miss = ~np.isfinite(m)
    return np.column_stack([np.ones(len(la)), la, la ** 2, np.where(miss, model_fill, m), miss.astype(float)])


@dataclass
class AdpBlend:
    coef: list[float]             # [1, log adp, log adp ^2, model total, model-missing flag]
    model_fill: float             # mean model total in the fit, used where a listed player has no projection
    model: str                    # projector the coefficients were fit for
    seasons: list[str]            # seasons whose (ADP, projection, outcome) rows were used
    n_rows: int
    fitted_utc: str = ""

    def predict(self, adp, model_total) -> np.ndarray:
        """Blended season total FP for ADP-listed players (caller handles unlisted players)."""
        return _design(adp, model_total, self.model_fill) @ np.asarray(self.coef, float)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "AdpBlend":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_adp_blend(frames: Mapping[str, pd.DataFrame], *, model: str) -> AdpBlend:
    """Least squares on ADP-listed player-seasons. Each frame: ``player_id, adp, proj_total_fp, actual_total_fp``.

    A listed player who was projected but did not play counts as 0 actual FP (the backtest's own convention); a listed
    player the model did not project has a missing model total and the missing flag.
    """
    parts = []
    for season, f in frames.items():
        d = f[f["adp"].notna()].copy()
        d["actual_total_fp"] = d["actual_total_fp"].fillna(0.0)
        parts.append(d.assign(season=season))
    if not parts:
        raise ValueError("no seasons to fit the ADP blend on")
    d = pd.concat(parts, ignore_index=True)
    if len(d) < MIN_ROWS:
        raise ValueError(f"only {len(d)} ADP-listed player-seasons; need at least {MIN_ROWS} to fit the blend")
    fill = float(np.nanmean(d["proj_total_fp"]))
    X = _design(d["adp"], d["proj_total_fp"], fill)
    coef, *_ = np.linalg.lstsq(X, d["actual_total_fp"].to_numpy(float), rcond=None)
    return AdpBlend([float(c) for c in coef], fill, model, list(frames), len(d),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))


def blend_totals(proj: pd.DataFrame, adp: pd.DataFrame, blend: AdpBlend) -> pd.Series:
    """Blended season total per projection row: the regression for ADP-listed players, the model total for the rest."""
    a = adp.drop_duplicates("player_id").set_index("player_id")["adp"].astype(float)
    adp_v = a.reindex(proj["player_id"]).to_numpy()
    model_v = proj["proj_total_fp"].to_numpy(float)
    out = model_v.copy()
    listed = np.isfinite(adp_v)
    if listed.any():
        out[listed] = blend.predict(adp_v[listed], model_v[listed])
    return pd.Series(np.maximum(out, 0.0), index=proj.index, name="blend_total_fp")


def add_blend_columns(board: pd.DataFrame, proj: pd.DataFrame, adp: pd.DataFrame, blend: AdpBlend, *,
                      cfg: dict | None = None, teams: int | None = None, bench_weight: float | None = None,
                      season_games: float = 82.0, positional: str = "auto", scarcity_threshold: float | None = None,
                      **tier_kwargs) -> pd.DataFrame:
    """``board`` plus ``blend_total_fp, blend_vorp, blend_rank, blend_tier`` (every existing column untouched).

    VORP is recomputed by the ordinary machinery on a copy of the projections whose ``proj_total_fp`` is the blend
    (``proj_fppg`` rescaled by the same factor so the two stay consistent), with the same league shape and options.
    """
    from src.value.league import load_league
    from src.value.replacement import DEFAULT_SCARCITY_THRESHOLD, league_shape
    from src.value.tiers import assign_tiers
    from src.value.vorp import compute_vorp

    proj = proj.reset_index(drop=True)
    total = blend_totals(proj, adp, blend)
    p2 = proj.copy()
    p2["proj_fppg"] = np.where(proj["proj_gp"] > 0, total / np.where(proj["proj_gp"] > 0, proj["proj_gp"], 1.0),
                               proj["proj_fppg"])
    p2["proj_total_fp"] = total
    if "position" not in p2.columns and "position" in board.columns:
        pos = board.drop_duplicates("player_id").set_index("player_id")["position"]
        p2["position"] = pos.reindex(p2["player_id"]).to_numpy()
    shape = league_shape(cfg or load_league(), teams=teams)
    has_pos = "position" in p2.columns and p2["position"].notna().any()
    res = compute_vorp(p2 if has_pos else p2.drop(columns="position", errors="ignore"), shape, bench_weight=bench_weight,
                       season_games=season_games, positional=positional if has_pos else "off",
                       scarcity_threshold=DEFAULT_SCARCITY_THRESHOLD if scarcity_threshold is None else scarcity_threshold)
    t = pd.DataFrame({"player_id": proj["player_id"].to_numpy(), "blend_total_fp": total.to_numpy(),
                      "blend_vorp": res.frame["vorp"].to_numpy()})
    t = t.sort_values(["blend_vorp", "blend_total_fp", "player_id"], ascending=[False, False, True], kind="mergesort")
    t["blend_rank"] = range(1, len(t) + 1)
    t["blend_tier"] = assign_tiers(t["blend_vorp"].to_numpy(), floor=0.0, **tier_kwargs)
    return board.merge(t, on="player_id", how="left")


# --------------------------------------------------------------------------- fitting from the real history

def fit_from_history(model: str, seasons: str, *, data_dir: Path | None = None) -> AdpBlend:
    """Walk-forward ``model`` over ``seasons`` (default the completed history) and fit the blend on the results."""
    from src.backtest.benchmarks import load_adp  # noqa: F401  (kept import-light: the mapping lives in features.adp)
    from src.backtest.harness import expand_seasons, walk_forward
    from src.backtest.runner import load_run_tables, resolve_projector
    from src.features.adp import load_store_adp
    from src.value.league import load_league

    class _Args:
        synthetic = False

    tables = load_run_tables(_Args())
    adp = load_store_adp(data_dir)
    res = walk_forward(tables, resolve_projector(model), expand_seasons(seasons), scoring=load_league()["scoring"])
    frames = {}
    for s in res.seasons:
        f = res.season_frame(s)
        a = adp[adp["season"] == s][["player_id", "adp"]]
        frames[s] = f.merge(a, on="player_id", how="left")
    return fit_adp_blend(frames, model=model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.value.adp_blend", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the blend from the walk-forward history and write adp_blend.json")
    f.add_argument("--model", default="baseline_hurdle_adp_offseason_debut")
    f.add_argument("--seasons", default="2016-17:2025-26")
    f.add_argument("--data-dir", type=Path, default=None)
    f.add_argument("--out", type=Path, default=None, help=f"default: <data dir>/processed/{BLEND_FILE}")
    args = ap.parse_args(argv)
    from src.contracts import data_dir

    base = args.data_dir or data_dir()
    blend = fit_from_history(args.model, args.seasons, data_dir=args.data_dir)
    path = blend.save(args.out or (base / "processed" / BLEND_FILE))
    print(f"adp blend fit on {blend.n_rows} ADP-listed player-seasons ({blend.seasons[0]}..{blend.seasons[-1]}); "
          f"coef={['%.4g' % c for c in blend.coef]}; wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
