"""Calibrate and validate the draft-availability noise (ADR 0040) on this league's own past drafts.

    python -m src.value.draft_sim_calibrate [--league-id N] [--seasons 2022,2023,2024,2025,2026] [--sims 200]

Reads each earlier season's ``mDraftDetail`` through the existing read-only, cached ESPN league client (raw
payloads stay in the data dir) and ESPN ADP from ``processed/adp.parquet``. At a decision point every STEP picks
it predicts P(a player is still there k picks later) for the RELEVANT lowest-ADP players still available and
scores that against what happened (Brier). The noise grid is fitted leave-one-season-out on the ESPN-ADP seasons
and compared with the old default and a constant. ESPN season id ``s`` is the season ending in ``s``
(2022 = 2021-22). 2025-26 ADP is FantasyPros gap-fill, so that season is an extra check, never fitted.
"""
from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.value import draft_sim as DS

SEASONS = (2022, 2023, 2024, 2025, 2026)
GAPS = (3, 6, 10, 14, 18, 22, 26)    # picks between turns, as in 10-16 team snakes
TOP_ADP = 20.0                       # 'top of the draft' for the star-only Brier check (the uniform skip's flaw)
STEP = 3                              # a decision point every STEP picks
RELEVANT = 30                         # score the RELEVANT lowest-ADP players of the pool at each point
KEYS = ["sd_min", "sd_frac", "sd_max", "skip", "skip_from"]
# The noise grid is the neighbourhood of the first fit (ADR 0040: 2 / 0.5 / 25 / skip 0.2 for every player); the skip now has
# an ADP gate (``skip_from``): stars are not passed on entirely, and the skip share is searched past the old grid edge.
GRID = {"sd_min": (2.0, 6.0), "sd_frac": (0.3, 0.5), "sd_max": (25.0,),
        "skip": (0.0, 0.1, 0.2, 0.3), "skip_from": (0.0, 15.0, 30.0, 45.0)}
OLD_DEFAULT = (1.0, 0.2, 25.0, 0.0, 0.0)
UNIFORM_SKIP = (2.0, 0.5, 25.0, 0.2, 0.0)       # the first fit: skip applied to every player


@dataclass
class Draft:
    season_id: int
    teams: int
    picks: pd.DataFrame       # overall, team_id, espn_id (str), auto (autoDraftTypeId), in draft order
    adp: pd.DataFrame         # source_id (str), adp, adp_source


def load_drafts(league_id: int, seasons=SEASONS, client=None, data_root=None) -> list[Draft]:
    from src.contracts import data_dir, raw_dir
    from src.ingest.espn_adp import season_from_espn_id
    from src.ingest.espn_league import ESPNLeagueClient, fetch_draft_detail

    client = client or ESPNLeagueClient(cache_dir=raw_dir("espn_league"))
    adp_all = pd.read_parquet((data_root or data_dir()) / "processed" / "adp.parquet")
    out = []
    for s in seasons:
        payload = fetch_draft_detail(client, s, league_id)
        picks = [p for p in (payload.get("draftDetail") or {}).get("picks", []) if (p.get("playerId") or -1) > 0]
        if not picks:
            continue
        df = pd.DataFrame({"overall": [p["overallPickNumber"] for p in picks], "team_id": [p["teamId"] for p in picks],
                           "espn_id": [str(p["playerId"]) for p in picks],
                           "auto": [p.get("autoDraftTypeId") for p in picks]}).sort_values("overall").reset_index(drop=True)
        a = adp_all[adp_all["season"] == season_from_espn_id(s)][["source_id", "adp", "adp_source"]].copy()
        a["source_id"] = a["source_id"].astype(str)
        out.append(Draft(s, df["team_id"].nunique(), df, a.drop_duplicates("source_id")))
    return out


def _pool(d: Draft) -> tuple[np.ndarray, np.ndarray]:
    """Market ADP (``draft_sim.market_adp``) and the pick each player went at (inf = never drafted)."""
    known = set(d.adp["source_id"])
    ids = list(d.adp["source_id"]) + [i for i in d.picks["espn_id"] if i not in known]
    adp = np.concatenate([d.adp["adp"].to_numpy(float), np.full(len(ids) - len(d.adp), np.nan)])
    order = np.argsort(np.where(np.isnan(adp), np.inf, adp), kind="stable")
    rank = np.empty(len(ids))
    rank[order] = np.arange(1, len(ids) + 1)
    market, _ = DS.market_adp(pd.DataFrame({"rank": rank, "adp": adp}))
    pick_of = dict(zip(d.picks["espn_id"], d.picks["overall"]))
    return market, np.array([pick_of.get(i, np.inf) for i in ids], dtype=float)


def predictions(d: Draft, params: tuple, *, n_sims: int = 200, seed: int = 0) -> pd.DataFrame:
    """One row per (decision pick c, gap k, relevant player): predicted P(available at c + k) and the outcome."""
    sd_min, sd_frac, sd_max, skip, skip_from = params
    market, when = _pool(d)
    total = int(d.picks["overall"].max())
    rows = []
    for c in range(1, total, STEP):
        avail = when >= c
        m, w = market[avail], when[avail]
        rel = np.argsort(m, kind="stable")[:RELEVANT]
        ks = [k for k in GAPS if c + k <= total + 1]
        if not ks or not len(rel):
            continue
        p = DS.survival_probabilities(m, ks, n_sims=n_sims, seed=seed + c, sd_min=sd_min, sd_frac=sd_frac,
                                      sd_max=sd_max, skip=skip, skip_from=skip_from)
        for j, k in enumerate(ks):
            rows.append(pd.DataFrame({"season": d.season_id, "c": c, "k": k, "market": m[rel], "p": p[j, rel],
                                      "y": (w[rel] >= c + k).astype(float), "never": np.isinf(w[rel])}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def brier(df: pd.DataFrame) -> float:
    return float(((df["p"] - df["y"]) ** 2).mean())


def residual_table(drafts: list[Draft]) -> pd.DataFrame:
    """Robust spread of (actual pick - ADP) by ADP bucket, players drafted with an ADP well inside the draft."""
    rows = []
    for d in drafts:
        m = d.picks.merge(d.adp, left_on="espn_id", right_on="source_id")
        m = m[m["adp"] <= 0.6 * d.picks["overall"].max()]
        rows.append(m.assign(resid=m["overall"] - m["adp"]))
    r = pd.concat(rows)
    r["bucket"] = pd.cut(r["adp"], [0, 12, 24, 48, 72, 100])
    g = r.groupby("bucket", observed=True)["resid"]
    return pd.DataFrame({"n": g.size(), "median": g.median(),
                         "robust_sd": g.apply(lambda x: 1.4826 * (x - x.median()).abs().median())})


def never_drafted_share(drafts: list[Draft]) -> pd.DataFrame:
    """Share of the top-N ADP players (N = picks in that draft) this league never drafted."""
    out = []
    for d in drafts:
        market, when = _pool(d)
        top = np.argsort(market, kind="stable")[: len(d.picks)]
        out.append({"season": d.season_id, "teams": d.teams, "picks": len(d.picks),
                    "never_drafted_top_n": int(np.isinf(when[top]).sum()),
                    "auto_flagged_picks": int((d.picks["auto"].fillna(0) != 0).sum())})
    return pd.DataFrame(out)


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(df["p"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    g = df.groupby(b, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pred": g["p"].mean(), "observed": g["y"].mean()})


def _weighted(grid: pd.DataFrame) -> pd.Series:
    tot = grid.assign(w=grid["brier"] * grid["n"]).groupby(KEYS)[["w", "n"]].sum()
    return (tot["w"] / tot["n"]).sort_values()


def run(drafts: list[Draft], n_sims: int = 200) -> dict:
    fit = [d for d in drafts if len(d.adp) and (d.adp["adp_source"] == "espn").all()]
    combos = list(dict.fromkeys([*itertools.product(*(GRID[k] for k in KEYS)), OLD_DEFAULT, UNIFORM_SKIP]))   # references always scored
    cache = {(d.season_id, c): predictions(d, c, n_sims=n_sims) for d in fit for c in combos}
    grid = pd.DataFrame([{"season": s, **dict(zip(KEYS, c)), "brier": brier(df), "n": len(df)}
                         for (s, c), df in cache.items()])
    loso = []
    for d in fit:
        best = _weighted(grid[grid["season"] != d.season_id]).index[0]
        held = cache[(d.season_id, best)]
        base = float(held["y"].mean())
        loso.append({"held_out": d.season_id, **{f"fit_{k}": v for k, v in zip(KEYS, best)}, "brier_fit": brier(held),
                     "brier_old_default": brier(cache[(d.season_id, OLD_DEFAULT)]),
                     "brier_uniform_skip": brier(cache[(d.season_id, UNIFORM_SKIP)]),
                     **{f"top_{name}": brier(df[df["market"] <= TOP_ADP]) for name, df in
                        (("fit", held), ("uniform_skip", cache[(d.season_id, UNIFORM_SKIP)]))},
                     "brier_constant": float(((base - held["y"]) ** 2).mean())})
    pooled = _weighted(grid)
    best = pooled.index[0]
    loso_pred = pd.concat([cache[(r["held_out"], tuple(r[f"fit_{k}"] for k in KEYS))] for r in loso])
    return {"pooled": pooled.rename("brier").reset_index(), "loso": pd.DataFrame(loso), "best": best,
            "calibration_loso": calibration_table(loso_pred),
            "calibration_old": calibration_table(pd.concat([cache[(d.season_id, OLD_DEFAULT)] for d in fit])),
            "residuals": residual_table(fit), "never": never_drafted_share(drafts), "fit": fit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league-id", type=int, default=int(os.environ.get("ESPN_LEAGUE_ID") or 0),
                    help="ESPN league id (default $ESPN_LEAGUE_ID; never committed)")
    ap.add_argument("--seasons", default=",".join(map(str, SEASONS)))
    ap.add_argument("--sims", type=int, default=200)
    a = ap.parse_args(argv)
    if not a.league_id:
        ap.error("set $ESPN_LEAGUE_ID or pass --league-id")
    drafts = load_drafts(a.league_id, [int(s) for s in a.seasons.split(",")])
    pd.set_option("display.width", 200)
    res = run(drafts, a.sims)
    print("drafts:\n", res["never"].to_string(index=False))
    print("\nresidual (pick - ADP) by ADP bucket:\n", res["residuals"].round(2))
    print("\npooled Brier, best 10 of the grid:\n", res["pooled"].head(10).round(4).to_string(index=False))
    print("\nleave-one-season-out:\n", res["loso"].round(4).to_string(index=False))
    print(f"\nbest pooled {dict(zip(KEYS, res['best']))}")
    print("calibration, held-out seasons at their LOSO fit:\n", res["calibration_loso"].round(3))
    print("calibration, old default:\n", res["calibration_old"].round(3))
    for d in drafts:
        if d in res["fit"]:
            continue
        for params in (res["best"], OLD_DEFAULT):
            pr = predictions(d, params, n_sims=a.sims)
            print(f"season {d.season_id} (FantasyPros-filled ADP, not fitted) Brier at {params}: {brier(pr):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
