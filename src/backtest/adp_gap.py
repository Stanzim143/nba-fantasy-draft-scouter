"""ADP-gap study (docs/research/adp-gap-study.md): is there predictable structure in where the model and ADP disagree and ADP is right?

Walk-forward evaluation frames of one model (default: the board default) are joined to ESPN ADP and to pre-season
player descriptors (age, experience, team change, prior availability, preseason role signal, contract year). The study
then (1) describes, by tier and player type, how often ADP's side of a disagreement wins, and (2) tests pre-registered
walk-forward corrections of the ADR 0032 blend against the plain blend with the repo's paired bootstrap.

    python -m src.backtest.adp_gap collect   # walk-forward once (memory heavy), cache the eval frames as parquet
    python -m src.backtest.adp_gap report    # descriptive tables + corrections from the cached frames (light)

Rank convention (ADR 0028): ``rank_delta = model_rank - adp_rank`` among ADP-listed, projected players of a season, so
a positive value means ADP ranks the player *higher* (earlier) than the model does.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.contracts import season_start

log = logging.getLogger("src.backtest.adp_gap")

DEFAULT_MODEL = "baseline_hurdle_adp_depth_rookie_offseason_debut"
DEFAULT_SEASONS = "2016-17:2025-26"
FRAME_COLUMNS = ("season", "player_id", "player_name", "projected", "played", "proj_total_fp", "proj_gp",
                 "actual_total_fp", "actual_gp")


def cache_root() -> Path:
    from src.contracts import data_dir
    return data_dir() / "cache" / "adp-gap"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - a label only
        return "nogit"


# --------------------------------------------------------------------------- collect (memory heavy, run once)

def collect(model: str = DEFAULT_MODEL, seasons: str = DEFAULT_SEASONS, out: Path | None = None) -> Path:
    """Walk ``model`` forward over ``seasons`` and write the stacked eval frames (plus ADP) to parquet."""
    from src.backtest.harness import expand_seasons, walk_forward
    from src.backtest.runner import load_run_tables, resolve_projector
    from src.features.adp import load_store_adp
    from src.value.league import load_league

    class _Args:
        synthetic = False

    tables = load_run_tables(_Args())
    res = walk_forward(tables, resolve_projector(model), expand_seasons(seasons), scoring=load_league()["scoring"])
    adp = load_store_adp()
    f = res.players[list(FRAME_COLUMNS)].merge(adp, on=["season", "player_id"], how="left")
    path = Path(out) if out else cache_root() / f"{model}__{_git_sha()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    f.assign(model=model).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- pre-season descriptors

def player_descriptors(frame: pd.DataFrame, game_logs: pd.DataFrame, players: pd.DataFrame,
                       offseason_logs: pd.DataFrame | None = None, contracts: pd.DataFrame | None = None,
                       scoring: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Per (season, player_id) of ``frame``: descriptors known before the season (except ``team_change``, below).

    * ``age``: on 1 Oct of the season's start year.
    * ``exp``: earlier NBA regular seasons with a game (0 = rookie, 1 = sophomore).
    * ``prior_gp_share``: last season's games / 82 (NaN when he did not play last season).
    * ``team_change``: first team of the target season differs from last season's last team. The first team is
      observed at the first game, so this is a near pre-season proxy (a few early-season trades leak in); it is used
      descriptively and as a correction feature, and the writeup says so.
    * ``pre_fppg_delta``: preseason fantasy points per game minus last season's FPPG (NaN without both).
    * ``contract_year``: final known year of a veteran deal or a first-rounder's fourth season (ADR 0013/0019).
    """
    from src.value.frame import fantasy_points_frame as fantasy_points

    if scoring is None:
        from src.value.league import load_league
        scoring = load_league()["scoring"]
    keys = frame[["season", "player_id"]].drop_duplicates().reset_index(drop=True)
    gl = game_logs.copy()
    gl["fp"] = fantasy_points(gl, scoring)
    gl["_start"] = gl["season"].map(season_start)
    by = gl.groupby(["player_id", "_start"])
    season_tab = by.agg(gp=("game_id", "nunique"), fp=("fp", "sum")).reset_index()
    first_team = (gl.sort_values("game_date").groupby(["player_id", "_start"])["team_id"].first()
                  .rename("first_team").reset_index())
    last_team = (gl.sort_values("game_date").groupby(["player_id", "_start"])["team_id"].last()
                 .rename("last_team").reset_index())

    out = keys.copy()
    out["_start"] = out["season"].map(season_start)
    bd = players.drop_duplicates("player_id").set_index("player_id")["birthdate"]
    born = pd.to_datetime(out["player_id"].map(bd), errors="coerce")
    ref = pd.to_datetime(out["_start"].astype(str) + "-10-01")
    out["age"] = (ref - born).dt.days / 365.25

    exp = []
    starts = season_tab.groupby("player_id")["_start"].apply(lambda s: np.sort(s.to_numpy()))
    for pid, s0 in zip(out["player_id"], out["_start"]):
        arr = starts.get(pid)
        exp.append(int(np.searchsorted(arr, s0)) if arr is not None else 0)
    out["exp"] = exp

    prev = season_tab.assign(_start=season_tab["_start"] + 1)
    out = out.merge(prev.rename(columns={"gp": "prior_gp", "fp": "prior_fp"}), on=["player_id", "_start"], how="left")
    out["prior_gp_share"] = (out["prior_gp"] / 82.0).clip(upper=1.0)
    out["prior_fppg"] = out["prior_fp"] / out["prior_gp"]
    out = out.merge(last_team.assign(_start=last_team["_start"] + 1), on=["player_id", "_start"], how="left")
    out = out.merge(first_team, on=["player_id", "_start"], how="left")
    out["team_change"] = (out["last_team"].notna() & out["first_team"].notna()
                          & (out["last_team"] != out["first_team"]))

    out["pre_fppg_delta"] = np.nan
    if offseason_logs is not None and len(offseason_logs):
        ol = offseason_logs[offseason_logs["context"] == "preseason"].copy()
        if len(ol):
            ol["fp"] = fantasy_points(ol, scoring)
            pre = ol.groupby(["season", "player_id"]).agg(pre_fp=("fp", "sum"), pre_gp=("game_id", "nunique"))
            pre = pre.reset_index()
            pre = pre[pre["pre_gp"] >= 2]
            out = out.merge(pre, on=["season", "player_id"], how="left")
            out["pre_fppg_delta"] = out["pre_fp"] / out["pre_gp"] - out["prior_fppg"]

    out["contract_year"] = False
    if contracts is not None and len(contracts):
        from src.features.contract import contract_clock
        from src.features.contract_terms import terms_features
        cy = []
        for s, g in out.groupby("season", sort=False):
            pids = g["player_id"].to_numpy("int64")
            clock = contract_clock(players.set_index("player_id").reindex(pids).reset_index(), s)
            tf = terms_features(contracts, s, pids, clock)
            cy.append(pd.Series(tf.contract_year | tf.rookie_cy, index=g.index))
        out["contract_year"] = pd.concat(cy).reindex(out.index).fillna(False).astype(bool)
    cols = ["season", "player_id", "age", "exp", "prior_gp_share", "team_change", "pre_fppg_delta", "contract_year"]
    return out[cols]


# --------------------------------------------------------------------------- disagreement table

TIERS = (24, 50, 100)


def disagreement_frame(frame: pd.DataFrame, desc: pd.DataFrame | None = None) -> pd.DataFrame:
    """ADP-listed, projected rows with model/ADP/actual ranks, ``rank_delta`` and who was right.

    Ranks are within that season's ADP-listed, projected players (the comparable set). ``adp_right`` is 1 when the
    actual rank is closer to the ADP rank than to the model rank, 0 when closer to the model, 0.5 on a tie.
    ``tier`` is the better (earlier) of the two ranks, bucketed by ``TIERS`` (``>100`` beyond).
    """
    d = frame[frame["projected"] & frame["adp"].notna()].copy()
    d["actual_total_fp"] = d["actual_total_fp"].fillna(0.0)
    g = d.groupby("season")
    d["model_rank"] = g["proj_total_fp"].rank(ascending=False, method="first")
    d["adp_rank"] = g["adp"].rank(ascending=True, method="first")
    d["actual_rank"] = g["actual_total_fp"].rank(ascending=False, method="average")
    d["rank_delta"] = d["model_rank"] - d["adp_rank"]
    em, ea = (d["actual_rank"] - d["model_rank"]).abs(), (d["actual_rank"] - d["adp_rank"]).abs()
    d["adp_right"] = np.where(ea < em, 1.0, np.where(ea > em, 0.0, 0.5))
    best = np.minimum(d["model_rank"], d["adp_rank"])
    d["tier"] = pd.cut(best, [0, *TIERS, np.inf], labels=[f"top{TIERS[0]}", *[f"{a + 1}-{b}" for a, b in
                                                                           zip(TIERS, TIERS[1:])], f">{TIERS[-1]}"])
    d["direction"] = np.where(d["rank_delta"] > 0, "ADP higher", np.where(d["rank_delta"] < 0, "ADP lower", "same"))
    if desc is not None:
        d = d.merge(desc, on=["season", "player_id"], how="left")
    return d


def _wins_row(g: pd.DataFrame, min_gap: int) -> dict:
    big = g[g["rank_delta"].abs() >= min_gap]
    return {"n": len(g), "n_disagree": len(big),
            "adp_right": float(big["adp_right"].mean()) if len(big) else np.nan,
            "mean_abs_delta": float(g["rank_delta"].abs().mean()) if len(g) else np.nan,
            "seasons_adp_majority": f"{int((big.groupby('season')['adp_right'].mean() > 0.5).sum())}/"
                                    f"{big['season'].nunique()}" if len(big) else "0/0"}


def segment(d: pd.DataFrame) -> pd.Series:
    """One player-type label per row, by priority: rookie > sophomore > age>=31 > team change > prior GP < 50% >
    preseason up/down > contract year, else other."""
    lab = np.full(len(d), "other", dtype=object)
    rules = [
        ("contract year", d.get("contract_year", pd.Series(False, index=d.index)).fillna(False).astype(bool)),
        ("preseason down", d.get("pre_fppg_delta", pd.Series(np.nan, index=d.index)) <= -5),
        ("preseason up", d.get("pre_fppg_delta", pd.Series(np.nan, index=d.index)) >= 5),
        ("prior GP < 50%", d.get("prior_gp_share", pd.Series(np.nan, index=d.index)) < 0.5),
        ("team change", d.get("team_change", pd.Series(False, index=d.index)).fillna(False).astype(bool)),
        ("age >= 31", d.get("age", pd.Series(np.nan, index=d.index)) >= 31),
        ("sophomore", d.get("exp", pd.Series(-1, index=d.index)) == 1),
        ("rookie", d.get("exp", pd.Series(-1, index=d.index)) == 0),
    ]
    for name, mask in rules:           # later rules overwrite: rookie > sophomore > age > ... (priority order)
        lab[np.asarray(mask.fillna(False) if hasattr(mask, "fillna") else mask, bool)] = name
    return pd.Series(lab, index=d.index, name="segment")


def win_tables(d: pd.DataFrame, *, min_gap: int = 10) -> dict[str, pd.DataFrame]:
    """Share of disagreements (|rank_delta| >= ``min_gap``) that ADP's side wins, by tier, direction and segment."""
    d = d.assign(segment=segment(d))
    out = {}
    for name, keys in (("tier", ["tier"]), ("direction", ["tier", "direction"]), ("segment", ["segment"]),
                       ("segment_direction", ["segment", "direction"])):
        rows = []
        for k, g in d.groupby(keys, observed=True):
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **_wins_row(g, min_gap)})
        out[name] = pd.DataFrame(rows)
    out["overall"] = pd.DataFrame([_wins_row(d, min_gap)])
    return out


# --------------------------------------------------------------------------- pre-registered corrections

def _flags(d: pd.DataFrame) -> np.ndarray:
    """Interaction design for the feature blend: per flag, ``flag`` and ``flag * (log ADP - model-implied log rank)``."""
    f = []
    for col, fn in (("exp", lambda x: x.fillna(-1) <= 1), ("age", lambda x: x.fillna(0) >= 31),
                    ("team_change", lambda x: x.fillna(False).astype(bool)),
                    ("prior_gp_share", lambda x: x.fillna(1.0) < 0.5),
                    ("pre_fppg_delta", lambda x: x.fillna(0).abs() >= 5)):
        f.append(np.asarray(fn(d[col]) if col in d else np.zeros(len(d), bool), float))
    return np.column_stack(f)


FLAG_NAMES = ("young (exp<=1)", "age>=31", "team change", "prior GP<50%", "preseason |dFPPG|>=5")


def feature_blend_fit(train: pd.DataFrame) -> np.ndarray:
    """Least squares of actual total on the ADR 0032 design plus ``flag`` and ``flag * log ADP`` for each flag."""
    X = _feature_design(train)
    y = train["actual_total_fp"].fillna(0.0).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _feature_design(d: pd.DataFrame) -> np.ndarray:
    la = np.log(np.maximum(d["adp"].to_numpy(float), 1.0))
    m = d["proj_total_fp"].to_numpy(float)
    F = _flags(d)
    return np.column_stack([np.ones(len(d)), la, la ** 2, m, F, F * la[:, None], F * m[:, None]])


def shrink_group(d: pd.DataFrame) -> pd.Series:
    """Correction B's cell: the tier, split by whether any of the ``FLAG_NAMES`` flags is set (``tier|flag``/``tier|none``)."""
    any_flag = _flags(d).any(axis=1) if len(d) else np.zeros(0, bool)
    return d["tier"].astype(str) + np.where(any_flag, "|flag", "|none")


def rank_shrink(d: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    """Score whose order is ``(1 - w) * model_rank + w * adp_rank``, ``w`` per ``shrink_group`` cell (0 if unseen)."""
    w = shrink_group(d).map(weights).fillna(0.0).to_numpy(float)
    r = (1 - w) * d["model_rank"].to_numpy(float) + w * d["adp_rank"].to_numpy(float)
    return -r


def fit_shrink_weights(train: pd.DataFrame, grid: Sequence[float] = tuple(np.round(np.linspace(0, 1, 11), 2))
                       ) -> dict[str, float]:
    """Per ``shrink_group`` cell, the weight minimising mean |actual_rank - shrunk rank| on ``train`` (earlier seasons)."""
    out = {}
    for key, g in train.groupby(shrink_group(train)):
        errs = [np.mean(np.abs(g["actual_rank"] - ((1 - w) * g["model_rank"] + w * g["adp_rank"]))) for w in grid]
        out[str(key)] = float(grid[int(np.argmin(errs))])
    return out


def corrected_totals(frame: pd.DataFrame, desc: pd.DataFrame, *, min_rows: int = 150) -> pd.DataFrame:
    """Walk-forward arms per season (fit on earlier seasons only): ``model``, ``blend`` (ADR 0032), ``blend_feat``
    (pre-registered correction A) and ``blend_shrink`` (correction B, the blend for listed players re-ordered by a
    per-tier model/ADP rank shrink and mapped back onto the blend's own sorted totals, so magnitudes are unchanged).

    Returns ``frame`` (projected rows of the seasons that could be fit) with one ``arm_*`` column per arm.
    """
    from src.value.adp_blend import blend_totals, fit_adp_blend

    seasons = sorted(frame["season"].unique(), key=season_start)
    dis = disagreement_frame(frame, desc)
    out = []
    for s in seasons:
        prior_s = [p for p in seasons if season_start(p) < season_start(s)]
        prior = dis[dis["season"].isin(prior_s)]
        if len(prior) < min_rows:
            continue
        frames = {p: frame[frame["season"] == p] for p in prior_s}
        blend = fit_adp_blend(frames, model="study")
        coef = feature_blend_fit(prior)
        weights = fit_shrink_weights(prior)
        f = frame[(frame["season"] == s) & frame["projected"]].copy().reset_index(drop=True)
        f["arm_model"] = f["proj_total_fp"].astype(float)
        listed = f["adp"].notna().to_numpy()
        b = f["proj_total_fp"].to_numpy(float).copy()
        b[listed] = blend_totals(f.loc[listed], f.loc[listed, ["player_id", "adp"]], blend).to_numpy()
        f["arm_blend"] = b
        cur = dis[dis["season"] == s].set_index("player_id")
        fe, sh = b.copy(), b.copy()
        if listed.any():
            sub = cur.reindex(f.loc[listed, "player_id"]).reset_index()
            fe[listed] = np.maximum(_feature_design(sub) @ coef, 0.0)
            # re-rank listed players by the shrunk rank; assign them the blend's own totals in that new order
            score = rank_shrink(sub, weights)
            vals = np.sort(b[listed])[::-1]
            order = np.argsort(-score, kind="stable")
            new = np.empty(len(vals))
            new[order] = vals
            sh[listed] = new
        f["arm_blend_feat"], f["arm_blend_shrink"] = fe, sh
        out.append(f)
    if not out:
        raise ValueError("no season has enough earlier ADP-listed rows to fit")
    return pd.concat(out, ignore_index=True)


ARMS = ("model", "blend", "blend_feat", "blend_shrink")


def as_results(arms: pd.DataFrame, full: pd.DataFrame, *, ks: Sequence[int] = (12, 24, 50, 100)) -> dict:
    """One ``BacktestResult`` per arm over the same seasons and universe, so ``ablation.paired_lift`` applies.

    The universe is ``full`` (every eval row of each scored season: projected and played-not-projected), with
    ``proj_total_fp`` replaced by the arm's value for projected rows.
    """
    from src.backtest import metrics as M
    from src.backtest.harness import BacktestResult

    seasons = sorted(arms["season"].unique(), key=season_start)
    res = {}
    for arm in ARMS:
        frames, rows = [], []
        for s in seasons:
            f = full[full["season"] == s].drop(columns=["adp"], errors="ignore").copy()
            a = arms[arms["season"] == s].set_index("player_id")[f"arm_{arm}"]
            f["proj_total_fp"] = np.where(f["projected"], f["player_id"].map(a), np.nan)
            for c in ("band_games", "p50_games", "n_below_p10", "n_above_p90", "n_below_p50"):
                f[c] = 0.0                      # rank-only arms: no game-level bands to calibrate
            m = M.compute_season_metrics(f, ks=ks, rank_only=True)
            m["season"] = s
            rows.append(m)
            frames.append(f)
        sm = pd.DataFrame(rows).set_index("season")
        res[arm] = BacktestResult(arm, list(seasons), sm, pd.concat(frames, ignore_index=True),
                                  {"ks": list(ks), "min_gp": M.DEFAULT_MIN_GP, "rank_only": True}, {})
    return res


def lift_table(results: Mapping, base: str, arms: Sequence[str], metrics: Sequence[str], *,
               n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    from src.backtest.ablation import paired_lift
    rows = []
    for arm in arms:
        for m in metrics:
            pl = paired_lift(results[base], results[arm], m, n_boot=n_boot, seed=seed)
            rows.append({"arm": arm, "vs": base, "metric": m, "lift": pl.estimate, "lo": pl.lo, "hi": pl.hi,
                         "wins": f"{pl.wins}/{pl.n_seasons}", "verdict": pl.verdict})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- CLI

def _load_descriptor_tables():
    from src.store import read_table
    gl = read_table("game_logs")
    pl = read_table("players")
    try:
        from src.ingest.nba_offseason import read_offseason_logs
        ol = read_offseason_logs()
    except (ImportError, FileNotFoundError):
        ol = None
    try:
        from src.ingest.wiki_contracts import read_player_contracts
        ct = read_player_contracts()
    except (ImportError, FileNotFoundError):
        ct = None
    return gl, pl, ol, ct


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.adp_gap", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="walk-forward the model once and cache its eval frames (memory heavy)")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--seasons", default=DEFAULT_SEASONS)
    c.add_argument("--out", type=Path, default=None)
    r = sub.add_parser("report", help="descriptive tables and the pre-registered corrections from a cached frame")
    r.add_argument("frame", type=Path)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--min-gap", type=int, default=10)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if args.cmd == "collect":
        print(collect(args.model, args.seasons, args.out))
        return 0
    from src.backtest.mdutil import df_to_markdown

    frame = pd.read_parquet(args.frame)
    gl, pl, ol, ct = _load_descriptor_tables()
    desc = player_descriptors(frame, gl, pl, ol, ct)
    del gl
    d = disagreement_frame(frame, desc)
    for name, t in win_tables(d, min_gap=args.min_gap).items():
        print(f"\n## ADP side wins: by {name} (|rank_delta| >= {args.min_gap})\n")
        print(df_to_markdown(t.round(3), index=False))
    arms = corrected_totals(frame, desc)
    full = frame[frame["season"].isin(arms["season"].unique())]
    res = as_results(arms, full)
    metrics = ["top12_hit", "top24_hit", "top50_hit", "top100_hit", "spearman_total_fp", "ndcg_50"]
    t = lift_table(res, "blend", ["blend_feat", "blend_shrink"], metrics, n_boot=args.n_boot)
    print("\n## Corrections vs the ADR 0032 blend (paired bootstrap, stratified by season)\n")
    print(df_to_markdown(t.round(4), index=False))
    t2 = lift_table(res, "model", ["blend"], metrics, n_boot=args.n_boot)
    print("\n## Blend vs model (context)\n")
    print(df_to_markdown(t2.round(4), index=False))
    means = pd.DataFrame({a: r.season_metrics[metrics].mean() for a, r in res.items()})
    print("\n## Season means per arm\n")
    print(df_to_markdown(means.round(4).reset_index().rename(columns={"index": "metric"}), index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
