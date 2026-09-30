"""Coach analysis: does a coach's system follow him, and who benefits from which system? (ADR 0020)

    python -m src.backtest.coach_report [--players reports/real_.../players.parquet] [--out reports/coach_2026-09-25]

Four questions, each answered from the ten real walk-forward seasons and reported with intervals, never as a headline number
alone (the numbers in ``docs/coaches.md`` are this module's output):

1. :func:`style_persistence` -- year to year, how much of a team's style (star and top-five minutes, rotation depth, pace, three-point
   share, youth share) survives when the coach stays versus when he changes? If style were a roster property the two would match.
2. :func:`coach_transfer` -- when a coach *moves to a new team*, is the team's style shift predicted by his own earlier style? The
   sharper test of "the system follows the coach", because the roster he inherits is not his.
3. :func:`coach_profiles` -- each coach's career style (league-relative), the table a draft-day reader wants.
4. :func:`archetype_persistence` -- do coaches have lasting effects on *kinds of players* (young, veteran, star, bench) beyond what the
   baseline projection already captured? For each player-season the outcome is actual minus projected (minutes, fantasy points per
   game, games); the predictor is that coach's *earlier* mean residual for the same archetype (any team), shrunk. A slope near 0 means
   no usable lasting effect; near 1, a coach's past treatment of an archetype fully repeats. This needs the saved player-level
   backtest frame (``--players``), because the residuals come from the baseline's own walk-forward projections.

Leakage: every "prior" uses seasons strictly before the row's season. Coaches are the season's *opening* coach (what a preseason
decision knows); team-seasons with a mid-season change are excluded from persistence and transfer (the style cannot be attributed).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import HISTORY_TABLES
from src.features.coach import STYLE_COLS, opening_coaches, team_style
from src.models.shrink import wls

N_BOOT = 2000
K_GAMES = 300.0            # shrinkage (in games) for the archetype effect


def _coef_ci(cp: np.ndarray, prev: np.ndarray, cur: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float, float]:
    """Coefficient on the coach's prior style in ``cur ~ 1 + cp + prev`` (this season's team style on the coach's earlier style and
    the team's own last-season style), a bootstrap 95% interval, and a permutation p-value (share of 2000 shuffles of ``cp`` whose
    coefficient is at least as large). Regressing the *change* (cur - prev) on (cp - prev) would put ``prev`` on both sides and,
    because team style mean-reverts, show a positive slope even for a coach with no effect: an independent review found exactly that."""
    def coef(c, p, y):
        X = np.column_stack([np.ones(len(c)), c, p])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][1])

    est = coef(cp, prev, cur)
    rng = np.random.default_rng(seed)
    n = len(cp)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(cp[i]) > 0:
            bs.append(coef(cp[i], prev[i], cur[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    null = np.array([coef(rng.permutation(cp), prev, cur) for _ in range(2000)])
    return est, float(lo), float(hi), float((null >= est).mean())


def _styled(game_logs: pd.DataFrame, bio: pd.DataFrame, team_coaches: pd.DataFrame) -> pd.DataFrame:
    """Team-season style joined to the season's opening coach and whether the season had a single coach."""
    ts = team_style(game_logs, bio)
    oc = opening_coaches(team_coaches)[["season", "team_id", "coach_key", "coach_name", "single"]]
    return ts.merge(oc, on=["season", "team_id"], how="inner")


def style_persistence(styled: pd.DataFrame) -> pd.DataFrame:
    """Season-to-season correlation of league-relative style: same coach both seasons vs a coaching change."""
    nxt = styled.copy()
    nxt["s"] -= 1
    pair = styled.merge(nxt, on=["s", "team_id"], suffixes=("", "_n"))
    single = pair["single"] & pair["single_n"]
    same = pair[single & (pair["coach_key"] == pair["coach_key_n"])]
    new = pair[pair["single"] & (pair["coach_key"] != pair["coach_key_n"])]
    rows = []
    for c in STYLE_COLS:
        rows.append({"style": c, "n_same": len(same), "r_same_coach": float(same[f"{c}_x"].corr(same[f"{c}_x_n"])),
                     "n_new": len(new), "r_new_coach": float(new[f"{c}_x"].corr(new[f"{c}_x_n"]))})
    return pd.DataFrame(rows)


def _coach_prior(styled: pd.DataFrame, col: str, rows: pd.DataFrame | None = None) -> pd.Series:
    """For each of ``rows`` (default: every styled row) its coach's mean league-relative ``col`` over his *earlier* single-coach
    seasons (NaN if none). The result is indexed like ``rows``."""
    rows = styled if rows is None else rows
    one = styled[styled["single"]]
    out = []
    for r in rows.itertuples(index=False):
        prev = one[(one["coach_key"] == r.coach_key) & (one["s"] < r.s)]
        out.append(prev[f"{col}_x"].mean() if len(prev) else np.nan)
    return pd.Series(out, index=rows.index)


def coach_transfer(styled: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """New-coach team-seasons with a measurable coach history: regress the team's style this season on the coach's earlier style and
    the team's own last-season style (all league-relative). The coefficient on the coach's style is how much of his style the team
    takes on beyond what it already had (1 = fully, 0 = none); it is reported with a bootstrap interval and a permutation p-value."""
    prev = styled[["s", "team_id", *[f"{c}_x" for c in STYLE_COLS]]].copy()
    prev["s"] += 1
    prev.columns = ["s", "team_id", *[f"{c}_prev" for c in STYLE_COLS]]
    d = styled.merge(prev, on=["s", "team_id"])
    last_coach = styled[["s", "team_id", "coach_key"]].assign(s=lambda x: x["s"] + 1).rename(columns={"coach_key": "coach_prev"})
    d = d.merge(last_coach, on=["s", "team_id"], how="left")
    d = d[d["single"] & d["coach_prev"].notna() & (d["coach_key"] != d["coach_prev"])].copy()
    rows = []
    for c in STYLE_COLS:
        d[f"{c}_cp"] = _coach_prior(styled, c, d)
        g = d[d[f"{c}_cp"].notna()]
        if len(g) < 8:
            continue
        cp = g[f"{c}_cp"].to_numpy(float)
        prev = g[f"{c}_prev"].to_numpy(float)
        cur = g[f"{c}_x"].to_numpy(float)
        ok = np.isfinite(cp) & np.isfinite(prev) & np.isfinite(cur)
        cp, prev, cur = cp[ok], prev[ok], cur[ok]
        if len(cp) < 8 or np.ptp(cp) == 0:
            continue                                  # no variation to regress on (an axis every team shares): nothing to report
        coef, lo, hi, p = _coef_ci(cp, prev, cur, n_boot)
        rows.append({"style": c, "n": len(cp), "coef": coef, "lo": lo, "hi": hi, "p_perm": p,
                     "follows_coach": bool(lo > 0 and p < 0.05)})
    return pd.DataFrame(rows)


def coach_profiles(styled: pd.DataFrame) -> pd.DataFrame:
    """One row per coach: seasons as an opening coach, teams, and mean league-relative style over single-coach seasons."""
    one = styled[styled["single"]]
    g = one.groupby("coach_key")
    out = g[[f"{c}_x" for c in STYLE_COLS]].mean()
    out.insert(0, "seasons", g.size())
    out.insert(1, "first_season", g["season"].min())
    out.insert(2, "last_season", g["season"].max())
    out.insert(3, "coach", g["coach_name"].last())
    return out.reset_index().sort_values("seasons", ascending=False).reset_index(drop=True)


def archetype_frame(players: pd.DataFrame, game_logs: pd.DataFrame, bio: pd.DataFrame, opening: pd.DataFrame) -> pd.DataFrame:
    """Player-season rows (played and projected) with residuals, archetypes and the season's opening coach.

    ``players`` is the saved player-level backtest frame (``BacktestResult.players``: ``proj_*`` and ``actual_*`` columns, ``team_id``
    = the team he played most for). Archetypes: age (young <= 23, veteran >= 31), role by last-season minutes rank on his team
    (star = top 2, starter = 3rd to 5th, bench = the rest).
    """
    pl = players[players["played"] & players["projected"]].copy()
    pl["s"] = pl["season"].map(lambda x: int(x[:4]))
    pl["res_mpg"] = pl["actual_mpg"] - pl["proj_mpg"]
    pl["res_fppg"] = pl["actual_fppg"] - pl["proj_fppg"]
    pl["res_gp"] = pl["actual_gp"] - pl["proj_gp"]
    gl = game_logs.copy()
    gl["s"] = gl["season"].map(lambda x: int(x[:4]))
    pm = gl.groupby(["s", "player_id", "team_id"]).agg(g=("game_id", "size"), m=("min", "mean")).reset_index()
    pm = pm.sort_values(["s", "player_id", "g"], ascending=[True, True, False]).drop_duplicates(["s", "player_id"])
    pm["rk"] = pm[pm["g"] >= 20].groupby(["s", "team_id"])["m"].rank(ascending=False)
    last = pm[["s", "player_id", "rk"]].rename(columns={"rk": "rk_prev"}).assign(s=lambda x: x["s"] + 1)
    pl = pl.merge(last, on=["s", "player_id"], how="left")
    ages = bio.assign(s=bio["season"].map(lambda x: int(x[:4])))[["s", "player_id", "age_at_season_start"]]
    pl = pl.merge(ages, on=["s", "player_id"], how="left").dropna(subset=["age_at_season_start"])
    pl = pl.merge(opening[["s", "team_id", "coach_key"]], on=["s", "team_id"], how="inner")
    pl["is_young"] = (pl["age_at_season_start"] <= 23).astype(float)
    pl["is_vet"] = (pl["age_at_season_start"] >= 31).astype(float)
    pl["is_star"] = (pl["rk_prev"] <= 2).astype(float)
    pl["is_bench"] = ((pl["rk_prev"] > 5) | pl["rk_prev"].isna()).astype(float)
    pl["w"] = pl["actual_gp"].clip(lower=1)
    pl["cluster"] = pl["season"] + pl["team_id"].astype(str)
    return pl


def archetype_persistence(pl: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """For each outcome and archetype: slope of the residual on the same coach's *earlier* archetype residual (shrunk, any team)."""
    rows = []
    for outcome in ("res_mpg", "res_fppg", "res_gp"):
        for arch in ("is_young", "is_vet", "is_star", "is_bench"):
            d = pl[pl[arch] == 1].copy()
            d["wr"] = d[outcome] * d["w"]
            agg = d.groupby(["coach_key", "s"]).agg(wr=("wr", "sum"), w=("w", "sum")).reset_index()
            by_coach = {k: g.sort_values("s") for k, g in agg.groupby("coach_key")}
            xs, has = [], []
            for r in d.itertuples(index=False):
                g = by_coach.get(r.coach_key)
                prev = g[g["s"] < r.s] if g is not None else None
                W = float(prev["w"].sum()) if prev is not None and len(prev) else 0.0
                xs.append(float(prev["wr"].sum()) / (W + K_GAMES) if W > 0 else 0.0)
                has.append(W > 0)
            d["x"] = xs
            d = d[np.array(has, bool)]
            if len(d) < 80:
                continue
            X = np.column_stack([np.ones(len(d)), d["x"].to_numpy(float)])
            y, w = d[outcome].to_numpy(float), d["w"].to_numpy(float)
            beta = wls(X, y, w)
            cl = d["cluster"].to_numpy()
            ucl = np.unique(cl)
            idx = {c: np.where(cl == c)[0] for c in ucl}
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(n_boot):
                ii = np.concatenate([idx[c] for c in rng.choice(ucl, len(ucl))])
                bs.append(wls(X[ii], y[ii], w[ii])[1])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"outcome": outcome, "archetype": arch[3:], "n": len(d), "slope": float(beta[1]), "lo": float(lo), "hi": float(hi),
                         "sd_x": float(d["x"].std()), "lasting": bool(lo > 0)})
    return pd.DataFrame(rows)


def render(persist: pd.DataFrame, transfer: pd.DataFrame, profiles: pd.DataFrame, arche: pd.DataFrame | None) -> str:
    def t(df: pd.DataFrame, fmt: dict[str, str]) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.itertuples(index=False):
            out.append("| " + " | ".join(fmt.get(c, "{}").format(v) for c, v in zip(cols, r)) + " |")
        return "\n".join(out)

    parts = ["# Coach analysis", "",
             "Generated by `python -m src.backtest.coach_report` from real 2016-17..2025-26 data. Coaches are the season's opening head coach; "
             "seasons with a mid-season change are excluded. See ADR 0020 and docs/coaches.md.", "",
             "## 1. Does a team's style survive a coaching change?", "",
             "Correlation of league-relative style, season t to t+1, same team.", "",
             t(persist, {"r_same_coach": "{:.2f}", "r_new_coach": "{:.2f}"}), "",
             "## 2. Does the system follow the coach when he changes teams?", "",
             "Coefficient on the coach's earlier style in (team style this season ~ coach's earlier style + team's last style); 1 = the team fully "
             "takes on his style, 0 = none. 95% bootstrap interval over new-coach team-seasons, and the permutation p-value (share of shuffles of the "
             "coach's style that do at least as well). `follows_coach` needs both.", "",
             t(transfer, {"coef": "{:+.2f}", "lo": "{:+.2f}", "hi": "{:+.2f}", "p_perm": "{:.3f}"}), ""]
    if arche is not None and len(arche):
        parts += ["## 3. Do coaches have lasting effects on kinds of players (beyond the baseline)?", "",
                  "Slope of a player's (actual - baseline projection) on the same coach's earlier mean residual for the same archetype. "
                  "12 comparisons: expect about one to clear zero by chance at 95%.", "",
                  t(arche, {"slope": "{:+.2f}", "lo": "{:+.2f}", "hi": "{:+.2f}", "sd_x": "{:.2f}"}), ""]
    prof = profiles[profiles["seasons"] >= 3].copy()
    prof = prof[["coach", "seasons", "first_season", "last_season", *[f"{c}_x" for c in STYLE_COLS]]]
    parts += ["## 4. Coach profiles (3+ seasons as opening coach)", "",
              "League-relative career means: `star` / `top5` in minutes, `depth10` in players, `pace` in possessions, `three` / `young` / `old` as shares.", "",
              t(prof.round(3), {}), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.backtest.coach_report", description=__doc__.split("\n\n")[0])
    p.add_argument("--players", type=Path, default=None, help="saved player-level backtest frame (players.parquet) for the archetype test")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src.ingest.wiki_coaches import read_team_coaches
    from src.store import load_tables

    tables = load_tables(HISTORY_TABLES, base=args.data_dir)
    coaches = read_team_coaches(args.data_dir)
    styled = _styled(tables["game_logs"], tables["player_season_bio"], coaches)
    persist, transfer, profiles = style_persistence(styled), coach_transfer(styled, args.n_boot), coach_profiles(styled)
    arche = None
    if args.players is not None:
        pl = archetype_frame(pd.read_parquet(args.players), tables["game_logs"], tables["player_season_bio"], opening_coaches(coaches))
        arche = archetype_persistence(pl)
    text = render(persist, transfer, profiles, arche)
    out = args.out or Path("reports") / f"coach_{date.today():%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coach_analysis.md").write_text(text, encoding="utf-8")
    persist.to_csv(out / "style_persistence.csv", index=False)
    transfer.to_csv(out / "coach_transfer.csv", index=False)
    profiles.to_csv(out / "coach_profiles.csv", index=False)
    if arche is not None:
        arche.to_csv(out / "archetype_persistence.csv", index=False)
    print(text)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
