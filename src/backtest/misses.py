"""Miss analysis: the biggest fantasy-point misses per season, with heuristic explanations.

The tags are *heuristics computed from the data*, not causal findings: they say which observable
change accompanied the miss (many more or fewer games than projected, a new team, a large minutes
swing, a rookie, an age extreme), never why it happened.  A miss can carry several tags; the
``primary`` tag is the first that applies in this priority order:

    rookie/debut > availability > team change > role/minutes > age > unexplained

Definitions (thresholds are parameters):

* error = actual_total_fp - projected_total_fp  (unprojected players count as projected 0)
* **rookie/debut**: no game in any earlier season of ``game_logs``.
* **availability**: the games-played component dominates the error and |GP gap| >= ``gp_gap``.
  The exact decomposition ``error = (actual_fppg - proj_fppg) * actual_gp + proj_fppg * (actual_gp -
  proj_gp)`` gives a per-game-rate part and a games part; availability needs the games part to be
  >= half of the two parts' absolute sum.  Zero-game players are pure availability.
* **team change**: last team of the season differs from last team of the previous season, or the
  player appeared for >= 2 teams that season.
* **role/minutes**: actual minutes per game differs from the projected (else previous-season)
  minutes per game by >= ``mpg_gap``.
* **age**: age at season start >= 33 or <= 22.
* **unexplained**: none of the above (typically shooting-efficiency or usage variance).

This module only runs *after* scoring, so it may look at the target season's data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.backtest.harness import BacktestResult
from src.backtest.mdutil import df_to_markdown
from src.contracts import season_start

TAG_PRIORITY = ["rookie/debut", "availability", "team change", "role/minutes", "age"]
UNEXPLAINED = "unexplained"


@dataclass
class MissAnalysis:
    table: pd.DataFrame          # one row per miss
    tag_summary: pd.DataFrame    # per primary tag: count, share of |error|

    def to_markdown(self, per_season: bool = True) -> str:
        if self.table.empty:
            return "_No misses to report._"
        cols = ["player", "proj_total_fp", "actual_total_fp", "error", "proj_gp", "actual_gp",
                "mpg_delta", "age", "tags"]
        fmt = {"proj_total_fp": "{:.0f}", "actual_total_fp": "{:.0f}", "error": "{:+.0f}",
               "proj_gp": "{:.0f}", "actual_gp": "{:.0f}", "mpg_delta": "{:+.1f}", "age": "{:.1f}"}
        out = ["**Primary tag summary** (share of total absolute miss among the listed players)", "",
               df_to_markdown(self.tag_summary, formats={"share_abs_error": "{:.1%}"}), ""]
        for season, g in self.table.groupby("season", sort=True):
            out += [f"**{season}**", "", df_to_markdown(g[cols], index=False, formats=fmt), ""]
        return "\n".join(out).rstrip()


def _player_side_info(tables: Mapping[str, pd.DataFrame], season: str) -> pd.DataFrame:
    """Previous-season team / mpg, first-ever-season flag, age: per player_id."""
    gl = tables["game_logs"]
    s = season_start(season)
    starts = gl["season"].map(season_start)
    prev = gl[starts == s - 1].sort_values(["game_date", "game_id"], kind="stable")
    g = prev.groupby("player_id")
    prev_info = pd.DataFrame({"prev_team_id": g["team_id"].last(), "prev_mpg": g["min"].mean()})
    earlier = pd.Series(sorted(set(gl.loc[starts < s, "player_id"])), name="player_id")
    info = prev_info.reset_index().merge(earlier.to_frame().assign(has_history=True), on="player_id", how="outer")
    info["has_history"] = info["has_history"].fillna(False).astype(bool)

    bio = tables["player_season_bio"]
    age = bio[bio["season"] == season][["player_id", "age_at_season_start"]].rename(columns={"age_at_season_start": "age"})
    info = info.merge(age, on="player_id", how="outer")
    players = tables["players"]
    if "birthdate" in players.columns:
        bd = players[["player_id", "birthdate"]].dropna()
        approx = (pd.Timestamp(s, 10, 1) - bd["birthdate"]).dt.days / 365.25
        info = info.merge(pd.DataFrame({"player_id": bd["player_id"].to_numpy(), "_age_bd": approx.to_numpy()}),
                          on="player_id", how="left")
        info["age"] = info["age"].where(info["age"].notna(), info["_age_bd"])
        info = info.drop(columns="_age_bd")
    info["has_history"] = info["has_history"].fillna(False).astype(bool)
    return info


def analyze_misses(
    result: BacktestResult,
    tables: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 15,
    gp_gap: float = 20.0,
    mpg_gap: float = 5.0,
) -> MissAnalysis:
    """Top ``top_n`` absolute total-FP misses per season, tagged.  ``tables`` = the full tables."""
    rows = []
    for season in result.seasons:
        f = result.season_frame(season)
        f = f[f["projected"] | f["played"]].copy()
        f["proj_total_filled"] = f["proj_total_fp"].fillna(0.0)
        f["actual_total_filled"] = f["actual_total_fp"].fillna(0.0)
        f["error"] = f["actual_total_filled"] - f["proj_total_filled"]
        f = f.reindex(f["error"].abs().sort_values(ascending=False, kind="stable").index).head(top_n)
        info = _player_side_info(tables, season)
        f = f.merge(info, on="player_id", how="left")
        f["has_history"] = f["has_history"].fillna(False).astype(bool)
        for r in f.itertuples(index=False):
            rows.append(_tag_row(r, season, gp_gap, mpg_gap))
    cols = ["season", "player_id", "player", "proj_total_fp", "actual_total_fp", "error", "proj_gp",
            "actual_gp", "proj_mpg", "actual_mpg", "mpg_delta", "age", "avail_share", "tags", "primary"]
    table = pd.DataFrame(rows, columns=cols)
    if table.empty:
        return MissAnalysis(table, pd.DataFrame(columns=["count", "share_abs_error"]))
    table["_abs"] = table["error"].abs()
    summ = table.groupby("primary").agg(count=("player_id", "size"), _abs=("_abs", "sum"))
    summ["share_abs_error"] = summ["_abs"] / summ["_abs"].sum()
    summ = summ.drop(columns="_abs").sort_values("share_abs_error", ascending=False)
    return MissAnalysis(table.drop(columns="_abs"), summ)


def _tag_row(r, season: str, gp_gap: float, mpg_gap: float) -> dict:
    projected = bool(r.projected)
    actual_gp = 0.0 if pd.isna(r.actual_gp) else float(r.actual_gp)
    proj_gp = float(r.proj_gp) if projected and pd.notna(r.proj_gp) else np.nan
    proj_fppg = float(r.proj_fppg) if projected and pd.notna(r.proj_fppg) else np.nan
    actual_fppg = float(r.actual_fppg) if pd.notna(r.actual_fppg) else np.nan

    gp_comp = rate_comp = np.nan
    if projected and not np.isnan(proj_fppg):
        gp_comp = proj_fppg * (actual_gp - proj_gp)
        rate_comp = 0.0 if np.isnan(actual_fppg) else (actual_fppg - proj_fppg) * actual_gp
    denom = abs(gp_comp) + abs(rate_comp) if not np.isnan(gp_comp) else np.nan
    avail_share = abs(gp_comp) / denom if denom and not np.isnan(denom) else np.nan

    ref_mpg = r.proj_mpg if projected and pd.notna(r.proj_mpg) else r.prev_mpg
    mpg_delta = float(r.actual_mpg) - float(ref_mpg) if pd.notna(r.actual_mpg) and pd.notna(ref_mpg) else np.nan

    tags = []
    if not r.has_history:
        tags.append("rookie/debut")
    if projected and not np.isnan(avail_share) and avail_share >= 0.5 and abs(actual_gp - proj_gp) >= gp_gap:
        tags.append("availability")
    changed = pd.notna(r.prev_team_id) and pd.notna(r.team_id) and r.prev_team_id != r.team_id
    if changed or (pd.notna(r.n_teams) and r.n_teams >= 2):
        tags.append("team change")
    if not np.isnan(mpg_delta) and abs(mpg_delta) >= mpg_gap:
        tags.append("role/minutes")
    if pd.notna(r.age) and (r.age >= 33 or r.age <= 22):
        tags.append("age")
    primary = next((t for t in TAG_PRIORITY if t in tags), UNEXPLAINED)
    return {
        "season": season, "player_id": int(r.player_id), "player": r.player_name,
        "proj_total_fp": np.nan if not projected else float(r.proj_total_fp),
        "actual_total_fp": float(r.actual_total_fp) if pd.notna(r.actual_total_fp) else 0.0,
        "error": float(r.error), "proj_gp": proj_gp, "actual_gp": actual_gp,
        "proj_mpg": float(r.proj_mpg) if projected and pd.notna(r.proj_mpg) else np.nan,
        "actual_mpg": float(r.actual_mpg) if pd.notna(r.actual_mpg) else np.nan,
        "mpg_delta": mpg_delta, "age": float(r.age) if pd.notna(r.age) else np.nan,
        "avail_share": avail_share,
        "tags": ", ".join(tags) if tags else UNEXPLAINED, "primary": primary,
    }
