"""Minutes and usage trends, absences, and who inherits an absent teammate's minutes (from game logs).

    python -m src.inseason.signals --eval --seasons 2021-22:2025-26     # does the redistribution rule hold?

Everything here reads only game-log rows with ``game_date <= as_of`` (leakage tests in
``tests/inseason/test_signals.py``); ESPN injury status, when supplied, is a snapshot the caller vouches for.

Trend signals (``trend_signals``)
    Recent window = a player's last ``window`` (5) games; baseline = his earlier games this season (needs
    8+). ``min_delta`` = recent minus baseline minutes; ``min_z`` scales it by the baseline's game-to-game
    minute spread (floored at 3 minutes). ``rising_minutes`` needs ``min_delta >= 3`` and ``min_z >= 1.5``.
    Usage is a per-36 volume proxy ``(FGA + 0.44 FTA + TOV) / MIN * 36``; ``rising_usage`` needs a rise of at
    least 2.0 per 36 and the same z test. These are flags for a human to look at, not model inputs (ADR 0010
    and 0011 tested roster-context features as *model* inputs and found no lift; this is only an alert).

Absences (``absent_players``)
    A rotation player (season-to-date mpg >= 18 over 8+ games) whose team has played at least one game since
    his last appearance. An ESPN ``OUT`` status also counts, even before the season. ``streak`` is the
    number of team games missed in a row, ``long_term`` is a streak of 10+.

Beneficiaries (``beneficiaries``)
    Rule: the absent player's minutes go to his teammates in proportion to their *headroom* (36 minus their
    own mpg) times ``compat``, which is 1.5 for the same position group and 1.0 otherwise, each teammate
    capped at 36 minutes. (Weighting by current minutes instead was tried first and is anti-correlated with
    reality, spearman -0.13: starters already play 34 and gain little, the 15-minute players gain most.) When the
    absent player has already missed 3+ games this season, the rule's share is blended with the teammate's
    observed with/without minutes (weight ``n_out / (n_out + 4)``). ``python -m src.inseason.signals --eval``
    measures how well the rule alone tracks real with/without minute changes; the result is in
    ``docs/inseason.md``.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.contracts import season_start, seasons_between
from src.value.positions import position_group

WINDOW = 5
MIN_BASE_GAMES = 8
MIN_DELTA = 3.0
USG_DELTA = 2.0
Z_MIN = 1.5
ROTATION_MPG = 18.0
LONG_TERM_STREAK = 10
SAME_GROUP_BOOST = 1.5
MINUTES_CAP = 36.0
WITH_WITHOUT_K = 4.0


def usage36(df: pd.DataFrame) -> pd.Series:
    return (df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["min"].where(df["min"] > 0) * 36.0


def trend_signals(gl: pd.DataFrame, *, window: int = WINDOW, min_base: int = MIN_BASE_GAMES) -> pd.DataFrame:
    """Per-player recent-vs-baseline minutes, usage and FPPG. ``gl`` = this season's logs up to ``as_of``
    with a ``fp`` column."""
    cols = ["player_id", "recent_mpg", "base_mpg", "min_delta", "min_z", "recent_usg", "base_usg", "usg_delta",
            "usg_z", "recent_fppg", "base_fppg", "n_recent", "n_base", "rising_minutes", "rising_usage"]
    if gl.empty:
        return pd.DataFrame(columns=cols)
    g = gl.sort_values(["player_id", "game_date", "game_id"], kind="mergesort").copy()
    g["usg"] = usage36(g)
    g["rank_from_end"] = g.groupby("player_id").cumcount(ascending=False)
    recent = g[g["rank_from_end"] < window]
    base = g[g["rank_from_end"] >= window]
    r = recent.groupby("player_id").agg(recent_mpg=("min", "mean"), recent_usg=("usg", "mean"),
                                        recent_fppg=("fp", "mean"), n_recent=("min", "size"))
    b = base.groupby("player_id").agg(base_mpg=("min", "mean"), base_sd=("min", "std"), base_usg=("usg", "mean"),
                                      usg_sd=("usg", "std"), base_fppg=("fp", "mean"), n_base=("min", "size"))
    out = r.join(b, how="left")
    out = out[out["n_base"].fillna(0) >= min_base].copy()
    out["min_delta"] = out["recent_mpg"] - out["base_mpg"]
    out["usg_delta"] = out["recent_usg"] - out["base_usg"]
    se = lambda sd: np.maximum(sd.fillna(3.0), 3.0) / np.sqrt(out["n_recent"])  # noqa: E731
    out["min_z"] = out["min_delta"] / se(out["base_sd"])
    out["usg_z"] = out["usg_delta"] / (np.maximum(out["usg_sd"].fillna(3.0), 3.0) / np.sqrt(out["n_recent"]))
    out["rising_minutes"] = (out["min_delta"] >= MIN_DELTA) & (out["min_z"] >= Z_MIN)
    out["rising_usage"] = (out["usg_delta"] >= USG_DELTA) & (out["usg_z"] >= Z_MIN)
    return out.reset_index()[cols]


def _chronological(gl: pd.DataFrame) -> pd.DataFrame:
    """Game logs in (game_date, game_id) order, so ``groupby(...).agg(team_id="last")`` is the player's *latest* team."""
    return gl.sort_values(["game_date", "game_id"], kind="mergesort") if len(gl) else gl


def absent_players(gl: pd.DataFrame, tg: pd.DataFrame, *, injuries: pd.Series | None = None,
                   min_mpg: float = ROTATION_MPG, min_games: int = MIN_BASE_GAMES) -> pd.DataFrame:
    """Rotation players not playing now: ``player_id, team_id, mpg, gp, streak, long_term, source``."""
    rows = []
    gl = _chronological(gl)
    if len(gl):
        per = gl.groupby("player_id").agg(mpg=("min", "mean"), gp=("game_id", "size"), last=("game_date", "max"),
                                          team_id=("team_id", "last"))
        rot = per[(per["mpg"] >= min_mpg) & (per["gp"] >= min_games)]
        tg_dates = tg.groupby("team_id")["game_date"].apply(lambda s: np.sort(pd.to_datetime(s).to_numpy()))
        for pid, r in rot.iterrows():
            dates = tg_dates.get(r["team_id"])
            if dates is None:
                continue
            streak = int((dates > np.datetime64(r["last"])).sum())
            if streak >= 1:
                rows.append({"player_id": int(pid), "team_id": int(r["team_id"]), "mpg": float(r["mpg"]),
                             "gp": int(r["gp"]), "streak": streak, "source": "game log"})
    out = pd.DataFrame(rows, columns=["player_id", "team_id", "mpg", "gp", "streak", "source"])
    if injuries is not None and len(injuries) and len(gl):
        known = set(out["player_id"])
        per = gl.groupby("player_id").agg(mpg=("min", "mean"), gp=("game_id", "size"), team_id=("team_id", "last"))
        for pid, st in injuries.items():
            if st == "OUT" and pid not in known and pid in per.index and per.at[pid, "mpg"] >= min_mpg:
                r = per.loc[pid]
                out.loc[len(out)] = [int(pid), int(r["team_id"]), float(r["mpg"]), int(r["gp"]), 0, "ESPN OUT"]
    out["long_term"] = out["streak"] >= LONG_TERM_STREAK
    return out.reset_index(drop=True)


def _with_without(gl: pd.DataFrame, tg_team: pd.DataFrame, out_pid: int) -> tuple[pd.Series, int]:
    """Teammate mean minutes in team games the absent player missed minus games he played (this season)."""
    first = gl.loc[(gl["player_id"] == out_pid) & (gl["team_id"] == tg_team["team_id"].iloc[0]), "game_date"].min()         if len(tg_team) else pd.NaT
    if pd.notna(first):    # games before he joined this team (a trade) are not games he missed
        tg_team = tg_team[pd.to_datetime(tg_team["game_date"]) >= pd.Timestamp(first)]
    team_games = tg_team["game_id"].unique()
    played = set(gl.loc[gl["player_id"] == out_pid, "game_id"])
    out_games = [g for g in team_games if g not in played]
    in_games = [g for g in team_games if g in played]
    if len(out_games) < 3 or len(in_games) < 3:
        return pd.Series(dtype=float), len(out_games)
    m = gl[gl["game_id"].isin(team_games) & (gl["player_id"] != out_pid)]
    mo = m[m["game_id"].isin(out_games)].groupby("player_id")["min"].agg(["mean", "size"])
    mi = m[m["game_id"].isin(in_games)].groupby("player_id")["min"].agg(["mean", "size"])
    j = mo.join(mi, lsuffix="_out", rsuffix="_in", how="inner")
    j = j[(j["size_out"] >= 2) & (j["size_in"] >= 3)]
    return (j["mean_out"] - j["mean_in"]), len(out_games)


def allocate_minutes(freed: float, teammates: pd.DataFrame, out_group: str) -> pd.Series:
    """Rule allocation of ``freed`` minutes over ``teammates`` (``player_id`` index; ``mpg``, ``group``)."""
    room = (MINUTES_CAP - teammates["mpg"]).clip(lower=0.0)
    w = room * np.where(teammates["group"] == out_group, SAME_GROUP_BOOST, 1.0)
    gain = pd.Series(0.0, index=teammates.index)
    left = freed
    active = room > 0
    for _ in range(5):  # redistribute what capped players cannot absorb
        if left <= 1e-9 or not active.any():
            break
        share = w[active] / w[active].sum() * left
        take = np.minimum(share, (room - gain)[active])
        gain[active] += take
        left -= float(take.sum())
        active = (room - gain) > 1e-9
    return gain


def beneficiaries(gl: pd.DataFrame, tg: pd.DataFrame, absent: pd.DataFrame, positions: pd.Series,
                  *, fp_per_min: pd.Series | None = None) -> pd.DataFrame:
    """Teammates expected to gain minutes because ``absent`` players are out.

    Columns: ``player_id, team_id, out_player_id, freed_mpg, gain_mpg, gain_fppg, method``. ``positions`` is
    player_id -> position string; ``fp_per_min`` (optional) converts minutes to FP.
    """
    cols = ["player_id", "team_id", "out_player_id", "freed_mpg", "gain_mpg", "gain_fppg", "method"]
    if absent.empty or gl.empty:
        return pd.DataFrame(columns=cols)
    gl = _chronological(gl)
    per = gl.groupby("player_id").agg(mpg=("min", "mean"), gp=("game_id", "size"), team_id=("team_id", "last"))
    per["group"] = [position_group(p) for p in positions.reindex(per.index)]
    out_rows = []
    absent_ids = set(absent["player_id"])
    for _, a in absent.iterrows():
        team = per[(per["team_id"] == a["team_id"]) & (~per.index.isin(absent_ids)) & (per["gp"] >= 3)]
        if team.empty:
            continue
        out_group = position_group(positions.get(a["player_id"]))
        rule = allocate_minutes(float(a["mpg"]), team, out_group)
        emp, n_out = _with_without(gl, tg[tg["team_id"] == a["team_id"]], int(a["player_id"]))
        for pid in team.index:
            g_rule = float(rule.get(pid, 0.0))
            method = "rule"
            g = g_rule
            if pid in emp.index and n_out >= 3:
                wt = n_out / (n_out + WITH_WITHOUT_K)
                g = wt * float(emp[pid]) + (1 - wt) * g_rule
                method = f"rule+with/without (n={n_out})"
            if g > 0.25:
                fpm = float(fp_per_min.get(pid, np.nan)) if fp_per_min is not None else np.nan
                out_rows.append({"player_id": int(pid), "team_id": int(a["team_id"]), "out_player_id": int(a["player_id"]),
                                 "freed_mpg": float(a["mpg"]), "gain_mpg": g,
                                 "gain_fppg": g * fpm if np.isfinite(fpm) else np.nan, "method": method})
    return pd.DataFrame(out_rows, columns=cols).sort_values("gain_mpg", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- evidence for the rule

def evaluate_rule(tables, seasons: list[str], *, min_star_mpg: float = 25.0, min_out: int = 5) -> pd.DataFrame:
    """Event study: for each rotation star with 5+ missed games and 15+ played, compare the rule's predicted
    minute gain for each teammate with the observed with/without minute difference (same season)."""
    gl_all, tg_all, players = tables["game_logs"], tables["team_games"], tables["players"]
    pos = players.drop_duplicates("player_id").set_index("player_id")["position"]
    rows = []
    for season in seasons:
        gl = _chronological(gl_all[gl_all["season"] == season])
        tg = tg_all[tg_all["season"] == season]
        per = gl.groupby("player_id").agg(mpg=("min", "mean"), gp=("game_id", "size"), team_id=("team_id", "last"))
        per["group"] = [position_group(p) for p in pos.reindex(per.index)]
        for pid, s in per[(per["mpg"] >= min_star_mpg) & (per["gp"] >= 15)].iterrows():
            team_games = tg[tg["team_id"] == s["team_id"]]
            emp, n_out = _with_without(gl, team_games, int(pid))
            if n_out < min_out or emp.empty:
                continue
            team = per[(per["team_id"] == s["team_id"]) & (per.index != pid) & (per["gp"] >= 3)]
            in_games = gl[(gl["player_id"] == pid)]["game_id"]
            base = gl[gl["game_id"].isin(in_games) & gl["player_id"].isin(team.index)].groupby("player_id")["min"].mean()
            team = team.assign(mpg=base.reindex(team.index).fillna(team["mpg"]))
            pred = allocate_minutes(float(s["mpg"]), team, s["group"])
            common = emp.index.intersection(pred.index)
            for p in common:
                rows.append({"season": season, "star": int(pid), "mate": int(p), "pred_gain": float(pred[p]),
                             "actual_gain": float(emp[p]), "n_out": n_out})
    return pd.DataFrame(rows)


def summarize_rule(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {}
    by_star = ev.groupby(["season", "star"]).agg(pred=("pred_gain", "sum"), actual=("actual_gain", "sum"))
    sp = ev["pred_gain"].corr(ev["actual_gain"], method="spearman")
    big = ev[ev["pred_gain"] >= 1.5]
    return {"pairs": int(len(ev)), "events": int(len(by_star)), "spearman_pair": float(sp),
            "pearson_pair": float(ev["pred_gain"].corr(ev["actual_gain"])),
            "mean_pred_when_flagged": float(big["pred_gain"].mean()) if len(big) else float("nan"),
            "mean_actual_when_flagged": float(big["actual_gain"].mean()) if len(big) else float("nan"),
            "mean_actual_all": float(ev["actual_gain"].mean()), "pairs_flagged": int(len(big)),
            "share_flagged_positive": float((big["actual_gain"] > 0).mean()) if len(big) else float("nan"),
            "mean_pred_total_per_event": float(by_star["pred"].mean()),
            "mean_actual_total_per_event": float(by_star["actual"].mean())}


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.signals", description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="store_true", required=True, help="run the redistribution-rule event study")
    ap.add_argument("--seasons", default="2021-22:2025-26")
    args = ap.parse_args(argv)
    from src.contracts import HISTORY_TABLES
    from src.store import load_tables

    a, _, b = args.seasons.partition(":")
    seasons = seasons_between(season_start(a), season_start(b or a))
    ev = evaluate_rule(dict(load_tables(HISTORY_TABLES)), seasons)
    s = summarize_rule(ev)
    if not s:
        print("no events found", file=sys.stderr)
        return 2
    print(f"seasons {seasons[0]}..{seasons[-1]}: {s['events']} star-absence events, {s['pairs']} teammate pairs")
    for k, v in s.items():
        if k not in ("events", "pairs"):
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
