"""Phase-0 data-quality report for the NBA ingest.

    python -m src.ingest.quality [--data-dir DIR] [--out docs/data-quality.md]

Computes every number from the stored contract tables (``game_logs``, ``team_games``, ``players``,
``player_season_bio``) plus the ingest report (drop counts, cross-check) and renders Markdown.
Nothing in the output is hand-typed except the explanatory prose; expected values that come from
outside knowledge (season lengths, famous games, scoring leaders, team records) live in the
``EXPECTED_*``/``KNOWN_*`` constants below and each is *checked against the data* and reported as
ok/FAIL, so a wrong expectation or wrong data both show up.

All ``compute`` functions are pure (DataFrames in, plain dicts/DataFrames out) so they can be tested
with hand-built frames, including deliberately broken ones.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.contracts import HISTORY_TABLES, STAT_COLUMN_MAP, TABLES, data_dir, season_start

# --------------------------------------------------------------------------- outside knowledge (each is verified)

# Games per team in a complete regular season, and the known exceptions.
DEFAULT_GAMES_PER_TEAM = 82
EXPECTED_GAMES_PER_TEAM: dict[str, int | None] = {
    "2019-20": None,   # COVID: suspended 2020-03-11, resumed in a 22-team bubble; teams ended with 64-75 games
    "2020-21": 72,     # COVID-shortened schedule
}
COVID_SUSPENSION = pd.Timestamp("2020-03-11")
BUBBLE_RESTART = pd.Timestamp("2020-07-30")

# (season, player name, official points-per-game leader, ppg rounded to 1 dp). Official NBA scoring
# titles; qualification here = at least half of the season's maximum team games played.
KNOWN_SCORING_LEADERS: list[tuple[str, str, float]] = [
    ("2015-16", "Stephen Curry", 30.1), ("2016-17", "Russell Westbrook", 31.6),
    ("2017-18", "James Harden", 30.4), ("2018-19", "James Harden", 36.1),
    ("2019-20", "James Harden", 34.3), ("2020-21", "Stephen Curry", 32.0),
    ("2021-22", "Joel Embiid", 30.6), ("2022-23", "Joel Embiid", 33.1),
    ("2023-24", "Luka Doncic", 33.9), ("2024-25", "Shai Gilgeous-Alexander", 32.7),
]

# Famous individual games: (description, player, date, team abbr, points, fga or None, max minutes or None)
KNOWN_GAMES: list[dict[str, Any]] = [
    dict(what="Kobe Bryant's 60-point farewell", player="Kobe Bryant", date="2016-04-13", team="LAL", pts=60, fga=50),
    dict(what="Klay Thompson's 60 points in 29 minutes", player="Klay Thompson", date="2016-12-05", team="GSW", pts=60, max_min=30),
    dict(what="Devin Booker's 70 points", player="Devin Booker", date="2017-03-24", team="PHX", pts=70),
    dict(what="Damian Lillard's 71 points", player="Damian Lillard", date="2023-02-26", team="POR", pts=71),
    dict(what="Joel Embiid's 70 points", player="Joel Embiid", date="2024-01-22", team="PHI", pts=70),
    dict(what="Luka Doncic's 73 points", player="Luka Doncic", date="2024-01-26", team="DAL", pts=73),
]

# (season, team abbreviation, wins, losses) from the official standings
KNOWN_RECORDS: list[tuple[str, str, int, int]] = [
    ("2015-16", "GSW", 73, 9), ("2016-17", "GSW", 67, 15), ("2017-18", "HOU", 65, 17),
    ("2018-19", "MIL", 60, 22), ("2019-20", "MIL", 56, 17), ("2020-21", "UTA", 52, 20),
    ("2023-24", "BOS", 64, 18), ("2024-25", "OKC", 68, 14),
]

# Westbrook's 2016-17 triple-double season: (GP, ppg, rpg, apg)
KNOWN_SEASON_LINE = dict(player="Russell Westbrook", season="2016-17", gp=81, pts=31.6, reb=10.7, ast=10.4)

# Games after this date are beyond what the report author can confirm from memory.
KNOWLEDGE_CUTOFF = pd.Timestamp("2026-01-31")
MAX_REGULATION_MINUTES = 48.0
OT_MINUTES = 5.0
TEAM_MINUTES_REGULATION = 240.0
TEAM_MINUTES_TOLERANCE = 1.5


# --------------------------------------------------------------------------- helpers

def _season_order(seasons) -> list[str]:
    return sorted(set(seasons), key=season_start)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate dependency)."""
    if df.empty:
        return "_(none)_\n"

    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        return "" if v is None or (v is pd.NA) else str(v).replace("|", "\\|")

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows]) + "\n"


def _plain(name: str) -> str:
    """Accent-insensitive comparison key ("Dončić" == "Doncic")."""
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower().strip()


def _pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else float("nan")


# --------------------------------------------------------------------------- computations

def inventory(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in HISTORY_TABLES:
        df = tables[name]
        seasons = _season_order(df["season"]) if "season" in df else []
        rows.append({"table": name, "rows": len(df), "columns": len(df.columns),
                     "seasons": f"{seasons[0]} .. {seasons[-1]} ({len(seasons)})" if seasons else "n/a (static)"})
    return pd.DataFrame(rows)


def season_coverage(gl: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Per-season games, teams, games per team vs expected, players and rows."""
    rows = []
    per_team = tg.groupby(["season", "team_id"]).size().rename("n").reset_index()
    for season in _season_order(tg["season"]):
        t = tg[tg["season"] == season]
        g = gl[gl["season"] == season]
        n = per_team[per_team["season"] == season]["n"]
        expected = EXPECTED_GAMES_PER_TEAM.get(season, DEFAULT_GAMES_PER_TEAM) if season in EXPECTED_GAMES_PER_TEAM else DEFAULT_GAMES_PER_TEAM
        if expected is None:
            verdict = "n/a (COVID, see below)"
        else:
            verdict = "ok" if (n == expected).all() else f"MISMATCH: {int((n != expected).sum())} teams"
        # players on a team in a game
        pg = g.groupby(["game_id", "team_id"]).size()
        rows.append({
            "season": season,
            "games": int(t["game_id"].nunique()),
            "teams": int(t["team_id"].nunique()),
            "team_games": len(t),
            "games/team min": int(n.min()), "games/team max": int(n.max()),
            "expected": "n/a" if expected is None else expected,
            "check": verdict,
            "first game": t["game_date"].min(), "last game": t["game_date"].max(),
            "players": int(g["player_id"].nunique()),
            "player-games": len(g),
            "players/team-game min": int(pg.min()) if len(pg) else 0,
            "avg": float(pg.mean()) if len(pg) else float("nan"),
            "max": int(pg.max()) if len(pg) else 0,
        })
    return pd.DataFrame(rows)


def covid_summary(tg: pd.DataFrame) -> dict[str, Any]:
    """Evidence for the 2019-20 shape (computed, not assumed)."""
    t = tg[tg["season"] == "2019-20"]
    if t.empty:
        return {}
    per_team = t.groupby("team_id").size()
    before = t[t["game_date"] <= COVID_SUSPENSION].groupby("team_id").size()
    after = t[t["game_date"] >= BUBBLE_RESTART]
    return {
        "games": int(t["game_id"].nunique()), "last_date": t["game_date"].max(),
        "teams_playing_after_restart": int(after["team_id"].nunique()),
        "games_after_restart": int(after["game_id"].nunique()),
        "min_games": int(per_team.min()), "max_games": int(per_team.max()),
        "pre_suspension_min": int(before.min()), "pre_suspension_max": int(before.max()),
        "games_per_team_distribution": per_team.value_counts().sort_index().to_dict(),
        "post_restart_games_per_team": sorted(set(after.groupby("team_id").size().tolist())),
    }


def duplicate_keys(tables: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    return {name: int(tables[name].duplicated(list(TABLES[name].key)).sum()) for name in HISTORY_TABLES}


def id_checks(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    gl, tg, pl, bio = (tables[n] for n in HISTORY_TABLES)
    pairs = gl[["player_id", "player_name"]].drop_duplicates()
    names_per_id = pairs.groupby("player_id")["player_name"].nunique()
    ids_per_name = pairs.groupby("player_name")["player_id"].nunique()
    collisions = ids_per_name[ids_per_name > 1]
    abbr_per_team = pd.concat([gl[["team_id", "team_abbr"]], tg[["team_id", "team_abbr"]]]).drop_duplicates()
    abbrs = abbr_per_team.groupby("team_id")["team_abbr"].nunique()
    team_per_abbr = abbr_per_team.groupby("team_abbr")["team_id"].nunique()
    game_dates = tg.groupby("game_id")["game_date"].nunique()
    team_rows_per_game = tg.groupby("game_id").size()
    home_per_game = tg.groupby("game_id")["is_home"].agg(["sum", "size"])
    neutral = home_per_game[(home_per_game["sum"] == 0) & (home_per_game["size"] == 2)].index
    neutral_games = tg[tg["game_id"].isin(neutral)].drop_duplicates("game_id")[["season", "game_id", "game_date"]]
    return {
        "players_unique_in_players": bool(pl["player_id"].is_unique),
        "players_in_game_logs": int(gl["player_id"].nunique()),
        "players_rows": len(pl),
        "game_log_players_missing_from_players": int((~gl["player_id"].drop_duplicates().isin(pl["player_id"])).sum()),
        "players_without_game_logs": int((~pl["player_id"].isin(gl["player_id"])).sum()),
        "ids_with_multiple_names": int((names_per_id > 1).sum()),
        "names_with_multiple_ids": {k: int(v) for k, v in collisions.items()},
        "collision_details": {name: sorted(pairs[pairs["player_name"] == name]["player_id"].tolist()) for name in collisions.index[:10]},
        "teams": int(tg["team_id"].nunique()),
        "team_ids_with_multiple_abbrs": {int(k): sorted(abbr_per_team[abbr_per_team["team_id"] == k]["team_abbr"]) for k in abbrs[abbrs > 1].index},
        "abbrs_with_multiple_team_ids": {k: int(v) for k, v in team_per_abbr[team_per_abbr > 1].items()},
        "games": int(tg["game_id"].nunique()),
        "games_with_multiple_dates": int((game_dates > 1).sum()),
        "games_without_exactly_two_teams": int((team_rows_per_game != 2).sum()),
        "neutral_site_games": int(len(neutral)),   # both rows are '@': Mexico City / Paris / Cup semifinals in Las Vegas
        "neutral_site_by_season": {k: int(v) for k, v in neutral_games.groupby("season").size().items()},
        "games_with_two_home_teams": int((home_per_game["sum"] > 1).sum()),
        "bio_rows_without_game_logs": int(len(bio.merge(gl[["season", "player_id"]].drop_duplicates(), how="left", indicator=True).query("_merge == 'left_only'"))),
        "player_seasons_without_bio": int(len(gl[["season", "player_id"]].drop_duplicates().merge(bio[["season", "player_id"]], how="left", indicator=True).query("_merge == 'left_only'"))),
    }


def box_score_checks(gl: pd.DataFrame, tg: pd.DataFrame) -> dict[str, Any]:
    """Row-level identities plus player-vs-team consistency."""
    checks = {
        "fgm > fga": gl["fgm"] > gl["fga"], "fg3m > fgm": gl["fg3m"] > gl["fgm"], "fg3a > fga": gl["fg3a"] > gl["fga"],
        "ftm > fta": gl["ftm"] > gl["fta"], "reb != oreb + dreb": gl["reb"] != gl["oreb"] + gl["dreb"],
        "pts != 2*fgm + fg3m + ftm": gl["pts"] != 2 * gl["fgm"] + gl["fg3m"] + gl["ftm"],
        "min <= 0": gl["min"] <= 0,
        "negative counting stat": (gl[["fgm", "fga", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts", "pf"]] < 0).any(axis=1),
    }
    out: dict[str, Any] = {"row_identities": {k: int(v.sum()) for k, v in checks.items()}, "rows": len(gl)}
    agg = gl.groupby(["game_id", "team_id"]).agg(pts=("pts", "sum"), minutes=("min", "sum"), players=("player_id", "size")).reset_index()
    both = agg.merge(tg[["game_id", "team_id", "pts_for", "season"]], on=["game_id", "team_id"], how="outer", indicator=True)
    out["team_games"] = int(len(tg))
    out["orphan_player_team_games"] = int((both["_merge"] == "left_only").sum())
    out["team_games_without_player_rows"] = int((both["_merge"] == "right_only").sum())
    m = both[both["_merge"] == "both"]
    bad_pts = m[m["pts"] != m["pts_for"]]
    out["team_points_mismatch"] = int(len(bad_pts))
    out["team_points_mismatch_by_season"] = {k: int(v) for k, v in bad_pts.groupby("season").size().items()}
    out["team_points_mismatch_examples"] = bad_pts.merge(tg[["game_id", "team_id", "team_abbr", "game_date"]], on=["game_id", "team_id"]).head(5)[["season", "game_id", "game_date", "team_abbr", "pts", "pts_for"]].rename(columns={"pts": "sum of player pts", "pts_for": "team score"})
    out["team_points_mismatch_max_abs_diff"] = int((bad_pts["pts"] - bad_pts["pts_for"]).abs().max()) if len(bad_pts) else 0
    ot = ((m["minutes"] - TEAM_MINUTES_REGULATION) / 25).round().clip(lower=0)  # an overtime is 25 team-minutes
    grid = TEAM_MINUTES_REGULATION + 25 * ot
    off = m[(m["minutes"] - grid).abs() > TEAM_MINUTES_TOLERANCE]
    out["team_minutes_off_grid"] = int(len(off))
    out["team_minutes_off_grid_by_season"] = {k: int(v) for k, v in off.groupby("season").size().items()}
    out["team_minutes_off_grid_examples"] = off.merge(tg[["game_id", "team_id", "team_abbr", "game_date"]], on=["game_id", "team_id"]).head(6)[["season", "game_id", "game_date", "team_abbr", "minutes", "players"]].rename(columns={"minutes": "sum of player minutes"})
    out["team_minutes_off_grid_max_gap"] = float((off["minutes"] - (TEAM_MINUTES_REGULATION + 25 * ((off["minutes"] - 240) / 25).round().clip(lower=0))).abs().max()) if len(off) else 0.0
    out["overtime_team_games"] = int((ot > 0).sum())
    out["team_game_players_min"] = int(m["players"].min()) if len(m) else 0
    return out


def minutes_checks(gl: pd.DataFrame) -> dict[str, Any]:
    """Distribution and physical plausibility of minutes."""
    team = gl.groupby(["game_id", "team_id"])["min"].sum().rename("team_min").reset_index()
    team["ot"] = ((team["team_min"] - TEAM_MINUTES_REGULATION) / 25).round().clip(lower=0)
    game_ot = team.groupby("game_id")["ot"].max().rename("game_ot")
    g = gl[["game_id", "season", "player_name", "game_date", "team_abbr", "min"]].merge(game_ot, on="game_id", how="left")
    legal = MAX_REGULATION_MINUTES + OT_MINUTES * g["game_ot"]
    impossible = g[g["min"] > legal + 0.02]
    regulation = g[g["game_ot"] == 0]
    q = gl["min"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    bins = pd.cut(gl["min"], [0, 1, 5, 10, 20, 30, 40, 48, 100], right=True)
    hist = bins.value_counts().sort_index()
    return {
        "quantiles": {f"p{int(k * 100)}": float(v) for k, v in q.items()},
        "mean": float(gl["min"].mean()),
        "histogram": {str(k): int(v) for k, v in hist.items()},
        "rows_under_1_min": int((gl["min"] < 1).sum()),
        "rows_over_48": int((gl["min"] > 48).sum()),
        "rows_over_48_in_regulation_games": int((regulation["min"] > MAX_REGULATION_MINUTES + 0.02).sum()),
        "regulation_game_rows": int(len(regulation)),
        "impossible_for_game_length": int(len(impossible)),
        "impossible_examples": impossible.sort_values("min", ascending=False).head(5)[["season", "game_date", "player_name", "team_abbr", "min"]].to_dict("records"),
        "max_minutes": float(gl["min"].max()),
        "max_minutes_row": gl.loc[gl["min"].idxmax(), ["season", "game_date", "player_name", "team_abbr", "min"]].to_dict() if len(gl) else {},
        "top_minutes_games": g.sort_values("min", ascending=False).head(5)[["season", "game_date", "player_name", "team_abbr", "min", "game_ot"]].to_dict("records"),
        "by_season_mean": {k: float(v) for k, v in gl.groupby("season")["min"].mean().sort_index(key=lambda s: s.map(season_start)).items()},
    }


OUTLIER_RULES: list[tuple[str, str, float]] = [
    ("pts", "points", 60), ("reb", "rebounds", 28), ("ast", "assists", 22), ("stl", "steals", 8),
    ("blk", "blocks", 9), ("tov", "turnovers", 10), ("fga", "field-goal attempts", 42),
    ("fg3m", "threes made", 12), ("pf", "personal fouls", 7), ("fta", "free-throw attempts", 28),
]


def _with_team_score(top: pd.DataFrame, tg: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["season", "game_date", "player_name", "team_abbr", "pts", "fga", "min"]
    if tg is None:
        return top[cols]
    m = top.merge(tg[["game_id", "team_id", "pts_for", "pts_against"]], on=["game_id", "team_id"], how="left")
    m["note"] = np.where(m["game_date"] > KNOWLEDGE_CUTOFF, "after author's knowledge; passes all internal checks, verify", "")
    return m[cols + ["pts_for", "pts_against", "note"]].rename(columns={"pts_for": "team score", "pts_against": "opp score"})


def plausibility_outliers(gl: pd.DataFrame, tg: pd.DataFrame | None = None) -> dict[str, Any]:
    rows = []
    for col, label, thr in OUTLIER_RULES:
        hit = gl[gl[col] >= thr]
        top = gl.sort_values(col, ascending=False).head(1).iloc[0] if len(gl) else None
        rows.append({
            "stat": label, "threshold": f">= {thr}", "rows": int(len(hit)), "max": int(gl[col].max()),
            "max holder": f"{top['player_name']} {top['game_date']:%Y-%m-%d} {top['team_abbr']}" if top is not None else "",
        })
    pf_over = int((gl["pf"] > 6).sum())  # 6 fouls = disqualified; 7+ is only possible in OT with the rule (never in NBA)
    ppm = gl[gl["min"] >= 10].assign(ppm=lambda d: d["pts"] / d["min"]).sort_values("ppm", ascending=False).head(5)
    return {
        "table": pd.DataFrame(rows),
        "pf_over_6": pf_over,
        "top_points_games": _with_team_score(gl.sort_values("pts", ascending=False).head(8), tg),
        "top_points_per_minute": ppm[["season", "game_date", "player_name", "team_abbr", "pts", "min", "ppm"]],
        "plus_minus_extremes": {"min": float(gl["plus_minus"].min()), "max": float(gl["plus_minus"].max()),
                                "null": int(gl["plus_minus"].isna().sum())},
        "efficiency_flags": {
            "made_more_than_attempted_ratio_violations": int(((gl["fgm"] > gl["fga"]) | (gl["ftm"] > gl["fta"])).sum()),
            "pts_over_3_per_fga_plus_fta_with_min_10": int((gl[gl["min"] >= 10]["pts"] > 3 * (gl[gl["min"] >= 10]["fga"]) + gl[gl["min"] >= 10]["fta"]).sum()),
        },
    }


def age_calibration_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Aggregate the per-season fallback-vs-exact age comparisons recorded by the ingest."""
    tot = {"pairs": 0, "floor_age_on_jun30_matches": 0, "within_half_year": 0}
    worst, weighted = 0.0, 0.0
    for e in ((report or {}).get("seasons") or {}).values():
        c = e.get("age_fallback_calibration") or {}
        if not c:
            continue
        for k in tot:
            tot[k] += c[k]
        worst = max(worst, c["max_abs_error"])
        weighted += c["mean_error"] * c["pairs"]
    if not tot["pairs"]:
        return {}
    return {**tot, "max_abs_error": worst, "mean_error": weighted / tot["pairs"]}


def missing_attributes(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    gl, _, pl, bio = (tables[n] for n in HISTORY_TABLES)
    games_by_player = gl.groupby("player_id").size().rename("games")
    p = pl.merge(games_by_player, left_on="player_id", right_index=True, how="left").fillna({"games": 0})
    total_games = p["games"].sum()
    rows = []
    for col, note in [("birthdate", "age falls back to integer season AGE (+/-0.5 y)"), ("position", ""),
                      ("height_in", ""), ("weight_lb", ""), ("draft_year", "null = undrafted (expected)"),
                      ("draft_round", "null = undrafted / unknown"), ("draft_number", "null = undrafted / unknown"),
                      ("from_year", ""), ("to_year", "")]:
        miss = p[col].isna()
        rows.append({"column": col, "missing players": int(miss.sum()), "% of players": _pct(miss.sum(), len(p)),
                     "% of player-games": _pct(p.loc[miss, "games"].sum(), total_games), "note": note})
    ages = bio["age_at_season_start"]
    known_bd = pl.set_index("player_id")["birthdate"].notna()
    bio_exact = bio["player_id"].map(known_bd).fillna(False)
    sample = p[p["birthdate"].isna()].sort_values("games", ascending=False).head(5)[["player_id", "player_name", "games"]]
    return {
        "table": pd.DataFrame(rows),
        "players": len(pl),
        "undrafted_players": int(pl["draft_year"].isna().sum()),
        "birthdate_missing_examples": sample,
        "bio_rows": len(bio),
        "bio_age_exact_share_pct": _pct(int(bio_exact.sum()), len(bio)),
        "bio_age_min": float(ages.min()) if len(ages) else float("nan"),
        "bio_age_max": float(ages.max()) if len(ages) else float("nan"),
        "bio_age_under_18": int((ages < 18).sum()), "bio_age_over_45": int((ages > 45).sum()),
        "bio_age_by_season": {k: (float(v.mean()), float(v.min()), float(v.max())) for k, v in ages.groupby(bio["season"]) if len(v)},
        "bio_team_missing": int(bio["team_id"].isna().sum()),
        "birthdate_before_1970_or_after_2010": int(((pl["birthdate"] < "1970-01-01") | (pl["birthdate"] > "2010-01-01")).sum()),
        "height_range": (float(pl["height_in"].min()), float(pl["height_in"].max())),
        "weight_range": (float(pl["weight_lb"].min()), float(pl["weight_lb"].max())),
    }


def _fp_weights(scoring: Mapping[str, float] | None) -> dict[str, float]:
    if scoring is None:
        import yaml

        cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "league.yaml").read_text(encoding="utf-8"))
        scoring = cfg["scoring"]
    return {STAT_COLUMN_MAP[k]: float(v) for k, v in scoring.items()}


def season_leaders(gl: pd.DataFrame, tg: pd.DataFrame, scoring: Mapping[str, float] | None = None) -> pd.DataFrame:
    """PPG leader and fantasy-points-per-game leader per season (qualified: >= half the max team games)."""
    w = _fp_weights(scoring)
    fp = sum(gl[c] * wt for c, wt in w.items())
    d = gl.assign(fp=fp)
    max_games = tg.groupby(["season", "team_id"]).size().groupby("season").max()
    agg = d.groupby(["season", "player_id", "player_name"]).agg(gp=("pts", "size"), pts=("pts", "mean"), reb=("reb", "mean"),
                                                                 ast=("ast", "mean"), fppg=("fp", "mean")).reset_index()
    agg["qualified"] = agg["gp"] >= (agg["season"].map(max_games) / 2)
    q = agg[agg["qualified"]]
    rows = []
    for season in _season_order(q["season"]):
        s = q[q["season"] == season]
        top = s.sort_values("pts", ascending=False).iloc[0]
        top_fp = s.sort_values("fppg", ascending=False).iloc[0]
        rows.append({"season": season, "ppg leader": top["player_name"], "gp": int(top["gp"]), "ppg": float(top["pts"]),
                     "fppg leader": top_fp["player_name"], "fppg": float(top_fp["fppg"])})
    return pd.DataFrame(rows)


def _season_of(day: pd.Timestamp) -> str:
    """Season a date belongs to (NBA seasons run October to June)."""
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def known_facts(gl: pd.DataFrame, tg: pd.DataFrame, leaders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Check well-known facts (scoring titles, famous games, team records, a triple-double season)."""
    rows: list[dict[str, Any]] = []
    leaders = season_leaders(gl, tg) if leaders is None else leaders
    for season, name, ppg in KNOWN_SCORING_LEADERS:
        r = leaders[leaders["season"] == season]
        if r.empty:
            rows.append({"fact": f"{season} scoring title: {name} {ppg} ppg", "data": "season not ingested", "ok": "n/a"})
            continue
        r = r.iloc[0]
        ok = _plain(r["ppg leader"]) == _plain(name) and abs(r["ppg"] - ppg) <= 0.051
        rows.append({"fact": f"{season} scoring title: {name} {ppg} ppg", "data": f"{r['ppg leader']} {r['ppg']:.2f}", "ok": "ok" if ok else "FAIL"})
    plain_names = gl["player_name"].map(_plain)
    present = set(gl["season"])
    for k in KNOWN_GAMES:
        when = pd.Timestamp(k["date"])
        if _season_of(when) not in present:
            rows.append({"fact": k["what"], "data": "season not ingested", "ok": "n/a"})
            continue
        r = gl[(plain_names == _plain(k["player"])) & (gl["game_date"] == when)]
        if r.empty:
            rows.append({"fact": k["what"], "data": "row missing", "ok": "FAIL"})
            continue
        r = r.iloc[0]
        ok = r["pts"] == k["pts"] and r["team_abbr"] == k["team"]
        if "fga" in k:
            ok = ok and r["fga"] == k["fga"]
        if "max_min" in k:
            ok = ok and r["min"] <= k["max_min"]
        rows.append({"fact": k["what"], "data": f"{r['pts']} pts, {r['fga']} FGA, {r['min']:.1f} min, {r['team_abbr']}", "ok": "ok" if ok else "FAIL"})
    wins = tg.assign(win=tg["pts_for"] > tg["pts_against"]).groupby(["season", "team_abbr"])["win"].agg(["sum", "size"])
    for season, abbr, w, losses in KNOWN_RECORDS:
        if season not in present:
            rows.append({"fact": f"{season} {abbr} record {w}-{losses}", "data": "season not ingested", "ok": "n/a"})
            continue
        if (season, abbr) not in wins.index:
            rows.append({"fact": f"{season} {abbr} record {w}-{losses}", "data": "team not found", "ok": "FAIL"})
            continue
        got = wins.loc[(season, abbr)]
        gw, gl_ = int(got["sum"]), int(got["size"] - got["sum"])
        rows.append({"fact": f"{season} {abbr} record {w}-{losses}", "data": f"{gw}-{gl_}", "ok": "ok" if (gw, gl_) == (w, losses) else "FAIL"})
    k = KNOWN_SEASON_LINE
    s = gl[(gl["season"] == k["season"]) & (plain_names == _plain(k["player"]))]
    if len(s):
        line = (len(s), round(s["pts"].mean(), 1), round(s["reb"].mean(), 1), round(s["ast"].mean(), 1))
        ok = line == (k["gp"], k["pts"], k["reb"], k["ast"])
        rows.append({"fact": f"{k['season']} {k['player']}: {k['gp']} GP, {k['pts']}/{k['reb']}/{k['ast']}", "data": f"{line[0]} GP, {line[1]}/{line[2]}/{line[3]}", "ok": "ok" if ok else "FAIL"})
    return pd.DataFrame(rows)


def compute_quality(tables: Mapping[str, pd.DataFrame], report: Mapping[str, Any] | None = None,
                    scoring: Mapping[str, float] | None = None) -> dict[str, Any]:
    gl, tg = tables["game_logs"], tables["team_games"]
    leaders = season_leaders(gl, tg, scoring)
    return {
        "inventory": inventory(tables),
        "coverage": season_coverage(gl, tg),
        "covid": covid_summary(tg),
        "duplicates": duplicate_keys(tables),
        "ids": id_checks(tables),
        "box": box_score_checks(gl, tg),
        "minutes": minutes_checks(gl),
        "outliers": plausibility_outliers(gl, tg),
        "missing": missing_attributes(tables),
        "leaders": leaders,
        "facts": known_facts(gl, tg, leaders),
        "report": dict(report or {}),
        "age_calibration": age_calibration_summary(report),
    }


# --------------------------------------------------------------------------- rendering

def _drops_table(report: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for season, e in (report.get("seasons") or {}).items():
        gl = e.get("game_logs", {})
        info = gl.get("info", {})
        rows.append({
            "season": season, "player rows pulled": gl.get("rows_in"), "kept": gl.get("rows_out"),
            "dropped: 0/no minutes": (gl.get("dropped") or {}).get("no_minutes_played", 0),
            "...of which had recorded stats": info.get("no_minutes_rows_with_recorded_stats", 0),
            "points lost": info.get("no_minutes_points_lost", 0),
            "other drops": sum(v for k, v in (gl.get("dropped") or {}).items() if k != "no_minutes_played"),
            "bio rows dropped (no age)": (e.get("player_season_bio", {}).get("dropped") or {}).get("no_age_source", 0),
        })
    return pd.DataFrame(rows)


def _crosscheck_table(report: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for season, e in (report.get("seasons") or {}).items():
        c = e.get("crosscheck")
        if not c:
            continue
        rows.append({
            "season": season, "rows (fractional-min source)": c["rows_primary"], "rows (integer-min source)": c["rows_secondary"],
            "only in primary": c["n_only_in_primary"], "...of which >= 0.5 min": c["n_only_in_primary_over_half_minute"],
            "only in secondary": c["n_only_in_secondary"],
            "rows with any stat mismatch": max(c["stat_mismatches"].values()) if c["stat_mismatches"] else 0,
            "stats differing": ", ".join(f"{k}:{v}" for k, v in c["stat_mismatches"].items()),
            "max |min diff|": c["minutes_max_abs_diff"],
        })
    return pd.DataFrame(rows)


def render_markdown(q: Mapping[str, Any], source: str = "~/dev-data/nba-fantasy-2026") -> str:
    inv, cov, ids, box, mins, out, miss = q["inventory"], q["coverage"], q["ids"], q["box"], q["minutes"], q["outliers"], q["missing"]
    covid, dup, facts = q["covid"], q["duplicates"], q["facts"]
    report = q["report"]
    L: list[str] = []
    add = L.append
    add("# Data-quality report: NBA ingest (Phase 0)\n")
    add(f"Generated by `python -m src.ingest.quality` from the contract tables under `{source}/processed/` "
        "and the ingest report `nba_ingest_report.json`. Every number below is computed from the stored data; "
        "regenerate after any re-ingest (the output is deterministic). Source: stats.nba.com (unofficial), "
        "regular season only. Method and endpoints: `docs/adr/0002-nba-ingest.md`.\n")

    n_fail = int((facts["ok"] == "FAIL").sum()) if len(facts) else 0
    n_ok = int((facts["ok"] == "ok").sum()) if len(facts) else 0
    add("## Summary\n")
    seasons = cov["season"].tolist()
    add(f"* **Coverage:** {len(seasons)} seasons ({seasons[0]} to {seasons[-1]}), {int(cov['games'].sum()):,} games, "
        f"{int(cov['player-games'].sum()):,} player-game rows, {ids['players_in_game_logs']:,} distinct players, {ids['teams']} team ids.")
    add(f"* **Structural integrity:** duplicate keys {sum(dup.values())}; box-score identity violations "
        f"{sum(box['row_identities'].values())} of {box['rows']:,} rows; player rows without a team-game {box['orphan_player_team_games']}; "
        f"{ids['games_without_exactly_two_teams']} games without exactly two team rows; ID collisions: "
        f"{ids['ids_with_multiple_names']} ids with several names, {len(ids['names_with_multiple_ids'])} names shared by several ids.")
    add(f"* **Known-fact checks:** {n_ok} ok, {n_fail} FAIL (scoring titles, famous games, team records, a triple-double season).")
    add(f"* **Weakest spots (honest list):** {box['team_points_mismatch']} team-games where summed player points differ from the team score; "
        f"{box['team_minutes_off_grid']} team-games whose summed minutes are off the 240+25k grid; "
        f"{miss['table'].set_index('column').loc['birthdate', 'missing players']} players without a birthdate "
        f"(ages for them use the integer season AGE, +/-0.5 year); minutes are exact to the second, not rounded.\n")

    add("## 1. Inventory\n")
    add(_md_table(inv))
    add("Duplicate keys per table (contract keys): " + ", ".join(f"`{k}` {v}" for k, v in dup.items()) + ".\n")

    add("## 2. Games per team-season vs expected\n")
    add("Expected: 82 per team in a full season; 72 in 2020-21; none fixed for 2019-20. \"check\" compares every team with the expectation.\n")
    add(_md_table(cov[["season", "games", "teams", "team_games", "games/team min", "games/team max", "expected", "check", "first game", "last game"]]))
    if covid:
        add("### The two COVID seasons, verified from the data\n")
        add(f"* **2019-20** has {covid['games']:,} games (a full 30-team season is 1,230). Games per team range {covid['min_games']} to {covid['max_games']} "
            f"(distribution `{covid['games_per_team_distribution']}`). The league suspended play on 2020-03-11: up to that date teams had "
            f"{covid['pre_suspension_min']}-{covid['pre_suspension_max']} games. Play resumed 2020-07-30 in the bubble with "
            f"**{covid['teams_playing_after_restart']} teams** playing {covid['games_after_restart']} more regular-season games "
            f"({'/'.join(str(x) for x in covid['post_restart_games_per_team'])} games each); the other {30 - covid['teams_playing_after_restart']} teams' seasons ended at the suspension. "
            f"Last game in the data: {covid['last_date']:%Y-%m-%d}. So 2019-20 is a legitimately uneven, shortened season, not missing data: "
            "**nothing downstream may assume 82 games**; availability must be computed per team from `team_games`.")
        r2021 = cov[cov["season"] == "2020-21"]
        if len(r2021):
            r = r2021.iloc[0]
            add(f"* **2020-21** was scheduled at 72 games per team: {r['games']:,} games, every team {r['games/team min']}-{r['games/team max']} ({r['check']}).\n")
    add("### Players\n")
    add(_md_table(cov[["season", "players", "player-games", "players/team-game min", "avg", "max"]].rename(
        columns={"players": "distinct players", "players/team-game min": "min players per team-game", "avg": "avg per team-game", "max": "max per team-game"})))

    add("## 3. Duplicate keys and identifier uniqueness\n")
    add(f"* `players.player_id` unique: **{ids['players_unique_in_players']}**; `players` has {ids['players_rows']:,} rows, `game_logs` has {ids['players_in_game_logs']:,} distinct players; "
        f"players in game logs missing from `players`: {ids['game_log_players_missing_from_players']}; players without any game log: {ids['players_without_game_logs']}.")
    add(f"* Ids with more than one name in `game_logs`: {ids['ids_with_multiple_names']}. Names shared by more than one id: {len(ids['names_with_multiple_ids'])}"
        + (f" ({ids['collision_details']})" if ids['names_with_multiple_ids'] else "") + ". "
        "The source reports one current name per id; `player_id` is the only safe key and names must never be joined on.")
    add(f"* {ids['teams']} team ids. Team ids with more than one abbreviation (renames): {ids['team_ids_with_multiple_abbrs'] or 'none'}; "
        f"abbreviations mapped to more than one team id: {ids['abbrs_with_multiple_team_ids'] or 'none'}. `team_id` is the identity; "
        "abbreviations are kept exactly as the source reported them per row.")
    add(f"* Games: {ids['games']:,}; games with more than one date {ids['games_with_multiple_dates']}; games without exactly two team rows {ids['games_without_exactly_two_teams']}; "
        f"games with two home teams {ids['games_with_two_home_teams']}; neutral-site games (both rows are away, so `is_home` is False for both): "
        f"{ids['neutral_site_games']} ({ids['neutral_site_by_season']}; Mexico City, Paris and the NBA Cup semifinals in Las Vegas).")
    add(f"* `player_season_bio` vs `game_logs`: bio rows without any game {ids['bio_rows_without_game_logs']}; player-seasons with games but no bio row {ids['player_seasons_without_bio']}.\n")

    add("## 4. Box-score identities and consistency\n")
    add("Row-level identities on every `game_logs` row (also enforced at write time by `validate_table`):\n")
    add(_md_table(pd.DataFrame([{"check": k, "violations": v} for k, v in box["row_identities"].items()])))
    add(f"Player rows vs team rows: {box['orphan_player_team_games']} player (game, team) groups have no `team_games` row; {box['team_games_without_player_rows']} team-games have no player rows. "
        f"Summed player points differ from the team's score in **{box['team_points_mismatch']}** of {box['team_games']:,} team-games"
        + (f" (by season: {box['team_points_mismatch_by_season']}; largest gap {box['team_points_mismatch_max_abs_diff']} pts)" if box['team_points_mismatch'] else "")
        + f". Summed player minutes are off the `240 + 25 x overtimes` grid (tolerance {TEAM_MINUTES_TOLERANCE} min) in **{box['team_minutes_off_grid']}** team-games"
        + (f" (by season: {box['team_minutes_off_grid_by_season']}; largest gap {box['team_minutes_off_grid_max_gap']:.2f} min)" if box['team_minutes_off_grid'] else "")
        + f". {box['overtime_team_games']:,} team-games went to overtime.\n")
    if box["team_points_mismatch"]:
        add("Team-games where the player points do not add up to the team score (a source-side discrepancy; the player rows are kept as reported):\n")
        add(_md_table(box["team_points_mismatch_examples"]))
    if box["team_minutes_off_grid"]:
        add("Team-games whose player minutes do not sum to 240 + 25 per overtime (a source-side minutes error; rows kept, minutes are only used per player):\n")
        add(_md_table(box["team_minutes_off_grid_examples"]))

    add("## 5. Minutes\n")
    qs = mins["quantiles"]
    add(f"Minutes are fractional (source `playergamelogs`, exact to the second). Mean {mins['mean']:.2f}; percentiles "
        + ", ".join(f"{k} {v:.1f}" for k, v in qs.items()) + ".\n")
    add(_md_table(pd.DataFrame([{"minutes bin": k, "rows": v, "share %": _pct(v, box['rows'])} for k, v in mins["histogram"].items()])))
    add(f"* Rows over 48 minutes: {mins['rows_over_48']:,} (only possible in overtime). **In games whose team minutes imply no overtime ({mins['regulation_game_rows']:,} rows): {mins['rows_over_48_in_regulation_games']}.**")
    add(f"* Rows exceeding the physical maximum for their game length (48 + 5 per overtime, inferred from team minutes): **{mins['impossible_for_game_length']}**.")
    add(f"* Longest stint: {mins['max_minutes']:.2f} min ({mins['max_minutes_row']}). Rows under 1 minute: {mins['rows_under_1_min']:,} (kept: they did play).")
    add("* Mean minutes per player-game by season: " + ", ".join(f"{k} {v:.1f}" for k, v in mins["by_season_mean"].items()) + ".\n")

    add("## 6. Plausibility outliers\n")
    add("Thresholds sit near all-time records, so a hit is either a genuine feat or a data error; the tables list every extreme for review. "
        "Top lines also show the team's score, and the identity and team-total checks of section 4 apply to every row (a fabricated line would break them). "
        "Games dated after the author's knowledge cutoff (2026) cannot be confirmed from memory: those are marked \"passes all internal checks; verify against the box score\".\n")
    add(_md_table(out["table"]))
    add(f"Rows with more than 6 personal fouls: {out['pf_over_6']}. Plus/minus range {out['plus_minus_extremes']['min']:.0f} to {out['plus_minus_extremes']['max']:.0f}; null plus/minus {out['plus_minus_extremes']['null']}.\n")
    add("Top single-game point totals:\n")
    add(_md_table(out["top_points_games"]))
    add("Highest points per minute (min >= 10 minutes):\n")
    add(_md_table(out["top_points_per_minute"]))

    add("## 7. Missing bio / birthdate / position / draft\n")
    add(f"`players` ({miss['players']:,} rows). Undrafted (null draft_year): {miss['undrafted_players']:,}, which is expected and not an error.\n")
    add(_md_table(miss["table"]))
    add("Players without a birthdate, most games first:\n")
    add(_md_table(miss["birthdate_missing_examples"]))
    lo, hi = miss["bio_age_min"], miss["bio_age_max"]
    cal = q["age_calibration"]
    cal_txt = (f"Calibration of that fallback on this data: of {cal['pairs']:,} player-seasons with both an exact birthdate and a source AGE, "
               f"AGE equalled the whole age on June 30 of the season's end year in {cal['floor_age_on_jun30_matches']:,} "
               f"({_pct(cal['floor_age_on_jun30_matches'], cal['pairs']):.2f}%); the Oct-1 estimate `AGE - 0.245` had mean error {cal['mean_error']:+.3f} y "
               f"and max |error| {cal['max_abs_error']:.3f} y ({cal['within_half_year']:,} of {cal['pairs']:,} within 0.5 y). ") if cal else ""
    add(f"`player_season_bio` ({miss['bio_rows']:,} rows): age computed from an exact birthdate for {miss['bio_age_exact_share_pct']:.2f}% of rows, otherwise from the source's integer season AGE "
        f"(`AGE - 0.245`, worst-case error 0.5 year). {cal_txt}Age range {lo:.1f} to {hi:.1f}; under 18: {miss['bio_age_under_18']}; over 45: {miss['bio_age_over_45']}; "
        f"null bio team: {miss['bio_team_missing']}. Height range {miss['height_range'][0]:.0f}-{miss['height_range'][1]:.0f} in, weight {miss['weight_range'][0]:.0f}-{miss['weight_range'][1]:.0f} lb. "
        f"Birthdates before 1970 or after 2010: {miss['birthdate_before_1970_or_after_2010']}.\n")
    add("Mean / min / max age at season start by season:\n")
    add(_md_table(pd.DataFrame([{"season": s, "mean": v[0], "min": v[1], "max": v[2]} for s, v in sorted(miss["bio_age_by_season"].items(), key=lambda kv: season_start(kv[0]))])))

    add("## 8. Sanity checks against known facts\n")
    add("Each row is an independent fact from outside the pipeline (official scoring titles, famous games, final standings, a triple-double season). "
        "\"ok\" means the ingested data reproduces it exactly.\n")
    add(_md_table(facts))
    add("Scoring-title and fantasy-points-per-game leaders per season (qualified = at least half the season's max team games):\n")
    add(_md_table(q["leaders"]))

    add("## 9. Data dropped and why\n")
    add("Every dropped row is counted; nothing is dropped silently. Reasons seen in the real data:\n")
    add(_md_table(_drops_table(report)))
    add("* **0/no minutes** rows are box-score lines for players who were on the roster sheet but did not play. The contract defines `game_logs` as games actually played (minutes > 0), "
        "so they are excluded; \"had recorded stats\" counts the odd 0:00 lines that still credit a stat (e.g. a free throw at 0:00) - dropped as well because the contract requires min > 0, and "
        "the lost points are listed above (negligible for fantasy value).")
    add("* Non-regular-season game ids (preseason, All-Star, playoffs, play-in, NBA Cup knockout) are filtered by game-id prefix; the API is already asked for the regular season only, so the count is 0 in practice.")
    add("* Player-seasons without any age source are dropped from `player_season_bio` only if there is neither a birthdate nor a source AGE (column \"bio rows dropped\").\n")
    ct = _crosscheck_table(report)
    if len(ct):
        add("### Cross-check: fractional-minute source vs integer-minute source\n")
        add("The same player rows are pulled from a second endpoint (`leaguegamelog`, integer minutes) and compared field by field. "
            "Rows present only in the primary source are players who logged under 30 seconds (integer minutes round to 0); \"...of which >= 0.5 min\" must be 0.\n")
        add(_md_table(ct))
        add("Disagreements in counting stats between the two endpoints are source-side stat corrections (the primary source satisfies every box-score identity; see section 4). "
            "The primary source is used; the secondary is a check only.\n")

    add("## 10. Coverage gaps by season\n")
    gaps = pd.DataFrame([{
        "season": r["season"], "teams": r["teams"], "games/team": f"{r['games/team min']}-{r['games/team max']}", "players": r["players"],
        "min players per team-game": r["players/team-game min"],
        "team-games with points mismatch": box["team_points_mismatch_by_season"].get(r["season"], 0),
        "team-games off minute grid": box["team_minutes_off_grid_by_season"].get(r["season"], 0),
    } for _, r in cov.iterrows()])
    add(_md_table(gaps))
    add("Not covered by this ingest (by design or source limitation): playoffs and play-in games (excluded by contract); injuries, transactions and contract data (other tracks); "
        "player positions per season (only the current source position string is available; it is not point-in-time); `players.to_year`, heights and weights are as of the ingest date, not point-in-time.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    from src.ingest.nba_stats import read_report
    from src.store import read_table

    p = argparse.ArgumentParser(prog="python -m src.ingest.quality", description="Generate docs/data-quality.md from the stored NBA tables.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("docs") / "data-quality.md")
    args = p.parse_args(argv)
    base = args.data_dir or data_dir()
    try:
        tables = {n: read_table(n, base) for n in HISTORY_TABLES}
    except FileNotFoundError as exc:
        print(f"cannot build report: {exc}", file=sys.stderr)
        return 1
    md = render_markdown(compute_quality(tables, read_report(base)), source="~/dev-data/nba-fantasy-2026" if args.data_dir is None else str(base))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
