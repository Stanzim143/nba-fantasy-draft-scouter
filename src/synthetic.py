"""Deterministic synthetic NBA league that satisfies the data contract.

Purpose: let model, value, backtest and app code be built and unit-tested before (and
independently of) real data. It has the structure real data has (age curves, roles,
injuries, rookies, retirements, box-score identities) with a known generating process,
so tests can assert that a model recovers real signal.

It is NOT a substitute for real data in any reported result.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.contracts import season_str, validate_table

BASE_MPG = np.array([34, 32, 30, 28, 26, 22, 20, 17, 14, 11, 8, 6, 4, 2.5], dtype=float)
TEAM_ID0 = 1610612737
PLAYER_ID0 = 2_000_000
ROSTER_CAP = len(BASE_MPG)


def _age_delta(age: np.ndarray) -> np.ndarray:
    """Yearly change in latent skill by age: growth, plateau, decline."""
    return np.select([age < 23, age < 26, age < 30], [0.12, 0.05, 0.0], default=-0.08)


def make_synthetic_tables(
    first_start: int = 2015,
    last_start: int = 2018,
    n_teams: int = 30,
    games_per_team: int = 82,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Return ``{'game_logs','team_games','players','player_season_bio'}`` for the given seasons."""
    if n_teams % 2:
        raise ValueError("n_teams must be even")
    if last_start < first_start:
        raise ValueError("last_start must be >= first_start")
    rng = np.random.default_rng(seed)
    target_pool = n_teams * 13

    pool = _new_players(rng, 0, target_pool, first_start, rookies=False)
    everyone = [pool]
    next_idx = target_pool
    team_of: dict[int, int] = {}

    logs, team_games, bios = [], [], []
    for start in range(first_start, last_start + 1):
        season = season_str(start)
        if start > first_start:
            pool, rookies = _advance_season(rng, pool, next_idx, target_pool, start)
            next_idx += len(rookies)
            everyone.append(rookies)
        team_of = _assign_teams(rng, pool, team_of, n_teams)
        pool["team_id"] = pool["player_id"].map(team_of)

        season_logs, opp_of, home_of = _simulate_season(rng, pool, season, start, n_teams, games_per_team)
        logs.append(season_logs)
        team_games.append(_team_games(season_logs, opp_of, home_of, season, start, n_teams, games_per_team))

        b = pool[pool["player_id"].isin(set(season_logs["player_id"]))]
        bios.append(pd.DataFrame({
            "season": season,
            "player_id": b["player_id"].astype("int64").to_numpy(),
            "age_at_season_start": ((pd.Timestamp(start, 10, 1) - b["birthdate"]).dt.days / 365.25).to_numpy(),
            "team_id": (TEAM_ID0 + b["team_id"]).astype("int64").to_numpy(),
        }))

    game_logs = pd.concat(logs, ignore_index=True)
    out = {
        "game_logs": game_logs,
        "team_games": pd.concat(team_games, ignore_index=True),
        "players": _players_table(game_logs, pd.concat(everyone, ignore_index=True)),
        "player_season_bio": pd.concat(bios, ignore_index=True),
    }
    for name, df in out.items():
        validate_table(df, name)
    return out


# --------------------------------------------------------------------------- population

def _new_players(rng, first_idx: int, n: int, start: int, rookies: bool) -> pd.DataFrame:
    ids = np.arange(PLAYER_ID0 + first_idx, PLAYER_ID0 + first_idx + n)
    if rookies:
        age = rng.uniform(19, 23, n)
        skill = rng.normal(-0.3, 0.8, n)
    else:
        age = rng.uniform(20, 36, n)
        skill = rng.normal(0, 1, n)
    big = rng.beta(1.5, 1.5, n)
    drafted = rng.random(n) < 0.85
    years_ago = np.zeros(n, dtype=int) if rookies else rng.integers(0, 5, n)
    return pd.DataFrame({
        "player_id": ids.astype("int64"),
        "player_name": [f"Synth Player {i - PLAYER_ID0:05d}" for i in ids],
        "birthdate": pd.to_datetime([date(start, 10, 1) - timedelta(days=int(a * 365.25)) for a in age]),
        "skill": skill,
        "big": big,
        "usage": rng.normal(0, 1, n),
        "three": rng.beta(2, 4, n),
        "ft_skill": rng.normal(0, 1, n),
        "inj": rng.uniform(0.004, 0.03, n),
        "height_in": 72 + 12 * big + rng.normal(0, 1.5, n),
        "weight_lb": 180 + 70 * big + rng.normal(0, 8, n),
        "draft_year": np.where(drafted, start - years_ago, -1),
        "draft_round": np.where(rng.random(n) < 0.6, 1, 2),
        "draft_number": rng.integers(1, 60, n),
    })


def _advance_season(rng, pool, next_idx, target_pool, start):
    """Age everyone a year, retire some, add rookies to keep the league size steady."""
    age = ((pd.Timestamp(start, 10, 1) - pool["birthdate"]).dt.days / 365.25).to_numpy()
    pool = pool.copy()
    pool["skill"] = pool["skill"] + _age_delta(age) + rng.normal(0, 0.15, len(pool))
    p_retire = 0.02 + 0.3 * (age > 34) + 0.1 * (pool["skill"].to_numpy() < -1.0)
    pool = pool[rng.random(len(pool)) > p_retire].reset_index(drop=True)
    rookies = _new_players(rng, next_idx, max(target_pool - len(pool), 0), start, rookies=True)
    return pd.concat([pool, rookies], ignore_index=True), rookies


def _assign_teams(rng, pool, team_of, n_teams):
    """Most players stay put; teams stay balanced at <= ROSTER_CAP players."""
    roster: dict[int, list[int]] = {t: [] for t in range(n_teams)}
    floaters: list[int] = []
    for pid in rng.permutation(pool["player_id"].to_numpy()):
        pid = int(pid)
        prev = team_of.get(pid)
        if prev is not None and rng.random() < 0.85 and len(roster[prev]) < ROSTER_CAP - 1:
            roster[prev].append(pid)
        else:
            floaters.append(pid)
    for pid in floaters:
        t = min(roster, key=lambda k: (len(roster[k]), rng.random()))
        roster[t].append(pid)
    return {pid: t for t, pids in roster.items() for pid in pids}


# --------------------------------------------------------------------------- season sim

def _simulate_season(rng, pool, season, start, n_teams, G):
    n = len(pool)
    skill = pool["skill"].to_numpy()
    team = pool["team_id"].to_numpy()

    # Role: rank within team by (noisy) skill -> base minutes.
    base = np.zeros(n)
    for t in range(n_teams):
        idx = np.where(team == t)[0]
        order = idx[np.argsort(-skill[idx] + rng.normal(0, 0.3, len(idx)))]
        base[order] = BASE_MPG[: len(order)]

    # Availability: injury spells (geometric length) + occasional rest days.
    avail = np.ones((n, G), dtype=bool)
    remaining = np.zeros(n, dtype=int)
    inj = pool["inj"].to_numpy()
    for g in range(G):
        starting = (remaining == 0) & (rng.random(n) < inj)
        remaining[starting] = 1 + rng.geometric(1 / 9, starting.sum())
        avail[:, g] = remaining == 0
        remaining = np.maximum(remaining - 1, 0)
    avail &= rng.random((n, G)) > 0.04
    avail &= (base > 0)[:, None]

    mins = np.clip(base[:, None] * rng.lognormal(0, 0.18, (n, G)), 1.0, 44.0)
    mins = np.where(avail, mins, 0.0)

    big = pool["big"].to_numpy()[:, None]
    usage = pool["usage"].to_numpy()[:, None]
    three = pool["three"].to_numpy()[:, None]
    ftsk = pool["ft_skill"].to_numpy()[:, None]
    s = skill[:, None]
    guard = 1 - big

    def pois(per36):
        return rng.poisson(np.clip(per36, 0.05, None) / 36.0 * mins)

    fga = pois(14 + 3 * usage + 2 * s)
    fg3a = rng.binomial(fga, np.clip(three * (0.6 + 0.4 * guard), 0.02, 0.7))
    fg2m = rng.binomial(fga - fg3a, np.clip(0.48 + 0.04 * big + 0.02 * s, 0.35, 0.7))
    fg3m = rng.binomial(fg3a, np.clip(0.35 + 0.015 * s, 0.22, 0.45))
    fgm = fg2m + fg3m
    fta = pois(3.5 + 1.2 * usage + 0.6 * s)
    ftm = rng.binomial(fta, np.clip(0.72 + 0.04 * ftsk - 0.06 * big, 0.45, 0.93))
    reb = pois(4 + 6 * big + 0.8 * s)
    oreb = rng.binomial(reb, np.clip(0.08 + 0.2 * big, 0.05, 0.4))
    ast = pois(2 + 4 * guard + 0.8 * s)
    stl = pois(0.8 + 0.5 * guard + 0.2 * s)
    blk = pois(0.3 + 1.6 * big + 0.15 * s)
    tov = pois(1.6 + 0.6 * usage + 0.8 * guard)
    pf = pois(2.6 + 0.4 * big)
    pts = 2 * fgm + fg3m + ftm

    opp_of, home_of = _schedule(rng, n_teams, G)
    pi, gi = np.where(avail)
    tm = team[pi]
    is_home = home_of[tm, gi]
    opp = opp_of[tm, gi]
    abbr = np.array([f"T{t:02d}" for t in range(n_teams)])
    matchup = np.where(is_home,
                       [f"{abbr[a]} vs. {abbr[b]}" for a, b in zip(tm, opp)],
                       [f"{abbr[a]} @ {abbr[b]}" for a, b in zip(tm, opp)])
    home_team = np.where(is_home, tm, opp)

    def col(a):
        return a[pi, gi].astype("int64")

    logs = pd.DataFrame({
        "season": season,
        "game_id": _game_ids(start, gi, home_team),
        "game_date": pd.Timestamp(start, 10, 22) + pd.to_timedelta(gi * 2, unit="D"),
        "player_id": pool["player_id"].to_numpy()[pi].astype("int64"),
        "player_name": pool["player_name"].to_numpy()[pi],
        "team_id": (TEAM_ID0 + tm).astype("int64"),
        "team_abbr": abbr[tm],
        "matchup": matchup,
        "min": mins[pi, gi].round(2),
        "fgm": col(fgm), "fga": col(fga), "fg3m": col(fg3m), "fg3a": col(fg3a),
        "ftm": col(ftm), "fta": col(fta), "oreb": col(oreb), "dreb": col(reb - oreb),
        "reb": col(reb), "ast": col(ast), "stl": col(stl), "blk": col(blk),
        "tov": col(tov), "pf": col(pf), "pts": col(pts),
        "plus_minus": rng.normal(0, 8, len(pi)).round(0),
    })
    return logs, opp_of, home_of


def _game_ids(start, day, home_team):
    yy = f"{start % 100:02d}"
    return np.array([f"002{yy}{d:03d}{h:02d}" for d, h in zip(day, home_team)])


def _schedule(rng, n_teams, G):
    """opp[t, d] / home[t, d]: each day teams are paired off, so every team plays G games."""
    opp = np.zeros((n_teams, G), dtype=int)
    home = np.zeros((n_teams, G), dtype=bool)
    for d in range(G):
        perm = rng.permutation(n_teams)
        for a, b in zip(perm[0::2], perm[1::2]):
            opp[a, d], opp[b, d] = b, a
            home[a, d], home[b, d] = True, False
    return opp, home


def _team_games(logs, opp_of, home_of, season, start, n_teams, G):
    """Every team's every game, straight from the schedule (so fully-injured teams still appear)."""
    team_idx = np.repeat(np.arange(n_teams), G)
    day = np.tile(np.arange(G), n_teams)
    opp = opp_of[team_idx, day]
    home = home_of[team_idx, day]
    home_team = np.where(home, team_idx, opp)

    scored = (logs.assign(t=logs["team_id"] - TEAM_ID0, d=(logs["game_date"] - pd.Timestamp(start, 10, 22)).dt.days // 2)
              .groupby(["t", "d"])["pts"].sum())
    pts_for = scored.reindex(pd.MultiIndex.from_arrays([team_idx, day]), fill_value=0).to_numpy()
    pts_against = scored.reindex(pd.MultiIndex.from_arrays([opp, day]), fill_value=0).to_numpy()
    return pd.DataFrame({
        "season": season,
        "game_id": _game_ids(start, day, home_team),
        "game_date": pd.Timestamp(start, 10, 22) + pd.to_timedelta(day * 2, unit="D"),
        "team_id": (TEAM_ID0 + team_idx).astype("int64"),
        "team_abbr": [f"T{t:02d}" for t in team_idx],
        "is_home": home,
        "pts_for": pts_for.astype("int64"),
        "pts_against": pts_against.astype("int64"),
    })


def _players_table(game_logs, everyone):
    seasons = game_logs.groupby("player_id")["season"].agg(["min", "max"])
    p = everyone.drop_duplicates("player_id").set_index("player_id").loc[seasons.index]
    pos = np.select([p["big"] < 0.25, p["big"] < 0.45, p["big"] < 0.65, p["big"] < 0.85],
                    ["G", "G-F", "F", "F-C"], default="C")
    drafted = p["draft_year"] > 0
    return pd.DataFrame({
        "player_id": p.index.astype("int64"),
        "player_name": p["player_name"].to_numpy(),
        "birthdate": p["birthdate"].to_numpy(),
        "position": pos,
        "height_in": p["height_in"].round(1).to_numpy(),
        "weight_lb": p["weight_lb"].round(0).to_numpy(),
        # null = undrafted, consistent for all three columns (real ingest reports none of
        # DRAFT_YEAR/DRAFT_ROUND/DRAFT_NUMBER for a player the source never drafted; see
        # src/ingest/quality.py's "null = undrafted / unknown" and src/contracts.py's PLAYERS spec).
        "draft_year": p["draft_year"].where(drafted).astype("Int64").to_numpy(),
        "draft_round": p["draft_round"].where(drafted).astype("Int64").to_numpy(),
        "draft_number": p["draft_number"].where(drafted).astype("Int64").to_numpy(),
        "from_year": seasons["min"].str[:4].astype(int).to_numpy().astype("int64"),
        "to_year": seasons["max"].str[:4].astype(int).to_numpy().astype("int64"),
    }).reset_index(drop=True)
