"""Test-only helper: add an entire team-season roster with EXACT known minutes/position/box lines.

Unlike ``tests.models.model_testkit.add_player_seasons`` (one player at a time, sharing whatever
teammates already exist on that team-season in the synthetic league), this replaces every player's
box-score presence for one team-season with a fully-controlled roster, so pace/role_share/
pos_crowding have a known, hand-computable right answer -- nothing left over from the synthetic
league's own randomly-generated players on that team muddies the possession/minutes totals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_testkit import box_line  # noqa: E402


def add_team_season_roster(tables, season: str, players: list[dict], *, team_of_season=None,
                           birthdate="1996-03-01", draft_year=None) -> tuple[dict, int]:
    """Replace one team's game_logs rows for ``season`` with an exactly-controlled roster.

    Each entry in ``players`` is a dict with keys: ``player_id``, ``name``, ``position``,
    ``minutes`` (per game), ``n_games`` (how many of the team's scheduled games he appears in,
    starting from the first game date), and optionally box_line kwargs (``fga``, ``oreb``, ``tov``,
    ``fta``, etc.) to control the possession estimate precisely.

    Returns ``(new_tables, team_id)``. All *existing* game_logs rows for this team-season are
    dropped first, so the returned tables reflect exactly the given roster (no leftover synthetic
    teammates skewing team-level sums).
    """
    t = {k: v.copy() for k, v in tables.items()}
    tg = t["team_games"]
    team = (tg[tg["season"] == season].groupby("team_id").size().index[0]
            if team_of_season is None else team_of_season)
    sched = tg[(tg["season"] == season) & (tg["team_id"] == team)].sort_values("game_date").reset_index(drop=True)

    # Drop any existing game_logs rows for this team-season so the roster is fully controlled.
    gl = t["game_logs"]
    keep = ~((gl["season"] == season) & (gl["team_id"] == team))
    gl = gl[keep].copy()

    rows = []
    bio_rows = []
    player_rows = []
    for spec in players:
        pid = spec["player_id"]
        name = spec["name"]
        position = spec["position"]
        minutes = spec["minutes"]
        n_games = spec["n_games"]
        if n_games > len(sched):
            raise ValueError(f"{name} wants {n_games} games but the schedule only has {len(sched)}")
        box_kwargs = {k: v for k, v in spec.items()
                      if k not in ("player_id", "name", "position", "minutes", "n_games", "oreb")}
        oreb_override = spec.get("oreb")
        for i in range(n_games):
            g = sched.iloc[i]
            line = box_line(minutes, **box_kwargs)
            if oreb_override is not None:
                # box_line() itself has no oreb kwarg (always 0); override directly, keeping reb
                # (== dreb) consistent so the contract's reb = oreb + dreb identity still holds.
                line["oreb"] = oreb_override
                line["reb"] = line["dreb"] + oreb_override
            rows.append(dict(season=season, game_id=g["game_id"], game_date=g["game_date"], player_id=pid,
                             player_name=name, team_id=int(team), team_abbr=g["team_abbr"],
                             matchup=f"{g['team_abbr']} vs. X", plus_minus=0.0, **line))
        age = (pd.Timestamp(int(season[:4]), 10, 1) - pd.Timestamp(birthdate)).days / 365.25
        bio_rows.append(dict(season=season, player_id=pid, age_at_season_start=age, team_id=int(team)))
        row = {c: None for c in t["players"].columns}
        row.update(player_id=pid, player_name=name, birthdate=pd.Timestamp(birthdate), position=position,
                   height_in=78.0, weight_lb=210.0, draft_year=draft_year, draft_round=1, draft_number=10,
                   from_year=int(season[:4]), to_year=int(season[:4]))
        player_rows.append(row)

    new_gl = pd.concat([gl, pd.DataFrame(rows, columns=[c for c in gl.columns])], ignore_index=True) \
        if rows else gl
    for c in new_gl.columns:
        new_gl[c] = new_gl[c].astype(t["game_logs"][c].dtype)
    t["game_logs"] = new_gl

    # Drop existing player_season_bio rows for these player_ids/this season, then re-add.
    pids = [p["player_id"] for p in players]
    bio = t["player_season_bio"]
    bio = bio[~((bio["season"] == season) & (bio["player_id"].isin(pids)))]
    t["player_season_bio"] = pd.concat([bio, pd.DataFrame(bio_rows)[t["player_season_bio"].columns]],
                                       ignore_index=True)

    # Drop existing players rows for these player_ids, then re-add.
    pl = t["players"]
    pl = pl[~pl["player_id"].isin(pids)]
    new_pl = pd.DataFrame(player_rows)
    for c in t["players"].columns:
        new_pl[c] = new_pl[c].astype(t["players"][c].dtype)
    t["players"] = pd.concat([pl, new_pl], ignore_index=True)

    return t, int(team)
