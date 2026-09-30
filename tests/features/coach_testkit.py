"""Synthetic box scores, coach tables and histories for the coach tests."""
import numpy as np
import pandas as pd

from src.contracts import History

T1, T2 = 1, 2


def logs(spec):
    """spec: list of (season, team, game_id, [minutes...]) -> game_logs rows; every player also shoots (fga 10, fg3a 4)."""
    rows = []
    pid = 0
    for season, team, game, mins in spec:
        for i, m in enumerate(mins):
            rows.append({"season": season, "team_id": team, "game_id": f"{season}-{team}-{game}", "player_id": team * 100 + i, "min": float(m),
                         "fga": 10, "fta": 2, "oreb": 1, "tov": 2, "fg3a": 4, "player_name": "p", "game_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=game)})
            pid += 1
    return pd.DataFrame(rows)


def season_games(season, team, n, mins):
    return [(season, team, g, mins) for g in range(n)]


def bio_for(gl, age=27.0):
    return gl[["season", "player_id"]].drop_duplicates().assign(age_at_season_start=age, team_id=0)


def coach_table(rows):
    """rows: (season, team, [(name, status)...])"""
    out = []
    for season, team, coaches in rows:
        for i, (name, status) in enumerate(coaches):
            out.append({"season": season, "team_id": team, "seq": i, "coach_name": name, "coach_key": name.lower(), "status": status,
                        "is_opening": i == 0, "n_coaches": len(coaches), "source": "t", "start_date": pd.NaT})
    return pd.DataFrame(out)


def build_history(seasons_spec, target):
    tables = {"game_logs": logs(seasons_spec), "team_games": pd.DataFrame({"season": [], "game_id": [], "team_id": []}),
              "players": pd.DataFrame({"player_id": [1], "player_name": ["x"], "birthdate": [pd.Timestamp("1990-01-01")], "position": ["G"],
                                       "height_in": [70], "weight_lb": [180], "draft_year": [2010], "draft_round": [1], "draft_number": [1],
                                       "from_year": [2010], "to_year": [np.nan]}),
              "player_season_bio": pd.DataFrame({"season": [], "player_id": [], "age_at_season_start": [], "team_id": []})}
    tables["player_season_bio"] = bio_for(tables["game_logs"])
    return History(target_season=target, game_logs=tables["game_logs"], team_games=tables["team_games"], players=tables["players"],
                   player_season_bio=tables["player_season_bio"])


def two_team_history(target="2021-22"):
    """T1: coach 'star' (38-minute stars) in 2019-20 and 2020-21; T2: coach 'deep' (deep rotation) in both. Then, for the target
    season, 'star' moves to T2 (a coaching change there)."""
    spec = []
    for s in ("2018-19", "2019-20", "2020-21"):
        spec += season_games(s, T1, 25, [38, 36, 34, 32, 30, 12, 8, 6, 4, 2])
        spec += season_games(s, T2, 25, [28, 27, 26, 25, 24, 22, 20, 18, 12, 8])
    coaches = coach_table([("2018-19", T1, [("star", "")]), ("2018-19", T2, [("deep", "")]),
                           ("2019-20", T1, [("star", "")]), ("2019-20", T2, [("deep", "")]),
                           ("2020-21", T1, [("star", "")]), ("2020-21", T2, [("deep", "")]),
                           ("2021-22", T1, [("newbie", "")]), ("2021-22", T2, [("star", "")])])
    return build_history(spec, target), coaches
