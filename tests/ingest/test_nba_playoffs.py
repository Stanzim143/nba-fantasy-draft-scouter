"""nba_playoffs: only ``004<yy>`` playoff game ids, played minutes, newest rows replace a re-pulled season."""
import pandas as pd
from ingest_fakes import wrap

from src.ingest import nba_playoffs as po

HEADERS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "PLUS_MINUS",
           "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def row(pid, gid, minutes, date="2026-05-22"):
    return [pid, f"P{pid}", 1610612760, "OKC", gid, date, "OKC vs. SAS", minutes, 1.0, 5, 10, 2, 5, 3, 4, 1, 3, 4, 2, 1, 0, 1, 2, 15]


def payload(rows):
    return wrap("PlayerGameLogs", HEADERS, rows, "leaguegamelog")


def test_transform_keeps_only_playoff_ids_and_played_minutes():
    p = payload([row(1, "0042500301", 27), row(2, "0042500301", 0), row(3, "0022500001", 30),      # DNP, regular season
                 row(4, "0052500001", 30), row(5, "0042400301", 30)])                              # play-in, wrong season
    df = po.transform_playoff_logs(p, "2025-26")
    assert list(df["player_id"]) == [1] and (df["season"] == "2025-26").all() and df["min"].iloc[0] == 27.0
    assert list(df.columns) == po.KEEP


def test_playoff_params_request_the_playoffs_season_type():
    assert po.playoff_params("2025-26")["SeasonType"] == "Playoffs" and po.playoff_params("2025-26")["Season"] == "2025-26"
