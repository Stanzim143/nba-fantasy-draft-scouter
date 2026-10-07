"""The draft-day risk overlay: flags and advisory games; projections are never touched (ADR 0016)."""
import numpy as np
import pandas as pd

from src.features import risk as R


def test_preseason_flag_needs_a_rotation_veteran_on_a_team_that_played():
    pl = pd.Series({1: 0, 2: 1, 3: 3, 4: 0}, dtype=float)
    tm = pd.Series({10: 4, 20: 2}, dtype=float)
    team = pd.Series({1: 10, 2: 10, 3: 10, 4: 20})
    ids = [1, 2, 3, 4, 5]
    f = R.preseason_flags(ids, team.reindex(ids), np.full(5, 30.0), np.full(5, 3), pl, tm)
    assert list(f["pre_flag"]) == ["dnp_all", "partial", "", "", ""]      # team 20 played only 2 games; player 5 has no team
    assert R.preseason_flags([1], team.reindex([1]), np.array([30.0]), np.array([0]), pl, tm)["pre_flag"].iloc[0] == ""   # no history
    assert R.preseason_flags([1], team.reindex([1]), np.array([5.0]), np.array([3]), pl, tm)["pre_flag"].iloc[0] == ""    # not rotation


def test_status_flags_and_staleness():
    st = pd.DataFrame({"player_id": pd.array([1, 2, pd.NA], dtype="Int64"), "injury_status": ["OUT", "DAY_TO_DAY", "OUT"],
                       "last_news_date": pd.to_datetime(["2026-09-20", "2026-01-01", "2026-09-20"])})
    f = R.status_flags([1, 2, 3], st, today="2026-09-24")
    assert list(f["espn_status"]) == ["OUT", "DAY_TO_DAY", ""] and f["status_news_days"].iloc[0] == 4


def test_overlay_never_changes_projections_and_stale_news_is_not_discounted():
    board = pd.DataFrame({"player_id": [1, 2, 3], "proj_gp": [60.0, 60.0, 60.0], "proj_fppg": [30.0, 30.0, 30.0]})
    status = pd.DataFrame({"player_id": [1, 2], "espn_status": ["OUT", "OUT"], "status_news_days": [3.0, 200.0]})
    o = R.build_overlay(board, status=status, today="2026-09-24")
    assert list(o["proj_gp"]) == [60.0] * 3 and list(o["proj_fppg"]) == [30.0] * 3
    assert o["risk_level"].tolist() == ["high", "", ""] or o["risk_level"].tolist() == ["high", "high", ""]
    assert o["risk_gp"].iloc[0] == 60 * (1 - R.ASSUMED_STATUS_HAIRCUT["OUT"])
    assert o["risk_gp"].iloc[1] == 60.0 and "stale" in o["risk_flags"].iloc[1]     # shown, but not counted


def test_preseason_absence_warns_without_discounting():
    board = pd.DataFrame({"player_id": [1], "proj_gp": [60.0]})
    pre = pd.DataFrame({"player_id": [1], "pre_flag": ["dnp_all"], "team_pre_games": [4.0], "pre_gp": [0.0]})
    o = R.build_overlay(board, preseason=pre)
    assert o["risk_level"].iloc[0] == "watch" and "0 of 4" in o["risk_flags"].iloc[0]
    assert o["risk_gp_haircut"].iloc[0] == R.PRESEASON_HAIRCUT["dnp_all"] == 0.0


def test_context_flags_new_team_and_star_moves():
    roster = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [10, 10, 20], "team_abbr": ["AAA", "AAA", "BBB"]})
    last = pd.Series({1: 30, 2: 10, 3: 20})            # 1 arrived at AAA from team 30 and is a star
    names = pd.Series({1: "Star One", 2: "Vet Two", 3: "Guy Three"})
    r = R.context_flags(roster, last, {1}, names).flags.set_index("player_id")
    assert bool(r.loc[1, "new_team"]) and not bool(r.loc[2, "new_team"])
    assert "Star One" in r.loc[2, "star_arrivals"] and r.loc[1, "star_arrivals"] == "" and r.loc[3, "star_arrivals"] == ""


def test_suspended_status_is_a_watch_flag_with_a_ten_percent_haircut():
    """SUSPENDED is handled like DAY_TO_DAY (level watch, not high) but with a 10% assumed haircut (ADR 0016 D4 amendment)."""
    assert R.ASSUMED_STATUS_HAIRCUT == {"OUT": 0.20, "DAY_TO_DAY": 0.03, "SUSPENDED": 0.10}
    board = pd.DataFrame({"player_id": [1, 2, 3, 4], "proj_gp": [60.0] * 4, "proj_fppg": [30.0] * 4})
    status = pd.DataFrame({"player_id": [1, 2, 3, 4], "espn_status": ["SUSPENDED", "DAY_TO_DAY", "OUT", "SUSPENDED"],
                           "status_news_days": [3.0, 3.0, 3.0, 200.0]})
    o = R.build_overlay(board, status=status, today="2026-09-24")
    assert o["risk_level"].tolist() == ["watch", "watch", "high", ""]
    assert o["risk_gp"].tolist() == [54.0, 60.0 * 0.97, 48.0, 60.0]           # stale suspension: shown, not discounted
    assert o["risk_flags"].iloc[0].startswith("ESPN suspended") and "stale" in o["risk_flags"].iloc[3]


# --- minutes opened by departures (rotation depth) ---------------------------------------------------------

def _logs(rows):
    """rows: (player_id, team_id, games, minutes_each) for season 2025-26; every team plays 10 games."""
    recs = [{"season": "2025-26", "player_id": p, "team_id": t, "min": m, "game_id": f"{t}-{p}-{g}"} for p, t, n, m in rows for g in range(n)]
    return pd.DataFrame(recs)


def _team_games(teams, n=10):
    return pd.DataFrame([{"season": "2025-26", "team_id": t, "game_id": f"{t}-{g}"} for t in teams for g in range(n)])


NAMES = pd.Series({1: "Dort", 2: "Joe", 3: "Star", 4: "Newcomer", 5: "Bench"})


def test_minutes_opened_counts_departed_minus_arrived_per_team_game():
    # team 10 lost Dort (30 mpg), Joe (20) and a free agent still unsigned; Newcomer (from team 20) arrived at 10 mpg
    logs = _logs([(1, 10, 10, 30.0), (2, 10, 10, 20.0), (3, 10, 10, 35.0), (4, 20, 10, 10.0), (6, 10, 10, 5.0)])
    roster = pd.DataFrame({"player_id": [1, 3, 4, 5], "team_id": [30, 10, 10, 10], "team_abbr": ["XXX", "AAA", "AAA", "AAA"]})
    o = R.minutes_opened(logs, _team_games([10, 20, 30]), roster, NAMES).set_index("team_id")
    assert o.loc[10, "out_mpg"] == 30 + 20 + 5          # Dort (traded), Joe (on no roster), player 6 (on no roster)
    assert o.loc[10, "in_mpg"] == 10 and o.loc[10, "net_open"] == 45
    assert o.loc[10, "departed"].startswith("Dort 30, Joe 20")
    assert o.loc[30, "in_mpg"] == 30 and o.loc[30, "net_open"] == -30     # Dort's old minutes show up as arrived minutes at 30
    assert o.loc[20, "out_mpg"] == 10 and o.loc[20, "net_open"] == 10 - 0


def test_minutes_opened_splits_a_midseason_trade_by_team_games():
    logs = _logs([(1, 10, 5, 30.0), (1, 20, 5, 30.0)])        # 5 games for each of two teams that played 10
    roster = pd.DataFrame({"player_id": [1], "team_id": [20], "team_abbr": ["BBB"]})
    o = R.minutes_opened(logs, _team_games([10, 20]), roster, NAMES).set_index("team_id")
    assert o.loc[10, "out_mpg"] == 15 and o.loc[20, "out_mpg"] == 0 and o.loc[20, "in_mpg"] == 15


def test_depth_flag_only_for_rotation_range_players_on_a_team_that_opened_enough_minutes():
    opened = pd.DataFrame({"team_id": [10, 20], "out_mpg": [70.0, 30.0], "in_mpg": [8.0, 25.0],
                           "net_open": [62.0, 5.0], "departed": ["Dort 23, Joe 18", "X 30"]})
    roster = pd.DataFrame({"player_id": [1, 2, 3, 4], "team_id": [10, 10, 10, 20], "team_abbr": ["OKC"] * 3 + ["BBB"]})
    mpg = pd.Series({1: 18.0, 2: 34.0, 3: 3.0, 4: 18.0})     # 2 is a star, 3 is end-of-bench, 4's team did not open minutes
    f = R.depth_flags(roster, opened, mpg)
    assert list(f["player_id"]) == [1] and "OKC lost 70 mpg (Dort 23, Joe 18)" in f["depth_flag"].iloc[0]
    assert R.depth_flags(roster, opened.iloc[0:0], mpg).empty


# --- playoff role -------------------------------------------------------------------------------------------

def _po(pid, mins, season="2025-26"):
    return pd.DataFrame({"season": season, "player_id": pid, "game_id": [f"{pid}-{i:02d}" for i in range(len(mins))],
                         "game_date": pd.date_range("2026-04-20", periods=len(mins)), "min": mins,
                         "pts": 10, "reb": 2, "ast": 2, "stl": 0, "blk": 0, "tov": 1, "fgm": 4, "fga": 9, "ftm": 1, "fta": 2, "fg3m": 1})


def test_playoff_flag_uses_the_closing_run_of_role_players_not_starters():
    reg = pd.DataFrame({"season": "2025-26", "player_id": [1] * 4 + [2] * 4 + [3] * 4, "min": [18.0] * 4 + [34.0] * 4 + [18.0] * 4})
    po = pd.concat([_po(1, [8, 12, 15, 10, 26, 27, 24, 33, 27]),       # bench guard whose role grew late
                    _po(2, [40] * 9),                                  # starter: always more in the playoffs
                    _po(3, [10, 12, 14, 16]),                          # too few games
                    ])
    f = R.playoff_flags(po, reg, {"PTS": 1.0})
    assert list(f["player_id"]) == [1]
    assert "last 5 of 9 games" in f["playoff_flag"].iloc[0] and "(regular season 18 mpg)" in f["playoff_flag"].iloc[0]
    assert "10.0 FP/g" in f["playoff_flag"].iloc[0]


def test_playoff_flag_ignores_seasons_other_than_the_newest_and_empty_input():
    reg = pd.DataFrame({"season": "2025-26", "player_id": [1], "min": [18.0]})
    assert R.playoff_flags(pd.DataFrame(columns=["season"]), reg).empty
    old = _po(1, [30] * 8, season="2024-25")
    new = _po(1, [18] * 8)
    assert R.playoff_flags(pd.concat([old, new]), reg).empty             # newest season only: a flat 18 is not a bigger role


def test_overlay_appends_depth_and_playoff_text_without_changing_level_or_games():
    board = pd.DataFrame({"player_id": [1, 2], "proj_gp": [60.0, 60.0]})
    o = R.build_overlay(board, depth=pd.DataFrame({"player_id": [1], "depth_flag": ["minutes opened: OKC lost 70 mpg"]}),
                        playoffs=pd.DataFrame({"player_id": [1], "playoff_flag": ["playoffs 2025-26: 27 mpg"]}))
    assert o["risk_flags"].iloc[0] == "minutes opened: OKC lost 70 mpg; playoffs 2025-26: 27 mpg" and o["risk_flags"].iloc[1] == ""
    assert list(o["risk_level"]) == ["", ""] and list(o["risk_gp"]) == [60.0, 60.0]


def test_minutes_opened_survives_no_named_departures_and_no_departures_at_all():
    roster = pd.DataFrame({"player_id": [1], "team_id": [10], "team_abbr": ["AAA"]})
    small = _logs([(1, 10, 10, 30.0), (2, 10, 10, 3.0)])             # only a 3 mpg man left: too small to name
    o = R.minutes_opened(small, _team_games([10]), roster, NAMES)
    assert o["departed"].iloc[0] == "" and o["out_mpg"].iloc[0] == 3.0
    none = R.minutes_opened(_logs([(1, 10, 10, 30.0)]), _team_games([10]), roster, NAMES)
    assert none.empty and list(none.columns) == ["team_id", "out_mpg", "in_mpg", "net_open", "departed"]
