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
