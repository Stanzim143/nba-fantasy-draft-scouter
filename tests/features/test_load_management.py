"""Load-management proxy features (ADR 0024): interior runs, the isolated-absence and back-to-back counts, point in time."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_testkit import L, TEAM, _pat, _tables  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import History  # noqa: E402
from src.features import load_management as LM  # noqa: E402

def test_absence_shape_counts_only_short_interior_runs():
    b2b = np.zeros(L, bool)
    b2b[[5, 12]] = True
    # game 5 (b2b, single), games 9-10 (double), games 14-16 (triple: too long), lead block 0-1 and the last game (never interior)
    present = _pat(missed=[5, 9, 10, 14, 15, 16, L - 1], lead=2)
    assert LM.absence_shape(present, b2b) == (3, 2, 1)
    # a single miss that is not a second night, a b2b single miss in a run of two: neither is a "rest" absence
    p2 = _pat(missed=[3, 12, 13])
    assert LM.absence_shape(p2, b2b) == (3, 2, 0)
    assert LM.absence_shape(np.ones(L, bool), b2b) == (0, 0, 0)
    assert LM.absence_shape(np.zeros(L, bool), b2b) == (0, 0, 0), "no games played: the whole season is one lead block, never interior"


def test_profiles_from_game_logs_and_dates():
    pats = {(1, "2018-19"): _pat(missed=[5, 9, 10]), (2, "2018-19"): _pat(missed=[4, 8, 9, 10]), (3, "2018-19"): _pat()}
    t = _tables(pats, b2b={5, 9}, seasons=("2018-19",))
    h = History.until({**t}, "2019-20")
    p = LM.load_profiles(h).set_index("player_id")
    assert p.loc[1, ["iso_n", "iso_runs", "rest_n", "gp", "L", "missed"]].tolist() == [3, 2, 1, 27, 30, 3]
    assert p.loc[2, "iso_n"] == 1 and p.loc[2, "rest_n"] == 0 and p.loc[2, "iso_runs"] == 1, "the 3-game run does not count"
    assert p.loc[3, "iso_n"] == 0 and p.loc[3, "iso_frac"] == 0.0 and p.loc[3, "gp"] == 30
    assert p.loc[1, "iso_n82"] == pytest.approx(3 * 82 / 30)
    assert p.loc[1, "iso_frac"] == 1.0 and not p.loc[1, "near65"]


def test_traded_player_is_marked_multi_team_and_uses_the_primary_team():
    pats = {(1, "2018-19"): _pat()}
    t = _tables(pats, seasons=("2018-19",), other_team={(1, "2018-19", g): TEAM + 1 for g in range(20, 30)})
    h = History.until(t, "2019-20")
    p = LM.load_profiles(h).set_index("player_id")
    assert p.loc[1, "n_teams"] == 2 and p.loc[1, "team_id"] == TEAM


def test_prior_features_use_only_the_season_before_the_target_and_ignore_the_future():
    pats = {(1, s): _pat(missed=[5, 9]) for s in ("2017-18", "2018-19", "2019-20")}
    pats[(1, "2018-19")] = _pat(missed=[5, 6, 9, 10, 11, 20])
    t = _tables(pats, b2b={5})
    base = LM.prior_load_features(History.until(t, "2019-20"))
    r = base.set_index("player_id").loc[1]
    assert r["iso_n"] == 3 and r["veteran"] and r["gp"] == 24 and r["f"] == 0.8      # runs 5-6 (2), 9-11 is 3 long, 20 single
    assert "s" not in base.columns
    with scrambled_future(t, "2019-20"):
        scr = LM.prior_load_features(History.until(t, "2019-20"))
    pd.testing.assert_frame_equal(scr, base)
    # the same features one season earlier profile 2017-18 instead
    early = LM.prior_load_features(History.until(t, "2018-19")).set_index("player_id").loc[1]
    assert early["iso_n"] == 2


def test_empty_history_gives_an_empty_frame():
    h = History(target_season="2019-20", game_logs=pd.DataFrame(columns=["season", "game_id", "player_id", "team_id", "min"]),
                team_games=pd.DataFrame(columns=["season", "game_id", "game_date", "team_id"]),
                players=pd.DataFrame(columns=["player_id", "from_year", "to_year", "draft_year"]),
                player_season_bio=pd.DataFrame(columns=["season", "player_id"]))
    assert LM.load_profiles(h).empty and LM.prior_load_features(h).empty
