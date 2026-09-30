import pytest

from src.value.points import breakdown, fantasy_points

SCORING = {
    "PTS": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TO": -2,
    "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "3PM": 1,
}


def line(**kw):
    base = {k: 0 for k in SCORING}
    base.update(kw)
    return base


def test_made_two_is_worth_three():
    # 2 PTS + 2 FGM - 1 FGA
    assert fantasy_points(line(PTS=2, FGM=1, FGA=1), SCORING) == 3


def test_made_three_is_worth_five():
    # 3 PTS + 1 3PM + 2 FGM - 1 FGA
    assert fantasy_points(line(PTS=3, FGM=1, FGA=1, **{"3PM": 1}), SCORING) == 5


def test_missed_shot_and_missed_ft_cost_one():
    assert fantasy_points(line(FGA=1), SCORING) == -1
    assert fantasy_points(line(FTA=1), SCORING) == -1


def test_made_ft_is_worth_one():
    # 1 PTS + 1 FTM - 1 FTA
    assert fantasy_points(line(PTS=1, FTM=1, FTA=1), SCORING) == 1


def test_stocks_and_turnovers():
    assert fantasy_points(line(STL=1, BLK=1, TO=1), SCORING) == 6


def test_breakdown_sums_to_total():
    s = line(PTS=25, REB=8, AST=6, STL=1, BLK=1, TO=3, FGM=9, FGA=19, FTM=5, FTA=6, **{"3PM": 2})
    assert sum(breakdown(s, SCORING).values()) == fantasy_points(s, SCORING)


def test_missing_stat_raises():
    with pytest.raises(KeyError):
        fantasy_points({"PTS": 10}, SCORING)
