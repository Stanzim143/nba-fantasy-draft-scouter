import pandas as pd
import pytest

from src.contracts import PROJECTION_STATS
from src.synthetic import make_synthetic_tables
from src.value.frame import fantasy_points_frame
from src.value.league import load_league
from src.value.points import fantasy_points
from src.contracts import STAT_COLUMN_MAP

SCORING = load_league()["scoring"]


def test_matches_scalar_engine_row_by_row():
    g = make_synthetic_tables(2015, 2015, n_teams=4, games_per_team=10, seed=1)["game_logs"]
    vec = fantasy_points_frame(g, SCORING)
    for i in g.sample(50, random_state=0).index:
        stats = {k: g.at[i, c] for k, c in STAT_COLUMN_MAP.items()}
        assert vec[i] == pytest.approx(fantasy_points(stats, SCORING))


def test_known_line():
    df = pd.DataFrame([{c: 0 for c in STAT_COLUMN_MAP.values()}])
    df.loc[0, ["pts", "fgm", "fga"]] = [2, 1, 1]
    assert fantasy_points_frame(df, SCORING).iloc[0] == 3.0  # made two = +3


def test_projection_prefix():
    df = pd.DataFrame([{c: 1.0 for c in PROJECTION_STATS}])
    expected = sum(SCORING.values())
    assert fantasy_points_frame(df, SCORING, prefix="proj_").iloc[0] == pytest.approx(expected)


def test_preserves_index_and_handles_empty():
    df = pd.DataFrame({c: pd.Series(dtype="int64") for c in STAT_COLUMN_MAP.values()})
    out = fantasy_points_frame(df, SCORING)
    assert out.empty
    df2 = pd.DataFrame({c: [1, 2] for c in STAT_COLUMN_MAP.values()}, index=[10, 20])
    assert list(fantasy_points_frame(df2, SCORING).index) == [10, 20]


def test_missing_column_fails_loudly():
    df = pd.DataFrame({"pts": [1]})
    with pytest.raises(KeyError, match="missing stat columns"):
        fantasy_points_frame(df, SCORING)


def test_unmapped_scoring_key_fails_loudly():
    with pytest.raises(KeyError, match="no column mapping"):
        fantasy_points_frame(pd.DataFrame({"pts": [1]}), {"DD": 5})
