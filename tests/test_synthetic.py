import pandas as pd
import pytest

from src.contracts import season_start
from src.synthetic import make_synthetic_tables

KW = dict(first_start=2015, last_start=2018, n_teams=10, games_per_team=30, seed=3)


@pytest.fixture(scope="module")
def t():
    return make_synthetic_tables(**KW)


def test_deterministic():
    a = make_synthetic_tables(**{**KW, "last_start": 2016})
    b = make_synthetic_tables(**{**KW, "last_start": 2016})
    pd.testing.assert_frame_equal(a["game_logs"], b["game_logs"])


def test_seed_changes_output():
    a = make_synthetic_tables(**{**KW, "last_start": 2015})
    b = make_synthetic_tables(**{**KW, "last_start": 2015, "seed": 4})
    assert not a["game_logs"]["pts"].equals(b["game_logs"]["pts"])


def test_every_team_plays_every_game(t):
    tg = t["team_games"]
    assert (tg.groupby(["season", "team_id"]).size() == KW["games_per_team"]).all()
    assert tg.groupby("game_id")["team_id"].count().eq(2).all()
    assert tg.groupby("game_id")["is_home"].sum().eq(1).all()


def test_scores_are_consistent_between_opponents(t):
    tg = t["team_games"]
    other = tg.groupby("game_id")["pts_for"].transform("sum") - tg["pts_for"]
    assert (tg["pts_against"] == other).all()


def test_player_game_rows_match_a_team_game(t):
    key = ["game_id", "team_id"]
    merged = t["game_logs"][key].drop_duplicates().merge(t["team_games"][key], on=key, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_players_missing_games_is_realistic(t):
    """Availability varies: some players miss games, most play most of them."""
    g = t["game_logs"]
    gp = g.groupby(["season", "player_id"]).size()
    assert gp.max() <= KW["games_per_team"]
    assert (gp < KW["games_per_team"] * 0.5).any(), "expected some long absences"
    assert (gp >= KW["games_per_team"] * 0.8).any(), "expected some near-full seasons"


def test_minutes_track_role(t):
    g = t["game_logs"]
    mpg = g.groupby(["season", "player_id"])["min"].mean()
    assert 5 < mpg.mean() < 30
    assert mpg.max() <= 44


def test_skill_persists_year_to_year(t):
    """Year-over-year FPPG correlation must be clearly positive: that's the signal models find."""
    g = t["game_logs"]
    fp = (g["pts"] + g["reb"] + 2 * g["ast"] + 4 * g["stl"] + 4 * g["blk"] - 2 * g["tov"]
          + g["fgm"] - g["fga"] + g["ftm"] - g["fta"] + g["fg3m"])
    per = g.assign(fp=fp).groupby(["season", "player_id"])["fp"].mean().unstack(0)
    a, b = per.iloc[:, 0], per.iloc[:, 1]
    both = pd.concat([a, b], axis=1).dropna()
    assert both.corr().iloc[0, 1] > 0.5


def test_rookies_and_retirements_occur(t):
    p = t["players"]
    assert (p["from_year"] > KW["first_start"]).any()
    assert (p["to_year"] < KW["last_start"]).any()


def test_bio_ages_are_plausible_and_grow(t):
    b = t["player_season_bio"]
    assert b["age_at_season_start"].between(17, 45).all()
    two = b.pivot(index="player_id", columns="season", values="age_at_season_start").dropna().iloc[:, :2]
    assert ((two.iloc[:, 1] - two.iloc[:, 0]).round(0) == 1).all()


def test_bio_only_for_players_who_played(t):
    played = set(zip(t["game_logs"]["season"], t["game_logs"]["player_id"]))
    assert set(zip(t["player_season_bio"]["season"], t["player_season_bio"]["player_id"])) == played


def test_players_table_covers_all_loggers(t):
    assert set(t["game_logs"]["player_id"]) == set(t["players"]["player_id"])
    assert t["players"]["player_id"].is_unique


def test_no_state_leaks_between_calls():
    """Regression: an earlier draft cached players in a module global."""
    a = make_synthetic_tables(**{**KW, "last_start": 2015})
    make_synthetic_tables(**{**KW, "last_start": 2016, "seed": 99})
    b = make_synthetic_tables(**{**KW, "last_start": 2015})
    pd.testing.assert_frame_equal(a["players"], b["players"])


def test_dates_within_season(t):
    g = t["game_logs"]
    yr = g["season"].map(season_start)
    assert (g["game_date"].dt.year >= yr).all() and (g["game_date"].dt.year <= yr + 1).all()


def test_rejects_odd_team_count():
    with pytest.raises(ValueError):
        make_synthetic_tables(n_teams=9)


def test_rejects_reversed_seasons():
    with pytest.raises(ValueError):
        make_synthetic_tables(first_start=2018, last_start=2015)
