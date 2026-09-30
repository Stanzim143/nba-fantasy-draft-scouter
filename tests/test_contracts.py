import os
from pathlib import Path

import pandas as pd
import pytest

from src import contracts as C
from src.store import load_tables, read_table, table_exists, write_table
from src.synthetic import make_synthetic_tables

SMALL = dict(first_start=2015, last_start=2017, n_teams=8, games_per_team=20, seed=7)


@pytest.fixture(scope="module")
def tables():
    return make_synthetic_tables(**SMALL)


# ------------------------------------------------------------------ seasons

def test_season_roundtrip():
    assert C.season_str(2023) == "2023-24"
    assert C.season_str(1999) == "1999-00"
    assert C.season_start("2023-24") == 2023
    assert C.season_start("1999-00") == 1999


@pytest.mark.parametrize("bad", ["2023", "2023-25", "23-24", "2023-2024", "", "abcd-ef"])
def test_season_start_rejects_malformed(bad):
    with pytest.raises(ValueError):
        C.season_start(bad)


def test_backtest_seasons_span():
    assert C.BACKTEST_SEASONS[0] == "2015-16"
    assert C.BACKTEST_SEASONS[-1] == "2025-26"
    assert len(C.BACKTEST_SEASONS) == 11


def test_stat_map_covers_league_scoring():
    from src.value.league import load_league
    assert set(load_league()["scoring"]) == set(C.STAT_COLUMN_MAP)
    assert set(C.STAT_COLUMN_MAP.values()) <= set(C.GAME_LOGS.columns)


# ------------------------------------------------------------------ storage

def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    assert C.data_dir() == tmp_path
    assert C.table_path("game_logs") == tmp_path / "processed" / "game_logs.parquet"
    assert C.raw_dir("nba_api") == tmp_path / "raw" / "nba_api"


def test_data_dir_default_is_outside_repo():
    os.environ.pop("NBA_DATA_DIR", None)
    assert Path.home() in C.data_dir().parents


def test_unknown_table():
    with pytest.raises(KeyError):
        C.table_path("nope")


def test_store_roundtrip_all_tables(tables, tmp_path):
    for name, df in tables.items():
        write_table(df, name, base=tmp_path)
        assert table_exists(name, base=tmp_path)
        back = read_table(name, base=tmp_path)
        assert len(back) == len(df)
    loaded = load_tables(base=tmp_path)
    assert set(loaded) == set(C.HISTORY_TABLES)


def test_store_refuses_invalid_table(tables, tmp_path):
    bad = tables["game_logs"].drop(columns=["pts"])
    with pytest.raises(C.ContractError):
        write_table(bad, "game_logs", base=tmp_path)
    assert not table_exists("game_logs", base=tmp_path)


def test_read_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table("game_logs", base=tmp_path)


def test_write_leaves_no_temp_files(tables, tmp_path):
    write_table(tables["players"], "players", base=tmp_path)
    assert [p.name for p in (tmp_path / "processed").iterdir()] == ["players.parquet"]


# ------------------------------------------------------------------ validation

def test_synthetic_tables_valid(tables):
    for name, df in tables.items():
        C.validate_table(df, name)


def test_validate_catches_missing_column(tables):
    with pytest.raises(C.ContractError, match="missing columns"):
        C.validate_table(tables["game_logs"].drop(columns=["reb"]), "game_logs")


def test_validate_catches_duplicates(tables):
    g = tables["game_logs"]
    with pytest.raises(C.ContractError, match="duplicate"):
        C.validate_table(pd.concat([g, g.head(1)]), "game_logs")


def test_validate_catches_nulls(tables):
    g = tables["game_logs"].copy()
    g["player_name"] = g["player_name"].astype(object)
    g.loc[g.index[0], "player_name"] = None
    with pytest.raises(C.ContractError, match="not nullable"):
        C.validate_table(g, "game_logs")


def test_validate_catches_wrong_dtype(tables):
    g = tables["game_logs"].copy()
    g["pts"] = g["pts"].astype(float) + 0.5
    with pytest.raises(C.ContractError, match="dtype"):
        C.validate_table(g, "game_logs")


@pytest.mark.parametrize("mutate,msg", [
    (lambda g: g.assign(fgm=g["fga"] + 1), "fgm > fga"),
    (lambda g: g.assign(ftm=g["fta"] + 1), "ftm > fta"),
    (lambda g: g.assign(reb=g["reb"] + 1), "reb != oreb + dreb"),
    (lambda g: g.assign(pts=g["pts"] + 1), "pts != "),
    (lambda g: g.assign(min=0.0), "min <= 0"),
])
def test_validate_catches_box_score_violations(tables, mutate, msg):
    with pytest.raises(C.ContractError, match=msg.replace("+", r"\+").replace("*", r"\*")):
        C.validate_table(mutate(tables["game_logs"].copy()), "game_logs")


def test_extra_columns_allowed_unless_strict(tables):
    g = tables["game_logs"].assign(extra=1)
    C.validate_table(g, "game_logs")
    with pytest.raises(C.ContractError, match="unexpected"):
        C.validate_table(g, "game_logs", allow_extra=False)


def test_nullable_int_accepts_pandas_nullable_and_float_nan(tables):
    p = tables["players"].copy()
    p["draft_year"] = p["draft_year"].astype("float64")  # NaN-carrying ints from parquet/CSV
    C.validate_table(p, "players")


# ------------------------------------------------------------------ History leakage guard

def test_history_until_excludes_target_and_future(tables):
    h = C.History.until(tables, "2016-17")
    assert h.last_season == "2015-16"
    assert set(h.game_logs["season"]) == {"2015-16"}
    assert set(h.team_games["season"]) == {"2015-16"}
    assert set(h.player_season_bio["season"]) == {"2015-16"}
    h.assert_no_future()


def test_history_first_season_is_empty(tables):
    h = C.History.until(tables, "2015-16")
    assert h.game_logs.empty and h.last_season is None
    h.assert_no_future()


def test_assert_no_future_detects_leak(tables):
    h = C.History.until(tables, "2016-17")
    leaked = C.History(
        target_season="2016-17", game_logs=tables["game_logs"], team_games=h.team_games,
        players=h.players, player_season_bio=h.player_season_bio)
    with pytest.raises(AssertionError, match="leakage"):
        leaked.assert_no_future()


def test_history_slices_extras(tables):
    proj = pd.DataFrame({"season": ["2015-16", "2016-17"], "x": [1, 2]})
    h = C.History.until({**tables, "adp": proj}, "2016-17")
    assert list(h.extras["adp"]["x"]) == [1]


def test_history_sanitizes_players_from_year_and_to_year(tables):
    """Regression test: players.from_year/to_year are derived from the FULL dataset (every
    season ever ingested), not from ``target_season``, so History.until must scrub them or a
    projector can read who is about to debut or retire straight out of the "static" table."""
    target = "2016-17"
    s = C.season_start(target)
    raw_players = tables["players"]
    assert (raw_players["from_year"] >= s).any(), "fixture must contain a player debuting later"
    assert (raw_players["to_year"] >= s).any(), "fixture must contain a player active later"

    h = C.History.until(tables, target)
    assert not (h.players["from_year"].dropna() >= s).any()
    assert not (h.players["to_year"].dropna() >= s).any()
    assert not (pd.to_numeric(h.players["draft_year"], errors="coerce") > s).any()
    h.assert_no_future()  # now also checks players; must not raise on the sanitized frame

    # a player who debuted and left entirely within history is untouched
    long_retired = raw_players[raw_players["to_year"] < s - 5]
    if len(long_retired):
        row = long_retired.iloc[0]
        kept = h.players.loc[h.players["player_id"] == row["player_id"]]
        assert len(kept) == 1
        assert int(kept["to_year"].iloc[0]) == int(row["to_year"])

    # players.player_id is a superset relationship: sanitizing only drops future-only players
    assert set(h.players["player_id"]) <= set(raw_players["player_id"])


def test_history_players_sanitization_is_idempotent(tables):
    once = C.History.until(tables, "2018-19").players
    twice = C._sanitize_players(once, C.season_start("2018-19"), known_player_ids=set(once["player_id"]))
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


def test_history_keeps_undrafted_player_with_only_bio_row(tables):
    """A player known only via player_season_bio (no game_logs row that season, e.g. a source
    gap) but who genuinely played before cutoff must still be kept, not just game_logs members."""
    target = "2018-19"
    bio_only_id = int(tables["player_season_bio"].query("season < @target")["player_id"].iloc[0])
    gl = tables["game_logs"]
    stripped = {**tables, "game_logs": gl[gl["player_id"] != bio_only_id].reset_index(drop=True)}
    h = C.History.until(stripped, target)
    assert bio_only_id in set(h.players["player_id"])


def test_history_never_mutates_input_players_table(tables):
    before = tables["players"].copy(deep=True)
    C.History.until(tables, "2017-18")
    pd.testing.assert_frame_equal(tables["players"], before)


def test_assert_no_future_detects_players_leak(tables):
    target = "2016-17"
    s = C.season_start(target)
    h = C.History.until(tables, target)
    leaky_row = tables["players"][tables["players"]["from_year"] >= s]
    if leaky_row.empty:
        pytest.skip("fixture has no future debutant to reintroduce")
    h.players = pd.concat([h.players, leaky_row], ignore_index=True)
    with pytest.raises(AssertionError, match="leakage.*players"):
        h.assert_no_future()


def test_projector_protocol_runtime_check():
    class Good:
        name = "g"

        def project(self, history):
            return pd.DataFrame()

    class Bad:
        pass

    assert isinstance(Good(), C.Projector)
    assert not isinstance(Bad(), C.Projector)


def test_projection_spec_has_all_stat_columns():
    assert set(C.PROJECTION_STATS) <= set(C.PROJECTIONS.columns)
    assert len(C.PROJECTION_STATS) == len(C.STAT_COLUMN_MAP)
