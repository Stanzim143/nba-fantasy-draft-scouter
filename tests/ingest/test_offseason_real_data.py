"""Opt-in: sanity checks on the real ingested Summer League / preseason / roster tables (skipped when absent).

These pin down facts about the real data that shaped the design (ADR 0012), so a future re-ingest that changes
them fails loudly instead of silently changing model behaviour.
"""
import pandas as pd
import pytest

from src.contracts import data_dir, season_start, season_str
from src.ingest import nba_incoming as ni
from src.ingest import nba_offseason as no

_HAVE = (data_dir() / "processed" / "offseason_logs.parquet").exists() and (data_dir() / "processed" / "players.parquet").exists()
pytestmark = pytest.mark.skipif(not _HAVE, reason="real offseason data not ingested (python -m src.ingest.nba_offseason)")

LIVE = 2026
EVENT_SEASONS = [season_str(y) for y in range(2015, LIVE + 1)]


@pytest.fixture(scope="module")
def logs():
    return no.read_offseason_logs()


@pytest.fixture(scope="module")
def teams():
    return no.read_offseason_team_games()


@pytest.fixture(scope="module")
def players():
    return pd.read_parquet(data_dir() / "processed" / "players.parquet")


def test_both_tables_satisfy_their_schemas(logs, teams):
    no.validate_offseason_logs(logs)
    no.validate_offseason_team_games(teams)


def test_every_summer_league_exists_except_the_cancelled_2020_one(logs):
    have = set(logs.loc[logs["context"] == no.SUMMER_LEAGUE, "event_season"])
    assert have == set(EVENT_SEASONS) - {"2020-21"}


def test_every_completed_preseason_exists_and_the_live_one_may_not_yet(logs):
    have = set(logs.loc[logs["context"] == no.PRESEASON, "event_season"])
    assert set(EVENT_SEASONS[:-1]) <= have


def test_events_carry_the_season_they_follow_and_stay_inside_their_windows(logs):
    starts = logs["event_season"].map(season_start)
    assert (logs["season"].map(season_start) == starts - 1).all()
    for (ctx, ev), g in logs.groupby(["context", "event_season"]):
        lo, hi = no.event_window(ctx, ev)
        assert g["game_date"].between(lo, hi).all(), (ctx, ev)


def test_the_2019_20_bubble_games_are_not_in_the_preseason(logs):
    pre = logs[(logs["context"] == no.PRESEASON) & (logs["event_season"] == "2019-20")]
    assert len(pre) > 1000 and pre["game_date"].max() < pd.Timestamp(2020, 1, 1)


def test_box_scores_obey_the_identities_apart_from_the_known_2026_summer_league_quirk(logs):
    excess = no.unrecorded_points(logs)
    assert (excess >= 0).all()
    by_event = logs.assign(excess=excess).groupby(["event_season", "context"])["excess"].sum()
    normal = by_event.drop(labels=[("2026-27", no.SUMMER_LEAGUE)], errors="ignore")
    assert (normal == 0).all(), normal[normal != 0]
    assert by_event.get(("2026-27", no.SUMMER_LEAGUE), 0) > 0      # documented: free throws missing from ftm/fta in July 2026


MIN_ROWS_FOR_ID_SHARE = 300      # roughly ten box scores


def test_player_ids_join_to_the_players_table_and_minutes_are_fractional_where_available(logs, players):
    known = logs["player_id"].isin(set(players["player_id"]))
    for (ctx, ev), g in logs.groupby(["context", "event_season"]):
        if len(g) < MIN_ROWS_FOR_ID_SHARE:     # an event in progress (one preseason game) is a handful of camp players: noise, not a join defect
            continue
        share = known[g.index].mean()
        assert share >= (0.4 if ctx == no.SUMMER_LEAGUE else 0.85), (ctx, ev, round(float(share), 3))
    recent = logs[(logs["event_season"] >= "2022-23") & (logs["context"] == no.SUMMER_LEAGUE)]
    assert (recent["min"] % 1 != 0).mean() > 0.9                   # playergamelogs refinement was applied


def test_the_incoming_class_is_in_the_players_table_with_birthdates(players):
    cls = players[players["draft_year"] == LIVE]
    assert len(cls) >= 45 and cls["birthdate"].notna().all() and cls["draft_number"].notna().all()
    assert players["player_id"].is_unique


def test_roster_snapshot_covers_every_team_sensibly():
    snaps = ni.read_roster_snapshots()
    day = ni.latest_snapshot(snaps)
    assert day["team_id"].nunique() == 30 and day["player_id"].is_unique
    per_team = day.groupby("team_id").size()
    assert per_team.min() >= 13 and per_team.max() <= 25
    assert (day["draft_year"] == LIVE).sum() >= 45
    ni.validate_roster_snapshots(snaps)


def test_the_live_summer_league_players_mostly_join_now_that_the_rookies_are_present(logs, players):
    sl = logs[(logs["context"] == no.SUMMER_LEAGUE) & (logs["event_season"] == season_str(LIVE))]
    minutes_known = sl.loc[sl["player_id"].isin(set(players["player_id"])), "min"].sum() / sl["min"].sum()
    assert minutes_known > 0.5, f"only {minutes_known:.0%} of July 2026 minutes belong to players in the table"


def test_player_names_are_resolved_and_ids_are_unique_per_game(logs):
    """``leaguegamelog`` carries real names; the ``playergamelogs`` placeholder ids (blank names) never enter the table."""
    placeholder = logs["player_name"].str.startswith("player ")
    assert placeholder.mean() < 0.01, f"{placeholder.mean():.2%} of rows have no player name"
    assert not logs.duplicated(["game_id", "player_id"]).any()
