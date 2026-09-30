"""Test doubles and fixture builders for the ESPN league sync tests. No network, no real data.

The shapes mirror the *real* ESPN fantasy API responses observed while building
``src.ingest.espn_league`` against our actual league (id 1234567890, verified 2026-09-23; see
docs/adr/0008-espn-league-sync.md). Names, ids and owner GUIDs below are all invented -- nothing
here is real league data.
"""
from __future__ import annotations

from typing import Any

# Reuse the FakeResponse/FakeSession doubles already used by the NBA ingest tests.
from ingest_fakes import FakeClock, FakeResponse, FakeSession  # noqa: F401

# --------------------------------------------------------------------------- settings/teams/rosters

DEFAULT_LINEUP_SLOT_COUNTS = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 0, "8": 0, "9": 0, "10": 0,
    "11": 3, "12": 3, "13": 1, "14": 0,
}

DEFAULT_SCORING_ITEMS = [
    {"statId": 6, "points": 1.0, "isReverseItem": False},   # REB
    {"statId": 11, "points": -2.0, "isReverseItem": False},  # TO
    {"statId": 13, "points": 2.0, "isReverseItem": False},   # FGM
    {"statId": 14, "points": -1.0, "isReverseItem": False},  # FGA
    {"statId": 15, "points": 1.0, "isReverseItem": False},   # FTM
    {"statId": 16, "points": -1.0, "isReverseItem": False},  # FTA
    {"statId": 0, "points": 1.0, "isReverseItem": False},    # PTS
    {"statId": 17, "points": 1.0, "isReverseItem": False},   # 3PM
    {"statId": 1, "points": 4.0, "isReverseItem": False},    # BLK
    {"statId": 2, "points": 4.0, "isReverseItem": False},    # STL
    {"statId": 3, "points": 2.0, "isReverseItem": False},    # AST
]


def make_team(team_id: int, name: str, owner_id: str, *, division_id: int = 0, abbrev: str | None = None,
              roster_entries: list[dict] | None = None) -> dict:
    return {
        "id": team_id,
        "abbrev": abbrev or name[:4].upper(),
        "name": name,
        "divisionId": division_id,
        "owners": [owner_id],
        "primaryOwner": owner_id,
        "record": {"overall": {"wins": 0, "losses": 0, "ties": 0, "pointsFor": 0.0, "pointsAgainst": 0.0}},
        "roster": {"entries": roster_entries or [], "tradeReservedEntries": 0},
    }


def make_settings_payload(*, teams: list[dict] | None = None, team_count: int = 13, name: str = "Test League",
                          is_public: bool = True, drafted: bool = False, in_progress: bool = False,
                          lineup_slot_counts: dict | None = None, scoring_items: list[dict] | None = None,
                          pick_order: list[int] | None = None, deadline_epoch_ms: int = 1804870800000,
                          matchup_period_count: int = 20, playoff_team_count: int = 8,
                          matchup_acquisition_limit: float = 1.0, waiver_hours: int = 24,
                          season_limit: int = -1, trade_max: int = -1, veto_votes: int = 4,
                          review_hours: int = 48, keeper_count: int = 0, members: list[dict] | None = None) -> dict:
    if teams is None:
        teams = [make_team(i + 1, f"Team {i + 1}", f"OWNER-{i + 1}") for i in range(team_count)]
    return {
        "id": 1234567890,
        "seasonId": 2027,
        "members": members if members is not None else [
            {"id": t["owners"][0], "displayName": f"ESPNFAN{1000 + i}"} for i, t in enumerate(teams)
        ],
        "draftDetail": {"drafted": drafted, "inProgress": in_progress},
        "status": {"teamsJoined": len(teams), "isFull": len(teams) == team_count},
        "teams": teams,
        "settings": {
            "name": name,
            "isPublic": is_public,
            "size": len(teams),
            "acquisitionSettings": {
                "acquisitionType": "WAIVERS_TRADITIONAL",
                "acquisitionLimit": season_limit,
                "matchupAcquisitionLimit": matchup_acquisition_limit,
                "matchupLimitPerScoringPeriod": True,
                "waiverHours": waiver_hours,
            },
            "draftSettings": {
                "type": "SNAKE",
                "orderType": "MANUAL",
                "timePerSelection": 90,
                "isTradingEnabled": True,
                "keeperCount": keeper_count,
                "pickOrder": pick_order or [t["id"] for t in teams],
            },
            "rosterSettings": {"lineupSlotCounts": lineup_slot_counts or dict(DEFAULT_LINEUP_SLOT_COUNTS)},
            "scheduleSettings": {
                "matchupPeriodCount": matchup_period_count,
                "playoffTeamCount": playoff_team_count,
                "playoffReseed": False,
                "divisions": [{"id": 0, "name": "East", "size": len(teams)}],
            },
            "scoringSettings": {
                "scoringType": "H2H_POINTS",
                "scoringItems": scoring_items if scoring_items is not None else list(DEFAULT_SCORING_ITEMS),
            },
            "tradeSettings": {
                "max": trade_max,
                "deadlineDate": deadline_epoch_ms,
                "revisionHours": review_hours,
                "vetoVotesRequired": veto_votes,
            },
        },
    }


def make_roster_entry(espn_player_id: int, name: str, *, lineup_slot_id: int = 12,
                      default_position_id: int = 1, pro_team_id: int = 1,
                      injury_status: str = "ACTIVE") -> dict:
    return {
        "playerId": espn_player_id,
        "lineupSlotId": lineup_slot_id,
        "playerPoolEntry": {
            "player": {
                "id": espn_player_id,
                "fullName": name,
                "defaultPositionId": default_position_id,
                "proTeamId": pro_team_id,
                "injuryStatus": injury_status,
                "injured": False,
            }
        },
    }


# --------------------------------------------------------------------------- draft

def make_draft_payload(*, drafted: bool = False, in_progress: bool = False, picks: list[dict] | None = None,
                       team_ids: list[int] | None = None) -> dict:
    if picks is None:
        team_ids = team_ids or list(range(1, 14))
        picks = [
            {"id": i + 1, "overallPickNumber": i + 1, "roundId": 1, "roundPickNumber": i + 1,
             "teamId": tid, "playerId": -1, "keeper": False}
            for i, tid in enumerate(team_ids)
        ]
    return {"draftDetail": {"drafted": drafted, "inProgress": in_progress, "picks": picks}}


# --------------------------------------------------------------------------- free agents / transactions

def make_free_agent(espn_player_id: int, name: str, *, adp: float = 50.0, pct_owned: float = 10.0,
                    default_position_id: int = 1, pro_team_id: int = 1,
                    injury_status: str = "ACTIVE", on_team_id: int = 0) -> dict:
    return {
        "id": espn_player_id,
        "onTeamId": on_team_id,
        "player": {
            "id": espn_player_id,
            "fullName": name,
            "defaultPositionId": default_position_id,
            "proTeamId": pro_team_id,
            "injuryStatus": injury_status,
            "injured": False,
            "ownership": {"averageDraftPosition": adp, "percentOwned": pct_owned},
        },
    }


def make_free_agents_payload(players: list[dict] | None = None) -> dict:
    if players is None:
        players = [make_free_agent(1000 + i, f"Player {i}", adp=1.0 + i) for i in range(5)]
    return {"players": players}


def make_transactions_payload(transactions: list[dict] | None = None) -> dict:
    payload: dict[str, Any] = {"draftDetail": {"drafted": False}, "id": 1234567890}
    if transactions is not None:
        payload["transactions"] = transactions
    return payload
