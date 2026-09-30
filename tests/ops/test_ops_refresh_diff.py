"""Snapshot archive and the run-over-run diffs (offline, tiny fixtures)."""
from datetime import datetime, timezone

import pytest

from src.ops import refresh_diff as rd
from src.ops.daily_report import render_report


def raw(*rows):
    return [{"id": i, "fullName": n, "proTeamId": 1, "injuryStatus": st, "injured": st == "OUT",
             "ownership": {"averageDraftPosition": adp, "percentOwned": 50.0}} for i, n, st, adp in rows]


T1 = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc)


def test_snapshot_written_deduplicated_and_never_overwritten(tmp_path):
    p1, w1 = rd.write_snapshot(tmp_path, "2026-27", raw((1, "A", "ACTIVE", 5.0)), T1)
    assert w1 and p1.exists() and p1.parent == tmp_path / "archive" / "espn_players" / "2026-27"
    p1b, w1b = rd.write_snapshot(tmp_path, "2026-27", raw((1, "A", "ACTIVE", 5.0)), T2)
    assert not w1b and p1b == p1
    p2, w2 = rd.write_snapshot(tmp_path, "2026-27", raw((1, "A", "OUT", 5.0)), T2)
    assert w2 and p2 != p1 and len(rd.list_snapshots(tmp_path, "2026-27")) == 2
    assert rd.previous_snapshot(tmp_path, "2026-27", p2) == p1
    assert rd.read_snapshot(p2)["players"][0]["status"] == "OUT"


def test_empty_payload_is_refused(tmp_path):
    with pytest.raises(ValueError):
        rd.write_snapshot(tmp_path, "2026-27", [], T1)


def snap(*rows):
    return {"players": rd.compact_players(raw(*rows))}


def test_injury_changes_ignore_none_vs_active_and_sort_by_adp():
    a = snap((1, "A", None, 50.0), (2, "B", "ACTIVE", 3.0), (3, "C", "ACTIVE", 20.0))
    b = snap((1, "A", "ACTIVE", 50.0), (2, "B", "OUT", 3.0), (3, "C", "DAY_TO_DAY", 20.0))
    ch = rd.injury_changes(a, b)
    assert [c["name"] for c in ch] == ["B", "C"] and ch[0]["from"] == "ACTIVE" and ch[0]["to"] == "OUT"


def test_adp_moves():
    a = snap((1, "Riser", "ACTIVE", 30.0), (2, "Faller", "ACTIVE", 10.0), (3, "Flat", "ACTIVE", 20.0),
             (4, "Gone", "ACTIVE", 100.0), (5, "New", "ACTIVE", 139.99))
    b = snap((1, "Riser", "ACTIVE", 25.0), (2, "Faller", "ACTIVE", 14.0), (3, "Flat", "ACTIVE", 20.5),
             (4, "Gone", "ACTIVE", 139.99), (5, "New", "ACTIVE", 90.0))
    m = rd.adp_moves(a, b)
    assert [x["name"] for x in m["risers"]] == ["Riser"] and m["risers"][0]["delta"] == -5.0
    assert [x["name"] for x in m["fallers"]] == ["Faller"]
    assert [x["name"] for x in m["priced"]] == ["New"] and [x["name"] for x in m["unpriced"]] == ["Gone"]


def test_roster_and_watchlist_diffs_and_games():
    r = rd.roster_diff({"1": ["A", "BOS"], "2": ["B", "LAL"], "3": ["C", "NYK"]},
                       {"1": ["A", "MIA"], "2": ["B", "LAL"], "4": ["D", "DEN"]})
    assert r["moved"] == [{"name": "A", "from": "BOS", "to": "MIA"}]
    assert r["arrived"] == [{"name": "D", "to": "DEN"}] and r["departed"] == [{"name": "C", "from": "NYK"}]
    w = rd.watchlist_diff([{"player_id": 1, "name": "A", "rank": 1}, {"player_id": 2, "name": "B", "rank": 2},
                           {"player_id": 3, "name": "C", "rank": 3}],
                          [{"player_id": 3, "name": "C", "rank": 1}, {"player_id": 4, "name": "D", "rank": 9},
                           {"player_id": 1, "name": "A", "rank": 7}])
    assert [x["name"] for x in w["new"]] == ["D"] and [x["name"] for x in w["dropped"]] == ["B"]
    assert {x["name"] for x in w["moved"]} == {"A"}
    g = rd.games_delta({"preseason": {"games": 4}},
                       {"preseason": {"games": 10, "last_date": "2026-10-03"}, "summer_league": {"games": 76}})
    assert g["preseason"]["new"] == 6 and g["summer_league"]["new"] is None


def test_report_renders_all_sections():
    run = {"date": "2026-10-03", "outcome": "partial", "run_id": "X", "season": "2026-27", "finished_local": "t", "seconds": 5,
           "days_to_draft": 14, "draft_date": "2026-10-17", "draft_date_source": "fallback", "prev_run_id": "P", "prev_time": "p",
           "steps": [{"name": "adp", "status": "failed", "seconds": 1, "error": "boom"}],
           "diffs": {"roster": {"arrived": [], "departed": [], "moved": [{"name": "A", "from": "BOS", "to": "MIA"}]},
                     "injuries": [{"name": "B", "from": "ACTIVE", "to": "OUT", "adp": 3.0}],
                     "adp": {"risers": [{"name": "R", "from": 30.0, "to": 25.0, "delta": -5.0}], "fallers": [],
                             "priced": [], "unpriced": []},
                     "watchlist": {"new": [{"rank": 1, "name": "D", "team": "DEN", "uplift": 2.0}], "dropped": [], "moved": []},
                     "games": {"preseason": {"now": 10, "new": 6, "last_date": "2026-10-03"}}},
           "alert": {"reasons": ["2 consecutive runs failed"]}, "watchlist_top": []}
    md = render_report(run)
    for needle in ("PARTIAL", "draft in 14", "latest plausible", "A BOS -> MIA", "B: ACTIVE -> OUT", "riser R",
                   "NEW on the list", "+6 new", "ALERT", "boom"):
        assert needle in md
