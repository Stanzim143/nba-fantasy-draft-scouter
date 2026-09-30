"""nba_incoming: the incoming draft class joins ``players``; roster snapshots are dated, appended and diffable."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from ingest_fakes import BOS, GSW, NYK, common_player_info_payload, wrap

from src.contracts import validate_table
from src.ingest import nba_incoming as ni
from src.store import read_table, write_table

SEASON = "2026-27"
INDEX_HEADERS = ["PERSON_ID", "PLAYER_LAST_NAME", "PLAYER_FIRST_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "POSITION",
                 "HEIGHT", "WEIGHT", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER", "ROSTER_STATUS", "FROM_YEAR", "TO_YEAR"]


def irow(pid, first, last, team=BOS, abbr="BOS", pos="G", dy=2026.0, dr=1.0, dn=5.0, status=1.0, fy="2026", ty="2026"):
    return [pid, last, first, team, abbr, pos, "6-5", "200", dy, dr, dn, status, fy, ty]


def index_payload(rows):
    return wrap("PlayerIndex", INDEX_HEADERS, rows, "playerindex")


# a realistic mix: one veteran already in players, two 2026 draftees, one retired man whose TEAM_ID is still filled,
# one undrafted signee, one earlier draftee making his debut
ROWS = [
    irow(1, "Vet", "Eran", team=NYK, abbr="NYK", dy=2018.0, dr=1.0, dn=3.0, fy="2018", ty="2025"),
    irow(2, "Top", "Pick", team=BOS, abbr="BOS", dy=2026.0, dn=1.0),
    irow(3, "Late", "Second", team=GSW, abbr="GSW", pos="F", dy=2026.0, dr=2.0, dn=48.0),
    irow(4, "Long", "Retired", team=BOS, abbr="BOS", dy=2001.0, dn=10.0, status=None, fy="2001", ty="2012"),
    irow(5, "Undrafted", "Signee", team=GSW, abbr="GSW", dy=None, dr=None, dn=None),
    irow(6, "Stash", "Arrival", team=NYK, abbr="NYK", dy=2024.0, dr=1.0, dn=22.0, fy="2026"),
]


class FakeClient:
    offline = False
    min_interval = 0.0

    def __init__(self, index_rows=ROWS, cpi=None, offline=False):
        self.index_rows, self.offline = list(index_rows), offline
        self.cpi = cpi if cpi is not None else {
            2: common_player_info_payload(2, "2007-05-18T00:00:00", "Forward", "6-9", "215", ("2026", "1", "1"), "2026", "2026"),
            3: common_player_info_payload(3, "2004-02-02T00:00:00", "Guard", "6-4", "190", ("2026", "2", "48"), "2026", "2026")}
        self.calls, self.stats = [], SimpleNamespace(network_requests=0, cache_hits=0)

    def peek(self, endpoint, params):
        return None

    def get(self, endpoint, params, *, refresh=False):
        self.calls.append((endpoint, dict(params), refresh))
        self.stats.network_requests += 1
        if endpoint == "playerindex":
            return index_payload(self.index_rows)
        if endpoint == "commonplayerinfo":
            pid = int(params["PlayerID"])
            if pid not in self.cpi:
                return {"resultSets": [{"name": "CommonPlayerInfo", "headers": ["PERSON_ID"], "rowSet": []}]}
            return self.cpi[pid]
        raise AssertionError(endpoint)


def seed_players(base, ids=(1,)):
    rows = [dict(player_id=1, player_name="Vet Eran", birthdate=pd.Timestamp("1996-01-02"), position="G",
                 height_in=77.0, weight_lb=200.0, draft_year=2018, draft_round=1, draft_number=3, from_year=2018, to_year=2025)]
    df = pd.DataFrame([r for r in rows if r["player_id"] in ids])
    for c in ("draft_year", "draft_round", "draft_number", "from_year", "to_year"):
        df[c] = df[c].astype("Int64")
    df["birthdate"] = df["birthdate"].astype("datetime64[ns]")
    write_table(df, "players", base)


def run(base, client=None, day=date(2026, 9, 24), **kw):
    client = client or FakeClient()
    return ni.run_ingest(SEASON, client, base, snapshot_date=day, log=lambda *_: None, **kw), client


# --------------------------------------------------------------------------- transforms

def test_roster_membership_uses_roster_status_not_team_id():
    idx = ni._index(index_payload(ROWS))
    ros = ni.rostered(idx)
    assert set(ros["player_id"]) == {1, 2, 3, 5, 6}
    assert 4 not in set(ros["player_id"])          # retired: TEAM_ID filled, ROSTER_STATUS empty


def test_incoming_class_is_the_drafted_rostered_unknown_players_of_the_draft_year():
    idx = ni._index(index_payload(ROWS))
    assert set(ni.incoming_class(idx, SEASON, known_player_ids={1})["player_id"]) == {2, 3}
    assert set(ni.incoming_class(idx, SEASON, known_player_ids={1, 2})["player_id"]) == {3}
    assert ni.incoming_class(idx, "2025-26", known_player_ids=set()).empty       # nobody rostered was drafted in 2025


def test_new_player_rows_prefer_the_index_and_fill_gaps_from_commonplayerinfo():
    idx = ni._index(index_payload(ROWS))
    cls = ni.incoming_class(idx, SEASON, set())
    cpi = {2: {"birthdate": pd.Timestamp("2007-05-18"), "position": "F-C", "height_in": 81.0}}
    rows = ni.new_player_rows(cls, cpi).set_index("player_id")
    assert rows.at[2, "birthdate"] == pd.Timestamp("2007-05-18")
    assert rows.at[2, "position"] == idx.set_index("player_id").at[2, "position"]     # the index value wins
    assert pd.isna(rows.at[3, "birthdate"])                                            # no commonplayerinfo: unknown, not guessed
    assert rows.at[2, "draft_number"] == 1 and rows.at[3, "draft_round"] == 2


def test_roster_snapshot_frame_is_validated_and_typed():
    snap = ni.roster_snapshot_frame(ni._index(index_payload(ROWS)), SEASON, date(2026, 9, 24))
    assert len(snap) == 5 and set(snap["snapshot_date"]) == {pd.Timestamp(2026, 9, 24)}
    assert snap["player_id"].dtype == "int64" and snap["team_id"].dtype == "int64"
    assert pd.isna(snap.set_index("player_id").at[5, "draft_year"])
    ni.validate_roster_snapshots(snap)
    dup = pd.concat([snap, snap], ignore_index=True)
    with pytest.raises(ni.IncomingError, match="duplicate"):
        ni.validate_roster_snapshots(dup)
    with pytest.raises(ni.IncomingError, match="missing"):
        ni.validate_roster_snapshots(snap.drop(columns=["team_id"]))


def test_roster_moves_reports_arrivals_departures_and_team_changes():
    idx = ni._index(index_payload(ROWS))
    before = ni.roster_snapshot_frame(idx, SEASON, date(2026, 9, 1))
    moved_rows = [r[:] for r in ROWS]
    moved_rows[0][3], moved_rows[0][4] = GSW, "GSW"        # veteran traded NYK -> GSW
    moved_rows[2][11] = None                                # second-rounder waived
    moved_rows.append(irow(7, "New", "Signing", team=BOS, abbr="BOS", dy=None, dr=None, dn=None))
    after = ni.roster_snapshot_frame(ni._index(index_payload(moved_rows)), SEASON, date(2026, 9, 2))
    m = ni.roster_moves(before, after).set_index("player_id")
    assert m.at[1, "kind"] == "moved" and (m.at[1, "from_team"], m.at[1, "to_team"]) == ("NYK", "GSW")
    assert m.at[3, "kind"] == "departed" and m.at[7, "kind"] == "arrived"
    assert set(m.index) == {1, 3, 7}
    assert ni.roster_moves(before, before).empty


def test_snapshots_append_replace_the_same_day_and_pick_the_latest(tmp_path):
    idx = ni._index(index_payload(ROWS))
    d1 = ni.roster_snapshot_frame(idx, SEASON, date(2026, 9, 1))
    d2 = ni.roster_snapshot_frame(idx.iloc[:3], SEASON, date(2026, 9, 2))
    ni.write_roster_snapshot(d1, tmp_path)
    ni.write_roster_snapshot(d2, tmp_path)
    ni.write_roster_snapshot(d2, tmp_path)                      # same day again: replaced, not duplicated
    all_ = ni.read_roster_snapshots(tmp_path)
    assert all_.groupby("snapshot_date").size().to_dict() == {pd.Timestamp(2026, 9, 1): 5, pd.Timestamp(2026, 9, 2): 3}
    assert set(ni.latest_snapshot(all_)["snapshot_date"]) == {pd.Timestamp(2026, 9, 2)}
    assert set(ni.latest_snapshot(all_, before=date(2026, 9, 2))["snapshot_date"]) == {pd.Timestamp(2026, 9, 1)}
    assert ni.latest_snapshot(all_, before=date(2026, 8, 1)).empty


def test_reading_a_missing_snapshot_table_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="nba_incoming"):
        ni.read_roster_snapshots(tmp_path)


# --------------------------------------------------------------------------- pipeline

def test_pipeline_adds_only_the_drafted_class_and_never_touches_existing_players(tmp_path):
    seed_players(tmp_path)
    before = read_table("players", tmp_path).set_index("player_id").loc[1]
    result, client = run(tmp_path)
    players = read_table("players", tmp_path).set_index("player_id")
    assert set(players.index) == {1, 2, 3}                           # undrafted signee and the stash arrival stay out
    validate_table(players.reset_index(), "players")
    pd.testing.assert_series_equal(players.loc[1], before, check_names=False)
    assert players.at[2, "draft_number"] == 1 and players.at[2, "birthdate"] == pd.Timestamp("2007-05-18")
    assert (result.rostered, result.added_players, result.already_known) == (5, 2, 1)
    assert result.undrafted_rostered_not_in_players == 1 and result.earlier_draft_debuts_not_in_players == 1
    assert result.earlier_draft_debut_names == ["Stash Arrival"]
    assert result.missing_birthdates == 0
    assert any(c[0] == "playerindex" and c[2] is True for c in client.calls)       # the live index is always refreshed


def test_rerunning_adds_nothing_and_does_not_duplicate_the_snapshot(tmp_path):
    seed_players(tmp_path)
    run(tmp_path)
    before = read_table("players", tmp_path)
    result, client = run(tmp_path)
    assert result.added_players == 0 and result.already_known == 3
    pd.testing.assert_frame_equal(read_table("players", tmp_path), before)
    assert len(ni.read_roster_snapshots(tmp_path)) == 5
    assert not any(c[0] == "commonplayerinfo" for c in client.calls)


def test_a_later_snapshot_reports_the_moves_since_the_previous_one(tmp_path):
    seed_players(tmp_path)
    run(tmp_path, day=date(2026, 9, 24))
    moved = [r[:] for r in ROWS]
    moved[0][3], moved[0][4] = GSW, "GSW"
    result, _ = run(tmp_path, client=FakeClient(moved), day=date(2026, 9, 25))
    assert result.moves_since_previous == {"moved": 1}
    assert set(ni.read_roster_snapshots(tmp_path)["snapshot_date"]) == {pd.Timestamp(2026, 9, 24), pd.Timestamp(2026, 9, 25)}


def test_missing_birthdates_are_counted_not_invented(tmp_path):
    seed_players(tmp_path)
    result, _ = run(tmp_path, client=FakeClient(cpi={}))
    assert result.missing_birthdates == 2
    assert read_table("players", tmp_path).set_index("player_id")["birthdate"].loc[[2, 3]].isna().all()


def test_no_birthdates_flag_skips_the_per_player_pulls(tmp_path):
    seed_players(tmp_path)
    _, client = run(tmp_path, birthdates=False)
    assert not any(c[0] == "commonplayerinfo" for c in client.calls)


def test_works_without_an_existing_players_table(tmp_path):
    result, _ = run(tmp_path)
    assert result.added_players == 2 and result.already_known == 0
    # only the 2026 draft class is added; the 2018 veteran, the undrafted signee and the stash arrival stay out
    assert set(read_table("players", tmp_path)["player_id"]) == {2, 3}


def test_an_empty_roster_refuses_to_write_anything(tmp_path):
    empty = [irow(9, "Ghost", "Player", status=None)]
    with pytest.raises(ni.IncomingError, match="nobody on a roster"):
        run(tmp_path, client=FakeClient(empty))
    assert not ni.snapshots_path(tmp_path).exists()


def test_report_is_written_and_keyed_by_season(tmp_path):
    seed_players(tmp_path)
    run(tmp_path)
    rep = json.loads(ni.report_path(tmp_path).read_text(encoding="utf-8"))["runs"][SEASON]
    assert rep["added_players"] == 2 and rep["snapshot_date"] == "2026-09-24" and rep["rostered"] == 5


def test_offline_client_does_not_request_a_refresh(tmp_path):
    seed_players(tmp_path)
    _, client = run(tmp_path, client=FakeClient(offline=True))
    assert all(not c[2] for c in client.calls)


# --------------------------------------------------------------------------- CLI

def test_cli_runs_with_an_injected_client(tmp_path, capsys):
    seed_players(tmp_path)
    assert ni.main(["--season", SEASON, "--data-dir", str(tmp_path)], client=FakeClient()) == 0
    assert "+2 rookies" in capsys.readouterr().out


def test_cli_reports_an_unusable_index_as_exit_code_one(tmp_path, capsys):
    assert ni.main(["--season", SEASON, "--data-dir", str(tmp_path)], client=FakeClient([irow(9, "G", "P", status=None)])) == 1
    assert "ingest failed" in capsys.readouterr().err


def test_cli_rejects_a_malformed_season(tmp_path, capsys):
    assert ni.main(["--season", "2026", "--data-dir", str(tmp_path)], client=FakeClient()) == 1
    assert "ingest failed" in capsys.readouterr().err
