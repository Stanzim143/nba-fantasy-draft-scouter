"""nba_profiles: origin classes from the player index, draft-combine rows, birthdates, atomic writes (offline)."""
from types import SimpleNamespace

import pandas as pd
from ingest_fakes import common_player_info_payload, wrap

from src.ingest import nba_profiles as npf

HEAD = ["PERSON_ID", "PLAYER_LAST_NAME", "PLAYER_FIRST_NAME", "POSITION", "HEIGHT", "WEIGHT", "COLLEGE", "COUNTRY",
        "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER", "FROM_YEAR", "TO_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "ROSTER_STATUS"]


def row(pid, last, college, country, dy=2020.0, dr=1.0, dn=10.0):
    return [pid, last, "A", "F", "6-8", "220", college, country, dy, dr, dn, "2020", "2025", 1, "AAA", 1.0]


ROWS = [row(1, "One", "Duke", "USA"), row(2, "Two", "Duke", "USA"), row(3, "Three", "Duke", "USA"), row(4, "Four", "Club Zed", "Serbia"),
        row(5, "Five", "Oak Hill Academy", "USA"), row(6, "Six", "", None, None, None, None)]


def payload():
    return wrap("PlayerIndex", HEAD, ROWS, "playerindex")


def test_origin_and_previous_organisation_classes():
    p = npf.build_profiles(payload()).set_index("player_id")
    assert p.loc[1, "origin"] == "usa" and p.loc[1, "prev_org_type"] == "college"
    assert p.loc[4, "origin"] == "intl" and p.loc[4, "prev_org_type"] == "other"
    assert p.loc[5, "prev_org_type"] == "other"                      # a prep school shared by fewer than 3 players is not a college
    assert pd.isna(p.loc[6, "origin"]) and pd.isna(p.loc[6, "country"])


def test_combine_rows_and_duplicates():
    h = ["SEASON", "PLAYER_ID", "HEIGHT_WO_SHOES", "WINGSPAN", "MAX_VERTICAL_LEAP", "WEIGHT"]
    pl = wrap("DraftCombineStats", h, [["2025", 1, 80.5, 84.0, 33.5, "210.0"], ["2025", 1, 80.5, 84.0, 33.5, "210.0"], ["2025", 2, None, None, None, None]])
    c = npf.build_combine(pl)
    assert len(c) == 2 and c.loc[c["player_id"] == 1, "wingspan_in"].iloc[0] == 84.0
    assert npf.build_combine(wrap("DraftCombineStats", h, [])).empty


def test_birthdates_prefer_players_then_commonplayerinfo():
    prof = npf.build_profiles(payload())
    players = pd.DataFrame({"player_id": [1], "birthdate": pd.to_datetime(["2000-01-02"])})
    out = npf.fill_birthdates(prof, players, {4: "2001-05-06T00:00:00", 5: None}).set_index("player_id")
    assert out.loc[1, "birthdate"] == pd.Timestamp("2000-01-02") and out.loc[4, "birthdate"] == pd.Timestamp("2001-05-06")
    assert out["birthdate"].isna().sum() == 4


class FakeClient:
    offline, min_interval = False, 0.0

    def __init__(self):
        self.stats = SimpleNamespace(network_requests=0, cache_hits=0)

    def peek(self, endpoint, params):
        return None

    def get(self, endpoint, params, *, refresh=False):
        if endpoint == "playerindex":
            return payload()
        if endpoint == "draftcombinestats":
            return wrap("DraftCombineStats", ["SEASON", "PLAYER_ID", "HEIGHT_WO_SHOES"], [[params["SeasonYear"][:4], 1, 80.0]])
        if endpoint == "commonplayerinfo":
            return common_player_info_payload(int(params["PlayerID"]), "1999-03-04T00:00:00")
        raise AssertionError(endpoint)


def test_run_ingest_writes_tables_and_backs_up_once(tmp_path):
    (tmp_path / "processed").mkdir()
    pd.DataFrame({"snapshot_date": pd.to_datetime(["2026-09-24"]), "player_id": [6]}).to_parquet(tmp_path / "processed" / "roster_snapshots.parquet")
    r = npf.run_ingest("2026-27", FakeClient(), tmp_path, log=lambda *_: None)
    assert r.profiles == 6 and r.combine_rows == len(r.combine_years) and r.candidates_needing_birthdate == 1
    assert npf.read_profiles(tmp_path).set_index("player_id").loc[6, "birthdate"] == pd.Timestamp("1999-03-04")
    npf.run_ingest("2026-27", FakeClient(), tmp_path, log=lambda *_: None)
    assert (tmp_path / "processed" / "player_profiles.parquet.bak_pre_profiles").exists()
    assert npf.read_combine(tmp_path)["draft_year"].min() == 2015


class FailingYearClient(FakeClient):
    def __init__(self, bad_year):
        super().__init__()
        self.bad_year = bad_year

    def get(self, endpoint, params, *, refresh=False):
        if endpoint == "draftcombinestats" and params["SeasonYear"].startswith(str(self.bad_year)):
            raise npf.NBAClientError("boom")
        return super().get(endpoint, params, refresh=refresh)


def test_partial_combine_failure_never_truncates_the_table(tmp_path):
    (tmp_path / "processed").mkdir()
    # first run, a year fails and there is no earlier table: nothing is written
    r = npf.run_ingest("2026-27", FailingYearClient(2020), tmp_path, birthdates=False, log=lambda *_: None)
    assert r.combine_failed_years == [2020] and not (tmp_path / "processed" / "draft_combine.parquet").exists()
    npf.run_ingest("2026-27", FakeClient(), tmp_path, birthdates=False, log=lambda *_: None)
    full = npf.read_combine(tmp_path)
    assert 2020 in set(full["draft_year"])
    # later run with another year failing keeps that year's earlier rows
    r = npf.run_ingest("2026-27", FailingYearClient(2018), tmp_path, birthdates=False, log=lambda *_: None)
    assert r.combine_failed_years == [2018]
    assert len(npf.read_combine(tmp_path)) == len(full) and 2018 in set(npf.read_combine(tmp_path)["draft_year"])
