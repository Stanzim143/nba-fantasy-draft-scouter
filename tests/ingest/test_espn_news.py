"""ESPN injury-news feed: parsing, the append-only merge, the watch loop and the risk-overlay flag."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.features import risk
from src.ingest import espn_news

T0, T1 = datetime(2026, 10, 9, 8, 0), datetime(2026, 10, 9, 8, 30)


def note(nid, name, status="Day-To-Day", espn=None, **kw):
    athlete = {"displayName": name, "team": {"abbreviation": "MIA"},
               "links": [{"href": f"https://www.espn.com/nba/player/_/id/{espn}/x"}] if espn else []}
    return {"id": nid, "athlete": athlete, "status": status, "date": "2026-10-08T17:20Z", "source": "RotoWire",
            "shortComment": kw.get("short", "Wiggins is dealing with an illness."), "longComment": kw.get("long", "Long text."),
            "details": {"type": "Ankle", "location": "Leg", "detail": "Sprain", "side": "Left",
                        "returnDate": "2026-10-12", "fantasyStatus": {"abbreviation": "GTD"}}}


def feed(*notes):
    return {"injuries": [{"displayName": "Miami Heat", "injuries": list(notes)}]}


ID_MAP = pd.DataFrame({"source": ["espn"], "source_id": ["4397183"], "player_id": [1630]})


def test_parse_maps_ids_and_reads_the_athlete_id_from_links():
    df = espn_news.parse_feed(feed(note("1", "Aaron Wiggins", espn=4397183), note("2", "Nobody", espn=999)), T0, ID_MAP)
    assert list(df.columns) == espn_news.COLUMNS
    a = df.set_index("player_name")
    assert a.loc["Aaron Wiggins", "player_id"] == 1630 and pd.isna(a.loc["Nobody", "player_id"])
    assert a.loc["Aaron Wiggins", "injury_type"] == "Ankle" and a.loc["Aaron Wiggins", "return_date"] == pd.Timestamp("2026-10-12")


def test_parse_repairs_mojibake_and_skips_malformed_notes():
    df = espn_news.parse_feed(feed(note("1", "Luka", espn=1, short="AndrÃ© is out"), {"athlete": {}}), T0, None)
    assert len(df) == 1 and df["headline"].iloc[0] == "André is out"


def test_merge_is_append_only_and_keeps_first_seen():
    first = espn_news.parse_feed(feed(note("1", "A", espn=1)), T0, None)
    merged, fresh = espn_news.merge_news(None, first)
    again = espn_news.parse_feed(feed(note("1", "A", espn=1), note("2", "B", espn=2)), T1, None)
    merged, fresh = espn_news.merge_news(merged, again)
    assert list(fresh["note_id"]) == ["2"] and len(merged) == 2
    a = merged[merged["note_id"] == "1"].iloc[0]
    assert a["first_seen"] == pd.Timestamp(T0) and a["last_seen"] == pd.Timestamp(T1)


def test_a_status_change_on_the_same_note_is_a_new_row():
    m, _ = espn_news.merge_news(None, espn_news.parse_feed(feed(note("1", "A", espn=1)), T0, None))
    m, fresh = espn_news.merge_news(m, espn_news.parse_feed(feed(note("1", "A", status="Out", espn=1)), T1, None))
    assert len(m) == 2 and list(fresh["status"]) == ["Out"]


class FakeClient:
    def __init__(self, *payloads):
        self.payloads, self.calls = list(payloads), 0

    def get_json(self, label, url, params=None, *, headers=None, refresh=False):
        assert refresh, "the live feed must bypass the cache"
        self.calls += 1
        return self.payloads[min(self.calls - 1, len(self.payloads) - 1)]


def test_pull_writes_table_and_second_pull_reports_only_new(tmp_path):
    c = FakeClient(feed(note("1", "A", espn=1)), feed(note("1", "A", espn=1), note("2", "B", espn=2)))
    r1 = espn_news.run_pull(c, tmp_path, now=T0)
    r2 = espn_news.run_pull(c, tmp_path, now=T1)
    assert (r1["new"], r2["new"], r2["total"]) == (1, 1, 2)
    assert len(espn_news.read_news(tmp_path)) == 2


def test_pull_rejects_a_feed_without_the_injuries_key(tmp_path):
    with pytest.raises(Exception, match="shape may have changed"):
        espn_news.run_pull(FakeClient({"oops": 1}), tmp_path)


def test_watch_loop_sleeps_between_pulls_and_refuses_hammering(tmp_path, capsys):
    sleeps = []
    c = FakeClient(feed(note("1", "A", espn=1)))
    rc = espn_news.main(["--watch", "10", "--no-offcourt", "--data-dir", str(tmp_path)], client=c, sleep=sleeps.append, max_pulls=3)
    assert rc == 0 and c.calls == 3 and sleeps == [600.0, 600.0]
    assert espn_news.main(["--watch", "1", "--data-dir", str(tmp_path)], client=c) == 2


def test_latest_per_player_drops_players_who_left_the_feed(tmp_path):
    m, _ = espn_news.merge_news(None, espn_news.parse_feed(feed(note("1", "A", espn=1), note("2", "B", espn=2)), T0, ID_MAP))
    m, _ = espn_news.merge_news(m, espn_news.parse_feed(feed(note("2", "B", espn=2)), T1, ID_MAP))
    assert list(m[m["last_seen"] == m["last_seen"].max()]["note_id"]) == ["2"]


def test_news_flag_text_and_overlay_leaves_projection_risk_untouched():
    n = espn_news.parse_feed(feed(note("1", "Aaron Wiggins", espn=4397183)), T0, ID_MAP)
    f = risk.news_flags([1630, 7], n)
    assert f["news_flag"].tolist() == ["news: day-to-day, ankle (leg), est. return Oct 12", ""]
    board = pd.DataFrame({"player_id": [1630, 7], "proj_gp": [70.0, 70.0]})
    out = risk.build_overlay(board, news=f, today=pd.Timestamp("2026-10-09"))
    assert out["risk_flags"].iloc[0].startswith("news: day-to-day") and out["risk_level"].tolist() == ["", ""]
    assert out["risk_gp"].tolist() == [70.0, 70.0]
