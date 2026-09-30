"""wiki_coaches: infobox coach parsing, the current-coaches list, and the table writes. Offline; wikitext is in the pages' style."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingest import wiki_coaches as wc
from src.ingest.wiki_transactions import TEAM_WIKI_NAMES

GSW, BKN, CLE = 1610612744, 1610612751, 1610612739


def box(field):
    return f"{{{{Infobox NBA season\n| team = X\n| coach = {field}\n| owners = [[Someone]]\n}}}}\n"


def test_single_coach_and_disambiguated_link():
    (e,) = wc.parse_coach_field(box("[[Mike Brown (basketball, born 1970)|Mike Brown]]"))
    assert (e.name, e.status) == ("Mike Brown", "")


def test_fired_then_interim_and_refs_and_notes_are_ignored():
    text = box("[[Jacque Vaughn]] (fired)<ref>{{cite web|title=[[Not A Coach]]}}</ref><br> [[Kevin Ollie]] (interim)")
    assert [(e.name, e.status) for e in wc.parse_coach_field(text)] == [("Jacque Vaughn", "fired"), ("Kevin Ollie", "interim")]
    kerr = box("[[Steve Kerr]]<br>[[Luke Walton]]{{efn|Walton served as interim head coach while [[Steve Kerr]] recovered.<ref name=\"x\"/>}} <small>(interim)</small>")
    assert [(e.name, e.status) for e in wc.parse_coach_field(kerr)] == [("Steve Kerr", ""), ("Luke Walton", "interim")]


def test_ubl_list_and_absence_tags():
    text = box("{{ubl|[[Wes Unseld Jr.]] (fired)|[[Brian Keefe]] (interim)}}")
    assert [(e.name, e.status) for e in wc.parse_coach_field(text)] == [("Wes Unseld Jr.", "fired"), ("Brian Keefe", "interim")]
    text = box("[[Chauncey Billups]] (on leave)<br>[[Tiago Splitter]] (acting)")
    assert [e.status for e in wc.parse_coach_field(text)] == ["acting", "acting"]
    assert wc.parse_coach_field("no infobox here") == []


def test_coach_key_is_stable_across_spellings():
    assert wc.coach_key("Wes Unseld Jr.") == wc.coach_key("Wes Unseld") == "wes unseld"
    assert wc.coach_key("Rajaković") == wc.coach_key("Rajakovic")


def test_page_rows_flag_the_opening_coach_and_use_the_documented_override():
    rows, how = wc.page_rows(box("[[A One]] (fired)<br>[[B Two]]"), GSW, "2019-20")
    assert how == "infobox" and [r["is_opening"] for r in rows] == [True, False] and {r["n_coaches"] for r in rows} == {2}
    rows, how = wc.page_rows("no infobox", CLE, "2015-16")
    assert how == "override" and rows[0]["coach_name"] == "David Blatt" and rows[1]["coach_name"] == "Tyronn Lue"
    assert wc.page_rows("no infobox", GSW, "2019-20") == ([], "missing")


LIST = """==Coaches==
{| class="wikitable"
|-
| [[Atlanta Hawks]]
| [[File:x.jpg|120px]]
! {{sortname|Quin|Snyder}}
| [[Southeast Division (NBA)|Southeast]]
| [[2022–23 NBA season|{{Dts|2023|February|26}}]]
| 267 || 132
|-
| [[Los Angeles Clippers]]
| [[File:y.jpg]]
! style="background-color:#DDFFDD" |{{sortname|Tyronn|Lue}}*
| [[2020–21 NBA season|{{Dts|2020|October|20}}]]
|-
| [[New York Knicks]]
! {{sortname|Mike|Brown|dab=basketball, born 1970}}
| [[2025–26 NBA season|{{Dts|2025|July|7}}]]
|-
| [[Not A Team]]
! {{sortname|No|Body}}
| {{Dts|2020|May|1}}
|}
"""


def test_current_list_parsing():
    got = {r["team_id"]: r for r in wc.parse_current_list(LIST)}
    assert got[1610612737]["coach_name"] == "Quin Snyder" and got[1610612737]["start_date"] == pd.Timestamp("2023-02-26")
    assert got[1610612746]["coach_name"] == "Tyronn Lue"                 # 'Los Angeles Clippers'
    assert got[1610612752]["coach_name"] == "Mike Brown" and len(got) == 3   # the unknown team is skipped, not guessed


class Client:
    offline = False

    def __init__(self, text):
        self.text, self.stats = text, SimpleNamespace(network_requests=0, cache_hits=0)

    def get_text(self, label, url, params=None, *, headers=None, refresh=False):
        self.refreshed = refresh
        return self.text


def full_list():
    rows = ["==Coaches==", "{| class=\"wikitable\""]
    for i, name in enumerate(sorted(TEAM_WIKI_NAMES.values())):
        rows += ["|-", f"| [[{name}]]", f"! {{{{sortname|Coach|N{i}}}}}", "| {{Dts|2024|May|1}}"]
    return "\n".join(rows + ["|}"])


def test_current_run_needs_all_thirty_teams_and_replaces_only_that_season(tmp_path):
    with pytest.raises(wc.WikiCoachesError, match="3 of 30"):
        wc.run_current("2026-27", Client(LIST), tmp_path, log=lambda *_: None)
    assert not wc.table_path(tmp_path).exists()
    old, _ = wc.page_rows(box("[[A One]]"), GSW, "2024-25")
    wc._write(wc._typed(pd.DataFrame(old, columns=wc.COLUMNS)), tmp_path)
    c = Client(full_list())
    wc.run_current("2026-27", c, tmp_path, log=lambda *_: None)
    assert c.refreshed is True
    t = wc.read_team_coaches(tmp_path)
    assert (t["season"] == "2026-27").sum() == 30 and (t["season"] == "2024-25").sum() == 1
    assert set(t.loc[t["season"] == "2026-27", "source"]) == {wc.SOURCE_CURRENT}
    wc.run_current("2026-27", c, tmp_path, log=lambda *_: None)          # a second run replaces, never duplicates
    assert (wc.read_team_coaches(tmp_path)["season"] == "2026-27").sum() == 30


def test_run_ingest_writes_completed_seasons_and_reports_missing(tmp_path, monkeypatch):
    def fake_fetch(client, team_id, season, refresh=False):
        return box("[[Coach A]]<br>[[Coach B]] (interim)") if team_id != BKN else "page without a coach field"

    monkeypatch.setattr(wc, "fetch_team_season_wikitext", fake_fetch)
    res = wc.run_ingest(["2019-20"], Client(""), tmp_path, log=lambda *_: None)
    assert res.pages == 30 and len(res.missing) == 1 and res.missing[0]["team"] == "Brooklyn Nets"
    assert res.rows == 58 and res.multi_coach_seasons == 29
    t = wc.read_team_coaches(tmp_path)
    assert t["is_opening"].sum() == 29 and t["start_date"].isna().all()


def test_missing_table_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wiki_coaches"):
        wc.read_team_coaches(tmp_path)
