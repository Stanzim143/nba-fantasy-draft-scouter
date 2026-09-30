"""wiki_contracts: parsing of dated contract events from team-season wikitext (ADR 0019).

The fixtures below are tiny wikitext snippets written for this test in the shapes seen on real pages (the
2016-17 "Signed" column with contract text, the 2015-16 ``||`` inline rows, the 2019+ Date/Player/Terms
tables with ``rowspan``, and the ``{{sortname}}`` player template of the newest pages). No page text is copied.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ingest_fakes import FakeClock, FakeResponse
from src.ingest import wiki_contracts as wc
from src.ingest import wiki_transactions as wt
from src.ingest.http_cache import CachedHttpClient
from src.store import write_table

GSW, LAL, DAL = 1610612744, 1610612747, 1610612742

OLD_FORMAT = """
==Transactions==
===Trades===
{| class="wikitable"
|-
| July 7, 2016
| To '''Golden State Warriors'''<hr />• Future 2nd round-pick
| To '''[[Dallas Mavericks]]'''<hr />• [[Andrew Bogut]]
|}

===Free agency===

====Re-signed====
{| class="wikitable sortable"
! Player
! Signed
|- style="text-align: center"
| {{flagicon|USA}} [[Ian Clark (basketball)|Ian Clark]]<ref>{{cite web|url=http://x|title=Warriors re-sign Ian Clark|date=July 8, 2016|access-date=July 8, 2016}}</ref>
| 1-year contract worth $980,431
|}

====Additions====
{| class="wikitable sortable"
! Player
! Signed
! Former team
|- style="text-align: center"
| {{flagicon|USA}} [[Kevin Durant]]<ref>{{cite web|url=http://x|title=Warriors sign Kevin Durant|date=July 7, 2016}}</ref>
| 2-year contract worth $54.3&nbsp;million
| [[Oklahoma City Thunder]]
|- style="text-align: center"
| {{flagicon|USA}} [[Brianté Weber]]<ref>{{cite web|url=http://x|title=ten day|date=February 4, 2017}}</ref>
| 10-day contract worth $51,449<br />12-day contract worth $61,739
| [[Sioux Falls Skyforce]] {{small|([[NBA D-League]])}}
|- style="text-align: center"
| {{flagicon|USA}} [[Matt Barnes]]<ref>{{cite web|url=http://x|title=Warriors sign Matt Barnes|date=March 2, 2017}}</ref>
|
| [[Sacramento Kings]]
|}

====Subtractions====
{| class="wikitable sortable"
! Player
! Reason left
! New team
|- style="text-align: center"
| {{flagicon|USA}} [[Harrison Barnes]]<ref>{{cite web|url=http://x|title=Mavs sign Barnes|date=July 7, 2016}}</ref>
| 4-year contract worth $94&nbsp;million
| [[Dallas Mavericks]]
|- style="text-align: center"
| {{flagicon|BRA}} [[Anderson Varejão]]<ref>{{cite web|url=http://x|title=Warriors Waive Varejão|date=February 3, 2017}}</ref>
| Waived
| {{N/A}}
|- style="text-align: center"
| [[Leandro Barbosa]]<ref>{{cite web|url=http://x|title=Suns sign Barbosa|date=July 19, 2016}}</ref>
| Unrestricted free agent
| [[Phoenix Suns]]
|}

==Awards==
|-
| [[Not A Transaction]]
| 5-year contract worth $1
|}
"""

INLINE_FORMAT = """
==Transactions==
====Re-signed====
{| class="wikitable"
! Player
! Signed
! Former Team
|-
|| [[Paul Millsap]]<ref>{{cite web|title=Hawks reach agreement|date=July 9, 2015}}</ref> || Signed 3-year contract worth $59 Million || Atlanta Hawks
|-
|| [[Nikola Jokic]] || Multi–year extension<br>4 years, $46.5 million ||
|}
"""

NEW_FORMAT = """
== Draft picks ==
{| class="wikitable"
|-
| 2
| 55
| [[Lachlan Olbrich]]
|}
== Transactions ==
=== Free agency ===
==== Re-signed ====
{| class="wikitable"
! Date
! Player
! Contract Terms
! Ref.
|-
| July 6, 2025
| [[Jaxson Hayes]]
| 1-year $3.4M contract
|
|-
| July 22, 2025
| {{sortname|LeBron|James}}
| 2-year $101M contract
| <ref>{{cite web|title=Lakers re-sign LeBron|date=July 22, 2025}}</ref>
|}

==== Additions ====
{| class="wikitable"
! Date
! Player
! Contract Terms
! Former Team
! Ref.
|-
| rowspan='2' | July 6, 2025
| [[Deandre Ayton]]
| 2-year $16M contract
| [[Portland Trail Blazers]]
|
|-
| [[Jake LaRavia]]
| 2-year $12M contract
| [[Sacramento Kings]]
|
|-
| July 24, 2025
| [[Chris Mañon]]
| rowspan='3' | [[Two-way contract]]
| [[Vanderbilt Commodores men's basketball|Vanderbilt Commodores]]
|
|-
| September 30, 2025
| [[Nick Smith Jr.]]
| [[Charlotte Hornets]]
|
|-
| November 24, 2025
| Drew Timme
| [[Brooklyn Nets]]
|
|-
| July 30
| [[Lachlan Olbrich]]
| [[Two-way contract]]
| [[Illawarra Hawks]]
|
|-
| November 2
| [[Terrance Shannon]]
| [[Sioux Falls Skyforce]]
| <ref>{{cite web|title=Lakers sign Terrance Shannon to 10-day contract|date=November 2, 2025}}</ref>
|
<!--
|-
| July 1, 2025
| [[Ghost Player]]
| 5-year $500M contract
-->
|}

==== Subtractions ====
{| class="wikitable"
! Date
! Player
! Reason
! New Team
! Ref.
|-
| July 6, 2025
| [[Dorian Finney-Smith]]
| 4-year $53M contract
| [[Houston Rockets]]
|
|-
| rowspan='2' | July 20, 2025
| [[Jordan Goodwin]]
| rowspan='2' | Waived
| [[Phoenix Suns]]
|
|-
| [[Shake Milton]]
| [[KK Partizan]]
|
|-
| N/A
| [[Markieff Morris]]
| Expired contract
|
|-
| July 10
| [[Tim Hardaway Jr.]]
| [[Free agent|Unrestricted free agent]]
| [[Denver Nuggets]]
|
|}
"""


def by_name(rows: list[wc.ParsedRow]) -> dict[str, wc.ParsedRow]:
    out: dict[str, wc.ParsedRow] = {}
    for r in rows:
        out.setdefault(r.wiki_name, r)
    return out


# --------------------------------------------------------------------------- tags and dates

@pytest.mark.parametrize("date,tag,cov", [
    ("2016-07-07", 2015, 2016),      # summer: follows the season that just ended, covers the coming one
    ("2016-09-30", 2015, 2016),      # last day before the cutoff
    ("2016-10-01", 2016, 2016),      # opening month: follows (and covers) the new season
    ("2017-02-04", 2016, 2016),      # in-season
    ("2017-06-30", 2016, 2016),      # before the league year turns
    ("2017-07-01", 2016, 2017),      # the league year turns on 1 Jul
])
def test_tag_and_coverage_years(date, tag, cov):
    d = pd.Timestamp(date)
    assert wc.tag_start_year(d) == tag and wc.coverage_start_year(d) == cov


def test_tag_is_consistent_with_the_history_cutoff():
    """An event tagged T is visible to target season T+1 and never to T: History.until drops season >= target."""
    for date in ("2016-07-07", "2016-09-30", "2016-10-01", "2017-03-15"):
        tag = wc.tag_start_year(pd.Timestamp(date))
        visible_targets = [y for y in range(2014, 2020) if tag < y]
        d = pd.Timestamp(date)
        # visible exactly when the event is dated before 1 Oct of the target's start year
        assert visible_targets == [y for y in range(2014, 2020) if d < pd.Timestamp(y, 10, 1)]


def test_parse_cell_date_variants():
    assert wc.parse_cell_date("July 6, 2025", "2025-26") == (pd.Timestamp("2025-07-06"), "cell")
    assert wc.parse_cell_date("6 July 2025", "2025-26") == (pd.Timestamp("2025-07-06"), "cell")
    assert wc.parse_cell_date("July 6", "2025-26") == (pd.Timestamp("2025-07-06"), "cell_no_year")
    assert wc.parse_cell_date("February 4", "2016-17") == (pd.Timestamp("2017-02-04"), "cell_no_year")
    assert wc.parse_cell_date("Waived (December 5, 2016)", "2016-17")[0] == pd.Timestamp("2016-12-05")
    d, src = wc.parse_cell_date("1-year contract worth $980,431", "2016-17")
    assert pd.isna(d) and src == "none"


def test_parse_years_amount_and_words():
    assert wc.parse_years("2-year contract worth $54.3 million") == 2
    assert wc.parse_years("Signed three year deal") == 3
    assert wc.parse_years("4 years, $46.5 million") == 4
    assert wc.parse_years("4-yr/$75.6M extension") == 4
    assert wc.parse_years("10-day contract") is None
    assert wc.parse_years("a 20-year-old guard") is None
    assert wc.parse_amount("2-year contract worth $54.3 million") == pytest.approx(54.3e6)
    assert wc.parse_amount("1-year $3.4M contract") == pytest.approx(3.4e6)
    assert wc.parse_amount("contract worth $980,431") == pytest.approx(980431)
    assert wc.parse_amount("$388 thousand") == pytest.approx(388e3)
    assert wc.parse_amount("worth $5") is None            # implausible: not a salary
    assert wc.parse_amount("no money here") is None


# --------------------------------------------------------------------------- table grid

def test_table_grid_carries_rowspan_and_splits_inline_cells():
    body = "{|\n|-\n| rowspan='2' | July 6\n| [[A One]]\n| x\n|-\n| [[B Two]]\n| y\n|}"
    rows = wc.table_grid(body)
    assert [[c.raw.strip() for c in r] for r in rows] == [["July 6", "[[A One]]", "x"], ["July 6", "[[B Two]]", "y"]]
    assert [c.inherited for c in rows[1]] == [True, False, False]
    inline = wc.table_grid("{|\n|-\n|| [[A One]] || 2-year || Team\n|}")
    assert [c.raw.strip() for c in inline[0]] == ["[[A One]]", "2-year", "Team"]


# --------------------------------------------------------------------------- page parsing

def test_old_format_page():
    pg = wc.parse_team_season_contracts(OLD_FORMAT, GSW, "2016-17")
    r = by_name(pg.rows)
    assert set(r) == {"Ian Clark", "Kevin Durant", "Brianté Weber", "Matt Barnes", "Harrison Barnes", "Anderson Varejão",
                      "Leandro Barbosa"}                        # the Awards table and the Trades table are not parsed
    clark = r["Ian Clark"]
    assert (clark.event, clark.years, clark.amount_usd, clark.contract_type) == ("resign", 1, 980431.0, "standard")
    assert clark.date == pd.Timestamp("2016-07-08") and clark.date_source == "citation"
    durant = r["Kevin Durant"]
    assert (durant.event, durant.years, durant.team_id) == ("sign", 2, GSW)
    assert durant.amount_usd == pytest.approx(54.3e6)
    weber = r["Brianté Weber"]
    assert weber.is_ten_day and weber.contract_type == "ten_day" and weber.years is None and weber.days == 10
    barnes_matt = r["Matt Barnes"]
    assert barnes_matt.event == "sign" and not barnes_matt.has_terms and barnes_matt.years is None
    # a departure row with contract text is a signing by the NEW team
    hb = r["Harrison Barnes"]
    assert (hb.event, hb.years, hb.team_id, hb.section) == ("sign", 4, DAL, "sub")
    assert r["Anderson Varejão"].event == "waived" and r["Anderson Varejão"].date == pd.Timestamp("2017-02-03")
    assert r["Leandro Barbosa"].event == "expired"


def test_inline_pipe_format_and_multiyear_terms():
    pg = wc.parse_team_season_contracts(INLINE_FORMAT, 1610612737, "2015-16")
    r = by_name(pg.rows)
    assert (r["Paul Millsap"].event, r["Paul Millsap"].years) == ("resign", 3)
    assert r["Paul Millsap"].amount_usd == pytest.approx(59e6)
    assert r["Nikola Jokic"].event == "extension" and r["Nikola Jokic"].years == 4
    assert r["Nikola Jokic"].amount_usd == pytest.approx(46.5e6) and r["Nikola Jokic"].is_extension


def test_new_format_rowspan_sortname_plain_names_and_comments():
    pg = wc.parse_team_season_contracts(NEW_FORMAT, LAL, "2025-26")
    r = by_name(pg.rows)
    assert "Ghost Player" not in r                              # commented-out row
    assert r["LeBron James"].years == 2 and r["LeBron James"].event == "resign"     # {{sortname}} player
    assert r["Drew Timme"].date == pd.Timestamp("2025-11-24") and r["Drew Timme"].contract_type == "two_way"  # plain-text name, inherited two-way cell
    ayton, lar = r["Deandre Ayton"], r["Jake LaRavia"]
    assert ayton.date == lar.date == pd.Timestamp("2025-07-06") and (ayton.years, lar.years) == (2, 2)   # rowspan date
    mañon, smith = r["Chris Mañon"], r["Nick Smith Jr."]
    assert mañon.is_two_way and smith.is_two_way and smith.date == pd.Timestamp("2025-09-30")        # rowspan terms
    assert r["Lachlan Olbrich"].contract_type == "two_way" and r["Lachlan Olbrich"].is_rookie          # in the draft table
    assert r["Lachlan Olbrich"].date == pd.Timestamp("2025-07-30") and r["Lachlan Olbrich"].date_source == "cell_no_year"
    shannon = r["Terrance Shannon"]
    assert shannon.is_ten_day and shannon.terms_source == "title"    # terms only in the cited article's title
    assert r["Dorian Finney-Smith"].event == "sign" and r["Dorian Finney-Smith"].team_id == 1610612745
    assert r["Jordan Goodwin"].event == "waived" and r["Shake Milton"].event == "waived"
    assert r["Shake Milton"].date == pd.Timestamp("2025-07-20")     # inherited date
    assert r["Markieff Morris"].event == "expired" and pd.isna(r["Markieff Morris"].date)
    assert r["Tim Hardaway Jr."].event == "expired"                  # "[[Free agent|Unrestricted free agent]]" is text, not a player


def test_a_page_without_transactions_or_subsections_yields_nothing():
    assert wc.parse_team_season_contracts("== Roster ==\nnothing", GSW, "2016-17").rows == []
    assert wc.parse_team_season_contracts("==Transactions==\n===Trades===\n{|\n|}\n", GSW, "2016-17").rows == []


def test_long_deal_beats_ten_day_in_one_cell():
    txt = ("==Transactions==\n====Additions====\n{|\n|-\n| [[Archie Goodwin]]<ref>{{cite web|date=March 1, 2017}}</ref>\n"
           "| Two 10-day contracts / 2-year contract worth $2 million\n| [[Phoenix Suns]]\n|}")
    r = wc.parse_team_season_contracts(txt, 1610612751, "2016-17").rows[0]
    assert r.years == 2 and not r.is_ten_day and r.contract_type == "standard"


def test_undated_row_is_kept_and_tagged_conservatively():
    txt = ("==Transactions==\n====Additions====\n{|\n|-\n| [[Kris Humphries]]\n| Signed 1-year contract worth $388 thousand\n"
           "| [[Phoenix Suns]]\n|}")
    r = wc.parse_team_season_contracts(txt, 1610612737, "2015-16").rows[0]
    assert pd.isna(r.date) and r.date_source == "none" and r.years == 1
    f = wc._rows_to_frame([r])
    assert f.loc[0, "season"] == "2015-16" and f.loc[0, "start_year"] == 2015     # page season: invisible to that season's target


# --------------------------------------------------------------------------- frame, dedupe, schema

def _frame_from(rows, ids):
    raw = wc._rows_to_frame(rows)
    raw["player_id"] = [ids.get(n) for n in raw["wiki_name"]]
    return raw[raw["player_id"].notna()]


def test_dedupe_collapses_a_departure_and_its_arrival_preferring_the_arrival():
    dep = wc.parse_team_season_contracts(OLD_FORMAT, GSW, "2016-17")           # Harrison Barnes leaves for DAL
    arr_txt = ("==Transactions==\n====Additions====\n{|\n|-\n| [[Harrison Barnes]]<ref>{{cite web|date=July 9, 2016}}</ref>\n"
               "| 4-year contract worth $94 million\n| [[Golden State Warriors]]\n|}")
    arr = wc.parse_team_season_contracts(arr_txt, DAL, "2016-17")
    ids = {"Harrison Barnes": 7, "Ian Clark": 8}
    f = _frame_from(dep.rows + arr.rows, ids)
    out = wc.dedupe_events(f)
    hb = out[out["player_id"] == 7]
    assert len(hb) == 1 and hb.iloc[0]["section"] == "add" and hb.iloc[0]["team_id"] == DAL
    far = wc.dedupe_events(_frame_from(dep.rows + [wc.ParsedRow(**{**arr.rows[0].__dict__, "date": pd.Timestamp("2016-08-30")})], ids))
    assert len(far[far["player_id"] == 7]) == 2            # more than a week apart: two distinct events


def test_finalize_and_validate_roundtrip(tmp_path):
    pg = wc.parse_team_season_contracts(OLD_FORMAT, GSW, "2016-17")
    ids = {r.wiki_name: i + 1 for i, r in enumerate(pg.rows)}
    frame = wc.finalize_frame(wc.dedupe_events(_frame_from(pg.rows, ids)).drop(columns=["wiki_name", "has_terms"]))
    assert list(frame.columns) == wc.PLAYER_CONTRACTS_COLUMNS
    assert frame["player_id"].dtype == "int64" and str(frame["years"].dtype) == "Int64"
    assert "wiki_name" not in frame.columns                # no source text is stored
    path = wc.write_player_contracts(frame, tmp_path)
    back = wc.read_player_contracts(tmp_path)
    assert path.exists() and len(back) == len(frame)
    pd.testing.assert_frame_equal(back.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False)


def _valid_frame():
    pg = wc.parse_team_season_contracts(OLD_FORMAT, GSW, "2016-17")
    ids = {r.wiki_name: i + 1 for i, r in enumerate(pg.rows)}
    return wc.finalize_frame(wc.dedupe_events(_frame_from(pg.rows, ids)).drop(columns=["wiki_name", "has_terms"]))


def test_schema_rejects_bad_frames():
    good = _valid_frame()
    wc.validate_player_contracts(good)
    with pytest.raises(wc.WikiContractsError, match="missing columns"):
        wc.validate_player_contracts(good.drop(columns=["event"]))
    with pytest.raises(wc.WikiContractsError, match="event"):
        wc.validate_player_contracts(good.assign(event="dunked"))
    with pytest.raises(wc.WikiContractsError, match="contract_type"):
        wc.validate_player_contracts(good.assign(contract_type="max"))
    bad_years = good.copy()
    bad_years["years"] = pd.array([9] + [pd.NA] * (len(good) - 1), dtype="Int64")
    with pytest.raises(wc.WikiContractsError, match="years"):
        wc.validate_player_contracts(bad_years)
    bad_tag = good.copy()
    bad_tag["season"] = "2030-31"                          # tag disagrees with the date: the leak guard would lie
    with pytest.raises(wc.WikiContractsError, match="tag"):
        wc.validate_player_contracts(bad_tag)
    with pytest.raises(wc.WikiContractsError, match="malformed season"):
        wc.validate_player_contracts(good.assign(page_season="20xx"))
    with pytest.raises(wc.WikiContractsError, match="nulls"):
        wc.validate_player_contracts(good.assign(player_id=np.nan))
    with pytest.raises(wc.WikiContractsError, match="duplicate"):
        wc.validate_player_contracts(pd.concat([good, good.iloc[[0]]], ignore_index=True))


def test_read_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wiki_contracts"):
        wc.read_player_contracts(tmp_path)


# --------------------------------------------------------------------------- pipeline (fake HTTP)

class _Session:
    def __init__(self, by_title):
        self.by_title = by_title

    def get(self, url, params=None, headers=None, timeout=None):
        title = dict(params or {}).get("title")
        return FakeResponse(200, self.by_title[title]) if title in self.by_title else FakeResponse(404, b"missing")


def _players(names_ids):
    return pd.DataFrame({
        "player_id": [i for _, i in names_ids], "player_name": [n for n, _ in names_ids],
        "birthdate": pd.to_datetime(["1990-01-01"] * len(names_ids)), "position": ["F"] * len(names_ids),
        "height_in": [80.0] * len(names_ids), "weight_lb": [220.0] * len(names_ids),
        "draft_year": pd.array([2010] * len(names_ids), dtype="Int64"), "draft_round": pd.array([1] * len(names_ids), dtype="Int64"),
        "draft_number": pd.array([5] * len(names_ids), dtype="Int64"), "from_year": pd.array([2010] * len(names_ids), dtype="Int64"),
        "to_year": pd.array([2026] * len(names_ids), dtype="Int64")})


def test_run_ingest_end_to_end(tmp_path):
    base = tmp_path / "data"
    write_table(_players([("Kevin Durant", 501), ("Ian Clark", 502), ("Harrison Barnes", 503)]), "players", base)
    clock = FakeClock()
    client = CachedHttpClient(tmp_path / "raw_wiki", session=_Session({wt.team_season_page_title(GSW, "2016-17"): OLD_FORMAT}),
                              clock=clock, sleep=clock.sleep, offline=False, min_interval=0, jitter=0)
    res = wc.run_ingest(["2016-17"], client, base, log=lambda *_: None)
    assert res.fetched == 1 and len(res.skipped_pages) == 29
    df = wc.read_player_contracts(base)
    assert set(df["player_id"]) == {501, 502, 503}
    kd = df[df["player_id"] == 501].iloc[0]
    assert (kd["event"], int(kd["years"]), kd["signed_date"], kd["season"], int(kd["start_year"])) == \
        ("sign", 2, pd.Timestamp("2016-07-07"), "2015-16", 2016)
    assert kd["team_id"] == GSW
    hb = df[df["player_id"] == 503]
    assert len(hb) == 1 and hb.iloc[0]["team_id"] == DAL and hb.iloc[0]["event"] == "sign"
    rep = json.loads((base / "processed" / wc.REPORT_NAME).read_text(encoding="utf-8"))
    assert rep["totals"]["matched_rows"] == len(df) and rep["by_season"]["2016-17"]["pages"] == 1
    assert rep["totals"]["contract_event_rows"] >= 5 and "id_matching" in rep
    # the ad-hoc table is not a contract table and the id map is not touched
    assert not (base / "processed" / "player_id_map.parquet").exists()


def test_run_ingest_with_nothing_fetched_is_a_clear_error(tmp_path):
    base = tmp_path / "data"
    write_table(_players([("Kevin Durant", 501)]), "players", base)
    clock = FakeClock()
    client = CachedHttpClient(tmp_path / "raw_wiki", session=_Session({}), clock=clock, sleep=clock.sleep,
                              offline=False, min_interval=0, jitter=0)
    with pytest.raises(wc.WikiContractsError, match="no contract rows"):
        wc.run_ingest(["2016-17"], client, base, log=lambda *_: None)


def test_cli_rejects_bad_seasons(capsys):
    with pytest.raises(SystemExit):
        wc.main(["--seasons", "20xx"])
    assert "malformed season" in capsys.readouterr().err


def test_an_expired_short_deal_on_a_departure_row_is_not_a_signing():
    txt = ("==Transactions==\n====Subtractions====\n{|\n|-\n| [[Quincy Acy]]\n| Second 10-day contract expired\n"
           "| [[Texas Legends]] / [[Shenzhen New Century Leopards]]\n|}")
    r = wc.parse_team_season_contracts(txt, 1610612756, "2018-19").rows[0]
    assert r.event == "expired" and r.years is None


def test_a_plain_text_player_beats_a_later_linked_foreign_club():
    txt = ("==Transactions==\n====Subtractions====\n{|\n|-\n| Matt Janning\n| Waived\n| October 15, 2015\n"
           "| [[Hapoel Jerusalem B.C.]]\n|}")
    r = wc.parse_team_season_contracts(txt, 1610612743, "2015-16").rows[0]
    assert r.wiki_name == "Matt Janning" and r.event == "waived" and r.date == pd.Timestamp("2015-10-15")


# --------------------------------------------------------------------------- dts dates, citation window, yearless rolling (verifier fixes)

DTS_PAGE = """
==Transactions==
===Free agency===
====Additions====
{| class="wikitable"
! Date !! Player !! Terms
|-
| {{dts|July 7, 2016}} || [[Andrew Nicholson]] || 1-year contract
|-
| {{Dts|2016|July|9}} || [[Bo Two]] || 2-year contract
|-
| {{dts|format=dmy|2016-07-10}} || [[Cy Three]] || 1-year contract
|}
"""


def test_dts_template_dates_are_read_from_the_row_cell():
    assert wc.expand_dts("{{dts|July 7, 2016}}") == "July 7, 2016"
    assert wc.expand_dts("{{Dts|2016|July|9}}") == "July 9, 2016"
    assert wc.expand_dts("{{dts|2016|7|9}}") == "July 9, 2016"
    assert wc.expand_dts("{{dts|format=dmy|2016-07-10}}") == "July 10, 2016"
    assert wc.expand_dts("{{dts|weird|shape}}") == "{{dts|weird|shape}}"
    r = by_name(wc.parse_team_season_contracts(DTS_PAGE, GSW, "2016-17").rows)
    assert (r["Andrew Nicholson"].date, r["Andrew Nicholson"].date_source) == (pd.Timestamp("2016-07-07"), "cell")
    assert r["Bo Two"].date == pd.Timestamp("2016-07-09") and r["Cy Three"].date == pd.Timestamp("2016-07-10")
    assert r["Andrew Nicholson"].years == 1                     # terms text is unaffected by the expansion


CITE_PAGE = """
==Transactions==
===Free agency===
====Additions====
{| class="wikitable"
! Player !! Terms
|-
| [[Late Cite]]<ref>{{cite web|url=http://x|title=T|date=October 24, 2019|access-date=November 1, 2019}}</ref> || 2-year contract
|-
| [[Old Cite]]<ref>{{cite web|url=http://x|title=Background piece|date=March 3, 2018}}</ref> || 2-year contract
|-
| [[Prev Summer]]<ref>{{cite web|url=http://x|title=Piece from the summer before|date=August 3, 2018}}</ref> || 2-year contract
|-
| [[Prev Access]]<ref>{{cite web|url=http://x|title=No date|access-date=August 3, 2018}}</ref> || 2-year contract
|}
"""


def test_citation_date_is_on_or_after_event_and_never_earlier_than_page_window():
    r = by_name(wc.parse_team_season_contracts(CITE_PAGE, LAL, "2019-20").rows)
    late = r["Late Cite"]
    assert (late.date, late.date_source) == (pd.Timestamp("2019-10-24"), "citation")
    assert late.date >= pd.Timestamp("2019-07-01")              # may be LATER than the real signing, never earlier
    assert pd.isna(r["Old Cite"].date) and r["Old Cite"].date_source == "none"   # predates the page window: ignored
    # August of the PREVIOUS year is inside the loose window (1 Jun y0-1) but before the page's own 1 Jun y0:
    # the strict citation / access-date window rejects it (an earlier date would be the leaking direction)
    assert pd.isna(r["Prev Summer"].date) and pd.isna(r["Prev Access"].date)
    assert wc._plausible(pd.Timestamp("2018-08-03"), "2019-20") and not wc._plausible(pd.Timestamp("2018-08-03"), "2019-20", strict=True)


ROLL_PAGE = """
==Transactions==
===Free agency===
====Subtractions====
{| class="wikitable"
! Date !! Player !! Reason
|-
| January 23 || [[Early One]] || waived
|-
| June 30 || [[Summer After]] || contract expired
|-
| July 9 || [[Later Summer]] || contract expired
|}
====Additions====
{| class="wikitable"
! Date !! Player !! Terms
|-
| June 30 || [[Pat Alpha]] || 1-year contract
|-
| July 9 || [[Sam Beta]] || 1-year contract
|}
"""


def test_yearless_dates_after_a_later_dated_row_roll_to_the_next_year():
    r = by_name(wc.parse_team_season_contracts(ROLL_PAGE, GSW, "2023-24").rows)
    assert r["Early One"].date == pd.Timestamp("2024-01-23")
    assert r["Summer After"].date == pd.Timestamp("2024-06-30")     # not 2023: the table has already reached January 2024
    assert r["Later Summer"].date == pd.Timestamp("2024-07-09")
    # a separate table with no earlier later-dated row keeps the page-year reading (residual ambiguity, ADR 0019)
    assert r["Pat Alpha"].date == pd.Timestamp("2023-06-30") and r["Sam Beta"].date == pd.Timestamp("2023-07-09")


RAPTORS_PAGE = """
==Transactions==
===Free agency===
====Subtractions====
{| class="wikitable"
! Date !! Player !! Reason
|-
| April 17 || [[Jontay Porter]]<ref>{{cite web|url=http://x|title=Porter banned|date=April 17, 2024}}</ref> || banned
|-
| June 30 || [[Gueye Row]]<ref>{{cite web|url=http://x|title=Gueye|date=July 6, 2024}}</ref> || contract expired
|-
| July 6 || [[Trent Row]]<ref>{{cite web|url=http://x|title=Trent|date=July 6, 2024}}</ref> || contract expired
|-
| June 30 || [[Truly Rolled]]<ref>{{cite web|url=http://x|title=Late|date=July 1, 2025}}</ref> || contract expired
|}
"""


def test_yearless_roll_is_cross_checked_against_the_rows_own_citation():
    r = by_name(wc.parse_team_season_contracts(RAPTORS_PAGE, GSW, "2024-25").rows)
    # the table opens with an April 2024 event read as 2025-04-17 (a year late); it must not anchor a roll
    assert r["Jontay Porter"].date == pd.Timestamp("2025-04-17")
    assert r["Gueye Row"].date == pd.Timestamp("2024-06-30")           # citation says 2024: NOT rolled to 2025
    assert r["Trent Row"].date == pd.Timestamp("2024-07-06")
    # without the citation evidence the same run would have rolled; a citation that agrees with the roll allows it
    assert r["Truly Rolled"].date == pd.Timestamp("2024-06-30")        # run_max is Jul 2024, gap < 120d: no roll
    later = RAPTORS_PAGE.replace("| April 17 ||", "| January 12 ||").replace("April 17, 2024", "January 12, 2025")
    r2 = by_name(wc.parse_team_season_contracts(later, GSW, "2024-25").rows)
    assert r2["Jontay Porter"].date == pd.Timestamp("2025-01-12")
    assert r2["Gueye Row"].date == pd.Timestamp("2024-06-30")          # own citation (Jul 2024) contradicts a roll
    assert r2["Truly Rolled"].date == pd.Timestamp("2025-06-30")       # citation 2025 agrees: rolled
    for row in r2.values():                                            # a roll only ever moves a date later
        assert row.date >= pd.Timestamp("2024-06-01")
