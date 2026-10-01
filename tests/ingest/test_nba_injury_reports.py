"""NBA injury-report ingest (ADR 0033): the parser on three layouts, the fetcher's politeness and resume, id matching, the table.

The fixtures below are invented text in the shape ``pdftotext -table`` produces (no NBA.com content is copied), so the tests
need neither the network nor poppler.
"""
import urllib.error
from datetime import date

import pandas as pd
import pytest

from src.ingest import nba_injury_reports as R

LEGACY = """\
                                                                        Injury  Report: 01/10/19 05:30 PM

Game Date   Game Time   Matchup  Team                   Player Name             Category          Reason                        Current Status  Previous  Status

01/10/2019  07:00 (ET)  BOS@MIA  Boston Celtics         Smith, Al               Injury/Illness    Left Fourth Metacarpal        Out             -

                                                                                                  Fracture

                                                        Jones, Bo               Personal Reasons  -                             Out             -

                                 Miami Heat             Khan, Cy                G League Team     -                             Out             -

            09:00 (ET)  LAC@MIN  Los Angeles Clippers   Lee, Di                 Injury/Illness    Left hamstring tightness      Questionable    -

                                 Minnesota               Wu, Ed                  Injury/Illness    Right Knee Bone Bruise        Out             -

                                 Timberwolves

                                                        Ng, Flo                 Injury/Illness    Low back surgery              Probable        Out
"""

MODERN = """\
                                                 Injury  Report: 12/01/21          05:30 PM

Game Date   Game Time   Matchup  Team                    Player Name               Current Status  Reason

12/01/2021  07:00 (ET)  ATL@IND  Atlanta Hawks           Ray, Gus                  Out             Injury/Illness - Right Ankle; Sprain

                                                         Sun, Hal                  Out             G League - Two-Way

                                 Indiana Pacers          Holiday, Ian              Out             Health and Safety Protocols

                                                         Moore Jr., Wendell        Questionable    Injury/Illness - N/a; Non-Covid related illness

                                 Portland Trail Blazers  NOT YET SUBMITTED

            08:00 (ET)  DEN@ORL  Denver Nuggets          Ito, Jay                  Available       -

                                                         Kim, Kit                  Out             Not With Team
"""


# --------------------------------------------------------------------------- parser

def test_legacy_layout_splits_category_reason_and_previous_status():
    rows = R.parse_report_text(LEGACY)
    assert [r["player"] for r in rows] == ["Smith, Al", "Jones, Bo", "Khan, Cy", "Lee, Di", "Wu, Ed", "Ng, Flo"]
    a = rows[0]
    assert (a["status"], a["category"], a["detail"]) == ("Out", "Injury/Illness", "Left Fourth Metacarpal Fracture")
    assert a["report_ts"] == "2019-01-10 05:30 PM" and a["game_date"] == "01/10/2019" and a["matchup"] == "BOS@MIA"
    assert (rows[1]["category"], rows[1]["detail"]) == ("Personal Reasons", "")
    assert rows[2]["category"] == "G League Team" and rows[2]["team"] == "Miami Heat"
    assert rows[5]["status"] == "Probable" and rows[5]["detail"] == "Low back surgery"      # previous status is not part of the reason


def test_a_wrapped_team_name_is_not_glued_onto_a_reason():
    rows = R.parse_report_text(LEGACY)
    wu = next(r for r in rows if r["player"] == "Wu, Ed")
    assert wu["detail"] == "Right Knee Bone Bruise"
    assert R.team_abbr_of(wu["team"]) == "MIN"                    # the first line alone ("Minnesota") resolves
    assert R.team_abbr_of("Minnesota Timberwolves") == "MIN"


def test_modern_layout_splits_reason_on_the_first_dash_and_skips_unsubmitted_teams():
    rows = R.parse_report_text(MODERN)
    assert [r["player"] for r in rows] == ["Ray, Gus", "Sun, Hal", "Holiday, Ian", "Moore Jr., Wendell", "Ito, Jay", "Kim, Kit"]
    assert (rows[0]["category"], rows[0]["detail"]) == ("Injury/Illness", "Right Ankle; Sprain")
    assert (rows[1]["category"], rows[1]["detail"]) == ("G League", "Two-Way")
    assert (rows[2]["category"], rows[2]["detail"]) == ("Health and Safety Protocols", "")
    assert rows[4]["status"] == "Available" and rows[4]["category"] == ""
    assert rows[5]["category"] == "Not With Team" and rows[5]["game_time"] == "08:00 (ET)" and rows[5]["matchup"] == "DEN@ORL"


def test_split_reason_and_names():
    assert R.split_reason("Injury/Illness - Left Knee; Sprain") == ("Injury/Illness", "Left Knee; Sprain")
    assert R.split_reason("-") == ("", "") and R.split_reason("") == ("", "")
    assert R.split_reason("Rest") == ("Rest", "")
    assert R.display_name("Moore Jr., Wendell") == "Wendell Moore Jr."
    assert R.display_name("Nene") == "Nene"
    assert R.team_abbr_of("LA Clippers") == "LAC" and R.team_abbr_of("Los Angeles Lakers") == "LAL"
    assert R.team_abbr_of("Nowhere United") is None and R.team_abbr_of(None) is None


# --------------------------------------------------------------------------- id matching and seasons

def _players():
    return pd.DataFrame({"player_id": [1, 2, 3, 4], "player_name": ["Wendell Moore Jr.", "Gus Ray", "Chris Smith", "Chris Smith"]})


def _logs():
    return pd.DataFrame({"player_id": [3, 4], "season": ["2018-19", "2018-19"], "team_abbr": ["BOS", "MIA"]})


def test_resolver_uses_unique_names_and_breaks_ties_by_team_and_season():
    res = R.PlayerResolver.build(_players(), _logs())
    assert res.resolve("Wendell Moore Jr.", "MIN", "2021-22") == 1
    assert res.resolve("Wendell Moore", None, None) == 1                 # suffix-insensitive
    assert res.resolve("Chris Smith", "MIA", "2018-19") == 4
    assert res.resolve("Chris Smith", "BOS", "2018-19") == 3
    assert res.resolve("Chris Smith", "DEN", "2018-19") is None          # ambiguous and on neither team
    assert res.resolve("Chris Smith", None, "2018-19") is None
    assert res.resolve("Nobody Atall", "MIA", "2018-19") is None


def test_season_of_uses_team_games_and_handles_the_bubble():
    assert R.season_of(date(2018, 12, 25), None) == "2018-19"
    assert R.season_of(date(2019, 3, 1), None) == "2018-19"
    assert R.season_of(date(2020, 8, 10), None) == "2019-20"              # restart games played in August
    assert R.season_of(date(2020, 12, 25), None) == "2020-21"
    assert R.season_of(date(2020, 8, 10), {date(2020, 8, 10): "2019-20"}) == "2019-20"


def test_rows_to_frame_types_and_matching():
    res = R.PlayerResolver.build(_players(), _logs())
    frame = R.rows_to_frame(R.parse_report_text(MODERN), "05PM", res, None)
    assert list(frame.columns) == R.COLUMNS
    assert frame["season"].eq("2021-22").all() and (frame["slot"] == "05PM").all()
    assert str(frame["game_date"].dtype) == "datetime64[ms]" and str(frame["player_id"].dtype) == "Int64"
    by = frame.set_index("player_name")
    assert by.loc["Gus Ray", "player_id"] == 2 and by.loc["Wendell Moore Jr.", "player_id"] == 1
    assert pd.isna(by.loc["Hal Sun", "player_id"])                        # not in the players table
    assert by.loc["Gus Ray", "team_abbr"] == "ATL" and by.loc["Jay Ito", "game_date"] == pd.Timestamp("2021-12-01")


# --------------------------------------------------------------------------- fetcher


class FakeHost:
    """Opener that has reports only for the (date, slot) pairs given; everything else answers 403 like S3."""

    def __init__(self, have):
        self.have, self.calls = set(have), []

    def __call__(self, url):
        self.calls.append(url)
        key = url.rsplit("Injury-Report_", 1)[1][:-4]
        if key in self.have:
            return b"%PDF fake " + key.encode()
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _fetcher(tmp_path, have, **kw):
    clock = Clock()
    host = FakeHost(have)
    f = R.Fetcher(tmp_path, opener=host, sleep=clock.sleep, clock=clock.now, **kw)
    return f, host, clock


def test_fetch_takes_the_first_slot_that_exists_and_remembers_absent_ones(tmp_path):
    f, host, _ = _fetcher(tmp_path, {"2019-01-10_01PM"})
    assert f.fetch_day(date(2019, 1, 10)) == "01PM"
    assert [u.rsplit("_", 1)[1] for u in host.calls] == ["05PM.pdf", "06PM.pdf", "01PM.pdf"]
    assert f.pdf_file(date(2019, 1, 10), "01PM").read_bytes().startswith(b"%PDF")
    assert f.index["fetched"] == {"2019-01-10": "01PM"} and "2019-01-10" not in f.index["absent"]
    n = len(host.calls)
    assert f.fetch_day(date(2019, 1, 10)) == "01PM" and len(host.calls) == n        # cached: no request


def test_a_date_with_no_report_is_probed_once_then_skipped(tmp_path):
    f, host, _ = _fetcher(tmp_path, set())
    assert f.fetch_day(date(2019, 1, 11)) is None and len(host.calls) == len(R.SLOTS)
    assert f.done(date(2019, 1, 11))
    f.save_index()
    f2, host2, _ = _fetcher(tmp_path, set())                                        # a new process reads the index
    assert f2.done(date(2019, 1, 11)) and f2.fetch_day(date(2019, 1, 11)) is None and host2.calls == []


def test_requests_are_spaced_and_budgeted(tmp_path):
    f, host, clock = _fetcher(tmp_path, set(), interval=1.5, max_requests=3)
    assert f.fetch_day(date(2019, 1, 11)) is None
    assert len(host.calls) == 3 and f.requests == 3                                  # the budget stops it mid-date
    assert not f.done(date(2019, 1, 11))                                             # ... and the date stays pending
    assert len(clock.slept) == 2 and all(s == pytest.approx(1.5) for s in clock.slept)


def test_offline_never_touches_the_network(tmp_path):
    f, host, _ = _fetcher(tmp_path, {"2019-01-10_05PM"}, offline=True)
    assert f.fetch_day(date(2019, 1, 10)) is None and host.calls == []


def test_a_server_error_is_reported_not_swallowed(tmp_path):
    def boom(url):
        raise urllib.error.HTTPError(url, 500, "err", {}, None)

    f = R.Fetcher(tmp_path, opener=boom, sleep=lambda s: None)
    with pytest.raises(R.InjuryReportError, match="500"):
        f.fetch_day(date(2019, 1, 10))


# --------------------------------------------------------------------------- end to end (fake host, fake pdftotext)

def test_ingest_builds_the_table_from_the_cache_and_resumes(tmp_path, monkeypatch):
    from src.store import write_table
    from src.synthetic import make_synthetic_tables

    t = make_synthetic_tables(first_start=2018, last_start=2019, n_teams=6, games_per_team=10, seed=1)
    for name in ("team_games", "players", "game_logs"):
        write_table(t[name], name, tmp_path)
    days, _ = R.game_dates(t["team_games"], ["2019-20"], until=date(2030, 1, 1))
    assert days and all(d >= R.FIRST_REPORT_DATE for d in days)
    have = {f"{d.isoformat()}_05PM" for d in days[:3]}
    monkeypatch.setattr(R, "pdf_to_text", lambda pdf: MODERN.replace("12/01/2021", days[0].strftime("%m/%d/%Y")))
    host = FakeHost(have)
    res = R.ingest(tmp_path, seasons=["2019-20"], opener=host, sleep=lambda s: None, until=date(2030, 1, 1))
    assert res.n_fetched == 3 and res.n_dates == len(days) and res.n_pending == 0 and res.parse_failures == []
    table = R.read_injury_reports(tmp_path)
    assert len(table) == 3 * 6 and set(table.columns) == set(R.COLUMNS)
    n_calls = len(host.calls)
    again = R.ingest(tmp_path, seasons=["2019-20"], opener=host, sleep=lambda s: None, until=date(2030, 1, 1))
    assert len(host.calls) == n_calls and again.n_rows == res.n_rows                # nothing re-requested
    partial = R.ingest(tmp_path, seasons=["2019-20"], offline=True, until=date(2030, 1, 1))
    assert partial.requests == 0 and partial.n_rows == res.n_rows
    assert (tmp_path / "processed" / R.REPORT_NAME).exists()


def test_cli_rejects_a_hammering_interval(capsys):
    assert R.main(["--interval", "0.2"]) == 2
    assert "at least 1 second" in capsys.readouterr().err


def test_incremental_run_parses_only_new_days_and_rebuild_reparses_everything(tmp_path, monkeypatch):
    from src.store import write_table
    from src.synthetic import make_synthetic_tables

    t = make_synthetic_tables(first_start=2018, last_start=2019, n_teams=6, games_per_team=10, seed=1)
    for name in ("team_games", "players", "game_logs"):
        write_table(t[name], name, tmp_path)
    days, _ = R.game_dates(t["team_games"], ["2019-20"], until=date(2030, 1, 1))
    parsed = []

    def fake_text(pdf):
        parsed.append(pdf.name)
        d = date.fromisoformat(pdf.name.split("_")[0])
        return MODERN.replace("12/01/2021", d.strftime("%m/%d/%Y"))

    monkeypatch.setattr(R, "pdf_to_text", fake_text)
    kw = dict(seasons=["2019-20"], sleep=lambda s: None, until=date(2030, 1, 1))
    host = FakeHost({f"{d.isoformat()}_05PM" for d in days[:4]})
    R.ingest(tmp_path, opener=host, max_requests=2, **kw)
    assert len(parsed) == 2
    parsed.clear()
    res = R.ingest(tmp_path, opener=host, **kw)
    assert len(parsed) == 2 and res.n_fetched == 4                      # only the two new days were parsed
    table = R.read_injury_reports(tmp_path)
    assert table["game_date"].dt.date.nunique() == 4 and len(table) == 4 * 6 and not table.duplicated(
        ["game_date", "matchup", "team_abbr", "player_name"]).any()
    parsed.clear()
    R.ingest(tmp_path, opener=FakeHost(set()), rebuild=True, **kw)
    assert len(parsed) == 4                                             # --rebuild re-parses every cached PDF


def test_a_recent_date_without_a_report_is_probed_again_but_an_old_one_is_not(tmp_path):
    today = date.today()
    f, host, _ = _fetcher(tmp_path, set())
    assert f.fetch_day(today) is None and not f.done(today)            # all slots absent, yet published-late is possible
    n = len(host.calls)
    f.fetch_day(today)
    assert len(host.calls) == n + len(R.SLOTS)                          # probed again in full
    old = date(2019, 1, 11)
    f.fetch_day(old)
    m = len(host.calls)
    f.fetch_day(old)
    assert len(host.calls) == m and f.done(old)
