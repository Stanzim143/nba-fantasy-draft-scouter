"""Draft date/time from config, the pre-draft window and final runs, and the time zone arithmetic (ADR 0014)."""
import json
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import daily_refresh as dr
from src.ops import schedule as sc

NZ = ZoneInfo("Pacific/Auckland")
CET = ZoneInfo("Europe/Berlin")
UTC = timezone.utc
START = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)          # earliest plausible draft start
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def repo_with_config(tmp_path, draft_yaml):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "league.yaml").write_text("league:\n  draft:\n" + draft_yaml, encoding="utf-8")
    return tmp_path


CFG = '    date: 2026-10-17\n    timezone: Pacific/Auckland\n    start_utc: "2026-10-16T18:00:00Z"\n'


def parse(tmp_path, argv=(), env=None, local=None):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if local is not None:
        (data / "daily_refresh.json").write_text(json.dumps(local))
    args = dr.build_parser().parse_args(["--data-dir", str(data), "--repo-root", str(tmp_path), *argv])
    return dr.resolve_settings(args, env or {}, date(2026, 10, 1))


# ------------------------------------------------------------------ time zone arithmetic

def test_start_is_nz_morning_and_cest_evening_and_dst_boundaries():
    nz = START.astimezone(NZ)
    assert (nz.date(), nz.hour, nz.minute, nz.utcoffset()) == (date(2026, 10, 17), 7, 0, timedelta(hours=13))     # NZDT
    cest = START.astimezone(CET)
    assert (cest.date(), cest.hour, cest.utcoffset()) == (date(2026, 10, 16), 20, timedelta(hours=2))             # CEST
    # NZ DST starts Sunday 2026-09-27 02:00 NZST (= 2026-09-26 14:00Z); CEST ends Sunday 2026-10-25, i.e. AFTER the draft
    assert datetime(2026, 9, 26, 13, 59, tzinfo=UTC).astimezone(NZ).utcoffset() == timedelta(hours=12)
    assert datetime(2026, 9, 26, 14, 0, tzinfo=UTC).astimezone(NZ).utcoffset() == timedelta(hours=13)
    assert datetime(2026, 10, 25, 0, 59, tzinfo=UTC).astimezone(CET).utcoffset() == timedelta(hours=2)
    assert datetime(2026, 10, 25, 1, 0, tzinfo=UTC).astimezone(CET).utcoffset() == timedelta(hours=1)
    # the window of plausible starts, 18:00-20:00Z, is 07:00-09:00 NZDT on the 17th
    assert [START.replace(hour=h).astimezone(NZ).strftime("%d %H:%M") for h in (18, 20)] == ["17 07:00", "17 09:00"]


def test_final_runs_default_hours_land_in_the_intended_local_slots():
    hours = sc.parse_hours(sc.DEFAULT_FINAL_HOURS)
    assert hours == [15.5, 1.75]
    nz = sc.final_run_times(START, hours, NZ)
    assert nz == [datetime(2026, 10, 16, 15, 30), datetime(2026, 10, 17, 5, 15)]         # NZDT wall clock
    assert sc.final_run_times(START, hours, CET) == [datetime(2026, 10, 16, 4, 30), datetime(2026, 10, 16, 18, 15)]   # CEST
    assert sc.final_run_times(START, hours, UTC) == [datetime(2026, 10, 16, 2, 30), datetime(2026, 10, 16, 16, 15)]
    # last run is 105 minutes before the earliest start, and the earliest start still leaves the last preseason night final
    assert START - datetime(2026, 10, 16, 16, 15, tzinfo=UTC) == timedelta(minutes=105)


def test_final_runs_in_the_past_are_dropped_and_none_disables():
    now = datetime(2026, 10, 17, 4, 0, tzinfo=NZ)
    assert sc.final_run_times(START, [15.5, 1.75], NZ, now) == [datetime(2026, 10, 17, 5, 15)]
    assert sc.parse_hours("none") == [] and sc.parse_hours("") == []
    with pytest.raises(sc.ScheduleError):
        sc.parse_hours("abc")
    with pytest.raises(sc.ScheduleError):
        sc.parse_hours("0")


def test_naive_astimezone_is_not_dst_safe_but_the_named_zone_is():
    # an install on 2026-09-26 (NZST, +12) must not use today's fixed offset for October
    fixed = timezone(timedelta(hours=12))
    wrong = sc.final_run_times(START, [1.75], fixed)[0]
    right = sc.final_run_times(START, [1.75], NZ)[0]
    assert wrong == datetime(2026, 10, 17, 4, 15) and right == datetime(2026, 10, 17, 5, 15)


# ------------------------------------------------------------------ settings precedence

def test_config_is_the_default_source_and_overrides_win(tmp_path):
    repo = repo_with_config(tmp_path, CFG)
    s = parse(repo)
    assert (s.draft_date, s.draft_date_source, s.draft_start, s.draft_tz) == (date(2026, 10, 17), "config", START, "Pacific/Auckland")
    f = parse(repo, local={"draft_date": "2026-10-18"})
    assert (f.draft_date, f.draft_date_source) == (date(2026, 10, 18), "file") and f.draft_start is None   # config start disagrees
    e = parse(repo, env={"NBA_DRAFT_DATE": "2026-10-19"}, local={"draft_date": "2026-10-18"})
    assert (e.draft_date, e.draft_date_source) == (date(2026, 10, 19), "env")
    g = parse(repo, argv=["--draft-date", "2026-10-20"], env={"NBA_DRAFT_DATE": "2026-10-19"})
    assert (g.draft_date, g.draft_date_source) == (date(2026, 10, 20), "flag")
    same = parse(repo, argv=["--draft-date", "2026-10-17"])
    assert same.draft_date_source == "flag" and same.draft_start == START                                   # consistent: time kept


def test_explicit_start_overrides_config_date_and_bad_values_fail(tmp_path):
    repo = repo_with_config(tmp_path, CFG)
    s = parse(repo, argv=["--draft-start", "2026-10-16T20:00:00Z"])
    assert s.draft_start == datetime(2026, 10, 16, 20, 0, tzinfo=UTC) and s.draft_date == date(2026, 10, 17)
    later = parse(repo, env={"NBA_DRAFT_START": "2026-10-18T02:00:00+00:00"})
    assert later.draft_date == date(2026, 10, 18) and later.draft_date_source == "env"                       # NZ 15:00 on the 18th
    with pytest.raises(dr.ConfigError):
        parse(repo, argv=["--draft-start", "2026-10-16T18:00:00"])          # no offset: ambiguous, refused
    with pytest.raises(dr.ConfigError):
        parse(repo, argv=["--draft-start", "soon"])


def test_missing_or_broken_config_falls_back(tmp_path):
    assert parse(tmp_path).draft_date_source == "fallback"
    (tmp_path / "b").mkdir()
    broken = repo_with_config(tmp_path / "b", "    date: [unclosed" + chr(10))
    assert parse(broken).draft_date_source == "fallback"


def test_real_repo_config_and_dev_data_json_are_consistent():
    from src.contracts import data_dir
    from src.ops.daily_refresh import REPO_ROOT

    assert dr.config_draft(REPO_ROOT)["date"] == date(2026, 10, 17)
    local = data_dir() / "daily_refresh.json"
    if local.exists():
        d = json.loads(local.read_text(encoding="utf-8"))
        assert "draft_date" not in d and "draft_start" not in d           # config is the single source


# ------------------------------------------------------------------ the run gate ends at the draft

def test_gate_stops_at_the_draft_start_but_not_before(tmp_path):
    s = parse(repo_with_config(tmp_path, CFG))
    before = datetime(2026, 10, 17, 5, 15, tzinfo=NZ)
    at = datetime(2026, 10, 17, 7, 0, tzinfo=NZ)
    assert dr.in_window(before.date(), s, before) and not dr.in_window(at.date(), s, at)
    assert dr.in_window(date(2026, 10, 17), s)                                # date-only callers keep the old behaviour
    assert dr.hours_to_draft(s, before) == pytest.approx(1.75)
    assert dr.hours_to_draft(s, datetime(2026, 9, 26, 7, 30, tzinfo=NZ)) == pytest.approx(
        (START - datetime(2026, 9, 25, 19, 30, tzinfo=UTC)).total_seconds() / 3600)          # 07:30 NZST = 19:30Z the day before


def test_main_skips_after_the_draft_starts_and_reports_days_and_hours(tmp_path):
    repo = repo_with_config(tmp_path, CFG)
    calls = []

    def runner(step, timeout):
        calls.append(step)
        return dr.StepOutcome("ok")
    argv = ["--data-dir", str(tmp_path / "d"), "--reports-dir", str(tmp_path / "r"), "--repo-root", str(repo), "--quiet", "--skip", "league"]
    assert dr.main(argv, runner=runner, now=datetime(2026, 10, 17, 7, 30, tzinfo=NZ), env={}) == 0 and calls == []
    assert dr.main(argv, runner=runner, now=datetime(2026, 10, 17, 5, 15, tzinfo=NZ), env={}) == 0 and calls
    latest = (tmp_path / "r" / "latest.md").read_text(encoding="utf-8")
    assert "DRAFT DAY" in latest and "Draft start (earliest plausible)" in latest and "1.8 h from the start" in latest


# ------------------------------------------------------------------ installed triggers

def install_dry(tmp_path, *extra, today=date(2026, 9, 26)):
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True, exist_ok=True)
    (repo / "config" / "league.yaml").write_text("league:\n  draft:\n" + CFG, encoding="utf-8")
    (repo / "src" / "ops").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "ops" / "daily_refresh.py").write_text("")
    lines = []
    rc = sc.main(["install", "--dry-run", "--repo", str(repo), "--data-dir", str(tmp_path / "d"), *extra], today=today,
                 python_exe=Path("py.exe"), out=lines.append, local_tz=NZ, now=datetime(2026, 9, 26, 12, 0, tzinfo=NZ))
    assert rc == 0
    xml = lines[0].split("\n", 2)[2]
    return ET.fromstring(xml.replace('encoding="UTF-16"', "")), lines[0]


def test_install_from_config_daily_triggers_end_at_the_draft_and_final_runs_are_added(tmp_path):
    root, text = install_dry(tmp_path)
    cal = root.findall("t:Triggers/t:CalendarTrigger", NS)
    once = root.findall("t:Triggers/t:TimeTrigger", NS)
    assert [t.findtext("t:StartBoundary", namespaces=NS) for t in cal] == ["2026-09-26T07:30:00", "2026-09-26T19:00:00"]
    assert [t.findtext("t:StartBoundary", namespaces=NS) for t in once] == ["2026-10-16T15:30:00", "2026-10-17T05:15:00"]
    ends = {t.findtext("t:EndBoundary", namespaces=NS) for t in (*cal, *once)}
    assert ends == {"2026-10-17T07:00:00"}                     # = 2026-10-16T18:00Z in NZDT: covers 05:15, excludes 07:30 on the 17th
    assert "2026-10-17T07:30:00" > "2026-10-17T07:00:00" and "2026-10-17T05:15:00" < "2026-10-17T07:00:00"
    args = root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=NS)
    assert "--draft-date" not in args and "--draft-start" not in args           # config stays the live source
    assert "final runs at 2026-10-16 15:30, 2026-10-17 05:15" in text


def test_install_flags_and_disabling_final_runs(tmp_path):
    root, _ = install_dry(tmp_path, "--final-runs-before-hours", "none")
    assert not root.findall("t:Triggers/t:TimeTrigger", NS)
    root, _ = install_dry(tmp_path, "--draft-start", "2026-10-16T20:00:00Z", "--final-runs-before-hours", "3")
    assert [t.findtext("t:StartBoundary", namespaces=NS) for t in root.findall("t:Triggers/t:TimeTrigger", NS)] == ["2026-10-17T06:00:00"]
    assert root.findall("t:Triggers/t:CalendarTrigger", NS)[0].findtext("t:EndBoundary", namespaces=NS) == "2026-10-17T09:00:00"
    assert "--draft-start 2026-10-16T20:00:00Z" in root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=NS)


def test_date_only_install_keeps_the_old_end_of_day_boundary(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ops").mkdir(parents=True)
    (repo / "src" / "ops" / "daily_refresh.py").write_text("")
    lines = []
    assert sc.main(["install", "--dry-run", "--repo", str(repo), "--data-dir", str(tmp_path), "--draft-date", "2026-10-17"],
                   today=date(2026, 9, 26), python_exe=Path("py.exe"), out=lines.append, local_tz=NZ) == 0
    assert "<EndBoundary>2026-10-18T23:59:59</EndBoundary>" in lines[0] and "TimeTrigger" not in lines[0]


def test_machine_zone_mismatch_is_refused():
    real = datetime.now().astimezone().utcoffset()
    other = "Asia/Kolkata" if real != timedelta(hours=5, minutes=30) else "Pacific/Auckland"
    with pytest.raises(sc.ScheduleError):
        sc.machine_tz(other)


def test_xml_datetime_end_and_end_before_start_error():
    xml = sc.build_task_xml(python_exe=Path("p"), repo=Path("r"), data_dir=Path("d"), times=["07:30"], start=date(2026, 9, 26),
                            end=datetime(2026, 10, 17, 7, 0), draft_date=None, final_runs=[datetime(2026, 10, 17, 5, 15)])
    assert xml.count("<EndBoundary>2026-10-17T07:00:00</EndBoundary>") == 2
    with pytest.raises(sc.ScheduleError):
        sc.build_task_xml(python_exe=Path("p"), repo=Path("r"), data_dir=Path("d"), times=["07:30"], start=date(2026, 9, 26),
                          end=datetime(2026, 9, 1, 7, 0), draft_date=None)
