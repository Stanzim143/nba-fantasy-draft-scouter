"""Generated Task Scheduler XML and commands; nothing is registered."""
import json
import xml.etree.ElementTree as ET
from datetime import date
from zoneinfo import ZoneInfo
from pathlib import Path

import pytest

from src.ops import schedule as sc

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def make_xml(**kw):
    args = dict(python_exe=Path(r"C:\v\Scripts\pythonw.exe"), repo=Path(r"C:\repo & co"), data_dir=Path(r"C:\data dir"),
                times=["07:30", "19:00"], start=date(2026, 9, 24), end=date(2026, 10, 19), draft_date=date(2026, 10, 17))
    args.update(kw)
    return sc.build_task_xml(**args)


def test_xml_is_valid_and_has_the_required_settings():
    root = ET.fromstring(make_xml().replace('encoding="UTF-16"', ""))
    s = root.find("t:Settings", NS)
    assert s.findtext("t:StartWhenAvailable", namespaces=NS) == "true"
    assert s.findtext("t:RunOnlyIfNetworkAvailable", namespaces=NS) == "true"
    assert s.findtext("t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"
    assert s.findtext("t:ExecutionTimeLimit", namespaces=NS) == "PT1H"
    p = root.find("t:Principals/t:Principal", NS)
    assert p.findtext("t:LogonType", namespaces=NS) == "InteractiveToken"
    assert p.findtext("t:RunLevel", namespaces=NS) == "LeastPrivilege"
    trig = root.findall("t:Triggers/t:CalendarTrigger", NS)
    assert [t.findtext("t:StartBoundary", namespaces=NS) for t in trig] == ["2026-09-24T07:30:00", "2026-09-24T19:00:00"]
    assert all(t.findtext("t:EndBoundary", namespaces=NS) == "2026-10-19T23:59:59" for t in trig)
    ex = root.find("t:Actions/t:Exec", NS)
    assert ex.findtext("t:Command", namespaces=NS).endswith("pythonw.exe")
    args = ex.findtext("t:Arguments", namespaces=NS)
    assert "-m src.ops.daily_refresh" in args and '--data-dir "C:\\data dir"' in args and "--draft-date 2026-10-17" in args
    assert ex.findtext("t:WorkingDirectory", namespaces=NS) == r"C:\repo & co"       # XML-escaped and round-trips


def test_no_draft_date_means_no_flag_and_bad_inputs_fail():
    assert "--draft-date" not in sc.build_arguments(Path("d"), None)
    with pytest.raises(sc.ScheduleError):
        sc.parse_times("25:00")
    with pytest.raises(sc.ScheduleError):
        make_xml(end=date(2026, 9, 1))
    assert sc.parse_times("7:30, 19:00") == ["07:30", "19:00"]


def test_install_dry_run_registers_nothing_and_install_calls_schtasks(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ops").mkdir(parents=True)
    (repo / "src" / "ops" / "daily_refresh.py").write_text("")
    calls, lines = [], []

    def runner(cmd):
        calls.append(cmd)
        return 0, ""
    base = ["install", "--repo", str(repo), "--data-dir", str(tmp_path / "d"), "--draft-date", "2026-10-17"]
    kw = dict(runner=runner, today=date(2026, 9, 24), python_exe=Path("py.exe"), out=lines.append,
              local_tz=ZoneInfo("Pacific/Auckland"))
    assert sc.main([*base, "--dry-run"], **kw) == 0
    assert calls == [] and "<Task" in lines[0]
    assert sc.main(base, **kw) == 0
    assert calls[0][:3] == ["schtasks", "/Create", "/TN"] and "/XML" in calls[0] and "/F" in calls[0]
    xml_file = tmp_path / "d" / "daily_refresh" / "daily_refresh_task.xml"
    assert xml_file.read_bytes()[:2] == b"\xff\xfe"                                   # UTF-16 LE BOM
    calls.clear()
    assert sc.main(["remove", "--repo", str(repo), "--data-dir", str(tmp_path)], **kw) == 0
    assert calls[0][:2] == ["schtasks", "/Delete"]
    assert sc.main(["run-now", "--repo", str(repo), "--data-dir", str(tmp_path)], **kw) == 0
    assert calls[1][:2] == ["schtasks", "/Run"]


def test_install_refuses_a_checkout_without_the_feature(tmp_path):
    rc = sc.main(["install", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], runner=lambda c: (0, ""), out=print,
                 python_exe=Path("p.exe"))
    assert rc == 2


def test_status_and_health(tmp_path):
    ops = tmp_path / "daily_refresh"
    ops.mkdir()
    (ops / "status.json").write_text(json.dumps({"outcome": "partial", "run_id": "R", "consecutive_failures": 2,
                                                  "steps": {"adp": {"status": "failed"}, "roster": {"status": "ok"}},
                                                  "report": "x"}))
    lines = sc.describe_health(tmp_path, tmp_path)
    assert "partial" in lines[0] and any("adp=failed" in line for line in lines)
    out = []
    rc = sc.main(["status", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], runner=lambda c: (1, "not found"), out=out.append)
    assert rc == 1 and any("not registered" in line for line in out)


def test_status_without_job_shows_both_jobs_and_job_all_is_status_only(tmp_path):
    out = []
    rc = sc.main(["status", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], runner=lambda c: (1, "nf"), out=out.append)
    text = " ".join(out)
    assert rc == 1 and sc.TASK_NAME in text and sc.NIGHTLY_TASK_NAME in text
    assert "--job daily" in text and "--job nightly" in text
    out.clear()
    only = sc.main(["status", "--job", "nightly", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], runner=lambda c: (1, "nf"), out=out.append)
    assert only == 1 and sc.TASK_NAME not in " ".join(out)
    assert sc.main(["remove", "--job", "all", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], runner=lambda c: (0, "")) == 2


def test_health_flags_a_missing_report_file_and_accepts_an_existing_one(tmp_path):
    ops = tmp_path / "nightly"
    ops.mkdir()
    rel = "reports/nightly/latest.md"        # (the real status file records it with backslashes on Windows)
    (ops / "status.json").write_text(json.dumps({"outcome": "ok", "run_id": "N1", "consecutive_failures": 0, "steps": {}, "report": rel}))
    reports = tmp_path / "reports" / "nightly"
    lines = sc.describe_health(tmp_path, reports, ops="nightly")
    assert any("REPORT FILE MISSING" in line for line in lines)
    reports.mkdir(parents=True)
    (reports / "latest.md").write_text("x")
    lines = sc.describe_health(tmp_path, reports, ops="nightly")
    assert not any("MISSING" in line for line in lines) and any(line.startswith("report: ") for line in lines)


# --------------------------------------------------------------------------- the nightly in-season task (ADR 0018)

def nightly_xml(**kw):
    args = dict(python_exe=Path(r"C:\v\Scripts\pythonw.exe"), repo=Path(r"C:\repo"), data_dir=Path(r"C:\data"), times=["09:30"],
                start=date(2026, 10, 20), end=date(2027, 4, 6), draft_date=None, job="nightly")
    args.update(kw)
    return sc.build_task_xml(**args)


def test_nightly_xml_has_the_required_settings_window_and_action():
    root = ET.fromstring(nightly_xml().replace('encoding="UTF-16"', ""))
    s = root.find("t:Settings", NS)
    assert s.findtext("t:StartWhenAvailable", namespaces=NS) == "true" and s.findtext("t:RunOnlyIfNetworkAvailable", namespaces=NS) == "true"
    assert s.findtext("t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew" and s.findtext("t:ExecutionTimeLimit", namespaces=NS) == "PT90M"
    p = root.find("t:Principals/t:Principal", NS)
    assert p.findtext("t:LogonType", namespaces=NS) == "InteractiveToken" and p.findtext("t:RunLevel", namespaces=NS) == "LeastPrivilege"
    (trig,) = root.findall("t:Triggers/t:CalendarTrigger", NS)
    assert trig.findtext("t:StartBoundary", namespaces=NS) == "2026-10-20T09:30:00" and trig.findtext("t:EndBoundary", namespaces=NS) == "2027-04-06T23:59:59"
    ex = root.find("t:Actions/t:Exec", NS)
    args = ex.findtext("t:Arguments", namespaces=NS)
    assert "-m src.ops.nightly" in args and "daily_refresh" not in args and "--draft-date" not in args
    assert root.findtext("t:RegistrationInfo/t:Description", namespaces=NS).startswith("In-season nightly job")


def test_nightly_install_dry_run_and_real_call_leave_the_daily_task_alone(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src" / "ops").mkdir(parents=True)
    (repo / "src" / "ops" / "nightly.py").write_text("")
    calls, lines = [], []
    kw = dict(runner=lambda c: (calls.append(c), (0, ""))[1], today=date(2026, 9, 25), python_exe=Path("py.exe"), out=lines.append)
    base = ["install", "--job", "nightly", "--repo", str(repo), "--data-dir", str(tmp_path / "d"), "--start-date", "2026-10-20", "--end-date", "2027-04-06"]
    assert sc.main([*base, "--dry-run"], **kw) == 0 and calls == [] and "Nightly In-Season" in lines[0] and "09:30" in lines[0]
    assert sc.main([*base, "--times", "10:15"], **kw) == 0
    assert calls[0][:4] == ["schtasks", "/Create", "/TN", sc.NIGHTLY_TASK_NAME] and "/F" in calls[0]
    assert (tmp_path / "d" / "nightly" / "nightly_task.xml").read_bytes()[:2] == b"\xff\xfe"
    assert not (tmp_path / "d" / "daily_refresh").exists()                        # the pre-draft task's files are untouched
    calls.clear()
    assert sc.main(["remove", "--job", "nightly", "--repo", str(repo), "--data-dir", str(tmp_path)], **kw) == 0 and calls[0][3] == sc.NIGHTLY_TASK_NAME
    assert sc.main(["remove", "--repo", str(repo), "--data-dir", str(tmp_path)], **kw) == 0 and calls[1][3] == sc.TASK_NAME     # default job unchanged
    assert sc.TASK_NAME == "NBA Fantasy 2026 Daily Refresh" and sc.DEFAULT_TIMES == "07:30,19:00"
    assert sc.main(["install", "--job", "nightly", "--repo", str(tmp_path), "--data-dir", str(tmp_path)], **kw) == 2         # module missing


def test_nightly_window_defaults_to_the_season_window_and_status_reads_its_own_files(tmp_path):
    start, end = sc.nightly_window(tmp_path, "2026-27", date(2026, 9, 25), None, None)
    assert start == date(2026, 10, 20) and end == date(2026, 10, 20) + __import__("datetime").timedelta(days=166 + 2)
    assert sc.nightly_window(tmp_path, "2026-27", date(2026, 9, 25), "2026-10-22", "2027-01-01") == (date(2026, 10, 22), date(2027, 1, 1))
    ops = tmp_path / "nightly"
    ops.mkdir()
    (ops / "status.json").write_text(json.dumps({"outcome": "ok", "run_id": "N1", "consecutive_failures": 0, "steps": {}, "report": "r"}))
    assert "N1" in sc.describe_health(tmp_path, tmp_path, ops="nightly")[0]
