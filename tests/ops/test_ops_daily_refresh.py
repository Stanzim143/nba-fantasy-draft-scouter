"""Daily refresh orchestration with a fake step runner: gate, lock, isolation, budget, diffs, report, status, alert."""
import json
import sys
from datetime import datetime, timezone

import pytest

from src.ops import daily_refresh as dr
from src.ops import refresh_diff as rd
from src.ops.runlock import RunLock

NOW = datetime(2026, 10, 3, 8, 0, tzinfo=timezone.utc)


def raw(*rows):
    return [{"id": i, "fullName": n, "proTeamId": 1, "injuryStatus": st, "injured": False,
             "ownership": {"averageDraftPosition": adp, "percentOwned": 50.0}} for i, n, st, adp in rows]


class Fake:
    """Step runner: returns canned outcomes; ``snapshots`` feeds the snapshot step from real archive files."""

    def __init__(self, tmp, players, wl, roster, fail=(), preseason=0):
        self.tmp, self.players, self.wl, self.roster, self.fail, self.pre = tmp, players, wl, roster, set(fail), preseason
        self.calls = []
        self.taken = datetime(2026, 10, 3, 7, 59, tzinfo=timezone.utc)

    def __call__(self, step, timeout):
        self.calls.append((step, timeout))
        if step in self.fail:
            return dr.StepOutcome("failed", error=f"{step} exploded")
        if step == "roster":
            return dr.StepOutcome("ok", {"rostered": len(self.roster)}, {"roster_map": self.roster})
        if step == "offseason":
            s = {"summer_league": {"games": 76, "last_date": "2026-07-19"}, "preseason": {"games": self.pre, "last_date": None}}
            return dr.StepOutcome("ok", s, {"offseason": s})
        if step == "snapshot":
            taken = self.taken
            path, _ = rd.write_snapshot(self.tmp, "2026-27", self.players, taken)
            return dr.StepOutcome("ok", {}, {"snapshot": str(path), "snapshot_taken_at": taken.isoformat()})
        if step == "profiles":
            s = {"profiles": 500, "with_birthdate": 490, "combine_rows": 300, "combine_failed_years": []}
            return dr.StepOutcome("ok", s, {"profiles": s})
        if step == "status":
            by = {"ACTIVE": 400 + self.pre, "OUT": 5}
            return dr.StepOutcome("ok", {"players": 405, "by_status": by}, {"status_archive": {"players": 405, "by_status": by}})
        if step == "watchlist":
            return dr.StepOutcome("ok", {"model": "m"}, {"watchlist": self.wl, "model": "m"})
        if step == "board":
            b = {"baseline": {"path": "p.csv", "rows": 3, "top": [{"rank": 1, "name": "A"}]}}
            return dr.StepOutcome("ok", {"boards": b}, {})
        return dr.StepOutcome("ok")


def settings(tmp_path, **kw):
    args = dr.build_parser().parse_args(["--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "rep"),
                                         "--draft-date", "2026-10-17", "--league-id", "1", *kw.pop("argv", [])])
    s = dr.resolve_settings(args, {}, NOW.date())
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_window_gate_and_setting_precedence(tmp_path, monkeypatch):
    s = settings(tmp_path)
    assert (s.window_start, s.window_end) == (datetime(2026, 9, 17).date(), datetime(2026, 10, 18).date())
    assert dr.in_window(NOW.date(), s) and not dr.in_window(datetime(2026, 10, 19).date(), s)
    # local file < env < flag; without any, the fallback is used and flagged as such
    data = tmp_path / "data"
    data.mkdir()
    (data / "daily_refresh.json").write_text(json.dumps({"draft_date": "2026-10-17", "league_id": 7}))
    ns = dr.build_parser().parse_args(["--data-dir", str(data)])
    got = dr.resolve_settings(ns, {}, NOW.date())
    assert (str(got.draft_date), got.draft_date_source, got.league_id) == ("2026-10-17", "file", 7)
    assert dr.resolve_settings(ns, {"NBA_DRAFT_DATE": "2026-10-18"}, NOW.date()).draft_date_source == "env"
    empty = dr.resolve_settings(dr.build_parser().parse_args(["--data-dir", str(tmp_path / "none"), "--repo-root", str(tmp_path)]), {}, NOW.date())
    assert empty.draft_date_source == "fallback" and str(empty.draft_date) == "2026-10-18"
    with pytest.raises(dr.ConfigError):
        dr.resolve_settings(dr.build_parser().parse_args(["--draft-date", "soon"]), {}, NOW.date())


def test_outside_window_does_nothing_and_exits_zero(tmp_path):
    fake = Fake(tmp_path, raw((1, "S", "ACTIVE", 5.0)), [], {})
    late = datetime(2026, 12, 1, 8, 0, tzinfo=timezone.utc)
    rc = dr.main(["--data-dir", str(tmp_path / "d"), "--reports-dir", str(tmp_path / "r"), "--draft-date", "2026-10-17", "--quiet"],
                 runner=fake, now=late, env={})
    assert rc == 0 and fake.calls == [] and not (tmp_path / "r").exists()
    assert dr.main(["--data-dir", str(tmp_path / "d"), "--reports-dir", str(tmp_path / "r"), "--draft-date", "2026-10-17", "--quiet", "--force",
                    "--skip", "league"], runner=fake, now=late, env={}) == 0 and fake.calls


def test_full_run_then_second_run_diffs_and_alerts_clear(tmp_path):
    s = settings(tmp_path)
    p1 = raw((1, "Star", "ACTIVE", 5.0), (2, "Rook", "ACTIVE", 60.0))
    wl1 = [{"player_id": 10, "rank": 1, "name": "W1"}, {"player_id": 11, "rank": 2, "name": "W2"}]
    f1 = Fake(tmp_path / "data", p1, wl1, {"1": ["Star", "BOS"]})
    assert dr.execute(s, f1, now=NOW, notifier=lambda *a: None) == 0
    assert [c[0] for c in f1.calls] == list(dr.STEP_ORDER)
    latest = (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    assert "OK" in latest and "First archived run" in latest
    status = json.loads((tmp_path / "data" / "daily_refresh" / "status.json").read_text())
    assert status["outcome"] == "ok" and status["exit_code"] == 0 and status["consecutive_failures"] == 0
    assert not (tmp_path / "rep" / "ALERT.txt").exists() and not (tmp_path / "data" / "daily_refresh" / "lock.json").exists()

    later = datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc)
    p2 = raw((1, "Star", "OUT", 5.0), (2, "Rook", "ACTIVE", 52.0))
    wl2 = [{"player_id": 10, "rank": 1, "name": "W1"}, {"player_id": 12, "rank": 2, "name": "W3"}]
    f2 = Fake(tmp_path / "data", p2, wl2, {"1": ["Star", "MIA"]}, preseason=4)
    f2.taken = datetime(2026, 10, 3, 19, 59, tzinfo=timezone.utc)
    assert dr.execute(s, f2, now=later, notifier=lambda *a: None) == 0
    md = (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    assert "Star: ACTIVE -> OUT" in md and "riser Rook" in md and "Star BOS -> MIA" in md
    assert "NEW on the list: #2 W3" in md and "DROPPED off the list: W2" in md and "+4 new" in md
    assert (tmp_path / "rep" / f"{later:%Y-%m-%d}.md").exists() and len(list((tmp_path / "rep" / "runs").glob("*.md"))) == 2


def test_step_failure_is_isolated_alerts_after_two_runs_and_clears(tmp_path):
    s = settings(tmp_path)
    toasts = []
    players = raw((1, "Star", "ACTIVE", 5.0))
    bad = Fake(tmp_path / "data", players, [], {}, fail=("adp", "league"))
    assert dr.execute(s, bad, now=NOW, notifier=lambda t, m: toasts.append(m)) == 1
    assert [c[0] for c in bad.calls] == list(dr.STEP_ORDER)                       # nothing stopped the others
    assert not (tmp_path / "rep" / "ALERT.txt").exists()                          # one failing run: no alert yet
    st = json.loads((tmp_path / "data" / "daily_refresh" / "status.json").read_text())
    assert st["outcome"] == "partial" and st["steps"]["adp"]["status"] == "failed" and st["last_success_at"] is None
    assert dr.execute(s, bad, now=datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc), notifier=lambda t, m: toasts.append(m)) == 1
    alert = (tmp_path / "rep" / "ALERT.txt").read_text()
    assert "2 consecutive runs" in alert and len(toasts) == 1
    assert "adp exploded" in (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    good = Fake(tmp_path / "data", raw((1, "Star", "ACTIVE", 5.0)), [], {})
    assert dr.execute(s, good, now=datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc), notifier=lambda *a: None) == 0
    assert not (tmp_path / "rep" / "ALERT.txt").exists()


def test_stale_espn_data_raises_an_alert_even_without_failures(tmp_path):
    s = settings(tmp_path)

    class OldSnapshot(Fake):
        def __call__(self, step, timeout):
            if step == "snapshot":
                return dr.StepOutcome("ok", {}, {"snapshot": "x", "snapshot_taken_at": "2026-09-30T00:00:00+00:00"})
            return super().__call__(step, timeout)
    assert dr.execute(s, OldSnapshot(tmp_path / "data", [], [], {}), now=NOW, notifier=lambda *a: None) == 0
    assert "ESPN data is" in (tmp_path / "rep" / "ALERT.txt").read_text()


def test_league_skipped_cleanly_without_id_and_budget_exhaustion_is_a_failure(tmp_path):
    s = settings(tmp_path, league_id=None)
    f = Fake(tmp_path / "data", raw((1, "S", "ACTIVE", 5.0)), [], {})
    assert dr.execute(s, f, now=NOW, notifier=lambda *a: None) == 0
    assert "league" not in [c[0] for c in f.calls]
    st = json.loads((tmp_path / "data" / "daily_refresh" / "status.json").read_text())
    assert st["steps"]["league"]["status"] == "skipped"
    ticks = iter(range(0, 100000, 4000))
    s2 = settings(tmp_path, budget_s=5000.0)
    f2 = Fake(tmp_path / "data", raw((1, "S", "ACTIVE", 5.0)), [], {})
    assert dr.execute(s2, f2, now=NOW, notifier=lambda *a: None, clock=lambda: next(ticks)) == 1
    st = json.loads((tmp_path / "data" / "daily_refresh" / "status.json").read_text())
    assert any(v["status"] == "skipped" and "budget" in (v["error"] or "") for v in st["steps"].values())


def test_busy_lock_exits_three_and_crashing_step_is_isolated(tmp_path):
    s = settings(tmp_path)
    held = RunLock(s.ops_dir / "lock.json").acquire()
    f = Fake(tmp_path / "data", [], [], {})
    assert dr.execute(s, f, now=NOW, notifier=lambda *a: None) == dr.EXIT_BUSY and f.calls == []
    held.release()

    def crashing(step, timeout):
        if step == "roster":
            raise RuntimeError("kaboom")
        return dr.StepOutcome("ok")
    assert dr.execute(s, crashing, now=NOW, notifier=lambda *a: None) == 1
    assert "kaboom" in (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")


def test_dry_run_writes_nothing(tmp_path, capsys):
    rc = dr.main(["--dry-run", "--data-dir", str(tmp_path / "d"), "--reports-dir", str(tmp_path / "r"), "--draft-date", "2026-10-17"],
                 now=NOW, env={})
    assert rc == 0 and "INSIDE" in capsys.readouterr().out and not (tmp_path / "d").exists()


def test_process_runner_timeout_kills_and_reports(tmp_path):
    s = settings(tmp_path)
    code, out = dr.run_worker_process([sys.executable, "-c", "import time; print('hi', flush=True); time.sleep(30)"], 2,
                                      cwd=tmp_path, env={})
    assert code is None
    runner = dr.SubprocessRunner(s, process_runner=lambda cmd, t, cwd, env: (None, ""))
    assert runner("roster", 1).status == "timeout"
    runner = dr.SubprocessRunner(s, process_runner=lambda cmd, t, cwd, env: (1, "Traceback: nope"))
    o = runner("roster", 1)
    assert o.status == "failed" and "without a result" in o.error


def test_worker_result_file_roundtrip(tmp_path, monkeypatch):
    def fake_worker(a):
        return {"x": float("nan"), "n": 1}, {"k": [1]}
    monkeypatch.setitem(dr.WORKERS, "roster", fake_worker)
    res = tmp_path / "res.json"
    rc = dr.main(["--worker", "roster", "--result-file", str(res), "--data-dir", str(tmp_path), "--season", "2026-27"])
    assert rc == 0 and json.loads(res.read_text()) == {"ok": True, "summary": {"x": None, "n": 1}, "state": {"k": [1]}}

    def bad_worker(a):
        raise ValueError("bad")
    monkeypatch.setitem(dr.WORKERS, "roster", bad_worker)
    assert dr.main(["--worker", "roster", "--result-file", str(res), "--data-dir", str(tmp_path)]) == 1
    assert "ValueError: bad" in json.loads(res.read_text())["error"]


def test_extras_steps_order_report_and_isolation(tmp_path):
    assert dr.STEP_ORDER.index("adp") < dr.STEP_ORDER.index("profiles") < dr.STEP_ORDER.index("status") < dr.STEP_ORDER.index("snapshot")
    assert set(dr.STEP_ORDER) <= set(dr.DEFAULT_TIMEOUTS) and set(dr.STEP_ORDER) == set(dr.STEP_ORDER) & set(dr.WORKERS) | {"snapshot"}
    with pytest.raises(dr.ConfigError):
        dr.resolve_settings(dr.build_parser().parse_args(["--only", "profiles,bogus"]), {}, NOW.date())
    s = settings(tmp_path)
    f1 = Fake(tmp_path / "data", raw((1, "S", "ACTIVE", 5.0)), [], {})
    assert dr.execute(s, f1, now=NOW, notifier=lambda *a: None) == 0
    md = (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    assert "500 players (490 with a birthdate), 300 draft-combine rows" in md and "status archive: 405 players" in md
    f2 = Fake(tmp_path / "data", raw((1, "S", "ACTIVE", 5.0)), [], {}, preseason=3)
    f2.taken = datetime(2026, 10, 3, 19, 59, tzinfo=timezone.utc)
    assert dr.execute(s, f2, now=datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc), notifier=lambda *a: None) == 0
    assert "ACTIVE 400->403" in (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    # a failing extra is isolated: everything else still runs, the run is partial, the report names it
    bad = Fake(tmp_path / "data", raw((1, "S", "ACTIVE", 5.0)), [], {}, fail=("profiles", "status"))
    assert dr.execute(s, bad, now=datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc), notifier=lambda *a: None) == 1
    assert [c[0] for c in bad.calls] == list(dr.STEP_ORDER)
    st = json.loads((tmp_path / "data" / "daily_refresh" / "status.json").read_text())
    assert st["steps"]["status"]["status"] == "failed" and st["steps"]["board"]["status"] == "ok"


def test_extras_workers_delegate_to_refresh_steps(tmp_path, monkeypatch):
    seen = []

    def fake_refresh(name, season, base, offline):
        seen.append(name)
        return {"profiles": 1, "with_birthdate": 1, "combine_rows": 0, "combine_failed_years": []} if name == "profiles" else {"players": 2, "by_status": {"OUT": 1}}
    monkeypatch.setattr(dr, "_refresh_step", fake_refresh)
    a = dr.build_parser().parse_args(["--data-dir", str(tmp_path), "--season", "2026-27"])
    a.data_dir = tmp_path
    assert dr.worker_profiles(a)[1]["profiles"]["profiles"] == 1
    assert dr.worker_status(a)[1] == {"status_archive": {"players": 2, "by_status": {"OUT": 1}}}
    assert seen == ["profiles", "status"] and {"profiles", "status"} <= set(dr.WORKERS)


def test_default_boards_include_the_debut_board_and_report_says_so():
    from src.ops.daily_report import render_report

    assert dr.Settings.__dataclass_fields__["board_models"].default == ("baseline", "auto", "baseline_offseason_debut", "baseline_hurdle_adp_offseason_debut")
    run = {"date": "2026-10-01", "outcome": "ok", "run_id": "r", "season": "2026-27",
           "boards": {"baseline_offseason_debut": {"path": "x.csv", "rows": 796, "top": [{"rank": 1, "name": "A"}]},
                      "baseline": {"path": "y.csv", "rows": 739, "top": [{"rank": 1, "name": "A"}]}}}
    lines = [ln for ln in render_report(run).splitlines() if ln.startswith("- `")]
    assert "debutants and undrafted signees" in next(ln for ln in lines if "baseline_offseason_debut" in ln)
    assert "debutants" not in next(ln for ln in lines if "y.csv" in ln)


def test_history_reader_skips_torn_lines_and_append_repairs_missing_newline(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text('{"run_id": "a", "ok": true}\n{"run_id": "b", "ok": fal', encoding="utf-8")  # torn last line
    assert [r["run_id"] for r in dr.read_history(p)] == ["a"]
    dr.append_history(p, {"run_id": "c", "ok": False})
    assert [r["run_id"] for r in dr.read_history(p)] == ["a", "c"]
    p.write_text('{"run_id": "a"}\ngarbage\n[1]\n{"run_id": "z"}\n', encoding="utf-8")
    assert [r["run_id"] for r in dr.read_history(p)] == ["a", "z"]
