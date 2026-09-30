"""preseason_refresh: step order and isolation, the ADP step's live-season-only refresh, the summary and the CLI."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingest import preseason_refresh as pr
from src.store import write_table

SEASON = "2026-27"


class Client:
    offline = False


def clients():
    return dict(nba=Client(), espn=Client(), fp=Client())


def fake_steps(monkeypatch, *, fail=(), calls=None):
    calls = calls if calls is not None else []

    def make(name, summary):
        def step(*a, **k):
            calls.append(name)
            if name in fail:
                raise pr.nba_incoming.IncomingError(f"{name} broke")
            return summary
        return step

    monkeypatch.setattr(pr, "step_roster", make("roster", {
        "rostered": 584, "rookies_added": 52, "moves_since_previous_snapshot": {"moved": 2},
        "debuts_not_projected": ["Thomas Sorber"], "missing_birthdates": 0}))
    monkeypatch.setattr(pr, "step_offseason", make("offseason", {
        "summer_league": {"games": 76, "players": 447, "last_date": "2026-07-19"},
        "preseason": {"games": 0, "players": 0, "last_date": None}}))
    monkeypatch.setattr(pr, "step_adp", make("adp", {"adp_rows_for_season": 212, "unmapped_names": ["Wang Zhelin"],
                                                     "id_matching": "1367/1590 matched"}))
    return calls


def test_steps_always_run_in_dependency_order(monkeypatch, tmp_path):
    calls = fake_steps(monkeypatch)
    results = pr.run_refresh(SEASON, tmp_path, steps=("adp", "roster", "offseason"), log=lambda *_: None, **clients())
    assert calls == ["roster", "offseason", "adp"] and [r.name for r in results] == calls and all(r.ok for r in results)


def test_only_selected_steps_run(monkeypatch, tmp_path):
    calls = fake_steps(monkeypatch)
    pr.run_refresh(SEASON, tmp_path, steps=("offseason",), log=lambda *_: None, **clients())
    assert calls == ["offseason"]


def test_a_failing_step_never_stops_the_others(monkeypatch, tmp_path):
    calls = fake_steps(monkeypatch, fail=("offseason",))
    results = pr.run_refresh(SEASON, tmp_path, log=lambda *_: None, **clients())
    assert calls == ["roster", "offseason", "adp"]
    by = {r.name: r for r in results}
    assert by["roster"].ok and by["adp"].ok and not by["offseason"].ok and "offseason broke" in by["offseason"].error


def test_unexpected_exceptions_are_not_swallowed(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise KeyError("a genuine bug")

    monkeypatch.setattr(pr, "step_roster", boom)
    with pytest.raises(KeyError):
        pr.run_refresh(SEASON, tmp_path, steps=("roster",), log=lambda *_: None, **clients())


def test_unknown_steps_and_seasons_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown steps"):
        pr.run_refresh(SEASON, tmp_path, steps=("nope",), **clients())
    with pytest.raises(ValueError):
        pr.run_refresh("2026", tmp_path, **clients())


def test_summary_reports_what_matters_before_a_draft(monkeypatch, tmp_path):
    fake_steps(monkeypatch)
    text = pr.render_summary(SEASON, pr.run_refresh(SEASON, tmp_path, log=lambda *_: None, **clients()))
    assert "584 players on rosters; +52 rookies" in text and "{'moved': 2}" in text
    assert "Thomas Sorber" in text and "NOT projected" in text
    assert "summer_league: 76 games" in text and "no preseason games yet" in text
    assert "Wang Zhelin" in text and "Next: python -m src.value.breakouts" in text


def test_summary_of_a_failed_run_says_so(monkeypatch, tmp_path):
    fake_steps(monkeypatch, fail=("adp",))
    text = pr.render_summary(SEASON, pr.run_refresh(SEASON, tmp_path, log=lambda *_: None, **clients()))
    assert "[FAILED] adp" in text and "Some steps failed" in text and "Next:" not in text


def test_cli_exit_codes(monkeypatch, tmp_path, capsys):
    fake_steps(monkeypatch)
    assert pr.main(["--season", SEASON, "--data-dir", str(tmp_path)], **clients()) == 0
    assert "Preseason refresh for 2026-27" in capsys.readouterr().out
    fake_steps(monkeypatch, fail=("roster",))
    assert pr.main(["--season", SEASON, "--data-dir", str(tmp_path)], **clients()) == 1
    assert pr.main(["--season", SEASON, "--only", "bogus", "--data-dir", str(tmp_path)], **clients()) == 2
    assert "unknown steps" in capsys.readouterr().err


# --------------------------------------------------------------------------- real step functions, stubbed at the ingest boundary

def test_roster_and_offseason_steps_summarise_their_ingests(monkeypatch, tmp_path):
    monkeypatch.setattr(pr.nba_incoming, "run_ingest", lambda *a, **k: SimpleNamespace(
        rostered=584, added_players=52, moves_since_previous={}, earlier_draft_debut_names=["A B"], missing_birthdates=1))
    s = pr.step_roster(SEASON, tmp_path, Client(), lambda *_: None)
    assert s == {"rostered": 584, "rookies_added": 52, "moves_since_previous_snapshot": {},
                 "debuts_not_projected": ["A B"], "missing_birthdates": 1}
    seen = {}

    def fake_offseason(seasons, client, base, **kw):
        seen.update(seasons=seasons, kw=kw)
        ev = {"n_games": 3, "n_players": 40, "last_date": "2026-10-05"}
        return SimpleNamespace(events={f"{SEASON}/summer_league": ev, f"{SEASON}/preseason": ev})

    monkeypatch.setattr(pr.nba_offseason, "run_ingest", fake_offseason)
    s = pr.step_offseason(SEASON, tmp_path, Client(), lambda *_: None)
    assert seen["seasons"] == [SEASON] and seen["kw"]["live_start"] == 2026      # the live season is always re-downloaded
    assert s["preseason"] == {"games": 3, "players": 40, "last_date": "2026-10-05"}


def test_the_adp_step_refreshes_only_the_live_season_and_rebuilds_from_the_cache(monkeypatch, tmp_path):
    adp = pd.DataFrame({"season": [SEASON, SEASON, "2025-26"], "source": "espn", "source_id": ["1", "2", "3"],
                        "adp": [5.0, 88.0, 9.0], "name": ["Mapped One", "Zed Unmapped", "Old"], "adp_source": "espn"})
    (tmp_path / "processed").mkdir(parents=True)
    adp.to_parquet(tmp_path / "processed" / "adp.parquet")
    id_map = pd.DataFrame({"player_id": [11], "source": "espn", "source_id": ["1"], "source_name": "Mapped One",
                           "match_method": "exact", "confidence": 1.0})
    write_table(id_map, "player_id_map", tmp_path)
    fetched, ran = {}, {}
    monkeypatch.setattr(pr.espn_adp, "fetch_espn_players",
                        lambda client, sid, **kw: fetched.update(sid=sid, kw=kw) or [])
    monkeypatch.setattr(pr.espn_adp, "run_ingest", lambda seasons, client, base, **kw: ran.update(seasons=seasons, kw=kw) or
                        SimpleNamespace(match_report=SimpleNamespace(summary=lambda: "1/2 matched")))
    s = pr.step_adp(SEASON, tmp_path, Client(), Client(), lambda *_: None)
    assert fetched["sid"] == 2027 and fetched["kw"] == {"refresh": True}                   # only the live season is re-pulled
    assert ran["seasons"][0] == "2015-16" and ran["seasons"][-1] == SEASON and "refresh" not in ran["kw"]
    assert s["adp_rows_for_season"] == 2 and s["unmapped_names"] == ["Zed Unmapped"] and s["id_matching"] == "1/2 matched"


def test_the_adp_step_does_not_refresh_when_offline(monkeypatch, tmp_path):
    (tmp_path / "processed").mkdir(parents=True)
    pd.DataFrame({"season": [SEASON], "source": "espn", "source_id": ["1"], "adp": [5.0], "name": ["A"],
                  "adp_source": "espn"}).to_parquet(tmp_path / "processed" / "adp.parquet")
    write_table(pd.DataFrame({"player_id": [11], "source": "espn", "source_id": ["1"], "source_name": "A",
                              "match_method": "exact", "confidence": 1.0}), "player_id_map", tmp_path)
    kw = {}
    monkeypatch.setattr(pr.espn_adp, "fetch_espn_players", lambda client, sid, **k: kw.update(k) or [])
    monkeypatch.setattr(pr.espn_adp, "run_ingest", lambda *a, **k: SimpleNamespace(match_report=None))
    offline = Client()
    offline.offline = True
    pr.step_adp(SEASON, tmp_path, offline, Client(), lambda *_: None)
    assert kw == {"refresh": False}


# --------------------------------------------------------------------------- optional extra steps (ADR 0016)

def test_extra_steps_are_opt_in_and_run_after_the_default_three(monkeypatch, tmp_path):
    calls = fake_steps(monkeypatch)
    monkeypatch.setattr(pr, "step_profiles", lambda *a, **k: calls.append("profiles") or {
        "profiles": 5, "with_birthdate": 4, "combine_rows": 9, "combine_failed_years": []})
    monkeypatch.setattr(pr, "step_status", lambda *a, **k: calls.append("status") or {"players": 600, "by_status": {"OUT": 12}})
    pr.run_refresh(SEASON, tmp_path, log=lambda *_: None, **clients())
    assert calls == ["roster", "offseason", "adp"]                                  # the default is unchanged
    calls.clear()
    results = pr.run_refresh(SEASON, tmp_path, steps=("status", "profiles", "adp", "roster"), log=lambda *_: None, **clients())
    assert calls == ["roster", "adp", "profiles", "status"] and all(r.ok for r in results)
    text = pr.render_summary(SEASON, results)
    assert "5 player profiles" in text and "ESPN injury status archived for 600" in text
    assert pr.STEPS == ("roster", "offseason", "adp")


def test_with_extras_flag_adds_the_extra_steps(monkeypatch, tmp_path):
    calls = fake_steps(monkeypatch)
    monkeypatch.setattr(pr, "step_profiles", lambda *a, **k: calls.append("profiles") or {
        "profiles": 1, "with_birthdate": 1, "combine_rows": 1, "combine_failed_years": []})
    monkeypatch.setattr(pr, "step_status", lambda *a, **k: calls.append("status") or {"players": 1, "by_status": {}})
    monkeypatch.setattr(pr, "step_transactions", lambda *a, **k: calls.append("transactions") or {
        "feed_rows": 5, "records": 7, "new_rows": 2, "ledger_rows": 7, "by_kind": {"signed": 7}, "parse": {}, "matched": "5/6 matched", "unresolved": 1})
    monkeypatch.setattr(pr, "step_coaches", lambda *a, **k: calls.append("coaches") or {"teams": 30, "coaches": {}})
    assert pr.main(["--season", SEASON, "--with-extras", "--data-dir", str(tmp_path)], **clients()) == 0
    assert calls == ["roster", "offseason", "adp", "profiles", "status", "transactions", "coaches"]
