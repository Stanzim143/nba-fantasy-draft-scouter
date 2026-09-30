"""schedule_games: ESPN parsing, table IO, weekly team table (b2b, off nights, heavy/light, games left)."""
import json
from datetime import date

import pandas as pd
import pytest
from ise_testkit import GAMES, N_TEAMS, SEASON, make_tables

from src.contracts import ContractError
from src.inseason import schedule as S
from src.inseason.weeks import derive


def _game(gid, sp, home, away, tbd=False):
    # 2026-10-20 is scoring period 1; an 8 pm ET tip is the next UTC day, which must not shift the date
    ms = pd.Timestamp("2026-10-21 00:30:00").value // 10**6 + (sp - 1) * 86400_000
    return {"id": gid, "scoringPeriodId": sp, "date": ms, "homeProTeamId": home, "awayProTeamId": away,
            "startTimeTBD": tbd, "statsOfficial": True}


def _payload(games):
    by_team = {}
    for g in games:
        for t in (g["homeProTeamId"], g["awayProTeamId"]):
            by_team.setdefault(t, {}).setdefault(str(g["scoringPeriodId"]), []).append(g)
    teams = [{"id": t, "abbrev": str(t), "proGamesByScoringPeriod": v} for t, v in by_team.items()]
    return {"settings": {"proTeams": [{"id": 0, "proGamesByScoringPeriod": {}}] + teams}}


def test_parse_pro_schedule_uses_the_scoring_period_for_the_date_and_dedupes_games():
    games = [_game(1, 1, 2, 3), _game(2, 1, 1, 4), _game(3, 2, 3, 2), _game(4, 5, 1, 2)]
    df = S.parse_pro_schedule(_payload(games), "2026-27")
    assert len(df) == 4 and df["game_id"].is_unique       # each game appears under both teams in the payload
    d = df.set_index("game_id")
    assert d.at["1", "game_date"] == pd.Timestamp("2026-10-20")
    assert d.at["4", "game_date"] == pd.Timestamp("2026-10-24")
    assert d.at["1", "home_abbr"] == "BOS" and d.at["1", "away_abbr"] == "NOP"
    assert d.at["1", "home_team_id"] == 1610612738 and (df["source"] == "espn").all()
    assert not df["time_tbd"].any()


def test_parse_pro_schedule_errors_on_empty_and_unknown_teams():
    with pytest.raises(S.ScheduleError):
        S.parse_pro_schedule({"settings": {"proTeams": []}}, "2026-27")
    with pytest.raises(S.ScheduleError, match="unknown ESPN"):
        S.parse_pro_schedule(_payload([_game(1, 1, 2, 99)]), "2026-27")


def test_espn_team_table_is_a_bijection_onto_the_thirty_nba_ids():
    ids = [t for _, t in S.ESPN_TEAMS.values()]
    assert len(set(ids)) == 30 and sorted(ids) == list(range(1610612737, 1610612767))
    assert len({a for a, _ in S.ESPN_TEAMS.values()}) == 30


def test_validate_schedule_rejects_bad_tables():
    df = S.parse_pro_schedule(_payload([_game(1, 1, 2, 3)]), "2026-27")
    S.validate_schedule(df)
    with pytest.raises(ContractError, match="duplicate"):
        S.validate_schedule(pd.concat([df, df]))
    with pytest.raises(ContractError, match="plays itself"):
        S.validate_schedule(df.assign(away_team_id=df["home_team_id"]))
    with pytest.raises(ContractError, match="missing columns"):
        S.validate_schedule(df.drop(columns=["game_date"]))
    with pytest.raises(ContractError, match="unknown NBA team"):
        S.validate_schedule(df.assign(home_team_id=5))


def test_write_read_roundtrip_replaces_only_the_written_season(tmp_path):
    a = S.parse_pro_schedule(_payload([_game(1, 1, 2, 3)]), "2026-27")
    b = S.parse_pro_schedule(_payload([_game(9, 1, 2, 3), _game(10, 2, 1, 2)]), "2025-26")
    S.write_schedule(a, tmp_path)
    S.write_schedule(b, tmp_path)
    assert len(S.read_schedule(tmp_path)) == 3
    S.write_schedule(a.assign(game_id="77"), tmp_path)             # re-ingest replaces that season's rows
    got = S.read_schedule(tmp_path, "2026-27")
    assert list(got["game_id"]) == ["77"] and len(S.read_schedule(tmp_path, "2025-26")) == 2
    with pytest.raises(FileNotFoundError):
        S.read_schedule(tmp_path / "nowhere")


def test_from_team_games_pairs_home_and_away():
    t = make_tables()
    sch = S.from_team_games(t["team_games"], SEASON)
    assert len(sch) == N_TEAMS * GAMES // 2
    S.validate_schedule(sch)
    per_team = pd.concat([sch["home_team_id"], sch["away_team_id"]]).value_counts()
    assert (per_team == GAMES).all()
    with pytest.raises(S.ScheduleError):
        S.from_team_games(t["team_games"], "1999-00")


def _mini():
    """Four teams over 2 weeks: known back-to-backs and slate sizes."""
    ids = {"A": 1610612737, "B": 1610612738, "C": 1610612739, "D": 1610612740}
    rows = []

    def g(i, d, h, a):
        rows.append({"season": "s", "game_id": str(i), "scoring_period": None, "game_date": pd.Timestamp(d),
                     "home_team_id": ids[h], "away_team_id": ids[a], "home_abbr": h, "away_abbr": a,
                     "source": "team_games", "time_tbd": False})

    g(1, "2026-10-19", "A", "B")
    g(2, "2026-10-19", "C", "D")     # slate of 2 games
    g(3, "2026-10-20", "A", "C")                                     # A back-to-back (2nd night); slate of 1
    g(4, "2026-10-22", "B", "D")
    g(5, "2026-10-22", "A", "D")
    g(6, "2026-10-24", "A", "B")
    g(7, "2026-10-24", "C", "D")
    g(8, "2026-10-25", "B", "C")                                     # B: Oct 24 + 25 -> back-to-back
    g(9, "2026-10-26", "A", "B")
    g(10, "2026-10-28", "C", "D")      # week 2
    return S._frame(rows), ids


def test_weekly_team_table_counts_b2b_off_nights_and_games_left():
    sch, ids = _mini()
    cal = derive(date(2026, 10, 19), 2, regular_periods=2)
    wk = S.weekly_team_table(sch, cal, as_of="2026-10-22", off_max=1)
    w1 = wk[wk["week"] == 1].set_index("team_id")
    a, b = w1.loc[ids["A"]], w1.loc[ids["B"]]
    assert a["games"] == 4 and b["games"] == 4
    assert a["b2b"] == 1 and b["b2b"] == 1                      # A: Oct 19-20; B: Oct 24-25
    assert a["games_left"] == 1 and b["games_left"] == 2         # games dated after Oct 22 (the 22nd counts as played)
    assert a["off_night_games"] == 1                             # the lone Oct 20 game (slate of 1)
    assert bool(a["heavy"]) and not bool(b["light"])
    w2 = wk[wk["week"] == 2]
    assert w2["games"].sum() == 4 and w2["light"].sum() == 4     # week 2 is 1 game in 7 days for its two teams... and 0 for others
    left = S.games_remaining(sch, "2026-10-22")
    assert left[ids["A"]] == 2 and left[ids["B"]] == 3           # includes week 2


def test_expected_week_games_scales_by_availability():
    sch, ids = _mini()
    cal = derive(date(2026, 10, 19), 2, regular_periods=2)
    wk = S.weekly_team_table(sch, cal, as_of="2026-10-19")
    g = S.expected_week_games([ids["A"], ids["B"], 12345], wk, 1, availability=[0.5, 1.0, 1.0])
    assert list(g) == [0.5 * 3, 3.0, 0.0]                        # after Oct 19: A and B each have 3 left; unknown team 0
    assert list(S.expected_week_games([ids["A"]], wk, 1, remaining_only=False)) == [4.0]


def test_playoff_table_and_off_night_threshold():
    sch, _ = _mini()
    cal = derive(date(2026, 10, 19), 2, regular_periods=1)
    wk = S.weekly_team_table(sch, cal)
    p = S.playoff_weeks_table(wk)
    assert list(p.columns[:3]) == ["team_id", "abbr", "wk2"] and p["total"].sum() == 4
    assert S.off_night_max(pd.Series([2, 8, 10, 12, 14])) >= 2
    assert S.off_night_max(pd.Series(dtype=float)) == 0


def test_cli_from_team_games_and_report(tmp_path, capsys):
    from src.store import write_table

    t = make_tables()
    base = tmp_path / "d"
    write_table(t["team_games"], "team_games", base)
    rc = S.main(["--season", SEASON, "--from-team-games", "--data-dir", str(base), "--as-of", "2023-11-15", "--top", "3"])
    out = capsys.readouterr().out
    assert rc == 0 and "matchup calendar source: derived" in out and "League-wide by matchup week" in out
    assert S.read_schedule(base, SEASON).shape[0] == N_TEAMS * GAMES // 2
    rc = S.main(["--season", SEASON, "--data-dir", str(base), "--as-of", "2023-11-15", "--week", "2"])   # read stored
    assert rc == 0 and "Week 2, by team" in capsys.readouterr().out


def test_cli_errors_are_messages_not_tracebacks(tmp_path, capsys):
    assert S.main(["--season", "2030-31", "--data-dir", str(tmp_path)]) == 2                    # nothing stored
    assert "error:" in capsys.readouterr().err
    assert S.main(["--season", "2026-27", "--ingest", "--offline", "--data-dir", str(tmp_path)]) == 5   # cache miss
    assert "offline" in capsys.readouterr().err


def test_fetch_pro_schedule_is_one_cached_get(tmp_path):
    from src.ingest.http_cache import CachedHttpClient

    class Resp:
        status_code = 200
        headers = {}
        content = json.dumps(_payload([_game(1, 1, 2, 3)])).encode()

    class Session:
        calls = 0

        def get(self, url, params=None, headers=None, timeout=None):
            Session.calls += 1
            assert "proTeamSchedules_wl" in params["view"] and url.endswith("/seasons/2027")
            return Resp()

    c = CachedHttpClient(tmp_path, offline=False, session=Session(), sleep=lambda s: None, write_fetch_log=False)
    p1 = S.fetch_pro_schedule(c, "2026-27")
    p2 = S.fetch_pro_schedule(c, "2026-27")
    assert p1 == p2 and Session.calls == 1


def test_from_team_games_handles_games_with_no_home_flag_and_incomplete_pairs():
    tg = make_tables()["team_games"]
    tg = tg[tg["season"] == SEASON].copy()
    gid = tg["game_id"].iloc[0]
    tg.loc[tg["game_id"] == gid, "is_home"] = False               # neutral-site style quirk seen in real data
    orphan = tg["game_id"].iloc[2]
    tg = tg.drop(index=tg[tg["game_id"] == orphan].index[:1])     # one team row missing
    sch = S.from_team_games(tg, SEASON)
    assert gid in set(sch["game_id"]) and orphan not in set(sch["game_id"])
    assert len(sch) == N_TEAMS * GAMES // 2 - 1
