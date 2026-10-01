"""espn_adp: season parsing, wiped-season detection, end-to-end pipeline (fake HTTP, no network),
FantasyPros gap fill, and a real integration with the unmodified ``load_adp``/``AdpBenchmark``.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ingest_fakes import FakeClock, FakeResponse, FakeSession
from src.backtest.benchmarks import ADP_REQUIRED, AdpBenchmark, load_adp
from src.contracts import History, validate_table
from src.ingest import espn_adp as ea
from src.ingest.http_cache import CachedHttpClient, OfflineCacheMiss
from src.store import read_table, write_table
from src.synthetic import make_synthetic_tables

# --------------------------------------------------------------------------- fixtures

def espn_player(id_, name, adp, pro_team_id=9):
    return {"id": id_, "fullName": name, "proTeamId": pro_team_id,
            "ownership": {"averageDraftPosition": adp, "auctionValueAverage": 10.0}}


def espn_response(players):
    return FakeResponse(200, json.dumps(players).encode("utf-8"))


def fp_html(rows):
    """rows: list of (rank, name, team, pos, espn_adp, avg_adp)."""
    head = "<tr><th>Rank</th><th>Player</th><th>ESPN</th><th>AVG</th></tr>"
    body = "".join(
        f"<tr><td>{rank}</td><td>{name} ({team} - {pos})</td><td>{espn}</td><td>{avg}</td></tr>"
        for rank, name, team, pos, espn, avg in rows
    )
    return f"<html><body><table>{head}{body}</table></body></html>"


def make_client(tmp_path, script):
    clock = FakeClock()
    session = FakeSession(script)
    return CachedHttpClient(tmp_path / "raw_espn", session=session, clock=clock, sleep=clock.sleep,
                            offline=False, min_interval=0, jitter=0), clock


def make_players_table(base):
    df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Stephen Curry", "Jimmy Butler", "Enes Kanter"],
        "birthdate": pd.to_datetime(["1988-03-14", "1989-09-14", "1992-05-20"]),
        "position": ["G", "G-F", "C"],
        "height_in": [75.0, 79.0, 83.0],
        "weight_lb": [185.0, 230.0, 245.0],
        "draft_year": pd.array([2009, 2011, 2011], dtype="Int64"),
        "draft_round": pd.array([1, 1, 1], dtype="Int64"),
        "draft_number": pd.array([7, 8, 3], dtype="Int64"),
        "from_year": pd.array([2009, 2011, 2011], dtype="Int64"),
        "to_year": pd.array([2026, 2026, 2020], dtype="Int64"),
    })
    write_table(df, "players", base)
    return df


# --------------------------------------------------------------------------- season parsing

def test_espn_season_id_is_end_year():
    assert ea.espn_season_id("2026-27") == 2027
    assert ea.espn_season_id("2015-16") == 2016


def test_season_from_espn_id_round_trips():
    assert ea.season_from_espn_id(2027) == "2026-27"
    assert ea.season_from_espn_id(ea.espn_season_id("2020-21")) == "2020-21"


def test_parse_seasons_range():
    assert ea.parse_seasons("2015-16:2017-18") == ["2015-16", "2016-17", "2017-18"]


def test_parse_seasons_single_and_list():
    assert ea.parse_seasons("2023-24") == ["2023-24"]
    assert ea.parse_seasons("2015-16,2020-21") == ["2015-16", "2020-21"]


def test_parse_seasons_rejects_malformed():
    with pytest.raises(ValueError):
        ea.parse_seasons("2015")
    with pytest.raises(ValueError):
        ea.parse_seasons("")
    with pytest.raises(ValueError):
        ea.parse_seasons("2020-21:2015-16")  # reversed


# --------------------------------------------------------------------------- wiped-season detection

def test_detect_wiped_season_flags_all_sentinel_values():
    df = pd.DataFrame({"season": "2025-26", "source_id": ["1", "2", "3"], "name": ["A", "B", "C"],
                       "pro_team_id": [1, 2, 3], "adp": [140.0, 140.0, 140.0]})
    check = ea.detect_wiped_season(df, "2025-26")
    assert check.wiped is True
    assert check.n_real_adp == 0


def test_detect_wiped_season_one_stray_real_value_still_wiped():
    df = pd.DataFrame({"season": "2025-26", "source_id": [str(i) for i in range(20)],
                       "name": [f"P{i}" for i in range(20)], "pro_team_id": [1] * 20,
                       "adp": [140.0] * 19 + [127.04]})
    check = ea.detect_wiped_season(df, "2025-26", min_real=10)
    assert check.wiped is True
    assert check.n_real_adp == 1


def test_detect_wiped_season_normal_season_not_flagged():
    df = pd.DataFrame({"season": "2023-24", "source_id": [str(i) for i in range(20)],
                       "name": [f"P{i}" for i in range(20)], "pro_team_id": [1] * 20,
                       "adp": [float(i + 1) for i in range(20)]})
    check = ea.detect_wiped_season(df, "2023-24", min_real=10)
    assert check.wiped is False
    assert check.n_real_adp == 20
    assert check.best_adp == 1.0


def test_real_adp_rows_drops_sentinel_and_null_and_zero():
    df = pd.DataFrame({"adp": [1.5, 140.0, None, 0.0, 99.9]})
    out = ea.real_adp_rows(df)
    assert out["adp"].tolist() == [1.5, 99.9]


# --------------------------------------------------------------------------- end-to-end pipeline

def test_run_ingest_writes_adp_and_id_map(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)

    season = "2023-24"
    players = [espn_player(100, "Stephen Curry", 1.5), espn_player(200, "Jimmy Butler III", 12.0),
              espn_player(300, "Someone Unmatched", 50.0)]
    client, _ = make_client(tmp_path, [espn_response(players)])

    result = ea.run_ingest([season], client, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)

    assert result.wiped == []
    assert result.adp_rows == 3
    assert result.id_map_rows == 2  # "Someone Unmatched" stays unmatched, never guessed
    assert result.match_report.n_unmatched == 1

    adp = pd.read_parquet(ea.adp_path(base))
    assert set(ADP_REQUIRED).issubset(adp.columns)
    assert (adp["source"] == "espn").all()
    assert adp["adp"].notna().all()
    assert set(adp["source_id"]) == {"100", "200", "300"}

    id_map = read_table("player_id_map", base)
    validate_table(id_map, "player_id_map")
    assert set(id_map["source_id"]) == {"100", "200"}
    curry_row = id_map[id_map["source_id"] == "100"].iloc[0]
    assert curry_row["player_id"] == 1 and curry_row["match_method"] == "exact"
    butler_row = id_map[id_map["source_id"] == "200"].iloc[0]
    assert butler_row["player_id"] == 2  # "Jimmy Butler III" -> "Jimmy Butler" via suffix stripping


def test_run_ingest_excludes_wiped_season_from_adp(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)
    wiped_players = [espn_player(100, "Stephen Curry", 140.0), espn_player(200, "Jimmy Butler III", 140.0)]
    client, _ = make_client(tmp_path, [espn_response(wiped_players)])

    result = ea.run_ingest(["2025-26"], client, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)

    assert result.wiped == ["2025-26"]
    assert result.adp_rows == 0
    # the id map is still populated (matching doesn't depend on ADP being real)
    assert result.id_map_rows == 2


def test_run_ingest_fills_gap_season_from_fantasypros(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)

    good_season_players = [espn_player(100, "Stephen Curry", 2.0), espn_player(200, "Jimmy Butler III", 15.0)]
    wiped_players = [espn_player(100, "Stephen Curry", 140.0), espn_player(200, "Jimmy Butler III", 140.0)]
    fp_page = fp_html([
        (1, "Stephen Curry", "GSW", "PG", "1.8", "1.9"),
        (2, "Jimmy Butler", "MIA", "SF", "14.5", "14.0"),
    ])

    espn_script = [espn_response(good_season_players), espn_response(wiped_players)]
    espn_client, _ = make_client(tmp_path, espn_script)
    fp_client, _ = make_client(tmp_path, [FakeResponse(200, fp_page)])

    result = ea.run_ingest(["2023-24", "2025-26"], espn_client, base, fill_gap=True,
                           fp_client=fp_client, min_real_adp=1, log=lambda *_: None)

    assert result.wiped == ["2025-26"]
    assert result.gap_filled == ["2025-26"]

    adp = pd.read_parquet(ea.adp_path(base))
    gap_rows = adp[adp["season"] == "2025-26"]
    assert len(gap_rows) == 2
    assert set(gap_rows["adp_source"]) == {"fantasypros_fill"}
    assert set(gap_rows["source_id"]) == {"100", "200"}  # matched onto the SAME espn ids
    curry_gap = gap_rows[gap_rows["source_id"] == "100"].iloc[0]
    assert curry_gap["adp"] == pytest.approx(1.8)


def test_gap_fill_falls_back_to_avg_when_espn_column_blank(tmp_path):
    """A FantasyPros row with no ESPN column value (empty cell -> NaN, not None, once it round-trips
    through a pandas float column) must fall back to AVG, not silently produce a null adp."""
    base = tmp_path / "data"
    make_players_table(base)
    wiped_players = [espn_player(100, "Stephen Curry", 140.0)]
    fp_page = fp_html([(1, "Stephen Curry", "GSW", "PG", "", "2.4")])  # blank ESPN column

    espn_client, _ = make_client(tmp_path, [espn_response(wiped_players)])
    fp_client, _ = make_client(tmp_path, [FakeResponse(200, fp_page)])

    result = ea.run_ingest(["2025-26"], espn_client, base, fill_gap=True, fp_client=fp_client,
                           min_real_adp=1, log=lambda *_: None)

    assert result.gap_filled == ["2025-26"]
    adp = pd.read_parquet(ea.adp_path(base))
    assert len(adp) == 1
    assert adp.iloc[0]["adp_source"] == "fantasypros_fill_avg"
    assert adp.iloc[0]["adp"] == pytest.approx(2.4)
    assert adp["adp"].notna().all()


def test_run_ingest_is_resumable_via_client_cache(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)
    players = [espn_player(100, "Stephen Curry", 1.5)]
    client, _ = make_client(tmp_path, [espn_response(players)])
    ea.run_ingest(["2023-24"], client, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)
    assert client.stats.network_requests == 1

    client2, _ = make_client(tmp_path, [FakeResponse(500)])
    client2.cache_dir = client.cache_dir
    ea.run_ingest(["2023-24"], client2, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)
    assert client2.stats.network_requests == 0  # served entirely from disk cache


def test_run_ingest_offline_without_cache_raises(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)
    client, _ = make_client(tmp_path, [])
    client.offline = True
    with pytest.raises(OfflineCacheMiss):
        ea.run_ingest(["2023-24"], client, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)


def test_merge_id_map_replaces_only_its_own_source(tmp_path):
    existing = pd.DataFrame({"player_id": [9], "source": ["other"], "source_id": ["z"],
                             "source_name": ["Z"], "match_method": ["manual"], "confidence": [1.0]})
    new = pd.DataFrame({"player_id": [1], "source": ["espn"], "source_id": ["100"],
                        "source_name": ["Stephen Curry"], "match_method": ["exact"], "confidence": [1.0]})
    merged = ea.merge_id_map(existing, new, source="espn")
    assert set(merged["source"]) == {"other", "espn"}
    assert len(merged) == 2


def test_run_ingest_keeps_other_seasons_and_idmap_rows_from_earlier_runs(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)
    c1, _ = make_client(tmp_path, [espn_response([espn_player(100, "Stephen Curry", 1.5)])])
    ea.run_ingest(["2022-23"], c1, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)
    c2, _ = make_client(tmp_path / "b", [espn_response([espn_player(200, "Jimmy Butler", 9.0)])])
    res = ea.run_ingest(["2023-24"], c2, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)
    adp = pd.read_parquet(ea.adp_path(base))
    assert set(adp["season"]) == {"2022-23", "2023-24"} and res.adp_rows == 2
    assert set(read_table("player_id_map", base)["source_id"]) == {"100", "200"}
    # re-running a season replaces just that season
    c3, _ = make_client(tmp_path / "c", [espn_response([espn_player(200, "Jimmy Butler", 4.0)])])
    ea.run_ingest(["2023-24"], c3, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)
    adp = pd.read_parquet(ea.adp_path(base))
    assert len(adp) == 2 and adp[adp["season"] == "2023-24"]["adp"].iloc[0] == 4.0


def test_main_puts_the_raw_cache_under_data_dir(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ea, "run_ingest", lambda seasons, client, base, **kw: seen.append(client.cache_dir) or
                        ea.AdpIngestResult(seasons=seasons))
    base = tmp_path / "data"
    assert ea.main(["--seasons", "2023-24", "--data-dir", str(base), "--offline"]) == 0
    assert seen[0] == base / "raw" / ea.CACHE_DIR_NAME


# --------------------------------------------------------------------------- CLI parsing

def test_build_parser_seasons_required():
    parser = ea.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_defaults():
    parser = ea.build_parser()
    args = parser.parse_args(["--seasons", "2015-16:2026-27"])
    assert args.seasons == "2015-16:2026-27"
    assert args.offline is False
    assert args.min_interval == 2.0
    assert args.limit == 600


def test_main_enforces_politeness_floor(tmp_path, capsys):
    base = tmp_path / "data"
    make_players_table(base)
    client, _ = make_client(tmp_path, [espn_response([espn_player(100, "Stephen Curry", 1.5)])])
    rc = ea.main(["--seasons", "2023-24", "--min-interval", "0.1", "--data-dir", str(base),
                 "--no-gap-fill"], client=client, fp_client=client)
    assert rc == 0
    captured = capsys.readouterr()
    assert "politeness floor" in captured.err


def test_main_reports_failure_cleanly(tmp_path, capsys):
    base = tmp_path / "data"
    # no players table written -> run_ingest should fail with a clear, caught error
    client, _ = make_client(tmp_path, [espn_response([espn_player(100, "Stephen Curry", 1.5)])])
    rc = ea.main(["--seasons", "2023-24", "--data-dir", str(base), "--no-gap-fill"], client=client)
    assert rc == 1
    captured = capsys.readouterr()
    assert "espn adp ingest failed" in captured.err


# --------------------------------------------------------------------------- load_adp integration

def test_output_loads_cleanly_through_unmodified_load_adp(tmp_path):
    """The single most important check: our adp.parquet + player_id_map must satisfy the EXISTING,
    unmodified load_adp/AdpBenchmark contract, unchanged."""
    base = tmp_path / "data"
    make_players_table(base)
    players = [espn_player(100, "Stephen Curry", 1.5), espn_player(200, "Jimmy Butler III", 12.0)]
    client, _ = make_client(tmp_path, [espn_response(players)])
    ea.run_ingest(["2023-24"], client, base, fill_gap=False, min_real_adp=1, log=lambda *_: None)

    id_map = read_table("player_id_map", base)
    adp_data = load_adp(ea.adp_path(base), id_map, source="espn")
    assert adp_data.n_unmapped == 0
    assert adp_data.n_rows == 2
    assert set(adp_data.frame["player_id"]) == {1, 2}

    bench = AdpBenchmark(adp_data)
    # AdpBenchmark.project only reads history.target_season, but History.until needs the four
    # required tables to build at all: any contract-valid set (synthetic here) works.
    synth = make_synthetic_tables(first_start=2015, last_start=2018, n_teams=8, games_per_team=10, seed=3)
    history = History.until(synth, "2018-19")
    history.target_season = "2023-24"
    proj = bench.project(history)
    assert bench.rank_only is True
    assert set(proj.columns) >= {"season", "player_id", "player_name", "model", "proj_total_fp"}
    assert len(proj) == 2
    # lower ADP -> higher (better) ordinal score
    curry = proj[proj["player_id"] == 1].iloc[0]["proj_total_fp"]
    butler = proj[proj["player_id"] == 2].iloc[0]["proj_total_fp"]
    assert curry > butler
