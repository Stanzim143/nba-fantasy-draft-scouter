"""wiki_transactions: pure wikitext parsing against the five REAL Wikipedia fixtures (no network,
no invented examples -- see docs/adr/0011-transactions-layer.md), plus the ad-hoc table IO and a
small fake-HTTP end-to-end pipeline run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingest_fakes import FakeClock, FakeResponse
from src.ingest import wiki_transactions as wt
from src.ingest.http_cache import CachedHttpClient
from src.store import read_table, write_table

FIXTURES = Path(__file__).parent / "fixtures" / "wiki_transactions"

GSW, MEM, LAL, TOR, CHA = (
    1610612744, 1610612763, 1610612747, 1610612761, 1610612766,
)


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_players_table(base: Path) -> pd.DataFrame:
    df = pd.DataFrame({
        "player_id": range(1, 3),
        "player_name": ["Stephen Curry", "Jimmy Butler"],
        "birthdate": pd.to_datetime(["1988-03-14", "1989-09-14"]),
        "position": ["G", "G-F"],
        "height_in": [75.0, 79.0],
        "weight_lb": [185.0, 230.0],
        "draft_year": pd.array([2009, 2011], dtype="Int64"),
        "draft_round": pd.array([1, 1], dtype="Int64"),
        "draft_number": pd.array([7, 8], dtype="Int64"),
        "from_year": pd.array([2009, 2011], dtype="Int64"),
        "to_year": pd.array([2026, 2026], dtype="Int64"),
    })
    write_table(df, "players", base)
    return df


# --------------------------------------------------------------------------- team mapping / titles

def test_team_season_page_title_uses_en_dash():
    title = wt.team_season_page_title(GSW, "2016-17")
    assert title == "2016–17 Golden State Warriors season"
    assert "–" in title
    assert "-" not in title  # a plain hyphen would be wrong -- must be the en dash


def test_team_season_page_title_unknown_team_raises():
    with pytest.raises(wt.WikiTransactionsError):
        wt.team_season_page_title(999, "2016-17")


def test_team_wiki_names_has_all_thirty_teams():
    assert len(wt.TEAM_WIKI_NAMES) == 30
    assert len(set(wt.TEAM_WIKI_NAMES.values())) == 30


# --------------------------------------------------------------------------- season parsing

def test_parse_seasons_range():
    assert wt.parse_seasons("2015-16:2017-18") == ["2015-16", "2016-17", "2017-18"]


def test_parse_seasons_single_and_list():
    assert wt.parse_seasons("2023-24") == ["2023-24"]
    assert wt.parse_seasons("2015-16,2020-21") == ["2015-16", "2020-21"]


def test_parse_seasons_rejects_malformed():
    with pytest.raises(ValueError):
        wt.parse_seasons("2015")
    with pytest.raises(ValueError):
        wt.parse_seasons("")
    with pytest.raises(ValueError):
        wt.parse_seasons("2020-21:2015-16")


# --------------------------------------------------------------------------- Warriors 2016-17 (hand-verified)

def test_warriors_additions_includes_kevin_durant_with_citation_date():
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    adds = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "addition"}
    assert "Kevin Durant" in adds
    assert adds["Kevin Durant"]["direction"] == "in"
    assert adds["Kevin Durant"]["txn_date"] == pd.Timestamp("2016-07-07")


def test_warriors_subtractions_includes_harrison_barnes():
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    subs = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "subtraction"}
    assert "Harrison Barnes" in subs
    assert subs["Harrison Barnes"]["direction"] == "out"
    assert subs["Harrison Barnes"]["txn_date"] == pd.Timestamp("2016-07-07")


def test_warriors_trades_includes_patrick_mccaw_june_23():
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    trades = [r for r in page.rows if r["source_kind"] == "trade"]
    mccaw = [r for r in trades if r["wiki_name"] == "Patrick McCaw"]
    assert len(mccaw) == 1
    assert mccaw[0]["direction"] == "in"
    assert mccaw[0]["txn_date"] == pd.Timestamp("2016-06-23")
    # the other leg of that same trade: Bogut goes out to Dallas on July 7
    bogut = [r for r in trades if r["wiki_name"] == "Andrew Bogut"]
    assert len(bogut) == 1
    assert bogut[0]["direction"] == "out"
    assert bogut[0]["txn_date"] == pd.Timestamp("2016-07-07")


def test_warriors_no_citation_date_keeps_row_as_nat_not_dropped():
    """JaVale McGee's Additions row has no <ref> at all -- must be kept with txn_date=NaT, not dropped."""
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    adds = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "addition"}
    assert "JaVale McGee" in adds
    assert pd.isna(adds["JaVale McGee"]["txn_date"])


def test_warriors_does_not_trust_the_signed_column():
    """The Additions table's 'Signed' column header is misleading -- for Kevin Durant it actually
    holds '2-year contract worth $54.3 million', not a date. The parser must never parse that text
    as a date; the row's date must come only from the citation."""
    contract_text = "2-year contract worth $54.3 million"
    assert wt._looks_like_date(wt._strip_markup(contract_text)) is False
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    adds = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "addition"}
    # if the contract text had been misparsed as a date it would not equal the real citation date
    assert adds["Kevin Durant"]["txn_date"] == pd.Timestamp("2016-07-07")
    assert adds["David West"]["txn_date"] == pd.Timestamp("2016-07-09")


def test_warriors_re_signed_players_are_never_emitted():
    page = wt.parse_team_season_wikitext(load_fixture("2016-17_Golden_State_Warriors.wiki"), GSW, "2016-17")
    names = {r["wiki_name"] for r in page.rows}
    # Ian Clark, James Michael McAdoo, Anderson Varejao (re-signed) must not appear as an addition;
    # Anderson Varejao DOES legitimately appear as a subtraction (waived later that season).
    assert "Ian Clark" not in names
    subs = {r["wiki_name"] for r in page.rows if r["source_kind"] == "subtraction"}
    assert any("Varej" in n for n in subs)
    adds = {r["wiki_name"] for r in page.rows if r["source_kind"] == "addition"}
    assert not any("Varej" in n for n in adds)


# --------------------------------------------------------------------------- other real fixtures

def test_memphis_three_team_trade_does_not_misattribute_third_leg():
    """The Feb 6, 2020 three-team trade (MEM/MIA/MIN): Memphis's page describes what MEM gave to
    MIA and received from MIA/MIN. The MIA<->MIN-only leg (James Johnson) never touches Memphis and
    must not appear as an MEM row at all."""
    page = wt.parse_team_season_wikitext(load_fixture("2019-20_Memphis_Grizzlies.wiki"), MEM, "2019-20")
    names = {r["wiki_name"] for r in page.rows if r["source_kind"] == "trade"}
    assert "James Johnson" not in names
    assert page.trade_counts["n_no_own_side"] >= 1
    # the legitimate legs are still there
    assert "Justise Winslow" in names
    assert "Andre Iguodala" in names


def test_lakers_re_signed_and_date_column_layout_still_works():
    """The Lakers page uses a DIFFERENT column order (Date | Player | ... ) from the Warriors page
    (Player | Signed | ...) -- the parser must not assume a fixed column position."""
    page = wt.parse_team_season_wikitext(load_fixture("2020-21_Los_Angeles_Lakers.wiki"), LAL, "2020-21")
    adds = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "addition"}
    assert "Montrezl Harrell" in adds
    assert adds["Montrezl Harrell"]["txn_date"] == pd.Timestamp("2020-11-22")
    trades = [r for r in page.rows if r["source_kind"] == "trade"]
    schroder = [r for r in trades if "Schr" in r["wiki_name"]]
    assert schroder and schroder[0]["direction"] == "in"


def test_toronto_headings_with_extra_spacing_still_found():
    """The Raptors page spells every heading with extra spaces ('== Transactions ==', not
    '==Transactions=='); the section finder must be whitespace-tolerant."""
    page = wt.parse_team_season_wikitext(load_fixture("2021-22_Toronto_Raptors.wiki"), TOR, "2021-22")
    names = {r["wiki_name"] for r in page.rows}
    assert "Kyle Lowry" in names  # traded out to Miami
    assert "Precious Achiuwa" in names  # received in that trade


def test_charlotte_rowspan_additions_row_with_no_own_ref_gets_nat():
    """Charlotte's Sept 5 addition block rowspans the date/ref across 4 players; only the first
    physical row carries the citation. A continuation row's own text has no <ref> -- must be NaT,
    not a crash, not a guessed date from a sibling row."""
    page = wt.parse_team_season_wikitext(load_fixture("2023-24_Charlotte_Hornets.wiki"), CHA, "2023-24")
    adds = {r["wiki_name"]: r for r in page.rows if r["source_kind"] == "addition"}
    assert "Nathan Mensah" in adds
    assert pd.isna(adds["Nathan Mensah"]["txn_date"])


# --------------------------------------------------------------------------- malformed input never crashes

def test_empty_wikitext_yields_zero_rows():
    page = wt.parse_team_season_wikitext("", GSW, "2016-17")
    assert page.rows == []


def test_garbage_wikitext_does_not_crash():
    garbage = "==Transactions==\nnot a table at all\n|-\n| no closing brace\n{| broken\n"
    page = wt.parse_team_season_wikitext(garbage, GSW, "2016-17")
    assert isinstance(page.rows, list)  # didn't raise


def test_page_missing_a_subsection_is_normal_not_an_error():
    text = "==Transactions==\n===Trades===\n{|\n|}\n"  # no Free agency section at all
    page = wt.parse_team_season_wikitext(text, GSW, "2016-17")
    assert page.rows == []


# --------------------------------------------------------------------------- full-fixture counts (sanity)

@pytest.mark.parametrize("fname,team_id,season", [
    ("2016-17_Golden_State_Warriors.wiki", GSW, "2016-17"),
    ("2019-20_Memphis_Grizzlies.wiki", MEM, "2019-20"),
    ("2020-21_Los_Angeles_Lakers.wiki", LAL, "2020-21"),
    ("2021-22_Toronto_Raptors.wiki", TOR, "2021-22"),
    ("2023-24_Charlotte_Hornets.wiki", CHA, "2023-24"),
])
def test_every_fixture_produces_some_rows_of_each_present_kind(fname, team_id, season):
    page = wt.parse_team_season_wikitext(load_fixture(fname), team_id, season)
    kinds = {r["source_kind"] for r in page.rows}
    assert "trade" in kinds
    assert "addition" in kinds
    assert "subtraction" in kinds
    assert all(r["team_id"] == team_id and r["season"] == season for r in page.rows)
    assert all(r["direction"] in ("in", "out") for r in page.rows)


# --------------------------------------------------------------------------- ad-hoc table IO

def test_team_transactions_round_trip(tmp_path):
    base = tmp_path / "data"
    df = pd.DataFrame({
        "season": ["2016-17", "2016-17"],
        "team_id": [GSW, GSW],
        "player_id": [1, 2],
        "direction": ["in", "out"],
        "source_kind": ["addition", "subtraction"],
        "txn_date": pd.to_datetime(["2016-07-07", pd.NaT]),
    })
    path = wt.write_team_transactions(df, base)
    assert path == wt.team_transactions_path(base)
    out = wt.read_team_transactions(base)
    assert list(out.columns) == wt.TEAM_TRANSACTIONS_COLUMNS
    assert len(out) == 2
    assert out["txn_date"].isna().sum() == 1


def test_team_transactions_round_trip_via_nba_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "envdata"))
    df = pd.DataFrame({
        "season": ["2016-17"], "team_id": [GSW], "player_id": [1], "direction": ["in"],
        "source_kind": ["trade"], "txn_date": pd.to_datetime(["2016-06-23"]),
    })
    wt.write_team_transactions(df)  # base=None -> data_dir() -> NBA_DATA_DIR
    out = wt.read_team_transactions()
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == 1


def test_read_team_transactions_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        wt.read_team_transactions(tmp_path / "data")


def test_write_team_transactions_rejects_missing_columns(tmp_path):
    with pytest.raises(wt.WikiTransactionsError):
        wt.write_team_transactions(pd.DataFrame({"season": ["2016-17"]}), tmp_path / "data")


# --------------------------------------------------------------------------- merge_id_map

def test_merge_id_map_replaces_only_its_own_source():
    existing = pd.DataFrame({"player_id": [9], "source": ["espn"], "source_id": ["z"],
                             "source_name": ["Z"], "match_method": ["manual"], "confidence": [1.0]})
    new = pd.DataFrame({"player_id": [1], "source": [wt.SOURCE], "source_id": ["Kevin Durant__2016"],
                        "source_name": ["Kevin Durant"], "match_method": ["exact"], "confidence": [1.0]})
    merged = wt.merge_id_map(existing, new, source=wt.SOURCE)
    assert set(merged["source"]) == {"espn", wt.SOURCE}
    assert len(merged) == 2


# --------------------------------------------------------------------------- end-to-end pipeline (fake HTTP)

class WikiFakeSession:
    """Serves canned wikitext for known page titles; everything else 404s."""

    def __init__(self, by_title: dict[str, str]):
        self.by_title = by_title
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        title = params.get("title")
        if title in self.by_title:
            return FakeResponse(200, self.by_title[title])
        return FakeResponse(404, b"page not found")


def make_client(tmp_path, by_title):
    clock = FakeClock()
    session = WikiFakeSession(by_title)
    client = CachedHttpClient(tmp_path / "raw_wiki", session=session, clock=clock, sleep=clock.sleep,
                              offline=False, min_interval=0, jitter=0)
    return client, session


def test_run_ingest_end_to_end_fetches_parses_matches_writes(tmp_path):
    base = tmp_path / "data"
    make_players_table(base)
    # pre-seed an unrelated source in player_id_map to confirm the merge doesn't clobber it
    write_table(pd.DataFrame({"player_id": [2], "source": ["espn"], "source_id": ["e1"],
                              "source_name": ["Jimmy Butler"], "match_method": ["exact"],
                              "confidence": [1.0]}), "player_id_map", base)

    title = wt.team_season_page_title(GSW, "2016-17")
    by_title = {title: load_fixture("2016-17_Golden_State_Warriors.wiki")}
    client, session = make_client(tmp_path, by_title)

    result = wt.run_ingest(["2016-17"], client, base, log=lambda *_: None)

    assert result.fetched == 1
    assert len(result.skipped_pages) == 29  # every other team's page 404s
    assert result.n_addition_rows > 0
    assert result.n_subtraction_rows > 0
    assert result.n_trade_rows > 0

    txns = wt.read_team_transactions(base)
    # Curry/Butler (the only two players in the fake `players` table) aren't part of this Warriors
    # trade/addition/subtraction set, so nothing matches -- what matters is it doesn't crash and
    # still produces a valid, schema-correct (empty) frame with every team_id/season it does have
    # (none, here) restricted to the one page fetched.
    assert list(txns.columns) == wt.TEAM_TRANSACTIONS_COLUMNS
    assert set(txns["team_id"]).issubset({GSW})
    assert set(txns["season"]).issubset({"2016-17"})

    id_map = read_table("player_id_map", base)
    assert "espn" in set(id_map["source"])  # untouched
    assert session.calls  # the fake session was actually used


def test_run_ingest_matches_a_real_player_end_to_end(tmp_path):
    """Same as above but the local `players` table actually contains Kevin Durant, so we can check
    a genuine name match makes it all the way through to team_transactions.parquet."""
    base = tmp_path / "data"
    df = pd.DataFrame({
        "player_id": [501], "player_name": ["Kevin Durant"],
        "birthdate": pd.to_datetime(["1988-09-29"]), "position": ["F"], "height_in": [82.0],
        "weight_lb": [240.0], "draft_year": pd.array([2007], dtype="Int64"),
        "draft_round": pd.array([1], dtype="Int64"), "draft_number": pd.array([2], dtype="Int64"),
        "from_year": pd.array([2007], dtype="Int64"), "to_year": pd.array([2026], dtype="Int64"),
    })
    write_table(df, "players", base)

    title = wt.team_season_page_title(GSW, "2016-17")
    by_title = {title: load_fixture("2016-17_Golden_State_Warriors.wiki")}
    client, _ = make_client(tmp_path, by_title)

    result = wt.run_ingest(["2016-17"], client, base, log=lambda *_: None)
    assert result.match_report.n_matched == 1

    txns = wt.read_team_transactions(base)
    durant_rows = txns[txns["player_id"] == 501]
    assert len(durant_rows) == 1
    assert durant_rows.iloc[0]["direction"] == "in"
    assert durant_rows.iloc[0]["source_kind"] == "addition"
    assert durant_rows.iloc[0]["txn_date"] == pd.Timestamp("2016-07-07")


def test_run_ingest_offline_without_cache_skips_every_uncached_page(tmp_path):
    """Offline mode with nothing cached hits the same 'report and skip' path as a 404: every
    team/season page is skipped (never a crash), leaving an empty but schema-correct output."""
    base = tmp_path / "data"
    make_players_table(base)
    client = CachedHttpClient(base / "raw_wiki", offline=True, min_interval=0)
    result = wt.run_ingest(["2016-17"], client, base, log=lambda *_: None)
    assert result.fetched == 0
    assert len(result.skipped_pages) == 30
    txns = wt.read_team_transactions(base)
    assert len(txns) == 0
    assert list(txns.columns) == wt.TEAM_TRANSACTIONS_COLUMNS


# --------------------------------------------------------------------------- CLI

def test_build_parser_seasons_required():
    parser = wt.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_defaults():
    parser = wt.build_parser()
    args = parser.parse_args(["--seasons", "2015-16:2025-26"])
    assert args.seasons == "2015-16:2025-26"
    assert args.offline is False
    assert args.min_interval == 1.0


def test_main_reports_failure_cleanly(tmp_path, capsys):
    base = tmp_path / "data"  # no players table -> run_ingest fails cleanly
    client, _ = make_client(tmp_path, {})
    rc = wt.main(["--seasons", "2016-17", "--data-dir", str(base)], client=client)
    assert rc == 1
    captured = capsys.readouterr()
    assert "wiki transactions ingest failed" in captured.err


def test_merge_team_transactions_replaces_only_fetched_pages():
    def rows(season, team, pid):
        return pd.DataFrame({"season": [season], "team_id": [team], "player_id": [pid], "direction": ["in"],
                             "source_kind": ["addition"], "txn_date": pd.to_datetime(["2016-07-01"])})
    existing = pd.concat([rows("2015-16", GSW, 1), rows("2016-17", GSW, 2), rows("2016-17", MEM, 3)],
                         ignore_index=True)
    new = rows("2016-17", GSW, 9)
    merged = wt.merge_team_transactions(existing, new, {("2016-17", GSW)})
    assert sorted(merged["player_id"]) == [1, 3, 9]


def test_merge_id_map_keeps_this_sources_rows_outside_the_run():
    cols = dict(source=[wt.SOURCE], source_name=["x"], match_method=["exact"], confidence=[1.0])
    existing = pd.DataFrame({"player_id": [1], "source_id": ["Old Guy__2015"], **cols})
    new = pd.DataFrame({"player_id": [2], "source_id": ["Kevin Durant__2016"], **cols})
    merged = wt.merge_id_map(existing, new, source=wt.SOURCE)
    assert set(merged["source_id"]) == {"Old Guy__2015", "Kevin Durant__2016"}


def test_run_ingest_keeps_existing_rows_of_other_seasons_and_main_fails_on_skips(tmp_path, capsys):
    base = tmp_path / "data"
    make_players_table(base)
    wt.write_team_transactions(pd.DataFrame({
        "season": ["2015-16"], "team_id": [GSW], "player_id": [1], "direction": ["in"],
        "source_kind": ["addition"], "txn_date": pd.to_datetime(["2015-07-01"])}), base)
    title = wt.team_season_page_title(GSW, "2016-17")
    client, _ = make_client(tmp_path, {title: load_fixture("2016-17_Golden_State_Warriors.wiki")})
    assert wt.main(["--seasons", "2016-17", "--data-dir", str(base)], client=client) == 1
    assert "skipped" in capsys.readouterr().err
    assert 1 in set(wt.read_team_transactions(base)["player_id"])
