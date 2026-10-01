"""espn_transactions: grammar, matching, the append-only ledger, paging. Offline; descriptions are written in the feed's style."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingest import espn_transactions as et

ATL, MIA, MIL, MIN, UTA, CHA, DAL, OKC, CLE, LAC, LAL = (et.ESPN_TEAMS[k][0] for k in
                                                        ("ATL", "MIA", "MIL", "MIN", "UTAH", "CHA", "DAL", "OKC", "CLE", "LAC", "LAL"))


def kinds(desc):
    acts, unparsed = et.parse_description(desc)
    return [(a.kind, a.person, a.other_team[1] if a.other_team else None, a.detail) for a in acts], unparsed


# --------------------------------------------------------------------------- grammar

def test_clean_names_drops_positions_picks_and_cash():
    assert et.clean_names("Gs Buddy Hield, Ryan Nembhard and cash considerations") == ["Buddy Hield", "Ryan Nembhard"]
    assert et.clean_names("C Nic Claxton") == ["Nic Claxton"]
    assert et.clean_names("F Dorian Finney-Smith and three second-round picks") == ["Dorian Finney-Smith"]
    assert et.clean_names("draft considerations") == []
    assert et.clean_names("forward Cam Whitmore") == ["Cam Whitmore"]
    # a name that merely starts with a position letter is not eaten
    assert et.clean_names("C.J. McCollum and G Cole Anthony") == ["C.J. McCollum", "Cole Anthony"]


def test_signings_and_contract_types():
    out, bad = kinds("Signed G Michael Ajayi and Kylan Boswell to two-way contracts. Re-signed G Coby White to a contract.")
    assert bad == 0
    assert out == [("signed", "Michael Ajayi", None, "two_way"), ("signed", "Kylan Boswell", None, "two_way"),
                   ("resigned", "Coby White", None, "")]
    assert kinds("Signed G Ausar Thompson to a rookie-scale contract extension.")[0] == [("extended", "Ausar Thompson", None, "rookie_scale+extension")]
    assert kinds("Signed G Pelle Larsson to a veteran contract extension.")[0][0][0] == "extended"


def test_waive_claim_convert():
    assert kinds("Waived G DJ Armstrong and C Tre Carroll.")[0] == [("waived", "DJ Armstrong", None, ""), ("waived", "Tre Carroll", None, "")]
    assert kinds("Placed G Miles Kelly on waivers.")[0] == [("waived", "Miles Kelly", None, "")]
    assert kinds("Claimed G Bez Mbeng off waivers.")[0] == [("claimed", "Bez Mbeng", None, "")]
    assert kinds("Converted the contract of G A.J. Lawson to an NBA contract.")[0] == [("converted", "A.J. Lawson", None, "to_nba")]
    assert kinds("Converted G Malachi Smith to a two-way contract.")[0] == [("converted", "Malachi Smith", None, "two_way")]


def test_a_two_sided_trade_reads_both_directions_and_the_counterparty():
    out, bad = kinds("Acquired Gs Buddy Hield, Ryan Nembhard and cash considerations from Atlanta in exchange for F Dorian Finney-Smith.")
    assert bad == 0
    assert ("trade_in", "Buddy Hield", "ATL", "") in out and ("trade_in", "Ryan Nembhard", "ATL", "") in out
    assert ("trade_out", "Dorian Finney-Smith", "ATL", "") in out


def test_trade_with_for_clause_and_full_team_name():
    out, _ = kinds("Acquired F Naz Reid and draft considerations from Minnesota Timberwolves for Gs LaMelo Ball and Josh Green.")
    assert ("trade_in", "Naz Reid", "MIN", "") in out
    assert {(k, p) for k, p, *_ in out if k == "trade_out"} == {("trade_out", "LaMelo Ball"), ("trade_out", "Josh Green")}


def test_three_team_trade_assigns_each_leg_its_own_team():
    out, _ = kinds("Acquired G Ryan Nembhard from Dallas and G Luguentz Dort from Oklahoma City in a three-team trade that "
                   "sent F Zaccharie Risacher to Dallas. Dallas received draft consideration from Atlanta.")
    assert ("trade_in", "Ryan Nembhard", "DAL", "") in out
    assert ("trade_in", "Luguentz Dort", "OKC", "") in out
    assert ("trade_out", "Zaccharie Risacher", "DAL", "") in out


def test_picks_only_trade_has_no_person_and_is_not_unparsed():
    out, bad = kinds("Acquired draft considerations from Atlanta Hawks for G Aaron Wiggins.")
    assert out == [("trade_out", "Aaron Wiggins", "ATL", "")] and bad == 0
    assert kinds("Received draft considerations from LA Clippers.") == ([], 0)


def test_sentence_split_does_not_break_on_initials_or_suffix_periods():
    parts = et.split_sentences("Signed G A.J. Lawson to a contract. Waived F Larry Nance Jr. Signed G C.J. McCollum to a contract.")
    assert len(parts) == 3


def test_staff_sentences():
    assert kinds("Hired Micah Nori as head coach.")[0] == [("hired", "Micah Nori", None, "head_coach")]
    assert kinds("Fired head coach Jamahl Mosley.")[0] == [("fired", "Jamahl Mosley", None, "head_coach")]
    assert kinds("Announced the resignation of head coach Billy Donovan.")[0] == [("resigned_staff", "Billy Donovan", None, "head_coach")]
    out, _ = kinds("Fired executive vice president of basketball operations Arturas Karnisovas and general manager Marc Eversley.")
    assert {(k, p, d) for k, p, _, d in out} == {("fired", "Arturas Karnisovas", "president_ops"), ("fired", "Marc Eversley", "general_manager")}
    assert kinds("Signed head coach Charles Lee to a contract extension.")[0] == [("extended_staff", "Charles Lee", None, "head_coach")]
    out, _ = kinds("Named Jason Kidd, Lionel Hollins and Phil Handy assistant coaches.")
    assert [(k, p, d) for k, p, _, d in out] == [("hired", "Jason Kidd", "assistant_coach"), ("hired", "Lionel Hollins", "assistant_coach"),
                                                 ("hired", "Phil Handy", "assistant_coach")]


def test_an_unrecognised_sentence_is_counted_never_guessed():
    out, bad = kinds("Suspended G Somebody Else for two games.")
    assert out == [] and bad == 1


def test_team_names():
    assert et.team_from_text("the Boston Celtics") == et.ESPN_TEAMS["BOS"]
    assert et.team_from_text("L.A. Clippers") == et.ESPN_TEAMS["LAC"]
    assert et.team_from_text("Cleveland Guardians") == et.ESPN_TEAMS["CLE"]     # the feed has typos; the location wins
    assert et.team_from_text("Los Angeles") is None                              # ambiguous, never guessed
    assert et.team_from_espn("GS") == (1610612744, "GSW") and et.team_from_espn("nope") is None


# --------------------------------------------------------------------------- feed -> records

def row(day, abbr, text):
    return {"date": f"{day}T07:00Z", "description": text, "team": {"abbreviation": abbr}}


def test_records_keys_are_stable_and_ignore_the_parsed_counterparty():
    recs, stats = et.feed_rows_to_records([row("2026-07-08", "CHA", "Acquired G Dennis Schroder from Cleveland."),
                                           row("2026-07-08", "ZZZ", "Waived G Nobody Real.")])
    assert stats["unknown_team"] == 1 and len(recs) == 1
    again, _ = et.feed_rows_to_records([row("2026-07-08", "CHA", "Acquired G Dennis Schroder from Cleveland Cavaliers.")])
    assert recs[0]["txn_key"] == again[0]["txn_key"] and recs[0]["other_team_id"] == CLE


def test_a_row_with_nothing_readable_is_kept_as_other():
    recs, stats = et.feed_rows_to_records([row("2026-07-08", "LAL", "Something unexpected happened.")])
    assert [r["kind"] for r in recs] == ["other"] and stats["other_rows"] == 1 and stats["unparsed_sentences"] == 1


# --------------------------------------------------------------------------- matching

def universe(active, extra=()):
    def f(rows):
        return pd.DataFrame(rows, columns=["player_id", "player_name", "from_year", "to_year"]).astype(
            {"player_id": "int64", "from_year": "float64", "to_year": "float64"})
    a = f(active)
    return a, pd.concat([a, f(extra)], ignore_index=True)


def test_father_and_son_resolve_to_the_active_one():
    uni = universe([(1, "Tim Hardaway Jr.", 2013, float("nan"))], extra=[(2, "Tim Hardaway", 1989, 2003)])
    recs, _ = et.feed_rows_to_records([row("2026-07-06", "MIA", "Signed G Tim Hardaway Jr. to a contract.")])
    summary, unresolved = et.attach_player_ids(recs, uni)
    assert recs[0]["player_id"] == 1 and summary["matched"] == 1 and unresolved == []


def test_unmatched_camp_players_are_reported_and_keep_a_null_id():
    uni = universe([(1, "Real Player", 2020, float("nan"))])
    recs, _ = et.feed_rows_to_records([row("2026-09-22", "DEN", "Waived G Real Player and F Camp Guy.")])
    summary, unresolved = et.attach_player_ids(recs, uni)
    by = {r["person"]: r["player_id"] for r in recs}
    assert by["Real Player"] == 1 and by["Camp Guy"] is None
    assert [u["name"] for u in unresolved] == ["Camp Guy"] and summary["match_rate"] == 0.5


# --------------------------------------------------------------------------- ledger

def frame(records, when):
    return et._frame(records, datetime(*when, tzinfo=timezone.utc))


def test_merge_is_append_only_and_keeps_first_seen():
    r1, _ = et.feed_rows_to_records([row("2026-07-08", "CHA", "Acquired G Dennis Schroder from Cleveland.")])
    r2, _ = et.feed_rows_to_records([row("2026-07-08", "CHA", "Acquired G Dennis Schroder from Cleveland."),
                                     row("2026-07-09", "MIA", "Signed G New Guy to a contract.")])
    first = frame(r1, (2026, 7, 8, 12))
    merged, added = et.merge_ledger(None, first)
    assert added == 1
    second = frame(r2, (2026, 7, 9, 12))
    merged2, added2 = et.merge_ledger(merged, second)
    assert added2 == 1 and len(merged2) == 2
    old = merged2[merged2["person"] == "Dennis Schroder"].iloc[0]
    assert old["first_seen"] == pd.Timestamp("2026-07-08 12:00") and old["last_seen"] == pd.Timestamp("2026-07-09 12:00")
    merged3, added3 = et.merge_ledger(merged2, second)
    assert added3 == 0 and len(merged3) == 2


def test_history_is_not_news():
    recs, _ = et.feed_rows_to_records([row("2026-01-10", "MIA", "Signed G Old Move to a contract."),
                                       row("2026-09-20", "MIA", "Signed G Fresh Move to a contract.")])
    f = frame(recs, (2026, 9, 25, 12))
    old, fresh = f[f["person"] == "Old Move"].iloc[0], f[f["person"] == "Fresh Move"].iloc[0]
    assert old["first_seen"] == pd.Timestamp("2026-01-10") and old["last_seen"] == pd.Timestamp("2026-09-25 12:00")
    assert fresh["first_seen"] == pd.Timestamp("2026-09-25 12:00")


def test_merge_fills_a_missing_id_but_never_overwrites_one():
    r, _ = et.feed_rows_to_records([row("2026-07-08", "CHA", "Signed G Some Body to a contract.")])
    ledger, _ = et.merge_ledger(None, frame(r, (2026, 7, 8, 1)))
    assert pd.isna(ledger.loc[0, "player_id"])
    r[0]["player_id"] = 5
    ledger2, _ = et.merge_ledger(ledger, frame(r, (2026, 7, 9, 1)))
    assert ledger2.loc[0, "player_id"] == 5
    r[0]["player_id"] = 9
    ledger3, _ = et.merge_ledger(ledger2, frame(r, (2026, 7, 10, 1)))
    assert ledger3.loc[0, "player_id"] == 5


# --------------------------------------------------------------------------- fetching

class FakeClient:
    """Serves canned pages keyed by (dates, page); records every call, and whether it was told to refresh."""

    offline = False

    def __init__(self, pages):
        self.pages, self.calls = pages, []
        self.stats = SimpleNamespace(network_requests=0, cache_hits=0)

    def get_json(self, label, url, params=None, *, refresh=False, **_):
        self.calls.append((dict(params), refresh))
        self.stats.network_requests += 1
        return self.pages[(params["dates"], params["page"])]


def test_windows_cover_the_range_without_overlap():
    w = et.windows(date(2026, 1, 1), date(2026, 3, 15), 31)
    assert w[0][0] == date(2026, 1, 1) and w[-1][1] == date(2026, 3, 15)
    assert all(b[0] == a[1] + pd.Timedelta(days=1) for a, b in zip(w, w[1:]))


def test_fetch_window_follows_pages_and_refreshes_only_recent_windows():
    pages = {("20260901-20260910", 1): {"transactions": [row("2026-09-02", "MIA", "Waived G A Person.")], "pageCount": 2},
             ("20260901-20260910", 2): {"transactions": [row("2026-09-03", "MIA", "Waived G B Person.")], "pageCount": 2}}
    old = et.fetch_window(FakeClient(pages), date(2026, 9, 1), date(2026, 9, 10), today=date(2026, 12, 1))
    assert len(old) == 2
    c = FakeClient(pages)
    et.fetch_window(c, date(2026, 9, 1), date(2026, 9, 10), today=date(2026, 9, 11))
    assert [flag for _, flag in c.calls] == [True, True]
    c2 = FakeClient(pages)
    et.fetch_window(c2, date(2026, 9, 1), date(2026, 9, 10), today=date(2026, 12, 1))
    assert [flag for _, flag in c2.calls] == [False, False]


def test_a_payload_of_the_wrong_shape_is_a_clear_error():
    with pytest.raises(et.EspnTransactionsError):
        et.fetch_window(FakeClient({("20260901-20260901", 1): {"oops": 1}}), date(2026, 9, 1), date(2026, 9, 1), today=date(2026, 12, 1))


def test_an_empty_window_omits_the_key_and_is_not_an_error():
    empty = {"timestamp": "x", "status": "success", "count": 0, "pageIndex": 1, "pageSize": 300, "pageCount": 0}
    assert et.fetch_window(FakeClient({("20181021-20181121", 1): empty}), date(2018, 10, 21), date(2018, 11, 21), today=date(2026, 9, 25)) == []


# --------------------------------------------------------------------------- regressions found by the independent verifier

def test_exchange_without_for_and_three_team_without_a_hyphen():
    out, _ = kinds("Acquired G AJ Johnson, Fs Khris Middleton and Marvin Bagley III, and draft considerations from Washington in exchange "
                   "Gs Jaden Hardy, Dante Exum and D'Angelo Russel and F Anthony Davis in a three team trade.")
    ins = {p for k, p, *_ in out if k == "trade_in"}
    outs = {p for k, p, *_ in out if k == "trade_out"}
    assert "Anthony Davis" not in ins and not any("trade" in p for p in ins | outs)
    assert {"AJ Johnson", "Khris Middleton", "Marvin Bagley III"} <= ins


def test_source_team_prefix_and_trailing_clauses_never_become_part_of_a_name():
    out, _ = kinds("Acquired Minnesota G Mike Conley Jr. and Detroit G Jaden Ivey in a three-team trade.")
    assert {p for k, p, *_ in out} == {"Mike Conley Jr", "Jaden Ivey"} or {p for k, p, *_ in out} == {"Mike Conley Jr.", "Jaden Ivey"}
    for text, who in [("Signed G Cam Thomas off waivers after getting waived by Brooklyn.", "Cam Thomas"),
                      ("Signed G Jordan McRae for the remainder of this season.", "Jordan McRae"),
                      ("Signed G Some Guy through the 2017-18 season.", "Some Guy"),
                      ("Waived G Anthony Brown from Erie (NBADL).", "Anthony Brown")]:
        assert [p for _, p, *_ in kinds(text)[0]] == [who], text


def test_names_with_stray_words_are_refused_not_recorded():
    assert et.clean_names("F Tolu might") == [] and et.clean_names("Foo bar baz qux") == []
    assert et.valid_name("Wes Unseld Jr.") and et.valid_name("Luka Doncic") and et.valid_name("Kel'el Ware")
    assert not et.valid_name("Jordan McRae for the remainder of this season")


def test_slash_positions():
    assert et.clean_names("G/F Josh Okogie, F/C Drew Eubanks and C/F Foo Barr") == ["Josh Okogie", "Drew Eubanks", "Foo Barr"]


def test_compound_sentences_lose_nothing():
    out, _ = kinds("Signed G New Guy to a two-way contract and waived F Old Guy.")
    assert [(k, p) for k, p, *_ in out] == [("signed", "New Guy"), ("waived", "Old Guy")]
    out, _ = kinds("Acquired F A One from Chicago in exchange for G B Two and sent F Cee Three to Utah.")
    assert ("trade_out", "Cee Three", "UTA", "") in out


def test_traded_verb_reads_both_sides():
    out, bad = kinds("Traded F David Lee to Boston for F Gerald Wallace and G Chris Babb.")
    assert bad == 0
    assert ("trade_out", "David Lee", "BOS", "") in out and ("trade_in", "Gerald Wallace", "BOS", "") in out


def test_a_guessed_counterparty_is_only_used_when_there_is_exactly_one_candidate():
    out, _ = kinds("Acquired F Ousmane Dieng from Chicago and F Nigel Hayes from Phoenix in exchange for Gs Cole Anthony and Amir Coffey.")
    assert [o for k, p, o, _ in out if k == "trade_out"] == [None, None]


def test_staff_names_and_roles_are_validated():
    assert kinds("Promoted assistant coach J.B. Bickerstaff to interim head coach.")[0] == [("hired", "J.B. Bickerstaff", None, "head_coach")]
    assert kinds("Named Frank Vogel coach.")[0] == [("hired", "Frank Vogel", None, "head_coach")]
    assert not any("vice" in p for _, p, *_ in kinds("Named Brett Stefansson executive vice president, general manager of basketball operations.")[0])


def test_a_better_parse_supersedes_the_old_rows_and_keeps_first_seen():
    old = et._frame([{"txn_key": "old1", "txn_date": pd.Timestamp("2026-09-20"), "team_id": MIA, "team_abbr": "MIA", "kind": "signed",
                      "subject_type": "player", "person": "Foo Bar for the remainder of the season", "player_id": None, "other_team_id": None,
                      "detail": "", "description": "Signed G Foo Bar for the remainder of the season."}],
                    datetime(2026, 9, 21, 8, tzinfo=timezone.utc))
    recs, _ = et.feed_rows_to_records([row("2026-09-20", "MIA", "Signed G Foo Bar for the remainder of the season.")])
    merged, added = et.merge_ledger(old, et._frame(recs, datetime(2026, 9, 25, 8, tzinfo=timezone.utc)))
    assert list(merged["person"]) == ["Foo Bar"] and added == 1
    assert merged.iloc[0]["first_seen"] == pd.Timestamp("2026-09-21 08:00")


def test_duplicate_feed_rows_do_not_write_duplicate_keys():
    recs, _ = et.feed_rows_to_records([row("2026-09-20", "MIA", "Waived G Foo Bar."), row("2026-09-20", "MIA", "Waived G Foo Bar.")])
    merged, added = et.merge_ledger(None, frame(recs, (2026, 9, 25, 8)))
    assert len(merged) == 1 and added == 1


def test_round_two_review_regressions():
    out, _ = kinds("Named Lamar Skeeter, Josh Longstaff and Jermaine Bucknor assistant coaches and Zach Peterson as assistant coach/director "
                   "and Kemba Walker as player enhancement coach.")
    assert not any(d == "head_coach" for *_, d in out)
    assert not any(d == "head_coach" for *_, d in kinds("Named Jason Kidd, Lionel Hollins and Phil Handy assistant coaches.")[0])
    assert kinds("Promoted director of strategic planning Pat Garrity to assistant general manager, executive director Andrew Loomis to "
                 "chief of staff and assistant coach Bob Beyer to associate head coach.")[0] == []
    out, _ = kinds("Signed F Chase Budinger for the remainder of the season and F/C Alan Williams to a 10-day contract.")
    assert [(p, d) for _, p, _, d in out] == [("Chase Budinger", ""), ("Alan Williams", "10_day")]
    out, _ = kinds("Signed F Nikola Jokic to a contract, and G C.J. McCollum to a two-way contract.")
    assert [(p, d) for _, p, _, d in out] == [("Nikola Jokic", ""), ("C.J. McCollum", "two_way")]
    assert et.clean_names("G's Killian Hayes and Atlanta's 2025") == ["Killian Hayes"]
    out, _ = kinds("Acquired Foo Bar from Boston in exchange for Al Horford in a three-team trade.")
    assert ("trade_out", "Al Horford", None, "") in out


def test_round_three_review_regressions():
    for who in ("Charles Lee", "Mike Budenholzer", "Mike Brown"):
        assert kinds(f"Named {who} head coach.")[0] == [("hired", who, None, "head_coach")]
    assert kinds("Named Frank Vogel coach.")[0] == [("hired", "Frank Vogel", None, "head_coach")]
    out, _ = kinds("Signed F Hannes Steinbach and G Christian Anderson to rookie scale contracts.")
    assert [(p, d) for _, p, _, d in out] == [("Hannes Steinbach", "rookie_scale"), ("Christian Anderson", "rookie_scale")]
    assert kinds("Hired Masai Ujiri as president and alternate governor.")[0] == [("hired", "Masai Ujiri", None, "president_ops")]
    out, _ = kinds("Signed Gs A One and B Two to two-way contracts.")
    assert [d for *_, d in out] == ["two_way", "two_way"]


# --------------------------------------------------------------------------- end to end

def test_run_ingest_writes_the_ledger_and_a_rerun_adds_nothing(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    pd.DataFrame({"player_id": [1, 2], "player_name": ["Buddy Hield", "Dorian Finney-Smith"], "from_year": [2016, 2016],
                  "to_year": [2026, 2026]}).to_parquet(proc / "players.parquet")
    page = {"transactions": [row("2026-09-22", "CHA", "Waived G Camp Guy. Acquired Gs Buddy Hield and cash considerations from "
                                                   "Atlanta in exchange for F Dorian Finney-Smith."),
                             row("2026-05-04", "ORL", "Fired head coach Jamahl Mosley.")], "pageCount": 1}
    client = FakeClient({("20260922-20260925", 1): page})
    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    res = et.run_ingest(date(2026, 9, 22), date(2026, 9, 25), client, tmp_path, now=now, log=lambda *_: None)
    assert res.records == 4 and res.new_rows == 4 and res.match["matched"] == 2
    ledger = et.read_ledger(tmp_path)
    assert set(ledger["kind"]) == {"waived", "trade_in", "trade_out", "fired"}
    assert ledger.loc[ledger["person"] == "Buddy Hield", "player_id"].iloc[0] == 1
    assert ledger.loc[ledger["subject_type"] == "staff", "detail"].iloc[0] == "head_coach"
    again = et.run_ingest(date(2026, 9, 22), date(2026, 9, 25), client, tmp_path, now=now, log=lambda *_: None)
    assert again.new_rows == 0 and again.ledger_rows == 4
    assert (tmp_path / "processed" / et.REPORT_NAME).exists()


def test_a_reversed_window_is_refused(tmp_path):
    with pytest.raises(et.EspnTransactionsError):
        et.run_ingest(date(2026, 9, 25), date(2026, 9, 1), FakeClient({}), tmp_path)


def test_a_naive_now_is_treated_as_utc():
    naive = et._frame([], datetime(2026, 9, 25, 8))
    aware = et._frame([], datetime(2026, 9, 25, 8, tzinfo=timezone.utc))
    assert naive.equals(aware)
