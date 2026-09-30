"""txn_impact: collapse the ledger to one row per player move, join the board, render the digest."""
import pandas as pd

from src.features import txn_impact as ti
from src.ingest import espn_transactions as et

T = et.ESPN_TEAMS


def led(rows):
    """rows: (date, team_abbr(ESPN), kind, person, player_id, other_abbr(ESPN)|None, detail, subject)"""
    recs = []
    for i, (day, abbr, kind, person, pid, other, detail, *rest) in enumerate(rows):
        recs.append({"txn_key": f"k{i}", "txn_date": pd.Timestamp(day), "team_id": T[abbr][0], "team_abbr": T[abbr][1], "kind": kind,
                     "subject_type": rest[0] if rest else "player", "person": person, "player_id": pid,
                     "other_team_id": T[other][0] if other else None, "detail": detail, "description": f"row {i}"})
    return et._frame(recs, pd.Timestamp("2026-09-25 12:00", tz="UTC").to_pydatetime())


BOARD = pd.DataFrame({"player_id": [1, 2, 3, 4, 5], "rank": [3, 10, 40, 150, 400], "name": ["Star", "Mate One", "Mate Two", "Depth", "Fringe"],
                      "position": ["F", "F", "F", "G", "G"], "vorp": [2000.0, 1000, 500, 100, 5], "adp": [3.0, 9, 41, 150, float("nan")]})
ROSTER = pd.DataFrame({"player_id": [2, 3, 4], "team_abbr": ["MIA", "MIA", "MIA"]})


def test_a_trade_seen_from_both_sides_is_one_move():
    ledger = led([("2026-07-06", "MIA", "trade_in", "Star", 1, "MIL", ""), ("2026-07-06", "MIL", "trade_out", "Star", 1, "MIA", "")])
    ev = ti.events(ledger)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert (r.kind, r.from_abbr, r.to_abbr) == ("trade", "MIL", "MIA")


def test_a_one_sided_trade_row_still_names_both_teams_when_it_can():
    ev = ti.events(led([("2026-07-06", "MIA", "trade_in", "Star", 1, "MIL", "")]))
    assert (ev.iloc[0].from_abbr, ev.iloc[0].to_abbr) == ("MIL", "MIA")
    ev2 = ti.events(led([("2026-07-06", "MIL", "trade_out", "Star", 1, "MIA", "")]))
    assert (ev2.iloc[0].from_abbr, ev2.iloc[0].to_abbr) == ("MIL", "MIA")


def test_arrivals_and_departures():
    ev = ti.events(led([("2026-07-07", "MIA", "signed", "Star", 1, None, "two_way"), ("2026-07-08", "MIA", "waived", "Fringe", 5, None, ""),
                        ("2026-07-09", "MIA", "other", "", None, None, ""), ("2026-07-09", "MIA", "hired", "Coach", None, None, "head_coach", "staff")]))
    assert len(ev) == 2
    s = ev[ev["kind"] == "signed"].iloc[0]
    assert s.to_abbr == "MIA" and pd.isna(s.from_abbr) and s.detail == "two_way"
    w = ev[ev["kind"] == "waived"].iloc[0]
    assert w.from_abbr == "MIA" and pd.isna(w.to_abbr)


def test_staff_events():
    st = ti.staff_events(led([("2026-05-04", "ORL", "fired", "Jamahl Mosley", None, None, "head_coach", "staff"),
                              ("2026-07-07", "MIA", "signed", "Star", 1, None, "")]))
    assert list(st["person"]) == ["Jamahl Mosley"] and st.iloc[0]["role"] == "head_coach"


def test_annotate_adds_rank_position_and_crowding_context():
    ev = ti.events(led([("2026-07-06", "MIA", "trade_in", "Star", 1, "MIL", ""), ("2026-07-07", "MIA", "signed", "Fringe", 5, None, "")]))
    ann = ti.annotate(ev, BOARD, ROSTER)
    star = ann[ann["person"] == "Star"].iloc[0]
    assert star["rank"] == 3 and star["position"] == "F" and star["relevant"]
    assert "MIA best teammates: Mate One (#10), Mate Two (#40), Depth (#150)" in star["context"]
    fringe = ann[ann["person"] == "Fringe"].iloc[0]
    assert not fringe["relevant"] and fringe["context"] == ""


def test_crowded_position_is_called_out_only_when_three_share_it():
    roster = pd.DataFrame({"player_id": [2, 3, 4], "team_abbr": ["MIA"] * 3})
    board = BOARD.assign(position=["F", "F", "F", "F", "G"])
    ann = ti.annotate(ti.events(led([("2026-07-06", "MIA", "signed", "Star", 1, None, "")])), board, roster)
    assert "3 of the next 3 best teammates share his position" in ann.iloc[0]["context"]
    ann2 = ti.annotate(ti.events(led([("2026-07-06", "MIA", "signed", "Star", 1, None, "")])), BOARD.assign(position=["F", "G", "G", "G", "G"]), roster)
    assert "share his position" not in ann2.iloc[0]["context"]


def test_departure_context_lists_who_is_left_behind():
    ev = ti.events(led([("2026-07-06", "MIA", "trade_out", "Star", 1, "MIL", ""), ("2026-07-06", "MIL", "trade_in", "Star", 1, "MIA", "")]))
    ann = ti.annotate(ev, BOARD, pd.DataFrame({"player_id": [2, 3], "team_abbr": ["MIL", "MIL"]}))
    assert "MIL best teammates" in ann.iloc[0]["context"] or "left behind at MIL" in ann.iloc[0]["context"]


def test_without_a_board_nothing_is_relevant_and_nothing_crashes():
    ann = ti.annotate(ti.events(led([("2026-07-07", "MIA", "signed", "Star", 1, None, "")])), None, None)
    assert not ann["relevant"].any() and ann["rank"].isna().all()
    assert ti.digest(ann, relevant_only=False)[0].endswith("Star (unranked): to MIA")


def test_digest_lists_relevant_moves_best_first_and_counts_the_rest():
    ev = ti.events(led([("2026-07-07", "MIA", "signed", "Depth", 4, None, ""), ("2026-07-07", "MIA", "signed", "Star", 1, None, "extension"),
                        ("2026-07-07", "MIA", "signed", "Fringe", 5, None, "")]))
    lines = ti.digest(ti.annotate(ev, BOARD, ROSTER), include_context=False)
    assert "Star" in lines[0] and "Depth" in lines[1] and "[extension]" in lines[0]
    assert lines[-1].startswith("(1 other moves") and not any("Fringe" in x for x in lines)


def test_line_never_prints_nan_for_an_unknown_team():
    ev = ti.events(led([("2026-08-20", "DEN", "trade_out", "Star", 1, None, "")]))
    text = ti.digest(ti.annotate(ev, BOARD, None), relevant_only=False)[0]
    assert "nan" not in text and "DEN -> ?" in text


def test_filters():
    ann = ti.annotate(ti.events(led([("2026-07-07", "MIA", "signed", "Star", 1, None, ""), ("2026-07-07", "MIL", "waived", "Depth", 4, None, "")])), BOARD, None)
    assert list(ti.filter_events(ann, teams=["mia"])["person"]) == ["Star"]
    assert list(ti.filter_events(ann, kinds=["waived"])["person"]) == ["Depth"]
    assert list(ti.filter_events(ann, player="sta")["person"]) == ["Star"]
