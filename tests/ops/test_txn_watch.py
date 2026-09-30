"""txn_watch: window selection, the acknowledge watermark, and the CLI end to end on a tiny ledger."""
from datetime import date

import pandas as pd

from src.ingest import espn_transactions as et
from src.ops import txn_watch as tw


def ledger_file(base, rows):
    recs, _ = et.feed_rows_to_records(rows)
    for r in recs:
        r["player_id"] = {"Buddy Hield": 1, "Some Vet": 2}.get(r["person"])
    when = [pd.Timestamp("2026-09-20 10:00", tz="UTC"), pd.Timestamp("2026-09-24 10:00", tz="UTC")]
    frames = [et._frame([r for r in recs if r["txn_date"] < pd.Timestamp("2026-09-22")], when[0].to_pydatetime()),
              et._frame([r for r in recs if r["txn_date"] >= pd.Timestamp("2026-09-22")], when[1].to_pydatetime())]
    merged = pd.concat([f for f in frames if len(f)], ignore_index=True)
    et._atomic_parquet(merged, et.ledger_path(base))
    return merged


FEED = [{"date": "2026-09-19T07:00Z", "description": "Signed G Some Vet to a contract.", "team": {"abbreviation": "MIA"}},
        {"date": "2026-09-23T07:00Z", "description": "Acquired G Buddy Hield from Atlanta.", "team": {"abbreviation": "CHA"}},
        {"date": "2026-05-04T07:00Z", "description": "Fired head coach Jamahl Mosley.", "team": {"abbreviation": "ORL"}}]


def test_window_precedence_and_the_first_run_default(tmp_path):
    led = ledger_file(tmp_path, FEED)
    today = date(2026, 9, 25)
    w, label = tw.select_window(led, days=None, since=None, watermark=None, today=today)
    assert set(w["person"]) == {"Some Vet", "Buddy Hield"} and "nothing acknowledged" in label
    w, _ = tw.select_window(led, days=3, since=None, watermark=None, today=today)
    assert set(w["person"]) == {"Buddy Hield"}
    w, _ = tw.select_window(led, days=3, since=date(2026, 9, 1), watermark=None, today=today)      # --since beats --days
    assert len(w) == 2
    w, label = tw.select_window(led, days=None, since=None, watermark=pd.Timestamp("2026-09-22"), today=today)
    assert set(w["person"]) == {"Buddy Hield"} and label.startswith("new since")


def test_watermark_round_trip_and_garbage_is_ignored(tmp_path):
    assert tw.read_watermark(tmp_path) is None
    tw.write_watermark(tmp_path, pd.Timestamp("2026-09-24 10:00"))
    assert tw.read_watermark(tmp_path) == pd.Timestamp("2026-09-24 10:00")
    tw.watermark_path(tmp_path).write_text("not json", encoding="utf-8")
    assert tw.read_watermark(tmp_path) is None


def test_cli_lists_moves_with_ranks_and_acks(tmp_path, capsys):
    ledger_file(tmp_path, FEED)
    pd.DataFrame({"player_id": [1, 2], "rank": [20, 400], "name": ["Buddy Hield", "Some Vet"], "position": ["G", "G"],
                  "vorp": [900.0, 3.0], "adp": [22.0, float("nan")]}).to_csv(tmp_path / "board.csv", index=False)
    assert tw.main(["--data-dir", str(tmp_path), "--days", "30", "--board-csv", str(tmp_path / "board.csv"), "--ack"]) == 0
    out = capsys.readouterr().out
    assert "TRADE" in out and "Buddy Hield (G, #20): ATL -> CHA" in out
    assert "Some Vet" not in out and "1 other moves" in out                       # outside the board's top ranks: counted, not listed
    assert "acknowledged" in out and tw.read_watermark(tmp_path) is not None
    assert tw.main(["--data-dir", str(tmp_path), "--board-csv", str(tmp_path / "board.csv")]) == 0
    assert "no moves in this window" in capsys.readouterr().out                    # nothing new after the ack


def test_cli_staff_and_player_search(tmp_path, capsys):
    ledger_file(tmp_path, FEED)
    assert tw.main(["--data-dir", str(tmp_path), "--since", "2026-01-01", "--staff"]) == 0
    assert "Jamahl Mosley (head coach)" in capsys.readouterr().out
    pd.DataFrame({"player_id": [2], "rank": [400], "name": ["Some Vet"], "position": ["G"], "vorp": [1.0], "adp": [float("nan")]}).to_csv(tmp_path / "b.csv", index=False)
    assert tw.main(["--data-dir", str(tmp_path), "--since", "2026-01-01", "--player", "vet", "--board-csv", str(tmp_path / "b.csv")]) == 0
    assert "SIGNED" in capsys.readouterr().out                                    # a named search shows unranked players too


def test_cli_without_a_ledger_says_how_to_make_one(tmp_path, capsys):
    assert tw.main(["--data-dir", str(tmp_path)]) == 2
    assert "espn_transactions" in capsys.readouterr().err
