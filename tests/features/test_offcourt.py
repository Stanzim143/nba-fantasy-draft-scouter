"""Off-court flags (injury-report reasons, general-news keywords) and the hand-written notes file (ADR 0037)."""
from datetime import datetime

import pandas as pd
import pytest

from src.features import offcourt as OC
from src.features import risk as R
from src.ingest import espn_news, espn_offcourt

T0 = datetime(2026, 10, 9, 8, 0)


def reports(rows):
    return pd.DataFrame(rows, columns=["season", "game_date", "player_id", "status", "category", "detail"])


def test_report_kind_separates_off_court_from_everything_else():
    k = OC.report_kind
    assert k("Personal Reasons", "") == "personal reasons" and k("", "Personal Reasons") == "personal reasons"
    assert k("Not With Team", "") == "not with team" and k("Trade Pending", "") == "trade pending"
    assert k("League Suspension", "") == "suspension" and k("Team Suspension -", "") == "suspension"
    assert k("Not With Team", "Self Isolating") == "" and k("Not with Team", "Return to Competition Reconditioning") == ""
    assert k("Injury/Illness", "Left Knee; Patellar Tendinopathy") == "" and k("G League - Two-Way", "") == ""


def test_injury_report_flag_counts_distinct_games_last_season_only_and_applies_a_floor():
    rows = [("2025-26", f"2026-01-{d:02d}", 1, "Out", "Personal Reasons", "") for d in range(1, 8)]
    rows += [("2025-26", "2026-01-01", 1, "Out", "Personal Reasons", "")]                      # duplicate slot, same game
    rows += [("2025-26", "2026-02-01", 2, "Out", "Personal Reasons", ""), ("2025-26", "2026-02-02", 2, "Out", "Personal Reasons", "")]
    rows += [("2025-26", "2026-02-03", 3, "Questionable", "Personal Reasons", "")] * 5         # not Out
    rows += [("2025-26", f"2026-03-{d:02d}", 4, "Out", "Injury/Illness", "Right Calf; Strain") for d in range(1, 9)]
    rows += [("2024-25", f"2025-01-{d:02d}", 5, "Out", "Not With Team", "") for d in range(1, 6)]   # older season
    f = OC.injury_report_flags([1, 2, 3, 4, 5, 6], reports(rows), "2026-27").set_index("player_id")["report_flag"]
    assert f[1] == "7 games out for personal reasons in 2025-26"
    assert f[[2, 3, 4, 5, 6]].tolist() == [""] * 5
    assert OC.injury_report_flags([1], reports(rows), "2025-26")["report_flag"].iloc[0] == ""    # 2025-26 is not a prior season


def article(aid, headline, athletes, desc="", published="2026-09-12T10:00Z"):
    return {"id": aid, "headline": headline, "description": desc, "published": published,
            "links": {"web": {"href": f"https://espn.test/{aid}"}},
            "categories": [{"type": "league"}] + [{"type": "athlete", "athleteId": i, "description": n} for i, n in athletes]}


ID_MAP = pd.DataFrame({"source": ["espn"] * 3, "source_id": ["11", "22", "33"], "player_id": [1, 2, 3]})


def test_attribution_rule_keeps_few_athlete_articles_and_headline_names_of_big_ones():
    many = [(str(100 + i), f"Player {i}") for i in range(10)] + [("11", "Star One")]
    feed = {"articles": [article(1, "Star One requests trade", [("11", "Star One")]),
                         article(2, "Draft board ft. Star One", many),
                         article(3, "Nobody listed", []), {"id": 4, "headline": ""}]}
    df = espn_offcourt.parse_feed(feed, T0, ID_MAP)
    assert sorted(zip(df["article_id"], df["espn_id"])) == [("1", "11"), ("2", "11")]
    assert df["player_id"].tolist() == [1, 1] and sorted(df["n_athletes"]) == [1, 11]


def test_news_keywords_match_the_target_stories_and_not_routine_news():
    feed = {"articles": [
        article(1, "Star One requests a trade", [("11", "Star One")]),
        article(2, "Star Two skipping training camp in contract dispute", [("22", "Star Two")]),
        article(3, "Shams: Star Three contract negotiations a growing cloud", [("33", "Star Three")], published="2026-10-05T10:00Z"),
        article(4, "Star One scores 30 in preseason", [("11", "Star One")]),
        article(5, "Star Two out for personal reasons", [("22", "Star Two")], published="2026-06-01T10:00Z")]}
    news = espn_offcourt.parse_feed(feed, T0, ID_MAP)
    f = OC.news_flags([1, 2, 3, 9], news, today="2026-10-09").set_index("player_id")["offcourt_news_flag"]
    assert f[1].startswith("off-court news: trade request, Sep 12 (") and "Star One requests a trade" in f[1] and "scores 30" not in f[1]
    assert "holdout" in f[2] and "contract dispute" in f[2] and "leave of absence" not in f[2]    # June article is past the window
    assert f[3].startswith("off-court news: contract dispute, Oct 05") and f[9] == ""


def test_combined_flag_joins_report_and_news_parts():
    rows = [("2025-26", f"2026-01-{d:02d}", 1, "Out", "Not With Team", "") for d in range(1, 5)]
    news = espn_offcourt.parse_feed({"articles": [article(1, "Star One wants out", [("11", "Star One")])]}, T0, ID_MAP)
    f = OC.offcourt_flags([1, 2], reports(rows), news, "2026-27", today="2026-10-09")
    assert f["offcourt_flag"].iloc[0].startswith("4 games out for not with team in 2025-26; off-court news: trade request")
    assert f["offcourt_flag"].iloc[1] == ""
    assert OC.offcourt_flags([1], None, None, "2026-27", today="2026-10-09")["offcourt_flag"].tolist() == [""]


def test_merge_keeps_history_and_flags_do_not_touch_projections_or_risk():
    first = espn_offcourt.parse_feed({"articles": [article(1, "A wants out", [("11", "A")])]}, T0, ID_MAP)
    m, _ = espn_news.merge_news(None, first, key=espn_offcourt.KEY, columns=espn_offcourt.COLUMNS)
    again = espn_offcourt.parse_feed({"articles": [article(1, "A wants out", [("11", "A")]),
                                                   article(2, "B holdout", [("22", "B")])]}, datetime(2026, 10, 9, 9), ID_MAP)
    m, fresh = espn_news.merge_news(m, again, key=espn_offcourt.KEY, columns=espn_offcourt.COLUMNS)
    assert len(m) == 2 and fresh["article_id"].tolist() == ["2"]
    assert m[m["article_id"] == "1"]["first_seen"].iloc[0] == pd.Timestamp(T0)
    board = pd.DataFrame({"player_id": [1, 2], "proj_gp": [60.0, 60.0], "proj_fppg": [30.0, 30.0]})
    off = OC.offcourt_flags([1, 2], None, m, "2026-27", today="2026-10-09")
    notes = pd.DataFrame({"player_id": [1, 2], "note_flag": ["note: hi", ""]})
    o = R.build_overlay(board, offcourt=off, notes=notes, today="2026-10-09")
    assert o["risk_gp"].tolist() == [60.0, 60.0] and o["risk_level"].tolist() == ["", ""] and o["proj_fppg"].tolist() == [30.0, 30.0]
    assert "trade request" in o["risk_flags"].iloc[0] and "note: hi" in o["risk_flags"].iloc[0] and "holdout" in o["risk_flags"].iloc[1]


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, label, url, params=None, *, headers=None, refresh=False):
        assert refresh
        return self.payload


def test_pull_writes_and_rejects_a_changed_feed_shape(tmp_path):
    c = FakeClient({"articles": [article(1, "A wants out", [("11", "A")])]})
    r = espn_offcourt.run_pull(c, tmp_path, now=T0)
    assert (r["new"], r["total"]) == (1, 1) and len(espn_offcourt.read_offcourt(tmp_path)) == 1
    with pytest.raises(Exception, match="shape may have changed"):
        espn_offcourt.run_pull(FakeClient({"oops": 1}), tmp_path)


INJ = {"injuries": [{"displayName": "T", "injuries": [{"id": "1", "athlete": {"displayName": "A", "id": 11}, "status": "Out",
                                                        "details": {}}]}]}


def test_injury_news_pull_also_archives_offcourt_news_and_survives_its_failure(tmp_path, capsys):
    class Good:
        def get_json(self, label, url, params=None, *, headers=None, refresh=False):
            return {"articles": [article(1, "A wants out", [("11", "A")])]} if label == "offcourt_news" else INJ

    class Broken:
        def get_json(self, label, url, params=None, *, headers=None, refresh=False):
            return {"nope": 1} if label == "offcourt_news" else INJ

    assert espn_news.main(["--data-dir", str(tmp_path)], client=Good()) == 0
    assert len(espn_offcourt.read_offcourt(tmp_path)) == 1
    capsys.readouterr()
    assert espn_news.main(["--data-dir", str(tmp_path)], client=Broken()) == 0
    assert "off-court news pull failed" in capsys.readouterr().err


def test_notes_file_matches_names_ignoring_accents_and_reports_problems(tmp_path):
    f = tmp_path / "notes.yaml"
    f.write_text("notes:\n  - player: nikola jokic\n    note: fine\n    date: 2026-10-09\n  - player: Nobody\n    note: x\n"
                 "  - player_id: 2\n    note: by id\n  - player: Same Name\n    note: dup\n  - player: Giannis\n", encoding="utf-8")
    names = pd.Series({1: "Nikola Jokić", 2: "Other Guy", 3: "Same Name", 4: "Same Name"})
    frame, problems = OC.note_flags([1, 2, 3, 4], names, OC.read_notes(f))
    assert frame["note_flag"].tolist() == ["note: fine (2026-10-09)", "note: by id", "", ""]
    assert len(problems) == 3 and any("Nobody" in p for p in problems) and any("ambiguous" in p for p in problems)


def test_shipped_notes_file_parses_and_missing_or_malformed_files_behave(tmp_path):
    assert isinstance(OC.read_notes(), list)
    assert OC.read_notes(tmp_path / "absent.yaml") == []
    (tmp_path / "bad.yaml").write_text("notes: 5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        OC.read_notes(tmp_path / "bad.yaml")
