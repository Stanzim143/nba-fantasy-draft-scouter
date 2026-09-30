"""Advisory short-absence flag (ADR 0024): who is flagged, the fixed advisory rule, projections unchanged, point in time, degradation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from load_testkit import L, _pat, _tables  # noqa: E402
from model_testkit import make_league  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import History  # noqa: E402
from src.models.baseline import BaselineProjector  # noqa: E402
from src.value import load_flag as LF  # noqa: E402
from src.value.board import build_board  # noqa: E402

PRIOR, TARGET = "2018-19", "2019-20"
# pid: prior-season missed games (L = 30 team games; the universe needs >= 41 GP so these tests use a lowered bar via monkeypatch)
SHORT6 = [2, 5, 8, 11, 14, 17]                      # six one-game absences
CASES = {
    1: dict(missed=SHORT6),                         # 6 isolated: flag, advisory -2
    2: dict(missed=SHORT6[:5]),                     # 5 isolated: not flagged (just below the threshold)
    3: dict(missed=[2, 3, 6, 7, 10, 11]),           # three 2-game runs = 6 isolated: flag (runs of exactly ISO_MAX count)
    4: dict(missed=[2, 3, 4, 8, 9, 10]),            # two 3-game runs: not short, not flagged
    5: dict(missed=[6, 9, 12, 15, 18, 21], lead=True),            # same but also a lead block: lead games are not interior
    6: dict(missed=SHORT6, traded=True),            # traded mid-season: outside the universe
    7: dict(missed=SHORT6, debut=True),             # debutant: outside the universe
}


@pytest.fixture(autouse=True)
def _small_universe(monkeypatch):
    """The synthetic seasons have 30 team games, so a 41-GP bar would exclude everyone: scale it down for the tests only."""
    monkeypatch.setattr(LF, "MIN_GP", 10)


@pytest.fixture(scope="module")
def tables():
    pats, other, frm = {}, {}, {}
    for pid, c in CASES.items():
        pres = _pat(missed=c["missed"])
        if c.get("lead"):
            pres[:4] = False                           # the four lead games are not part of any counted run
        pats[(pid, PRIOR)] = pres
        pats[(pid, "2017-18")] = _pat()
        pats[(pid, TARGET)] = _pat()                    # the future: must never be read
        if c.get("traded"):
            other.update({(pid, PRIOR, g): 1610612738 for g in range(20, 30)})
        if c.get("debut"):
            frm[pid] = 2018
            del pats[(pid, "2017-18")]
    t = _tables(pats, other_team=other, b2b={2, 5})
    t["players"].loc[t["players"]["player_id"].isin(frm), "from_year"] = 2018
    return t


def _board(pids, *, fppg=40.0, gp=30.0):
    n = len(pids)
    return pd.DataFrame({"rank": range(1, n + 1), "player_id": pids, "name": [f"P{p}" for p in pids], "position": "F", "proj_fppg": fppg,
                         "proj_gp": gp, "proj_total_fp": fppg * gp, "vorp": np.linspace(900, 100, n), "vorp_per_game": 5.0,
                         "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0, "tier": 1,
                         "risk_level": ["watch"] + [""] * (n - 1), "risk_flags": ["ESPN day-to-day"] + [""] * (n - 1),
                         "risk_gp_haircut": 0.0, "risk_gp": gp})


def _run(tables, season=TARGET, pids=None, **kw):
    h = History.until(tables, season)
    b = _board(pids if pids is not None else sorted(CASES), **kw)
    o, notes = LF.compute_load_flag(b, h, season_games=float(L))
    return b, LF.attach_load_flag(b, o), o, notes


def test_flag_appears_only_for_six_or_more_short_interior_absences(tables):
    _, out, _, notes = _run(tables)
    assert set(out.loc[out["lm_flag"] != "", "player_id"]) == {1, 3, 5}
    r = out.set_index("player_id")
    assert r.loc[1, "lm_iso_n"] == 6 and r.loc[2, "lm_iso_n"] == 5 and r.loc[4, "lm_iso_n"] == 0
    assert r.loc[1, "lm_rest_n"] == 2 and r.loc[3, "lm_rest_n"] == 0, "one-game absences on the second night of a back-to-back, information only"
    assert np.isnan(r.loc[6, "lm_iso_n"]) and np.isnan(r.loc[7, "lm_iso_n"]), "traded and debutant seasons are outside the universe"
    assert any("3 of 7" in n and "not a projection" in n for n in notes)


def test_advisory_rule_is_minus_two_games_never_below_zero(tables):
    _, out, _, _ = _run(tables)
    r = out.set_index("player_id")
    assert r.loc[1, "lm_gp_risk_adv"] == -2.0 and r.loc[1, "lm_fp_risk_adv"] == pytest.approx(-80.0)
    assert r.loc[2, "lm_gp_risk_adv"] == 0.0 and r.loc[2, "lm_fp_risk_adv"] == 0.0
    assert _run(tables, gp=1.0)[1].set_index("player_id").loc[1, "lm_gp_risk_adv"] == -1.0
    assert _run(tables, gp=0.0)[1].set_index("player_id").loc[1, "lm_gp_risk_adv"] == 0.0
    assert LF.advisory_gp_risk(np.array([5, 6, 9, np.nan]), np.full(4, 30.0)).tolist() == [0.0, -2.0, -2.0, 0.0]


def test_readable_sentence_is_appended_to_risk_flags_and_says_the_cause_is_unknown(tables):
    b, out, _, _ = _run(tables)
    r = out.set_index("player_id")["risk_flags"]
    assert r[1].startswith("ESPN day-to-day; many short absences last season: missed 6 games in 1-2 game absences (played 24 of 30; 2 were")
    assert "cause unknown, rest or minor injury" in r[1] and "advisory -2 GP (a judgement, not in the projection)" in r[1]
    unflagged = out["lm_flag"] == ""
    assert (out.loc[unflagged, "risk_flags"] == b.loc[unflagged, "risk_flags"]).all()
    assert (out[LF.VALIDATION_COLUMN] == LF.VALIDATION_LABEL).all()


PROTECTED = ["rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp", "vorp", "vorp_per_game", "fppg_p10",
             "fppg_p50", "fppg_p90", "tier", "risk_level", "risk_gp_haircut", "risk_gp"]


def test_attaching_the_flag_leaves_every_projection_rank_and_risk_column_byte_identical(tables):
    b, out, _, _ = _run(tables)
    assert list(out.columns[:len(b.columns)]) == list(b.columns) and len(out.columns) == len(b.columns) + 6
    assert out[PROTECTED].to_csv(index=False) == b[PROTECTED].to_csv(index=False)
    pd.testing.assert_frame_equal(out[PROTECTED], b[PROTECTED], check_exact=True)
    assert (out["proj_total_fp"] == out["proj_fppg"] * out["proj_gp"]).all()


def test_real_pipeline_board_with_and_without_the_flag_is_identical_on_every_original_column():
    lg = make_league()
    h = History.until(lg, "2019-20")
    board = build_board(BaselineProjector().project(h), h.players, season_games=60)
    overlay, notes = LF.compute_load_flag(board, h, season_games=60.0)
    flagged = LF.attach_load_flag(board, overlay)
    assert flagged[list(board.columns)].to_csv(index=False) == board.to_csv(index=False)
    assert flagged.attrs.get("replacement") == board.attrs.get("replacement") and notes
    assert (flagged["lm_gp_risk_adv"] <= 0).all()


def test_flag_ignores_the_target_season_and_anything_later(tables):
    _, base, _, _ = _run(tables)
    with scrambled_future(tables, TARGET):
        _, scr, _, _ = _run(tables)
    pd.testing.assert_frame_equal(scr, base)


def test_flag_at_an_earlier_target_uses_only_the_season_before_it(tables):
    _, out, _, notes = _run(tables, season=PRIOR, pids=[1, 2])      # profiles 2017-18: everybody played every game
    assert (out["lm_flag"] == "").all() and (out["lm_iso_n"] == 0).all() and any("0 of 2" in n for n in notes)


def test_empty_history_gives_no_flags_and_a_note_not_an_error():
    h = History(target_season=TARGET, game_logs=pd.DataFrame(columns=["season", "game_id", "player_id", "team_id", "min"]),
                team_games=pd.DataFrame(columns=["season", "game_id", "game_date", "team_id"]),
                players=pd.DataFrame(columns=["player_id", "from_year", "to_year", "draft_year"]),
                player_season_bio=pd.DataFrame(columns=["season", "player_id"]))
    b = _board([1, 2])
    o, notes = LF.compute_load_flag(b, h)
    out = LF.attach_load_flag(b, o)
    assert (out["lm_flag"] == "").all() and (out["lm_gp_risk_adv"] == 0).all() and notes
    assert out[PROTECTED].equals(b[PROTECTED]) and out["risk_flags"].tolist() == b["risk_flags"].tolist()


def test_board_without_projection_columns_or_risk_overlay_still_works(tables):
    h = History.until(tables, TARGET)
    b = _board(sorted(CASES)).drop(columns=["proj_gp", "proj_fppg", "risk_level", "risk_flags", "risk_gp", "risk_gp_haircut"])
    o, notes = LF.compute_load_flag(b, h)
    out = LF.attach_load_flag(b, o)
    assert set(out.loc[out["lm_flag"] != "", "player_id"]) == {1, 3, 5}
    assert (out["lm_gp_risk_adv"] == 0).all() and "risk_flags" not in out.columns and any("not sized" in n for n in notes)


def test_a_board_player_the_history_has_never_seen_is_simply_unflagged(tables):
    _, out, _, _ = _run(tables, pids=[1, 999999])
    r = out.set_index("player_id")
    assert r.loc[999999, "lm_flag"] == "" and np.isnan(r.loc[999999, "lm_iso_n"]) and r.loc[1, "lm_flag"] == LF.FLAG_LABEL


def test_the_universe_bar_is_the_adr_default_and_matches_the_study():
    from src.backtest import load_report as LR

    import importlib
    fresh = importlib.reload(LF)
    assert (fresh.MIN_MPG, fresh.MIN_GP, fresh.FLAG_MIN_ISO) == (LR.MIN_MPG, LR.MIN_GP, LR.HI_ISO) == (20.0, 41, 6)
    assert fresh.ADVISORY_GP == 2.0


def test_board_cli_writes_the_flag_columns_and_survives_a_failure_in_the_overlay(tmp_path, monkeypatch, capsys):
    from src.store import write_table
    from src.value import board as board_mod

    lg = make_league()
    for name in ("game_logs", "team_games", "players", "player_season_bio"):
        write_table(lg[name], name, tmp_path / "data")
    with_flag, without = tmp_path / "with.csv", tmp_path / "without.csv"
    args = ["--season", "2019-20", "--model", "baseline", "--data-dir", str(tmp_path / "data")]
    assert board_mod.main([*args, "--out", str(with_flag)]) == 0

    def boom(*a, **k):
        raise FileNotFoundError("no game logs")

    monkeypatch.setattr(LF, "compute_load_flag", boom)
    assert board_mod.main([*args, "--out", str(without)]) == 0
    assert "short-absence flag unavailable (no game logs)" in capsys.readouterr().out
    a, b = pd.read_csv(with_flag), pd.read_csv(without)
    assert {"lm_flag", "lm_iso_n", "lm_rest_n", "lm_gp_risk_adv", "lm_fp_risk_adv"} <= set(a.columns) and "lm_flag" not in b.columns
    assert a[list(b.columns)].equals(b), "every pre-existing column is identical with and without the flag"
