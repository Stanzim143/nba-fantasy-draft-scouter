"""Advisory return-from-absence flag (ADR 0023): cohort membership, the advisory rule, point-in-time, projections unchanged,
graceful degradation. Every number the flag touches is display-only; the tests prove the board's own columns never move."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from model_testkit import box_line, make_league  # noqa: E402
from src.backtest import return_report as R  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import History  # noqa: E402
from src.models.baseline import BaselineProjector  # noqa: E402
from src.value import return_flag as RF  # noqa: E402
from src.value.board import build_board  # noqa: E402

TEAM, OTHER = 1610612737, 1610612738
L = 40
PRIOR, TARGET = "2018-19", "2019-20"


def _pat(*spans):
    a = np.zeros(L, bool)
    for s, e, v in spans:
        a[s:e] = v
    return a


def _tables(patterns, *, seasons=("2017-18", "2018-19", "2019-20"), from_year=None):
    """One team, ``L`` games a season; ``patterns[(pid, season)]`` = bool array of games the player appeared in."""
    tg, gl = [], []
    for season in seasons:
        y = int(season[:4])
        for g in range(L):
            tg.append(dict(season=season, game_id=f"{season}-{g:03d}", game_date=pd.Timestamp(y, 10, 20) + pd.Timedelta(days=2 * g),
                           team_id=TEAM, team_abbr="AAA", is_home=True, pts_for=100, pts_against=99))
    tg = pd.DataFrame(tg)
    for (pid, season), present in patterns.items():
        for g in np.flatnonzero(present):
            row = tg[tg["season"] == season].iloc[g]
            gl.append(dict(season=season, game_id=row["game_id"], game_date=row["game_date"], player_id=pid, player_name=f"P{pid}",
                           team_id=TEAM, team_abbr="AAA", matchup="AAA vs. X", plus_minus=0.0, **box_line(28.0)))
    pids = sorted({p for p, _ in patterns})
    frm = from_year or {}
    players = pd.DataFrame({"player_id": pids, "player_name": [f"P{p}" for p in pids], "birthdate": pd.Timestamp("1995-01-01"),
                            "position": "F", "height_in": 78.0, "weight_lb": 210.0, "draft_year": 2015, "draft_round": 1,
                            "draft_number": 5, "from_year": [frm.get(p, 2015) for p in pids], "to_year": 2030})
    bio = pd.DataFrame([dict(season=s, player_id=p, age_at_season_start=24.0, team_id=TEAM) for (p, s) in patterns])
    return {"game_logs": pd.DataFrame(gl), "team_games": tg, "players": players, "player_season_bio": bio}


FULL = np.ones(L, bool)
_poor_tail = _pat((25, 40, True))
_poor_tail[26:33] = False
# pid: prior-season (2018-19) pattern
CASES = {
    1: _pat((25, 40, True)),                          # Tatum-like: out 25 of 40 (62.5%), then plays 15 of 15 -> flag, +5 GP
    2: np.tile([True, True, False], 14)[:L],          # spread absence, no lead block -> no flag
    3: _pat((25, 40, True)),                          # debutant (from_year 2018): a lead block is not an absence -> no flag
    4: _pat((25, 40, True)),                          # traded mid-season (games relabelled below) -> no flag
    5: FULL,                                          # no prior season (only appears in 2017-18, see below) -> no flag
    6: _pat((0, 10, True), (30, 40, True)),           # interior block, not a lead block -> no flag
    7: _pat((8, 40, True)),                           # lead block 20% < 25% -> no flag
    8: _pat((12, 40, True)),                          # lead block 30%, tail 28 of 28 -> flag, block < 60% so 0 advisory GP
    9: _pat((24, 40, True)),                          # lead block exactly 60% (24/40), tail 16 of 16 -> flag, +5 GP (boundary)
    10: _poor_tail,                                   # like 1, but the tail is only 8 of 15 games (53%) -> no flag
}


@pytest.fixture(scope="module")
def tables():
    pats = {}
    for pid, a in CASES.items():
        if pid == 5:
            pats[(pid, "2017-18")] = FULL
        else:
            pats[(pid, PRIOR)] = a
            if pid != 3:
                pats[(pid, "2017-18")] = FULL
        pats[(pid, TARGET)] = FULL                    # the target season only supplies "the future"; must never be read
    t = _tables(pats, from_year={3: 2018})
    gl = t["game_logs"]
    mask = (gl["player_id"] == 4) & (gl["season"] == PRIOR) & (gl["game_id"].str[-3:].astype(int) < 30)
    gl.loc[mask, "team_id"] = OTHER                   # 5 of his 15 games for another team: n_teams == 2
    return t


def _board(pids, *, fppg=40.0, gp=30.0):
    n = len(pids)
    return pd.DataFrame({"rank": range(1, n + 1), "player_id": pids, "name": [f"P{p}" for p in pids], "position": "F",
                         "proj_fppg": fppg, "proj_gp": gp, "proj_total_fp": fppg * gp, "vorp": np.linspace(900, 100, n),
                         "vorp_per_game": 5.0, "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0, "tier": 1,
                         "risk_level": ["watch"] + [""] * (n - 1), "risk_flags": ["ESPN day-to-day"] + [""] * (n - 1),
                         "risk_gp_haircut": 0.0, "risk_gp": gp})


def _run(tables, season=TARGET, pids=None, **kw):
    h = History.until(tables, season)
    b = _board(pids if pids is not None else sorted(CASES), **kw)
    o, notes = RF.compute_return_flag(b, h, season_games=float(L))
    return b, RF.attach_return_flag(b, o), o, notes


# ------------------------------------------------------------------ who is flagged

def test_flag_appears_for_the_tatum_like_player_and_not_for_the_look_alikes(tables):
    _, out, _, notes = _run(tables)
    flagged = set(out.loc[out["return_flag"] != "", "player_id"])
    assert flagged == {1, 8, 9}, "spread absence, debutant, traded, no-prior-season, interior block, short block, poor tail: no flag"
    r = out.set_index("player_id")
    assert r.loc[1, "return_flag"] == RF.FLAG_LABEL and r.loc[1, "return_tail"] == "15/15"
    assert r.loc[1, "return_block_pct"] == pytest.approx(62.5) and r.loc[8, "return_tail"] == "28/28"
    for pid in (2, 3, 4, 5, 6, 7, 10):
        assert r.loc[pid, "return_tail"] == "" and np.isnan(r.loc[pid, "return_block_pct"]) and r.loc[pid, "return_gp_upside_adv"] == 0.0
    assert any("3 of 10" in n and "advisory judgement, not a projection" in n for n in notes)


def test_flag_agrees_with_the_adr_0021_cohort_functions(tables):
    h = History.until(tables, TARGET)
    feat = R.prior_features(h)
    feat["season"] = TARGET
    cohort = set(feat.loc[R.in_cohort(feat, R.PRIMARY), "player_id"])
    assert cohort == set(RF.return_profile(h)["player_id"]) == {1, 8, 9}


def test_advisory_rule_is_five_games_from_a_sixty_percent_block_capped_at_the_schedule(tables):
    _, out, _, _ = _run(tables)
    r = out.set_index("player_id")
    assert r.loc[1, "return_gp_upside_adv"] == 5.0 and r.loc[9, "return_gp_upside_adv"] == 5.0    # 62.5% and exactly 60%
    assert r.loc[8, "return_gp_upside_adv"] == 0.0 and r.loc[8, "return_fp_upside_adv"] == 0.0     # 30% block: flagged, no advisory games
    assert r.loc[1, "return_fp_upside_adv"] == pytest.approx(5.0 * 40.0)                           # games x proj_fppg = total FP
    # capped so proj_gp + advisory never passes the schedule
    near = _run(tables, gp=float(L) - 2.0)[1].set_index("player_id")
    assert near.loc[1, "return_gp_upside_adv"] == 2.0
    full = _run(tables, gp=float(L))[1].set_index("player_id")
    assert full.loc[1, "return_gp_upside_adv"] == 0.0
    assert RF.advisory_gp_upside(np.array([0.59, 0.6, 0.9, np.nan]), np.full(4, 30.0)).tolist() == [0.0, 5.0, 5.0, 0.0]


def test_readable_flag_is_appended_to_risk_flags_and_names_the_numbers(tables):
    b, out, _, _ = _run(tables)
    r = out.set_index("player_id")["risk_flags"]
    assert r[1] == ("ESPN day-to-day; returned from long absence (missed the first 25 of 40 team games), healthy since: played 15 of the "
                    "last 15 team games; advisory +5 GP (a judgement, not in the projection)")
    assert "advisory" not in r[8] and "missed the first 12 of 40" in r[8]
    unflagged = out["return_flag"] == ""
    assert (out.loc[unflagged, "risk_flags"] == b.loc[unflagged, "risk_flags"]).all()
    assert (out[RF.VALIDATION_COLUMN] == RF.VALIDATION_LABEL).all()


# ------------------------------------------------------------------ projections unchanged

PROTECTED = ["rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp", "vorp", "vorp_per_game", "fppg_p10",
             "fppg_p50", "fppg_p90", "tier", "risk_level", "risk_gp_haircut", "risk_gp"]


def test_attaching_the_flag_leaves_every_projection_rank_and_risk_column_byte_identical(tables):
    b, out, _, _ = _run(tables)
    assert list(out.columns[:len(b.columns)]) == list(b.columns) and len(out.columns) == len(b.columns) + 6
    assert out[PROTECTED].to_csv(index=False) == b[PROTECTED].to_csv(index=False)
    pd.testing.assert_frame_equal(out[PROTECTED], b[PROTECTED], check_exact=True)
    assert out["player_id"].tolist() == b["player_id"].tolist()


def test_real_pipeline_board_with_and_without_the_flag_is_identical_on_every_original_column():
    lg = make_league()
    h = History.until(lg, "2019-20")
    proj = BaselineProjector().project(h)
    board = build_board(proj, h.players, season_games=60)
    overlay, notes = RF.compute_return_flag(board, h, season_games=60.0)
    flagged = RF.attach_return_flag(board, overlay)
    assert flagged[list(board.columns)].to_csv(index=False) == board.to_csv(index=False)
    assert flagged.attrs.get("replacement") == board.attrs.get("replacement")
    assert (flagged["return_gp_upside_adv"] <= RF.UPSIDE_GP).all() and notes


def test_rank_and_total_fp_stay_the_deciding_columns(tables):
    """The advisory FP is a side column: the board is still ordered by vorp/total FP, not by any return column."""
    b, out, _, _ = _run(tables)
    assert out["vorp"].is_monotonic_decreasing and out["rank"].tolist() == b["rank"].tolist()
    assert (out["proj_total_fp"] == out["proj_fppg"] * out["proj_gp"]).all()


# ------------------------------------------------------------------ point in time

def test_flag_ignores_the_target_season_and_anything_later(tables):
    _, base, _, _ = _run(tables)
    with scrambled_future(tables, TARGET):
        _, scr, _, _ = _run(tables)
    pd.testing.assert_frame_equal(scr, base)
    pats = {(p, s): FULL for p in (1, 2) for s in ("2017-18", "2018-19", "2019-20", "2020-21")}
    pats[(1, PRIOR)] = CASES[1]
    a = _tables({k: v for k, v in pats.items() if k[1] != "2020-21"}, seasons=("2017-18", "2018-19", "2019-20"))
    b = _tables(pats, seasons=("2017-18", "2018-19", "2019-20", "2020-21"))
    fa = RF.compute_return_flag(_board([1, 2]), History.until(a, TARGET), season_games=40.0)[0]
    fb = RF.compute_return_flag(_board([1, 2]), History.until(b, TARGET), season_games=40.0)[0]
    pd.testing.assert_frame_equal(fa, fb)
    assert fa.set_index("player_id").loc[1, "return_flag"] == RF.FLAG_LABEL


def test_flag_at_an_earlier_target_uses_only_the_season_before_it(tables):
    """The same tables read at 2018-19 profile 2017-18 (everyone played every game): nobody is flagged."""
    _, out, _, notes = _run(tables, season=PRIOR, pids=[1, 2, 5])
    assert (out["return_flag"] == "").all() and any("0 of 3" in n for n in notes)


# ------------------------------------------------------------------ degradation

def test_empty_history_gives_no_flags_and_a_note_not_an_error():
    h = History(target_season=TARGET,
                game_logs=pd.DataFrame(columns=["season", "game_id", "player_id", "team_id", "min"]),
                team_games=pd.DataFrame(columns=["season", "game_id", "game_date", "team_id"]),
                players=pd.DataFrame(columns=["player_id", "from_year", "to_year", "draft_year"]),
                player_season_bio=pd.DataFrame(columns=["season", "player_id"]))
    b = _board([1, 2])
    o, notes = RF.compute_return_flag(b, h)
    out = RF.attach_return_flag(b, o)
    assert (out["return_flag"] == "").all() and (out["return_gp_upside_adv"] == 0).all() and notes
    assert out[PROTECTED].equals(b[PROTECTED]) and out["risk_flags"].tolist() == b["risk_flags"].tolist()


def test_board_without_projection_columns_or_risk_overlay_still_works(tables):
    h = History.until(tables, TARGET)
    b = _board(sorted(CASES)).drop(columns=["proj_gp", "proj_fppg", "risk_level", "risk_flags", "risk_gp", "risk_gp_haircut"])
    o, notes = RF.compute_return_flag(b, h)
    out = RF.attach_return_flag(b, o)
    assert set(out.loc[out["return_flag"] != "", "player_id"]) == {1, 8, 9}
    assert (out["return_gp_upside_adv"] == 0).all() and "risk_flags" not in out.columns and any("not sized" in n for n in notes)


def test_a_board_player_the_history_has_never_seen_is_simply_unflagged(tables):
    _, out, _, _ = _run(tables, pids=[1, 999999])
    r = out.set_index("player_id")
    assert r.loc[999999, "return_flag"] == "" and r.loc[1, "return_flag"] == RF.FLAG_LABEL


def test_board_cli_with_and_without_the_flag_has_identical_original_columns(tmp_path, monkeypatch, capsys):
    """A failure inside the overlay is a printed note; the CSV is still written, and every pre-existing column is unchanged."""
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

    monkeypatch.setattr(RF, "compute_return_flag", boom)
    assert board_mod.main([*args, "--out", str(without)]) == 0
    assert "return flag unavailable (no game logs)" in capsys.readouterr().out
    a, b = pd.read_csv(with_flag), pd.read_csv(without)
    assert {"return_flag", "return_tail", "return_gp_upside_adv", "return_fp_upside_adv"} <= set(a.columns) and "return_flag" not in b.columns
    assert a[list(b.columns)].equals(b), "every pre-existing column is identical with and without the flag"


# ------------------------------------------------------------------ real data (skipped when the store is not ingested)

def _real_tables():
    from src.contracts import HISTORY_TABLES, table_path
    from src.store import load_tables

    if [t for t in HISTORY_TABLES if not table_path(t).exists()]:
        pytest.skip("real data not ingested")
    return load_tables(HISTORY_TABLES)


def test_real_data_flag_matches_the_adr_0021_current_cohort_and_tatum_numbers():
    tables = _real_tables()
    last = max(int(s[:4]) for s in tables["game_logs"]["season"].unique())
    target = f"{last + 1}-{str(last + 2)[-2:]}"
    h = History.until(tables, target)
    feat = R.prior_features(h)
    feat["season"] = target
    cohort = set(feat.loc[R.in_cohort(feat, R.PRIMARY), "player_id"])
    prof = RF.return_profile(h)
    assert set(prof["player_id"]) == cohort and len(cohort) >= 1
    tatum = h.players[h.players["player_name"] == "Jayson Tatum"]["player_id"]
    if len(tatum) and int(tatum.iloc[0]) in cohort and target == "2026-27":
        r = prof.set_index("player_id").loc[int(tatum.iloc[0])]
        assert (int(r["first_idx"]), int(r["L"]), int(r["gp"]), int(r["tail_len"])) == (62, 82, 16, 20)


def test_nan_proj_gp_gives_zero_advisory_games_not_nan():
    up = RF.advisory_gp_upside(np.array([0.9, 0.9]), np.array([np.nan, 30.0]))
    assert up.tolist() == [0.0, 5.0]


def test_public_veteran_flags_agrees_with_the_private_helper(tables):
    h = History.until(tables, TARGET)
    pids = np.array(sorted(CASES)), np.full(len(CASES), 2018)
    assert R.veteran_flags(h, *pids).tolist() == R._veteran_flags(h, *pids).tolist()
