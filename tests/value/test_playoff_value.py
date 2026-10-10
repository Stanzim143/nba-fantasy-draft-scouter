"""Advisory fantasy-playoff value (ADR 0039): team playoff-week games, the availability term, rank invariance, degradation."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_testkit import make_league  # noqa: E402

from src.contracts import History  # noqa: E402
from src.inseason import schedule as S  # noqa: E402
from src.inseason.weeks import derive  # noqa: E402
from src.models.baseline import BaselineProjector  # noqa: E402
from src.store import write_table  # noqa: E402
from src.value import playoff_value as PV  # noqa: E402
from src.value.board import build_board  # noqa: E402

A, B, C = 1610612737, 1610612738, 1610612739


def _sched(rows):
    out = []
    for i, (d, h, a) in enumerate(rows):
        out.append({"season": "s", "game_id": str(i), "scoring_period": None, "game_date": pd.Timestamp(d),
                    "home_team_id": h, "away_team_id": a, "home_abbr": S.ABBR_OF_TEAM_ID[h], "away_abbr": S.ABBR_OF_TEAM_ID[a],
                    "source": "team_games", "time_tbd": False})
    return S._frame(out)


def test_playoff_team_games_counts_only_the_playoff_weeks():
    # week 1 (Mon Oct 19 - Sun Oct 25) is regular, week 2 (Oct 26 - Nov 1) is the playoff week
    sch = _sched([("2026-10-19", A, B), ("2026-10-21", A, C),                          # regular week: ignored
                  ("2026-10-26", A, B), ("2026-10-28", A, C), ("2026-10-30", A, B)])   # playoff week: A 3, B 2, C 1
    cal = derive(date(2026, 10, 19), 2, regular_periods=1)
    g = PV.playoff_team_games(sch, cal)
    assert g.to_dict() == {A: 3.0, B: 2.0, C: 1.0}


def _board():
    return pd.DataFrame({"player_id": [1, 2, 3, 4], "proj_fppg": [40.0, 40.0, 30.0, 20.0], "proj_gp": [82.0, 82.0, 41.0, 90.0]})


def test_more_playoff_games_means_more_playoff_fp_and_unknown_team_is_nan():
    team_of = pd.Series({1: A, 2: B, 3: A, 4: B})
    tg = pd.Series({A: 12.0, B: 10.0})
    o = PV.playoff_overlay(_board(), team_of, tg, season_games=82.0).set_index("player_id")
    assert o.loc[1, "playoff_games"] == 12 and o.loc[2, "playoff_games"] == 10
    assert o.loc[1, "playoff_fp"] == pytest.approx(40 * 12) and o.loc[2, "playoff_fp"] == pytest.approx(40 * 10)
    assert o.loc[1, "playoff_fp"] > o.loc[2, "playoff_fp"]                     # same player, team with more games ranks higher here
    assert o.loc[3, "playoff_avail"] == pytest.approx(0.5) and o.loc[3, "playoff_fp"] == pytest.approx(30 * 12 * 0.5)
    assert o.loc[4, "playoff_avail"] == 1.0                                    # proj_gp above the schedule is clipped, not > 1
    assert o.loc[1, "playoff_team"] == "ATL" and o.loc[2, "playoff_team"] == "BOS"
    o2 = PV.playoff_overlay(_board(), pd.Series({1: A}), tg, season_games=82.0).set_index("player_id")
    assert np.isnan(o2.loc[2, "playoff_games"]) and np.isnan(o2.loc[2, "playoff_fp"]) and o2.loc[2, "playoff_team"] == ""


def test_availability_is_the_models_own_share_of_team_games():
    assert list(PV.per_game_availability([0.0, 41.0, 82.0, 100.0], 82.0)) == [0.0, 0.5, 1.0, 1.0]
    assert np.isnan(PV.per_game_availability([41.0], 0.0)).all()


# --------------------------------------------------------------------------- end to end on a synthetic league

TARGET = "2019-20"


@pytest.fixture(scope="module")
def tables():
    return make_league(n_teams=14, games_per_team=40, seed=8)


@pytest.fixture(scope="module")
def board_and_hist(tables):
    h = History.until(tables, TARGET)
    proj = BaselineProjector().project(h)
    return build_board(proj, h.players, season_games=40), h


@pytest.fixture()
def data_dir(tmp_path, tables, monkeypatch):
    """A data dir with the target season's realised team_games (its schedule) and a short league: 8 regular + 3 playoff weeks."""
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    import copy

    from src.value.league import load_league

    cfg = copy.deepcopy(load_league())
    cfg["league"]["schedule"].update(regular_season_matchups=8, playoff_rounds_weeks=[1, 1, 1])
    monkeypatch.setattr("src.value.league.load_league", lambda *a, **k: cfg)
    # the synthetic schedule is perfectly even; drop two playoff-week games so some teams play fewer (a postponement, say)
    tg = tables["team_games"]
    cal = S.build_calendar(S.from_team_games(tg, TARGET), TARGET, cfg=cfg)
    po = [w for w in cal.weeks if w.kind == "playoff"]
    d = pd.to_datetime(tg["game_date"])
    in_po = (tg["season"] == TARGET) & (d >= pd.Timestamp(po[0].start)) & (d <= pd.Timestamp(po[-1].end))
    drop = tg.loc[in_po, "game_id"].drop_duplicates().iloc[:2]
    write_table(tg[~tg["game_id"].isin(drop)].reset_index(drop=True), "team_games", tmp_path)
    return tmp_path


def test_board_columns_rank_vorp_and_projections_stay_byte_identical(board_and_hist, data_dir):
    board, h = board_and_hist
    before = board.copy()
    overlay, notes = PV.compute_playoff_value(board, h, TARGET, data_dir, season_games=40.0)
    out = PV.attach_playoff_value(board, overlay)
    pd.testing.assert_frame_equal(out[list(before.columns)], before, check_exact=True)
    assert out[list(before.columns)].to_csv(index=False) == before.to_csv(index=False)
    pd.testing.assert_frame_equal(board, before, check_exact=True)        # the input board itself is not mutated
    assert out.attrs == before.attrs
    assert set(PV.PLAYOFF_COLUMNS) <= set(out.columns)
    assert any("weeks 9-11" in n for n in notes)
    # teams really differ in playoff-week games, and playoff_fp follows them exactly
    known = out["playoff_games"].notna()
    assert known.mean() > 0.9 and out.loc[known, "playoff_games"].nunique() > 1
    exp = out["proj_fppg"] * out["playoff_games"] * np.clip(out["proj_gp"] / 40.0, 0, 1)
    np.testing.assert_allclose(out["playoff_fp"], exp)                      # NaN exactly where the team is unknown


def test_player_on_team_with_more_playoff_games_gets_more(board_and_hist, data_dir):
    board, h = board_and_hist
    out = PV.attach_playoff_value(board, PV.compute_playoff_value(board, h, TARGET, data_dir, season_games=40.0)[0])
    hi, lo = out["playoff_games"].max(), out["playoff_games"].min()
    a = out[out["playoff_games"] == hi].iloc[0]
    b = out[out["playoff_games"] == lo].iloc[0]
    assert a["playoff_fp"] / max(a["proj_fppg"] * a["playoff_avail"], 1e-9) == pytest.approx(hi)
    assert b["playoff_fp"] / max(b["proj_fppg"] * b["playoff_avail"], 1e-9) == pytest.approx(lo)


def test_roster_snapshot_team_wins_over_last_season(board_and_hist, data_dir, monkeypatch):
    board, h = board_and_hist
    pid = int(board["player_id"].iloc[0])
    snap = pd.DataFrame({"season": [TARGET], "snapshot_date": [pd.Timestamp("2019-10-01")], "player_id": [pid], "team_id": [C]})
    monkeypatch.setattr("src.ingest.nba_incoming.read_roster_snapshots", lambda *_a, **_k: snap)
    team, note = PV.current_teams(h, TARGET, data_dir)
    assert team[pid] == C and note is None
    assert len(team) > 1                                         # everyone else falls back to last season's team


def test_missing_schedule_degrades_to_a_note(board_and_hist, tmp_path, monkeypatch):
    board, h = board_and_hist
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    overlay, notes = PV.compute_playoff_value(board, h, TARGET, tmp_path, season_games=40.0)
    out = PV.attach_playoff_value(board, overlay)
    assert out["playoff_games"].isna().all() and out["playoff_fp"].isna().all()
    assert any("no schedule" in n for n in notes)
