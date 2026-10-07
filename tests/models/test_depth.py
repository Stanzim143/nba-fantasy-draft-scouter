"""Departed-minutes / playoff-uplift minutes layer (ADR 0035): feature rules, leakage and fallback."""
import numpy as np
import pandas as pd
from model_testkit import history_for, make_league

from src.features.depth import DepthContext, DepthFeatures
from src.models.baseline import BaselineProjector
from src.models.depth_baseline import BaselineDepthProjector
from src.models.registry import get_projector


class _H:
    """The two History fields DepthContext reads."""
    def __init__(self, game_logs, team_games, target_season="2025-26"):
        self.game_logs, self.team_games, self.target_season = game_logs, team_games, target_season


def _hist(rows, teams=(10, 20), games=10, season="2024-25"):
    gl = pd.DataFrame([{"season": season, "player_id": p, "team_id": t, "game_id": f"{p}-{i}", "min": m}
                       for p, t, m in rows for i in range(games)])
    tg = pd.DataFrame([{"season": season, "team_id": t, "game_id": f"{t}-{i}"} for t in teams for i in range(games)])
    return _H(gl, tg)


def _txn(rows):
    return pd.DataFrame([{"season": "2025-26", "team_id": t, "player_id": p, "direction": d, "source_kind": "x",
                          "txn_date": pd.Timestamp(day)} for t, p, d, day in rows])


def test_open_net_is_departed_minus_arrived_for_stayers_in_the_rotation_range_only():
    h = _hist([(1, 10, 20.0), (2, 10, 30.0), (3, 10, 6.0), (4, 20, 10.0), (5, 10, 35.0)])
    tx = _txn([(10, 2, "out", "2025-07-05"), (10, 4, "in", "2025-07-06")])
    f = DepthContext.build(h, tx, None).for_players(np.array([1, 3, 5, 4]), np.array([2025] * 4))
    assert f[0, 0] == (30 - 10) / 10          # stayer at 20 mpg: Joe-like 30 left, 10 arrived
    assert f[1, 0] == 0                        # 6 mpg: below the rotation range
    assert f[2, 0] == 0                        # 35 mpg: a star is not the one helped
    assert f[3, 0] == 0                        # the arrival changed team: his role is not this signal


def test_transactions_after_the_october_cutoff_are_ignored():
    h = _hist([(1, 10, 20.0), (2, 10, 30.0)])
    tx = _txn([(10, 2, "out", "2025-10-20")])
    assert DepthContext.build(h, tx, None).for_players(np.array([1]), np.array([2025]))[0, 0] == 0


def _po(pid, mins, season="2024-25"):
    return pd.DataFrame({"season": season, "player_id": pid, "game_id": [f"{pid}-{i:02d}" for i in range(len(mins))],
                         "game_date": pd.date_range("2025-04-20", periods=len(mins)), "min": mins})


def test_playoff_uplift_uses_the_closing_run_and_ignores_thin_or_flat_samples_and_the_target_season():
    h = _hist([(1, 10, 18.0), (2, 10, 18.0), (3, 10, 18.0), (4, 10, 18.0)])
    po = pd.concat([_po(1, [10, 12, 14, 12, 28, 28, 26, 30, 28]), _po(2, [12, 28, 28]), _po(3, [18] * 8),
                    _po(4, [40] * 8, season="2025-26")])
    f = DepthContext.build(h, None, po).for_players(np.array([1, 2, 3, 4]), np.array([2025] * 4))
    assert abs(f[0, 1] - 1.0) < 1e-9     # last five average 28 against 18 in the regular season, in tens
    assert list(f[1:, 1]) == [0, 0, 0]


def test_fit_falls_back_to_zero_with_too_few_rows_and_build_never_exceeds_the_cap():
    h = _hist([(1, 10, 20.0), (2, 10, 30.0)])
    z = DepthFeatures.fit(h, None, None, np.array([1]), np.array([2025]), np.array([20.0]), np.array([21.0]), np.array([10.0]))
    assert (z.beta == 0).all() and z.build(np.array([1]), np.array([2025]))[0] == 0
    big = DepthFeatures(DepthContext.build(h, _txn([(10, 2, "out", "2025-07-05")]), None), np.array([0.0, 1000.0, 0.0]))
    assert big.build(np.array([1]), np.array([2025]))[0] == 4.0


def test_projector_is_registered_and_identical_to_baseline_without_the_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "empty"))
    h = history_for(make_league())
    assert isinstance(get_projector("baseline_depth"), BaselineDepthProjector)
    a = BaselineProjector().project(h).drop(columns=["model"])
    b = BaselineDepthProjector().project(h).drop(columns=["model"])
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_assert_projector_ignores_future_passes(tmp_path, monkeypatch):
    from model_testkit import TARGET

    from src.backtest.leakage import assert_projector_ignores_future
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "empty"))
    assert_projector_ignores_future(BaselineDepthProjector(), make_league(), TARGET, seed=3)
