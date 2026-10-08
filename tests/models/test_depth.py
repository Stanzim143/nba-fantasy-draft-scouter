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


def _external_inputs(tables, *, with_future):
    """Synthetic ``team_transactions`` and ``playoff_logs`` built from the league's own players.

    Legit rows: preseason moves of the target season (before 1 October) and playoffs up to the season before the target.
    Future rows (``with_future``): a move dated after the cutoff, moves of the next season, and playoffs of the target season
    and after, with absurd minutes so any leak would move the projection.
    """
    gl = tables["game_logs"]
    last = gl[gl["season"] == "2017-18"]
    top = last.groupby(["team_id", "player_id"])["min"].sum().reset_index().sort_values(["team_id", "min"], ascending=[True, False])
    tx = []
    for tid, g in top.groupby("team_id"):
        pids = g["player_id"].tolist()
        tx.append(dict(season="2018-19", team_id=tid, player_id=pids[0], direction="out", source_kind="subtraction", txn_date="2018-07-05"))
        if with_future:
            tx.append(dict(season="2018-19", team_id=tid, player_id=pids[1], direction="out", source_kind="subtraction", txn_date="2018-10-20"))
            tx.append(dict(season="2019-20", team_id=tid, player_id=pids[2], direction="out", source_kind="subtraction", txn_date="2019-07-05"))
    tx = pd.DataFrame(tx)
    tx["txn_date"] = pd.to_datetime(tx["txn_date"])
    po = []
    for season, mins in (("2016-17", 24.0), ("2017-18", 24.0)) + ((("2018-19", 60.0), ("2019-20", 60.0)) if with_future else ()):
        for pid in top.groupby("team_id").nth(4)["player_id"].tolist():
            for k in range(7):
                po.append(dict(season=season, game_id=f"{season}-{pid}-{k}", game_date=pd.Timestamp(f"{season[:4]}-04-20") + pd.Timedelta(days=int(k) + 365 * (int(season[:4]) - 2016) * 0), player_id=pid, min=mins))
    return tx, pd.DataFrame(po)


def _write_inputs(root, tx, po):
    (root / "processed").mkdir(parents=True)
    tx.to_parquet(root / "processed" / "team_transactions.parquet", index=False)
    po.to_parquet(root / "processed" / "playoff_logs.parquet", index=False)


def test_future_dated_inputs_do_not_change_any_feature_value():
    """Feature-level leak guard: for every player and every season the features are identical with and without the future rows
    (and non-zero for some players, so the comparison is not vacuous)."""
    tables = make_league()
    h = history_for(tables)
    pids = tables["game_logs"]["player_id"].drop_duplicates().to_numpy()
    mats = {}
    for name, future in (("legit", False), ("with_future", True)):
        tx, po = _external_inputs(tables, with_future=future)
        ctx = DepthContext.build(h, tx, po)
        mats[name] = np.vstack([ctx.for_players(pids, np.full(len(pids), s)) for s in (2017, 2018, 2019)])
    np.testing.assert_array_equal(mats["legit"], mats["with_future"])
    assert (mats["legit"][:, 0] != 0).any() and (mats["legit"][:, 1] != 0).any()


def test_future_dated_transactions_and_playoffs_do_not_change_the_projection(tmp_path, monkeypatch):
    """The real leak guard for the file inputs: rows dated after the cutoff, next season's moves and the target season's own
    playoffs are present in one data dir and absent in another; the projections must be identical. The legit inputs must matter."""
    tables = make_league()
    h = history_for(tables)
    outs = {}
    for name, future in (("legit", False), ("with_future", True)):
        root = tmp_path / name
        _write_inputs(root, *_external_inputs(tables, with_future=future))
        monkeypatch.setenv("NBA_DATA_DIR", str(root))
        outs[name] = BaselineDepthProjector().project(h)
    pd.testing.assert_frame_equal(outs["legit"], outs["with_future"], check_exact=True)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "none"))
    assert not outs["legit"]["proj_mpg"].equals(BaselineDepthProjector().project(h)["proj_mpg"])


def test_assert_projector_ignores_future_passes(tmp_path, monkeypatch):
    from model_testkit import TARGET

    from src.backtest.leakage import assert_projector_ignores_future
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "empty"))
    assert_projector_ignores_future(BaselineDepthProjector(), make_league(), TARGET, seed=3)
