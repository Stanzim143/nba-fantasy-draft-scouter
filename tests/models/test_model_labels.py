"""Labelled-injury projectors (ADR 0033): registry, contract, degradation without reports, use of past reports, no future."""
import numpy as np
import pandas as pd
import pytest
from model_testkit import TARGET, history_for, make_league

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import Projector, season_start, validate_table
from src.models.adp_baseline import BaselineLabelsProjector
from src.models.baseline import BaselineProjector
from src.models.registry import get_projector

EMPTY = pd.DataFrame({"season": pd.Series([], dtype=object), "game_date": pd.Series([], dtype="datetime64[ms]"),
                      "player_id": pd.array([], dtype="Int64"), "status": pd.Series([], dtype=object),
                      "category": pd.Series([], dtype=object), "detail": pd.Series([], dtype=object),
                      "team_abbr": pd.Series([], dtype=object)})


def make_reports(tables, seed=0, frac=0.12):
    """One report per team game date; a random ``frac`` of each team's rotation is listed Out with an injury reason."""
    rng = np.random.default_rng(seed)
    tg = tables["team_games"]
    gl = tables["game_logs"]
    per = gl.groupby(["season", "team_abbr"])["player_id"].agg(lambda x: sorted(set(x)))
    rows = []
    injured = {}
    for (season, abbr), pids in per.items():
        for pid in pids:
            if rng.random() < 0.4:
                injured[(season, abbr, pid)] = (int(rng.integers(0, 40)), int(rng.integers(3, 30)))   # start game, length
    dates = {(s, a): sorted(g["game_date"]) for (s, a), g in tg.groupby(["season", "team_abbr"])}
    for (season, abbr, pid), (start, length) in injured.items():
        d = dates[(season, abbr)]
        for k in range(start, min(start + length, len(d))):
            rows.append((season, pd.Timestamp(d[k]), pid, "Out", "Injury/Illness", "Left Knee; Sprain", abbr))
    # make every game date "covered" with a filler Available row
    for (season, abbr), d in dates.items():
        for gd in d:
            rows.append((season, pd.Timestamp(gd), -1, "Available", "", "", abbr))
    df = pd.DataFrame(rows, columns=["season", "game_date", "player_id", "status", "category", "detail", "team_abbr"])
    df["player_id"] = pd.array(df["player_id"], dtype="Int64")
    df["game_date"] = df["game_date"].astype("datetime64[ms]")
    return df


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def reports(tables):
    return make_reports(tables)


@pytest.fixture(scope="module")
def hist(tables):
    return history_for(tables)


def test_registry_and_contract(hist, reports):
    for name in ("baseline_labels", "baseline_hurdle_labels", "baseline_hurdle_adp_labels"):
        p = get_projector(name, injury_reports=reports, adp=pd.DataFrame({"season": [], "player_id": [], "adp": []}))
        assert isinstance(p, Projector) and p.name == name
        validate_table(p.project(hist), "projections")
    assert get_projector("baseline_hurdle_labels").config.appearance_hurdle


def test_without_reports_it_is_the_base_model(hist):
    base = BaselineProjector().project(hist)
    out = BaselineLabelsProjector(injury_reports=EMPTY).project(hist)
    num = [c for c in base.columns if c != "model"]
    assert np.allclose(out[num].select_dtypes("number").to_numpy(), base[num].select_dtypes("number").to_numpy(),
                       atol=1e-6, equal_nan=True)


def test_reports_change_availability_but_not_the_rates(hist, reports):
    base = BaselineProjector().project(hist).set_index("player_id")
    out = BaselineLabelsProjector(injury_reports=reports).project(hist).set_index("player_id")
    assert np.allclose(out["proj_fppg"], base.loc[out.index, "proj_fppg"])        # only availability moves
    assert not np.allclose(out["proj_gp"], base.loc[out.index, "proj_gp"])


def test_reports_of_the_target_season_and_later_are_ignored(tables, hist, reports):
    ref = BaselineLabelsProjector(injury_reports=reports).project(hist)
    future = reports["season"].map(season_start) >= season_start(TARGET)
    assert future.any()
    shuffled = reports.copy()
    shuffled.loc[future, "status"] = "Out"
    shuffled.loc[future, "detail"] = "Right Ankle; Sprain"
    pd.testing.assert_frame_equal(ref, BaselineLabelsProjector(injury_reports=shuffled).project(hist))


def test_labelled_projector_ignores_the_future_of_the_data(tables, reports):
    t = dict(tables)
    t["injury_reports"] = reports
    assert_projector_ignores_future(get_projector("baseline_hurdle_labels"), t, TARGET)
