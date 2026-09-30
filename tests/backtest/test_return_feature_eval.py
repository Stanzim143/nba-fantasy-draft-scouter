"""Secondary analyses for the return-health feature (ADR 0022): cohort subsetting, sign conventions, determinism."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import return_feature_eval as E


def _rows():
    n = 40
    rng = np.random.default_rng(0)
    return pd.DataFrame({"season": ["2018-19"] * 20 + ["2019-20"] * 20, "player_id": np.arange(n),
                         "proj_gp_base": 40.0, "proj_gp_var": np.r_[np.full(10, 50.0), np.full(30, 40.0)],
                         "proj_total_fp_base": 1000.0, "proj_total_fp_var": np.r_[np.full(10, 1250.0), np.full(30, 1000.0)],
                         "actual_gp": np.r_[np.full(10, 50.0), rng.integers(30, 50, 30)].astype(float),
                         "actual_total_fp": np.r_[np.full(10, 1250.0), rng.integers(800, 1200, 30)].astype(float)})


def test_lift_is_base_error_minus_variant_error_and_restricts_to_the_cohort():
    rows = _rows()
    coh = pd.DataFrame({"season": rows["season"], "player_id": rows["player_id"],
                        "in_primary": rows["player_id"] < 10, "in_block60": rows["player_id"] < 4})
    t = E.lift_table(rows, coh, n_boot=200)
    prim = t[(t["subset"] == "ADR 0021 primary cohort") & (t["metric"] == "MAE games played")].iloc[0]
    assert prim["n"] == 10 and prim["MAE base"] == pytest.approx(10.0) and prim["MAE variant"] == pytest.approx(0.0)
    assert prim["lift"] == pytest.approx(10.0) and prim["bias base"] == pytest.approx(10.0) and prim["bias variant"] == pytest.approx(0.0)
    assert prim["lo"] <= prim["lift"] <= prim["hi"]
    assert set(t["subset"]) == {"all projected", "ADR 0021 primary cohort", "cohort, block >= 60%"}
    assert (t[t["subset"] == "cohort, block >= 60%"]["n"] == 4).all()
    pd.testing.assert_frame_equal(t, E.lift_table(rows, coh, n_boot=200))
