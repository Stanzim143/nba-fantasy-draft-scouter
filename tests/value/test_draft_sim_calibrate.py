"""The ADR 0040 calibration harness on a synthetic past draft (no network, no data dir)."""
from __future__ import annotations

import pandas as pd

from src.value import draft_sim_calibrate as C


def _draft(season, n=60, picks=40):
    adp = pd.DataFrame({"source_id": [str(i) for i in range(n)], "adp": [float(i + 1) for i in range(n)], "adp_source": "espn"})
    order = list(range(picks))
    order[3], order[9] = order[9], order[3]          # one reach, one faller
    p = pd.DataFrame({"overall": range(1, picks + 1), "team_id": [i % 10 for i in range(picks)],
                      "espn_id": [str(i) for i in order], "auto": 0})
    return C.Draft(season, 10, p, adp)


def test_predictions_outcomes_and_loso_run(monkeypatch):
    d = _draft(2022)
    pr = C.predictions(d, C.OLD_DEFAULT, n_sims=50)
    assert {"p", "y", "k", "c"} <= set(pr.columns) and pr["p"].between(0, 1).all()
    first = pr[(pr["c"] == 1) & (pr["k"] == 3)]
    assert first.sort_values("market")["y"].tolist()[:4] == [0, 0, 0, 1]   # ADP 1-3 gone after 3 picks, ADP 4 (the faller) not
    monkeypatch.setattr(C, "GRID", {"sd_min": (2.0,), "sd_frac": (0.2, 0.5), "sd_max": (25.0,), "skip": (0.0,)})
    res = C.run([d, _draft(2023)], n_sims=50)
    assert len(res["loso"]) == 2 and {"brier_fit", "brier_old_default", "brier_constant"} <= set(res["loso"].columns)
    assert res["never"]["never_drafted_top_n"].tolist() == [0, 0]
