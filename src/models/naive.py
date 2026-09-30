"""NaiveLastSeason: the benchmark every real model has to beat (PLANNING.md section 5).

Projection for season S = what each player did in the last completed season, as-is:

* per-game stats = last season's per-game averages (all eleven scored stats and minutes),
* ``proj_gp`` = last season's games played (no regression, no age, no schedule adjustment),
* floor / median / ceiling = the 10th / 50th / 90th percentile of the player's own game-level FP
  last season.

Players with **no game in the last completed season** (rookies, returners from a full-season
absence, retirees) get **no row**. The naive model has no information about them and this
benchmark does not invent any; a backtest should compare models on the players both cover and
report coverage separately (the baseline projects rookies and ignores retirees).
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.contracts import PROJECTION_STATS, STAT_COLUMN_MAP, History, validate_table
from src.models.panel import canonical_logs, season_starts
from src.value.frame import fantasy_points_frame
from src.value.league import load_league


class NaiveLastSeason:
    name = "naive_last_season"

    def __init__(self, scoring: Mapping[str, float] | None = None, name: str | None = None):
        self.scoring: Mapping[str, float] = dict(scoring) if scoring is not None else dict(load_league()["scoring"])
        if name is not None:
            self.name = name

    def project(self, history: History) -> pd.DataFrame:
        history.assert_no_future()
        gl = canonical_logs(history.game_logs)
        if gl.empty:
            raise ValueError("history has no game logs; nothing to project from")
        last = int(season_starts(gl["season"]).max())
        gl = gl[season_starts(gl["season"]) == last].copy()
        gl["fp"] = fantasy_points_frame(gl, self.scoring)
        stat_cols = list(STAT_COLUMN_MAP.values())
        grp = gl.groupby("player_id", sort=True)
        per_game = grp[stat_cols + ["min"]].mean()
        q = grp["fp"].quantile([0.10, 0.50, 0.90]).unstack()
        out = pd.DataFrame({
            "season": history.target_season,
            "player_id": per_game.index.to_numpy("int64"),
            "player_name": grp["player_name"].last().reindex(per_game.index).to_numpy(),
            "model": self.name,
            "proj_gp": grp.size().reindex(per_game.index).to_numpy(float),
            "proj_mpg": per_game["min"].to_numpy(float),
        })
        for c in stat_cols:
            out[f"proj_{c}"] = per_game[c].to_numpy(float)
        assert set(f"proj_{c}" for c in stat_cols) == set(PROJECTION_STATS)
        out["proj_fppg"] = fantasy_points_frame(out, self.scoring, prefix="proj_").to_numpy()
        out["proj_total_fp"] = out["proj_fppg"] * out["proj_gp"]
        out["fppg_p10"] = q[0.10].reindex(per_game.index).to_numpy(float)
        out["fppg_p50"] = q[0.50].reindex(per_game.index).to_numpy(float)
        out["fppg_p90"] = q[0.90].reindex(per_game.index).to_numpy(float)
        return validate_table(out.reset_index(drop=True), "projections")
