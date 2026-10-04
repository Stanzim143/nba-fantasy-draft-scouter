"""Hyper-parameters of the baseline model, in one place.

Most numbers that shape projections are *estimated from the history* (age curves, shrinkage
constants, recency decay, dispersion). What lives here are (a) search grids and lower limits for
those estimates and (b) documented fall-back values used only when the history is too thin to
estimate (for example a single season, where no season pair exists).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaselineConfig:
    n_lags: int = 4                                   # seasons of history a projection looks back
    decay_grid: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)   # recency weights searched (per season back)
    default_decay: float = 0.6                        # used when too little history to search
    age_degree: int = 2                               # polynomial degree of age curves
    age_min_pairs: int = 40                           # season pairs needed to trust an age curve
    min_fit_rows: int = 30                            # rows needed to fit a shrinkage constant
    availability_decay: float = 0.6                   # recency weight on games-played fraction
    availability_C: float = 1.0                       # inverse L2 strength of the availability logit
    availability_min_rows: int = 60                   # rows to fit the availability model
    # Hurdle stage of availability (ADR 0031): P(player appears at all) x conditional games played. Off by default so
    # `baseline` stays bit-identical; `baseline_hurdle` turns it on.
    appearance_hurdle: bool = False
    appearance_C: float = 1.0                         # inverse L2 strength of the appearance logit
    appearance_min_rows: int = 100                    # historical (player, season) rows needed to fit it
    # Season-total intervals (ADR 0033): adds proj_total_fp_p10/p50/p90 columns; every existing column is unchanged.
    season_intervals: bool = False
    # Interval calibration (``calibrate_quantiles``, ``fit_season_uncertainty``) uses out-of-fold residuals: the most recent
    # ``oof_folds`` training seasons are each projected by a model refit on the history before them (expanding window),
    # instead of by the model fit on those same rows, which understates the spread. A fold needs ``oof_min_seasons`` earlier
    # seasons; with no usable fold the in-sample residuals are the fall-back. False restores the in-sample calibration.
    oof_calibration: bool = True
    oof_folds: int = 3
    oof_min_seasons: int = 3
    min_rookie_rows: int = 20                         # historical rookies needed for the draft-slot prior
    active_seasons: int = 2                           # player must have played in one of this many latest seasons
    vol_min_games: int = 10                           # games for a season's FP std to inform volatility
    vol_quantile_min_games: int = 20                  # games for a season to feed pooled FP quantiles
    mpg_max: float = 44.0
    # Shrinkage tuning (decay/kappa) feeds the rate prior the *predicted* minutes, as forecasting does. False
    # restores the old in-sample tuning on the target season's actual MPG (kept only to measure the delta, ADR 0030).
    tune_with_predicted_mpg: bool = True
    # Position group (G/F/C) enters the rate priors and rookie priors. The `players.position` it comes from is the
    # player's *current* label, not point-in-time (no dated position history is ingested), so it can leak a little
    # (a player later moved G->F is already F in older seasons). False sets every player to 'U' (leak-free ablation).
    use_position_priors: bool = True
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90)
    season_games: int | None = None                   # override the target-season schedule length
    # Fall-back shrinkage constants (exposure units: minutes for rates, attempts for pct, games otherwise).
    default_kappa: dict[str, float] = field(default_factory=lambda: {
        "mpg": 25.0, "fga": 400.0, "fta": 400.0, "fg3a": 500.0, "reb": 400.0, "ast": 400.0,
        "stl": 800.0, "blk": 800.0, "tov": 500.0,
        "fg_pct": 300.0, "ft_pct": 150.0, "fg3_pct": 250.0, "vol": 15.0,
    })
