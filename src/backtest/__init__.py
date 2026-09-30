"""Walk-forward backtest harness, metrics, leakage guards, ablation and reporting.

See docs/backtest.md.  Public surface::

    from src.backtest import walk_forward, assert_projector_ignores_future, run_ablation
"""
from src.backtest.ablation import AblationResult, ablation_from_results, paired_lift, run_ablation
from src.backtest.actuals import build_eval_frame, season_actuals
from src.backtest.benchmarks import AdpBenchmark, NaiveLastSeason, load_adp, stats_to_projection
from src.backtest.errors import (BacktestError, LeakageError, ProjectionValidationError,
                                 ProjectorNotAvailable)
from src.backtest.harness import BacktestResult, expand_seasons, validate_projections, walk_forward
from src.backtest.leakage import assert_projector_ignores_future, build_history, data_hash
from src.backtest.misses import analyze_misses
from src.backtest.report import write_report

__all__ = [
    "AblationResult", "AdpBenchmark", "BacktestError", "BacktestResult", "LeakageError", "NaiveLastSeason",
    "ProjectionValidationError", "ProjectorNotAvailable", "ablation_from_results", "analyze_misses",
    "assert_projector_ignores_future", "build_eval_frame", "build_history", "data_hash", "expand_seasons",
    "load_adp", "paired_lift", "run_ablation", "season_actuals", "stats_to_projection",
    "validate_projections", "walk_forward", "write_report",
]
