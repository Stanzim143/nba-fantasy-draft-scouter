"""Exception types for the backtest package."""
from __future__ import annotations


class BacktestError(Exception):
    """Base class: the backtest cannot produce a trustworthy result."""


class ProjectionValidationError(BacktestError, ValueError):
    """A projector returned a frame that violates the ``projections`` contract (or its semantics)."""


class LeakageError(BacktestError, AssertionError):
    """Future information reached a projector, or a projector's output depends on it."""


class ProjectorNotAvailable(BacktestError):
    """A named projector cannot be resolved (model registry missing, or unknown name)."""
