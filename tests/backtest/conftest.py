import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # bt_helpers

from bt_helpers import SMALL  # noqa: E402
from src.synthetic import make_synthetic_tables  # noqa: E402


@pytest.fixture(scope="session")
def _tables_master():
    return make_synthetic_tables(**SMALL)


@pytest.fixture()
def tables(_tables_master):
    """Fresh deep copy per test: tests may (and the leakage check does) mutate frames in place."""
    return {k: v.copy(deep=True) for k, v in _tables_master.items()}
