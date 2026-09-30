"""The backtest runner passes the optional offseason tables through History.extras when they exist."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from src.backtest import runner  # noqa: E402
from src.contracts import HISTORY_TABLES, History  # noqa: E402


def _args():
    return SimpleNamespace(synthetic=False)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from model_testkit import make_league

    from src.store import write_table

    tables = make_league()
    for name in HISTORY_TABLES:
        write_table(tables[name], name, tmp_path)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    return tmp_path, tables


def test_without_offseason_tables_only_the_history_tables_load(store):
    _, _ = store
    loaded = runner.load_run_tables(_args())
    assert set(loaded) == set(HISTORY_TABLES)


def test_offseason_tables_travel_as_extras_and_are_sliced_by_history(store):
    from offseason_testkit import make_offseason

    from src.ingest import nba_offseason as no

    base, tables = store
    logs, tg = make_offseason(tables, "signal", seed=4)
    no.write_offseason_table(logs, "offseason_logs", base)
    no.write_offseason_table(tg, "offseason_team_games", base)
    loaded = runner.load_run_tables(_args())
    assert {"offseason_logs", "offseason_team_games"} <= set(loaded)
    h = History.until(loaded, "2018-19")
    assert set(h.extras) == {"offseason_logs", "offseason_team_games"}
    assert h.extras["offseason_logs"]["season"].map(lambda s: int(s[:4])).max() < 2018    # nothing tagged for 2018-19 or later
    assert (h.extras["offseason_logs"]["event_season"] == "2018-19").any()               # the July/October before it is visible
    h.assert_no_future()
