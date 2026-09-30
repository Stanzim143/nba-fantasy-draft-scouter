import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """No in-season test may read or write the real data directory."""
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    monkeypatch.delenv("ESPN_TEAM_ID", raising=False)
