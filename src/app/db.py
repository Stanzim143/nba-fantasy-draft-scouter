"""Read-only DuckDB query layer over the existing parquet store.

This is deliberately thin: it opens an **in-memory** DuckDB connection (nothing persisted to
disk — a fresh connection per process/session is cheap and there is nothing to keep consistent
across restarts) and registers a ``VIEW`` per contract table directly over
``read_parquet(...)`` for whichever parquet files already exist under ``src.contracts.data_dir()``.

It does **not** duplicate anything ``src/store.py`` already does: no schema validation, no
writing, no atomic-replace logic. Those parquet files are already validated on the way in by
``src.store.write_table``; this module only adds a SQL layer on top of already-trusted data for
ad-hoc querying (season lists, quick aggregates, future reporting) that the app or a notebook
wants without loading full DataFrames through pandas first.

Library use::

    from src.app.db import connect, available_seasons, query

    con = connect()                      # real data dir (NBA_DATA_DIR / ~/dev-data/...)
    available_seasons(con)                # ['2015-16', '2016-17', ...]
    query(con, "SELECT * FROM players LIMIT 5")

``connect(base=...)`` takes an explicit data directory (handy for tests against a tiny fixture
parquet file, or a demo dataset, without touching ``NBA_DATA_DIR``).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.contracts import TABLES, data_dir, table_path


def connect(base: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with one VIEW per contract table that has a parquet
    file on disk. Tables whose parquet file does not exist yet (e.g. ingest hasn't run, or this
    is a fresh checkout) are simply skipped — see ``available_tables`` to check what landed."""
    con = duckdb.connect(database=":memory:")
    root = base or data_dir()
    for name in TABLES:
        path = table_path(name, root)
        if path.exists():
            # Paths come from the local filesystem (data_dir()/table_path), never from
            # untrusted input, so simple quote-escaping is enough for this DDL string.
            escaped = path.as_posix().replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")
    return con


def available_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Contract tables that currently have a VIEW registered (i.e. their parquet file exists)."""
    return sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())


def available_seasons(con: duckdb.DuckDBPyConnection, table: str = "game_logs") -> list[str]:
    """Distinct ``season`` values in ``table``, sorted. Empty list if the table isn't registered
    or has no ``season`` column."""
    if table not in available_tables(con):
        return []
    columns = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    if "season" not in columns:
        return []
    rows = con.execute(f"SELECT DISTINCT season FROM {table} ORDER BY season").fetchall()
    return [r[0] for r in rows]


def query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run ``sql`` (optionally parameterised with ``?`` placeholders) and return a DataFrame."""
    return con.execute(sql, params or []).df()
