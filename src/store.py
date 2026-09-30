"""Parquet table IO with contract validation and atomic writes.

Atomic (write temp file, then rename) so parallel readers never see a half-written table.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.contracts import HISTORY_TABLES, TABLES, table_path, validate_table


def write_table(df: pd.DataFrame, name: str, base: Path | None = None) -> Path:
    validate_table(df, name)
    path = table_path(name, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_table(name: str, base: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    path = table_path(name, base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has the {name} ingest run?")
    df = pd.read_parquet(path)
    return validate_table(df, name) if validate else df


def table_exists(name: str, base: Path | None = None) -> bool:
    return table_path(name, base).exists()


def load_tables(names: tuple[str, ...] = HISTORY_TABLES, base: Path | None = None) -> Mapping[str, pd.DataFrame]:
    """Load several tables for History.until(...)."""
    unknown = [n for n in names if n not in TABLES]
    if unknown:
        raise KeyError(unknown)
    return {n: read_table(n, base) for n in names}
