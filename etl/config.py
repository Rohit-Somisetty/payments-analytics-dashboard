"""Centralized configuration for the payments analytics pipeline."""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_MARTS_DIR: Path = DATA_DIR / "marts"
DOCS_DIR: Path = BASE_DIR / "docs"
SQL_DIR: Path = BASE_DIR / "sql"
SCHEMA_SQL_PATH: Path = SQL_DIR / "schema.sql"
KPI_SQL_PATH: Path = SQL_DIR / "kpis.sql"
SQL_ANALYSIS_DIR: Path = SQL_DIR / "analysis"
RAW_MERCHANTS_PATH: Path = DATA_RAW_DIR / "merchants.parquet"
RAW_CUSTOMERS_PATH: Path = DATA_RAW_DIR / "customers.parquet"
RAW_TRANSACTIONS_DIR: Path = DATA_RAW_DIR / "transactions"
RAW_TRANSACTIONS_GLOB: str = (RAW_TRANSACTIONS_DIR / "**" / "*.parquet").as_posix()
RUN_SUMMARY_PATH: Path = DOCS_DIR / "run_summary.md"
DUCKDB_PATH: Path = DATA_DIR / "analytics.duckdb"
RANDOM_SEED: int = 42


def ensure_directories() -> None:
    """Create required directory structure if it is missing."""
    for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_MARTS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
