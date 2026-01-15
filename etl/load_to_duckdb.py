"""DuckDB orchestration helpers for the payments analytics pipeline."""
from __future__ import annotations

from pathlib import Path

import duckdb

import config


def connect_duckdb(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection using the configured path."""
    target = db_path or config.DUCKDB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(target.as_posix())


def create_schema_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Execute the schema DDL script to ensure baseline tables exist."""
    ddl = config.SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    con.execute(ddl)


def load_dimensions(con: duckdb.DuckDBPyConnection) -> None:
    """Load merchants and customers dimension tables from parquet."""
    if not config.RAW_MERCHANTS_PATH.exists() or not config.RAW_CUSTOMERS_PATH.exists():
        raise FileNotFoundError("Missing raw merchants/customers parquet files. Run data generation first.")

    con.execute("DELETE FROM merchants")
    con.execute(
        f"""
        INSERT INTO merchants
        SELECT
            merchant_id::BIGINT,
            industry,
            region,
            merchant_size,
            sales_channel,
            signup_date::DATE AS signup_date
        FROM read_parquet('{config.RAW_MERCHANTS_PATH.as_posix()}')
        """
    )

    con.execute("DELETE FROM customers")
    con.execute(
        f"""
        INSERT INTO customers
        SELECT
            customer_id::BIGINT,
            segment,
            region,
            join_date::DATE AS join_date
        FROM read_parquet('{config.RAW_CUSTOMERS_PATH.as_posix()}')
        """
    )


def load_transactions(con: duckdb.DuckDBPyConnection) -> None:
    """Load the fact transactions table from partitioned parquet files."""
    if not config.RAW_TRANSACTIONS_DIR.exists() or not any(
        config.RAW_TRANSACTIONS_DIR.rglob("*.parquet")
    ):
        raise FileNotFoundError("No transaction parquet files found under data/raw/transactions")

    glob_path = config.RAW_TRANSACTIONS_GLOB
    con.execute("DELETE FROM transactions")
    con.execute(
        f"""
        INSERT INTO transactions
        SELECT
            txn_id::BIGINT AS txn_id,
            ts::TIMESTAMP AS ts,
            date_trunc('day', ts)::DATE AS txn_date,
            merchant_id::BIGINT AS merchant_id,
            customer_id::BIGINT AS customer_id,
            amount::DOUBLE AS amount,
            currency AS currency,
            approved::BOOLEAN AS approved,
            payment_method,
            decline_reason,
            fee::DOUBLE AS fee,
            channel,
            year::INTEGER AS partition_year,
            month::INTEGER AS partition_month
        FROM read_parquet('{glob_path}', hive_partitioning=1)
        """
    )
