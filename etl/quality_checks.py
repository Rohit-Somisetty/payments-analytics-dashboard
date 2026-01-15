"""DuckDB-backed quality checks for the synthetic payments data."""
from __future__ import annotations

from pathlib import Path

import duckdb

import config


def run_all() -> None:
    """Run the minimum set of quality checks required for Step 2."""
    _assert_data_presence()
    _check_nulls()
    _check_ranges()
    _check_fk_integrity()


def _duckdb_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _transactions_glob() -> str:
    return (config.RAW_TRANSACTIONS_DIR / "**" / "*.parquet").as_posix()


def _assert_data_presence() -> None:
    missing = []
    if not config.RAW_MERCHANTS_PATH.exists():
        missing.append("merchants")
    if not config.RAW_CUSTOMERS_PATH.exists():
        missing.append("customers")
    if not config.RAW_TRANSACTIONS_DIR.exists() or not any(config.RAW_TRANSACTIONS_DIR.rglob("*.parquet")):
        missing.append("transactions")
    if missing:
        raise FileNotFoundError(
            "Raw data missing for: " + ", ".join(missing) + ". Run the generation step first."
        )


def _check_nulls() -> None:
    con = _duckdb_con()
    checks = {
        config.RAW_MERCHANTS_PATH: ("merchant_id", "industry", "region"),
        config.RAW_CUSTOMERS_PATH: ("customer_id", "segment", "region"),
    }
    for path, columns in checks.items():
        query = "SELECT " + ", ".join(
            [f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls" for col in columns]
        ) + f" FROM read_parquet('{path.as_posix()}')"
        row = con.execute(query).fetchone()
        violations = {
            col: row[idx]
            for idx, col in enumerate(columns)
            if row[idx] and row[idx] > 0
        }
        if violations:
            con.close()
            raise ValueError(f"Null values detected in {path.name}: {violations}")

    txn_query = "SELECT " + ", ".join(
        [
            "SUM(CASE WHEN txn_id IS NULL THEN 1 ELSE 0 END) AS txn_id_nulls",
            "SUM(CASE WHEN ts IS NULL THEN 1 ELSE 0 END) AS ts_nulls",
            "SUM(CASE WHEN merchant_id IS NULL THEN 1 ELSE 0 END) AS merchant_nulls",
            "SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS customer_nulls",
            "SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS amount_nulls",
            "SUM(CASE WHEN approved IS NULL THEN 1 ELSE 0 END) AS approved_nulls",
        ]
    ) + f" FROM read_parquet('{_transactions_glob()}')"
    txn_row = con.execute(txn_query).fetchone()
    txn_columns = [
        "txn_id",
        "ts",
        "merchant_id",
        "customer_id",
        "amount",
        "approved",
    ]
    txn_violations = {
        col: txn_row[idx]
        for idx, col in enumerate(txn_columns)
        if txn_row[idx] and txn_row[idx] > 0
    }
    con.close()
    if txn_violations:
        raise ValueError(f"Null values detected in transactions: {txn_violations}")


def _check_ranges() -> None:
    con = _duckdb_con()
    query = f"""
        SELECT
            SUM(CASE WHEN amount <= 0 THEN 1 ELSE 0 END) AS invalid_amounts,
            SUM(CASE WHEN fee < 0 THEN 1 ELSE 0 END) AS invalid_fees,
            SUM(CASE WHEN approved NOT IN (TRUE, FALSE) THEN 1 ELSE 0 END) AS invalid_approved
        FROM read_parquet('{_transactions_glob()}')
    """
    invalid_amounts, invalid_fees, invalid_approved = con.execute(query).fetchone()
    con.close()
    errors = {}
    if invalid_amounts:
        errors["amount"] = int(invalid_amounts)
    if invalid_fees:
        errors["fee"] = int(invalid_fees)
    if invalid_approved:
        errors["approved"] = int(invalid_approved)
    if errors:
        raise ValueError(f"Range checks failed: {errors}")


def _check_fk_integrity() -> None:
    con = _duckdb_con()
    txn_glob = _transactions_glob()
    missing_merchants = con.execute(
        f"""
        WITH txn AS (SELECT DISTINCT merchant_id FROM read_parquet('{txn_glob}'))
        SELECT COUNT(*)
        FROM txn t
        LEFT JOIN read_parquet('{config.RAW_MERCHANTS_PATH.as_posix()}') m USING (merchant_id)
        WHERE m.merchant_id IS NULL
        """
    ).fetchone()[0]
    missing_customers = con.execute(
        f"""
        WITH txn AS (SELECT DISTINCT customer_id FROM read_parquet('{txn_glob}'))
        SELECT COUNT(*)
        FROM txn t
        LEFT JOIN read_parquet('{config.RAW_CUSTOMERS_PATH.as_posix()}') c USING (customer_id)
        WHERE c.customer_id IS NULL
        """
    ).fetchone()[0]
    con.close()
    errors = {}
    if missing_merchants:
        errors["merchant_id"] = int(missing_merchants)
    if missing_customers:
        errors["customer_id"] = int(missing_customers)
    if errors:
        raise ValueError(f"Foreign key integrity failed: {errors}")
