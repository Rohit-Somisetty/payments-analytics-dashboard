"""Build merchant-week ML features directly from DuckDB."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = DATA_DIR / "ml"
DUCKDB_PATH = DATA_DIR / "analytics.duckdb"
FEATURES_PATH = ML_DIR / "features_merchant_week.parquet"
FEATURE_SQL_PATH = BASE_DIR / "sql" / "ml_features.sql"

logger = logging.getLogger(__name__)

_DEFAULT_QUERY = (
    "WITH weekly AS (\n"
    "    SELECT\n"
    "        merchant_id,\n"
    "        CAST(date_trunc('week', txn_date) AS DATE) AS week_start_date,\n"
    "        COUNT(*) AS txns,\n"
    "        SUM(CASE WHEN approved THEN 1 ELSE 0 END) AS approved_txns,\n"
    "        SUM(CASE WHEN NOT approved THEN 1 ELSE 0 END) AS decline_txns,\n"
    "        SUM(CASE WHEN approved THEN amount ELSE 0 END) AS gpv,\n"
    "        AVG(amount) AS avg_amount,\n"
    "        SUM(CASE WHEN approved THEN fee ELSE 0 END) AS total_fees,\n"
    "        COUNT(DISTINCT txn_date) AS active_days,\n"
    "        STDDEV_POP(amount) AS amount_volatility,\n"
    "        SUM(CASE WHEN payment_method = 'card' AND approved THEN amount ELSE 0 END) AS card_gpv,\n"
    "        SUM(CASE WHEN payment_method = 'ach' AND approved THEN amount ELSE 0 END) AS ach_gpv,\n"
    "        SUM(CASE WHEN payment_method = 'wallet' AND approved THEN amount ELSE 0 END) AS wallet_gpv,\n"
    "        SUM(CASE WHEN channel = 'pos' AND approved THEN amount ELSE 0 END) AS pos_gpv,\n"
    "        SUM(CASE WHEN channel = 'ecom' AND approved THEN amount ELSE 0 END) AS ecom_gpv,\n"
    "        SUM(CASE WHEN channel IN ('in_app', 'inapp') AND approved THEN amount ELSE 0 END) AS inapp_gpv\n"
    "    FROM transactions\n"
    "    GROUP BY 1, 2\n"
    "),\n"
    "daily AS (\n"
    "    SELECT\n"
    "        merchant_id,\n"
    "        CAST(date_trunc('week', txn_date) AS DATE) AS week_start_date,\n"
    "        txn_date,\n"
    "        SUM(CASE WHEN approved THEN amount ELSE 0 END) AS daily_gpv\n"
    "    FROM transactions\n"
    "    GROUP BY 1, 2, 3\n"
    "),\n"
    "volatility AS (\n"
    "    SELECT\n"
    "        merchant_id,\n"
    "        week_start_date,\n"
    "        STDDEV_POP(daily_gpv) AS gpv_volatility\n"
    "    FROM daily\n"
    "    GROUP BY 1, 2\n"
    ")\n"
    "SELECT\n"
    "    w.merchant_id,\n"
    "    w.week_start_date,\n"
    "    w.txns,\n"
    "    w.approved_txns,\n"
    "    w.decline_txns,\n"
    "    w.gpv,\n"
    "    w.avg_amount,\n"
    "    w.total_fees,\n"
    "    w.active_days,\n"
    "    COALESCE(w.amount_volatility, 0) AS amount_volatility,\n"
    "    COALESCE(v.gpv_volatility, 0) AS gpv_volatility,\n"
    "    CASE WHEN w.txns > 0 THEN w.decline_txns::DOUBLE / w.txns ELSE 0 END AS decline_rate,\n"
    "    CASE WHEN w.active_days > 0 THEN w.txns::DOUBLE / w.active_days ELSE 0 END AS txn_frequency,\n"
    "    CASE WHEN w.gpv > 0 THEN w.total_fees / w.gpv ELSE 0 END AS fee_yield,\n"
    "    CASE WHEN w.gpv > 0 THEN w.card_gpv / w.gpv ELSE 0 END AS share_card,\n"
    "    CASE WHEN w.gpv > 0 THEN w.ach_gpv / w.gpv ELSE 0 END AS share_ach,\n"
    "    CASE WHEN w.gpv > 0 THEN w.wallet_gpv / w.gpv ELSE 0 END AS share_wallet,\n"
    "    CASE WHEN w.gpv > 0 THEN w.pos_gpv / w.gpv ELSE 0 END AS share_pos,\n"
    "    CASE WHEN w.gpv > 0 THEN w.ecom_gpv / w.gpv ELSE 0 END AS share_ecom,\n"
    "    CASE WHEN w.gpv > 0 THEN w.inapp_gpv / w.gpv ELSE 0 END AS share_inapp\n"
    "FROM weekly w\n"
    "LEFT JOIN volatility v USING (merchant_id, week_start_date)\n"
    "ORDER BY 1, 2;"
)


def _load_query(sql_path: Path) -> str:
    if sql_path.exists():
        return sql_path.read_text()
    logger.warning("%s not found; falling back to in-module SQL", sql_path)
    return _DEFAULT_QUERY


def build_features(duckdb_path: Path = DUCKDB_PATH, output_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Aggregate merchant-week features and persist them to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {duckdb_path}")

    query = _load_query(FEATURE_SQL_PATH)
    logger.info("Running feature SQL against %s", duckdb_path)
    with duckdb.connect(str(duckdb_path)) as conn:
        df = conn.execute(query).fetch_df()

    if df.empty:
        raise ValueError("Feature query returned no rows. Did you run the ETL pipeline?")

    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["month"] = df["week_start_date"].dt.month.astype("int16")
    df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype("int16")
    df.sort_values(["merchant_id", "week_start_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    logger.info("Saved %s rows of merchant-week features to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
