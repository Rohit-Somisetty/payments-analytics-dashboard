"""Synthetic dataset generation for the Payments Analytics sandbox."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import shutil
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config


@dataclass
class RunMetadata:
    """Captured metrics from the most recent data generation or scan."""

    start_date: date
    end_date: date
    merchants: int
    customers: int
    transactions: int
    approval_rate: float
    industry_gpv: list[tuple[str, float]]
    regenerated: bool


@dataclass
class DuckDBSummary:
    """Metadata about DuckDB views and exported marts."""

    db_path: Path
    views: list[str]
    mart_rows: dict[str, int]


def _log(message: str, logger: Callable[[str], None] | None) -> None:
    if logger:
        logger(message)


def raw_data_exists() -> bool:
    """Check if merchants, customers, and transaction partitions exist."""
    merchants_exists = config.RAW_MERCHANTS_PATH.exists()
    customers_exists = config.RAW_CUSTOMERS_PATH.exists()
    transactions_exists = config.RAW_TRANSACTIONS_DIR.exists() and any(
        config.RAW_TRANSACTIONS_DIR.rglob("*.parquet")
    )
    return merchants_exists and customers_exists and transactions_exists


def ensure_data(
    *,
    num_merchants: int,
    num_customers: int,
    num_transactions: int,
    start_date: date,
    end_date: date,
    logger: Callable[[str], None] | None = None,
) -> RunMetadata:
    """Generate datasets when missing or summarize the existing snapshot."""
    if raw_data_exists():
        _log("Raw data already present. Skipping generation and scanning existing files.", logger)
        return summarize_existing_data(logger=logger)

    return generate_all(
        num_merchants=num_merchants,
        num_customers=num_customers,
        num_transactions=num_transactions,
        start_date=start_date,
        end_date=end_date,
        logger=logger,
    )


def generate_all(
    *,
    num_merchants: int,
    num_customers: int,
    num_transactions: int,
    start_date: date,
    end_date: date,
    logger: Callable[[str], None] | None = None,
) -> RunMetadata:
    """Generate merchants, customers, and transactions datasets."""
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    _log("Generating synthetic merchants and customers", logger)
    rng = np.random.default_rng(config.RANDOM_SEED)

    merchant_df = _generate_merchants(num_merchants, start_date, rng)
    customer_df = _generate_customers(num_customers, start_date, rng)

    config.RAW_MERCHANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_CUSTOMERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merchant_df.to_parquet(config.RAW_MERCHANTS_PATH, index=False)
    customer_df.to_parquet(config.RAW_CUSTOMERS_PATH, index=False)

    _log("Generating synthetic transactions", logger)
    transactions_df, txn_context = _generate_transactions(
        merchant_df=merchant_df,
        customer_df=customer_df,
        num_transactions=num_transactions,
        start_date=start_date,
        end_date=end_date,
        rng=rng,
    )

    _log("Writing partitioned transactions parquet dataset", logger)
    _write_partitioned_transactions(transactions_df)

    industry_gpv = _top_industries_by_gpv(
        industries=txn_context["industries"],
        amounts=txn_context["amounts"],
        approved=txn_context["approved"],
    )
    metadata = RunMetadata(
        start_date=transactions_df["ts"].min().date(),
        end_date=transactions_df["ts"].max().date(),
        merchants=len(merchant_df),
        customers=len(customer_df),
        transactions=len(transactions_df),
        approval_rate=float(txn_context["approved"].mean()),
        industry_gpv=industry_gpv,
        regenerated=True,
    )
    return metadata


def summarize_existing_data(*, logger: Callable[[str], None] | None = None) -> RunMetadata:
    """Scan the parquet files to produce summary statistics without regenerating."""
    _log("Summarizing existing raw parquet files", logger)

    merchants = _count_rows(config.RAW_MERCHANTS_PATH)
    customers = _count_rows(config.RAW_CUSTOMERS_PATH)

    con = duckdb.connect(database=":memory:")
    txn_path = _duckdb_glob(config.RAW_TRANSACTIONS_DIR)
    stats = con.execute(
        f"""
        WITH txn AS (
            SELECT * FROM read_parquet('{txn_path}')
        )
        SELECT
            COUNT(*) AS txn_count,
            MIN(ts)::DATE AS start_date,
            MAX(ts)::DATE AS end_date,
            AVG(CASE WHEN approved THEN 1 ELSE 0 END) AS approval_rate
        FROM txn
        """
    ).fetchone()

    industry_gpv = con.execute(
        f"""
        WITH txn AS (SELECT * FROM read_parquet('{txn_path}') WHERE approved)
        SELECT industry, SUM(amount) AS gpv
        FROM txn
        INNER JOIN read_parquet('{config.RAW_MERCHANTS_PATH.as_posix()}') USING (merchant_id)
        GROUP BY industry
        ORDER BY gpv DESC
        LIMIT 5
        """
    ).fetchall()

    con.close()

    return RunMetadata(
        start_date=stats[1],
        end_date=stats[2],
        merchants=merchants,
        customers=customers,
        transactions=stats[0],
        approval_rate=float(stats[3]),
        industry_gpv=[(row[0], float(row[1])) for row in industry_gpv],
        regenerated=False,
    )


def write_run_summary(metadata: RunMetadata, duckdb_summary: DuckDBSummary | None = None) -> Path:
    """Persist the latest run metadata to docs/run_summary.md."""
    lines = [
        "# Data Generation Summary",
        "",
        f"*Date range:* {metadata.start_date} → {metadata.end_date}",
        f"*Merchants:* {metadata.merchants:,}",
        f"*Customers:* {metadata.customers:,}",
        f"*Transactions:* {metadata.transactions:,}",
        f"*Approval rate:* {metadata.approval_rate:.2%}",
        f"*Generated this run:* {'Yes' if metadata.regenerated else 'No (existing data reused)'}",
        "",
        "## Top Industries by GPV",
    ]
    if metadata.industry_gpv:
        for idx, (industry, gpv) in enumerate(metadata.industry_gpv, start=1):
            lines.append(f"{idx}. {industry}: ${gpv:,.0f}")
    else:
        lines.append("No approved transactions available.")

    if duckdb_summary:
        lines.extend(
            [
                "",
                "## DuckDB & KPI Views",
                f"*Database:* {duckdb_summary.db_path}",
                "*Views:*",
            ]
        )
        for view in duckdb_summary.views:
            lines.append(f"- {view}")
        lines.extend(["", "*Exported Marts:*"])
        if duckdb_summary.mart_rows:
            for mart, rows in duckdb_summary.mart_rows.items():
                lines.append(f"- {mart}: {rows:,} rows")
        else:
            lines.append("- None exported")

    config.RUN_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return config.RUN_SUMMARY_PATH


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_merchants(num_merchants: int, start_date: date, rng: np.random.Generator) -> pd.DataFrame:
    industries = [
        "Retail",
        "Food & Beverage",
        "Travel",
        "Entertainment",
        "SaaS",
        "Gaming",
        "Marketplace",
    ]
    industry_probs = [0.25, 0.15, 0.1, 0.1, 0.15, 0.1, 0.15]
    regions = ["NA", "EMEA", "APAC", "LATAM", "Africa"]
    region_probs = [0.4, 0.25, 0.2, 0.1, 0.05]
    merchant_sizes = ["small", "medium", "enterprise"]
    size_probs = [0.55, 0.35, 0.10]
    sales_channels = ["direct", "partner", "online"]
    sales_probs = [0.4, 0.25, 0.35]

    signup_start = start_date - timedelta(days=365 * 3)
    signup_offsets = rng.integers(0, 365 * 3, size=num_merchants, endpoint=False)
    signup_dates = np.array([signup_start + timedelta(days=int(o)) for o in signup_offsets])

    merchant_df = pd.DataFrame(
        {
            "merchant_id": np.arange(1, num_merchants + 1, dtype=np.int64),
            "industry": rng.choice(industries, size=num_merchants, p=industry_probs),
            "region": rng.choice(regions, size=num_merchants, p=region_probs),
            "merchant_size": rng.choice(merchant_sizes, size=num_merchants, p=size_probs),
            "sales_channel": rng.choice(sales_channels, size=num_merchants, p=sales_probs),
            "signup_date": signup_dates,
        }
    )
    return merchant_df


def _generate_customers(num_customers: int, start_date: date, rng: np.random.Generator) -> pd.DataFrame:
    segments = ["consumer", "smb", "enterprise"]
    segment_probs = [0.8, 0.18, 0.02]
    regions = ["NA", "EMEA", "APAC", "LATAM", "Africa"]
    region_probs = [0.45, 0.25, 0.18, 0.07, 0.05]
    join_start = start_date - timedelta(days=365 * 2)
    join_offsets = rng.integers(0, 365 * 2, size=num_customers, endpoint=False)
    join_dates = np.array([join_start + timedelta(days=int(o)) for o in join_offsets])

    customer_df = pd.DataFrame(
        {
            "customer_id": np.arange(1, num_customers + 1, dtype=np.int64),
            "segment": rng.choice(segments, size=num_customers, p=segment_probs),
            "region": rng.choice(regions, size=num_customers, p=region_probs),
            "join_date": join_dates,
        }
    )
    return customer_df


def _generate_transactions(
    *,
    merchant_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    num_transactions: int,
    start_date: date,
    end_date: date,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    merchants = merchant_df.set_index("merchant_id")
    size_weights = merchants["merchant_size"].map({"small": 0.6, "medium": 1.0, "enterprise": 2.5}).to_numpy()
    merchant_prob = size_weights / size_weights.sum()
    merchant_indices = rng.choice(len(merchants), size=num_transactions, p=merchant_prob)
    merchant_ids = merchant_indices + 1

    merchant_industries = merchants["industry"].to_numpy()
    merchant_regions = merchants["region"].to_numpy()
    merchant_channels = merchants["sales_channel"].to_numpy()

    base_channels = merchant_channels[merchant_indices]

    channels = _sample_channels(base_channels, rng)
    payment_methods = _sample_payment_methods(channels, rng)

    customer_ids = rng.integers(1, len(customer_df) + 1, size=num_transactions, dtype=np.int64)

    timestamps = _sample_timestamps(num_transactions, start_date, end_date, rng)

    merchant_sizes = merchants["merchant_size"].to_numpy()
    sizes = merchant_sizes[merchant_indices]
    amounts = _sample_amounts(sizes, rng)
    fees = _sample_fees(amounts, rng)

    industries = merchant_industries[merchant_indices]
    regions = merchant_regions[merchant_indices]

    approvals = _sample_approvals(
        industries=industries,
        regions=regions,
        channels=channels,
        rng=rng,
    )

    decline_reasons = _assign_decline_reasons(approvals, rng)

    transactions_df = pd.DataFrame(
        {
            "txn_id": np.arange(1, num_transactions + 1, dtype=np.int64),
            "ts": pd.to_datetime(timestamps),
            "merchant_id": merchant_ids.astype(np.int64),
            "customer_id": customer_ids,
            "amount": amounts,
            "currency": np.full(num_transactions, "USD", dtype=object),
            "approved": approvals,
            "payment_method": payment_methods,
            "decline_reason": decline_reasons,
            "fee": fees,
            "channel": channels,
        }
    )

    context = {
        "industries": industries,
        "amounts": amounts,
        "approved": approvals,
    }
    return transactions_df, context


def _sample_channels(base_channels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    channel_profiles = {
        "direct": (0.65, 0.25, 0.10),  # pos, ecom, inapp
        "partner": (0.45, 0.35, 0.20),
        "online": (0.20, 0.55, 0.25),
    }
    labels = np.array(["pos", "ecom", "inapp"], dtype=object)
    channels = np.empty_like(base_channels)
    for origin, probs in channel_profiles.items():
        mask = base_channels == origin
        if not mask.any():
            continue
        channels[mask] = rng.choice(labels, size=mask.sum(), p=probs)
    return channels


def _sample_payment_methods(channels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    payment_profiles = {
        "pos": (0.8, 0.05, 0.15),  # card, ach, wallet
        "ecom": (0.7, 0.1, 0.2),
        "inapp": (0.6, 0.05, 0.35),
    }
    labels = np.array(["card", "ach", "wallet"], dtype=object)
    methods = np.empty_like(channels)
    for channel, probs in payment_profiles.items():
        mask = channels == channel
        if not mask.any():
            continue
        methods[mask] = rng.choice(labels, size=mask.sum(), p=probs)
    return methods


def _sample_timestamps(num_rows: int, start_date: date, end_date: date, rng: np.random.Generator) -> np.ndarray:
    start = np.datetime64(start_date)
    stop = np.datetime64(end_date) + np.timedelta64(1, "D")
    days = np.arange(start, stop, dtype="datetime64[D]")
    weekdays = pd.to_datetime(days).dayofweek.to_numpy()
    weekday_weights = np.where(weekdays < 5, 1.1, 0.85)
    months = pd.to_datetime(days).month.to_numpy()
    month_weights = np.ones_like(months, dtype=float)
    month_weights[(months == 11)] = 1.35
    month_weights[(months == 12)] = 1.45
    month_weights[(months == 1)] = 0.9
    weights = weekday_weights * month_weights
    weights = weights / weights.sum()

    sampled_indices = rng.choice(len(days), size=num_rows, p=weights)
    sampled_days = days[sampled_indices].astype("datetime64[s]")
    seconds = rng.integers(0, 24 * 60 * 60, size=num_rows, dtype=np.int64)
    timestamps = sampled_days + seconds.astype("timedelta64[s]")
    return timestamps.astype("datetime64[ns]")


def _sample_amounts(merchant_sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size_to_mu_sigma = {
        "small": (3.5, 0.55),
        "medium": (4.5, 0.5),
        "enterprise": (5.2, 0.45),
    }
    mus = np.array([size_to_mu_sigma[size][0] for size in merchant_sizes])
    sigmas = np.array([size_to_mu_sigma[size][1] for size in merchant_sizes])
    samples = rng.lognormal(mean=mus, sigma=sigmas)
    return np.round(samples, 2)


def _sample_fees(amounts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rates = rng.normal(loc=0.018, scale=0.003, size=len(amounts))
    rates = np.clip(rates, 0.01, 0.035)
    fees = np.round(amounts * rates, 2)
    return fees


def _sample_approvals(
    *,
    industries: np.ndarray,
    regions: np.ndarray,
    channels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    industry_base = {
        "Retail": 0.89,
        "Food & Beverage": 0.87,
        "Travel": 0.82,
        "Entertainment": 0.84,
        "SaaS": 0.93,
        "Gaming": 0.8,
        "Marketplace": 0.88,
    }
    channel_adj = {"pos": 0.02, "ecom": -0.04, "inapp": -0.02}
    region_adj = {"LATAM": -0.06, "Africa": -0.08, "APAC": -0.01, "EMEA": 0.0, "NA": 0.02}

    base = np.array([industry_base.get(ind, 0.85) for ind in industries])
    channel_delta = np.array([channel_adj.get(ch, 0.0) for ch in channels])
    region_delta = np.array([region_adj.get(reg, 0.0) for reg in regions])
    probs = np.clip(base + channel_delta + region_delta, 0.5, 0.99)
    approvals = rng.random(len(probs)) < probs
    return approvals


def _assign_decline_reasons(approved: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    decline_reasons = np.full(len(approved), None, dtype=object)
    decline_options = [
        "insufficient_funds",
        "fraud_suspected",
        "network_error",
        "limit_exceeded",
        "regulatory_block",
    ]
    decline_probs = [0.45, 0.25, 0.15, 0.1, 0.05]
    declines = np.where(~approved)[0]
    if declines.size:
        decline_reasons[declines] = rng.choice(decline_options, size=declines.size, p=decline_probs)
    return decline_reasons


def _write_partitioned_transactions(transactions_df: pd.DataFrame) -> None:
    if config.RAW_TRANSACTIONS_DIR.exists():
        shutil.rmtree(config.RAW_TRANSACTIONS_DIR)
    config.RAW_TRANSACTIONS_DIR.mkdir(parents=True, exist_ok=True)

    grouped = transactions_df.groupby([transactions_df["ts"].dt.year, transactions_df["ts"].dt.month])
    for (year, month), group in grouped:
        output_dir = config.RAW_TRANSACTIONS_DIR / f"year={year:04d}" / f"month={month:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "part-000.parquet"
        group.to_parquet(output_path, index=False)


def _top_industries_by_gpv(
    *,
    industries: np.ndarray,
    amounts: np.ndarray,
    approved: np.ndarray,
    limit: int = 5,
) -> list[tuple[str, float]]:
    if not approved.any():
        return []
    mask = approved.astype(bool)
    df = pd.DataFrame({"industry": industries[mask], "amount": amounts[mask]})
    ranking = (
        df.groupby("industry", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(limit)
    )
    return list(zip(ranking["industry"], ranking["amount"].astype(float)))


def _duckdb_glob(path: Path) -> str:
    return (path / "**" / "*.parquet").as_posix()


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix != ".parquet":
        raise ValueError(f"Unsupported file type for counting rows: {path}")
    return pq.ParquetFile(path).metadata.num_rows