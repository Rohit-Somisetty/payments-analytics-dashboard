"""Entrypoint for the Payments Analytics pipeline."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Any, Callable, Iterable

import config
import generate_data
import load_to_duckdb
import quality_checks

NUM_MERCHANTS = 50_000
NUM_CUSTOMERS = 2_000_000
KPI_VIEWS = (
    "v_kpi_daily",
    "v_kpi_monthly",
    "v_industry_heatmap",
    "v_merchant_cohorts",
    "v_top_merchants",
    "v_decline_drivers",
)
MART_EXPORTS = {
    "v_kpi_daily": "kpi_daily.csv",
    "v_kpi_monthly": "kpi_monthly.csv",
    "v_industry_heatmap": "industry_heatmap.csv",
    "v_merchant_cohorts": "merchant_cohorts.csv",
    "v_top_merchants": "top_merchants.csv",
    "v_decline_drivers": "decline_drivers.csv",
}


def log(message: str) -> None:
    """Lightweight logger for human-friendly CLI output."""
    print(f"[pipeline] {message}")


def generate_data_phase(rows: int, start: date, end: date, context: dict[str, Any]) -> None:
    """Generate synthetic datasets or reuse existing ones."""
    metadata = generate_data.ensure_data(
        num_merchants=NUM_MERCHANTS,
        num_customers=NUM_CUSTOMERS,
        num_transactions=rows,
        start_date=start,
        end_date=end,
        logger=log,
    )
    context["run_metadata"] = metadata


def run_quality_checks() -> None:
    """Execute data quality validations on the raw parquet files."""
    log("Running data quality checks")
    quality_checks.run_all()
    log("Quality checks passed")


def write_run_summary(context: dict[str, Any]) -> None:
    """Persist the run summary for downstream consumers."""
    metadata = context.get("run_metadata")
    if metadata is None:
        metadata = generate_data.summarize_existing_data(logger=log)
        context["run_metadata"] = metadata
    duckdb_summary = generate_data.DuckDBSummary(
        db_path=config.DUCKDB_PATH,
        views=list(context.get("kpi_views", [])),
        mart_rows=context.get("mart_rows", {}),
    )
    summary_path = generate_data.write_run_summary(metadata, duckdb_summary=duckdb_summary)
    log(f"Wrote run summary to {summary_path.relative_to(config.BASE_DIR)}")


def load_to_duckdb_phase() -> None:
    """Load merchants, customers, and transactions into DuckDB."""
    log("Loading data into DuckDB")
    con = load_to_duckdb.connect_duckdb(config.DUCKDB_PATH)
    try:
        load_to_duckdb.create_schema_tables(con)
        # Clear dependent fact table first to avoid FK violations during dimension reloads.
        con.execute("DELETE FROM transactions")
        load_to_duckdb.load_dimensions(con)
        load_to_duckdb.load_transactions(con)
    finally:
        con.close()
    log("DuckDB tables refreshed")


def build_kpis_phase(context: dict[str, Any]) -> None:
    """Create KPI views from the DuckDB SQL script."""
    log("Building KPI views")
    sql_text = config.KPI_SQL_PATH.read_text(encoding="utf-8")
    con = load_to_duckdb.connect_duckdb(config.DUCKDB_PATH)
    try:
        con.execute(sql_text)
    finally:
        con.close()
    context["kpi_views"] = KPI_VIEWS
    log("KPI views created")


def export_marts_phase(context: dict[str, Any]) -> None:
    """Export KPI views to CSV marts for downstream dashboards."""
    log("Exporting KPI marts to CSV")
    config.DATA_MARTS_DIR.mkdir(parents=True, exist_ok=True)
    mart_rows: dict[str, int] = {}
    con = load_to_duckdb.connect_duckdb(config.DUCKDB_PATH)
    try:
        for view, filename in MART_EXPORTS.items():
            output_path = config.DATA_MARTS_DIR / filename
            if output_path.exists():
                output_path.unlink()
            con.execute(
                f"COPY (SELECT * FROM {view}) TO '{output_path.as_posix()}' (HEADER, DELIMITER ',')"
            )
            row_count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            mart_rows[filename] = int(row_count)
            log(f"Exported {view} -> {filename} ({row_count:,} rows)")
    finally:
        con.close()
    context["mart_rows"] = mart_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the payments analytics pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned steps without touching the filesystem",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2_000_000,
        help="Number of synthetic transactions to generate",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=date(2024, 1, 1),
        help="Inclusive start date for transactions (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=date(2024, 12, 31),
        help="Inclusive end date for transactions (YYYY-MM-DD)",
    )
    return parser.parse_args()


def execute_phases(phases: Iterable[tuple[str, Callable[[], None]]], dry_run: bool) -> None:
    """Run each pipeline phase sequentially."""
    for name, fn in phases:
        log(f"Starting phase: {name}")
        if dry_run:
            log(f"Dry run enabled. Skipping execution of {name}.")
            continue
        fn()
        log(f"Completed phase: {name}")


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be a positive integer")
    config.ensure_directories()
    context: dict[str, Any] = {}
    phases = (
        (
            "generate_data",
            lambda: generate_data_phase(args.rows, args.start_date, args.end_date, context),
        ),
        ("quality_checks", run_quality_checks),
        ("load_to_duckdb", load_to_duckdb_phase),
        ("build_kpis", lambda: build_kpis_phase(context)),
        ("export_marts", lambda: export_marts_phase(context)),
        ("write_run_summary", lambda: write_run_summary(context)),
    )
    execute_phases(phases, dry_run=args.dry_run)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD format.") from exc


if __name__ == "__main__":
    main()
