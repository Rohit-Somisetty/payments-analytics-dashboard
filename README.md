# Payments Analytics & Executive Dashboard

A production-style payments intelligence stack that consolidates acquisition, conversion, and retention insights for Sales, Marketing, Finance, Risk, and Operations. The repo shows how one team can own synthetic data generation, data quality, DuckDB modeling, KPI marts, and Tableau/Looker dashboards from a single codebase.

## Tech Stack
- **Python 3.11** for ETL orchestration and synthetic data generation.
- **DuckDB + SQL** for schema management, KPI views, and window-function analytics.
- **PyArrow / Parquet** for efficient columnar storage and hive-style partitions.
- **Tableau / Looker Studio / Looker** for dashboard delivery (with Python notebook fallback).
- **Rich documentation** (Markdown specs, run summaries, shot lists) for executive storytelling.

## Architecture Overview
```
Raw sources → Python ETL → DuckDB (analytics.duckdb) → KPI marts (CSV/Parquet) → Tableau / Looker dashboards
```
- **Data layer:** `/data/raw`, `/data/processed`, and `/data/marts` isolate landing, staging, and consumption tiers (only `.gitkeep` files tracked).
- **Compute:** `etl/` modules configure paths via `pathlib`, generate synthetic merchants/customers/transactions, run DuckDB DDL/DML, and export marts.
- **BI layer:** `sql/kpis.sql`, `dashboards/dashboard_spec.md`, and `dashboards/README_dashboard.md` describe exactly how dashboards consume the marts.
- **Narrative:** `docs/executive_summary.md` and `dashboards/screenshots/SHOTLIST.md` keep the story stakeholder-ready.

## Business Questions Answered
- How are GPV, approval rates, and fee yield trending week-over-week and month-over-month?
- Which merchant segments, industries, or regions are driving outperformance or risk?
- Where are declines concentrated (reason × channel), and what remediation actions matter most?
- What is the retention health of newly onboarded merchants/cohorts, and which account managers need to intervene?
- How concentrated is revenue in top merchants, and where do we see early signs of churn?

## Local Setup & Execution
1. Install Python 3.11 and clone this repository.
2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run the end-to-end pipeline (customize `--rows`, `--start-date`, `--end-date` as needed):
   ```bash
   python etl/run_pipeline.py --rows 2000000
   ```
4. Output artifacts:
   - Synthetic Parquet under `data/raw/` (Hive partitions).
   - DuckDB database at `data/analytics.duckdb`.
   - KPI marts at `data/marts/*.csv`.
   - Run summary appended to `docs/run_summary.md`.
5. Review the dashboard spec/build guide and connect Tableau/Looker to the refreshed marts.

## Dashboard Layer & Screenshots
- **Specification:** [dashboards/dashboard_spec.md](dashboards/dashboard_spec.md) covers stakeholders, KPI tiles, visuals, calculations, drilldowns, and acceptance criteria.
- **Build Guide:** [dashboards/README_dashboard.md](dashboards/README_dashboard.md) documents Tableau CSV/ODBC steps and Looker/Looker Studio modeling patterns.
- **Shotlist:** [dashboards/screenshots/SHOTLIST.md](dashboards/screenshots/SHOTLIST.md) lists the required captures — use placeholders such as [01_exec_overview.png](dashboards/screenshots/01_exec_overview.png), [02_exec_overview_filters.png](dashboards/screenshots/02_exec_overview_filters.png), ..., [07_monthly_growth_trends.png](dashboards/screenshots/07_monthly_growth_trends.png) once screenshots are exported.
- **Portfolio Narrative:** Mirrors a Chase Payment Solutions engagement where Analytics partners with Sales/Marketing/Finance to deliver trustworthy KPIs, decline insights, and retention playbooks.

## What This Demonstrates
- End-to-end ownership: synthetic data, QA, modeling, KPI views, dashboards, and executive storytelling inside one repo.
- Modern analytics engineering practices: parameterized ETL, DuckDB SQL-as-code, partitioned Parquet, reproducible marts.
- BI enablement: detailed Tableau/Looker build scripts, screenshot governance, and stakeholder-ready messaging.
- Engineering hygiene: `pathlib` everywhere, type hints, docstrings, and automated summary artifacts for audits.

See [PROJECT_NOTES.md](PROJECT_NOTES.md) for guidance on synthetic data expectations, regeneration commands, and why KPI views precompute metrics for BI tooling.

## Roadmap
- Add CAC/LTV and profitability KPIs plus anomaly detection notebooks.
- Parameterize Looker Studio community visualizations for faster UAT.
- Schedule ETL via GitHub Actions / Airflow and publish nightly extracts.
- Automate executive summary narration and screenshot capture.
