# Payments Analytics & Merchant Churn ML Platform

Production-ready analytics engineering plus applied ML for payments. This repo combines synthetic data generation, DuckDB marts, curated Looker/Tableau dashboards, and a merchant churn / revenue-decline modeling workflow so a single data team can prove end-to-end ownership.

## Architecture Overview
```
Raw Data → Python ETL → DuckDB (analytics.duckdb) → KPI Marts → Looker/Tableau Dashboards → Merchant Churn ML Models
```
- **Raw + ETL:** `etl/run_pipeline.py` synthesizes merchants/customers/transactions, enforces schema defined in `sql/schema.sql`, and persists to DuckDB + Parquet.
- **KPI marts:** `sql/kpis.sql` materializes executive metrics that downstream dashboards expect.
- **Dashboards:** Specs and build guides under `dashboards/` document Tableau/Looker tiles, filters, drilldowns, and screenshot governance.
- **ML system:** `ml/` modules create merchant-week features straight from DuckDB, label churn and revenue-decline events, train/evaluate sklearn baselines plus TPOT search, and ship outreach lists.

## Tech Stack
- Python 3.11, `rich` CLI logging, and type-hinted modules
- DuckDB + SQL + PyArrow/Parquet for analytics storage
- Looker Studio / Tableau for BI delivery
- scikit-learn, Gradient Boosting, Random Forest, KNN
- TPOT for automated pipeline search, matplotlib for lift curves

## Project 1 — Payments Analytics Dashboard
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Refresh synthetic data, DuckDB, and KPI marts:
   ```bash
   python etl/run_pipeline.py
   ```
3. Connect Tableau/Looker to `data/marts/` (CSV) or directly to `data/analytics.duckdb`, following [dashboards/dashboard_spec.md](dashboards/dashboard_spec.md) and [dashboards/README_dashboard.md](dashboards/README_dashboard.md).

<img width="1197" height="723" alt="Executive Overview" src="https://github.com/user-attachments/assets/688f5352-09c4-4cbf-8f57-ee1e1bb3a936" />

## Project 2 — Merchant Churn ML System
1. Ensure DuckDB is current (run the ETL step above if needed).
2. Launch the ML workflow (default label = churn_4w, TPOT enabled):
   ```bash
   python ml/pipeline.py --label churn_4w --tpot
   ```
   Useful flags: `--label rev_decline_40pct`, `--train-end-date YYYY-MM-DD`, `--no-score-latest`, or omit `--tpot` for faster runs.
3. Outputs (gitignored under `data/ml/`): merchant-week features, labels, trained `.joblib` files + `model_card.json`, evaluation metrics/lift tables, feature importances, and `scores_latest_week.csv` outreach lists. Stakeholder summary lives in [docs/churn_model_readout.md](docs/churn_model_readout.md) with the lift plot at [docs/churn_lift_curve.png](docs/churn_lift_curve.png).

## Dashboards & ML Outputs
- **Executive dashboards:** KPI tiles (GPV, approvals, fee yield), decline drilldowns, cohort retention, and playbooks defined in `dashboards/`. Screenshot placeholders are governed by [dashboards/screenshots/SHOTLIST.md](dashboards/screenshots/SHOTLIST.md).
- **ML artifacts:** Weekly churn-risk rankings with driver-aware recommendations (`scores_latest_week.csv`), evaluation JSON + lift CSV for enablement, and feature importance exports to guide AM playbooks.

## Synthetic Data Only
Every dataset in this repository is programmatically generated for demo purposes. **All data is synthetic. No real payment or customer data.**

## What This Demonstrates (Recruiter Cliff Notes)
- One team can own ETL, DuckDB modeling, KPI marts, dashboards, and ML scoring in a single repo.
- Strong engineering hygiene: parameterized pipelines, pathlib paths, docstrings, tests, gitignored data, and reproducible environments.
- Applied ML that respects business context (time-based splits, TPOT automation, lift/feature importance deliverables) and turns into outreach-ready action.
- Portfolio-ready documentation for executives (specs, run summaries, dashboards, churn model readout) plus automation hooks for future CI/CD.

See [PROJECT_NOTES.md](PROJECT_NOTES.md) for additional guidance on synthetic data controls and regeneration commands.
