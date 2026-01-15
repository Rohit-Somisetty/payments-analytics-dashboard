# Project Notes

## Synthetic Data Philosophy
- Synthetic merchants, customers, and transactions are generated via `etl/generate_data.py` to mimic realistic payments pipelines (GPV, approval rates, fee structures, declines, merchant cohorts, and account managers).
- `faker`-style randomness is avoided; instead, distributions are derived from configurable defaults in `etl/config.py` so KPI trends stay reproducible for storytelling.
- Each run produces deterministic partitions keyed by `ingest_batch_id` and by hive-style `dt` directories, ensuring downstream data quality checks can assert record counts per slice.

## Regeneration & Reset Instructions
- Delete the `data/raw/`, `data/processed/`, `data/marts/`, and `data/analytics.duckdb` artifacts to force a clean rebuild.
- Run `python etl/run_pipeline.py --rows <N> --start-date YYYY-MM-DD --end-date YYYY-MM-DD` to repopulate all layers (defaults cover ~24 months of data if not supplied).
- Quality checks run automatically; inspect `docs/run_summary.md` to confirm row counts and KPI metrics before refreshing dashboards.

## KPI View Strategy
- All dashboard-facing metrics are materialized in DuckDB via `sql/kpis.sql` so Tableau/Looker can query wide tables without recomputing window functions.
- Views intentionally precompute week-over-week and month-over-month growth, rolling 28-day GPV, and decline rate decompositions to keep BI tools lightweight and to standardize executive definitions.
- Exports in `data/marts/*.csv` mirror each view (daily, monthly, cohort, heatmap, decline_drivers) and are regenerated on every pipeline run; downstream tools should treat them as immutable outputs per run.

## Screenshot Governance
- The `dashboards/screenshots/SHOTLIST.md` file documents the scenes required for portfolio reviews; keep PNGs named `01_*`, `02_*`, etc. for consistent ordering.
- PNGs themselves stay untracked to avoid bloating the repo—store them externally or re-export after each pipeline run using the latest marts.

## Git Hygiene
- `.gitignore` keeps heavy artifacts out of version control while preserving `.gitkeep` placeholders and the shotlist; verify `git status` is clean before publishing.
- Run `pytest` (unit layer) or the full pipeline before commits so documentation stays truthful about data freshness and KPI availability.
