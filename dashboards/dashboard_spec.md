# Payments Analytics Dashboard Specification

## Dashboard 1 – Executive Overview
- **Purpose / Stakeholders:** Give Finance leadership and the CRO a single-glance view of volume, approvals, and revenue contribution.
- **KPI Tiles:**
  - Daily GPV (sum of `gpv` from `kpi_daily`).
  - Approval rate (approved_txns ÷ txns from `kpi_daily`).
  - Average ticket (`avg_amount`).
  - Fee yield (`total_fees ÷ gpv`).
- **Visuals:**
  - Dual-axis sparkline showing GPV vs approval rate over time (line + area) using `kpi_daily`.
  - Monthly MoM growth waterfall sourced from `kpi_monthly` with color encoding for positive/negative deltas.
  - KPI annotation card listing top 3 industries by GPV using `industry_heatmap` (latest year only).
- **Calculations:**
  - `approval_rate = approved_txns / NULLIF(txns, 0)`.
  - `mom_growth = (gpv - LAG(gpv)) / LAG(gpv)` from `kpi_monthly`.
  - Rolling 7-day approval average (Tableau table calc or LookML window function).
- **Drill-down behavior:** Selecting a month on the waterfall filters all tiles and opens the Sales Drilldown dashboard.
- **Tooltips:** Include day/month, GPV, approval rate, and delta vs prior period with arrows (▲/▼) for immediate interpretation.

## Dashboard 2 – Sales & Merchant Performance Drilldown
- **Purpose / Stakeholders:** Equip Sales and Account Management teams with merchant-level performance rankings.
- **KPI Tiles:**
  - Total merchants active (distinct `merchant_id` in `top_merchants`).
  - Share of GPV by top 10 merchants.
  - YoY growth for selected region/industry.
- **Visuals:**
  - Ranked bar chart of GPV by merchant with `gpv_rank` field; color by industry.
  - Detail table showing merchant size, region, GPV, txns, approval rate (calculated on the fly).
  - Map or filled region chart showing GPV share by region using aggregated `top_merchants` data.
- **Calculations:**
  - `approval_rate = SUM(approved_txns) / SUM(txns)` after blending in `kpi_daily` filtered to merchant.
  - `rank_percentile = 1 - (gpv_rank / WINDOW_MAX(gpv_rank))` for highlighting rising merchants.
- **Drill-down behavior:** Clicking a merchant opens a detail pop-up (Tableau action) showing daily trend from `kpi_daily` filtered to that merchant, or links to CRM record.
- **Tooltips:** Display merchant metadata (industry, channel mix, signup_month) plus MoM change from `kpi_monthly` derived field.

## Dashboard 3 – Approval & Decline Operations View
- **Purpose / Stakeholders:** Inform Risk and Payments Operations of emerging decline drivers.
- **KPI Tiles:**
  - Decline rate (overall from `decline_drivers`).
  - Declined dollars (sum of `amount` where approved = false via DuckDB view if needed).
  - Channels with highest drop vs prior week.
- **Visuals:**
  - Stacked bar chart of decline_reason by channel with percentages from `decline_drivers`.
  - Heatmap of region × channel decline rate using `industry_heatmap` filtered by channel.
  - Rolling 7-day approval trend line referencing `kpi_daily`.
- **Calculations:**
  - `decline_rate = declined_txns / total_txns` (already provided but recompute for filtered views).
  - Rolling 7d = WINDOW_AVG(approval_rate, -6, 0).
  - Contribution to decline = (declined_txns / SUM(declined_txns)).
- **Drill-down behavior:** Selecting a decline reason filters a secondary sheet listing impacted merchants from `top_merchants` with highest decline exposure.
- **Tooltips:** Show absolute counts, rates, and share of total declines plus last-updated timestamp pulled from `docs/run_summary.md` if desired.

## Dashboard 4 – Retention & Cohort Health
- **Purpose / Stakeholders:** Support Marketing/Growth with visibility into cohort stickiness.
- **KPI Tiles:**
  - Month 0 cohort size (from `merchant_cohorts`).
  - Retention rate at month_index 1, 3, 6.
  - Number of at-risk merchants (retention_rate < 40%).
- **Visuals:**
  - Cohort heatmap (signup_month vs month_index, colored by retention_rate).
  - Line chart showing cohort curves for selected signup months.
  - Bar chart summarizing retention by merchant_size and region.
- **Calculations:**
  - `retention_rate` already provided; create calculated field for drop-off = 1 - retention.
  - Weighted retention = SUM(retained_merchants) / SUM(cohort size) across filtered cohorts.
- **Drill-down behavior:** Clicking a cell in the heatmap filters a detail table listing merchants in that cohort with account manager assignments (join to CRM export when available).
- **Tooltips:** Include signup_month, month_index, retained_merchants, retention_rate %, and delta vs prior cohort.

## Acceptance Criteria
1. Executives can answer five core questions within two minutes: current GPV, approval trend, top/bottom merchants, decline hotspots, and retention health.
2. All charts respond to the global filters (date, region, industry, merchant_size, channel, payment_method) without breaking calculations.
3. Industry/region heatmap supports drilldown (Tableau action or Looker drill fields) to expose merchant-level rows.
4. KPI tiles stay in sync with selected filters and always display the latest refreshed timestamp from `docs/run_summary.md`.
5. Dashboard layout matches the screenshot checklist defined in `dashboards/screenshots/SHOTLIST.md`.
