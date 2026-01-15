# Data Generation Summary

*Date range:* 2024-01-01 → 2024-12-31
*Merchants:* 50,000
*Customers:* 2,000,000
*Transactions:* 2,000,000
*Approval rate:* 85.68%
*Generated this run:* No (existing data reused)

## Top Industries by GPV
1. Retail: $46,248,001
2. SaaS: $29,578,312
3. Marketplace: $27,171,041
4. Food & Beverage: $26,741,134
5. Entertainment: $18,461,970

## DuckDB & KPI Views
*Database:* E:\WAYMO\payments-analytics-dashboard\data\analytics.duckdb
*Views:*
- v_kpi_daily
- v_kpi_monthly
- v_industry_heatmap
- v_merchant_cohorts
- v_top_merchants
- v_decline_drivers

*Exported Marts:*
- kpi_daily.csv: 366 rows
- kpi_monthly.csv: 12 rows
- industry_heatmap.csv: 35 rows
- merchant_cohorts.csv: 432 rows
- top_merchants.csv: 50,000 rows
- decline_drivers.csv: 126 rows