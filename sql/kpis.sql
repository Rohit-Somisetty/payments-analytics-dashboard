CREATE OR REPLACE VIEW v_kpi_daily AS
WITH daily AS (
	SELECT
		txn_date AS day,
		SUM(amount) FILTER (WHERE approved) AS gpv,
		COUNT(*) AS txns,
		SUM(CASE WHEN approved THEN 1 ELSE 0 END) AS approved_txns,
		AVG(amount) AS avg_amount,
		SUM(fee) AS total_fees
	FROM transactions
	GROUP BY txn_date
)
SELECT
	day,
	COALESCE(gpv, 0) AS gpv,
	txns,
	CASE WHEN txns = 0 THEN NULL ELSE approved_txns::DOUBLE / txns END AS approval_rate,
	avg_amount,
	total_fees
FROM daily;


CREATE OR REPLACE VIEW v_kpi_monthly AS
WITH monthly AS (
	SELECT
		date_trunc('month', txn_date) AS month,
		SUM(amount) FILTER (WHERE approved) AS gpv,
		COUNT(*) AS txns,
		SUM(CASE WHEN approved THEN 1 ELSE 0 END) AS approved_txns
	FROM transactions
	GROUP BY 1
), ranked AS (
	SELECT
		month,
		gpv,
		txns,
		CASE WHEN txns = 0 THEN NULL ELSE approved_txns::DOUBLE / txns END AS approval_rate,
		LAG(gpv) OVER (ORDER BY month) AS prev_gpv
	FROM monthly
)
SELECT
	month,
	gpv,
	txns,
	approval_rate,
	(gpv - prev_gpv) / NULLIF(prev_gpv, 0) AS mom_growth
FROM ranked;


CREATE OR REPLACE VIEW v_industry_heatmap AS
WITH yearly AS (
	SELECT
		m.region,
		m.industry,
		t.partition_year AS metric_year,
		SUM(t.amount) FILTER (WHERE t.approved) AS gpv,
		SUM(CASE WHEN t.approved THEN 1 ELSE 0 END) AS approved_txns,
		COUNT(*) AS txns
	FROM transactions t
	JOIN merchants m USING (merchant_id)
	GROUP BY 1, 2, 3
), scored AS (
	SELECT
		region,
		industry,
		metric_year,
		gpv,
		CASE WHEN txns = 0 THEN NULL ELSE approved_txns::DOUBLE / txns END AS approval_rate,
		LAG(gpv) OVER (PARTITION BY region, industry ORDER BY metric_year) AS prev_gpv,
		ROW_NUMBER() OVER (PARTITION BY region, industry ORDER BY metric_year DESC) AS row_rank
	FROM yearly
)
SELECT
	region,
	industry,
	gpv,
	approval_rate,
	CASE
		WHEN prev_gpv IS NULL OR prev_gpv = 0 THEN NULL
		ELSE (gpv - prev_gpv) / prev_gpv
	END AS yoy_growth
FROM scored
WHERE row_rank = 1;


CREATE OR REPLACE VIEW v_merchant_cohorts AS
WITH merchant_cohort AS (
	SELECT
		merchant_id,
		date_trunc('month', signup_date) AS signup_month
	FROM merchants
), cohort_size AS (
	SELECT signup_month, COUNT(*) AS cohort_merchants
	FROM merchant_cohort
	GROUP BY 1
), merchant_activity AS (
	SELECT DISTINCT
		t.merchant_id,
		date_trunc('month', t.txn_date) AS activity_month
	FROM transactions t
	WHERE t.approved
), cohort_activity AS (
	SELECT
		mc.signup_month,
		ma.activity_month,
		ma.merchant_id,
		(datediff('month', mc.signup_month, ma.activity_month)) AS month_index
	FROM merchant_cohort mc
	JOIN merchant_activity ma USING (merchant_id)
	WHERE ma.activity_month >= mc.signup_month
)
SELECT
	ca.signup_month,
	ca.month_index,
	COUNT(DISTINCT ca.merchant_id) AS retained_merchants,
	CASE WHEN cs.cohort_merchants = 0 THEN NULL
		 ELSE COUNT(DISTINCT ca.merchant_id)::DOUBLE / cs.cohort_merchants
	END AS retention_rate
FROM cohort_activity ca
JOIN cohort_size cs USING (signup_month)
GROUP BY 1, 2, cs.cohort_merchants;


CREATE OR REPLACE VIEW v_top_merchants AS
WITH merchant_perf AS (
	SELECT
		m.merchant_id,
		m.industry,
		m.region,
		SUM(t.amount) FILTER (WHERE t.approved) AS gpv,
		COUNT(*) AS txns
	FROM transactions t
	JOIN merchants m USING (merchant_id)
	GROUP BY 1, 2, 3
)
SELECT
	merchant_id,
	industry,
	region,
	gpv,
	txns,
	DENSE_RANK() OVER (ORDER BY gpv DESC) AS gpv_rank
FROM merchant_perf;


CREATE OR REPLACE VIEW v_decline_drivers AS
WITH base AS (
	SELECT
		COALESCE(t.decline_reason, 'unknown') AS decline_reason,
		t.channel,
		m.industry,
		SUM(CASE WHEN t.approved THEN 0 ELSE 1 END) AS declined_txns,
		COUNT(*) AS total_txns
	FROM transactions t
	JOIN merchants m USING (merchant_id)
	GROUP BY 1, 2, 3
)
SELECT
	decline_reason,
	channel,
	industry,
	CASE WHEN total_txns = 0 THEN NULL ELSE declined_txns::DOUBLE / total_txns END AS decline_rate
FROM base;
