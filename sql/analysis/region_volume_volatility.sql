-- Region-level GPV volatility using standard deviation windowing.
WITH region_daily AS (
    SELECT
        m.region,
        t.txn_date,
        SUM(t.amount) FILTER (WHERE t.approved) AS gpv
    FROM transactions t
    JOIN merchants m USING (merchant_id)
    GROUP BY 1, 2
)
SELECT
    region,
    txn_date,
    gpv,
    AVG(gpv) OVER (PARTITION BY region ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS gpv_avg_7d,
    STDDEV_SAMP(gpv) OVER (PARTITION BY region ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS gpv_stddev_7d
FROM region_daily
ORDER BY region, txn_date;
