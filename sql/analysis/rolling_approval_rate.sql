-- Rolling seven-day approval rate with window functions.
WITH daily AS (
    SELECT
        txn_date,
        SUM(CASE WHEN approved THEN 1 ELSE 0 END) AS approved_txns,
        COUNT(*) AS total_txns
    FROM transactions
    GROUP BY txn_date
), metrics AS (
    SELECT
        txn_date,
        approved_txns::DOUBLE / NULLIF(total_txns, 0) AS daily_approval
    FROM daily
)
SELECT
    txn_date,
    daily_approval,
    AVG(daily_approval) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS approval_rate_7d
FROM metrics
ORDER BY txn_date;
