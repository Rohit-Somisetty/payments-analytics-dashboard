-- Fee take rate rolling trend using windowed averages.
WITH daily AS (
    SELECT
        txn_date,
        SUM(fee) AS total_fees,
        SUM(amount) FILTER (WHERE approved) AS approved_volume
    FROM transactions
    GROUP BY 1
)
SELECT
    txn_date,
    total_fees,
    approved_volume,
    total_fees / NULLIF(approved_volume, 0) AS take_rate,
    AVG(total_fees / NULLIF(approved_volume, 0)) OVER (
        ORDER BY txn_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
    ) AS take_rate_14d
FROM daily
ORDER BY txn_date;
