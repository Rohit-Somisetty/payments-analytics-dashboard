-- Payment method contribution with channel mix and rolling averages.
WITH daily_method AS (
    SELECT
        txn_date,
        channel,
        payment_method,
        SUM(amount) FILTER (WHERE approved) AS gpv
    FROM transactions
    GROUP BY 1, 2, 3
), ranked AS (
    SELECT
        txn_date,
        channel,
        payment_method,
        gpv,
        gpv / NULLIF(SUM(gpv) OVER (PARTITION BY txn_date, channel), 0) AS share,
        AVG(gpv) OVER (
            PARTITION BY channel, payment_method
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS gpv_7d
    FROM daily_method
)
SELECT *
FROM ranked
ORDER BY txn_date, channel, gpv DESC;
