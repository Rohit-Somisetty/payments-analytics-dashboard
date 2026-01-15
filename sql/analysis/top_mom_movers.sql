-- Identify merchants with the largest month-over-month GPV swing.
WITH merchant_month AS (
    SELECT
        merchant_id,
        date_trunc('month', txn_date) AS month,
        SUM(amount) FILTER (WHERE approved) AS gpv
    FROM transactions
    GROUP BY 1, 2
), ranked AS (
    SELECT
        merchant_id,
        month,
        gpv,
        LAG(gpv) OVER (PARTITION BY merchant_id ORDER BY month) AS prev_gpv
    FROM merchant_month
), deltas AS (
    SELECT
        merchant_id,
        month,
        gpv,
        prev_gpv,
        CASE WHEN prev_gpv IS NULL OR prev_gpv = 0 THEN NULL ELSE (gpv - prev_gpv) / prev_gpv END AS mom_growth
    FROM ranked
)
SELECT *
FROM deltas
WHERE mom_growth IS NOT NULL
QUALIFY ABS(mom_growth) = MAX(ABS(mom_growth)) OVER ()
ORDER BY ABS(mom_growth) DESC;
