-- Decline rate matrix by industry and region with contribution ranking.
WITH declines AS (
    SELECT
        m.industry,
        m.region,
        SUM(CASE WHEN t.approved THEN 0 ELSE 1 END) AS declined_txns,
        COUNT(*) AS total_txns
    FROM transactions t
    JOIN merchants m USING (merchant_id)
    GROUP BY 1, 2
)
SELECT
    industry,
    region,
    declined_txns,
    total_txns,
    declined_txns::DOUBLE / NULLIF(total_txns, 0) AS decline_rate,
    RANK() OVER (ORDER BY declined_txns DESC) AS decline_rank
FROM declines
ORDER BY decline_rank;
