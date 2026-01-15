-- Channel-level YoY change by region leveraging window functions.
WITH channel_year AS (
    SELECT
      m.region,
      t.channel,
      t.partition_year AS metric_year,
      SUM(t.amount) FILTER (WHERE t.approved) AS gpv
    FROM transactions t
    JOIN merchants m USING (merchant_id)
    GROUP BY 1, 2, 3
), ranked AS (
    SELECT
      region,
      channel,
      metric_year,
      gpv,
      LAG(gpv) OVER (PARTITION BY region, channel ORDER BY metric_year) AS prev_gpv
    FROM channel_year
)
SELECT
    region,
    channel,
    metric_year,
    gpv,
    CASE WHEN prev_gpv IS NULL OR prev_gpv = 0 THEN NULL ELSE (gpv - prev_gpv) / prev_gpv END AS yoy_growth
FROM ranked
ORDER BY region, channel, metric_year;
