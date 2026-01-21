-- Merchant-week aggregation query for churn ML features.
WITH weekly AS (
    SELECT
        merchant_id,
        CAST(date_trunc('week', txn_date) AS DATE) AS week_start_date,
        COUNT(*) AS txns,
        SUM(CASE WHEN approved THEN 1 ELSE 0 END) AS approved_txns,
        SUM(CASE WHEN NOT approved THEN 1 ELSE 0 END) AS decline_txns,
        SUM(CASE WHEN approved THEN amount ELSE 0 END) AS gpv,
        AVG(amount) AS avg_amount,
        SUM(CASE WHEN approved THEN fee ELSE 0 END) AS total_fees,
        COUNT(DISTINCT txn_date) AS active_days,
        STDDEV_POP(amount) AS amount_volatility,
        SUM(CASE WHEN payment_method = 'card' AND approved THEN amount ELSE 0 END) AS card_gpv,
        SUM(CASE WHEN payment_method = 'ach' AND approved THEN amount ELSE 0 END) AS ach_gpv,
        SUM(CASE WHEN payment_method = 'wallet' AND approved THEN amount ELSE 0 END) AS wallet_gpv,
        SUM(CASE WHEN channel = 'pos' AND approved THEN amount ELSE 0 END) AS pos_gpv,
        SUM(CASE WHEN channel = 'ecom' AND approved THEN amount ELSE 0 END) AS ecom_gpv,
        SUM(CASE WHEN channel IN ('in_app', 'inapp') AND approved THEN amount ELSE 0 END) AS inapp_gpv
    FROM transactions
    GROUP BY 1, 2
),

daily AS (
    SELECT
        merchant_id,
        CAST(date_trunc('week', txn_date) AS DATE) AS week_start_date,
        txn_date,
        SUM(CASE WHEN approved THEN amount ELSE 0 END) AS daily_gpv
    FROM transactions
    GROUP BY 1, 2, 3
),

volatility AS (
    SELECT
        merchant_id,
        week_start_date,
        STDDEV_POP(daily_gpv) AS gpv_volatility
    FROM daily
    GROUP BY 1, 2
)
SELECT
    w.merchant_id,
    w.week_start_date,
    w.txns,
    w.approved_txns,
    w.decline_txns,
    w.gpv,
    w.avg_amount,
    w.total_fees,
    w.active_days,
    COALESCE(w.amount_volatility, 0) AS amount_volatility,
    COALESCE(v.gpv_volatility, 0) AS gpv_volatility,
    CASE WHEN w.txns > 0 THEN w.decline_txns::DOUBLE / w.txns ELSE 0 END AS decline_rate,
    CASE WHEN w.active_days > 0 THEN w.txns::DOUBLE / w.active_days ELSE 0 END AS txn_frequency,
    CASE WHEN w.gpv > 0 THEN w.total_fees / w.gpv ELSE 0 END AS fee_yield,
    CASE WHEN w.gpv > 0 THEN w.card_gpv / w.gpv ELSE 0 END AS share_card,
    CASE WHEN w.gpv > 0 THEN w.ach_gpv / w.gpv ELSE 0 END AS share_ach,
    CASE WHEN w.gpv > 0 THEN w.wallet_gpv / w.gpv ELSE 0 END AS share_wallet,
    CASE WHEN w.gpv > 0 THEN w.pos_gpv / w.gpv ELSE 0 END AS share_pos,
    CASE WHEN w.gpv > 0 THEN w.ecom_gpv / w.gpv ELSE 0 END AS share_ecom,
    CASE WHEN w.gpv > 0 THEN w.inapp_gpv / w.gpv ELSE 0 END AS share_inapp
FROM weekly w
LEFT JOIN volatility v USING (merchant_id, week_start_date)
ORDER BY 1, 2;
