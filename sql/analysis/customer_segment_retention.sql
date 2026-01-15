-- Customer retention by segment using cohort-style month indexes.
WITH cohort AS (
    SELECT
        customer_id,
        date_trunc('month', join_date) AS join_month,
        segment
    FROM customers
), activity AS (
    SELECT DISTINCT
        customer_id,
        date_trunc('month', txn_date) AS activity_month
    FROM transactions
    WHERE approved
), cohort_activity AS (
    SELECT
        c.segment,
        c.join_month,
        datediff('month', c.join_month, a.activity_month) AS month_index,
        a.customer_id
    FROM cohort c
    JOIN activity a USING (customer_id)
    WHERE a.activity_month >= c.join_month
), cohort_size AS (
    SELECT segment, join_month, COUNT(*) AS customers
    FROM cohort
    GROUP BY 1, 2
)
SELECT
    ca.segment,
    ca.join_month,
    ca.month_index,
    COUNT(DISTINCT ca.customer_id) AS retained_customers,
    COUNT(DISTINCT ca.customer_id)::DOUBLE / cs.customers AS retention_rate
FROM cohort_activity ca
JOIN cohort_size cs USING (segment, join_month)
GROUP BY 1, 2, 3, cs.customers
ORDER BY 1, 2, 3;
