# Metrics Dictionary (Draft)

| Metric Name | Definition | Grain | Owner | Notes |
| --- | --- | --- | --- | --- |
| Gross Payment Volume (GPV) |  | Merchant x Day | Finance Ops | TODO: derive from `transactions.amount` sums |
| Approval Rate |  | Merchant x Payment Method x Day | Risk | TODO: success / attempts with PSP reason codes |
| Retention Cohort % |  | Acquisition Cohort x Week | Growth Analytics | TODO: compute survival curve from customer lifecycle |
| Chargeback Ratio |  | Merchant x Month | Compliance | TODO: disputes / settled volume |
| Marketing ROI |  | Campaign x Week | Marketing Science | TODO: tie campaign spend to incremental GPV |
