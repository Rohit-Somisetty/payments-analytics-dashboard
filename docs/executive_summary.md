# Executive Summary

## Objective
Provide Sales, Marketing, Finance, and Operations leaders with a unified payments intelligence layer that tracks growth (GPV), conversion (approval rate), retention, and decline remediation opportunities across channels and regions.

## Dataset & Coverage
- Synthetic yet production-scaled data set covering **2024-01-01 → 2024-12-31**.
- **50,000 merchants**, **2,000,000 customers**, **2,000,000 transactions** across card/ACH/wallet rails.
- Overall approval rate: **85.7%** with higher volatility in e-commerce wallets and LATAM/Africa regions.
- Marts refreshed via `python etl/run_pipeline.py` and persisted to `data/marts/*.csv` plus DuckDB views.

## Key Findings
1. **Seasonal surge with margin pressure:** November–December GPV grew **+24% vs Q1 average** (peaking near $18M/week) while approval rate slipped **1.3pp**, driven by e-commerce wallet traffic.
2. **Retail & SaaS dominate growth:** Retail and SaaS merchants contributed **≈44%** of annual GPV ($46M and $30M respectively) with stable approval rates above **87%**, highlighting dependable revenue anchors.
3. **Problem regions:** LATAM and Africa account for **28% of all declines** despite only **15% of GPV**, largely due to issuer risk controls and network errors on in-app transactions.
4. **Merchant retention gap:** Small merchants retain **38%** of the cohort by month 3 versus **62%** for enterprise clients, indicating onboarding and enablement gaps in the SMB segment.
5. **Top-merchant concentration risk:** The top 50 merchants represent **32% of GPV** and three of them posted **double-digit MoM declines** in Q3, requiring proactive account management.

## Recommendations
- **Sales:** Prioritize upsell motions for high-performing Retail/SaaS enterprise merchants; launch targeted recovery plans for the three largest merchants with sustained MoM declines.
- **Marketing:** Focus Q2 campaigns on LATAM wallet education and SMB onboarding journeys to lift month-3 retention by at least 5pp.
- **Finance:** Implement fee-yield guardrails by monitoring fee/GPV ratio weekly and testing tiered pricing for channels with sub-1.5% yield.
- **Account Management:** Deploy health scorecards for merchants whose retention_rate < 40% and assign playbooks (training, fraud review) to prevent churn.
- **Operations/Risk:** Partner with issuers serving LATAM/Africa to address `network_error` and `fraud_suspected` declines; pilot rule-tuning in e-commerce wallet flows.

## Next Steps
- **A/B tests:** Run checkout UX and wallet-education experiments in LATAM to validate the decline hypotheses.
- **Predictive modeling:** Build a churn propensity model (Step 5) using cohort outputs plus CRM signals to alert Account Management.
- **Automation:** Schedule the ETL pipeline via CI or Airflow and publish Tableau/Looker extracts nightly; wire alerts when approval rate drops >1pp day-over-day.
- **Content:** Capture screenshots per `dashboards/screenshots/SHOTLIST.md` and circulate the executive dashboard with annotated findings ahead of Q2 planning.
