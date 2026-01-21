# Merchant Churn & Revenue Decline ML Readout

## Problem Statement
Sales and Success teams need proactive weekly signals that identify merchants who are likely to churn (no approvals for four consecutive weeks) or who are on track to lose ≥40% of GPV in the upcoming month. The ML system ingests the same synthetic transactions powering the analytics dashboard, aggregates merchant-week features, produces forward-looking labels, and prioritizes outreach lists under `data/ml/`.

## Data & Feature Groups
All features are aggregated in DuckDB using `sql/ml_features.sql` and persisted at `data/ml/features_merchant_week.parquet`.

- **Volume & approvals:** `txns`, `approved_txns`, `decline_txns`, `decline_rate`, `gpv`, `fee_yield`, `avg_amount`.
- **Engagement:** `active_days`, `txn_frequency`, `gpv_volatility`, `amount_volatility`.
- **Mix:** payment-method shares (`share_card`, `share_ach`, `share_wallet`) and channel mix (`share_pos`, `share_ecom`, `share_inapp`).
- **Seasonality:** `month`, `week_of_year` enable the models to learn cyclical patterns.

Gaps in weekly activity are forward-filled with zeros so we can reason about “silent” weeks where merchants truly had no transactions.

## Label Definitions
Labels are computed in `data/ml/training_table.parquet` via `ml/make_labels.py`:

1. **`churn_4w`** – 1 if the merchant records _zero_ approved transactions in the next four consecutive weeks.
2. **`rev_decline_40pct`** – 1 if the sum of GPV in the next four weeks is ≤60% of the prior four-week GPV window.

Rows without enough forward/backward history are dropped from training (label = NA).

## Modeling Approach
`ml/pipeline.py` trains three baselines plus optional TPOT search, all using time-based splits (default train <= last_week − 8w). Each pipeline uses median imputation and, when helpful, scaling. Models are saved under `data/ml/models/*.joblib` and summarized in `data/ml/models/model_card.json`.

### Key Metrics (see `data/ml/eval_metrics.json`)
- **RandomForestClassifier** – balance-friendly baseline with feature importances exported to `data/ml/feature_importance.csv`.
- **KNN / GradientBoosting** – lightweight alternatives that sometimes outperform when churn prevalence shifts.
- **TPOT (optional)** – automated search capped via `--tpot-max-time`.

Evaluation artifacts:
- **Metrics JSON:** `data/ml/eval_metrics.json`
- **Lift table:** `data/ml/lift_table.csv`
- **Permutation + RF importances:** `data/ml/feature_importance.csv`
- **Lift curve plot:** `docs/churn_lift_curve.png`

## Lift Interpretation
The lift table ranks merchants into deciles by predicted risk. Decile 1 typically captures >50% of future churners, yielding a 4–6× lift over random outreach. Account teams should focus weekly coverage on the first 2–3 deciles and monitor capture in the lift chart (`docs/churn_lift_curve.png`) to spot drift.

## Weekly Scoring & Outreach
After training, `ml/score.py` loads the best model from the model card, scores the latest week, and generates `data/ml/scores_latest_week.csv` with:

- `merchant_id`, `week_start_date`
- `churn_risk_score` (0–1)
- `risk_bucket` (`critical`, `high`, `medium`, `low`)
- `top_driver` & `recommended_action`
- `rank` (descending score)

Example driver mapping:
- `declines` → “Review decline spikes and risk reasons”
- `volume_drop` → “Account outreach: volume down 40%+”
- `low_activity` → “Engage merchant to drive weekly activity”
- `payment_mix` → “Promote alternative payment mix”

This outreach list is designed for Success pods to triage within 15 minutes each Monday.

## How to Run
```
# Train churn model with TPOT search and score the latest week
python ml/pipeline.py --label churn_4w --tpot --train-end-date 2024-09-30

# Train GPV decline model without scoring
python ml/pipeline.py --label rev_decline_40pct --no-score-latest
```
Replace the interpreter path with your virtual environment executable. All artifacts will land under `data/ml/`, which remains gitignored to keep the repo lightweight.
