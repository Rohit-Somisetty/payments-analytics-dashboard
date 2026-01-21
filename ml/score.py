"""Score the latest merchant-week and generate an outreach list."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = DATA_DIR / "ml"
FEATURES_PATH = ML_DIR / "features_merchant_week.parquet"
MODEL_CARD_PATH = ML_DIR / "models" / "model_card.json"
SCORES_PATH = ML_DIR / "scores_latest_week.csv"

logger = logging.getLogger(__name__)


def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except TypeError:
            proba = model.predict_proba(np.asarray(X))
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
        except TypeError:
            scores = model.decision_function(np.asarray(X))
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    raise AttributeError("Loaded model cannot produce probability scores")


def _risk_bucket(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _recommendation(row: pd.Series, medians: Dict[str, float]) -> tuple[str, str]:
    if row.get("decline_rate", 0) >= 0.25:
        return ("declines", "Review decline spikes and risk reasons")
    if row.get("gpv", 0) <= 0.6 * medians.get("gpv", 1):
        return ("volume_drop", "Account outreach: volume down 40%+")
    if row.get("txn_frequency", 0) < 1:
        return ("low_activity", "Engage merchant to drive weekly activity")
    if row.get("share_card", 0) > 0.9:
        return ("payment_mix", "Promote alternative payment mix")
    return ("steady", "Monitor health via CSM check-in")


def score_latest_week(
    model_card_path: Path = MODEL_CARD_PATH,
    features_path: Path = FEATURES_PATH,
    output_path: Path = SCORES_PATH,
) -> pd.DataFrame:
    """Score the most recent week and persist ranked outreach list."""
    if not model_card_path.exists():
        raise FileNotFoundError("Model card missing. Train models before scoring.")
    if not features_path.exists():
        raise FileNotFoundError("Feature table missing. Run build_features first.")

    card = json.loads(model_card_path.read_text())
    model_rel_path = card["best_model"]["path"]
    feature_names = card["feature_names"]
    model_path = BASE_DIR / model_rel_path
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found at {model_path}")

    features = pd.read_parquet(features_path)
    features["week_start_date"] = pd.to_datetime(features["week_start_date"])
    latest_week = features["week_start_date"].max()
    latest_df = features[features["week_start_date"] == latest_week].copy()
    if latest_df.empty:
        raise ValueError("No rows available for the latest week")

    model = joblib.load(model_path)
    X_latest = latest_df[feature_names]
    scores = _predict_proba(model, X_latest)
    latest_df["churn_risk_score"] = scores
    latest_df["risk_bucket"] = latest_df["churn_risk_score"].apply(_risk_bucket)

    medians = {
        "gpv": features["gpv"].median(),
    }
    recommendations = latest_df.apply(lambda row: _recommendation(row, medians), axis=1, result_type="expand")
    latest_df["top_driver"] = recommendations[0]
    latest_df["recommended_action"] = recommendations[1]
    latest_df.sort_values("churn_risk_score", ascending=False, inplace=True)
    latest_df["rank"] = np.arange(1, len(latest_df) + 1)

    outreach_cols = [
        "merchant_id",
        "week_start_date",
        "churn_risk_score",
        "risk_bucket",
        "top_driver",
        "recommended_action",
        "rank",
    ]
    outreach = latest_df[outreach_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outreach.to_csv(output_path, index=False)
    logger.info(
        "Saved latest-week scoring (%s merchants, week %s) to %s",
        len(outreach),
        latest_week.date(),
        output_path,
    )
    return outreach


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    score_latest_week()
