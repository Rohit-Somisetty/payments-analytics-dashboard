"""Evaluate trained models and persist metrics, lift, and feature importance."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from .train import TrainArtifacts

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = DATA_DIR / "ml"
DOCS_DIR = BASE_DIR / "docs"
METRICS_PATH = ML_DIR / "eval_metrics.json"
LIFT_TABLE_PATH = ML_DIR / "lift_table.csv"
FEATURE_IMPORTANCE_PATH = ML_DIR / "feature_importance.csv"
MODEL_CARD_PATH = ML_DIR / "models" / "model_card.json"
LIFT_PLOT_PATH = DOCS_DIR / "churn_lift_curve.png"

logger = logging.getLogger(__name__)


def _precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, pct: float) -> Dict[str, float]:
    if y_true.size == 0:
        return {"precision": float("nan"), "recall": float("nan")}
    cutoff = max(1, int(len(scores) * pct))
    order = np.argsort(scores)[::-1][:cutoff]
    selected = y_true[order]
    positives = y_true.sum()
    precision = float(selected.sum() / len(selected)) if len(selected) else float("nan")
    recall = float(selected.sum() / positives) if positives else 0.0
    return {"precision": precision, "recall": recall}


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    for pct in (0.01, 0.05, 0.10):
        stats = _precision_recall_at_k(y_true, scores, pct)
        suffix = f"{int(pct * 100)}pct"
        metrics[f"precision_at_{suffix}"] = stats["precision"]
        metrics[f"recall_at_{suffix}"] = stats["recall"]
    return metrics


def _build_lift_table(y_true: np.ndarray, scores: np.ndarray, deciles: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "score": scores})
    frame.sort_values("score", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)

    total_positive = frame["y_true"].sum()
    baseline = total_positive / len(frame) if len(frame) else 0
    slice_size = int(np.ceil(len(frame) / deciles)) if deciles else len(frame)

    records: List[Dict[str, Any]] = []
    cumulative_positive = 0.0

    for decile in range(1, deciles + 1):
        start = (decile - 1) * slice_size
        end = min(decile * slice_size, len(frame))
        if start >= len(frame):
            break
        bucket = frame.iloc[start:end]
        bucket_pos = bucket["y_true"].sum()
        cumulative_positive += bucket_pos
        bucket_rate = bucket_pos / len(bucket) if len(bucket) else 0
        lift = (bucket_rate / baseline) if baseline else 0
        capture = cumulative_positive / total_positive if total_positive else 0
        records.append(
            {
                "decile": decile,
                "population": len(bucket),
                "avg_score": bucket["score"].mean(),
                "positives": bucket_pos,
                "lift": lift,
                "cumulative_capture": capture,
            }
        )

    return pd.DataFrame.from_records(records)


def _extract_rf_importance(model: object, feature_names: List[str]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
    else:
        clf = getattr(model, "clf", None)
    if clf is None or not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()
    importances = clf.feature_importances_
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "importance_type": "random_forest_gini",
        }
    )


def _permutation_importance(model: object, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    try:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=8,
            random_state=42,
            scoring="roc_auc",
        )
    except Exception:
        result = permutation_importance(
            model,
            X.values,
            y,
            n_repeats=8,
            random_state=42,
            scoring="roc_auc",
        )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
            "importance_type": "permutation_best_model",
        }
    )


def _model_params(model: object) -> Dict[str, Any]:
    if hasattr(model, "get_params"):
        try:
            params = model.get_params(deep=False)
            return {k: v for k, v in params.items() if isinstance(k, str)}
        except Exception:
            return {}
    return {}


def evaluate_models(
    artifacts: TrainArtifacts,
    metrics_path: Path = METRICS_PATH,
    lift_table_path: Path = LIFT_TABLE_PATH,
    importance_path: Path = FEATURE_IMPORTANCE_PATH,
    lift_plot_path: Path = LIFT_PLOT_PATH,
    model_card_path: Path = MODEL_CARD_PATH,
) -> Dict[str, Any]:
    """Compute evaluation artifacts and persist them to disk."""
    metrics: Dict[str, Dict[str, float]] = {}
    for name, scores in artifacts.predictions.items():
        metrics[name] = _compute_metrics(artifacts.y_test, scores)

    metrics_payload = {
        "label": artifacts.label,
        "split_date": str(artifacts.split_date.date()),
        "rows": artifacts.rows,
        "train_range": artifacts.train_range,
        "test_range": artifacts.test_range,
        "models": metrics,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Wrote evaluation metrics to %s", metrics_path)

    def _roc_key(item: Any) -> float:
        value = item[1].get("roc_auc")
        if value is None or np.isnan(value):
            return float("-inf")
        return value

    best_model_name = max(metrics.items(), key=_roc_key)[0]
    best_scores = artifacts.predictions[best_model_name]
    lift_table = _build_lift_table(artifacts.y_test, best_scores)
    lift_table.to_csv(lift_table_path, index=False)
    logger.info("Saved lift table to %s", lift_table_path)

    lift_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(lift_table["decile"], lift_table["cumulative_capture"], marker="o")
    plt.title("Cumulative Capture by Decile")
    plt.xlabel("Decile (High Risk → Low Risk)")
    plt.ylabel("Capture Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(lift_plot_path, dpi=150)
    plt.close()

    rf_importance = _extract_rf_importance(artifacts.models.get("random_forest"), artifacts.feature_names)
    perm_importance = _permutation_importance(
        artifacts.models[best_model_name], artifacts.X_test, artifacts.y_test, artifacts.feature_names
    )
    feature_importance = pd.concat([rf_importance, perm_importance], ignore_index=True)
    feature_importance.sort_values("importance", ascending=False, inplace=True)
    feature_importance.to_csv(importance_path, index=False)
    logger.info("Saved feature importance to %s", importance_path)

    model_card = {
        "label": artifacts.label,
        "feature_names": artifacts.feature_names,
        "best_model": {
            "name": best_model_name,
            "path": str(artifacts.model_paths[best_model_name].as_posix()),
            "metrics": metrics[best_model_name],
            "parameters": _model_params(artifacts.models[best_model_name]),
        },
        "training": {
            "split_date": str(artifacts.split_date.date()),
            "rows": artifacts.rows,
            "train_range": artifacts.train_range,
            "test_range": artifacts.test_range,
        },
    }
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(json.dumps(model_card, indent=2))
    logger.info("Saved model card to %s", model_card_path)

    return {
        "best_model_name": best_model_name,
        "best_model_path": artifacts.model_paths[best_model_name],
        "metrics": metrics_payload,
        "lift_table": lift_table,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("Run evaluate via ml/pipeline.py after training.")
