"""Train baseline and automated models for merchant churn."""
from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tpot import TPOTClassifier
except ImportError:  # pragma: no cover - optional dependency
    TPOTClassifier = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = DATA_DIR / "ml"
TRAINING_TABLE_PATH = ML_DIR / "training_table.parquet"
MODELS_DIR = ML_DIR / "models"

RANDOM_STATE = 42
logger = logging.getLogger(__name__)
TPOT_MAX_SAMPLE_SIZE = 200_000
TPOT_RECENT_WEEKS = 26
TPOT_MAX_EVAL_MINS = 15


def _tpot_supported_params() -> Set[str]:
    if TPOTClassifier is None:
        return set()
    signature = inspect.signature(TPOTClassifier.__init__)
    return set(signature.parameters.keys())


def _create_local_dask_client() -> Optional[object]:
    try:
        from distributed import Client
    except ImportError:
        logger.warning("distributed is not installed; TPOT will run on a single core")
        return None
    try:
        n_cpus = os.cpu_count() or 2
        n_workers = max(1, min(2, n_cpus - 1))
        client = Client(
            processes=False,
            n_workers=n_workers,
            threads_per_worker=2,
            silence_logs=True,
        )
        return client
    except Exception as err:
        logger.warning("Failed to start local Dask client (%s); continuing without distributed", err)
        return None


def _build_tpot_kwargs(
    max_time_mins: int,
    params: Optional[Set[str]] = None,
    client: Optional[object] = None,
) -> Dict[str, object]:
    if params is None:
        params = _tpot_supported_params()
    kwargs: Dict[str, object] = {
        "max_time_mins": max_time_mins,
        "random_state": RANDOM_STATE,
    }
    if "cv" in params:
        kwargs["cv"] = 3
    if "n_jobs" in params:
        kwargs["n_jobs"] = -1
    if "max_eval_time_mins" in params:
        kwargs["max_eval_time_mins"] = TPOT_MAX_EVAL_MINS
    if "metric" in params:
        kwargs["metric"] = "roc_auc"
    elif "scoring" in params:
        kwargs["scoring"] = "roc_auc"
    if "verbose" in params:
        kwargs["verbose"] = 2
    elif "verbosity" in params:
        kwargs["verbosity"] = 2
    if client is not None and "client" in params:
        kwargs["client"] = client
    return kwargs


@dataclass
class TrainArtifacts:
    label: str
    feature_names: List[str]
    models: Dict[str, object]
    model_paths: Dict[str, Path]
    predictions: Dict[str, np.ndarray]
    X_test: pd.DataFrame
    y_test: np.ndarray
    test_meta: pd.DataFrame
    train_range: Dict[str, str]
    test_range: Dict[str, str]
    rows: Dict[str, int]
    split_date: pd.Timestamp


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
    raise AttributeError("Model does not expose predict_proba or decision_function")


def _determine_split(df: pd.DataFrame, explicit_date: Optional[str]) -> pd.Timestamp:
    if explicit_date:
        return pd.to_datetime(explicit_date)
    latest_week = df["week_start_date"].max()
    return (latest_week - pd.Timedelta(weeks=8)).normalize()


def _build_models() -> Dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=20,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=0.05, max_depth=3),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
    }


def train_models(
    training_table_path: Path = TRAINING_TABLE_PATH,
    label: str = "churn_4w",
    train_end_date: Optional[str] = None,
    enable_tpot: bool = False,
    tpot_max_time_mins: int = 5,
) -> TrainArtifacts:
    """Train models using a time-based split and persist them under data/ml/models."""
    if not training_table_path.exists():
        raise FileNotFoundError(f"Training table not found at {training_table_path}")

    df = pd.read_parquet(training_table_path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    if label not in df.columns:
        raise KeyError(f"Label {label} not found in training table")

    df = df.dropna(subset=[label])
    if df.empty:
        raise ValueError("No rows available after dropping NA labels")

    drop_cols = {"merchant_id", "week_start_date", "churn_4w", "rev_decline_40pct"}
    feature_names = [c for c in df.columns if c not in drop_cols]

    split_date = _determine_split(df, train_end_date)
    if (df["week_start_date"] > split_date).sum() == 0:
        split_date = df["week_start_date"].quantile(0.8)
        logger.warning("Adjusted split date to %s to ensure holdout rows", split_date.date())

    train_df = df[df["week_start_date"] <= split_date]
    test_df = df[df["week_start_date"] > split_date]

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced empty train or test set")

    X_train = train_df[feature_names]
    y_train = train_df[label].astype(int)
    X_test = test_df[feature_names]
    y_test = test_df[label].astype(int)

    models = _build_models()
    predictions: Dict[str, np.ndarray] = {}
    model_paths: Dict[str, Path] = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        preds = _predict_proba(model, X_test)
        predictions[name] = preds
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_paths[name] = model_path

    if enable_tpot:
        if TPOTClassifier is None:
            raise ImportError("tpot is not installed; add it to requirements or disable --tpot")
        logger.info("Running TPOT search (max %s mins)", tpot_max_time_mins)
        tpot_train_df = train_df
        recent_cutoff = train_df["week_start_date"].max() - pd.Timedelta(weeks=TPOT_RECENT_WEEKS)
        recent_slice = train_df[train_df["week_start_date"] >= recent_cutoff]
        if len(recent_slice) >= 5000:
            tpot_train_df = recent_slice
        if len(tpot_train_df) > TPOT_MAX_SAMPLE_SIZE:
            tpot_train_df = tpot_train_df.sample(n=TPOT_MAX_SAMPLE_SIZE, random_state=RANDOM_STATE)
        tpot_X_train = tpot_train_df[feature_names]
        tpot_y_train = tpot_train_df[label].astype(int)
        logger.info(
            "TPOT training rows: %s (%.1f%% of time-eligible data)",
            len(tpot_train_df),
            100 * len(tpot_train_df) / max(1, len(train_df)),
        )
        params = _tpot_supported_params()
        tpot_client = _create_local_dask_client()
        tpot_common_kwargs = _build_tpot_kwargs(tpot_max_time_mins, params=params, client=tpot_client)
        try:
            try:
                tpot = TPOTClassifier(**tpot_common_kwargs)
            except TypeError as err:
                logger.warning("TPOT instantiation failed (%s); retrying with minimal args", err)
                minimal_kwargs = {"max_time_mins": tpot_max_time_mins, "random_state": RANDOM_STATE}
                for key in ("cv", "n_jobs", "metric", "scoring", "verbose", "verbosity", "client"):
                    if key in tpot_common_kwargs:
                        minimal_kwargs[key] = tpot_common_kwargs[key]
                tpot = TPOTClassifier(**minimal_kwargs)

            best_pipeline = None
            try:
                tpot.fit(tpot_X_train.values, tpot_y_train.values)
                best_pipeline = tpot.fitted_pipeline_
            except Exception as err:  # pragma: no cover - TPOT instability
                logger.warning("TPOT search failed after %s mins (%s); skipping automated model", tpot_max_time_mins, err)

            if best_pipeline is not None:
                tpot_name = "tpot"
                preds = _predict_proba(best_pipeline, X_test.values)
                predictions[tpot_name] = preds
                tpot_path = MODELS_DIR / f"{tpot_name}.joblib"
                joblib.dump(best_pipeline, tpot_path)
                models[tpot_name] = best_pipeline
                model_paths[tpot_name] = tpot_path
        finally:
            if tpot_client is not None:
                tpot_client.close()

    train_range = {
        "start": str(train_df["week_start_date"].min().date()),
        "end": str(train_df["week_start_date"].max().date()),
    }
    test_range = {
        "start": str(test_df["week_start_date"].min().date()),
        "end": str(test_df["week_start_date"].max().date()),
    }

    artifacts = TrainArtifacts(
        label=label,
        feature_names=feature_names,
        models=models,
        model_paths=model_paths,
        predictions=predictions,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.values,
        test_meta=test_df[["merchant_id", "week_start_date"]].reset_index(drop=True),
        train_range=train_range,
        test_range=test_range,
        rows={"train": len(train_df), "test": len(test_df)},
        split_date=pd.to_datetime(split_date),
    )
    return artifacts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
