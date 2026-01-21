"""Generate churn and revenue decline labels for merchant-week features."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = DATA_DIR / "ml"
FEATURES_PATH = ML_DIR / "features_merchant_week.parquet"
TRAINING_TABLE_PATH = ML_DIR / "training_table.parquet"

logger = logging.getLogger(__name__)


def _expand_to_weekly(grid: pd.DataFrame) -> pd.DataFrame:
    """Ensure each merchant has contiguous weekly rows (fill zeros for gaps)."""
    if grid.empty:
        raise ValueError("Feature table is empty; nothing to label.")

    numeric_cols: List[str] = [c for c in grid.select_dtypes(include=["number"]).columns if c != "merchant_id"]
    expanded_frames: List[pd.DataFrame] = []

    for merchant_id, group in grid.groupby("merchant_id"):
        group = group.sort_values("week_start_date")
        week_index = pd.date_range(group["week_start_date"].min(), group["week_start_date"].max(), freq="W-MON")
        aligned = group.set_index("week_start_date").reindex(week_index)
        aligned["merchant_id"] = merchant_id
        aligned.index.name = "week_start_date"
        expanded_frames.append(aligned.reset_index().rename(columns={"index": "week_start_date"}))

    expanded = pd.concat(expanded_frames, ignore_index=True)
    if numeric_cols:
        expanded[numeric_cols] = expanded[numeric_cols].fillna(0)

    expanded["week_start_date"] = pd.to_datetime(expanded["week_start_date"])
    expanded["month"] = expanded["week_start_date"].dt.month.astype("int16")
    expanded["week_of_year"] = expanded["week_start_date"].dt.isocalendar().week.astype("int16")
    expanded.sort_values(["merchant_id", "week_start_date"], inplace=True)
    expanded.reset_index(drop=True, inplace=True)
    return expanded


def _compute_labels(group: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    group = group.sort_values("week_start_date").reset_index(drop=True)

    future_approved = sum(group["approved_txns"].shift(-i).fillna(0) for i in range(1, horizon + 1))
    future_complete = group["approved_txns"].shift(-horizon).notna()
    churn = np.where(future_complete & (future_approved == 0), 1.0, 0.0)
    churn[~future_complete] = np.nan
    group["churn_4w"] = churn

    past_gpv = group["gpv"].rolling(window=horizon, min_periods=horizon).sum()
    future_gpv = sum(group["gpv"].shift(-i).fillna(0) for i in range(1, horizon + 1))
    future_counts = sum(group["gpv"].shift(-i).notna().astype(int) for i in range(1, horizon + 1))
    valid_future = future_counts == horizon
    valid_past = past_gpv.notna() & (past_gpv > 0)
    decline_flag = valid_future & valid_past & (future_gpv <= 0.6 * past_gpv)
    decline = np.where(decline_flag, 1.0, 0.0)
    decline[~(valid_future & valid_past)] = np.nan
    group["rev_decline_40pct"] = decline

    return group


def make_labels(
    features_path: Path = FEATURES_PATH,
    output_path: Path = TRAINING_TABLE_PATH,
) -> pd.DataFrame:
    """Attach forward-looking churn and revenue-decline labels to the feature table."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found at {features_path}")

    logger.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)
    features["week_start_date"] = pd.to_datetime(features["week_start_date"])

    expanded = _expand_to_weekly(features)
    labeled = (
        expanded.groupby("merchant_id", group_keys=False)
        .apply(_compute_labels)
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output_path, index=False)
    logger.info("Saved labeled training table to %s (%s rows)", output_path, len(labeled))
    return labeled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_labels()
