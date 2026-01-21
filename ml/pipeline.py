"""End-to-end CLI pipeline for the merchant churn ML system."""
from __future__ import annotations

import argparse
import logging

from ml.build_features import build_features
from ml.evaluate import evaluate_models
from ml.make_labels import make_labels
from ml.score import score_latest_week
from ml.train import train_models


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merchant churn & revenue-decline ML pipeline")
    parser.add_argument(
        "--label",
        choices=["churn_4w", "rev_decline_40pct"],
        default="churn_4w",
        help="Target label to train on (default: churn_4w)",
    )
    parser.add_argument(
        "--train-end-date",
        dest="train_end_date",
        help="Inclusive week_start_date for the training split (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--tpot",
        action="store_true",
        help="Enable TPOT automated model search (default off)",
    )
    parser.add_argument(
        "--tpot-max-time",
        dest="tpot_max_time",
        type=int,
        default=5,
        help="TPOT max_time_mins budget (default: 5)",
    )
    parser.add_argument(
        "--score-latest",
        dest="score_latest",
        action="store_true",
        help="Score the latest week after evaluation (default)",
    )
    parser.add_argument(
        "--no-score-latest",
        dest="score_latest",
        action="store_false",
        help="Skip scoring the latest week",
    )
    parser.set_defaults(score_latest=True)
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger("ml.pipeline")

    logger.info("Step 1/5: Building merchant-week features")
    features_df = build_features()
    logger.info("Feature table shape: %s", features_df.shape)

    logger.info("Step 2/5: Generating labels")
    training_df = make_labels()
    logger.info("Training table shape: %s", training_df.shape)

    logger.info("Step 3/5: Training models for label=%s", args.label)
    artifacts = train_models(
        label=args.label,
        train_end_date=args.train_end_date,
        enable_tpot=args.tpot,
        tpot_max_time_mins=args.tpot_max_time,
    )

    logger.info("Step 4/5: Evaluating models")
    eval_summary = evaluate_models(artifacts)
    best_model = eval_summary["best_model_name"]
    logger.info("Best model: %s", best_model)

    if args.score_latest:
        logger.info("Step 5/5: Scoring latest week with %s", best_model)
        score_latest_week()
    else:
        logger.info("Skipping latest-week scoring")


if __name__ == "__main__":
    main()
