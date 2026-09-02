from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from risk_engine.gradient_boosting import (
    GradientBoostingConfig,
    GradientBoostingRiskModel,
)


DEFAULT_FEATURES_DIR = Path("ml/datasets/features")
DEFAULT_MODEL_DIR = Path("ml/models/gradient_boosting")
DEFAULT_OUTPUT_DIR = Path("ml/datasets/gbm")


def train_ticker(
    input_path: Path,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Train and persist one ticker's Phase 1 GBM model."""
    ticker = input_path.stem.upper()

    dataframe = pd.read_csv(input_path)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
        utc=True,
    )

    model = GradientBoostingRiskModel(
        config=GradientBoostingConfig(
            n_components=0.95,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )
    )

    split, report = model.fit_split(dataframe)

    model_path = (
        model_dir
        / f"{ticker}_gradient_boosting.joblib"
    )

    model.save(model_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = model.predict(split.test)
    probabilities = model.predict_proba(split.test)

    predictions_dataframe = split.test[
        ["timestamp", "target_direction_1d"]
    ].copy()

    predictions_dataframe["predicted_direction"] = predictions
    predictions_dataframe["probability_down"] = probabilities[:, 0]
    predictions_dataframe["probability_up"] = probabilities[:, 1]

    predictions_dataframe.to_csv(
        output_dir / f"{ticker}_test_predictions.csv",
        index=False,
    )

    model.explained_variance().to_csv(
        output_dir / f"{ticker}_train_pca_variance.csv",
        index=False,
    )

    model.component_importance().to_csv(
        output_dir / f"{ticker}_component_importance.csv",
        index=False,
    )

    metrics_dataframe = pd.DataFrame(
        [
            {
                "ticker": report.ticker,
                "partition": "train",
                "rows": report.train_rows,
                "accuracy": report.train_metrics.accuracy,
                "roc_auc": report.train_metrics.roc_auc,
                "log_loss": report.train_metrics.log_loss,
            },
            {
                "ticker": report.ticker,
                "partition": "validation",
                "rows": report.validation_rows,
                "accuracy": report.validation_metrics.accuracy,
                "roc_auc": report.validation_metrics.roc_auc,
                "log_loss": report.validation_metrics.log_loss,
            },
            {
                "ticker": report.ticker,
                "partition": "test",
                "rows": report.test_rows,
                "accuracy": report.test_metrics.accuracy,
                "roc_auc": report.test_metrics.roc_auc,
                "log_loss": report.test_metrics.log_loss,
            },
        ]
    )

    metrics_dataframe.to_csv(
        output_dir / f"{ticker}_metrics.csv",
        index=False,
    )

    print(f"\n=== {ticker} ===")
    print(f"Rows: {report.total_rows}")
    print(f"Train: {report.train_rows}")
    print(f"Validation: {report.validation_rows}")
    print(f"Test: {report.test_rows}")
    print(f"Train-fitted PCA components: {report.pca_components}")

    print("\nValidation:")
    print(
        f"  Accuracy: {report.validation_metrics.accuracy:.4f}"
    )
    print(
        f"  ROC-AUC:  {report.validation_metrics.roc_auc:.4f}"
    )
    print(
        f"  Log loss: {report.validation_metrics.log_loss:.4f}"
    )

    print("\nTest:")
    print(
        f"  Accuracy: {report.test_metrics.accuracy:.4f}"
    )
    print(
        f"  ROC-AUC:  {report.test_metrics.roc_auc:.4f}"
    )
    print(
        f"  Log loss: {report.test_metrics.log_loss:.4f}"
    )

    print(f"\nModel: {model_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Sentinel-AI Phase 1 Gradient Boosting "
            "models using leakage-safe chronological splits."
        )
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    """Train GBM models for every feature dataset."""
    args = parse_args()

    feature_files = sorted(
        args.features_dir.glob("*.csv")
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No feature datasets found in {args.features_dir}"
        )

    args.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for input_path in feature_files:
        train_ticker(
            input_path=input_path,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()