from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from risk_engine.evaluation import EvaluationReport, RiskModelEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[4]

GBM_DIR = PROJECT_ROOT / "ml" / "datasets" / "gbm"
FEATURES_DIR = PROJECT_ROOT / "ml" / "datasets" / "features"
RISK_SCORES_DIR = PROJECT_ROOT / "ml" / "datasets" / "risk_scores"

OUTPUT_DIR = GBM_DIR

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_direction_1d"
RISK_SCORE_COLUMN = "risk_score"


def _flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested dictionaries into a single-level dictionary."""

    flattened: dict[str, Any] = {}

    for key, value in data.items():
        full_key = (
            f"{prefix}_{key}"
            if prefix
            else key
        )

        if isinstance(value, dict):
            flattened.update(
                _flatten_dict(
                    value,
                    prefix=full_key,
                )
            )
        else:
            flattened[full_key] = value

    return flattened


def report_to_dict(
    report: EvaluationReport,
) -> dict[str, Any]:
    """Convert an EvaluationReport dataclass into a flat dictionary."""

    if not is_dataclass(report):
        raise TypeError(
            "EvaluationReport must be a dataclass instance."
        )

    report_dict = asdict(report)

    return _flatten_dict(report_dict)


def load_evaluation_dataset(
    ticker: str,
) -> pd.DataFrame:
    """
    Load and combine GBM predictions and risk scores.

    The GBM prediction file already contains the true target,
    so the target does not need to be merged from the feature
    dataset.
    """

    prediction_path = (
        GBM_DIR
        / f"{ticker}_test_predictions.csv"
    )

    feature_path = (
        FEATURES_DIR
        / f"{ticker}.csv"
    )

    risk_score_path = (
        RISK_SCORES_DIR
        / f"{ticker}_risk_scores.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"GBM prediction file not found: "
            f"{prediction_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{feature_path}"
        )

    if not risk_score_path.exists():
        raise FileNotFoundError(
            f"Risk score file not found: "
            f"{risk_score_path}"
        )

    predictions = pd.read_csv(
        prediction_path
    )

    features = pd.read_csv(
        feature_path
    )

    risk_scores = pd.read_csv(
        risk_score_path
    )

    for dataframe, name in (
        (predictions, "GBM predictions"),
        (features, "features"),
        (risk_scores, "risk scores"),
    ):
        if TIMESTAMP_COLUMN not in dataframe.columns:
            raise ValueError(
                f"{name} dataset is missing required "
                f"column '{TIMESTAMP_COLUMN}'."
            )

        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            utc=True,
        )

    if TARGET_COLUMN not in predictions.columns:
        raise ValueError(
            f"GBM prediction dataset is missing required "
            f"column '{TARGET_COLUMN}'."
        )

    if TARGET_COLUMN not in features.columns:
        raise ValueError(
            f"Feature dataset is missing required "
            f"column '{TARGET_COLUMN}'."
        )

    if RISK_SCORE_COLUMN not in risk_scores.columns:
        raise ValueError(
            f"Risk score dataset is missing required "
            f"column '{RISK_SCORE_COLUMN}'."
        )

    prediction_columns = [
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        "predicted_direction",
        "probability_down",
        "probability_up",
    ]

    missing_prediction_columns = [
        column
        for column in prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            "GBM prediction dataset is missing required "
            f"columns: {missing_prediction_columns}"
        )

    evaluation = predictions[
        prediction_columns
    ].copy()

    risk_score_frame = risk_scores[
        [
            TIMESTAMP_COLUMN,
            RISK_SCORE_COLUMN,
        ]
    ].copy()

    evaluation = evaluation.merge(
        risk_score_frame,
        on=TIMESTAMP_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    evaluation = (
        evaluation
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    if evaluation.empty:
        raise ValueError(
            f"Evaluation dataset for {ticker} is empty "
            "after joining GBM predictions with risk scores."
        )

    required_columns = [
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        "predicted_direction",
        "probability_down",
        "probability_up",
        RISK_SCORE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in evaluation.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Evaluation dataset for {ticker} is missing "
            f"required columns: {missing_columns}"
        )

    return evaluation


def save_evaluation_report(
    report: EvaluationReport,
    ticker: str,
) -> None:
    """Save a single ticker evaluation report."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        OUTPUT_DIR
        / f"{ticker}_evaluation.csv"
    )

    report_data = report_to_dict(report)

    report_dataframe = pd.DataFrame(
        [report_data]
    )

    report_dataframe.to_csv(
        report_path,
        index=False,
    )


def save_evaluation_summary(
    reports: list[EvaluationReport],
) -> Path:
    """Save the combined evaluation summary."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "evaluation_summary.csv"
    )

    summary_dataframe = pd.DataFrame(
        [
            report_to_dict(report)
            for report in reports
        ]
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    return summary_path


def main() -> None:
    """Run Phase 1 evaluation for every available ticker."""

    GBM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_files = sorted(
        GBM_DIR.glob(
            "*_test_predictions.csv"
        )
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"No GBM prediction files found in {GBM_DIR}"
        )

    evaluator = RiskModelEvaluator()

    reports: list[EvaluationReport] = []

    for prediction_file in prediction_files:
        ticker = prediction_file.name.replace(
            "_test_predictions.csv",
            "",
        )

        print(
            f"Evaluating {ticker}..."
        )

        evaluation_dataset = load_evaluation_dataset(
            ticker
        )

        report = evaluator.evaluate(
            ticker=ticker,
            dataframe=evaluation_dataset,
        )

        save_evaluation_report(
            report=report,
            ticker=ticker,
        )

        reports.append(report)

        print(
            f"  rows={report.rows}"
            f" accuracy={report.gbm.accuracy:.4f}"
            f" roc_auc={report.gbm.roc_auc:.4f}"
            f" log_loss={report.gbm.log_loss:.4f}"
            f" brier={report.gbm.brier_score:.4f}"
            f" baseline_accuracy="
            f"{report.baseline.accuracy:.4f}"
            f" accuracy_lift="
            f"{report.accuracy_lift:.4f}"
        )

    summary_path = save_evaluation_summary(
        reports
    )

    print()
    print(
        f"Evaluation summary saved to: "
        f"{summary_path}"
    )

    print(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()