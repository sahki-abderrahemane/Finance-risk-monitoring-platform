from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

PREDICTIONS_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
    / "finbert_test_predictions.csv"
)

OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
)

METRICS_PATH: Final[Path] = (
    OUTPUT_DIR / "finbert_evaluation.json"
)

CONFUSION_MATRIX_PATH: Final[Path] = (
    OUTPUT_DIR / "finbert_confusion_matrix.csv"
)


class SentimentEvaluationError(RuntimeError):
    """Raised when sentiment evaluation cannot be completed."""


def load_predictions(
    path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Load FinBERT test predictions from CSV."""
    if not path.exists():
        raise SentimentEvaluationError(
            f"Prediction file does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise SentimentEvaluationError(
            f"Prediction file is empty: {path}"
        )

    return dataframe


def _find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """Find the first matching column from a list of candidates."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise SentimentEvaluationError(
        f"Could not find {description} column. "
        f"Expected one of: {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def _normalize_label(value: object) -> str:
    """Normalize a sentiment label."""
    return str(value).strip().lower()


def evaluate_predictions(
    dataframe: pd.DataFrame,
) -> dict:
    """Calculate classification metrics from FinBERT predictions."""

    true_column = _find_column(
        dataframe,
        [
            "sentiment_label",
            "true_label",
            "actual_label",
            "label",
        ],
        "ground-truth sentiment",
    )

    predicted_column = _find_column(
        dataframe,
        [
            "predicted_sentiment",
            "predicted_label",
            "prediction",
            "pred_label",
        ],
        "predicted sentiment",
    )

    y_true = dataframe[true_column].map(_normalize_label)
    y_pred = dataframe[predicted_column].map(_normalize_label)

    if y_true.empty:
        raise SentimentEvaluationError(
            "Ground-truth labels are empty."
        )

    if y_pred.empty:
        raise SentimentEvaluationError(
            "Predicted labels are empty."
        )

    labels = sorted(
        set(y_true.tolist()) | set(y_pred.tolist())
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    classification_metrics = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in labels
        ],
        columns=[
            f"predicted_{label}"
            for label in labels
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_dataframe.to_csv(
        CONFUSION_MATRIX_PATH,
    )

    metrics = {
        "rows": int(len(dataframe)),
        "ground_truth_column": true_column,
        "prediction_column": predicted_column,
        "labels": labels,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_metrics,
        "confusion_matrix": matrix.tolist(),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    """Run FinBERT evaluation."""
    dataframe = load_predictions()

    metrics = evaluate_predictions(
        dataframe,
    )

    print("FinBERT evaluation completed.")
    print(
        f"Rows evaluated: {metrics['rows']}"
    )
    print(
        f"Ground truth:   {metrics['ground_truth_column']}"
    )
    print(
        f"Predictions:    {metrics['prediction_column']}"
    )
    print(
        f"Accuracy:       {metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1:       {metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1:    {metrics['weighted_f1']:.4f}"
    )
    print(
        f"Metrics:        {METRICS_PATH}"
    )
    print(
        f"Confusion:      {CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    main()