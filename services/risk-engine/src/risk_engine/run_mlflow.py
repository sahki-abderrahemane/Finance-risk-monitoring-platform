from __future__ import annotations

from pathlib import Path

import pandas as pd
import sklearn

from risk_engine.mlflow_tracking import (
    MLflowConfig,
    MLflowExperimentTracker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

GBM_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "gbm"
)

RISK_SCORES_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "risk_scores"
)

GBM_MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "gradient_boosting"
)

PCA_MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pca"
)

EVALUATION_FILE = (
    GBM_DIR
    / "evaluation_summary.csv"
)

TRACKING_DIR = (
    PROJECT_ROOT
    / "ml"
    / "experiments"
)

MLFLOW_DB = (
    TRACKING_DIR
    / "mlflow.db"
)

EXPERIMENT_NAME = (
    "sentinel-ai/phase1-core-engine"
)


def load_evaluation_summary() -> pd.DataFrame:
    """Load the completed Phase 1 evaluation summary."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            "Evaluation summary not found: "
            f"{EVALUATION_FILE}\n"
            "Run risk_engine.run_evaluation first."
        )

    dataframe = pd.read_csv(
        EVALUATION_FILE
    )

    if dataframe.empty:
        raise ValueError(
            "Evaluation summary is empty."
        )

    if "ticker" not in dataframe.columns:
        raise ValueError(
            "Evaluation summary is missing "
            "the required 'ticker' column."
        )

    return dataframe


def get_float(
    row: pd.Series,
    column: str,
) -> float | None:
    """Extract a numeric value from an evaluation row."""

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return float(value)


def model_artifact_for_ticker(
    ticker: str,
) -> Path:
    """Return the GBM model artifact for a ticker."""

    path = (
        GBM_MODEL_DIR
        / f"{ticker}_gradient_boosting.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"GBM model artifact not found: {path}"
        )

    return path


def pca_artifacts_for_ticker(
    ticker: str,
) -> list[Path]:
    """Return available PCA artifacts for a ticker."""

    candidates = [
        PCA_MODEL_DIR
        / f"{ticker}_pca.joblib",
        GBM_DIR
        / f"{ticker}_train_pca_variance.csv",
    ]

    return [
        path
        for path in candidates
        if path.exists()
    ]


def prediction_artifact_for_ticker(
    ticker: str,
) -> Path:
    """Return the GBM test prediction artifact."""

    path = (
        GBM_DIR
        / f"{ticker}_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"GBM prediction artifact not found: {path}"
        )

    return path


def risk_score_artifact_for_ticker(
    ticker: str,
) -> Path:
    """Return the risk-score artifact."""

    path = (
        RISK_SCORES_DIR
        / f"{ticker}_risk_scores.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Risk-score artifact not found: {path}"
        )

    return path


def log_ticker_run(
    tracker: MLflowExperimentTracker,
    row: pd.Series,
) -> None:
    """Create and populate one MLflow run for one ticker."""

    ticker = str(row["ticker"])

    model_path = (
        model_artifact_for_ticker(
            ticker
        )
    )

    prediction_path = (
        prediction_artifact_for_ticker(
            ticker
        )
    )

    risk_score_path = (
        risk_score_artifact_for_ticker(
            ticker
        )
    )

    pca_paths = (
        pca_artifacts_for_ticker(
            ticker
        )
    )

    run_name = (
        f"phase1-gbm-{ticker}"
    )

    with tracker.start_run(
        run_name=run_name
    ):
        tracker.log_tags(
            {
                "phase": "phase1",
                "component": "risk-engine",
                "asset": ticker,
                "task": "direction-classification",
                "data_type": "historical",
                "execution_mode": (
                    "research-simulation-only"
                ),
            }
        )

        tracker.set_model_metadata(
            model_type=(
                "GradientBoostingClassifier"
            ),
            framework="scikit-learn",
            framework_version=(
                sklearn.__version__
            ),
        )

        tracker.log_params(
            {
                "ticker": ticker,
                "pca_variance_target": 0.95,
                "model_artifact": model_path.name,
                "prediction_artifact": (
                    prediction_path.name
                ),
                "rows_evaluated": int(
                    row["rows"]
                ),
            }
        )

        metrics: dict[str, float] = {}

        metric_columns = [
            "gbm_accuracy",
            "gbm_roc_auc",
            "gbm_log_loss",
            "gbm_brier_score",
            "baseline_accuracy",
            "baseline_positive_class_ratio",
            "accuracy_lift",
            "risk_score_mean",
            "risk_score_median",
            "high_risk_positive_ratio",
            "low_risk_positive_ratio",
        ]

        for column in metric_columns:
            value = get_float(
                row,
                column,
            )

            if value is not None:
                metrics[column] = value

        tracker.log_metrics(
            metrics
        )

        tracker.log_artifact(
            model_path,
            artifact_path="models",
        )

        tracker.log_artifact(
            prediction_path,
            artifact_path="predictions",
        )

        tracker.log_artifact(
            risk_score_path,
            artifact_path="risk_scores",
        )

        for pca_path in pca_paths:
            tracker.log_artifact(
                pca_path,
                artifact_path="pca",
            )

        evaluation_path = (
            GBM_DIR
            / f"{ticker}_evaluation.csv"
        )

        if evaluation_path.exists():
            tracker.log_artifact(
                evaluation_path,
                artifact_path="evaluation",
            )

        print(
            f"Tracked MLflow run for {ticker}"
        )


def main() -> None:
    """Track all completed Phase 1 experiments."""

    TRACKING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        f"sqlite:///{MLFLOW_DB}"
    )

    tracker = MLflowExperimentTracker(
        MLflowConfig(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=tracking_uri,
        )
    )

    print(
        "MLflow tracking URI:"
    )
    print(
        f"  {tracker.tracking_uri}"
    )

    print(
        "MLflow experiment:"
    )
    print(
        f"  {tracker.experiment_name}"
    )

    print()

    evaluation_summary = (
        load_evaluation_summary()
    )

    for _, row in evaluation_summary.iterrows():
        log_ticker_run(
            tracker=tracker,
            row=row,
        )

    print()
    print(
        "MLflow Phase 1 tracking completed."
    )

    print(
        "MLflow database:"
    )
    print(
        f"  {MLFLOW_DB}"
    )


if __name__ == "__main__":
    main()