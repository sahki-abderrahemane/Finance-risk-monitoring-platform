from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[4]

MLFLOW_DB = PROJECT_ROOT / "ml" / "experiments" / "mlflow.db"
EXPERIMENT_NAME = "sentinel-ai/phase2-multimodal-engine"

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
DATASETS_DIR = PROJECT_ROOT / "ml" / "datasets"

FINBERT_MODEL_DIR = MODELS_DIR / "finbert"
CNN_CHECKPOINT = MODELS_DIR / "cnn" / "chart_cnn_weighted_best.pt"
UNIFIED_RISK_CHECKPOINT = (
    MODELS_DIR / "multimodal" / "unified_risk_model_best.pt"
)

FUSION_DIR = DATASETS_DIR / "multimodal" / "features"
EVALUATION_DIR = DATASETS_DIR / "multimodal" / "evaluation"


def configure_mlflow() -> None:
    """Configure the local SQLite MLflow tracking backend."""
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{MLFLOW_DB}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)


def _log_file_if_exists(path: Path, artifact_path: str | None = None) -> None:
    """Log a single artifact if it exists."""
    if not path.exists():
        print(f"WARNING: artifact does not exist: {path}")
        return

    mlflow.log_artifact(
        local_path=str(path),
        artifact_path=artifact_path,
    )


def _log_directory_if_exists(
    directory: Path,
    artifact_path: str,
) -> None:
    """Log a directory recursively if it exists."""
    if not directory.exists():
        print(f"WARNING: artifact directory does not exist: {directory}")
        return

    mlflow.log_artifacts(
        local_dir=str(directory),
        artifact_path=artifact_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return an object."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return value


def _log_numeric_metrics(
    metrics: dict[str, Any],
    prefix: str = "",
) -> None:
    """Log finite scalar numeric values as MLflow metrics."""
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, bool):
            mlflow.log_metric(metric_name, float(value))
            continue

        if isinstance(value, int | float):
            mlflow.log_metric(metric_name, float(value))


def track_fusion_run() -> str:
    """Track the feature-fusion artifacts."""
    configure_mlflow()

    manifest_path = FUSION_DIR / "fusion_manifest.json"

    with mlflow.start_run(run_name="feature-fusion") as run:
        mlflow.set_tags(
            {
                "phase": "phase2",
                "component": "feature-fusion",
                "task": "multimodal-representation-learning",
                "leakage_policy": "train_only_fitted_transforms",
            }
        )

        mlflow.log_params(
            {
                "market_representation": "StandardScaler + PCA",
                "text_representation": "mean-pooled FinBERT encoder",
                "vision_representation": "CNN penultimate representation",
                "fusion_strategy": "concatenation",
            }
        )

        _log_file_if_exists(
            manifest_path,
            artifact_path="fusion",
        )

        _log_file_if_exists(
            FUSION_DIR / "market_pca_pipeline.joblib",
            artifact_path="fusion",
        )

        for filename in (
            "train_fused_metadata.csv",
            "validation_fused_metadata.csv",
            "test_fused_metadata.csv",
        ):
            _log_file_if_exists(
                FUSION_DIR / filename,
                artifact_path="fusion/metadata",
            )

        return run.info.run_id


def track_nlp_run() -> str:
    """Track the FinBERT NLP model and sentiment artifacts."""
    configure_mlflow()

    with mlflow.start_run(run_name="finbert-sentiment") as run:
        mlflow.set_tags(
            {
                "phase": "phase2",
                "component": "nlp-engine",
                "model_family": "FinBERT",
                "model_name": "ProsusAI/finbert",
                "task": "financial-sentiment-classification",
            }
        )

        mlflow.log_params(
            {
                "base_model": "ProsusAI/finbert",
                "max_length": 256,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "epochs": 3,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "seed": 42,
                "label_negative": 0,
                "label_neutral": 1,
                "label_positive": 2,
            }
        )

        _log_directory_if_exists(
            FINBERT_MODEL_DIR,
            artifact_path="models/finbert",
        )

        _log_file_if_exists(
            DATASETS_DIR / "news" / "raw" / "financial_news.csv",
            artifact_path="datasets/news",
        )

        _log_file_if_exists(
            DATASETS_DIR / "news" / "sentiment_daily.csv",
            artifact_path="datasets/news",
        )

        _log_file_if_exists(
            DATASETS_DIR / "news" / "finbert_test_predictions.csv",
            artifact_path="evaluation",
        )

        return run.info.run_id


def track_vision_run() -> str:
    """Track the trained CNN vision model and evaluation artifacts."""
    configure_mlflow()

    with mlflow.start_run(run_name="chart-cnn") as run:
        mlflow.set_tags(
            {
                "phase": "phase2",
                "component": "vision-engine",
                "task": "chart-pattern-classification",
                "model_family": "CNN",
                "checkpoint_type": "weighted",
            }
        )

        mlflow.log_params(
            {
                "image_size": 224,
                "input_channels": 3,
                "num_classes": 3,
                "base_channels": 32,
                "dropout": 0.30,
                "window_size": 30,
                "forecast_horizon": 5,
                "seed": 42,
                "weighted_loss": True,
            }
        )

        _log_file_if_exists(
            CNN_CHECKPOINT,
            artifact_path="models/cnn",
        )

        _log_file_if_exists(
            DATASETS_DIR / "charts" / "chart_metadata.csv",
            artifact_path="datasets/charts",
        )

        _log_file_if_exists(
            DATASETS_DIR / "charts" / "dataset_summary.json",
            artifact_path="datasets/charts",
        )

        evaluation_files = (
            DATASETS_DIR / "vision" / "evaluation",
        )

        for directory in evaluation_files:
            _log_directory_if_exists(
                directory,
                artifact_path="evaluation/vision",
            )

        return run.info.run_id


def track_unified_risk_run() -> str:
    """Track the unified multimodal risk model."""
    configure_mlflow()

    summary_path = (
        MODELS_DIR
        / "multimodal"
        / "unified_risk_training_summary.json"
    )

    with mlflow.start_run(run_name="unified-risk-model") as run:
        mlflow.set_tags(
            {
                "phase": "phase2",
                "component": "multimodal-engine",
                "task": "unified-risk-classification",
                "architecture": "MLP",
                "target": "target_direction_1d",
                "risk_definition": "model_uncertainty",
            }
        )

        mlflow.log_params(
            {
                "input_dimension": 901,
                "hidden_dimension_1": 256,
                "hidden_dimension_2": 64,
                "dropout": 0.25,
                "loss": "BCEWithLogitsLoss",
                "early_stopping": True,
                "split_strategy": "chronological",
                "seed": 42,
            }
        )

        _log_file_if_exists(
            UNIFIED_RISK_CHECKPOINT,
            artifact_path="models/unified-risk",
        )

        _log_file_if_exists(
            summary_path,
            artifact_path="training",
        )

        if summary_path.exists():
            summary = _read_json(summary_path)

            _log_numeric_metrics(
                summary.get("metrics", {}),
                prefix="",
            )

            _log_numeric_metrics(
                summary.get("training", {}),
                prefix="training_",
            )

        return run.info.run_id


def track_multimodal_evaluation_run() -> str:
    """Track final multimodal evaluation and ablation results."""
    configure_mlflow()

    results_path = (
        EVALUATION_DIR / "multimodal_evaluation_results.csv"
    )
    predictions_path = (
        EVALUATION_DIR / "multimodal_test_predictions.csv"
    )
    summary_path = (
        EVALUATION_DIR / "multimodal_evaluation_summary.json"
    )

    with mlflow.start_run(run_name="multimodal-evaluation") as run:
        mlflow.set_tags(
            {
                "phase": "phase2",
                "component": "multimodal-engine",
                "task": "multimodal-evaluation",
                "evaluation_strategy": "chronological-test-evaluation",
                "ablation_study": True,
                "test_set_used_for_model_selection": False,
            }
        )

        mlflow.log_params(
            {
                "market_features": 5,
                "text_features": 768,
                "vision_features": 128,
                "full_feature_dimension": 901,
                "test_rows": 1862,
                "seed": 42,
            }
        )

        _log_file_if_exists(
            results_path,
            artifact_path="evaluation",
        )

        _log_file_if_exists(
            predictions_path,
            artifact_path="evaluation",
        )

        _log_file_if_exists(
            summary_path,
            artifact_path="evaluation",
        )

        if summary_path.exists():
            summary = _read_json(summary_path)

            for key in (
                "full_multimodal",
                "market_only",
                "text_only",
                "vision_only",
                "market_text",
                "market_vision",
                "text_vision",
            ):
                result = summary.get(key)

                if isinstance(result, dict):
                    _log_numeric_metrics(
                        result,
                        prefix=f"{key}_",
                    )

        return run.info.run_id


def track_all_phase2() -> dict[str, str]:
    """Track all completed Phase 2 components."""
    return {
        "nlp": track_nlp_run(),
        "vision": track_vision_run(),
        "fusion": track_fusion_run(),
        "unified_risk": track_unified_risk_run(),
        "evaluation": track_multimodal_evaluation_run(),
    }


def main() -> None:
    """CLI entry point."""
    print("SENTINEL-AI PHASE 2 MLflow TRACKING")
    print("=" * 48)

    run_ids = track_all_phase2()

    print()
    print("MLflow tracking complete.")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Tracking DB: {MLFLOW_DB}")
    print()

    for component, run_id in run_ids.items():
        print(f"{component:15s} -> {run_id}")


if __name__ == "__main__":
    main()