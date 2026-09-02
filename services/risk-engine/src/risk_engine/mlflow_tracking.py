from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for Sentinel-AI MLflow tracking."""

    experiment_name: str = "sentinel-ai/phase1-core-engine"

    # SQLite is used for local Phase 1 development.
    #
    # Later, this can be replaced with PostgreSQL:
    #
    # postgresql://user:password@postgres:5432/mlflow
    #
    # without changing the tracking API.
    tracking_uri: str | None = None

    registry_name: str = (
        "sentinel-ai-gradient-boosting"
    )

    artifact_subdirectory: str = "phase1"


class MLflowExperimentTracker:
    """
    MLflow tracking adapter for Sentinel-AI Phase 1.

    Responsibilities:
        - configure MLflow
        - create/select the Phase 1 experiment
        - start runs
        - log parameters
        - log metrics
        - log tags
        - log artifacts
        - log sklearn models
    """

    def __init__(
        self,
        config: MLflowConfig | None = None,
    ) -> None:
        self.config = (
            config
            or MLflowConfig()
        )

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(
                self.config.tracking_uri
            )

        experiment = mlflow.set_experiment(
            self.config.experiment_name
        )

        self.experiment_id = (
            experiment.experiment_id
        )

    @property
    def tracking_uri(self) -> str:
        """Return the active MLflow tracking URI."""

        return mlflow.get_tracking_uri()

    @property
    def experiment_name(self) -> str:
        """Return the configured experiment name."""

        return self.config.experiment_name

    def start_run(
        self,
        run_name: str,
    ):
        """
        Start an MLflow run.

        Usage:

            with tracker.start_run("AAPL"):
                ...
        """

        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )

    @staticmethod
    def log_params(
        params: dict[str, Any],
    ) -> None:
        """Log scalar experiment parameters."""

        normalized: dict[
            str,
            str | int | float | bool,
        ] = {}

        for key, value in params.items():
            if value is None:
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                normalized[key] = value
            else:
                normalized[key] = str(value)

        if normalized:
            mlflow.log_params(
                normalized
            )

    @staticmethod
    def log_metrics(
        metrics: dict[str, float],
    ) -> None:
        """Log numeric experiment metrics."""

        normalized: dict[str, float] = {}

        for key, value in metrics.items():
            if value is None:
                continue

            normalized[key] = float(value)

        if normalized:
            mlflow.log_metrics(
                normalized
            )

    @staticmethod
    def log_tags(
        tags: dict[str, str],
    ) -> None:
        """Log descriptive run tags."""

        if tags:
            mlflow.set_tags(
                tags
            )

    @staticmethod
    def log_artifact(
        path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log a single file as an MLflow artifact."""

        artifact = Path(path)

        if not artifact.exists():
            raise FileNotFoundError(
                "MLflow artifact does not exist: "
                f"{artifact}"
            )

        if not artifact.is_file():
            raise ValueError(
                "MLflow artifact must be a file: "
                f"{artifact}"
            )

        mlflow.log_artifact(
            str(artifact),
            artifact_path=artifact_path,
        )

    @staticmethod
    def log_artifacts(
        directory: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log all files in a directory as MLflow artifacts."""

        artifact_directory = Path(
            directory
        )

        if not artifact_directory.exists():
            raise FileNotFoundError(
                "MLflow artifact directory does not exist: "
                f"{artifact_directory}"
            )

        if not artifact_directory.is_dir():
            raise ValueError(
                "MLflow artifact path must be a directory: "
                f"{artifact_directory}"
            )

        mlflow.log_artifacts(
            str(artifact_directory),
            artifact_path=artifact_path,
        )

    @staticmethod
    def log_text(
        text: str,
        artifact_file: str,
    ) -> None:
        """Log text content as an MLflow artifact."""

        mlflow.log_text(
            text,
            artifact_file,
        )

    @staticmethod
    def log_sklearn_model(
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log a scikit-learn model to MLflow."""

        import mlflow.sklearn

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=(
                registered_model_name
            ),
        )

    @staticmethod
    def set_model_metadata(
        *,
        model_type: str,
        framework: str,
        framework_version: str,
    ) -> None:
        """Attach standardized model metadata."""

        mlflow.set_tags(
            {
                "model_type": model_type,
                "framework": framework,
                "framework_version": (
                    framework_version
                ),
            }
        )

    @staticmethod
    def end_run() -> None:
        """End the active MLflow run if one exists."""

        if mlflow.active_run() is not None:
            mlflow.end_run()