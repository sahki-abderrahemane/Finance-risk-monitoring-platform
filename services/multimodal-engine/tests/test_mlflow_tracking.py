from __future__ import annotations

from pathlib import Path

import mlflow

from multimodal_engine import mlflow_tracking


def test_experiment_configuration(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "mlflow.db"

    monkeypatch.setattr(
        mlflow_tracking,
        "MLFLOW_DB",
        database,
    )

    monkeypatch.setattr(
        mlflow_tracking,
        "EXPERIMENT_NAME",
        "sentinel-ai/test-phase2",
    )

    mlflow_tracking.configure_mlflow()

    experiment = mlflow.get_experiment_by_name(
        "sentinel-ai/test-phase2"
    )

    assert experiment is not None
    assert experiment.name == "sentinel-ai/test-phase2"


def test_numeric_metrics_only() -> None:
    mlflow_tracking.configure_mlflow()

    metrics = {
        "accuracy": 0.52,
        "loss": 0.69,
        "epoch": 10,
        "enabled": True,
        "ignored": "text",
        "ignored_none": None,
    }

    mlflow_tracking._log_numeric_metrics(metrics)


def test_missing_artifact_is_safe(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"

    mlflow_tracking.configure_mlflow()

    mlflow_tracking._log_file_if_exists(missing)


def test_missing_directory_is_safe(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    mlflow_tracking.configure_mlflow()

    mlflow_tracking._log_directory_if_exists(
        missing,
        artifact_path="test",
    )