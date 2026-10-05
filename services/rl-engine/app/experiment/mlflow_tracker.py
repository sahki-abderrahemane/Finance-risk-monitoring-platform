from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mlflow

from app.experiment.config import ExperimentConfig
from app.experiment.runner import ExperimentResult


class MLflowExperimentTracker:
    

    def __init__(
        self,
        experiment_name: str = "sentinel-ai-rl",
    ) -> None:
        if not experiment_name.strip():
            raise ValueError(
                "experiment_name cannot be empty."
            )

        self._experiment_name = experiment_name

        mlflow.set_experiment(
            experiment_name
        )

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def start_run(
        self,
        config: ExperimentConfig,
    ):
        """
        Start an MLflow run and record the complete experiment
        configuration as parameters.
        """

        run = mlflow.start_run()

        self._log_config(config)

        return run

    def log_result(
        self,
        result: ExperimentResult,
    ) -> None:
        """
        Record the primary experiment metrics.
        """

        training_rewards = (
            result.training.total_rewards
        )

        evaluation_rewards = (
            result.evaluation.total_rewards
        )

        if training_rewards:
            mlflow.log_metric(
                "training_average_reward",
                result.training.average_reward,
            )

            mlflow.log_metric(
                "training_final_reward",
                training_rewards[-1],
            )

        if evaluation_rewards:
            mlflow.log_metric(
                "evaluation_average_reward",
                result.evaluation.average_reward,
            )

            mlflow.log_metric(
                "evaluation_final_reward",
                evaluation_rewards[-1],
            )

        mlflow.log_metric(
            "hold_baseline_total_reward",
            result.hold_baseline_total_reward,
        )

        mlflow.log_metric(
            "q_table_size",
            float(result.q_table_size),
        )

        mlflow.set_tag(
            "seed",
            str(result.seed),
        )

    def end_run(self) -> None:
        """
        End the active MLflow run.
        """

        active_run = mlflow.active_run()

        if active_run is not None:
            mlflow.end_run()

    def _log_config(
        self,
        config: ExperimentConfig,
    ) -> None:
        parameters = self._flatten_config(
            asdict(config)
        )

        mlflow.log_params(parameters)

    @staticmethod
    def _flatten_config(
        values: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, str]:
        """
        Convert nested dataclass configuration into MLflow's
        flat parameter representation.

        Example:

            {
                "q_learning": {
                    "learning_rate": 0.1
                }
            }

        becomes:

            {
                "q_learning.learning_rate": "0.1"
            }
        """

        flattened: dict[str, str] = {}

        for key, value in values.items():
            full_key = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            if isinstance(value, dict):
                flattened.update(
                    MLflowExperimentTracker
                    ._flatten_config(
                        value,
                        full_key,
                    )
                )
            else:
                flattened[full_key] = str(value)

        return flattened