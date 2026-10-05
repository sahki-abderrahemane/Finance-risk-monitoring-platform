from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BiasAuditSummary:
  

    group_count: int
    minimum_group_size: int
    maximum_group_size: int
    metrics: tuple[str, ...]


class BiasAuditReport:
   

    REQUIRED_COLUMNS = (
        "group",
        "sample_count",
        "positive_rate",
        "accuracy",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    )

    @classmethod
    def validate(
        cls,
        metrics: pd.DataFrame,
    ) -> None:
        """
        Validate the schema of a subgroup metric table.
        """
        if metrics.empty:
            raise ValueError(
                "metrics cannot be empty."
            )

        missing = set(
            cls.REQUIRED_COLUMNS
        ).difference(
            metrics.columns
        )

        if missing:
            raise ValueError(
                "metrics is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if (
            metrics["sample_count"]
            .astype(int)
            .le(0)
            .any()
        ):
            raise ValueError(
                "sample_count must be greater than zero."
            )

        metric_columns = [
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ]

        for column in metric_columns:
            values = metrics[column].astype(float)

            if not values.between(
                0.0,
                1.0,
            ).all():
                raise ValueError(
                    f"{column} must contain values between 0 and 1."
                )

    @classmethod
    def summarize(
        cls,
        metrics: pd.DataFrame,
    ) -> BiasAuditSummary:
        """
        Produce a compact structural summary of the audit.
        """
        cls.validate(metrics)

        metric_columns = (
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        )

        return BiasAuditSummary(
            group_count=int(
                metrics["group"].nunique()
            ),
            minimum_group_size=int(
                metrics["sample_count"].min()
            ),
            maximum_group_size=int(
                metrics["sample_count"].max()
            ),
            metrics=metric_columns,
        )

    @classmethod
    def to_long_format(
        cls,
        metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert subgroup metrics into dashboard/monitoring-friendly form.

        The output contains one row per:

            subgroup × metric

        This format is convenient for MLflow logging and visualization.
        """
        cls.validate(metrics)

        metric_columns = [
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ]

        return metrics.melt(
            id_vars=[
                "group",
                "sample_count",
            ],
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        ).sort_values(
            [
                "metric",
                "group",
            ],
            ignore_index=True,
        )