from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupClassificationMetrics:
  

    group: str
    sample_count: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class GroupMetricDifference:
   
    metric: str
    group_a: str
    group_b: str
    value_a: float
    value_b: float
    absolute_difference: float
    signed_difference: float


class SubgroupBiasAuditor:
   

    def __init__(
        self,
        positive_label: int = 1,
    ) -> None:
        self._positive_label = positive_label

    def evaluate(
        self,
        y_true: Sequence[int] | np.ndarray,
        y_pred: Sequence[int] | np.ndarray,
        groups: Sequence[str] | np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculate classification metrics independently for each subgroup.

        Parameters
        ----------
        y_true:
            Ground-truth binary labels.

        y_pred:
            Binary model predictions.

        groups:
            Explicit subgroup labels aligned with ``y_true`` and ``y_pred``.

        Returns
        -------
        pandas.DataFrame
            One row per subgroup.
        """
        true = self._validate_labels(
            y_true,
            "y_true",
        )

        predicted = self._validate_labels(
            y_pred,
            "y_pred",
        )

        subgroup = self._validate_groups(
            groups,
        )

        if not (
            len(true)
            == len(predicted)
            == len(subgroup)
        ):
            raise ValueError(
                "y_true, y_pred, and groups must have identical lengths."
            )

        records: list[dict[str, float | int | str]] = []

        for group in pd.unique(subgroup):
            mask = subgroup == group

            metrics = self._calculate_group_metrics(
                true[mask],
                predicted[mask],
                str(group),
            )

            records.append(
                {
                    "group": metrics.group,
                    "sample_count": metrics.sample_count,
                    "positive_rate": metrics.positive_rate,
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "false_positive_rate": (
                        metrics.false_positive_rate
                    ),
                    "false_negative_rate": (
                        metrics.false_negative_rate
                    ),
                }
            )

        return pd.DataFrame(
            records,
            columns=[
                "group",
                "sample_count",
                "positive_rate",
                "accuracy",
                "precision",
                "recall",
                "false_positive_rate",
                "false_negative_rate",
            ],
        )

    def compare(
        self,
        metrics: pd.DataFrame,
        group_a: str,
        group_b: str,
    ) -> pd.DataFrame:
        """
        Compare two explicitly selected subgroups.

        Returns descriptive metric differences. No ranking or overall
        fairness score is produced.
        """
        required_columns = {
            "group",
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        }

        missing = required_columns.difference(
            metrics.columns,
        )

        if missing:
            raise ValueError(
                "metrics is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if group_a == group_b:
            raise ValueError(
                "group_a and group_b must be different."
            )

        rows_a = metrics.loc[
            metrics["group"].astype(str) == group_a
        ]

        rows_b = metrics.loc[
            metrics["group"].astype(str) == group_b
        ]

        if rows_a.empty:
            raise ValueError(
                f"Group '{group_a}' was not found."
            )

        if rows_b.empty:
            raise ValueError(
                f"Group '{group_b}' was not found."
            )

        row_a = rows_a.iloc[0]
        row_b = rows_b.iloc[0]

        metric_names = [
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ]

        rows: list[dict[str, float | str]] = []

        for metric_name in metric_names:
            value_a = float(row_a[metric_name])
            value_b = float(row_b[metric_name])
            difference = value_a - value_b

            rows.append(
                {
                    "metric": metric_name,
                    "group_a": group_a,
                    "group_b": group_b,
                    "value_a": value_a,
                    "value_b": value_b,
                    "absolute_difference": abs(
                        difference
                    ),
                    "signed_difference": difference,
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "metric",
                "group_a",
                "group_b",
                "value_a",
                "value_b",
                "absolute_difference",
                "signed_difference",
            ],
        )

    @staticmethod
    def _calculate_group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: str,
    ) -> GroupClassificationMetrics:
        """Calculate binary classification metrics for one subgroup."""
        sample_count = int(len(y_true))

        if sample_count == 0:
            raise ValueError(
                f"Group '{group}' contains no observations."
            )

        true_positive = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 1)
            )
        )

        true_negative = int(
            np.sum(
                (y_true == 0)
                & (y_pred == 0)
            )
        )

        false_positive = int(
            np.sum(
                (y_true == 0)
                & (y_pred == 1)
            )
        )

        false_negative = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 0)
            )
        )

        positive_rate = float(
            np.mean(y_pred == 1)
        )

        accuracy = float(
            (true_positive + true_negative)
            / sample_count
        )

        precision_denominator = (
            true_positive + false_positive
        )

        precision = (
            float(true_positive / precision_denominator)
            if precision_denominator > 0
            else 0.0
        )

        recall_denominator = (
            true_positive + false_negative
        )

        recall = (
            float(true_positive / recall_denominator)
            if recall_denominator > 0
            else 0.0
        )

        false_positive_denominator = (
            false_positive + true_negative
        )

        false_positive_rate = (
            float(
                false_positive
                / false_positive_denominator
            )
            if false_positive_denominator > 0
            else 0.0
        )

        false_negative_denominator = (
            false_negative + true_positive
        )

        false_negative_rate = (
            float(
                false_negative
                / false_negative_denominator
            )
            if false_negative_denominator > 0
            else 0.0
        )

        return GroupClassificationMetrics(
            group=group,
            sample_count=sample_count,
            positive_rate=positive_rate,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
        )

    @staticmethod
    def _validate_labels(
        labels: Sequence[int] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate binary classification labels."""
        values = np.asarray(
            labels,
        )

        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.isin(
            values,
            [0, 1],
        ).all():
            raise ValueError(
                f"{name} must contain only binary labels 0 and 1."
            )

        return values.astype(int)

    @staticmethod
    def _validate_groups(
        groups: Sequence[str] | np.ndarray,
    ) -> np.ndarray:
        """Validate explicit subgroup labels."""
        values = np.asarray(
            groups,
            dtype=object,
        )

        if values.ndim != 1:
            raise ValueError(
                "groups must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "groups cannot be empty."
            )

        if pd.isna(values).any():
            raise ValueError(
                "groups cannot contain missing values."
            )

        if any(
            not str(value).strip()
            for value in values
        ):
            raise ValueError(
                "groups cannot contain empty labels."
            )

        return values