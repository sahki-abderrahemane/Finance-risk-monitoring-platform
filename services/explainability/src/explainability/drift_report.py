from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .drift_detection import DriftMetric, DriftReport


@dataclass(frozen=True)
class DriftSummary:
    """
    Structural summary of a drift-monitoring run.

    This summary intentionally reports measurements without converting
    them into a model-quality verdict.
    """

    feature_count: int
    metric_count: int
    psi_measurements: int
    ks_measurements: int
    maximum_psi: float
    maximum_ks: float


class DriftReportBuilder:
    """
    Transform drift measurements into monitoring-friendly structures.
    """

    @staticmethod
    def to_dataframe(
        report: DriftReport,
    ) -> pd.DataFrame:
        """
        Convert a DriftReport into a flat DataFrame.

        Output columns:

        - feature_name
        - metric_name
        - statistic
        - sample_reference
        - sample_current
        - p_value
        """
        if not report.metrics:
            raise ValueError(
                "report cannot contain zero metrics."
            )

        records = [
            {
                "feature_name": metric.feature_name,
                "metric_name": metric.metric_name,
                "statistic": metric.statistic,
                "sample_reference": metric.sample_reference,
                "sample_current": metric.sample_current,
                "p_value": metric.p_value,
            }
            for metric in report.metrics
        ]

        return pd.DataFrame(
            records,
            columns=[
                "feature_name",
                "metric_name",
                "statistic",
                "sample_reference",
                "sample_current",
                "p_value",
            ],
        ).sort_values(
            [
                "metric_name",
                "feature_name",
            ],
            ignore_index=True,
        )

    @staticmethod
    def summarize(
        report: DriftReport,
    ) -> DriftSummary:
        """
        Produce aggregate structural measurements from a drift report.
        """
        if not report.metrics:
            raise ValueError(
                "report cannot contain zero metrics."
            )

        metrics = report.metrics

        psi_values = [
            metric.statistic
            for metric in metrics
            if metric.metric_name == "psi"
        ]

        ks_values = [
            metric.statistic
            for metric in metrics
            if metric.metric_name == "ks"
        ]

        if not psi_values:
            raise ValueError(
                "report does not contain PSI measurements."
            )

        if not ks_values:
            raise ValueError(
                "report does not contain KS measurements."
            )

        return DriftSummary(
            feature_count=len(
                {
                    metric.feature_name
                    for metric in metrics
                }
            ),
            metric_count=len(metrics),
            psi_measurements=len(psi_values),
            ks_measurements=len(ks_values),
            maximum_psi=max(psi_values),
            maximum_ks=max(ks_values),
        )

    @staticmethod
    def feature_metrics(
        report: DriftReport,
        feature_name: str,
    ) -> tuple[DriftMetric, ...]:
        """
        Retrieve all drift measurements for one feature.
        """
        if not feature_name.strip():
            raise ValueError(
                "feature_name cannot be empty."
            )

        metrics = tuple(
            metric
            for metric in report.metrics
            if metric.feature_name == feature_name
        )

        if not metrics:
            raise ValueError(
                f"No drift measurements found for "
                f"feature '{feature_name}'."
            )

        return metrics