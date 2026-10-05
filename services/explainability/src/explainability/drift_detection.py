from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftMetric:
   

    feature_name: str
    metric_name: str
    statistic: float
    sample_reference: int
    sample_current: int
    p_value: float | None = None


@dataclass(frozen=True)
class DriftReport:
    """
    Collection of drift measurements for a monitoring window.
    """

    metrics: tuple[DriftMetric, ...]

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError(
                "DriftReport requires at least one metric."
            )


class DriftDetector:
   
    def __init__(
        self,
        number_of_bins: int = 10,
        epsilon: float = 1e-6,
    ) -> None:
        if number_of_bins < 2:
            raise ValueError(
                "number_of_bins must be at least 2."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        self._number_of_bins = number_of_bins
        self._epsilon = epsilon

    @property
    def number_of_bins(self) -> int:
        """Return the number of PSI bins."""
        return self._number_of_bins

    @property
    def epsilon(self) -> float:
        """Return the numerical stability constant."""
        return self._epsilon

    def population_stability_index(
        self,
        reference: Sequence[float] | np.ndarray,
        current: Sequence[float] | np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index.

        Bins are derived from reference quantiles and then applied to both
        reference and current distributions.

        A small epsilon prevents division or logarithm operations from
        becoming numerically undefined when a bin has zero probability.
        """
        reference_values = self._validate_vector(
            reference,
            "reference",
        )

        current_values = self._validate_vector(
            current,
            "current",
        )

        edges = self._build_reference_bins(
            reference_values,
        )

        reference_counts = self._histogram(
            reference_values,
            edges,
        )

        current_counts = self._histogram(
            current_values,
            edges,
        )

        reference_proportions = self._safe_proportions(
            reference_counts,
        )

        current_proportions = self._safe_proportions(
            current_counts,
        )

        return float(
            np.sum(
                (
                    current_proportions
                    - reference_proportions
                )
                * np.log(
                    current_proportions
                    / reference_proportions,
                )
            )
        )

    def kolmogorov_smirnov(
        self,
        reference: Sequence[float] | np.ndarray,
        current: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        """
        Calculate the two-sample Kolmogorov-Smirnov statistic.

        Returns
        -------
        tuple[float, float]
            KS statistic and two-sided p-value.
        """
        reference_values = self._validate_vector(
            reference,
            "reference",
        )

        current_values = self._validate_vector(
            current,
            "current",
        )

        result = ks_2samp(
            reference_values,
            current_values,
            alternative="two-sided",
            method="auto",
        )

        return float(result.statistic), float(result.pvalue)

    def compare_feature(
        self,
        feature_name: str,
        reference: Sequence[float] | np.ndarray,
        current: Sequence[float] | np.ndarray,
    ) -> tuple[DriftMetric, DriftMetric]:
        """
        Calculate PSI and KS measurements for one feature.
        """
        if not feature_name.strip():
            raise ValueError(
                "feature_name cannot be empty."
            )

        reference_values = self._validate_vector(
            reference,
            "reference",
        )

        current_values = self._validate_vector(
            current,
            "current",
        )

        psi = self.population_stability_index(
            reference_values,
            current_values,
        )

        ks_statistic, p_value = self.kolmogorov_smirnov(
            reference_values,
            current_values,
        )

        return (
            DriftMetric(
                feature_name=feature_name,
                metric_name="psi",
                statistic=psi,
                sample_reference=len(reference_values),
                sample_current=len(current_values),
            ),
            DriftMetric(
                feature_name=feature_name,
                metric_name="ks",
                statistic=ks_statistic,
                sample_reference=len(reference_values),
                sample_current=len(current_values),
                p_value=p_value,
            ),
        )

    def compare_matrix(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_names: Sequence[str],
    ) -> DriftReport:
        """
        Compare every feature in two aligned numerical matrices.
        """
        reference_values = self._validate_matrix(
            reference,
            "reference",
        )

        current_values = self._validate_matrix(
            current,
            "current",
        )

        if reference_values.shape[1] != current_values.shape[1]:
            raise ValueError(
                "reference and current must have the same "
                "number of features."
            )

        names = tuple(feature_names)

        if len(names) != reference_values.shape[1]:
            raise ValueError(
                "feature_names count must match the number of features."
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "feature_names must be unique."
            )

        metrics: list[DriftMetric] = []

        for index, feature_name in enumerate(names):
            feature_metrics = self.compare_feature(
                feature_name=feature_name,
                reference=reference_values[:, index],
                current=current_values[:, index],
            )

            metrics.extend(feature_metrics)

        return DriftReport(
            metrics=tuple(metrics),
        )

    def _build_reference_bins(
        self,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Construct approximately equal-frequency bins from reference data.
        """
        quantiles = np.linspace(
            0.0,
            1.0,
            self._number_of_bins + 1,
        )

        edges = np.quantile(
            reference,
            quantiles,
        )

        edges = np.unique(edges)

        if edges.size < 2:
            minimum = float(reference[0])
            maximum = minimum + self._epsilon

            edges = np.array(
                [minimum, maximum],
                dtype=float,
            )

        return edges

    @staticmethod
    def _histogram(
        values: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Return counts for a predefined set of bins."""
        counts, _ = np.histogram(
            values,
            bins=edges,
        )

        return counts.astype(float)

    def _safe_proportions(
        self,
        counts: np.ndarray,
    ) -> np.ndarray:
        """Convert counts to numerically stable proportions."""
        total = float(np.sum(counts))

        if total <= 0:
            raise ValueError(
                "Histogram contains no observations."
            )

        proportions = counts / total

        proportions = np.where(
            proportions <= 0.0,
            self._epsilon,
            proportions,
        )

        return proportions

    @staticmethod
    def _validate_vector(
        values: Sequence[float] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate one numerical observation vector."""
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if array.size == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return array

    @staticmethod
    def _validate_matrix(
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate a numerical feature matrix."""
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.ndim != 2:
            raise ValueError(
                f"{name} must be two-dimensional."
            )

        if array.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if array.shape[1] == 0:
            raise ValueError(
                f"{name} must contain at least one feature."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return array