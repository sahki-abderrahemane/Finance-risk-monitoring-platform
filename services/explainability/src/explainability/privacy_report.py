from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .differential_privacy import PrivacyRelease


@dataclass(frozen=True)
class PrivacyUtilityMetrics:
  
    mean_absolute_error: float
    root_mean_squared_error: float
    maximum_absolute_error: float
    mean_noise_magnitude: float


class PrivacyUtilityEvaluator:
  

    @staticmethod
    def evaluate(
        release: PrivacyRelease,
    ) -> PrivacyUtilityMetrics:
        """
        Calculate distortion metrics for one privacy release.
        """
        clipped = np.asarray(
            release.clipped,
            dtype=float,
        )

        noisy = np.asarray(
            release.noisy,
            dtype=float,
        )

        if clipped.shape != noisy.shape:
            raise ValueError(
                "clipped and noisy arrays must have identical shapes."
            )

        error = noisy - clipped
        absolute_error = np.abs(error)

        return PrivacyUtilityMetrics(
            mean_absolute_error=float(
                np.mean(absolute_error)
            ),
            root_mean_squared_error=float(
                np.sqrt(np.mean(error**2))
            ),
            maximum_absolute_error=float(
                np.max(absolute_error)
            ),
            mean_noise_magnitude=float(
                np.mean(np.abs(error))
            ),
        )

    @staticmethod
    def compare_releases(
        releases: Sequence[PrivacyRelease],
    ) -> pd.DataFrame:
        """
        Compare utility distortion across multiple privacy budgets.

        The resulting table is useful for offline experiments such as:

            epsilon = 0.1
            epsilon = 0.5
            epsilon = 1.0
            epsilon = 5.0

        Smaller epsilon generally means a tighter privacy budget, but the
        resulting utility depends on the mechanism and sensitivity.
        """
        if not releases:
            raise ValueError(
                "releases cannot be empty."
            )

        rows: list[dict[str, float]] = []

        for release in releases:
            metrics = PrivacyUtilityEvaluator.evaluate(
                release,
            )

            rows.append(
                {
                    "epsilon": release.epsilon,
                    "delta": release.delta,
                    "sensitivity": release.sensitivity,
                    "noise_scale": release.noise_scale,
                    "mean_absolute_error": (
                        metrics.mean_absolute_error
                    ),
                    "root_mean_squared_error": (
                        metrics.root_mean_squared_error
                    ),
                    "maximum_absolute_error": (
                        metrics.maximum_absolute_error
                    ),
                    "mean_noise_magnitude": (
                        metrics.mean_noise_magnitude
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            "epsilon",
            ascending=True,
            ignore_index=True,
        )