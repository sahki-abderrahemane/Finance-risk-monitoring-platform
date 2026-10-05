from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    random_state: int = 42
    clip_min: float = -1.0
    clip_max: float = 1.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        if not 0.0 < self.delta < 1.0:
            raise ValueError(
                "delta must be strictly between zero and one."
            )

        if self.sensitivity <= 0:
            raise ValueError(
                "sensitivity must be greater than zero."
            )

        if self.clip_min >= self.clip_max:
            raise ValueError(
                "clip_min must be smaller than clip_max."
            )


@dataclass(frozen=True)
class PrivacyRelease:
   

    original: np.ndarray
    clipped: np.ndarray
    noisy: np.ndarray
    epsilon: float
    delta: float
    sensitivity: float
    noise_scale: float


class GaussianPrivacyMechanism:
   

    def __init__(
        self,
        config: DifferentialPrivacyConfig | None = None,
    ) -> None:
        self._config = (
            config or DifferentialPrivacyConfig()
        )

    @property
    def config(self) -> DifferentialPrivacyConfig:
        """Return the configured privacy parameters."""
        return self._config

    def release(
        self,
        values: np.ndarray,
    ) -> PrivacyRelease:
        """
        Produce a noisy experimental release.

        Parameters
        ----------
        values:
            Numerical array to be clipped and perturbed.

        Returns
        -------
        PrivacyRelease
            Original, clipped, noisy values and privacy metadata.
        """
        original = self._validate_values(values)

        clipped = np.clip(
            original,
            self._config.clip_min,
            self._config.clip_max,
        )

        noise_scale = self._calculate_noise_scale()

        rng = np.random.default_rng(
            self._config.random_state,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=clipped.shape,
        )

        noisy = clipped + noise

        return PrivacyRelease(
            original=original.copy(),
            clipped=clipped,
            noisy=noisy,
            epsilon=self._config.epsilon,
            delta=self._config.delta,
            sensitivity=self._config.sensitivity,
            noise_scale=noise_scale,
        )

    def _calculate_noise_scale(self) -> float:
        """
        Calculate the Gaussian noise standard deviation.

        This implementation uses the common analytical Gaussian
        mechanism form:

            sigma =
                sensitivity *
                sqrt(2 * ln(1.25 / delta)) /
                epsilon

        The result should be interpreted together with the assumptions
        under which that mechanism is being used.
        """
        return float(
            self._config.sensitivity
            * np.sqrt(
                2.0
                * np.log(
                    1.25 / self._config.delta,
                )
            )
            / self._config.epsilon
        )

    @staticmethod
    def _validate_values(
        values: np.ndarray,
    ) -> np.ndarray:
        """Validate a finite numerical array."""
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.size == 0:
            raise ValueError(
                "values cannot be empty."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "values contain NaN or infinite values."
            )

        return array