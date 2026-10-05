from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.environment.state import MarketState


@dataclass(frozen=True)
class DiscreteState:
    

    return_bucket: int
    volatility_bucket: int
    position_bucket: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (
            self.return_bucket,
            self.volatility_bucket,
            self.position_bucket,
        )


class StateDiscretizer:
    """
    Converts continuous Sentinel market states into discrete
    states suitable for tabular Q-Learning.
    """

    def __init__(
        self,
        return_boundaries: Sequence[float] = (
            -0.02,
            0.0,
            0.02,
        ),
        volatility_boundaries: Sequence[float] = (
            0.01,
            0.03,
        ),
        position_boundaries: Sequence[float] = (
            0.0,
            1.0,
        ),
    ) -> None:
        self._return_boundaries = self._validate_boundaries(
            return_boundaries,
            "return_boundaries",
        )

        self._volatility_boundaries = self._validate_boundaries(
            volatility_boundaries,
            "volatility_boundaries",
        )

        self._position_boundaries = self._validate_boundaries(
            position_boundaries,
            "position_boundaries",
        )

    def transform(
        self,
        state: MarketState,
    ) -> DiscreteState:
        """
        Convert a continuous MarketState into a DiscreteState.
        """

        return DiscreteState(
            return_bucket=self._bucket(
                state.return_value,
                self._return_boundaries,
            ),
            volatility_bucket=self._bucket(
                state.volatility,
                self._volatility_boundaries,
            ),
            position_bucket=self._bucket(
                state.position,
                self._position_boundaries,
            ),
        )

    @staticmethod
    def _bucket(
        value: float,
        boundaries: tuple[float, ...],
    ) -> int:
        """
        Assign a value to an integer bucket.

        Example boundaries:

            [-0.02, 0.0, 0.02]

        produce four buckets:

            0: value < -0.02
            1: -0.02 <= value < 0.0
            2: 0.0 <= value < 0.02
            3: value >= 0.02
        """

        for index, boundary in enumerate(boundaries):
            if value < boundary:
                return index

        return len(boundaries)

    @staticmethod
    def _validate_boundaries(
        boundaries: Sequence[float],
        name: str,
    ) -> tuple[float, ...]:
        """
        Validate that bucket boundaries are strictly increasing.
        """

        normalized = tuple(float(value) for value in boundaries)

        if any(
            left >= right
            for left, right in zip(
                normalized,
                normalized[1:],
            )
        ):
            raise ValueError(
                f"{name} must be strictly increasing."
            )

        return normalized