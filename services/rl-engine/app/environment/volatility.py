from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Iterable


class RollingVolatility:
  

    def __init__(
        self,
        window_size: int = 5,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        self._window_size = window_size
        self._returns: deque[float] = deque(
            maxlen=window_size
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def observations(self) -> tuple[float, ...]:
        return tuple(self._returns)

    def reset(self) -> None:
        """
        Clear all observations.
        """

        self._returns.clear()

    def update(
        self,
        return_value: float,
    ) -> float:
        """
        Add a return observation and return the current
        rolling volatility.
        """

        self._returns.append(
            float(return_value)
        )

        return self.value()

    def value(self) -> float:
        """
        Return population standard deviation of the current
        rolling return window.

        A single observation has zero measurable dispersion.
        """

        if len(self._returns) < 2:
            return 0.0

        values = tuple(self._returns)

        average = sum(values) / len(values)

        variance = sum(
            (value - average) ** 2
            for value in values
        ) / len(values)

        return sqrt(variance)

    def calculate(
        self,
        returns: Iterable[float],
    ) -> float:
       

        self.reset()

        volatility = 0.0

        for return_value in returns:
            volatility = self.update(
                return_value
            )

        return volatility