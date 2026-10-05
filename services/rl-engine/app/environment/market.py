from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MarketStep:
    """
    Result of one simulated market-environment transition.
    """

    timestamp: int
    price: float
    return_value: float
    done: bool


class SimulatedMarketEnvironment:
 

    def __init__(
        self,
        prices: Sequence[float],
    ) -> None:
        if len(prices) < 2:
            raise ValueError(
                "The environment requires at least two prices."
            )

        normalized_prices = [
            float(price)
            for price in prices
        ]

        if any(price <= 0 for price in normalized_prices):
            raise ValueError(
                "All market prices must be greater than zero."
            )

        self._prices = tuple(normalized_prices)
        self._current_step = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current_price(self) -> float:
        return self._prices[self._current_step]

    @property
    def observation_size(self) -> int:
        """
        Current minimal observation consists of the current price.
        """

        return 1

    def reset(self) -> tuple[float]:
        """
        Reset the simulation to its initial market state.
        """

        self._current_step = 0

        return self._observation()

    def step(
        self,
    ) -> MarketStep:
        """
        Advance the simulation by one market observation.

        The environment itself does not select an RL action yet.
        Action-aware portfolio transitions will be introduced after
        the market-state foundation is established.
        """

        if self._current_step >= len(self._prices) - 1:
            raise RuntimeError(
                "The market episode has already terminated."
            )

        previous_price = self._prices[
            self._current_step
        ]

        self._current_step += 1

        current_price = self._prices[
            self._current_step
        ]

        return_value = (
            current_price - previous_price
        ) / previous_price

        done = (
            self._current_step
            >= len(self._prices) - 1
        )

        return MarketStep(
            timestamp=self._current_step,
            price=current_price,
            return_value=return_value,
            done=done,
        )

    def observation(self) -> tuple[float]:
        """
        Return the current environment observation.
        """

        return self._observation()

    def _observation(self) -> tuple[float]:
        return (
            self._prices[self._current_step],
        )