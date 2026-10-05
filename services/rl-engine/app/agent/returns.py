from __future__ import annotations

from typing import Sequence


class DiscountedReturnCalculator:
    """
    Calculates discounted Monte-Carlo returns.

    For timestep t:

        G_t =
            r_t
            + γ r_{t+1}
            + γ² r_{t+2}
            + ...

    The implementation works backward through the trajectory.
    """

    def __init__(
        self,
        discount_factor: float = 0.99,
    ) -> None:
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0, 1]."
            )

        self._discount_factor = float(
            discount_factor
        )

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def calculate(
        self,
        rewards: Sequence[float],
    ) -> tuple[float, ...]:
        """
        Calculate one discounted return for every timestep.
        """

        if not rewards:
            return ()

        returns = [0.0] * len(rewards)

        running_return = 0.0

        for index in range(
            len(rewards) - 1,
            -1,
            -1,
        ):
            running_return = (
                float(rewards[index])
                + self._discount_factor
                * running_return
            )

            returns[index] = running_return

        return tuple(returns)