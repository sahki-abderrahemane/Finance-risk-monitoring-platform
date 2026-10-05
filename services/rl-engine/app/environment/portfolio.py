from __future__ import annotations

from dataclasses import dataclass

from app.environment.state import Action


@dataclass(frozen=True)
class PortfolioTransition:
    """
    Result of applying an RL action to the simulated portfolio.
    """

    previous_position: float
    new_position: float
    action: int


class SimulatedPortfolio:
   

    def __init__(
        self,
        initial_position: float = 0.0,
        min_position: float = 0.0,
        max_position: float = 1.0,
    ) -> None:
        if min_position > max_position:
            raise ValueError(
                "min_position cannot exceed max_position."
            )

        if not (
            min_position
            <= initial_position
            <= max_position
        ):
            raise ValueError(
                "initial_position must be inside "
                "the configured position bounds."
            )

        self._initial_position = float(initial_position)
        self._min_position = float(min_position)
        self._max_position = float(max_position)

        self._position = self._initial_position

    @property
    def position(self) -> float:
        """
        Current simulated portfolio exposure.
        """

        return self._position

    @property
    def min_position(self) -> float:
        return self._min_position

    @property
    def max_position(self) -> float:
        return self._max_position

    def reset(self) -> float:
        """
        Reset the simulated portfolio to its initial position.
        """

        self._position = self._initial_position

        return self._position

    def apply_action(
        self,
        action: int,
    ) -> PortfolioTransition:
        """
        Apply a discrete RL action to the simulated position.

        REDUCE:
            Move exposure toward the minimum.

        HOLD:
            Keep the current exposure.

        INCREASE:
            Move exposure toward the maximum.
        """

        Action.validate(action)

        previous_position = self._position

        if action == Action.REDUCE:
            self._position = self._min_position

        elif action == Action.HOLD:
            self._position = previous_position

        elif action == Action.INCREASE:
            self._position = self._max_position

        return PortfolioTransition(
            previous_position=previous_position,
            new_position=self._position,
            action=action,
        )