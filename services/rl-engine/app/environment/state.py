from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketState:
   

    return_value: float
    volatility: float
    position: float

    def as_tuple(self) -> tuple[float, float, float]:
        """
        Convert the state into the numerical representation
        consumed by an RL agent.
        """

        return (
            self.return_value,
            self.volatility,
            self.position,
        )


class Action:
    """
    Discrete actions available to the simulated RL agent.
    """

    REDUCE = 0
    HOLD = 1
    INCREASE = 2

    @classmethod
    def values(cls) -> tuple[int, int, int]:
        """
        Return all supported discrete actions.
        """

        return (
            cls.REDUCE,
            cls.HOLD,
            cls.INCREASE,
        )

    @classmethod
    def validate(cls, action: int) -> None:
        """
        Validate that an action belongs to the supported
        simulation action space.
        """

        if action not in cls.values():
            raise ValueError(
                f"Unsupported action: {action}. "
                f"Expected one of {cls.values()}."
            )