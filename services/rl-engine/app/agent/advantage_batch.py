from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.agent.advantage import (
    GeneralizedAdvantageEstimator,
)


@dataclass(frozen=True)
class AdvantageBatch:
  

    advantages: tuple[float, ...]
    returns: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.advantages) != len(
            self.returns
        ):
            raise ValueError(
                "advantages and returns must have "
                "the same length."
            )

    @property
    def size(self) -> int:
        return len(self.advantages)


class AdvantageBatchBuilder:
    """
    Build actor and critic targets from a simulated trajectory.

    The actor uses the estimated advantages.

    The critic uses return targets:

        return_t = advantage_t + V(s_t)
    """

    def __init__(
        self,
        estimator: GeneralizedAdvantageEstimator,
    ) -> None:
        self._estimator = estimator

    def build(
        self,
        rewards: Sequence[float],
        state_values: Sequence[float],
        next_state_values: Sequence[float],
        dones: Sequence[bool],
    ) -> AdvantageBatch:
        advantages = self._estimator.calculate(
            rewards=rewards,
            state_values=state_values,
            next_state_values=next_state_values,
            dones=dones,
        )

        returns = tuple(
            advantage + float(state_value)
            for advantage, state_value
            in zip(
                advantages,
                state_values,
            )
        )

        return AdvantageBatch(
            advantages=advantages,
            returns=returns,
        )