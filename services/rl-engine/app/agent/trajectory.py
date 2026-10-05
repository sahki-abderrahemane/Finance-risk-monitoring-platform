from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ActorCriticStep:
  

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool

    action_log_probability: float | None = None
    state_value: float | None = None


@dataclass(frozen=True)
class ActorCriticTrajectory:
 
    steps: tuple[ActorCriticStep, ...]

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def states(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            step.state
            for step in self.steps
        )

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(
            step.action
            for step in self.steps
        )

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(
            step.reward
            for step in self.steps
        )

    @property
    def dones(self) -> tuple[bool, ...]:
        return tuple(
            step.done
            for step in self.steps
        )

    @property
    def log_probabilities(
        self,
    ) -> tuple[float, ...]:
        values: list[float] = []

        for step in self.steps:
            if step.action_log_probability is None:
                raise ValueError(
                    "Action log probability is missing "
                    "from the trajectory."
                )

            values.append(
                step.action_log_probability
            )

        return tuple(values)

    @property
    def state_values(self) -> tuple[float, ...]:
        values: list[float] = []

        for step in self.steps:
            if step.state_value is None:
                raise ValueError(
                    "State value is missing "
                    "from the trajectory."
                )

            values.append(step.state_value)

        return tuple(values)

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[ActorCriticStep],
    ) -> ActorCriticTrajectory:
        normalized_steps = tuple(steps)

        if not normalized_steps:
            raise ValueError(
                "A trajectory must contain at least one step."
            )

        return cls(
            steps=normalized_steps
        )

    def total_reward(self) -> float:
        return sum(
            step.reward
            for step in self.steps
        )