from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from app.agent.ppo_policy import PPOPolicy
from app.agent.trajectory import (
    ActorCriticStep,
    ActorCriticTrajectory,
)
from app.environment.state import MarketState
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)


@dataclass(frozen=True)
class RolloutResult:
   

    trajectory: ActorCriticTrajectory
    final_state: MarketState
    terminated: bool


class PPORolloutCollector:
   
    def __init__(
        self,
        environment: ReinforcementLearningEnvironment,
        policy: PPOPolicy,
        device: torch.device | None = None,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._device = device or torch.device("cpu")

    def collect(
        self,
        max_steps: int,
    ) -> RolloutResult:
        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        state = self._environment.reset()

        steps: list[ActorCriticStep] = []

        for _ in range(max_steps):
            observation = self._state_to_tensor(
                state
            )

            with torch.no_grad():
                (
                    action_tensor,
                    log_probability_tensor,
                    value_tensor,
                ) = self._policy.sample_action(
                    observation
                )

            action = int(
                action_tensor.item()
            )

            old_log_probability = float(
                log_probability_tensor.item()
            )

            state_value = float(
                value_tensor.item()
            )

            transition = self._environment.step(
                action
            )

            next_state = transition.next_state

            next_observation = (
                self._state_to_tensor(next_state)
            )

            with torch.no_grad():
                next_value_tensor = (
                    self._policy.network.state_values(
                        next_observation
                    )
                )

            next_state_value = float(
                next_value_tensor.item()
            )

            steps.append(
                ActorCriticStep(
                    state=self._state_tuple(state),
                    action=action,
                    reward=transition.reward,
                    next_state=self._state_tuple(
                        next_state
                    ),
                    done=transition.done,
                    action_log_probability=(
                        old_log_probability
                    ),
                    state_value=state_value,
                )
            )

            state = next_state

            if transition.done:
                return RolloutResult(
                    trajectory=(
                        ActorCriticTrajectory.from_steps(
                            steps
                        )
                    ),
                    final_state=state,
                    terminated=True,
                )

        return RolloutResult(
            trajectory=(
                ActorCriticTrajectory.from_steps(
                    steps
                )
            ),
            final_state=state,
            terminated=False,
        )

    def final_bootstrap_value(
        self,
        rollout: RolloutResult,
    ) -> float:
        """
        Return V(s_T) for a truncated rollout.

        A truly terminal episode has no future value, so its
        bootstrap value is zero.

        A rollout stopped only because max_steps was reached
        still has a possible future value and therefore uses
        the critic's estimate.
        """

        if rollout.terminated:
            return 0.0

        observation = self._state_to_tensor(
            rollout.final_state
        )

        with torch.no_grad():
            value = self._policy.network.state_values(
                observation
            )

        return float(value.item())

    def _state_to_tensor(
        self,
        state: MarketState,
    ) -> Tensor:
        return torch.tensor(
            [self._state_tuple(state)],
            dtype=torch.float32,
            device=self._device,
        )

    @staticmethod
    def _state_tuple(
        state: MarketState,
    ) -> tuple[float, ...]:
        return state.as_tuple()