from __future__ import annotations

from dataclasses import dataclass

import torch

from app.agent.ppo_policy import PPOPolicy
from app.environment.state import Action, MarketState
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.evaluation.metrics import (
    EvaluationMetrics,
    RiskMetricsCalculator,
)


@dataclass(frozen=True)
class PPOEvaluationResult:
    """
    Result of evaluating a trained PPO policy.

    The result contains both the risk metrics and basic
    behavioral information about the evaluated policy.
    """

    metrics: EvaluationMetrics
    actions: tuple[int, ...]
    deterministic: bool

    @property
    def action_counts(self) -> dict[int, int]:
        counts = {
            action: 0
            for action in Action.values()
        }

        for action in self.actions:
            counts[action] += 1

        return counts


class PPOEvaluator:
    """
    Evaluate a trained PPO policy in the simulated environment.

    Evaluation never modifies policy parameters.

    Two action-selection modes are supported:

        deterministic=True:
            select argmax action.

        deterministic=False:
            sample from the policy distribution.

    Both modes use the same frozen network.
    """

    def __init__(
        self,
        environment: ReinforcementLearningEnvironment,
        policy: PPOPolicy,
        metrics_calculator: RiskMetricsCalculator,
        device: torch.device | None = None,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._metrics_calculator = (
            metrics_calculator
        )
        self._device = device or torch.device("cpu")

    def evaluate(
        self,
        deterministic: bool = True,
    ) -> PPOEvaluationResult:
        """
        Evaluate one complete simulated episode.

        No gradients are calculated and no optimizer step occurs.
        """

        state = self._environment.reset()

        step_returns: list[float] = []
        rewards: list[float] = []
        actions: list[int] = []

        was_training = self._policy.network.training

        self._policy.network.eval()

        try:
            while True:
                observation = self._state_to_tensor(
                    state
                )

                action = self._select_action(
                    observation=observation,
                    deterministic=deterministic,
                )

                transition = self._environment.step(
                    action
                )

                step_returns.append(
                    transition.next_state.return_value
                    * transition.state.position
                )

                rewards.append(
                    transition.reward
                )

                actions.append(action)

                state = transition.next_state

                if transition.done:
                    break

        finally:
            self._policy.network.train(
                was_training
            )

        metrics = (
            self._metrics_calculator.calculate(
                step_returns=step_returns,
                rewards=rewards,
            )
        )

        return PPOEvaluationResult(
            metrics=metrics,
            actions=tuple(actions),
            deterministic=deterministic,
        )

    def _select_action(
        self,
        observation: torch.Tensor,
        deterministic: bool,
    ) -> int:
        with torch.no_grad():
            distribution = self._policy.distribution(
                observation
            )

            if deterministic:
                action = torch.argmax(
                    distribution.probs,
                    dim=-1,
                )
            else:
                action = distribution.sample()

        return int(action.item())

    def _state_to_tensor(
        self,
        state: MarketState,
    ) -> torch.Tensor:
        return torch.tensor(
            [state.as_tuple()],
            dtype=torch.float32,
            device=self._device,
        )