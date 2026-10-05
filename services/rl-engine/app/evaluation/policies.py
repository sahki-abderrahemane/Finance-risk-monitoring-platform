from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.agent.q_learning import QLearningAgent
from app.environment.state import Action, MarketState
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.evaluation.metrics import (
    EvaluationMetrics,
    RiskMetricsCalculator,
)


PolicyFunction = Callable[[MarketState], int]


@dataclass(frozen=True)
class PolicyEvaluation:
    """
    Evaluation result for one policy.
    """

    policy_name: str
    metrics: EvaluationMetrics


class BaselinePolicies:
    """
    Collection of simple non-learning policies used as
    research baselines.
    """

    @staticmethod
    def hold(
        _state: MarketState,
    ) -> int:
        return Action.HOLD

    @staticmethod
    def always_increase(
        _state: MarketState,
    ) -> int:
        return Action.INCREASE

    @staticmethod
    def always_reduce(
        _state: MarketState,
    ) -> int:
        return Action.REDUCE


class PolicyEvaluator:
    """
    Evaluates deterministic or learned policies on the same
    simulated environment.
    """

    def __init__(
        self,
        environment: ReinforcementLearningEnvironment,
        metrics_calculator: RiskMetricsCalculator,
    ) -> None:
        self._environment = environment
        self._metrics_calculator = (
            metrics_calculator
        )

    def evaluate(
        self,
        policy_name: str,
        policy: PolicyFunction,
    ) -> PolicyEvaluation:
        """
        Evaluate a policy for one complete simulation episode.
        """

        state = self._environment.reset()

        step_returns: list[float] = []
        rewards: list[float] = []

        while True:
            action = policy(state)

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

            state = transition.next_state

            if transition.done:
                break

        metrics = (
            self._metrics_calculator.calculate(
                step_returns=step_returns,
                rewards=rewards,
            )
        )

        return PolicyEvaluation(
            policy_name=policy_name,
            metrics=metrics,
        )

    def evaluate_q_learning(
        self,
        policy_name: str,
        agent: QLearningAgent,
    ) -> PolicyEvaluation:
        """
        Evaluate a learned Q-Learning policy greedily.

        The Q-table is used only for action selection.
        """

        def policy(
            state: MarketState,
        ) -> int:
            return agent.select_action(state)

        original_exploration_rate = (
            agent.exploration_rate
        )

        agent._exploration_rate = 0.0

        try:
            return self.evaluate(
                policy_name=policy_name,
                policy=policy,
            )
        finally:
            agent._exploration_rate = (
                original_exploration_rate
            )