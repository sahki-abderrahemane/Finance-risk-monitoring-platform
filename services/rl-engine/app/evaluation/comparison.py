from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.ppo import PPOEvaluationResult


@dataclass(frozen=True)
class AlgorithmEvaluation:
   

    algorithm_name: str
    metrics: EvaluationMetrics
    action_counts: dict[int, int]


@dataclass(frozen=True)
class RLAlgorithmComparison:


    q_learning: AlgorithmEvaluation
    ppo: AlgorithmEvaluation

    @property
    def algorithms(self) -> tuple[str, str]:
        return (
            self.q_learning.algorithm_name,
            self.ppo.algorithm_name,
        )

    @property
    def total_returns(self) -> dict[str, float]:
        return {
            self.q_learning.algorithm_name: (
                self.q_learning.metrics.total_return
            ),
            self.ppo.algorithm_name: (
                self.ppo.metrics.total_return
            ),
        }

    @property
    def volatilities(self) -> dict[str, float]:
        return {
            self.q_learning.algorithm_name: (
                self.q_learning.metrics.volatility
            ),
            self.ppo.algorithm_name: (
                self.ppo.metrics.volatility
            ),
        }

    @property
    def maximum_drawdowns(self) -> dict[str, float]:
        return {
            self.q_learning.algorithm_name: (
                self.q_learning.metrics.maximum_drawdown
            ),
            self.ppo.algorithm_name: (
                self.ppo.metrics.maximum_drawdown
            ),
        }

    @property
    def total_rewards(self) -> dict[str, float]:
        return {
            self.q_learning.algorithm_name: (
                self.q_learning.metrics.total_reward
            ),
            self.ppo.algorithm_name: (
                self.ppo.metrics.total_reward
            ),
        }

    @property
    def action_distributions(
        self,
    ) -> dict[str, dict[int, int]]:
        return {
            self.q_learning.algorithm_name: dict(
                self.q_learning.action_counts
            ),
            self.ppo.algorithm_name: dict(
                self.ppo.action_counts
            ),
        }

    def as_dict(
        self,
    ) -> dict[str, dict[str, float | int | dict[int, int]]]:
        return {
            self.q_learning.algorithm_name: (
                self._algorithm_dict(
                    self.q_learning
                )
            ),
            self.ppo.algorithm_name: (
                self._algorithm_dict(
                    self.ppo
                )
            ),
        }

    @staticmethod
    def _algorithm_dict(
        evaluation: AlgorithmEvaluation,
    ) -> dict[
        str,
        float | int | dict[int, int],
    ]:
        metrics = evaluation.metrics

        return {
            "total_return": metrics.total_return,
            "average_step_return": (
                metrics.average_step_return
            ),
            "volatility": metrics.volatility,
            "maximum_drawdown": (
                metrics.maximum_drawdown
            ),
            "total_reward": metrics.total_reward,
            "steps": metrics.steps,
            "action_counts": dict(
                evaluation.action_counts
            ),
        }


class RLComparisonBuilder:
    """
    Converts algorithm-specific evaluation results into the
    common Sentinel-AI comparison representation.
    """

    @staticmethod
    def from_results(
        q_learning_metrics: EvaluationMetrics,
        q_learning_actions: tuple[int, ...],
        ppo_result: PPOEvaluationResult,
    ) -> RLAlgorithmComparison:
        q_learning_action_counts = (
            RLComparisonBuilder._count_actions(
                q_learning_actions
            )
        )

        ppo_action_counts = ppo_result.action_counts

        return RLAlgorithmComparison(
            q_learning=AlgorithmEvaluation(
                algorithm_name="q_learning",
                metrics=q_learning_metrics,
                action_counts=q_learning_action_counts,
            ),
            ppo=AlgorithmEvaluation(
                algorithm_name="ppo",
                metrics=ppo_result.metrics,
                action_counts=ppo_action_counts,
            ),
        )

    @staticmethod
    def _count_actions(
        actions: tuple[int, ...],
    ) -> dict[int, int]:
        counts = {
            0: 0,
            1: 0,
            2: 0,
        }

        for action in actions:
            if action not in counts:
                raise ValueError(
                    f"Unsupported action: {action}."
                )

            counts[action] += 1

        return counts