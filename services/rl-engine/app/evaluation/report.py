from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.ppo import PPOEvaluationResult


@dataclass(frozen=True)
class PPOEvaluationReport:
    """
    Serializable summary of one PPO evaluation.

    This class is intentionally independent of PyTorch so it
    can later be passed to API, MLflow, or reporting layers.
    """

    total_return: float
    average_step_return: float
    volatility: float
    maximum_drawdown: float
    total_reward: float
    steps: int

    deterministic: bool

    reduce_actions: int
    hold_actions: int
    increase_actions: int

    @classmethod
    def from_result(
        cls,
        result: PPOEvaluationResult,
    ) -> PPOEvaluationReport:
        action_counts = result.action_counts

        return cls(
            total_return=result.metrics.total_return,
            average_step_return=(
                result.metrics.average_step_return
            ),
            volatility=result.metrics.volatility,
            maximum_drawdown=(
                result.metrics.maximum_drawdown
            ),
            total_reward=result.metrics.total_reward,
            steps=result.metrics.steps,
            deterministic=result.deterministic,
            reduce_actions=action_counts[0],
            hold_actions=action_counts[1],
            increase_actions=action_counts[2],
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        """
        Convert the report into a structure suitable for
        logging or serialization.
        """

        return {
            "total_return": self.total_return,
            "average_step_return": (
                self.average_step_return
            ),
            "volatility": self.volatility,
            "maximum_drawdown": (
                self.maximum_drawdown
            ),
            "total_reward": self.total_reward,
            "steps": self.steps,
            "deterministic": self.deterministic,
            "reduce_actions": self.reduce_actions,
            "hold_actions": self.hold_actions,
            "increase_actions": self.increase_actions,
        }