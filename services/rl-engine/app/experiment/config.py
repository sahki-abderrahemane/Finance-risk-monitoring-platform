from __future__ import annotations

from dataclasses import dataclass

from app.agent.q_learning import QLearningConfig


@dataclass(frozen=True)
class EnvironmentConfig:
   

    volatility_window: int = 5
    initial_position: float = 0.0
    min_position: float = 0.0
    max_position: float = 1.0

    def __post_init__(self) -> None:
        if self.volatility_window <= 0:
            raise ValueError(
                "volatility_window must be greater than zero."
            )

        if self.min_position > self.max_position:
            raise ValueError(
                "min_position cannot exceed max_position."
            )

        if not (
            self.min_position
            <= self.initial_position
            <= self.max_position
        ):
            raise ValueError(
                "initial_position must be within the "
                "configured position bounds."
            )


@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration for the simulated risk-aware reward.
    """

    risk_penalty_coefficient: float = 0.0
    transaction_cost_coefficient: float = 1.0
    drawdown_penalty_coefficient: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.risk_penalty_coefficient,
            self.transaction_cost_coefficient,
            self.drawdown_penalty_coefficient,
        )

        if any(value < 0 for value in values):
            raise ValueError(
                "Reward coefficients cannot be negative."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for Q-Learning training.
    """

    episodes: int = 100

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for deterministic evaluation.
    """

    episodes: int = 1

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )


@dataclass(frozen=True)
class ExperimentConfig:
    

    seed: int = 42

    environment: EnvironmentConfig = EnvironmentConfig()
    reward: RewardConfig = RewardConfig()
    q_learning: QLearningConfig = QLearningConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )