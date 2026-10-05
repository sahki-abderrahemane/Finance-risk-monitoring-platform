from __future__ import annotations

from pydantic import BaseModel, Field


class RLComparisonRequest(BaseModel):
    """
    Request for a Q-Learning vs PPO simulation experiment.

    Prices must represent historical or synthetic data only.
    """

    prices: list[float] = Field(
        ...,
        min_length=8,
        description=(
            "Historical or synthetic prices used exclusively "
            "for simulation."
        ),
    )

    training_episodes: int = Field(
        default=100,
        gt=0,
        description="Number of training episodes.",
    )

    ppo_rollout_steps: int = Field(
        default=128,
        gt=0,
        description="Maximum PPO rollout length.",
    )

    ppo_optimization_epochs: int = Field(
        default=4,
        gt=0,
        description="PPO optimization epochs per rollout.",
    )

    ppo_minibatch_size: int = Field(
        default=32,
        gt=0,
        description="PPO minibatch size.",
    )

    seed: int = Field(
        default=42,
        ge=0,
        description="Reproducibility seed.",
    )


class RLAlgorithmMetrics(BaseModel):
    """
    Evaluation metrics for one RL algorithm.
    """

    total_return: float
    average_step_return: float
    volatility: float
    maximum_drawdown: float
    total_reward: float
    steps: int
    action_counts: dict[int, int]


class RLComparisonResponse(BaseModel):
    """
    API representation of a Q-Learning vs PPO comparison.

    The response exposes measurements without ranking the
    algorithms.
    """

    q_learning: RLAlgorithmMetrics
    ppo: RLAlgorithmMetrics

    train_size: int
    evaluation_size: int
    seed: int