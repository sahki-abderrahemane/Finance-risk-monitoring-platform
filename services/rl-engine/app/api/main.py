from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from sentinel_observability.config import get_observability_config
from sentinel_observability.logging import configure_logging, get_logger
from sentinel_observability.metrics import SentinelMetrics
from sentinel_observability.middleware import add_observability_middleware
from sentinel_observability.tracing import configure_tracing

from app.api.schemas import (
    RLAlgorithmMetrics,
    RLComparisonRequest,
    RLComparisonResponse,
)
from app.evaluation.comparison_runner import (
    RLComparisonRunner,
)
from app.experiment.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    RewardConfig,
    TrainingConfig,
)
from app.training.ppo_trainer import (
    PPOTrainingConfig,
)

config = get_observability_config()
config.service_name = "rl-engine"

configure_logging(
    service="rl-engine",
    environment=config.environment,
    log_level=config.log_level,
    log_format=config.log_format,
)

tracer_provider = configure_tracing(service_name="rl-engine")

logger = get_logger("rl-engine")

app = FastAPI(
    title="Sentinel-AI RL Engine",
    version="0.1.0",
    description=(
        "Research and simulation API for Sentinel-AI "
        "reinforcement-learning experiments."
    ),
)

add_observability_middleware(app, config=config)


@app.get("/health")
def health() -> dict[str, str]:
    """
    Service health endpoint.
    """

    return {
        "status": "ok",
        "service": "rl-engine",
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    """
    Readiness probe — always ready for the RL engine
    (no external dependencies required at startup).
    """

    return {"status": "ready", "service": "rl-engine"}


@app.post(
    "/v1/rl/compare",
    response_model=RLComparisonResponse,
)
def compare_rl_algorithms(
    request: RLComparisonRequest,
) -> RLComparisonResponse:
    """
    Run a Q-Learning vs PPO simulation experiment.

    The supplied prices must be historical or synthetic data.
    No live trading or execution is performed.
    """

    if any(
        price <= 0
        for price in request.prices
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "All prices must be greater than zero."
            ),
        )

    try:
        experiment_config = ExperimentConfig(
            seed=request.seed,
            environment=EnvironmentConfig(),
            reward=RewardConfig(),
            training=TrainingConfig(
                episodes=request.training_episodes,
            ),
            evaluation=EvaluationConfig(
                episodes=1,
            ),
        )

        ppo_config = PPOTrainingConfig(
            rollout_steps=request.ppo_rollout_steps,
            optimization_epochs=(
                request.ppo_optimization_epochs
            ),
            minibatch_size=(
                request.ppo_minibatch_size
            ),
        )

        runner = RLComparisonRunner(
            prices=request.prices,
            config=experiment_config,
            ppo_training_config=ppo_config,
        )

        result = runner.run()

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    comparison = result.comparison

    q_learning = comparison.q_learning
    ppo = comparison.ppo

    return RLComparisonResponse(
        q_learning=RLAlgorithmMetrics(
            total_return=(
                q_learning.metrics.total_return
            ),
            average_step_return=(
                q_learning.metrics.average_step_return
            ),
            volatility=(
                q_learning.metrics.volatility
            ),
            maximum_drawdown=(
                q_learning.metrics.maximum_drawdown
            ),
            total_reward=(
                q_learning.metrics.total_reward
            ),
            steps=q_learning.metrics.steps,
            action_counts=dict(
                q_learning.action_counts
            ),
        ),
        ppo=RLAlgorithmMetrics(
            total_return=(
                ppo.metrics.total_return
            ),
            average_step_return=(
                ppo.metrics.average_step_return
            ),
            volatility=(
                ppo.metrics.volatility
            ),
            maximum_drawdown=(
                ppo.metrics.maximum_drawdown
            ),
            total_reward=(
                ppo.metrics.total_reward
            ),
            steps=ppo.metrics.steps,
            action_counts=dict(
                ppo.action_counts
            ),
        ),
        train_size=result.train_size,
        evaluation_size=result.evaluation_size,
        seed=result.seed,
    )
