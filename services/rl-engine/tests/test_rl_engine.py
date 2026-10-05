from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.agent.advantage import (
    GeneralizedAdvantageEstimator,
)
from app.agent.advantage_batch import (
    AdvantageBatchBuilder,
)
from app.agent.advantage_normalization import (
    AdvantageNormalizer,
)
from app.agent.networks import ActorCriticNetwork
from app.agent.ppo_objective import PPOObjective
from app.agent.ppo_policy import PPOPolicy
from app.agent.q_learning import (
    QLearningAgent,
    QLearningConfig,
)
from app.agent.returns import (
    DiscountedReturnCalculator,
)
from app.agent.state_discretizer import (
    StateDiscretizer,
)
from app.agent.trajectory import (
    ActorCriticStep,
    ActorCriticTrajectory,
)
from app.api.main import app
from app.environment.market import (
    SimulatedMarketEnvironment,
)
from app.environment.portfolio import (
    SimulatedPortfolio,
)
from app.environment.reward import (
    RiskAdjustedReward,
)
from app.environment.state import (
    Action,
    MarketState,
)
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.environment.volatility import (
    RollingVolatility,
)
from app.evaluation.comparison import (
    RLComparisonBuilder,
)
from app.evaluation.metrics import (
    RiskMetricsCalculator,
)
from app.evaluation.ppo import (
    PPOEvaluator,
)
from app.evaluation.policies import (
    BaselinePolicies,
    PolicyEvaluator,
)
from app.evaluation.report import (
    PPOEvaluationReport,
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
from app.training.episode import (
    EpisodeRunner,
)
from app.training.ppo_rollout import (
    PPORolloutCollector,
)
from app.training.ppo_trainer import (
    PPOTrainer,
    PPOTrainingConfig,
)
from app.training.trainer import (
    QLearningTrainer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> tuple[float, ...]:
    return (
        100.0,
        101.0,
        99.0,
        102.0,
        103.0,
        101.0,
        104.0,
        105.0,
        103.0,
        106.0,
        107.0,
        105.0,
        108.0,
        109.0,
        107.0,
        110.0,
        111.0,
        109.0,
        112.0,
        113.0,
    )


@pytest.fixture
def environment() -> ReinforcementLearningEnvironment:
    market = SimulatedMarketEnvironment(
        prices=(
            100.0,
            101.0,
            102.0,
            100.0,
            103.0,
        )
    )

    portfolio = SimulatedPortfolio(
        initial_position=0.0,
        min_position=0.0,
        max_position=1.0,
    )

    reward = RiskAdjustedReward()

    return ReinforcementLearningEnvironment(
        market=market,
        portfolio=portfolio,
        reward_function=reward,
    )


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        environment=EnvironmentConfig(),
        reward=RewardConfig(
            risk_penalty_coefficient=0.0,
            transaction_cost_coefficient=0.0,
            drawdown_penalty_coefficient=0.0,
        ),
        q_learning=QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.95,
            minimum_exploration_rate=0.01,
        ),
        training=TrainingConfig(
            episodes=5,
        ),
        evaluation=EvaluationConfig(
            episodes=1,
        ),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_market_environment_reset_and_step() -> None:
    environment = SimulatedMarketEnvironment(
        prices=(
            100.0,
            105.0,
            110.0,
        )
    )

    observation = environment.reset()

    assert observation == (100.0,)
    assert environment.current_step == 0
    assert environment.current_price == 100.0

    transition = environment.step()

    assert transition.timestamp == 1
    assert transition.price == 105.0
    assert transition.return_value == pytest.approx(0.05)
    assert transition.done is False


def test_market_environment_rejects_invalid_prices() -> None:
    with pytest.raises(ValueError):
        SimulatedMarketEnvironment(
            prices=(100.0,)
        )

    with pytest.raises(ValueError):
        SimulatedMarketEnvironment(
            prices=(100.0, 0.0)
        )


def test_market_environment_terminates() -> None:
    environment = SimulatedMarketEnvironment(
        prices=(
            100.0,
            101.0,
        )
    )

    environment.reset()

    transition = environment.step()

    assert transition.done is True

    with pytest.raises(RuntimeError):
        environment.step()


# ---------------------------------------------------------------------------
# State and actions
# ---------------------------------------------------------------------------


def test_action_values_and_validation() -> None:
    assert Action.values() == (
        Action.REDUCE,
        Action.HOLD,
        Action.INCREASE,
    )

    Action.validate(Action.REDUCE)
    Action.validate(Action.HOLD)
    Action.validate(Action.INCREASE)

    with pytest.raises(ValueError):
        Action.validate(999)


def test_market_state_as_tuple() -> None:
    state = MarketState(
        return_value=0.02,
        volatility=0.01,
        position=1.0,
    )

    assert state.as_tuple() == (
        0.02,
        0.01,
        1.0,
    )


def test_state_discretizer() -> None:
    discretizer = StateDiscretizer()

    state = MarketState(
        return_value=0.01,
        volatility=0.02,
        position=0.5,
    )

    discrete_state = discretizer.transform(
        state
    )

    assert len(discrete_state.as_tuple()) == 3
    assert all(
        isinstance(value, int)
        for value in discrete_state.as_tuple()
    )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def test_portfolio_actions() -> None:
    portfolio = SimulatedPortfolio(
        initial_position=0.0,
        min_position=0.0,
        max_position=1.0,
    )

    assert portfolio.position == 0.0

    increase = portfolio.apply_action(
        Action.INCREASE
    )

    assert increase.previous_position == 0.0
    assert increase.new_position == 1.0

    hold = portfolio.apply_action(
        Action.HOLD
    )

    assert hold.new_position == 1.0

    reduce = portfolio.apply_action(
        Action.REDUCE
    )

    assert reduce.new_position == 0.0


def test_portfolio_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        SimulatedPortfolio(
            initial_position=2.0,
            min_position=0.0,
            max_position=1.0,
        )

    with pytest.raises(ValueError):
        SimulatedPortfolio(
            initial_position=0.0,
            min_position=1.0,
            max_position=0.0,
        )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_risk_adjusted_reward() -> None:
    reward = RiskAdjustedReward(
        risk_penalty_coefficient=0.5,
        transaction_cost_coefficient=1.0,
        drawdown_penalty_coefficient=2.0,
    )

    result = reward.calculate(
        position=1.0,
        market_return=0.10,
        risk_value=0.02,
        transaction_cost=0.01,
        drawdown=0.05,
    )

    assert result.portfolio_return == pytest.approx(
        0.10
    )

    assert result.risk_penalty == pytest.approx(
        0.01
    )

    assert result.transaction_cost == pytest.approx(
        0.01
    )

    assert result.drawdown_penalty == pytest.approx(
        0.10
    )

    assert result.total_reward == pytest.approx(
        -0.02
    )


# ---------------------------------------------------------------------------
# Environment transition
# ---------------------------------------------------------------------------


def test_rl_environment_step(environment) -> None:
    state = environment.reset()

    assert state.position == 0.0
    assert state.return_value == 0.0

    transition = environment.step(
        Action.INCREASE
    )

    assert transition.action == Action.INCREASE
    assert transition.state.position == 0.0
    assert transition.next_state.position == 1.0
    assert transition.done is False


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------


def test_rolling_volatility() -> None:
    volatility = RollingVolatility(
        window_size=3
    )

    assert volatility.value() == 0.0

    volatility.update(0.01)

    assert volatility.value() == 0.0

    value = volatility.calculate(
        (
            0.01,
            0.02,
            0.03,
        )
    )

    assert value > 0.0
    assert len(volatility.observations) == 3


# ---------------------------------------------------------------------------
# Q-Learning
# ---------------------------------------------------------------------------


def test_q_learning_agent_selects_valid_action() -> None:
    agent = QLearningAgent(
        discretizer=StateDiscretizer(),
        config=QLearningConfig(
            exploration_rate=1.0,
        ),
        seed=42,
    )

    state = MarketState(
        return_value=0.01,
        volatility=0.01,
        position=0.0,
    )

    action = agent.select_action(state)

    assert action in Action.values()


def test_q_learning_update_changes_q_value(
    environment,
) -> None:
    agent = QLearningAgent(
        discretizer=StateDiscretizer(),
        seed=42,
    )

    state = environment.reset()

    transition = environment.step(
        Action.INCREASE
    )

    before = dict(
        agent.q_table
    )

    td_error = agent.update(
        transition
    )

    after = agent.q_table

    assert isinstance(td_error, float)
    assert len(after) >= len(before)


def test_q_learning_exploration_decay() -> None:
    config = QLearningConfig(
        exploration_rate=1.0,
        exploration_decay=0.5,
        minimum_exploration_rate=0.1,
    )

    agent = QLearningAgent(
        discretizer=StateDiscretizer(),
        config=config,
        seed=42,
    )

    agent.decay_exploration()

    assert agent.exploration_rate == pytest.approx(
        0.5
    )

    agent.decay_exploration()
    agent.decay_exploration()

    assert agent.exploration_rate == pytest.approx(
        0.125
    )


# ---------------------------------------------------------------------------
# Episode and Q-Learning training
# ---------------------------------------------------------------------------


def test_episode_runner(environment) -> None:
    agent = QLearningAgent(
        discretizer=StateDiscretizer(),
        seed=42,
    )

    runner = EpisodeRunner(
        environment=environment,
        agent=agent,
    )

    result = runner.run_training_episode()

    assert result.steps == 4
    assert isinstance(result.total_reward, float)
    assert 0.0 <= result.final_position <= 1.0


def test_q_learning_trainer(environment) -> None:
    agent = QLearningAgent(
        discretizer=StateDiscretizer(),
        seed=42,
    )

    runner = EpisodeRunner(
        environment=environment,
        agent=agent,
    )

    trainer = QLearningTrainer(
        runner=runner,
        agent=agent,
    )

    history = trainer.train(
        episodes=3
    )

    assert len(history.episodes) == 3
    assert len(history.total_rewards) == 3
    assert isinstance(
        history.average_reward,
        float,
    )


# ---------------------------------------------------------------------------
# Returns and policy gradient foundations
# ---------------------------------------------------------------------------


def test_discounted_returns() -> None:
    calculator = DiscountedReturnCalculator(
        discount_factor=0.9
    )

    returns = calculator.calculate(
        rewards=(1.0, 1.0, 1.0)
    )

    assert returns == pytest.approx(
        (
            2.71,
            1.9,
            1.0,
        )
    )


# ---------------------------------------------------------------------------
# Actor-Critic / neural network
# ---------------------------------------------------------------------------


def test_actor_critic_network_shapes() -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    observations = torch.tensor(
        (
            (0.01, 0.02, 0.0),
            (0.02, 0.01, 1.0),
        ),
        dtype=torch.float32,
    )

    logits, values = network(
        observations
    )

    assert logits.shape == (2, 3)
    assert values.shape == (2,)


def test_actor_critic_network_rejects_wrong_input() -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
    )

    invalid = torch.tensor(
        ((1.0, 2.0),),
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
    ):
        network(invalid)


# ---------------------------------------------------------------------------
# PPO policy
# ---------------------------------------------------------------------------


def test_ppo_policy_sampling() -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    policy = PPOPolicy(
        network=network
    )

    observations = torch.tensor(
        ((0.01, 0.02, 0.0),),
        dtype=torch.float32,
    )

    (
        actions,
        log_probabilities,
        values,
    ) = policy.sample_action(
        observations
    )

    assert actions.shape == (1,)
    assert log_probabilities.shape == (1,)
    assert values.shape == (1,)

    assert 0 <= int(actions.item()) <= 2


def test_ppo_policy_evaluates_actions() -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
    )

    policy = PPOPolicy(
        network=network
    )

    observations = torch.tensor(
        (
            (0.01, 0.02, 0.0),
            (0.02, 0.01, 1.0),
        ),
        dtype=torch.float32,
    )

    actions = torch.tensor(
        (0, 2),
        dtype=torch.long,
    )

    result = policy.evaluate_actions(
        observations=observations,
        actions=actions,
    )

    assert result.action_log_probabilities.shape == (
        2,
    )

    assert result.state_values.shape == (
        2,
    )

    assert result.entropy.shape == (
        2,
    )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def test_actor_critic_trajectory() -> None:
    trajectory = ActorCriticTrajectory.from_steps(
        (
            ActorCriticStep(
                state=(0.0, 0.0, 0.0),
                action=1,
                reward=0.1,
                next_state=(0.01, 0.02, 1.0),
                done=False,
                action_log_probability=-0.5,
                state_value=0.2,
            ),
            ActorCriticStep(
                state=(0.01, 0.02, 1.0),
                action=2,
                reward=0.2,
                next_state=(0.02, 0.03, 1.0),
                done=True,
                action_log_probability=-0.4,
                state_value=0.3,
            ),
        )
    )

    assert trajectory.length == 2
    assert trajectory.actions == (1, 2)
    assert trajectory.rewards == (
        0.1,
        0.2,
    )
    assert trajectory.total_reward() == pytest.approx(
        0.3
    )


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------


def test_gae() -> None:
    estimator = GeneralizedAdvantageEstimator(
        discount_factor=0.99,
        gae_lambda=0.95,
    )

    advantages = estimator.calculate(
        rewards=(1.0, 0.5),
        state_values=(0.5, 0.2),
        next_state_values=(0.2, 0.0),
        dones=(False, True),
    )

    assert len(advantages) == 2
    assert all(
        isinstance(value, float)
        for value in advantages
    )


def test_advantage_batch_builder() -> None:
    estimator = GeneralizedAdvantageEstimator()

    builder = AdvantageBatchBuilder(
        estimator=estimator
    )

    batch = builder.build(
        rewards=(1.0, 0.5),
        state_values=(0.5, 0.2),
        next_state_values=(0.2, 0.0),
        dones=(False, True),
    )

    assert batch.size == 2
    assert len(batch.advantages) == 2
    assert len(batch.returns) == 2


def test_advantage_normalization() -> None:
    normalizer = AdvantageNormalizer()

    advantages = torch.tensor(
        (1.0, 2.0, 3.0),
        dtype=torch.float32,
    )

    normalized = normalizer.normalize(
        advantages
    )

    assert normalized.mean().item() == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert normalized.shape == advantages.shape


# ---------------------------------------------------------------------------
# PPO objective
# ---------------------------------------------------------------------------


def test_ppo_objective() -> None:
    objective = PPOObjective(
        clip_epsilon=0.2,
        value_loss_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    new_log_probabilities = torch.tensor(
        (-0.1, -0.2),
        dtype=torch.float32,
    )

    old_log_probabilities = torch.tensor(
        (-0.15, -0.25),
        dtype=torch.float32,
    )

    advantages = torch.tensor(
        (1.0, -1.0),
        dtype=torch.float32,
    )

    state_values = torch.tensor(
        (0.5, 0.2),
        dtype=torch.float32,
    )

    return_targets = torch.tensor(
        (1.0, -0.5),
        dtype=torch.float32,
    )

    entropy = torch.tensor(
        (0.8, 0.7),
        dtype=torch.float32,
    )

    result = objective.calculate(
        new_log_probabilities=(
            new_log_probabilities
        ),
        old_log_probabilities=(
            old_log_probabilities
        ),
        advantages=advantages,
        state_values=state_values,
        return_targets=return_targets,
        entropy=entropy,
    )

    assert result.policy_loss.ndim == 0
    assert result.value_loss.ndim == 0
    assert result.entropy_bonus.ndim == 0
    assert result.total_loss.ndim == 0

    assert result.probability_ratio.shape == (
        2,
    )


# ---------------------------------------------------------------------------
# PPO rollout
# ---------------------------------------------------------------------------


def test_ppo_rollout_collector(environment) -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    policy = PPOPolicy(
        network=network
    )

    collector = PPORolloutCollector(
        environment=environment,
        policy=policy,
    )

    result = collector.collect(
        max_steps=10
    )

    assert result.trajectory.length > 0
    assert result.trajectory.length <= 10
    assert result.terminated is True


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------


def test_ppo_training_update(environment) -> None:
    torch.manual_seed(42)

    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    policy = PPOPolicy(
        network=network
    )

    trainer = PPOTrainer(
        environment=environment,
        policy=policy,
        advantage_estimator=(
            GeneralizedAdvantageEstimator()
        ),
        objective=PPOObjective(),
        config=PPOTrainingConfig(
            rollout_steps=4,
            optimization_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
            gradient_clip_norm=0.5,
        ),
    )

    result = trainer.train_update()

    assert result.updates > 0
    assert isinstance(
        result.policy_loss,
        float,
    )
    assert isinstance(
        result.value_loss,
        float,
    )
    assert isinstance(
        result.entropy,
        float,
    )
    assert isinstance(
        result.total_loss,
        float,
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_risk_metrics() -> None:
    calculator = RiskMetricsCalculator()

    metrics = calculator.calculate(
        step_returns=(
            0.01,
            -0.02,
            0.03,
        ),
        rewards=(
            0.01,
            -0.02,
            0.03,
        ),
    )

    assert metrics.steps == 3
    assert metrics.total_return != 0.0
    assert metrics.volatility > 0.0
    assert metrics.maximum_drawdown < 0.0


def test_risk_metrics_empty_input() -> None:
    calculator = RiskMetricsCalculator()

    metrics = calculator.calculate(
        step_returns=(),
        rewards=(),
    )

    assert metrics.steps == 0
    assert metrics.total_return == 0.0
    assert metrics.volatility == 0.0
    assert metrics.maximum_drawdown == 0.0


# ---------------------------------------------------------------------------
# Baseline policy evaluation
# ---------------------------------------------------------------------------


def test_baseline_policies(environment) -> None:
    assert (
        BaselinePolicies.hold(
            environment.state
        )
        == Action.HOLD
    )

    assert (
        BaselinePolicies.always_increase(
            environment.state
        )
        == Action.INCREASE
    )

    assert (
        BaselinePolicies.always_reduce(
            environment.state
        )
        == Action.REDUCE
    )


def test_policy_evaluator(environment) -> None:
    evaluator = PolicyEvaluator(
        environment=environment,
        metrics_calculator=RiskMetricsCalculator(),
    )

    result = evaluator.evaluate(
        policy_name="hold",
        policy=BaselinePolicies.hold,
    )

    assert result.policy_name == "hold"
    assert result.metrics.steps == 4


# ---------------------------------------------------------------------------
# PPO evaluation and report
# ---------------------------------------------------------------------------


def test_ppo_evaluation(environment) -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    policy = PPOPolicy(
        network=network
    )

    evaluator = PPOEvaluator(
        environment=environment,
        policy=policy,
        metrics_calculator=RiskMetricsCalculator(),
    )

    result = evaluator.evaluate(
        deterministic=True
    )

    assert result.deterministic is True
    assert len(result.actions) == 4
    assert sum(
        result.action_counts.values()
    ) == 4


def test_ppo_evaluation_report(environment) -> None:
    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
    )

    policy = PPOPolicy(
        network=network
    )

    evaluator = PPOEvaluator(
        environment=environment,
        policy=policy,
        metrics_calculator=RiskMetricsCalculator(),
    )

    result = evaluator.evaluate()

    report = PPOEvaluationReport.from_result(
        result
    )

    data = report.as_dict()

    assert "total_return" in data
    assert "volatility" in data
    assert "maximum_drawdown" in data
    assert "reduce_actions" in data
    assert "hold_actions" in data
    assert "increase_actions" in data


# ---------------------------------------------------------------------------
# Q-Learning vs PPO comparison
# ---------------------------------------------------------------------------


def test_comparison_builder(
    environment,
) -> None:
    q_metrics = RiskMetricsCalculator().calculate(
        step_returns=(0.01, -0.01),
        rewards=(0.01, -0.01),
    )

    network = ActorCriticNetwork(
        observation_size=3,
        action_size=3,
        hidden_sizes=(16, 16),
    )

    policy = PPOPolicy(
        network=network
    )

    evaluator = PPOEvaluator(
        environment=environment,
        policy=policy,
        metrics_calculator=RiskMetricsCalculator(),
    )

    ppo_result = evaluator.evaluate()

    comparison = RLComparisonBuilder.from_results(
        q_learning_metrics=q_metrics,
        q_learning_actions=(0, 1),
        ppo_result=ppo_result,
    )

    assert comparison.algorithms == (
        "q_learning",
        "ppo",
    )

    assert "q_learning" in (
        comparison.total_returns
    )

    assert "ppo" in (
        comparison.total_returns
    )

    assert "q_learning" in (
        comparison.volatilities
    )

    assert "ppo" in (
        comparison.volatilities
    )

    assert "q_learning" in (
        comparison.maximum_drawdowns
    )

    assert "ppo" in (
        comparison.maximum_drawdowns
    )


def test_comparison_runner(
    prices,
    experiment_config,
) -> None:
    runner = RLComparisonRunner(
        prices=prices,
        config=experiment_config,
        ppo_training_config=PPOTrainingConfig(
            rollout_steps=8,
            optimization_epochs=1,
            minibatch_size=4,
            learning_rate=1e-3,
            gradient_clip_norm=0.5,
        ),
    )

    result = runner.run()

    assert result.train_size == 16
    assert result.evaluation_size == 4
    assert result.seed == 42

    assert (
        result.comparison.algorithms
        == (
            "q_learning",
            "ppo",
        )
    )

    assert (
        result.comparison
        .q_learning
        .metrics
        .steps
        == 3
    )

    assert (
        result.comparison
        .ppo
        .metrics
        .steps
        == 3
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "rl-engine",
    }


def test_rl_comparison_api() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rl/compare",
        json={
            "prices": [
                100.0,
                101.0,
                99.0,
                102.0,
                103.0,
                101.0,
                104.0,
                105.0,
                103.0,
                106.0,
            ],
            "training_episodes": 2,
            "ppo_rollout_steps": 8,
            "ppo_optimization_epochs": 1,
            "ppo_minibatch_size": 4,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "q_learning" in data
    assert "ppo" in data

    assert data["train_size"] == 8
    assert data["evaluation_size"] == 2
    assert data["seed"] == 42

    for algorithm in (
        "q_learning",
        "ppo",
    ):
        assert "total_return" in data[algorithm]
        assert "volatility" in data[algorithm]
        assert "maximum_drawdown" in data[algorithm]
        assert "total_reward" in data[algorithm]
        assert "action_counts" in data[algorithm]


def test_rl_comparison_api_rejects_invalid_prices() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rl/compare",
        json={
            "prices": [
                100.0,
                101.0,
                0.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
            ],
        },
    )

    assert response.status_code == 422


def test_rl_comparison_api_requires_enough_prices() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rl/compare",
        json={
            "prices": [
                100.0,
                101.0,
                102.0,
            ],
        },
    )

    assert response.status_code == 422