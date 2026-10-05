from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from app.agent.advantage import (
    GeneralizedAdvantageEstimator,
)
from app.agent.networks import ActorCriticNetwork
from app.agent.ppo_objective import PPOObjective
from app.agent.ppo_policy import PPOPolicy
from app.agent.q_learning import QLearningAgent
from app.agent.state_discretizer import StateDiscretizer
from app.environment.market import (
    SimulatedMarketEnvironment,
)
from app.environment.portfolio import (
    SimulatedPortfolio,
)
from app.environment.reward import (
    RiskAdjustedReward,
)
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.evaluation.comparison import (
    RLAlgorithmComparison,
    RLComparisonBuilder,
)
from app.evaluation.metrics import (
    EvaluationMetrics,
    RiskMetricsCalculator,
)
from app.evaluation.ppo import PPOEvaluator
from app.experiment.config import ExperimentConfig
from app.training.episode import EpisodeRunner
from app.training.ppo_trainer import (
    PPOTrainer,
    PPOTrainingConfig,
)
from app.training.trainer import QLearningTrainer


@dataclass(frozen=True)
class ComparisonExperimentResult:
    """
    Complete result of a Q-Learning vs PPO experiment.

    Training and evaluation data are separated chronologically.
    Both algorithms are evaluated on the same held-out data.
    """

    comparison: RLAlgorithmComparison
    train_size: int
    evaluation_size: int
    seed: int


class RLComparisonRunner:
    """
    Runs Q-Learning and PPO under the same simulated conditions.

    Experimental controls:

    - same input price sequence
    - same chronological train/evaluation split
    - same action space
    - same portfolio configuration
    - same reward configuration
    - same evaluation metrics

    The experiment is strictly research/simulation oriented.
    No live market, brokerage, or execution system is used.
    """

    def __init__(
        self,
        prices: Sequence[float],
        config: ExperimentConfig | None = None,
        ppo_training_config: PPOTrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        if len(prices) < 8:
            raise ValueError(
                "At least eight prices are required "
                "for a comparison experiment."
            )

        normalized_prices = tuple(
            float(price)
            for price in prices
        )

        if any(
            price <= 0
            for price in normalized_prices
        ):
            raise ValueError(
                "All prices must be greater than zero."
            )

        self._prices = normalized_prices

        self._config = (
            config
            if config is not None
            else ExperimentConfig()
        )

        self._ppo_training_config = (
            ppo_training_config
            if ppo_training_config is not None
            else PPOTrainingConfig()
        )

        self._device = (
            device
            if device is not None
            else torch.device("cpu")
        )

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @property
    def ppo_training_config(
        self,
    ) -> PPOTrainingConfig:
        return self._ppo_training_config

    def run(self) -> ComparisonExperimentResult:
        """
        Run the complete Q-Learning vs PPO experiment.
        """

        train_prices, evaluation_prices = (
            self._split_prices()
        )

        q_learning_agent = (
            self._train_q_learning(
                train_prices
            )
        )

        (
            q_learning_metrics,
            q_learning_actions,
        ) = self._evaluate_q_learning(
            agent=q_learning_agent,
            prices=evaluation_prices,
        )

        ppo_policy = self._train_ppo(
            train_prices
        )

        ppo_result = self._evaluate_ppo(
            policy=ppo_policy,
            prices=evaluation_prices,
        )

        comparison = (
            RLComparisonBuilder.from_results(
                q_learning_metrics=(
                    q_learning_metrics
                ),
                q_learning_actions=(
                    q_learning_actions
                ),
                ppo_result=ppo_result,
            )
        )

        return ComparisonExperimentResult(
            comparison=comparison,
            train_size=len(train_prices),
            evaluation_size=len(evaluation_prices),
            seed=self._config.seed,
        )

    def _split_prices(
        self,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """
        Perform an 80/20 chronological split.

        The first 80% is used exclusively for training.

        The final 20% is held out and used exclusively for
        evaluation.

        No random shuffling is performed because this is
        time-series simulation data.
        """

        split_index = int(
            len(self._prices) * 0.8
        )

        if split_index < 2:
            raise ValueError(
                "Training split must contain at least "
                "two prices."
            )

        if (
            len(self._prices)
            - split_index
            < 2
        ):
            raise ValueError(
                "Evaluation split must contain at least "
                "two prices."
            )

        return (
            self._prices[:split_index],
            self._prices[split_index:],
        )

    def _train_q_learning(
        self,
        prices: Sequence[float],
    ) -> QLearningAgent:
        """
        Train Q-Learning exclusively on the training prices.
        """

        environment = self._build_environment(
            prices
        )

        agent = QLearningAgent(
            discretizer=StateDiscretizer(),
            config=self._config.q_learning,
            seed=self._config.seed,
        )

        runner = EpisodeRunner(
            environment=environment,
            agent=agent,
        )

        trainer = QLearningTrainer(
            runner=runner,
            agent=agent,
        )

        trainer.train(
            episodes=self._config.training.episodes
        )

        return agent

    def _evaluate_q_learning(
        self,
        agent: QLearningAgent,
        prices: Sequence[float],
    ) -> tuple[
        EvaluationMetrics,
        tuple[int, ...],
    ]:
        """
        Evaluate the trained Q-Learning agent on held-out data.

        Exploration is disabled during evaluation.
        """

        environment = self._build_environment(
            prices
        )

        state = environment.reset()

        metrics_calculator = (
            RiskMetricsCalculator()
        )

        step_returns: list[float] = []
        rewards: list[float] = []
        actions: list[int] = []

        original_exploration_rate = (
            agent.exploration_rate
        )

        agent._exploration_rate = 0.0

        try:
            while True:
                action = agent.select_action(
                    state
                )

                transition = environment.step(
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
            agent._exploration_rate = (
                original_exploration_rate
            )

        metrics = (
            metrics_calculator.calculate(
                step_returns=step_returns,
                rewards=rewards,
            )
        )

        return metrics, tuple(actions)

    def _train_ppo(
        self,
        prices: Sequence[float],
    ) -> PPOPolicy:
        """
        Train PPO exclusively on the training prices.
        """

        environment = self._build_environment(
            prices
        )

        network = ActorCriticNetwork(
            observation_size=3,
            action_size=3,
            hidden_sizes=(64, 64),
        )

        network.to(self._device)

        policy = PPOPolicy(
            network=network
        )

        estimator = (
            GeneralizedAdvantageEstimator(
                discount_factor=(
                    self._config
                    .q_learning
                    .discount_factor
                ),
                gae_lambda=0.95,
            )
        )

        objective = PPOObjective(
            clip_epsilon=0.2,
            value_loss_coefficient=0.5,
            entropy_coefficient=0.01,
        )

        trainer = PPOTrainer(
            environment=environment,
            policy=policy,
            advantage_estimator=estimator,
            objective=objective,
            config=self._ppo_training_config,
            device=self._device,
        )

        torch.manual_seed(
            self._config.seed
        )

        for _ in range(
            self._config.training.episodes
        ):
            trainer.train_update()

        return policy

    def _evaluate_ppo(
        self,
        policy: PPOPolicy,
        prices: Sequence[float],
    ):
        """
        Evaluate PPO on the same held-out data used for
        Q-Learning evaluation.
        """

        environment = self._build_environment(
            prices
        )

        evaluator = PPOEvaluator(
            environment=environment,
            policy=policy,
            metrics_calculator=(
                RiskMetricsCalculator()
            ),
            device=self._device,
        )

        return evaluator.evaluate(
            deterministic=True
        )

    def _build_environment(
        self,
        prices: Sequence[float],
    ) -> ReinforcementLearningEnvironment:
        """
        Construct an independent simulation environment.

        Separate environments are created for training and
        evaluation so their internal state cannot leak between
        experiments.
        """

        market = SimulatedMarketEnvironment(
            prices=prices
        )

        portfolio = SimulatedPortfolio(
            initial_position=(
                self._config
                .environment
                .initial_position
            ),
            min_position=(
                self._config
                .environment
                .min_position
            ),
            max_position=(
                self._config
                .environment
                .max_position
            ),
        )

        reward_function = RiskAdjustedReward(
            risk_penalty_coefficient=(
                self._config
                .reward
                .risk_penalty_coefficient
            ),
            transaction_cost_coefficient=(
                self._config
                .reward
                .transaction_cost_coefficient
            ),
            drawdown_penalty_coefficient=(
                self._config
                .reward
                .drawdown_penalty_coefficient
            ),
        )

        return ReinforcementLearningEnvironment(
            market=market,
            portfolio=portfolio,
            reward_function=reward_function,
        )
