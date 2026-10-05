from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.agent.q_learning import QLearningAgent
from app.agent.state_discretizer import StateDiscretizer
from app.environment.market import SimulatedMarketEnvironment
from app.environment.portfolio import SimulatedPortfolio
from app.environment.reward import RiskAdjustedReward
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.evaluation.metrics import RiskMetricsCalculator
from app.evaluation.policies import (
    BaselinePolicies,
    PolicyEvaluation,
    PolicyEvaluator,
)
from app.experiment.config import ExperimentConfig
from app.training.episode import EpisodeRunner
from app.training.trainer import (
    QLearningTrainer,
    TrainingHistory,
)


@dataclass(frozen=True)
class ExperimentResult:
    """
    Structured result of one complete Q-Learning experiment.
    """

    training: TrainingHistory
    evaluation: TrainingHistory
    hold_baseline_total_reward: float
    q_table_size: int
    seed: int


class ExperimentRunner:
   

    def __init__(
        self,
        prices: Sequence[float],
        config: ExperimentConfig | None = None,
    ) -> None:
        if len(prices) < 4:
            raise ValueError(
                "At least four prices are required "
                "for an experiment."
            )

        self._prices = tuple(
            float(price)
            for price in prices
        )

        if any(price <= 0 for price in self._prices):
            raise ValueError(
                "All prices must be greater than zero."
            )

        self._config = (
            config
            if config is not None
            else ExperimentConfig()
        )

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    def run(self) -> ExperimentResult:
        """
        Execute one complete training and evaluation experiment.
        """

        environment = self._build_environment()

        agent = self._build_agent()

        episode_runner = EpisodeRunner(
            environment=environment,
            agent=agent,
        )

        trainer = QLearningTrainer(
            runner=episode_runner,
            agent=agent,
        )

        training_history = trainer.train(
            episodes=self._config.training.episodes
        )

        evaluation_history = trainer.evaluate(
            episodes=self._config.evaluation.episodes
        )

        hold_baseline = (
            episode_runner.run_hold_baseline()
        )

        return ExperimentResult(
            training=training_history,
            evaluation=evaluation_history,
            hold_baseline_total_reward=(
                hold_baseline.total_reward
            ),
            q_table_size=len(agent.q_table),
            seed=self._config.seed,
        )

    def run_with_tracking(
        self,
        tracker,
    ) -> ExperimentResult:
        """
        Execute an experiment while recording the run through
        an injected MLflow-compatible tracker.

        The tracker is intentionally injected rather than created
        inside the runner, keeping experiment execution decoupled
        from the tracking implementation.
        """

        tracker.start_run(self._config)

        try:
            result = self.run()

            tracker.log_result(result)

            return result
        finally:
            tracker.end_run()

    def evaluate_policies(
        self,
    ) -> tuple[PolicyEvaluation, ...]:
        """
        Train the Q-Learning agent and evaluate it alongside the
        deterministic baseline policies.

        This method is intended for research comparison.
        """

        environment = self._build_environment()

        agent = self._build_agent()

        episode_runner = EpisodeRunner(
            environment=environment,
            agent=agent,
        )

        trainer = QLearningTrainer(
            runner=episode_runner,
            agent=agent,
        )

        trainer.train(
            episodes=self._config.training.episodes
        )

        evaluator = PolicyEvaluator(
            environment=environment,
            metrics_calculator=RiskMetricsCalculator(),
        )

        return (
            evaluator.evaluate(
                policy_name="hold",
                policy=BaselinePolicies.hold,
            ),
            evaluator.evaluate(
                policy_name="always_increase",
                policy=BaselinePolicies.always_increase,
            ),
            evaluator.evaluate(
                policy_name="always_reduce",
                policy=BaselinePolicies.always_reduce,
            ),
            evaluator.evaluate_q_learning(
                policy_name="q_learning",
                agent=agent,
            ),
        )

    def _build_agent(
        self,
    ) -> QLearningAgent:
        discretizer = StateDiscretizer()

        return QLearningAgent(
            discretizer=discretizer,
            config=self._config.q_learning,
            seed=self._config.seed,
        )

    def _build_environment(
        self,
    ) -> ReinforcementLearningEnvironment:
        market = SimulatedMarketEnvironment(
            prices=self._prices
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