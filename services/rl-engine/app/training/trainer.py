from __future__ import annotations

from dataclasses import dataclass

from app.agent.q_learning import QLearningAgent
from app.training.episode import (
    EpisodeResult,
    EpisodeRunner,
)


@dataclass(frozen=True)
class TrainingHistory:
    """
    Collection of metrics from the training process.
    """

    episodes: tuple[EpisodeResult, ...]

    @property
    def total_rewards(self) -> tuple[float, ...]:
        return tuple(
            episode.total_reward
            for episode in self.episodes
        )

    @property
    def average_reward(self) -> float:
        if not self.episodes:
            return 0.0

        return sum(
            episode.total_reward
            for episode in self.episodes
        ) / len(self.episodes)


class QLearningTrainer:
    """
    Coordinates repeated Q-Learning training episodes.
    """

    def __init__(
        self,
        runner: EpisodeRunner,
        agent: QLearningAgent,
    ) -> None:
        self._runner = runner
        self._agent = agent

    def train(
        self,
        episodes: int,
    ) -> TrainingHistory:
        """
        Train the agent for the requested number of episodes.
        """

        if episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )

        history: list[EpisodeResult] = []

        for _ in range(episodes):
            result = (
                self._runner.run_training_episode()
            )

            history.append(result)

        return TrainingHistory(
            episodes=tuple(history)
        )

    def evaluate(
        self,
        episodes: int,
    ) -> TrainingHistory:
        """
        Evaluate the learned policy using greedy action
        selection rather than exploration.

        The agent's Q-table is not modified.
        """

        if episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )

        original_exploration_rate = (
            self._agent.exploration_rate
        )

        self._set_exploration_rate(0.0)

        history: list[EpisodeResult] = []

        try:
            for _ in range(episodes):
                result = (
                    self._run_greedy_episode()
                )

                history.append(result)

        finally:
            self._set_exploration_rate(
                original_exploration_rate
            )

        return TrainingHistory(
            episodes=tuple(history)
        )

    def _run_greedy_episode(
        self,
    ) -> EpisodeResult:
        """
        Run an evaluation episode with epsilon = 0.

        The existing EpisodeRunner training method cannot be
        reused here because it intentionally updates the Q-table.
        """

        environment = self._runner._environment

        state = environment.reset()

        total_reward = 0.0
        steps = 0
        positive_rewards = 0
        negative_rewards = 0

        while True:
            action = self._agent.select_action(state)

            transition = environment.step(action)

            total_reward += transition.reward

            if transition.reward > 0:
                positive_rewards += 1
            elif transition.reward < 0:
                negative_rewards += 1

            steps += 1
            state = transition.next_state

            if transition.done:
                break

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            positive_rewards=positive_rewards,
            negative_rewards=negative_rewards,
            final_position=state.position,
        )

    def _set_exploration_rate(
        self,
        value: float,
    ) -> None:
        """
        Temporarily override exploration for evaluation.

        The QLearningAgent intentionally exposes exploration
        through a read-only property, so the trainer needs a
        controlled internal override for evaluation.
        """

        self._agent._exploration_rate = value