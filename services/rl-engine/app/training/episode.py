from __future__ import annotations

from dataclasses import dataclass

from app.agent.q_learning import QLearningAgent
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.environment.state import Action


@dataclass(frozen=True)
class EpisodeResult:
    """
    Metrics produced by one simulated RL episode.
    """

    total_reward: float
    steps: int
    positive_rewards: int
    negative_rewards: int
    final_position: float


class EpisodeRunner:
    """
    Executes one interaction episode between a Q-Learning
    agent and the simulated market environment.
    """

    def __init__(
        self,
        environment: ReinforcementLearningEnvironment,
        agent: QLearningAgent,
    ) -> None:
        self._environment = environment
        self._agent = agent

    def run_training_episode(self) -> EpisodeResult:
        """
        Run one episode while updating the Q-table.
        """

        state = self._environment.reset()

        total_reward = 0.0
        steps = 0
        positive_rewards = 0
        negative_rewards = 0

        while True:
            action = self._agent.select_action(state)

            transition = self._environment.step(
                action
            )

            self._agent.update(transition)

            total_reward += transition.reward

            if transition.reward > 0:
                positive_rewards += 1
            elif transition.reward < 0:
                negative_rewards += 1

            steps += 1

            state = transition.next_state

            if transition.done:
                break

        self._agent.decay_exploration()

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            positive_rewards=positive_rewards,
            negative_rewards=negative_rewards,
            final_position=state.position,
        )

    def run_hold_baseline(self) -> EpisodeResult:
        """
        Run a non-learning HOLD policy.

        The baseline always chooses HOLD and does not modify
        the Q-table or exploration rate.
        """

        state = self._environment.reset()

        total_reward = 0.0
        steps = 0
        positive_rewards = 0
        negative_rewards = 0

        while True:
            transition = self._environment.step(
                Action.HOLD
            )

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