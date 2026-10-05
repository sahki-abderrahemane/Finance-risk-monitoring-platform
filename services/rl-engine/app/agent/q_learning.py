from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from app.environment.state import Action
from app.environment.transition import EnvironmentTransition
from app.environment.state import MarketState
from app.agent.state_discretizer import (
    DiscreteState,
    StateDiscretizer,
)


@dataclass(frozen=True)
class QLearningConfig:
    """
    Hyperparameters controlling the Q-Learning algorithm.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    minimum_exploration_rate: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                "learning_rate must be in (0, 1]."
            )

        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0, 1]."
            )

        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError(
                "exploration_rate must be in [0, 1]."
            )

        if not 0.0 < self.exploration_decay <= 1.0:
            raise ValueError(
                "exploration_decay must be in (0, 1]."
            )

        if not 0.0 <= self.minimum_exploration_rate <= 1.0:
            raise ValueError(
                "minimum_exploration_rate must be in [0, 1]."
            )


class QLearningAgent:
    """
    Tabular Q-Learning agent for Sentinel-AI simulation.

    The agent learns action values from historical or synthetic
    market simulations. It has no real-world trading capability.
    """

    def __init__(
        self,
        discretizer: StateDiscretizer,
        config: QLearningConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self._discretizer = discretizer
        self._config = config or QLearningConfig()

        self._q_table: dict[
            DiscreteState,
            dict[int, float],
        ] = {}

        self._rng = random.Random(seed)

        self._exploration_rate = (
            self._config.exploration_rate
        )

    @property
    def exploration_rate(self) -> float:
        return self._exploration_rate

    @property
    def q_table(
        self,
    ) -> Mapping[
        DiscreteState,
        Mapping[int, float],
    ]:
        return self._q_table

    def select_action(
        self,
        state: MarketState,
    ) -> int:
        """
        Select an action using epsilon-greedy exploration.

        With probability epsilon:
            choose a random action.

        Otherwise:
            choose the action with the highest Q-value.
        """

        discrete_state = self._discretizer.transform(
            state
        )

        self._ensure_state(discrete_state)

        if self._rng.random() < self._exploration_rate:
            return self._rng.choice(
                Action.values()
            )

        return self._best_action(discrete_state)

    def update(
        self,
        transition: EnvironmentTransition,
    ) -> float:
        """
        Apply the Q-Learning Bellman update.

        Returns the temporal-difference error.
        """

        state = self._discretizer.transform(
            transition.state
        )

        next_state = self._discretizer.transform(
            transition.next_state
        )

        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self._q_table[state][
            transition.action
        ]

        if transition.done:
            future_q = 0.0
        else:
            future_q = max(
                self._q_table[next_state].values()
            )

        target = (
            transition.reward
            + self._config.discount_factor
            * future_q
        )

        td_error = target - current_q

        updated_q = (
            current_q
            + self._config.learning_rate
            * td_error
        )

        self._q_table[state][
            transition.action
        ] = updated_q

        return td_error

    def decay_exploration(self) -> None:
        """
        Reduce epsilon after an episode.

        Exploration never falls below the configured minimum.
        """

        self._exploration_rate = max(
            self._config.minimum_exploration_rate,
            self._exploration_rate
            * self._config.exploration_decay,
        )

    def _ensure_state(
        self,
        state: DiscreteState,
    ) -> None:
        if state not in self._q_table:
            self._q_table[state] = {
                action: 0.0
                for action in Action.values()
            }

    def _best_action(
        self,
        state: DiscreteState,
    ) -> int:
        action_values = self._q_table[state]

        maximum_value = max(
            action_values.values()
        )

        best_actions = [
            action
            for action, value
            in action_values.items()
            if value == maximum_value
        ]

        return self._rng.choice(
            best_actions
        )