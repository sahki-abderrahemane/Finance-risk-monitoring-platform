from __future__ import annotations

from typing import Sequence


class GeneralizedAdvantageEstimator:
   

    def __init__(
        self,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0, 1]."
            )

        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must be in [0, 1]."
            )

        self._discount_factor = float(
            discount_factor
        )
        self._gae_lambda = float(
            gae_lambda
        )

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def gae_lambda(self) -> float:
        return self._gae_lambda

    def calculate(
        self,
        rewards: Sequence[float],
        state_values: Sequence[float],
        next_state_values: Sequence[float],
        dones: Sequence[bool],
    ) -> tuple[float, ...]:
        """
        Calculate one advantage estimate per transition.

        All input sequences must have identical lengths.

        For a non-terminal transition:

            delta_t =
                reward_t
                + gamma * V(s_{t+1})
                - V(s_t)

        For a terminal transition:

            delta_t =
                reward_t
                - V(s_t)

        The resulting advantages are calculated backward through
        the trajectory.
        """

        normalized_rewards = tuple(
            float(value)
            for value in rewards
        )

        normalized_values = tuple(
            float(value)
            for value in state_values
        )

        normalized_next_values = tuple(
            float(value)
            for value in next_state_values
        )

        normalized_dones = tuple(
            bool(value)
            for value in dones
        )

        self._validate_lengths(
            normalized_rewards,
            normalized_values,
            normalized_next_values,
            normalized_dones,
        )

        if not normalized_rewards:
            return ()

        advantages = [0.0] * len(
            normalized_rewards
        )

        running_advantage = 0.0

        for index in range(
            len(normalized_rewards) - 1,
            -1,
            -1,
        ):
            if normalized_dones[index]:
                bootstrap_value = 0.0
                continuation = 0.0
            else:
                bootstrap_value = (
                    normalized_next_values[index]
                )
                continuation = 1.0

            delta = (
                normalized_rewards[index]
                + self._discount_factor
                * bootstrap_value
                - normalized_values[index]
            )

            running_advantage = (
                delta
                + self._discount_factor
                * self._gae_lambda
                * continuation
                * running_advantage
            )

            advantages[index] = running_advantage

        return tuple(advantages)

    @staticmethod
    def _validate_lengths(
        rewards: Sequence[float],
        state_values: Sequence[float],
        next_state_values: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        expected_length = len(rewards)

        if len(state_values) != expected_length:
            raise ValueError(
                "state_values must have the same length "
                "as rewards."
            )

        if len(next_state_values) != expected_length:
            raise ValueError(
                "next_state_values must have the same length "
                "as rewards."
            )

        if len(dones) != expected_length:
            raise ValueError(
                "dones must have the same length "
                "as rewards."
            )