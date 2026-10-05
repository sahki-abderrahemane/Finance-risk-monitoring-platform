from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PPOObjectiveResult:
    """
    Components of the PPO optimization objective.

    """

    policy_loss: Tensor
    value_loss: Tensor
    entropy_bonus: Tensor
    total_loss: Tensor
    probability_ratio: Tensor


class PPOObjective:
    

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
    ) -> None:
        if not 0.0 < clip_epsilon < 1.0:
            raise ValueError(
                "clip_epsilon must be in (0, 1)."
            )

        if value_loss_coefficient < 0:
            raise ValueError(
                "value_loss_coefficient cannot be negative."
            )

        if entropy_coefficient < 0:
            raise ValueError(
                "entropy_coefficient cannot be negative."
            )

        self._clip_epsilon = float(
            clip_epsilon
        )
        self._value_loss_coefficient = float(
            value_loss_coefficient
        )
        self._entropy_coefficient = float(
            entropy_coefficient
        )

    @property
    def clip_epsilon(self) -> float:
        return self._clip_epsilon

    @property
    def value_loss_coefficient(self) -> float:
        return self._value_loss_coefficient

    @property
    def entropy_coefficient(self) -> float:
        return self._entropy_coefficient

    def calculate(
        self,
        new_log_probabilities: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        state_values: Tensor,
        return_targets: Tensor,
        entropy: Tensor,
    ) -> PPOObjectiveResult:
        """
        Calculate the complete PPO loss.

        Args:
            new_log_probabilities:
                log pi_new(a|s)

            old_log_probabilities:
                log pi_old(a|s)

            advantages:
                GAE advantage estimates

            state_values:
                critic predictions V(s)

            return_targets:
                target values used to train the critic

            entropy:
                policy entropy for exploration regularization.
        """

        self._validate_shapes(
            new_log_probabilities,
            old_log_probabilities,
            advantages,
            state_values,
            return_targets,
            entropy,
        )

        probability_ratio = torch.exp(
            new_log_probabilities
            - old_log_probabilities
        )

        clipped_ratio = torch.clamp(
            probability_ratio,
            1.0 - self._clip_epsilon,
            1.0 + self._clip_epsilon,
        )

        unclipped_objective = (
            probability_ratio
            * advantages
        )

        clipped_objective = (
            clipped_ratio
            * advantages
        )

        surrogate_objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        policy_loss = -surrogate_objective.mean()

        value_error = (
            state_values
            - return_targets
        )

        value_loss = (
            0.5
            * value_error.pow(2).mean()
        )

        entropy_bonus = entropy.mean()

        total_loss = (
            policy_loss
            + self._value_loss_coefficient
            * value_loss
            - self._entropy_coefficient
            * entropy_bonus
        )

        return PPOObjectiveResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_bonus=entropy_bonus,
            total_loss=total_loss,
            probability_ratio=probability_ratio,
        )

    @staticmethod
    def _validate_shapes(
        new_log_probabilities: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        state_values: Tensor,
        return_targets: Tensor,
        entropy: Tensor,
    ) -> None:
        expected_shape = new_log_probabilities.shape

        tensors = {
            "old_log_probabilities": old_log_probabilities,
            "advantages": advantages,
            "state_values": state_values,
            "return_targets": return_targets,
            "entropy": entropy,
        }

        for name, tensor in tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape "
                    f"{expected_shape}, "
                    f"received {tensor.shape}."
                )