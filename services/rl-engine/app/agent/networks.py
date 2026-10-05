from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class SharedFeatureEncoder(nn.Module):
    

    def __init__(
        self,
        observation_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError(
                "observation_size must be greater than zero."
            )

        normalized_hidden_sizes = tuple(
            int(size)
            for size in hidden_sizes
        )

        if not normalized_hidden_sizes:
            raise ValueError(
                "hidden_sizes cannot be empty."
            )

        if any(
            size <= 0
            for size in normalized_hidden_sizes
        ):
            raise ValueError(
                "All hidden sizes must be greater than zero."
            )

        layers: list[nn.Module] = []

        input_size = observation_size

        for hidden_size in normalized_hidden_sizes:
            layers.append(
                nn.Linear(
                    input_size,
                    hidden_size,
                )
            )
            layers.append(nn.Tanh())

            input_size = hidden_size

        self.network = nn.Sequential(*layers)

        self.output_size = input_size

    def forward(
        self,
        observations: Tensor,
    ) -> Tensor:
        if observations.ndim != 2:
            raise ValueError(
                "observations must have shape "
                "(batch_size, observation_size)."
            )

        return self.network(observations)


class ActorCriticNetwork(nn.Module):
    """
    Shared actor-critic neural network for Sentinel-AI PPO.

    Architecture:

        observation
             |
             v
        shared encoder
          /       \
         v         v
      actor      critic
       head       head
         |         |
       logits     V(s)

    The actor produces action logits.

    The critic produces a scalar state-value estimate.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()

        if action_size <= 0:
            raise ValueError(
                "action_size must be greater than zero."
            )

        self.encoder = SharedFeatureEncoder(
            observation_size=observation_size,
            hidden_sizes=hidden_sizes,
        )

        feature_size = self.encoder.output_size

        self.actor_head = nn.Linear(
            feature_size,
            action_size,
        )

        self.critic_head = nn.Linear(
            feature_size,
            1,
        )

        self.observation_size = observation_size
        self.action_size = action_size

    def forward(
        self,
        observations: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return actor logits and critic values.

        Returns:
            action_logits:
                Shape (batch_size, action_size)

            state_values:
                Shape (batch_size,)
        """

        features = self.encoder(observations)

        action_logits = self.actor_head(
            features
        )

        state_values = self.critic_head(
            features
        ).squeeze(-1)

        return action_logits, state_values

    def policy_logits(
        self,
        observations: Tensor,
    ) -> Tensor:
        """
        Return only the actor output.
        """

        features = self.encoder(observations)

        return self.actor_head(features)

    def state_values(
        self,
        observations: Tensor,
    ) -> Tensor:
        """
        Return only the critic output.
        """

        features = self.encoder(observations)

        return self.critic_head(
            features
        ).squeeze(-1)