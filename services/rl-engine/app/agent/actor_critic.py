from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ActorCriticConfig:
    

    observation_size: int
    action_size: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    shared_encoder: bool = True

    def __post_init__(self) -> None:
        if self.observation_size <= 0:
            raise ValueError(
                "observation_size must be greater than zero."
            )

        if self.action_size <= 0:
            raise ValueError(
                "action_size must be greater than zero."
            )

        if not self.hidden_sizes:
            raise ValueError(
                "hidden_sizes cannot be empty."
            )

        if any(
            size <= 0
            for size in self.hidden_sizes
        ):
            raise ValueError(
                "All hidden layer sizes must be greater "
                "than zero."
            )


@dataclass(frozen=True)
class ActorOutput:
   
    action_logits: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.action_logits:
            raise ValueError(
                "action_logits cannot be empty."
            )


@dataclass(frozen=True)
class CriticOutput:
    

    state_value: float


@dataclass(frozen=True)
class ActorCriticOutput:
   

    actor: ActorOutput
    critic: CriticOutput

    @property
    def action_logits(
        self,
    ) -> tuple[float, ...]:
        return self.actor.action_logits

    @property
    def state_value(self) -> float:
        return self.critic.state_value


class ActorCriticArchitecture:
  

    def __init__(
        self,
        config: ActorCriticConfig,
    ) -> None:
        self._config = config

    @property
    def config(self) -> ActorCriticConfig:
        return self._config

    @property
    def observation_size(self) -> int:
        return self._config.observation_size

    @property
    def action_size(self) -> int:
        return self._config.action_size

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return self._config.hidden_sizes

    @property
    def shared_encoder(self) -> bool:
        return self._config.shared_encoder

    def validate_observation(
        self,
        observation: Sequence[float],
    ) -> tuple[float, ...]:
        """
        Validate and normalize an observation before it reaches
        the future actor/critic networks.
        """

        normalized = tuple(
            float(value)
            for value in observation
        )

        if len(normalized) != self.observation_size:
            raise ValueError(
                "Observation size mismatch: "
                f"expected {self.observation_size}, "
                f"received {len(normalized)}."
            )

        return normalized

    def validate_action_logits(
        self,
        action_logits: Sequence[float],
    ) -> tuple[float, ...]:
        """
        Validate the actor output dimensionality.
        """

        normalized = tuple(
            float(value)
            for value in action_logits
        )

        if len(normalized) != self.action_size:
            raise ValueError(
                "Action-logit size mismatch: "
                f"expected {self.action_size}, "
                f"received {len(normalized)}."
            )

        return normalized