from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from app.agent.networks import ActorCriticNetwork


@dataclass(frozen=True)
class PolicyEvaluation:
    

    action_log_probabilities: Tensor
    state_values: Tensor
    entropy: Tensor


class PPOPolicy:
    

    def __init__(
        self,
        network: ActorCriticNetwork,
    ) -> None:
        self._network = network

    @property
    def network(self) -> ActorCriticNetwork:
        return self._network

    def distribution(
        self,
        observations: Tensor,
    ) -> Categorical:
        

        action_logits = self._network.policy_logits(
            observations
        )

        return Categorical(
            logits=action_logits
        )

    def sample_action(
        self,
        observations: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
   

        distribution = self.distribution(
            observations
        )

        actions = distribution.sample()

        log_probabilities = distribution.log_prob(
            actions
        )

        state_values = self._network.state_values(
            observations
        )

        return (
            actions,
            log_probabilities,
            state_values,
        )

    def evaluate_actions(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> PolicyEvaluation:
       

        distribution = self.distribution(
            observations
        )

        log_probabilities = distribution.log_prob(
            actions
        )

        state_values = self._network.state_values(
            observations
        )

        entropy = distribution.entropy()

        return PolicyEvaluation(
            action_log_probabilities=log_probabilities,
            state_values=state_values,
            entropy=entropy,
        )