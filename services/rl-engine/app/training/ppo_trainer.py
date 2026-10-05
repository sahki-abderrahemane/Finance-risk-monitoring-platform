from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.agent.advantage import (
    GeneralizedAdvantageEstimator,
)
from app.agent.advantage_normalization import (
    AdvantageNormalizer,
)
from app.agent.ppo_objective import (
    PPOObjective,
)
from app.agent.ppo_policy import PPOPolicy
from app.agent.trajectory import (
    ActorCriticTrajectory,
)
from app.environment.transition import (
    ReinforcementLearningEnvironment,
)
from app.training.ppo_rollout import (
    PPORolloutCollector,
)


@dataclass(frozen=True)
class PPOTrainingConfig:
    """
    Configuration for one PPO training process.
    """

    rollout_steps: int = 128
    optimization_epochs: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    gradient_clip_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.rollout_steps <= 0:
            raise ValueError(
                "rollout_steps must be greater than zero."
            )

        if self.optimization_epochs <= 0:
            raise ValueError(
                "optimization_epochs must be greater "
                "than zero."
            )

        if self.minibatch_size <= 0:
            raise ValueError(
                "minibatch_size must be greater than zero."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be greater "
                "than zero."
            )


@dataclass(frozen=True)
class PPOUpdateResult:
    """
    Metrics produced by one PPO optimization phase.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    updates: int


class PPOTrainer:
    """
    PPO training loop for Sentinel-AI.

    High-level process:

        1. collect rollout
        2. calculate GAE
        3. normalize advantages
        4. create minibatches
        5. optimize for multiple epochs
        6. collect a fresh rollout

    The trainer operates exclusively on the simulated
    reinforcement-learning environment.
    """

    def __init__(
        self,
        environment: ReinforcementLearningEnvironment,
        policy: PPOPolicy,
        advantage_estimator: (
            GeneralizedAdvantageEstimator
        ),
        objective: PPOObjective,
        config: PPOTrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._advantage_estimator = (
            advantage_estimator
        )
        self._objective = objective
        self._config = (
            config
            if config is not None
            else PPOTrainingConfig()
        )
        self._device = device or torch.device("cpu")

        self._optimizer = torch.optim.Adam(
            self._policy.network.parameters(),
            lr=self._config.learning_rate,
        )

        self._normalizer = AdvantageNormalizer()

        self._rollout_collector = (
            PPORolloutCollector(
                environment=self._environment,
                policy=self._policy,
                device=self._device,
            )
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def train_update(self) -> PPOUpdateResult:
        """
        Perform one complete PPO rollout + optimization cycle.
        """

        rollout = self._rollout_collector.collect(
            max_steps=self._config.rollout_steps
        )

        trajectory = rollout.trajectory

        (
            states,
            actions,
            old_log_probabilities,
            state_values,
            rewards,
            next_state_values,
            dones,
        ) = self._trajectory_tensors(
            trajectory,
            rollout,
        )

        advantages = (
            self._advantage_estimator.calculate(
                rewards=rewards.tolist(),
                state_values=state_values.tolist(),
                next_state_values=(
                    next_state_values.tolist()
                ),
                dones=dones.tolist(),
            )
        )

        advantage_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self._device,
        )

        return_targets = (
            advantage_tensor
            + state_values
        )

        normalized_advantages = (
            self._normalizer.normalize(
                advantage_tensor
            )
        )

        return self._optimize(
            states=states,
            actions=actions,
            old_log_probabilities=(
                old_log_probabilities
            ),
            advantages=normalized_advantages,
            return_targets=return_targets,
        )

    def _optimize(
        self,
        states: Tensor,
        actions: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        return_targets: Tensor,
    ) -> PPOUpdateResult:
        sample_count = states.shape[0]

        if sample_count == 0:
            raise ValueError(
                "Cannot optimize an empty rollout."
            )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        updates = 0

        for _ in range(
            self._config.optimization_epochs
        ):
            permutation = torch.randperm(
                sample_count,
                device=self._device,
            )

            for start in range(
                0,
                sample_count,
                self._config.minibatch_size,
            ):
                indices = permutation[
                    start:
                    start
                    + self._config.minibatch_size
                ]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = (
                    old_log_probabilities[indices]
                )
                batch_advantages = (
                    advantages[indices]
                )
                batch_return_targets = (
                    return_targets[indices]
                )

                evaluation = (
                    self._policy.evaluate_actions(
                        observations=batch_states,
                        actions=batch_actions,
                    )
                )

                result = self._objective.calculate(
                    new_log_probabilities=(
                        evaluation
                        .action_log_probabilities
                    ),
                    old_log_probabilities=(
                        batch_old_log_probs
                    ),
                    advantages=batch_advantages,
                    state_values=(
                        evaluation.state_values
                    ),
                    return_targets=(
                        batch_return_targets
                    ),
                    entropy=evaluation.entropy,
                )

                self._optimizer.zero_grad(
                    set_to_none=True
                )

                result.total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self._policy.network.parameters(),
                    self._config.gradient_clip_norm,
                )

                self._optimizer.step()

                total_policy_loss += float(
                    result.policy_loss.detach()
                )

                total_value_loss += float(
                    result.value_loss.detach()
                )

                total_entropy += float(
                    result.entropy_bonus.detach()
                )

                total_loss += float(
                    result.total_loss.detach()
                )

                updates += 1

        return PPOUpdateResult(
            policy_loss=(
                total_policy_loss / updates
            ),
            value_loss=(
                total_value_loss / updates
            ),
            entropy=(
                total_entropy / updates
            ),
            total_loss=(
                total_loss / updates
            ),
            updates=updates,
        )

    def _trajectory_tensors(
        self,
        trajectory: ActorCriticTrajectory,
        rollout,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        states = torch.tensor(
            trajectory.states,
            dtype=torch.float32,
            device=self._device,
        )

        actions = torch.tensor(
            trajectory.actions,
            dtype=torch.long,
            device=self._device,
        )

        old_log_probabilities = torch.tensor(
            trajectory.log_probabilities,
            dtype=torch.float32,
            device=self._device,
        )

        state_values = torch.tensor(
            trajectory.state_values,
            dtype=torch.float32,
            device=self._device,
        )

        rewards = torch.tensor(
            trajectory.rewards,
            dtype=torch.float32,
            device=self._device,
        )

        next_state_values = torch.empty_like(
            state_values
        )

        for index, step in enumerate(
            trajectory.steps
        ):
            if step.done:
                next_state_values[index] = 0.0
            elif (
                index + 1
                < trajectory.length
            ):
                next_state_values[index] = (
                    state_values[index + 1]
                )
            else:
                next_state_values[index] = (
                    self._rollout_collector
                    .final_bootstrap_value(
                        rollout
                    )
                )

        dones = torch.tensor(
            trajectory.dones,
            dtype=torch.bool,
            device=self._device,
        )

        return (
            states,
            actions,
            old_log_probabilities,
            state_values,
            rewards,
            next_state_values,
            dones,
        )