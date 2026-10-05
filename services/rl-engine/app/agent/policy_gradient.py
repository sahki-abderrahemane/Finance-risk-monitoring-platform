from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math


@dataclass(frozen=True)
class PolicyGradientSample:
    """
    One policy-gradient training sample.
    """

    action_probability: float
    return_value: float

    def __post_init__(self) -> None:
        if not 0.0 < self.action_probability <= 1.0:
            raise ValueError(
                "action_probability must be in (0, 1]."
            )


class PolicyGradientCalculator:
    """
    Mathematical policy-gradient helper.
 """

    @staticmethod
    def log_probability(
        action_probability: float,
    ) -> float:
        if not 0.0 < action_probability <= 1.0:
            raise ValueError(
                "action_probability must be in (0, 1]."
            )

        return math.log(action_probability)

    @classmethod
    def sample_objective(
        cls,
        sample: PolicyGradientSample,
    ) -> float:
        """
        Calculate:

            log π(a|s) * G

        A larger value represents a stronger positive contribution
        to the gradient-ascent objective.
        """

        return (
            cls.log_probability(
                sample.action_probability
            )
            * sample.return_value
        )

    @classmethod
    def policy_loss(
        cls,
        samples: Sequence[PolicyGradientSample],
    ) -> float:
        """
        Calculate the negative mean policy-gradient objective.

        Neural-network frameworks minimize losses, so:

            loss =
                -mean(log π(a|s) * G)
        """

        if not samples:
            raise ValueError(
                "At least one sample is required."
            )

        objective = sum(
            cls.sample_objective(sample)
            for sample in samples
        ) / len(samples)

        return -objective