from __future__ import annotations

import torch
from torch import Tensor


class AdvantageNormalizer:
  

    def __init__(
        self,
        epsilon: float = 1e-8,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        self._epsilon = float(epsilon)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def normalize(
        self,
        advantages: Tensor,
    ) -> Tensor:
        if advantages.numel() == 0:
            raise ValueError(
                "advantages cannot be empty."
            )

        mean = advantages.mean()

        standard_deviation = advantages.std(
            unbiased=False
        )

        return (
            advantages - mean
        ) / (
            standard_deviation
            + self._epsilon
        )