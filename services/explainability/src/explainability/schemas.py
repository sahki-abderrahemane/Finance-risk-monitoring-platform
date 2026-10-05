from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class FeatureContribution:
    feature_name: str
    feature_value: float
    shap_value: float

@dataclass(frozen=True)
class ShapExplanation:
    

    base_value: float
    prediction: float
    contributions: tuple[FeatureContribution, ...]
    feature_names: tuple[str, ...]
    output_index: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty.")

        if len(self.contributions) != len(self.feature_names):
            raise ValueError(
                "Number of contributions must match number of feature names."
            )

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        if self.output_index is not None and self.output_index < 0:
            raise ValueError("output_index must be non-negative.")

    def top_contributions(
        self,
        limit: int = 10,
    ) -> tuple[FeatureContribution, ...]:
        
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        return tuple(
            sorted(
                self.contributions,
                key=lambda contribution: abs(contribution.shap_value),
                reverse=True,
            )[:limit]
        )

    @property
    def reconstruction_error(self) -> float:
       
        reconstructed = self.base_value + sum(
            contribution.shap_value
            for contribution in self.contributions
        )

        return abs(self.prediction - reconstructed)
