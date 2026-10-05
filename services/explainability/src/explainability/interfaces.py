from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from .schemas import ShapExplanation


class LocalExplainer(Protocol):
    
    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered model feature names."""
        ...

    @property
    def output_index(self) -> int | None:
      
        ...

    def explain(
        self,
        instance: np.ndarray | pd.DataFrame | Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> ShapExplanation:
       
        ...


def validate_feature_names(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
   
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    normalized = tuple(feature_names)

    if any(
        not isinstance(name, str) or not name.strip()
        for name in normalized
    ):
        raise ValueError(
            "Every feature name must be a non-empty string."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "feature_names must contain unique values."
        )

    return normalized