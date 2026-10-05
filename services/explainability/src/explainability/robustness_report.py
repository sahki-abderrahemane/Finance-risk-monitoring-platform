from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import pandas as pd

from .adversarial import RobustnessResult


class RobustnessReport:
    
    @staticmethod
    def to_dataframe(
        results: Sequence[RobustnessResult],
    ) -> pd.DataFrame:
        """
        Convert robustness results into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per perturbation budget.
        """
        if not results:
            raise ValueError(
                "results cannot be empty."
            )

        records = [
            asdict(result)
            for result in results
        ]

        return pd.DataFrame(
            records,
            columns=[
                "sample_count",
                "changed_predictions",
                "flip_rate",
                "stability_rate",
                "mean_probability_change",
                "max_probability_change",
                "epsilon",
                "norm",
            ],
        ).sort_values(
            "epsilon",
            ascending=True,
            ignore_index=True,
        )

    @staticmethod
    def summarize(
        results: Sequence[RobustnessResult],
    ) -> dict[str, float | int | str]:
        """
        Produce a compact summary of the robustness experiment.

        The summary deliberately reports measurements rather than assigning
        qualitative labels such as "safe", "unsafe", "robust", or "weak".
        """
        if not results:
            raise ValueError(
                "results cannot be empty."
            )

        highest_budget = max(
            results,
            key=lambda result: result.epsilon,
        )

        lowest_budget = min(
            results,
            key=lambda result: result.epsilon,
        )

        return {
            "budget_count": len(results),
            "minimum_epsilon": lowest_budget.epsilon,
            "maximum_epsilon": highest_budget.epsilon,
            "minimum_flip_rate": lowest_budget.flip_rate,
            "maximum_flip_rate": highest_budget.flip_rate,
            "minimum_stability_rate": lowest_budget.stability_rate,
            "maximum_stability_rate": highest_budget.stability_rate,
            "maximum_mean_probability_change": max(
                result.mean_probability_change
                for result in results
            ),
            "maximum_probability_change": max(
                result.max_probability_change
                for result in results
            ),
            "norm": highest_budget.norm,
        }