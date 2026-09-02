from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd


RISK_SCORE_MIN = 0.0
RISK_SCORE_MAX = 100.0

REQUIRED_COLUMNS = {
    "timestamp",
    "volatility_20",
}

PREDICTED_PROBABILITY_COLUMNS = {
    "probability_down",
    "probability_up",
}


@dataclass(frozen=True)
class RiskScoringConfig:
    """
    Configuration for deterministic research risk scoring.

    The score combines:
      - model uncertainty
      - realized market volatility

    Both components are normalized independently before being combined.
    """

    model_uncertainty_weight: float = 0.5
    volatility_weight: float = 0.5

    volatility_lookback: int = 252

    low_threshold: float = 25.0
    moderate_threshold: float = 50.0
    high_threshold: float = 75.0

    def __post_init__(self) -> None:
        if self.model_uncertainty_weight < 0:
            raise ValueError("model_uncertainty_weight must be non-negative.")

        if self.volatility_weight < 0:
            raise ValueError("volatility_weight must be non-negative.")

        total_weight = (
            self.model_uncertainty_weight + self.volatility_weight
        )

        if not np.isclose(total_weight, 1.0):
            raise ValueError("Risk-scoring weights must sum to 1.0.")

        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be at least 2.")

        thresholds = (
            self.low_threshold,
            self.moderate_threshold,
            self.high_threshold,
        )

        if not (
            0.0 <= self.low_threshold
            < self.moderate_threshold
            < self.high_threshold
            <= 100.0
        ):
            raise ValueError(
                "Risk thresholds must satisfy "
                "0 <= low < moderate < high <= 100."
            )


@dataclass(frozen=True)
class RiskScoreReport:
    rows: int
    mean_score: float
    median_score: float
    minimum_score: float
    maximum_score: float
    low_count: int
    moderate_count: int
    high_count: int
    extreme_count: int


class ResearchRiskScorer:
    """
    Deterministic risk scoring engine.

    This class does NOT generate financial advice or trading actions.

    The resulting score represents a research-oriented estimate of
    market/model risk based on:
      1. GBM predictive uncertainty.
      2. Realized volatility.

    Model uncertainty:

        U = 1 - max(P(up), P(down))

    Volatility normalization:

        V_t = volatility_t / rolling_max(volatility)

    Final score:

        R_t = 100 * (w_u * U_t + w_v * V_t)

    where:

        w_u + w_v = 1
    """

    def __init__(
        self,
        config: RiskScoringConfig | None = None,
    ) -> None:
        self.config = config or RiskScoringConfig()

    def validate_input(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the input dataframe.
        """
        missing = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        missing_probabilities = (
            PREDICTED_PROBABILITY_COLUMNS - set(dataframe.columns)
        )

        if missing_probabilities:
            raise ValueError(
                "Missing probability columns: "
                f"{sorted(missing_probabilities)}"
            )

        if dataframe.empty:
            raise ValueError("Cannot score an empty dataframe.")

        if dataframe["timestamp"].isna().any():
            raise ValueError("timestamp contains null values.")

        probability_columns = [
            "probability_down",
            "probability_up",
        ]

        for column in probability_columns:
            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"{column} contains non-numeric or null values."
                )

            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(
                    f"{column} must contain probabilities between 0 and 1."
                )

        volatility = pd.to_numeric(
            dataframe["volatility_20"],
            errors="coerce",
        )

        if volatility.isna().any():
            raise ValueError(
                "volatility_20 contains non-numeric or null values."
            )

        if (volatility < 0.0).any():
            raise ValueError(
                "volatility_20 cannot contain negative values."
            )

    @staticmethod
    def calculate_model_uncertainty(
        probability_down: pd.Series,
        probability_up: pd.Series,
    ) -> pd.Series:
        """
        Calculate predictive uncertainty.

        A confident prediction such as:
            P(up)=0.95, P(down)=0.05

        produces:

            uncertainty = 0.05

        An uncertain prediction such as:
            P(up)=0.50, P(down)=0.50

        produces:

            uncertainty = 0.50
        """
        maximum_probability = pd.concat(
            [probability_down, probability_up],
            axis=1,
        ).max(axis=1)

        return 1.0 - maximum_probability

    def calculate_volatility_risk(
        self,
        volatility: pd.Series,
    ) -> pd.Series:
        """
        Normalize volatility using a historical rolling maximum.

        Values are clipped to [0, 1] so that the volatility component
        remains bounded.
        """
        rolling_max = (
            volatility
            .rolling(
                window=self.config.volatility_lookback,
                min_periods=1,
            )
            .max()
        )

        normalized = volatility.divide(
            rolling_max.replace(0.0, np.nan)
        )

        return (
            normalized
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    def classify_score(self, score: float) -> str:
        """
        Convert a numeric score into a qualitative risk band.
        """
        if not np.isfinite(score):
            raise ValueError("Risk score must be finite.")

        if score < self.config.low_threshold:
            return "LOW"

        if score < self.config.moderate_threshold:
            return "MODERATE"

        if score < self.config.high_threshold:
            return "HIGH"

        return "EXTREME"

    def score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for every row.

        Returns a new dataframe and never mutates the input.
        """
        self.validate_input(dataframe)

        result = dataframe.copy()

        probability_down = pd.to_numeric(
            result["probability_down"],
            errors="raise",
        )

        probability_up = pd.to_numeric(
            result["probability_up"],
            errors="raise",
        )

        volatility = pd.to_numeric(
            result["volatility_20"],
            errors="raise",
        )

        model_uncertainty = self.calculate_model_uncertainty(
            probability_down=probability_down,
            probability_up=probability_up,
        )

        volatility_risk = self.calculate_volatility_risk(
            volatility=volatility,
        )

        weighted_score = (
            self.config.model_uncertainty_weight
            * model_uncertainty
            + self.config.volatility_weight
            * volatility_risk
        )

        risk_score = (
            weighted_score
            * RISK_SCORE_MAX
        ).clip(
            RISK_SCORE_MIN,
            RISK_SCORE_MAX,
        )

        result["model_uncertainty"] = model_uncertainty
        result["volatility_risk"] = volatility_risk
        result["risk_score"] = risk_score
        result["risk_band"] = result["risk_score"].apply(
            self.classify_score
        )

        return result

    def summarize(
        self,
        scored_dataframe: pd.DataFrame,
    ) -> RiskScoreReport:
        """
        Produce aggregate statistics from scored data.
        """
        if scored_dataframe.empty:
            raise ValueError("Cannot summarize an empty dataframe.")

        if "risk_score" not in scored_dataframe.columns:
            raise ValueError(
                "Dataframe must contain risk_score before summarization."
            )

        scores = pd.to_numeric(
            scored_dataframe["risk_score"],
            errors="raise",
        )

        band_counts = (
            scored_dataframe["risk_band"]
            .value_counts()
            .to_dict()
        )

        return RiskScoreReport(
            rows=len(scored_dataframe),
            mean_score=float(scores.mean()),
            median_score=float(scores.median()),
            minimum_score=float(scores.min()),
            maximum_score=float(scores.max()),
            low_count=int(band_counts.get("LOW", 0)),
            moderate_count=int(band_counts.get("MODERATE", 0)),
            high_count=int(band_counts.get("HIGH", 0)),
            extreme_count=int(band_counts.get("EXTREME", 0)),
        )

    @staticmethod
    def save(
        scorer: "ResearchRiskScorer",
        path: str | Path,
    ) -> None:
        """
        Persist the scorer configuration.
        """
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(scorer, output_path)

    @staticmethod
    def load(
        path: str | Path,
    ) -> "ResearchRiskScorer":
        """
        Load a persisted scorer.
        """
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Risk scorer artifact does not exist: {input_path}"
            )

        scorer = joblib.load(input_path)

        if not isinstance(scorer, ResearchRiskScorer):
            raise TypeError(
                "Loaded artifact is not a ResearchRiskScorer."
            )

        return scorer