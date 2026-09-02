from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.risk_scoring import (
    ResearchRiskScorer,
    RiskScoringConfig,
)


def make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="D",
            ),
            "volatility_20": [
                0.01,
                0.02,
                0.03,
                0.015,
                0.025,
            ],
            "probability_down": [
                0.05,
                0.20,
                0.50,
                0.10,
                0.40,
            ],
            "probability_up": [
                0.95,
                0.80,
                0.50,
                0.90,
                0.60,
            ],
        }
    )


def test_model_uncertainty() -> None:
    scorer = ResearchRiskScorer()

    result = scorer.calculate_model_uncertainty(
        probability_down=pd.Series([0.05, 0.50]),
        probability_up=pd.Series([0.95, 0.50]),
    )

    assert np.allclose(
        result.to_numpy(),
        np.array([0.05, 0.50]),
    )


def test_volatility_risk_is_bounded() -> None:
    scorer = ResearchRiskScorer()

    volatility = pd.Series(
        [0.01, 0.02, 0.03, 0.015]
    )

    result = scorer.calculate_volatility_risk(volatility)

    assert (result >= 0.0).all()
    assert (result <= 1.0).all()

    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(1.0)


def test_score_creates_expected_columns() -> None:
    scorer = ResearchRiskScorer()

    result = scorer.score(make_dataframe())

    expected_columns = {
        "model_uncertainty",
        "volatility_risk",
        "risk_score",
        "risk_band",
    }

    assert expected_columns.issubset(result.columns)


def test_risk_score_is_between_zero_and_one_hundred() -> None:
    scorer = ResearchRiskScorer()

    result = scorer.score(make_dataframe())

    assert (result["risk_score"] >= 0.0).all()
    assert (result["risk_score"] <= 100.0).all()


def test_maximum_uncertainty_produces_higher_model_risk() -> None:
    scorer = ResearchRiskScorer()

    result = scorer.score(
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01",
                    periods=2,
                    freq="D",
                ),
                "volatility_20": [0.01, 0.01],
                "probability_down": [0.99, 0.50],
                "probability_up": [0.01, 0.50],
            }
        )
    )

    assert (
        result.loc[1, "model_uncertainty"]
        > result.loc[0, "model_uncertainty"]
    )


def test_risk_band_classification() -> None:
    scorer = ResearchRiskScorer()

    assert scorer.classify_score(0.0) == "LOW"
    assert scorer.classify_score(24.99) == "LOW"
    assert scorer.classify_score(25.0) == "MODERATE"
    assert scorer.classify_score(49.99) == "MODERATE"
    assert scorer.classify_score(50.0) == "HIGH"
    assert scorer.classify_score(74.99) == "HIGH"
    assert scorer.classify_score(75.0) == "EXTREME"
    assert scorer.classify_score(100.0) == "EXTREME"


def test_summary() -> None:
    scorer = ResearchRiskScorer()

    scored = scorer.score(make_dataframe())
    report = scorer.summarize(scored)

    assert report.rows == 5
    assert 0.0 <= report.minimum_score <= 100.0
    assert 0.0 <= report.maximum_score <= 100.0

    total_bands = (
        report.low_count
        + report.moderate_count
        + report.high_count
        + report.extreme_count
    )

    assert total_bands == 5


def test_input_is_not_mutated() -> None:
    scorer = ResearchRiskScorer()

    original = make_dataframe()
    original_columns = list(original.columns)

    scorer.score(original)

    assert list(original.columns) == original_columns


def test_missing_probability_column() -> None:
    scorer = ResearchRiskScorer()

    dataframe = make_dataframe().drop(
        columns=["probability_up"]
    )

    with pytest.raises(ValueError, match="Missing probability"):
        scorer.score(dataframe)


def test_empty_dataframe() -> None:
    scorer = ResearchRiskScorer()

    dataframe = pd.DataFrame(
        columns=[
            "timestamp",
            "volatility_20",
            "probability_down",
            "probability_up",
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        scorer.score(dataframe)


def test_invalid_probability() -> None:
    scorer = ResearchRiskScorer()

    dataframe = make_dataframe()
    dataframe.loc[0, "probability_up"] = 1.5

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        scorer.score(dataframe)


def test_negative_volatility() -> None:
    scorer = ResearchRiskScorer()

    dataframe = make_dataframe()
    dataframe.loc[0, "volatility_20"] = -0.01

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        scorer.score(dataframe)


def test_invalid_weights() -> None:
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        RiskScoringConfig(
            model_uncertainty_weight=0.7,
            volatility_weight=0.7,
        )


def test_custom_weights() -> None:
    scorer = ResearchRiskScorer(
        RiskScoringConfig(
            model_uncertainty_weight=0.7,
            volatility_weight=0.3,
        )
    )

    result = scorer.score(make_dataframe())

    assert len(result) == 5
    assert result["risk_score"].between(0.0, 100.0).all()