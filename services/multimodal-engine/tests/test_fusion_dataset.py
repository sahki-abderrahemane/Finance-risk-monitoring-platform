"""Tests for the multimodal fusion dataset validation layer."""

from __future__ import annotations

import pandas as pd
import pytest

from multimodal_engine.fusion_dataset import (
    FusionDatasetError,
    validate_fusion_dataset,
)


def _valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid fusion dataset."""

    return pd.DataFrame(
        {
            "timestamp": [
                "2025-01-02",
                "2025-01-03",
            ],
            "ticker": [
                "aapl",
                "MSFT",
            ],
            "return_1d": [
                0.01,
                -0.02,
            ],
            "log_return_1d": [
                0.0099,
                -0.0202,
            ],
            "sma_5": [
                150.0,
                200.0,
            ],
            "sma_20": [
                148.0,
                195.0,
            ],
            "ema_5": [
                150.0,
                200.0,
            ],
            "ema_20": [
                148.0,
                195.0,
            ],
            "volatility_20": [
                0.02,
                0.03,
            ],
            "volume_sma_20": [
                1_000_000.0,
                2_000_000.0,
            ],
            "volume_ratio": [
                1.1,
                0.9,
            ],
            "high_low_spread": [
                0.03,
                0.04,
            ],
            "open_close_spread": [
                0.01,
                0.02,
            ],
            "target_return_1d": [
                0.01,
                -0.02,
            ],
            "target_direction_1d": [
                1,
                0,
            ],
            "sentiment_score_mean": [
                0.2,
                -0.1,
            ],
            "sentiment_score_std": [
                0.1,
                0.2,
            ],
            "positive_ratio": [
                0.5,
                0.2,
            ],
            "neutral_ratio": [
                0.3,
                0.4,
            ],
            "negative_ratio": [
                0.2,
                0.4,
            ],
            "probability_negative_mean": [
                0.2,
                0.4,
            ],
            "probability_neutral_mean": [
                0.3,
                0.4,
            ],
            "probability_positive_mean": [
                0.5,
                0.2,
            ],
            "confidence_mean": [
                0.8,
                0.7,
            ],
            "sample_id": [
                "aapl_001",
                "msft_001",
            ],
            "image_path": [
                "aapl/chart.png",
                "msft/chart.png",
            ],
            "window_size": [
                30,
                30,
            ],
            "future_return": [
                0.02,
                -0.01,
            ],
        }
    )


def test_valid_dataset_is_normalized() -> None:
    """Valid datasets should be normalized and sorted."""

    dataframe = _valid_dataframe()

    result = validate_fusion_dataset(dataframe)

    assert result["ticker"].tolist() == [
        "AAPL",
        "MSFT",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )


def test_missing_column_is_rejected() -> None:
    """Required columns must exist."""

    dataframe = _valid_dataframe().drop(
        columns=["confidence_mean"]
    )

    with pytest.raises(
        FusionDatasetError,
        match="missing required columns",
    ):
        validate_fusion_dataset(dataframe)


def test_duplicate_keys_are_rejected() -> None:
    """Ticker/timestamp pairs must be unique."""

    dataframe = _valid_dataframe()

    duplicate = dataframe.iloc[[0]].copy()

    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        FusionDatasetError,
        match="Duplicate ticker/timestamp keys",
    ):
        validate_fusion_dataset(dataframe)


def test_probability_values_must_be_in_range() -> None:
    """Probability-like fields must be bounded."""

    dataframe = _valid_dataframe()

    dataframe.loc[
        0,
        "confidence_mean",
    ] = 1.5

    with pytest.raises(
        FusionDatasetError,
        match="outside \\[0, 1\\]",
    ):
        validate_fusion_dataset(dataframe)


def test_sentiment_ratios_must_sum_to_one() -> None:
    """Sentiment ratios must form a valid distribution."""

    dataframe = _valid_dataframe()

    dataframe.loc[
        0,
        "positive_ratio",
    ] = 0.9

    with pytest.raises(
        FusionDatasetError,
        match="must sum to approximately 1.0",
    ):
        validate_fusion_dataset(dataframe)


def test_target_direction_is_binary() -> None:
    """The Phase 1 target must remain binary."""

    dataframe = _valid_dataframe()

    dataframe.loc[
        0,
        "target_direction_1d",
    ] = 2

    with pytest.raises(
        FusionDatasetError,
        match="must contain only binary values",
    ):
        validate_fusion_dataset(dataframe)


def test_window_size_must_be_positive() -> None:
    """Chart windows must have positive lengths."""

    dataframe = _valid_dataframe()

    dataframe.loc[
        0,
        "window_size",
    ] = 0

    with pytest.raises(
        FusionDatasetError,
        match="must be greater than zero",
    ):
        validate_fusion_dataset(dataframe)


def test_input_dataframe_is_not_mutated() -> None:
    """Validation should not mutate the caller's dataframe."""

    dataframe = _valid_dataframe()

    original_ticker = dataframe["ticker"].tolist()

    validate_fusion_dataset(dataframe)

    assert dataframe["ticker"].tolist() == original_ticker