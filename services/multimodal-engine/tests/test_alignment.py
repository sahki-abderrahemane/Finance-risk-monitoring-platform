from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from multimodal_engine.alignment import (
    AlignmentError,
    align_multimodal_data,
)
from multimodal_engine.fusion_dataset import (
    MARKET_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
)
from multimodal_engine.schemas import AlignmentConfig


def build_market() -> pd.DataFrame:
    rows = []

    for ticker in ("AAPL", "MSFT"):
        for index in range(3):
            row = {
                "ticker": ticker,
                "timestamp": datetime(2026, 1, 1)
                + timedelta(days=index),
                "return_1d": 0.01,
                "log_return_1d": 0.0099,
                "sma_5": 100.0,
                "sma_20": 99.0,
                "ema_5": 100.0,
                "ema_20": 99.0,
                "volatility_20": 0.02,
                "volume_sma_20": 1_000_000.0,
                "volume_ratio": 1.1,
                "high_low_spread": 0.03,
                "open_close_spread": 0.01,
                "target_return_1d": 0.005,
                "target_direction_1d": 1,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def build_sentiment() -> pd.DataFrame:
    rows = []

    for ticker in ("AAPL", "MSFT"):
        for index in range(3):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": datetime(2026, 1, 1)
                    + timedelta(days=index),
                    "sentiment_score_mean": 0.10,
                    "sentiment_score_std": 0.05,
                    "positive_ratio": 0.50,
                    "neutral_ratio": 0.30,
                    "negative_ratio": 0.20,
                    "probability_negative_mean": 0.20,
                    "probability_neutral_mean": 0.30,
                    "probability_positive_mean": 0.50,
                    "confidence_mean": 0.90,
                }
            )

    return pd.DataFrame(rows)


def build_vision() -> pd.DataFrame:
    rows = []

    for ticker in ("AAPL", "MSFT"):
        for index in range(3):
            rows.append(
                {
                    "sample_id": f"{ticker}_{index}",
                    "ticker": ticker,
                    "timestamp": datetime(2026, 1, 1)
                    + timedelta(days=index),
                    "image_path": (
                        f"charts/{ticker}_{index}.png"
                    ),
                    "label": "UP",
                    "future_return": 0.02,
                    "window_size": 30,
                }
            )

    return pd.DataFrame(rows)


def test_alignment_produces_expected_rows() -> None:
    aligned, report = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    assert len(aligned) == 6
    assert report.market_rows == 6
    assert report.sentiment_rows == 6
    assert report.vision_rows == 6
    assert report.aligned_rows == 6
    assert report.ticker_count == 2


def test_alignment_preserves_market_features() -> None:
    aligned, _ = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    for column in MARKET_FEATURE_COLUMNS:
        assert column in aligned.columns

    assert aligned.iloc[0]["return_1d"] == pytest.approx(
        0.01
    )


def test_alignment_preserves_sentiment_features() -> None:
    aligned, _ = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    for column in SENTIMENT_FEATURE_COLUMNS:
        assert column in aligned.columns

    assert aligned.iloc[0][
        "sentiment_score_mean"
    ] == pytest.approx(0.10)


def test_alignment_adds_vision_metadata() -> None:
    aligned, _ = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    assert "sample_id" in aligned.columns
    assert "image_path" in aligned.columns
    assert "window_size" in aligned.columns

    assert aligned.iloc[0][
        "window_size"
    ] == 30


def test_alignment_keeps_target_separate() -> None:
    aligned, _ = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    assert "target_return_1d" in aligned.columns
    assert "target_direction_1d" in aligned.columns

    assert aligned.iloc[0][
        "target_direction_1d"
    ] == 1


def test_alignment_rejects_duplicate_market_keys() -> None:
    market = build_market()

    duplicate = market.iloc[[0]].copy()

    market = pd.concat(
        [market, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        AlignmentError,
        match="duplicate",
    ):
        align_multimodal_data(
            market=market,
            sentiment=build_sentiment(),
            vision=build_vision(),
        )


def test_alignment_rejects_missing_market_column() -> None:
    market = build_market().drop(
        columns=["return_1d"]
    )

    with pytest.raises(
        AlignmentError,
        match="missing required columns",
    ):
        align_multimodal_data(
            market=market,
            sentiment=build_sentiment(),
            vision=build_vision(),
        )


def test_alignment_requires_vision_by_default() -> None:
    vision = build_vision().iloc[:-1].copy()

    market = build_market()

    # Remove all vision rows for the first ticker so that
    # the market timeline still contains observations but
    # no complete multimodal match for that subset.
    vision = vision[
        vision["ticker"] == "MSFT"
    ].copy()

    config = AlignmentConfig(
        allowed_tickers=("AAPL",)
    )

    aligned, report = align_multimodal_data(
        market=market,
        sentiment=build_sentiment(),
        vision=vision,
        config=config,
    )

    assert len(aligned) == 0
    assert report.aligned_rows == 0


def test_sentiment_can_match_within_configured_gap() -> None:
    market = build_market()

    sentiment = build_sentiment()

    sentiment["timestamp"] = sentiment[
        "timestamp"
    ] + timedelta(hours=12)

    aligned, report = align_multimodal_data(
        market=market,
        sentiment=sentiment,
        vision=build_vision(),
        config=AlignmentConfig(
            max_sentiment_gap_days=1,
        ),
    )

    assert report.aligned_rows == 6
    assert len(aligned) == 6


def test_sentiment_outside_gap_is_rejected() -> None:
    market = build_market()

    sentiment = build_sentiment()

    sentiment["timestamp"] = sentiment[
        "timestamp"
    ] + timedelta(days=5)

    aligned, report = align_multimodal_data(
        market=market,
        sentiment=sentiment,
        vision=build_vision(),
        config=AlignmentConfig(
            max_sentiment_gap_days=1,
        ),
    )

    assert len(aligned) == 0
    assert report.aligned_rows == 0


def test_alignment_is_sorted_chronologically() -> None:
    market = build_market().sample(
        frac=1.0,
        random_state=42,
    )

    sentiment = build_sentiment().sample(
        frac=1.0,
        random_state=42,
    )

    vision = build_vision().sample(
        frac=1.0,
        random_state=42,
    )

    aligned, _ = align_multimodal_data(
        market=market,
        sentiment=sentiment,
        vision=vision,
    )

    expected = aligned.sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        aligned,
        expected,
    )


def test_alignment_has_unique_keys() -> None:
    aligned, _ = align_multimodal_data(
        market=build_market(),
        sentiment=build_sentiment(),
        vision=build_vision(),
    )

    assert not aligned.duplicated(
        subset=["ticker", "timestamp"]
    ).any()