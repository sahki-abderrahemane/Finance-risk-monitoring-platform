from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from multimodal_engine.schemas import (
    AlignmentConfig,
    MarketFeatures,
    MultimodalKey,
    MultimodalLabel,
    MultimodalSample,
    SentimentFeatures,
    VisionMetadata,
)


def build_market_features() -> MarketFeatures:
    return MarketFeatures(
        return_1d=0.01,
        log_return_1d=0.00995,
        sma_5=101.0,
        sma_20=99.0,
        ema_5=100.5,
        ema_20=99.5,
        volatility_20=0.02,
        volume_sma_20=1_000_000.0,
        volume_ratio=1.2,
        high_low_spread=0.03,
        open_close_spread=0.01,
    )


def build_sentiment_features() -> SentimentFeatures:
    return SentimentFeatures(
        sentiment_score_mean=0.15,
        sentiment_score_std=0.10,
        positive_ratio=0.50,
        neutral_ratio=0.30,
        negative_ratio=0.20,
        probability_negative_mean=0.20,
        probability_neutral_mean=0.30,
        probability_positive_mean=0.50,
        confidence_mean=0.85,
    )


def build_vision_metadata() -> VisionMetadata:
    return VisionMetadata(
        sample_id="AAPL_20260101",
        image_path="charts/AAPL_20260101.png",
        window_size=30,
        future_return=0.02,
        label=MultimodalLabel.UP,
    )


def test_multimodal_key_normalizes_ticker() -> None:
    key = MultimodalKey(
        ticker=" aapl ",
        timestamp=datetime(2026, 1, 1),
    )

    assert key.ticker == "AAPL"


def test_market_features_accept_phase1_features() -> None:
    features = build_market_features()

    assert features.return_1d == pytest.approx(0.01)
    assert features.volatility_20 == pytest.approx(0.02)


def test_sentiment_features_validate_probabilities() -> None:
    features = build_sentiment_features()

    assert features.positive_ratio == pytest.approx(0.50)


def test_sentiment_features_reject_invalid_probability() -> None:
    with pytest.raises(ValidationError):
        SentimentFeatures(
            sentiment_score_mean=0.0,
            sentiment_score_std=0.0,
            positive_ratio=1.5,
            neutral_ratio=0.0,
            negative_ratio=0.0,
            probability_negative_mean=0.0,
            probability_neutral_mean=0.0,
            probability_positive_mean=0.0,
            confidence_mean=0.0,
        )


def test_vision_metadata_requires_positive_window() -> None:
    with pytest.raises(ValidationError):
        VisionMetadata(
            sample_id="sample",
            image_path="image.png",
            window_size=0,
        )


def test_multimodal_sample_accepts_complete_contract() -> None:
    sample = MultimodalSample(
        key=MultimodalKey(
            ticker="NVDA",
            timestamp=datetime(2026, 1, 1),
        ),
        market=build_market_features(),
        sentiment=build_sentiment_features(),
        vision=build_vision_metadata(),
        target_return_1d=0.01,
        target_direction_1d=1,
    )

    assert sample.key.ticker == "NVDA"
    assert sample.target_direction_1d == 1
    assert sample.vision.label == MultimodalLabel.UP


def test_multimodal_sample_rejects_invalid_target() -> None:
    with pytest.raises(ValidationError):
        MultimodalSample(
            key=MultimodalKey(
                ticker="AAPL",
                timestamp=datetime(2026, 1, 1),
            ),
            market=build_market_features(),
            sentiment=build_sentiment_features(),
            vision=build_vision_metadata(),
            target_return_1d=0.01,
            target_direction_1d=2,
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MarketFeatures(
            return_1d=0.01,
            log_return_1d=0.01,
            sma_5=1.0,
            sma_20=1.0,
            ema_5=1.0,
            ema_20=1.0,
            volatility_20=0.01,
            volume_sma_20=1.0,
            volume_ratio=1.0,
            high_low_spread=0.01,
            open_close_spread=0.01,
            unexpected=123,
        )


def test_alignment_config_defaults() -> None:
    config = AlignmentConfig()

    assert config.max_sentiment_gap_days == 1
    assert config.require_sentiment is True
    assert config.require_vision is True
    assert len(config.allowed_tickers) == 5