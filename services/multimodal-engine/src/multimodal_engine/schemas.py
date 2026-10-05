from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MultimodalLabel(StrEnum):
    """Unified supervised target used by the multimodal model."""

    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"


class MultimodalKey(BaseModel):
    """Unique temporal key for a multimodal observation."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    timestamp: datetime

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()

        if not normalized:
            raise ValueError("ticker cannot be empty")

        return normalized


class MarketFeatures(BaseModel):
    """Phase 1 market feature vector."""

    model_config = ConfigDict(extra="forbid")

    return_1d: float
    log_return_1d: float
    sma_5: float
    sma_20: float
    ema_5: float
    ema_20: float
    volatility_20: float
    volume_sma_20: float
    volume_ratio: float
    high_low_spread: float
    open_close_spread: float


class SentimentFeatures(BaseModel):
    """Daily aggregated FinBERT sentiment features."""

    model_config = ConfigDict(extra="forbid")

    sentiment_score_mean: float
    sentiment_score_std: float
    positive_ratio: float
    neutral_ratio: float
    negative_ratio: float
    probability_negative_mean: float
    probability_neutral_mean: float
    probability_positive_mean: float
    confidence_mean: float

    @field_validator(
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "probability_negative_mean",
        "probability_neutral_mean",
        "probability_positive_mean",
        "confidence_mean",
    )
    @classmethod
    def validate_probability_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Expected value in [0, 1], received {value}"
            )

        return value


class VisionMetadata(BaseModel):
    """Metadata identifying the chart associated with an observation."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    window_size: int = Field(gt=0)

    future_return: float | None = None
    label: MultimodalLabel | None = None


class MultimodalSample(BaseModel):
    """
    Fully aligned multimodal observation.

    This contract intentionally stores vision metadata rather than the
    neural embedding. The embedding is generated later by the vision
    inference component.
    """

    model_config = ConfigDict(extra="forbid")

    key: MultimodalKey
    market: MarketFeatures
    sentiment: SentimentFeatures
    vision: VisionMetadata

    target_return_1d: float
    target_direction_1d: int

    @field_validator("target_direction_1d")
    @classmethod
    def validate_direction(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(
                "target_direction_1d must be 0 or 1"
            )

        return value


class AlignmentConfig(BaseModel):
    """Configuration controlling multimodal temporal alignment."""

    model_config = ConfigDict(extra="forbid")

    max_sentiment_gap_days: int = Field(
        default=1,
        ge=0,
    )

    require_sentiment: bool = True
    require_vision: bool = True
    drop_missing_market: bool = True

    allowed_tickers: tuple[str, ...] = (
        "AAPL",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
    )

    @field_validator("allowed_tickers")
    @classmethod
    def normalize_tickers(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            ticker.strip().upper()
            for ticker in value
            if ticker.strip()
        )

        if not normalized:
            raise ValueError(
                "allowed_tickers cannot be empty"
            )

        return normalized