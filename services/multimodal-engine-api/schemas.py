from __future__ import annotations

from pydantic import BaseModel, Field


class MarketFeatures(BaseModel):
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
    sentiment_score_mean: float
    sentiment_score_std: float
    positive_ratio: float = Field(ge=0.0, le=1.0)
    neutral_ratio: float = Field(ge=0.0, le=1.0)
    negative_ratio: float = Field(ge=0.0, le=1.0)
    probability_negative_mean: float = Field(ge=0.0, le=1.0)
    probability_neutral_mean: float = Field(ge=0.0, le=1.0)
    probability_positive_mean: float = Field(ge=0.0, le=1.0)
    confidence_mean: float = Field(ge=0.0, le=1.0)


class MultimodalPredictRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    market_features: MarketFeatures
    sentiment: SentimentFeatures
    news_text: str | None = Field(default=None, max_length=10_000)


class MultimodalPredictResponse(BaseModel):
    ticker: str
    risk_score: int
    risk_label: str
    probability: float
    uncertainty: float
    model_version: str
    components: dict[str, int]


class RiskScoreRequest(BaseModel):
    features: list[float] = Field(min_length=1)


class RiskScoreResponse(BaseModel):
    risk_score: int
    risk_label: str
    probability: float
    uncertainty: float


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    service: str
