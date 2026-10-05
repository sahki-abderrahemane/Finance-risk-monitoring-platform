from __future__ import annotations

from pydantic import BaseModel, Field


class RiskFeatures(BaseModel):
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


class RiskScoreRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    features: RiskFeatures


class RiskScoreResponse(BaseModel):
    ticker: str
    risk_score: float
    risk_label: str
    model_uncertainty: float
    volatility_risk: float
    probabilities: dict[str, float]
    model_name: str
    model_version: str


class BatchRiskScoreRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    features: RiskFeatures


class BatchRiskScoreResponse(BaseModel):
    results: list[RiskScoreResponse]


class FeaturesFromOHLCVRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    data: list[dict[str, float]] = Field(min_length=1)


class FeaturesFromOHLCVResponse(BaseModel):
    ticker: str
    features: RiskFeatures


class ModelInfoResponse(BaseModel):
    tickers: list[str]
    model_type: str


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    service: str
