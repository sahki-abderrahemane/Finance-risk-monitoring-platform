"""Multimodal engine API routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from dependencies import fuse_features, model_loaded, predict_from_fused
from schemas import (
    HealthResponse,
    MarketFeatures,
    MultimodalPredictRequest,
    MultimodalPredictResponse,
    ReadyResponse,
    RiskScoreRequest,
    RiskScoreResponse,
    SentimentFeatures,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service="multimodal-engine-api",
        model_loaded=model_loaded,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse | dict:
    if not model_loaded:
        return {"status": "not_ready", "service": "multimodal-engine-api", "reason": "model_not_loaded"}
    return ReadyResponse(status="ready", service="multimodal-engine-api")


@router.post("/multimodal/predict", response_model=MultimodalPredictResponse)
def predict_multimodal(request: MultimodalPredictRequest) -> MultimodalPredictResponse:
    market_arr = np.array([
        request.market_features.return_1d,
        request.market_features.log_return_1d,
        request.market_features.sma_5,
        request.market_features.sma_20,
        request.market_features.ema_5,
        request.market_features.ema_20,
        request.market_features.volatility_20,
        request.market_features.volume_sma_20,
        request.market_features.volume_ratio,
        request.market_features.high_low_spread,
        request.market_features.open_close_spread,
    ], dtype=np.float32)

    sentiment_arr = np.array([
        request.sentiment.sentiment_score_mean,
        request.sentiment.sentiment_score_std,
        request.sentiment.positive_ratio,
        request.sentiment.neutral_ratio,
        request.sentiment.negative_ratio,
        request.sentiment.probability_negative_mean,
        request.sentiment.probability_neutral_mean,
        request.sentiment.probability_positive_mean,
        request.sentiment.confidence_mean,
    ], dtype=np.float32)

    try:
        fused = fuse_features(market_arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature fusion failed: {e}")

    try:
        risk_score, risk_label, probability, uncertainty = predict_from_fused(fused)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return MultimodalPredictResponse(
        ticker=request.ticker.upper(),
        risk_score=risk_score,
        risk_label=risk_label,
        probability=round(probability, 4),
        uncertainty=uncertainty,
        model_version="1.0",
        components={
            "market_pca_dim": 5,
            "text_embedding_dim": 768,
            "vision_embedding_dim": 128,
        },
    )


@router.post("/multimodal/risk-score", response_model=RiskScoreResponse)
def risk_score_from_features(request: RiskScoreRequest) -> RiskScoreResponse:
    features = np.array(request.features, dtype=np.float32)

    try:
        fused = fuse_features(features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature fusion failed: {e}")

    try:
        risk_score, risk_label, probability, uncertainty = predict_from_fused(fused)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return RiskScoreResponse(
        risk_score=risk_score,
        risk_label=risk_label,
        probability=round(probability, 4),
        uncertainty=uncertainty,
    )
