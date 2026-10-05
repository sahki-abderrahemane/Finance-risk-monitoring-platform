"""Risk engine API routes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from dependencies import (
    AVAILABLE_TICKERS,
    compute_features_from_ohlcv,
    get_gbm_model,
    get_risk_scorer,
    models_loaded,
)
from schemas import (
    BatchRiskScoreRequest,
    BatchRiskScoreResponse,
    FeaturesFromOHLCVRequest,
    FeaturesFromOHLCVResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    RiskScoreRequest,
    RiskScoreResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        service="risk-engine-api",
        models_loaded=models_loaded,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse | dict:
    if not models_loaded:
        return {"status": "not_ready", "service": "risk-engine-api", "reason": "models_not_loaded"}
    return ReadyResponse(status="ready", service="risk-engine-api")


@router.get("/risk/models", response_model=ModelInfoResponse)
def list_models() -> ModelInfoResponse:
    loaded = [t for t in AVAILABLE_TICKERS if t in _get_gbm_models()]
    return ModelInfoResponse(tickers=loaded, model_type="gradient_boosting")


@router.post("/risk/score", response_model=RiskScoreResponse)
def score_risk(request: RiskScoreRequest) -> RiskScoreResponse:
    ticker = request.ticker.upper()

    model = _get_gbm_model(ticker)
    scorer = _get_risk_scorer()

    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(tz="UTC"),
        **request.features.model_dump(),
    }])

    proba = model.predict_proba(row)
    p_down, p_up = float(proba[0, 0]), float(proba[0, 1])

    uncertainty = 1.0 - max(p_down, p_up)

    vol = request.features.volatility_20
    vol_normalized = min(vol / 0.5, 1.0) if vol > 0 else 0.0

    config = scorer.config
    risk_score = 100.0 * (
        config.model_uncertainty_weight * uncertainty
        + config.volatility_weight * vol_normalized
    )
    risk_score = max(0.0, min(100.0, risk_score))
    risk_label = scorer.classify_score(risk_score)

    return RiskScoreResponse(
        ticker=ticker,
        risk_score=round(risk_score, 2),
        risk_label=risk_label,
        model_uncertainty=round(uncertainty, 4),
        volatility_risk=round(vol_normalized, 4),
        probabilities={"down": round(p_down, 4), "up": round(p_up, 4)},
        model_name="gradient_boosting",
        model_version="1.0",
    )


@router.post("/risk/score/batch", response_model=BatchRiskScoreResponse)
def score_risk_batch(request: BatchRiskScoreRequest) -> BatchRiskScoreResponse:
    results = []
    for ticker in request.tickers:
        try:
            req = RiskScoreRequest(ticker=ticker, features=request.features)
            results.append(score_risk(req))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return BatchRiskScoreResponse(results=results)


@router.post("/risk/features", response_model=FeaturesFromOHLCVResponse)
def compute_features(request: FeaturesFromOHLCVRequest) -> FeaturesFromOHLCVResponse:
    try:
        features = compute_features_from_ohlcv(request.data, request.ticker)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return FeaturesFromOHLCVResponse(
        ticker=request.ticker.upper(),
        features=features,
    )


def _get_gbm_models() -> dict:
    from dependencies import gbm_models
    return gbm_models


def _get_risk_scorer():
    return get_risk_scorer()
