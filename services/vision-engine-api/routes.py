"""Vision engine API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dependencies import classify_image, decode_image, model_loaded
from schemas import (
    BatchClassifyItemResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    ChartProbabilities,
    ClassifyChartRequest,
    ClassifyChartResponse,
    HealthResponse,
    ReadyResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service="vision-engine-api",
        model_loaded=model_loaded,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse | dict:
    if not model_loaded:
        return {"status": "not_ready", "service": "vision-engine-api", "reason": "model_not_loaded"}
    return ReadyResponse(status="ready", service="vision-engine-api")


@router.post("/vision/classify-chart", response_model=ClassifyChartResponse)
def classify_chart(request: ClassifyChartRequest) -> ClassifyChartResponse:
    try:
        image = decode_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image: {e}")

    try:
        label, confidence, probs = classify_image(image)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ClassifyChartResponse(
        label=label,
        confidence=round(confidence, 4),
        probabilities=ChartProbabilities(**probs),
        ticker=request.ticker,
        model="ChartCNN",
    )


@router.post("/vision/classify-charts", response_model=BatchClassifyResponse)
def classify_charts_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    results = []
    for item in request.images:
        try:
            image = decode_image(item.image_base64)
            label, confidence, probs = classify_image(image)
            results.append(BatchClassifyItemResponse(
                id=item.id,
                label=label,
                confidence=round(confidence, 4),
                probabilities=ChartProbabilities(**probs),
            ))
        except Exception:
            continue

    return BatchClassifyResponse(results=results, count=len(results))
