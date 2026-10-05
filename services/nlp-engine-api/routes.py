"""NLP engine API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dependencies import get_preprocessor, get_sentiment_model, model_loaded
from schemas import (
    AnalyzeArticleRequest,
    AnalyzeArticleResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    ReadyResponse,
    SentimentProbabilities,
    SentimentRequest,
    SentimentResponse,
)

router = APIRouter()


def _to_response(prediction, ticker: str | None = None) -> SentimentResponse:
    return SentimentResponse(
        label=prediction.label,
        confidence=round(prediction.confidence, 4),
        probabilities=SentimentProbabilities(
            negative=round(prediction.probability_negative, 4),
            neutral=round(prediction.probability_neutral, 4),
            positive=round(prediction.probability_positive, 4),
        ),
        ticker=ticker,
        model="ProsusAI/finbert",
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service="nlp-engine-api",
        model_loaded=model_loaded,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse | dict:
    if not model_loaded:
        return {"status": "not_ready", "service": "nlp-engine-api", "reason": "model_not_loaded"}
    return ReadyResponse(status="ready", service="nlp-engine-api")


@router.post("/nlp/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    model = get_sentiment_model()
    try:
        prediction = model.predict_one(request.text)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _to_response(prediction, ticker=request.ticker)


@router.post("/nlp/sentiment/batch", response_model=BatchSentimentResponse)
def analyze_sentiment_batch(request: BatchSentimentRequest) -> BatchSentimentResponse:
    model = get_sentiment_model()
    try:
        predictions = model.predict(request.texts)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    results = [_to_response(p, ticker=request.ticker) for p in predictions]
    return BatchSentimentResponse(predictions=results, count=len(results))


@router.post("/nlp/analyze-article", response_model=AnalyzeArticleResponse)
def analyze_article(request: AnalyzeArticleRequest) -> AnalyzeArticleResponse:
    model = get_sentiment_model()
    preprocessor = get_preprocessor()

    headline_pred = model.predict_one(request.headline)
    full_text_pred = model.predict_one(request.text)

    combined = f"{request.headline} [SEP] {request.text}"

    return AnalyzeArticleResponse(
        ticker=request.ticker,
        source=request.source,
        headline_sentiment=_to_response(headline_pred, ticker=request.ticker),
        full_text_sentiment=_to_response(full_text_pred, ticker=request.ticker),
        combined_text=combined[:500],
    )
