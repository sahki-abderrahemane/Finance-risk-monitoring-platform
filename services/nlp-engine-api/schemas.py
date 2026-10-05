from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    ticker: str | None = Field(default=None, min_length=1, max_length=10)


class SentimentProbabilities(BaseModel):
    negative: float
    neutral: float
    positive: float


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    probabilities: SentimentProbabilities
    ticker: str | None
    model: str


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50)
    ticker: str | None = Field(default=None, min_length=1, max_length=10)


class BatchSentimentResponse(BaseModel):
    predictions: list[SentimentResponse]
    count: int


class AnalyzeArticleRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=10_000)
    source: str = Field(min_length=1, max_length=200)
    ticker: str = Field(min_length=1, max_length=10)


class AnalyzeArticleResponse(BaseModel):
    ticker: str
    source: str
    headline_sentiment: SentimentResponse
    full_text_sentiment: SentimentResponse
    combined_text: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    service: str
