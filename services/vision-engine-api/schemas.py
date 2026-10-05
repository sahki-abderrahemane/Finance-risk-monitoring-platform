from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyChartRequest(BaseModel):
    image_base64: str = Field(min_length=1, description="Base64-encoded PNG image")
    ticker: str | None = Field(default=None, min_length=1, max_length=10)


class ChartProbabilities(BaseModel):
    DOWN: float
    FLAT: float
    UP: float


class ClassifyChartResponse(BaseModel):
    label: str
    confidence: float
    probabilities: ChartProbabilities
    ticker: str | None
    model: str


class BatchClassifyItem(BaseModel):
    id: str = Field(min_length=1)
    image_base64: str = Field(min_length=1)


class BatchClassifyRequest(BaseModel):
    images: list[BatchClassifyItem] = Field(min_length=1, max_length=20)
    ticker: str | None = Field(default=None, min_length=1, max_length=10)


class BatchClassifyItemResponse(BaseModel):
    id: str
    label: str
    confidence: float
    probabilities: ChartProbabilities


class BatchClassifyResponse(BaseModel):
    results: list[BatchClassifyItemResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    service: str
