from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.explanation.evidence import EvidenceChunk


class RiskOutputRequest(BaseModel):
    """
    API representation of an already-computed Sentinel-AI
    risk result.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    risk_score: float
    risk_label: str = Field(
        min_length=1,
    )

    model_name: str | None = Field(
        default=None,
        min_length=1,
    )

    model_version: str | None = Field(
        default=None,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ExplanationRequest(BaseModel):
    """
    Request for a grounded explanation.

    Evidence is supplied explicitly so the API does not silently
    introduce external context.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    risk_output: RiskOutputRequest
    evidence: list[EvidenceChunk] = Field(
        default_factory=list,
    )


class ExplanationResponse(BaseModel):
    """
    Validated grounded explanation returned by Sentinel-AI.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    explanation: str
    model: str
    provider: str
    evidence_count: int
    citations: list[str]