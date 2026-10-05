from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievalRequest(BaseModel):
    """
    Request contract for semantic evidence retrieval.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        description="Natural-language evidence query.",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of evidence chunks.",
    )

    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum accepted vector distance.",
    )

    ticker: str | None = Field(
        default=None,
        min_length=1,
        description="Optional ticker filter.",
    )

    source_type: str | None = Field(
        default=None,
        min_length=1,
        description="Optional evidence-source filter.",
    )


class RetrievedEvidence(BaseModel):
    """
    API representation of one retrieved evidence chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    content: str
    distance: float
    similarity: float

    document_id: str | None = None
    chunk_id: str | None = None

    source: str | None = None
    source_type: str | None = None
    ticker: str | None = None

    publication_date: str | None = None
    page: int | None = None


class RetrievalResponse(BaseModel):
    """
    Response contract for evidence retrieval.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str
    results: list[RetrievedEvidence]
    result_count: int