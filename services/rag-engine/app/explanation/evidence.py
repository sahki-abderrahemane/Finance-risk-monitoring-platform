from __future__ import annotations

from datetime import date
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class EvidenceChunk(BaseModel):
    """
    Immutable evidence contract passed from retrieval into
    grounded explanation.

    Sentinel-AI treats this object as source evidence, not as
    model-generated knowledge.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    content: str = Field(
        min_length=1,
        description="Retrieved source text.",
    )

    distance: float = Field(
        ge=0.0,
        description="Vector distance returned by pgvector.",
    )

    similarity: float = Field(
        ge=0.0,
        description="Normalized retrieval similarity.",
    )

    document_id: str = Field(
        min_length=1,
    )

    chunk_id: str = Field(
        min_length=1,
    )

    source: str = Field(
        min_length=1,
    )

    source_type: str = Field(
        min_length=1,
    )

    ticker: str | None = Field(
        default=None,
        min_length=1,
    )

    publication_date: date | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    def citation_label(self) -> str:
        """
        Produce a stable human-readable citation identifier.
        """

        if self.page is not None:
            return (
                f"{self.source} | "
                f"{self.document_id} | "
                f"page {self.page}"
            )

        return (
            f"{self.source} | "
            f"{self.document_id} | "
            f"{self.chunk_id}"
        )


class EvidenceSet(BaseModel):
    """
    Validated collection of retrieved evidence.

    An empty evidence set is valid and explicitly represents
    insufficient retrieval evidence.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    items: tuple[EvidenceChunk, ...] = ()

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def has_evidence(self) -> bool:
        return bool(self.items)

    def citation_labels(self) -> list[str]:
        return [
            item.citation_label()
            for item in self.items
        ]

    @classmethod
    def from_sequence(
        cls,
        evidence: Sequence[EvidenceChunk],
    ) -> EvidenceSet:
        return cls(
            items=tuple(evidence),
        )