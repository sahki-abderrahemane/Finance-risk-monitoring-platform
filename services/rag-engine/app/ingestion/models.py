from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class DocumentSourceType(StrEnum):
    """Supported evidence source categories for Sentinel-AI."""

    SEC_FILING = "sec_filing"
    FINANCIAL_NEWS = "financial_news"
    ECONOMIC_REPORT = "economic_report"


class FinancialDocumentMetadata(BaseModel):
    """
    Provenance metadata attached to every Sentinel-AI source document.

    This contract is intentionally domain-specific. LangChain handles
    document transport and retrieval, while Sentinel owns the meaning
    and validation of financial evidence metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_id: str = Field(
        min_length=1,
        description="Stable identifier for the source document.",
    )

    source: str = Field(
        min_length=1,
        description="Human-readable source name.",
    )

    source_type: DocumentSourceType

    ticker: str | None = Field(
        default=None,
        min_length=1,
    )

    publication_date: date | None = None

    filing_type: str | None = Field(
        default=None,
        min_length=1,
    )

    section: str | None = Field(
        default=None,
        min_length=1,
    )

    url: str | None = Field(
        default=None,
        min_length=1,
    )

    page: int | None = Field(
        default=None,
        ge=1,
    )

    ingestion_timestamp: datetime

    embedding_model: str | None = Field(
        default=None,
        min_length=1,
    )

    def to_langchain_metadata(self) -> dict[str, Any]:
        """
        Convert the Sentinel metadata contract into LangChain-compatible
        metadata.

        Dates are serialized as ISO strings because vector stores should
        receive JSON-compatible metadata.
        """

        return {
            "document_id": self.document_id,
            "source": self.source,
            "source_type": self.source_type.value,
            "ticker": self.ticker,
            "publication_date": (
                self.publication_date.isoformat()
                if self.publication_date is not None
                else None
            ),
            "filing_type": self.filing_type,
            "section": self.section,
            "url": self.url,
            "page": self.page,
            "ingestion_timestamp": (
                self.ingestion_timestamp.isoformat()
            ),
            "embedding_model": self.embedding_model,
        }


def create_langchain_document(
    *,
    text: str,
    metadata: FinancialDocumentMetadata,
) -> Document:
    """
    Create a LangChain Document from Sentinel-validated metadata.

    The function prevents callers from constructing unvalidated
    financial evidence metadata directly.
    """

    normalized_text = text.strip()

    if not normalized_text:
        raise ValueError(
            "Document text cannot be empty."
        )

    return Document(
        page_content=normalized_text,
        metadata=metadata.to_langchain_metadata(),
    )