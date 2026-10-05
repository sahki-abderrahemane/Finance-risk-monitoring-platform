from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.documents import Document

from app.config import get_settings
from app.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    One piece of evidence returned by the Sentinel-AI retriever.

    `distance` is the native vector-store distance. Lower values indicate
    closer vectors.

    `similarity` is a normalized convenience metric derived from distance:

        similarity = 1 - distance

    This normalized value is intended for downstream reporting and
    evaluation. Retrieval filtering itself always uses `distance`.
    """

    document: Document
    distance: float

    @property
    def similarity(self) -> float:
        """
        Convert vector distance into a bounded similarity-like score.

        For non-negative distances:

            distance = 0   -> similarity = 1
            distance -> inf -> similarity -> 0
        """

        return 1.0 - self.distance

    @property
    def score(self) -> float:
        """
        Backward-compatible alias for similarity.

        New code should prefer the explicit `similarity` property.
        """

        return self.similarity

    @property
    def document_id(self) -> str | None:
        """Return the originating source-document identifier."""

        value = self.document.metadata.get("document_id")

        return str(value) if value is not None else None

    @property
    def chunk_id(self) -> str | None:
        """Return the originating retrieval-chunk identifier."""

        value = self.document.metadata.get("chunk_id")

        return str(value) if value is not None else None

    @property
    def source(self) -> str | None:
        """Return the human-readable evidence source."""

        value = self.document.metadata.get("source")

        return str(value) if value is not None else None

    @property
    def source_type(self) -> str | None:
        """Return the Sentinel evidence-source category."""

        value = self.document.metadata.get("source_type")

        return str(value) if value is not None else None

    @property
    def ticker(self) -> str | None:
        """Return the associated ticker when available."""

        value = self.document.metadata.get("ticker")

        return str(value) if value is not None else None

    @property
    def publication_date(self) -> str | None:
        """Return the source publication date when available."""

        value = self.document.metadata.get(
            "publication_date"
        )

        return str(value) if value is not None else None

    @property
    def page(self) -> int | None:
        """Return the source page when available."""

        value = self.document.metadata.get("page")

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Expose the complete source metadata."""

        return self.document.metadata


class Retriever:
    """
    Sentinel-AI semantic retriever.

    Retrieval is deliberately separated from generation. This component
    returns evidence only; it never asks an LLM to produce a financial
    prediction or recommendation.

    pgvector/LangChain returns a vector distance from
    `similarity_search_with_score()`.

    Therefore:

        lower distance = better match

    and retrieval filtering uses `max_distance`.

    The retriever also exposes a normalized similarity-like value through
    RetrievalResult.similarity for evaluation and API presentation.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> None:
        settings = get_settings()

        resolved_top_k = (
            top_k
            if top_k is not None
            else settings.retrieval_top_k
        )

        resolved_max_distance = (
            max_distance
            if max_distance is not None
            else settings.retrieval_max_distance
        )

        if resolved_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if resolved_max_distance < 0.0:
            raise ValueError(
                "max_distance cannot be negative."
            )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

        self.top_k = resolved_top_k
        self.max_distance = resolved_max_distance

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_distance: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve semantically similar evidence.

        Parameters
        ----------
        query:
            Natural-language retrieval query.

        top_k:
            Maximum number of vector-store results requested.

        max_distance:
            Maximum accepted vector distance. Lower is better.

        filter:
            Optional pgvector metadata filter.

        Returns
        -------
        list[RetrievalResult]
            Evidence results whose vector distance is at or below the
            configured maximum distance.

        Raises
        ------
        ValueError
            If the query or retrieval parameters are invalid.
        """

        normalized_query = self._validate_query(query)

        result_limit = (
            top_k
            if top_k is not None
            else self.top_k
        )

        distance_threshold = (
            max_distance
            if max_distance is not None
            else self.max_distance
        )

        self._validate_parameters(
            top_k=result_limit,
            max_distance=distance_threshold,
        )

        retrieved = (
            self.vector_store
            .similarity_search_with_score(
                normalized_query,
                k=result_limit,
                filter=filter,
            )
        )

        return self._build_results(
            retrieved,
            max_distance=distance_threshold,
        )

    def retrieve_for_ticker(
        self,
        query: str,
        *,
        ticker: str,
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence constrained to one ticker.

        This prevents evidence belonging to unrelated entities from
        entering a ticker-specific explanation context.
        """

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError(
                "ticker cannot be empty."
            )

        return self.retrieve(
            query,
            top_k=top_k,
            max_distance=max_distance,
            filter={
                "ticker": normalized_ticker,
            },
        )

    def retrieve_for_source_type(
        self,
        query: str,
        *,
        source_type: str,
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve evidence constrained to one Sentinel source category.

        Supported categories include:

            sec_filing
            financial_news
            economic_report
        """

        normalized_source_type = source_type.strip()

        if not normalized_source_type:
            raise ValueError(
                "source_type cannot be empty."
            )

        return self.retrieve(
            query,
            top_k=top_k,
            max_distance=max_distance,
            filter={
                "source_type": normalized_source_type,
            },
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        """Normalize and validate the retrieval query."""

        if not isinstance(query, str):
            raise TypeError(
                "Retrieval query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        return normalized_query

    @staticmethod
    def _validate_parameters(
        *,
        top_k: int,
        max_distance: float,
    ) -> None:
        """Validate retrieval-specific runtime parameters."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if max_distance < 0.0:
            raise ValueError(
                "max_distance cannot be negative."
            )

    @staticmethod
    def _build_results(
        retrieved: list[tuple[Document, float]],
        *,
        max_distance: float,
    ) -> list[RetrievalResult]:
        """
        Convert vector-store output into Sentinel retrieval results.

        The vector-store distance is preserved without transformation.
        Filtering occurs against the native distance value.
        """

        results: list[RetrievalResult] = []

        for document, raw_distance in retrieved:
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Vector store returned a non-numeric "
                    "retrieval distance."
                ) from exc

            if distance < 0.0:
                raise RuntimeError(
                    "Vector store returned a negative "
                    "retrieval distance."
                )

            if distance > max_distance:
                continue

            results.append(
                RetrievalResult(
                    document=document,
                    distance=distance,
                )
            )

        return results