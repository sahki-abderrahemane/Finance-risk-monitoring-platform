from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.retrieval.retriever import (
    RetrievalResult,
    Retriever,
)


def make_document(
    *,
    document_id: str = "sec-aapl-2025",
    chunk_id: str = "sec-aapl-2025:0",
    source: str = "SEC",
    source_type: str = "sec_filing",
    ticker: str | None = "AAPL",
    publication_date: str = "2025-10-31",
    page: int | None = 12,
    content: str = "The company faces material economic risks.",
) -> Document:
    """Create a LangChain document with Sentinel provenance."""

    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "source": source,
            "source_type": source_type,
            "ticker": ticker,
            "publication_date": publication_date,
            "page": page,
        },
    )


class TestRetrievalResult:
    """Tests for structured retrieval evidence."""

    def test_distance_and_similarity(self) -> None:
        result = RetrievalResult(
            document=make_document(),
            distance=0.25,
        )

        assert result.distance == 0.25
        assert result.score == 0.25
        assert result.similarity == pytest.approx(
            1.0 / 1.25
        )

    def test_zero_distance_has_maximum_similarity(self) -> None:
        result = RetrievalResult(
            document=make_document(),
            distance=0.0,
        )

        assert result.similarity == 1.0

    def test_provenance_properties(self) -> None:
        result = RetrievalResult(
            document=make_document(),
            distance=0.3,
        )

        assert result.document_id == "sec-aapl-2025"
        assert result.chunk_id == "sec-aapl-2025:0"
        assert result.source == "SEC"
        assert result.source_type == "sec_filing"
        assert result.ticker == "AAPL"
        assert result.publication_date == "2025-10-31"
        assert result.page == 12

    def test_missing_optional_metadata(self) -> None:
        result = RetrievalResult(
            document=Document(
                page_content="Evidence.",
                metadata={},
            ),
            distance=0.4,
        )

        assert result.document_id is None
        assert result.chunk_id is None
        assert result.source is None
        assert result.source_type is None
        assert result.ticker is None
        assert result.publication_date is None
        assert result.page is None

    def test_invalid_page_returns_none(self) -> None:
        result = RetrievalResult(
            document=Document(
                page_content="Evidence.",
                metadata={"page": "invalid"},
            ),
            distance=0.4,
        )

        assert result.page is None


class TestRetrieverInitialization:
    """Tests for retriever configuration."""

    def test_custom_configuration(self) -> None:
        vector_store = Mock()

        retriever = Retriever(
            vector_store=vector_store,
            top_k=10,
            max_distance=0.75,
        )

        assert retriever.vector_store is vector_store
        assert retriever.top_k == 10
        assert retriever.max_distance == 0.75

    def test_rejects_zero_top_k(self) -> None:
        with pytest.raises(
            ValueError,
            match="top_k must be greater than zero",
        ):
            Retriever(
                vector_store=Mock(),
                top_k=0,
            )

    def test_rejects_negative_top_k(self) -> None:
        with pytest.raises(
            ValueError,
            match="top_k must be greater than zero",
        ):
            Retriever(
                vector_store=Mock(),
                top_k=-1,
            )

    def test_rejects_negative_max_distance(self) -> None:
        with pytest.raises(
            ValueError,
            match="max_distance cannot be negative",
        ):
            Retriever(
                vector_store=Mock(),
                max_distance=-0.1,
            )

    def test_accepts_zero_max_distance(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
            max_distance=0.0,
        )

        assert retriever.max_distance == 0.0


class TestRetrieverQueryValidation:
    """Tests for retrieval query validation."""

    def test_rejects_empty_query(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="query cannot be empty",
        ):
            retriever.retrieve("")

    def test_rejects_whitespace_query(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="query cannot be empty",
        ):
            retriever.retrieve("   ")

    def test_rejects_non_string_query(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            TypeError,
            match="query must be a string",
        ):
            retriever.retrieve(123)  # type: ignore[arg-type]


class TestBasicRetrieval:
    """Tests for basic semantic retrieval."""

    def test_returns_results(self) -> None:
        vector_store = Mock()

        document = make_document()

        vector_store.similarity_search_with_score.return_value = [
            (document, 0.25),
        ]

        retriever = Retriever(
            vector_store=vector_store,
            top_k=5,
            max_distance=1.0,
        )

        results = retriever.retrieve(
            "economic risk",
        )

        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].distance == 0.25
        assert results[0].similarity == pytest.approx(
            1.0 / 1.25
        )

    def test_passes_top_k_to_vector_store(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
            top_k=7,
        )

        retriever.retrieve(
            "supply chain risk",
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "supply chain risk",
            k=7,
            filter=None,
        )

    def test_strips_query(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
        )

        retriever.retrieve(
            "   economic risk   ",
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "economic risk",
            k=5,
            filter=None,
        )

    def test_runtime_top_k_overrides_default(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
            top_k=5,
        )

        retriever.retrieve(
            "risk",
            top_k=2,
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "risk",
            k=2,
            filter=None,
        )


class TestDistanceFiltering:
    """Tests for native vector-distance filtering."""

    def test_filters_distance_above_threshold(self) -> None:
        vector_store = Mock()

        close_document = make_document(
            chunk_id="close",
        )

        far_document = make_document(
            chunk_id="far",
        )

        vector_store.similarity_search_with_score.return_value = [
            (close_document, 0.20),
            (far_document, 0.80),
        ]

        retriever = Retriever(
            vector_store=vector_store,
            max_distance=0.50,
        )

        results = retriever.retrieve(
            "financial risk",
        )

        assert len(results) == 1
        assert results[0].chunk_id == "close"
        assert results[0].distance == 0.20

    def test_accepts_distance_equal_to_threshold(self) -> None:
        vector_store = Mock()

        document = make_document()

        vector_store.similarity_search_with_score.return_value = [
            (document, 0.50),
        ]

        retriever = Retriever(
            vector_store=vector_store,
            max_distance=0.50,
        )

        results = retriever.retrieve(
            "financial risk",
        )

        assert len(results) == 1

    def test_returns_empty_when_all_distances_fail(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = [
            (make_document(chunk_id="one"), 0.80),
            (make_document(chunk_id="two"), 0.90),
        ]

        retriever = Retriever(
            vector_store=vector_store,
            max_distance=0.50,
        )

        assert retriever.retrieve("risk") == []

    def test_zero_distance_is_accepted(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = [
            (make_document(), 0.0),
        ]

        retriever = Retriever(
            vector_store=vector_store,
            max_distance=0.0,
        )

        results = retriever.retrieve("risk")

        assert len(results) == 1
        assert results[0].distance == 0.0
        assert results[0].similarity == 1.0


class TestMetadataFiltering:
    """Tests for provenance-aware retrieval filters."""

    def test_retrieve_for_ticker(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
        )

        retriever.retrieve_for_ticker(
            "revenue risk",
            ticker="aapl",
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "revenue risk",
            k=5,
            filter={
                "ticker": "AAPL",
            },
        )

    def test_retrieve_for_ticker_rejects_empty_ticker(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="ticker cannot be empty",
        ):
            retriever.retrieve_for_ticker(
                "risk",
                ticker="   ",
            )

    def test_retrieve_for_source_type(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
        )

        retriever.retrieve_for_source_type(
            "inflation risk",
            source_type="economic_report",
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "inflation risk",
            k=5,
            filter={
                "source_type": "economic_report",
            },
        )

    def test_retrieve_for_source_type_rejects_empty_value(
        self,
    ) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="source_type cannot be empty",
        ):
            retriever.retrieve_for_source_type(
                "risk",
                source_type="   ",
            )

    def test_custom_filter_is_forwarded(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
        )

        metadata_filter = {
            "ticker": "AAPL",
            "source_type": "sec_filing",
        }

        retriever.retrieve(
            "risk factors",
            filter=metadata_filter,
        )

        vector_store.similarity_search_with_score.assert_called_once_with(
            "risk factors",
            k=5,
            filter=metadata_filter,
        )


class TestRuntimeValidation:
    """Tests for per-request retrieval parameters."""

    def test_rejects_zero_runtime_top_k(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="top_k must be greater than zero",
        ):
            retriever.retrieve(
                "risk",
                top_k=0,
            )

    def test_rejects_negative_runtime_top_k(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="top_k must be greater than zero",
        ):
            retriever.retrieve(
                "risk",
                top_k=-1,
            )

    def test_rejects_negative_runtime_distance(self) -> None:
        retriever = Retriever(
            vector_store=Mock(),
        )

        with pytest.raises(
            ValueError,
            match="max_distance cannot be negative",
        ):
            retriever.retrieve(
                "risk",
                max_distance=-0.1,
            )


class TestVectorStoreValidation:
    """Tests for defensive vector-store response handling."""

    def test_rejects_non_numeric_distance(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = [
            (
                make_document(),
                "invalid",
            )
        ]

        retriever = Retriever(
            vector_store=vector_store,
        )

        with pytest.raises(
            RuntimeError,
            match="non-numeric retrieval distance",
        ):
            retriever.retrieve("risk")

    def test_rejects_negative_distance(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = [
            (
                make_document(),
                -0.1,
            )
        ]

        retriever = Retriever(
            vector_store=vector_store,
        )

        with pytest.raises(
            RuntimeError,
            match="negative retrieval distance",
        ):
            retriever.retrieve("risk")

    def test_empty_vector_store_result_is_valid(self) -> None:
        vector_store = Mock()

        vector_store.similarity_search_with_score.return_value = []

        retriever = Retriever(
            vector_store=vector_store,
        )

        assert retriever.retrieve("risk") == []