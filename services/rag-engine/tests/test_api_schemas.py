from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    RetrievedEvidence,
    RetrievalRequest,
    RetrievalResponse,
)


class TestRetrievalRequest:
    def test_valid_request(self) -> None:
        request = RetrievalRequest(
            query="AAPL supply chain risk",
            top_k=5,
            max_distance=0.8,
            ticker="AAPL",
            source_type="sec_filing",
        )

        assert request.query == "AAPL supply chain risk"
        assert request.top_k == 5
        assert request.max_distance == 0.8
        assert request.ticker == "AAPL"
        assert request.source_type == "sec_filing"

    def test_defaults_are_none(self) -> None:
        request = RetrievalRequest(
            query="economic risk",
        )

        assert request.top_k is None
        assert request.max_distance is None
        assert request.ticker is None
        assert request.source_type is None

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="")

    def test_rejects_zero_top_k(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(
                query="risk",
                top_k=0,
            )

    def test_rejects_negative_distance(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(
                query="risk",
                max_distance=-0.1,
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(
                query="risk",
                unsupported="value",  # type: ignore[call-arg]
            )


class TestRetrievedEvidence:
    def test_valid_evidence(self) -> None:
        evidence = RetrievedEvidence(
            content="The company reports supply chain exposure.",
            distance=0.25,
            similarity=0.8,
            document_id="sec-aapl-2025",
            chunk_id="sec-aapl-2025:2",
            source="SEC",
            source_type="sec_filing",
            ticker="AAPL",
            publication_date="2025-10-31",
            page=12,
        )

        assert evidence.content.startswith(
            "The company"
        )
        assert evidence.distance == 0.25
        assert evidence.similarity == 0.8
        assert evidence.document_id == "sec-aapl-2025"
        assert evidence.page == 12

    def test_optional_metadata(self) -> None:
        evidence = RetrievedEvidence(
            content="Evidence",
            distance=0.4,
            similarity=0.7,
        )

        assert evidence.document_id is None
        assert evidence.chunk_id is None
        assert evidence.source is None
        assert evidence.ticker is None
        assert evidence.page is None


class TestRetrievalResponse:
    def test_response_count(self) -> None:
        evidence = RetrievedEvidence(
            content="Evidence",
            distance=0.2,
            similarity=0.83,
        )

        response = RetrievalResponse(
            query="risk",
            results=[evidence],
            result_count=1,
        )

        assert response.query == "risk"
        assert response.result_count == 1
        assert len(response.results) == 1

    def test_empty_response(self) -> None:
        response = RetrievalResponse(
            query="unknown risk",
            results=[],
            result_count=0,
        )

        assert response.results == []
        assert response.result_count == 0