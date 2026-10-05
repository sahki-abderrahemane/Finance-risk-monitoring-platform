from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.ingestion.loaders import DocumentLoader
from app.ingestion.models import (
    DocumentSourceType,
    FinancialDocumentMetadata,
    create_langchain_document,
)
from app.ingestion.pipeline import (
    IngestionPipeline,
    IngestionResult,
)
from app.ingestion.splitter import DocumentSplitter


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def financial_metadata() -> FinancialDocumentMetadata:
    """
    Standard provenance metadata used throughout ingestion tests.
    """

    return FinancialDocumentMetadata(
        document_id="sec-10k-aapl-2025",
        source="SEC",
        source_type=DocumentSourceType.SEC_FILING,
        ticker="AAPL",
        publication_date=date(2025, 10, 31),
        filing_type="10-K",
        section="Risk Factors",
        url="https://example.com/sec/aapl-10k-2025",
        ingestion_timestamp=datetime.now(timezone.utc),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture
def sample_document(
    financial_metadata: FinancialDocumentMetadata,
) -> Document:
    """
    Valid LangChain document containing Sentinel provenance.
    """

    return create_langchain_document(
        text=(
            "The company faces material risks related to global "
            "economic conditions, supply chain disruptions, foreign "
            "exchange rates, and changing consumer demand."
        ),
        metadata=financial_metadata,
    )


# ---------------------------------------------------------------------------
# FinancialDocumentMetadata tests
# ---------------------------------------------------------------------------


class TestFinancialDocumentMetadata:
    """Tests for the Sentinel financial-document metadata contract."""

    def test_metadata_requires_document_id(self) -> None:
        with pytest.raises(ValueError):
            FinancialDocumentMetadata(
                document_id="",
                source="SEC",
                source_type=DocumentSourceType.SEC_FILING,
                ingestion_timestamp=datetime.now(timezone.utc),
            )

    def test_metadata_requires_source(self) -> None:
        with pytest.raises(ValueError):
            FinancialDocumentMetadata(
                document_id="doc-001",
                source="",
                source_type=DocumentSourceType.SEC_FILING,
                ingestion_timestamp=datetime.now(timezone.utc),
            )

    def test_metadata_serializes_enum_and_dates(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        metadata = financial_metadata.to_langchain_metadata()

        assert metadata["document_id"] == "sec-10k-aapl-2025"
        assert metadata["source"] == "SEC"
        assert metadata["source_type"] == "sec_filing"
        assert metadata["ticker"] == "AAPL"
        assert metadata["publication_date"] == "2025-10-31"
        assert metadata["filing_type"] == "10-K"
        assert metadata["section"] == "Risk Factors"
        assert metadata["page"] is None
        assert metadata["embedding_model"] == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_metadata_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValueError):
            FinancialDocumentMetadata(
                document_id="doc-001",
                source="SEC",
                source_type=DocumentSourceType.SEC_FILING,
                ingestion_timestamp=datetime.now(timezone.utc),
                unexpected_field="not-allowed",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# LangChain document creation tests
# ---------------------------------------------------------------------------


class TestCreateLangChainDocument:
    """Tests for converting Sentinel metadata into LangChain Documents."""

    def test_creates_document_with_normalized_text(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        document = create_langchain_document(
            text="   Example financial text.   ",
            metadata=financial_metadata,
        )

        assert isinstance(document, Document)
        assert document.page_content == "Example financial text."
        assert document.metadata["document_id"] == (
            "sec-10k-aapl-2025"
        )
        assert document.metadata["ticker"] == "AAPL"

    def test_rejects_empty_text(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            create_langchain_document(
                text="   ",
                metadata=financial_metadata,
            )


# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    """Tests for source-document loading."""

    def test_rejects_missing_file(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        loader = DocumentLoader()

        missing_path = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            loader.load(
                missing_path,
                metadata=financial_metadata,
            )

    def test_rejects_directory(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        loader = DocumentLoader()

        directory = tmp_path / "documents"
        directory.mkdir()

        with pytest.raises(ValueError, match="not a file"):
            loader.load(
                directory,
                metadata=financial_metadata,
            )

    def test_rejects_unsupported_file_type(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        loader = DocumentLoader()

        source = tmp_path / "document.docx"
        source.write_text(
            "Unsupported document.",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unsupported document type"):
            loader.load(
                source,
                metadata=financial_metadata,
            )

    def test_loads_text_file(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        loader = DocumentLoader()

        source = tmp_path / "financial_report.txt"
        source.write_text(
            "Revenue increased during the reporting period.",
            encoding="utf-8",
        )

        documents = loader.load(
            source,
            metadata=financial_metadata,
        )

        assert len(documents) == 1

        document = documents[0]

        assert document.page_content == (
            "Revenue increased during the reporting period."
        )

        assert document.metadata["document_id"] == (
            financial_metadata.document_id
        )

        assert document.metadata["source"] == (
            financial_metadata.source
        )

        assert document.metadata["ticker"] == "AAPL"

        assert document.metadata["source_type"] == (
            "sec_filing"
        )

    def test_text_loader_preserves_provenance(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        loader = DocumentLoader()

        source = tmp_path / "risk_factors.txt"
        source.write_text(
            "Supply chain risks remain material.",
            encoding="utf-8",
        )

        documents = loader.load(
            source,
            metadata=financial_metadata,
        )

        assert len(documents) == 1

        metadata = documents[0].metadata

        assert metadata["document_id"] == (
            "sec-10k-aapl-2025"
        )
        assert metadata["source"] == "SEC"
        assert metadata["source_type"] == "sec_filing"
        assert metadata["ticker"] == "AAPL"
        assert metadata["filing_type"] == "10-K"
        assert metadata["section"] == "Risk Factors"


# ---------------------------------------------------------------------------
# DocumentSplitter tests
# ---------------------------------------------------------------------------


class TestDocumentSplitter:
    """Tests for retrieval-oriented document chunking."""

    def test_rejects_invalid_chunk_size(self) -> None:
        with pytest.raises(
            ValueError,
            match="chunk_size must be greater than zero",
        ):
            DocumentSplitter(
                chunk_size=0,
                chunk_overlap=0,
            )

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(
            ValueError,
            match="chunk_overlap cannot be negative",
        ):
            DocumentSplitter(
                chunk_size=100,
                chunk_overlap=-1,
            )

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with pytest.raises(
            ValueError,
            match="smaller than chunk_size",
        ):
            DocumentSplitter(
                chunk_size=100,
                chunk_overlap=100,
            )

    def test_splits_long_document(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        document = create_langchain_document(
            text=(
                "Revenue increased substantially during the year. "
                "Operating expenses also increased due to higher "
                "research and development expenditure. "
                "The company continues to monitor macroeconomic "
                "conditions and supply chain risks."
            ),
            metadata=financial_metadata,
        )

        splitter = DocumentSplitter(
            chunk_size=80,
            chunk_overlap=10,
        )

        chunks = splitter.split([document])

        assert len(chunks) > 1

        for chunk in chunks:
            assert chunk.page_content.strip()

    def test_preserves_document_provenance(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        document = create_langchain_document(
            text=(
                "The company operates in multiple geographic "
                "markets and is exposed to currency risk. "
                "Changes in exchange rates may affect reported "
                "financial results."
            ),
            metadata=financial_metadata,
        )

        splitter = DocumentSplitter(
            chunk_size=60,
            chunk_overlap=10,
        )

        chunks = splitter.split([document])

        assert chunks

        for chunk in chunks:
            assert chunk.metadata["document_id"] == (
                "sec-10k-aapl-2025"
            )

            assert chunk.metadata["ticker"] == "AAPL"
            assert chunk.metadata["source"] == "SEC"

    def test_assigns_unique_chunk_ids(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        document = create_langchain_document(
            text=(
                "First section of the document. "
                "Second section of the document. "
                "Third section of the document. "
                "Fourth section of the document."
            ),
            metadata=financial_metadata,
        )

        splitter = DocumentSplitter(
            chunk_size=40,
            chunk_overlap=5,
        )

        chunks = splitter.split([document])

        chunk_ids = [
            str(chunk.metadata["chunk_id"])
            for chunk in chunks
        ]

        assert len(chunk_ids) == len(set(chunk_ids))

        for index, chunk_id in enumerate(chunk_ids):
            assert chunk_id == (
                f"sec-10k-aapl-2025:{index}"
            )

    def test_empty_document_list_returns_empty_list(self) -> None:
        splitter = DocumentSplitter(
            chunk_size=100,
            chunk_overlap=10,
        )

        assert splitter.split([]) == []


# ---------------------------------------------------------------------------
# IngestionPipeline tests
# ---------------------------------------------------------------------------


class TestIngestionPipeline:
    """Tests for the complete ingestion orchestration layer."""

    def test_ingests_document_successfully(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="First source page.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "source": financial_metadata.source,
                    "ticker": financial_metadata.ticker,
                },
            )
        ]

        chunks = [
            Document(
                page_content="First retrieval chunk.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "source": financial_metadata.source,
                    "chunk_id": (
                        f"{financial_metadata.document_id}:0"
                    ),
                },
            ),
            Document(
                page_content="Second retrieval chunk.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "source": financial_metadata.source,
                    "chunk_id": (
                        f"{financial_metadata.document_id}:1"
                    ),
                },
            ),
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = chunks
        vector_store.add_documents.return_value = [
            "vector-001",
            "vector-002",
        ]

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        result = pipeline.ingest_file(
            "financial_report.txt",
            metadata=financial_metadata,
        )

        assert isinstance(result, IngestionResult)

        assert result.document_id == (
            "sec-10k-aapl-2025"
        )

        assert result.source == "SEC"
        assert result.loaded_documents == 1
        assert result.generated_chunks == 2
        assert result.stored_vectors == 2
        assert result.vector_ids == (
            "vector-001",
            "vector-002",
        )

        loader.load.assert_called_once()
        splitter.split.assert_called_once_with(
            loaded_documents
        )
        vector_store.add_documents.assert_called_once_with(
            chunks
        )

    def test_pipeline_rejects_no_loaded_documents(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loader.load.return_value = []

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(
            ValueError,
            match="No readable content",
        ):
            pipeline.ingest_file(
                "empty.txt",
                metadata=financial_metadata,
            )

        splitter.split.assert_not_called()
        vector_store.add_documents.assert_not_called()

    def test_pipeline_rejects_no_generated_chunks(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = []

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(
            ValueError,
            match="No retrieval chunks",
        ):
            pipeline.ingest_file(
                "source.txt",
                metadata=financial_metadata,
            )

        vector_store.add_documents.assert_not_called()

    def test_pipeline_rejects_provenance_mismatch(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        invalid_chunks = [
            Document(
                page_content="Chunk with wrong provenance.",
                metadata={
                    "document_id": "different-document",
                    "chunk_id": "different-document:0",
                },
            )
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = invalid_chunks

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(
            ValueError,
            match="Chunk provenance mismatch",
        ):
            pipeline.ingest_file(
                "source.txt",
                metadata=financial_metadata,
            )

        vector_store.add_documents.assert_not_called()

    def test_pipeline_rejects_missing_chunk_id(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        invalid_chunks = [
            Document(
                page_content="Chunk without identifier.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = invalid_chunks

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(
            ValueError,
            match="missing its chunk_id",
        ):
            pipeline.ingest_file(
                "source.txt",
                metadata=financial_metadata,
            )

        vector_store.add_documents.assert_not_called()

    def test_pipeline_rejects_vector_count_mismatch(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        chunks = [
            Document(
                page_content="Chunk one.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "chunk_id": (
                        f"{financial_metadata.document_id}:0"
                    ),
                },
            ),
            Document(
                page_content="Chunk two.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "chunk_id": (
                        f"{financial_metadata.document_id}:1"
                    ),
                },
            ),
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = chunks

        # Simulate an incomplete vector-store write.
        vector_store.add_documents.return_value = [
            "vector-001",
        ]

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(
            RuntimeError,
            match="insertion count does not match",
        ):
            pipeline.ingest_file(
                "source.txt",
                metadata=financial_metadata,
            )

    def test_pipeline_passes_metadata_to_loader(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loaded_documents = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        chunks = [
            Document(
                page_content="Chunk content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                    "chunk_id": (
                        f"{financial_metadata.document_id}:0"
                    ),
                },
            )
        ]

        loader.load.return_value = loaded_documents
        splitter.split.return_value = chunks
        vector_store.add_documents.return_value = [
            "vector-001",
        ]

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        pipeline.ingest_file(
            Path("report.txt"),
            metadata=financial_metadata,
        )

        loader.load.assert_called_once_with(
            Path("report.txt"),
            metadata=financial_metadata,
        )

    def test_pipeline_does_not_write_when_provenance_fails(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        loader = Mock()
        splitter = Mock()
        vector_store = Mock()

        loader.load.return_value = [
            Document(
                page_content="Source content.",
                metadata={
                    "document_id": financial_metadata.document_id,
                },
            )
        ]

        splitter.split.return_value = [
            Document(
                page_content="Invalid chunk.",
                metadata={
                    "document_id": "wrong-id",
                    "chunk_id": "wrong-id:0",
                },
            )
        ]

        pipeline = IngestionPipeline(
            loader=loader,
            splitter=splitter,
            vector_store=vector_store,
        )

        with pytest.raises(ValueError):
            pipeline.ingest_file(
                "report.txt",
                metadata=financial_metadata,
            )

        vector_store.add_documents.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end ingestion behavior using real loader + splitter
# ---------------------------------------------------------------------------


class TestIngestionLoaderAndSplitterIntegration:
    """
    Lightweight integration tests for the real loader/splitter pair.

    The vector database is intentionally excluded here because database
    availability belongs to the infrastructure/integration test layer.
    """

    def test_text_file_to_chunks_preserves_provenance(
        self,
        financial_metadata: FinancialDocumentMetadata,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "sec_report.txt"

        source.write_text(
            (
                "The company is exposed to economic uncertainty. "
                "Changes in consumer demand may affect revenue. "
                "Supply chain disruptions may increase costs. "
                "Foreign exchange movements may affect reported "
                "results."
            ),
            encoding="utf-8",
        )

        loader = DocumentLoader()

        splitter = DocumentSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

        loaded_documents = loader.load(
            source,
            metadata=financial_metadata,
        )

        chunks = splitter.split(
            loaded_documents
        )

        assert loaded_documents
        assert chunks

        for chunk in chunks:
            assert chunk.page_content.strip()

            assert chunk.metadata["document_id"] == (
                financial_metadata.document_id
            )

            assert chunk.metadata["source"] == (
                financial_metadata.source
            )

            assert chunk.metadata["source_type"] == (
                financial_metadata.source_type.value
            )

            assert chunk.metadata["ticker"] == (
                financial_metadata.ticker
            )

            assert chunk.metadata["chunk_id"]

    def test_multiple_documents_get_independent_chunk_sequences(
        self,
        financial_metadata: FinancialDocumentMetadata,
    ) -> None:
        second_metadata = financial_metadata.model_copy(
            update={
                "document_id": "sec-10q-aapl-2026",
                "filing_type": "10-Q",
            }
        )

        documents = [
            create_langchain_document(
                text=(
                    "First document contains risk information. "
                    "Additional information follows here."
                ),
                metadata=financial_metadata,
            ),
            create_langchain_document(
                text=(
                    "Second document contains economic information. "
                    "Additional information follows here."
                ),
                metadata=second_metadata,
            ),
        ]

        splitter = DocumentSplitter(
            chunk_size=50,
            chunk_overlap=5,
        )

        chunks = splitter.split(documents)

        first_document_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"]
            == financial_metadata.document_id
        ]

        second_document_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"]
            == second_metadata.document_id
        ]

        assert first_document_chunks
        assert second_document_chunks

        assert (
            first_document_chunks[0].metadata["chunk_id"]
            == "sec-10k-aapl-2025:0"
        )

        assert (
            second_document_chunks[0].metadata["chunk_id"]
            == "sec-10q-aapl-2026:0"
        )