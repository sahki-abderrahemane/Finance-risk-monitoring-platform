from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from app.ingestion.models import (
    FinancialDocumentMetadata,
    create_langchain_document,
)


class DocumentLoader:
    """
    Load supported financial source documents into LangChain Documents.

    This loader is intentionally responsible only for reading source
    content and attaching Sentinel provenance metadata.

    Chunking, embeddings, and vector storage are separate concerns.
    """

    SUPPORTED_SUFFIXES = {
        ".txt",
        ".pdf",
    }

    def load(
        self,
        path: str | Path,
        *,
        metadata: FinancialDocumentMetadata,
    ) -> list[Document]:
        """
        Load a source document and return LangChain Documents.

        PDF files are loaded page-by-page so page provenance can be
        preserved. Text files are loaded as a single source document.
        """

        file_path = Path(path)

        self._validate_path(file_path)

        if file_path.suffix.lower() == ".pdf":
            return self._load_pdf(
                file_path,
                metadata=metadata,
            )

        if file_path.suffix.lower() == ".txt":
            return self._load_text(
                file_path,
                metadata=metadata,
            )

        raise ValueError(
            f"Unsupported document type: {file_path.suffix}"
        )

    def _load_pdf(
        self,
        path: Path,
        *,
        metadata: FinancialDocumentMetadata,
    ) -> list[Document]:
        """Load a PDF while preserving page-level provenance."""

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        documents: list[Document] = []

        for page_index, page in enumerate(pages, start=1):
            page_metadata = metadata.model_copy(
                update={"page": page_index}
            )

            documents.append(
                create_langchain_document(
                    text=page.page_content,
                    metadata=page_metadata,
                )
            )

        return documents

    def _load_text(
        self,
        path: Path,
        *,
        metadata: FinancialDocumentMetadata,
    ) -> list[Document]:
        """Load a UTF-8 text document."""

        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

        loaded_documents = loader.load()

        return [
            create_langchain_document(
                text=document.page_content,
                metadata=metadata,
            )
            for document in loaded_documents
        ]

    def _validate_path(self, path: Path) -> None:
        """Validate the source file before handing it to LangChain."""

        if not path.exists():
            raise FileNotFoundError(
                f"Document does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Document path is not a file: {path}"
            )

        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported document type: {path.suffix}. "
                f"Supported types: {sorted(self.SUPPORTED_SUFFIXES)}"
            )