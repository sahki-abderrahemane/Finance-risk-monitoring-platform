from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


class DocumentSplitter:
    """
    Split source documents into retrieval-ready LangChain Documents.

    RecursiveCharacterTextSplitter is used instead of a custom word
    slicing implementation because it attempts to preserve natural
    textual boundaries before falling back to smaller separators.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        resolved_chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunk_size
        )

        resolved_chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        if resolved_chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if resolved_chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if resolved_chunk_overlap >= resolved_chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                ", ",
                " ",
                "",
            ],
        )

    def split(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Split source documents while preserving their metadata.

        LangChain copies the metadata from the parent document onto
        each generated chunk.
        """

        if not documents:
            return []

        chunks = self.splitter.split_documents(
            list(documents)
        )

        return self._assign_chunk_ids(chunks)

    @staticmethod
    def _assign_chunk_ids(
        chunks: Sequence[Document],
    ) -> list[Document]:
        """
        Add deterministic chunk identifiers to the metadata.

        Chunk IDs are generated per source document rather than globally,
        allowing provenance to remain tied to the original document.
        """

        counters: dict[str, int] = {}
        result: list[Document] = []

        for chunk in chunks:
            metadata = dict(chunk.metadata)

            document_id = str(
                metadata.get("document_id", "unknown")
            )

            chunk_number = counters.get(document_id, 0)
            counters[document_id] = chunk_number + 1

            metadata["chunk_id"] = (
                f"{document_id}:{chunk_number}"
            )

            result.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        return result