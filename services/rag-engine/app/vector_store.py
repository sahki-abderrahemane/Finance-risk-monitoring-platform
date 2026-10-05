from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_postgres import PGVector

from app.config import get_settings


class VectorStore:
    """
    Sentinel-AI vector-store adapter.

    LangChain owns the vector-store interaction while this class defines
    the application-level contract used by the Sentinel RAG service.
    """

    def __init__(
        self,
        *,
        connection: str | None = None,
        collection_name: str = "sentinel_documents",
    ) -> None:
        settings = get_settings()

        self.connection = (
            connection or settings.postgres_dsn
        )
        self.collection_name = collection_name

        self.store = PGVector(
            embeddings=self._create_embedding_function(),
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
        )

    @staticmethod
    def _create_embedding_function() -> Any:
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        settings = get_settings()

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": settings.embedding_device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.embedding_batch_size,
            },
        )

    def add_documents(
        self,
        documents: Sequence[Document],
    ) -> list[str]:
        if not documents:
            return []

        ids = [
            self._require_chunk_id(document)
            for document in documents
        ]

        return self.store.add_documents(
            list(documents),
            ids=ids,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not texts:
            return []

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                "metadatas must contain one entry for every text."
            )

        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                "ids must contain one entry for every text."
            )

        return self.store.add_texts(
            list(texts),
            metadatas=(
                list(metadatas)
                if metadatas is not None
                else None
            ),
            ids=list(ids) if ids is not None else None,
        )

    def similarity_search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        settings = get_settings()

        result_count = (
            k
            if k is not None
            else settings.retrieval_top_k
        )

        if result_count <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        return self.store.similarity_search(
            query,
            k=result_count,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        settings = get_settings()

        result_count = (
            k
            if k is not None
            else settings.retrieval_top_k
        )

        if result_count <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        return self.store.similarity_search_with_score(
            query,
            k=result_count,
            filter=filter,
        )

    @staticmethod
    def _require_chunk_id(
        document: Document,
    ) -> str:
        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if not isinstance(chunk_id, str):
            raise ValueError(
                "Every document must contain a string "
                "'chunk_id' before vector storage."
            )

        chunk_id = chunk_id.strip()

        if not chunk_id:
            raise ValueError(
                "Document chunk_id cannot be empty."
            )

        return chunk_id