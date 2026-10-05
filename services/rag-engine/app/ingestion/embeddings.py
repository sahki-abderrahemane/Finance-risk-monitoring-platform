from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import get_settings


class EmbeddingProvider:

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()

        resolved_model_name = (
            model_name
            if model_name is not None
            else settings.embedding_model
        )

        resolved_device = (
            device
            if device is not None
            else settings.embedding_device
        )

        resolved_batch_size = (
            batch_size
            if batch_size is not None
            else settings.embedding_batch_size
        )

        if not resolved_model_name.strip():
            raise ValueError(
                "Embedding model name cannot be empty."
            )

        if not resolved_device.strip():
            raise ValueError(
                "Embedding device cannot be empty."
            )

        if resolved_batch_size <= 0:
            raise ValueError(
                "Embedding batch size must be greater than zero."
            )

        self.model_name = resolved_model_name
        self.device = resolved_device
        self.batch_size = resolved_batch_size

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.batch_size,
            },
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        

        return self._embeddings

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
       

        if not texts:
            return []

        normalized_texts = [
            self._validate_text(text)
            for text in texts
        ]

        return self._embeddings.embed_documents(
            normalized_texts
        )

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate a normalized embedding for a retrieval query.
        """

        normalized_query = self._validate_text(query)

        return self._embeddings.embed_query(
            normalized_query
        )

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize text before embedding.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Embedding input must be a string."
            )

        normalized = text.strip()

        if not normalized:
            raise ValueError(
                "Embedding input cannot be empty."
            )

        return normalized


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    
    return EmbeddingProvider()