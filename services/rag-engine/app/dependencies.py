from __future__ import annotations

from functools import lru_cache

from app.retrieval.retriever import Retriever
from app.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    

    return VectorStore()


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    

    return Retriever(
        vector_store=get_vector_store(),
    )