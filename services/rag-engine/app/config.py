from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime configuration for the Sentinel-AI RAG engine.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "sentinel-rag-engine"
    app_version: str = "0.1.0"
    environment: str = "development"

    host: str = "0.0.0.0"
    port: int = Field(default=8003, ge=1, le=65535)

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_database: str = "sentinel"
    postgres_user: str = "sentinel"
    postgres_password: str = ""

    @property
    def postgres_dsn(self) -> str:
        return (
            "postgresql+psycopg://"
            f"{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_database}"
        )

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
    )

    chunk_size: int = Field(
        default=512,
        ge=1,
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
    )

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
    )
    retrieval_max_distance: float = Field(
        default=1.0,
        ge=0.0,
    )

    max_upload_size_mb: int = Field(
        default=25,
        ge=1,
        le=500,
    )

    allowed_document_extensions: tuple[str, ...] = (
        ".pdf",
        ".txt",
    )

    llm_provider: str = "none"
    llm_model: str = ""
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    langsmith_tracing: bool = False
    langsmith_project: str = "sentinel-ai-rag"
    langsmith_endpoint: str = (
        "https://api.smith.langchain.com"
    )
    langsmith_api_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide settings instance."""

    return Settings()