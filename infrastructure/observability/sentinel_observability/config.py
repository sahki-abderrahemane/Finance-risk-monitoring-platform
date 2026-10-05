from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilityConfig(BaseSettings):
    """Configuration for the Sentinel-AI observability stack."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="development")

    # Prometheus metrics
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9091, ge=1, le=65535)

    # Structured logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # OpenTelemetry tracing
    tracing_enabled: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="sentinel-ai")
    sample_rate: float = Field(default=1.0)

    # Alertmanager
    alertmanager_url: str = Field(
        default="http://localhost:9093",
    )


@lru_cache(maxsize=1)
def get_observability_config() -> ObservabilityConfig:
    return ObservabilityConfig()
