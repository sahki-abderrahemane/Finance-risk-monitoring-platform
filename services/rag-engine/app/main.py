from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sentinel_observability.config import get_observability_config
from sentinel_observability.logging import configure_logging, get_logger
from sentinel_observability.metrics import SentinelMetrics
from sentinel_observability.middleware import add_observability_middleware
from sentinel_observability.tracing import configure_tracing

from app.api.ingestion_routes import (
    router as ingestion_router,
)
from app.api.routes import router as retrieval_router
from app.config import get_settings
from app.database import get_database

from app.api.explanation_routes import (
    router as explanation_router,
)
from app.observability.langsmith import (
    configure_langsmith,
)

configure_langsmith()
settings = get_settings()

config = get_observability_config()
config.service_name = "rag-engine"

configure_logging(
    service="rag-engine",
    environment=config.environment,
    log_level=config.log_level,
    log_format=config.log_format,
)

tracer_provider = configure_tracing(service_name="rag-engine")

logger = get_logger("rag-engine")

app = FastAPI(
    title="Sentinel-AI RAG Engine",
    version=settings.app_version,
    description=(
        "Grounded financial evidence ingestion and retrieval "
        "service for Sentinel-AI."
    ),
)

add_observability_middleware(app, config=config)

app.include_router(
    retrieval_router,
)

app.include_router(
    ingestion_router,
)
app.include_router(
    explanation_router,
)


@app.get(
    "/health",
    tags=["system"],
)
def health() -> dict[str, object]:
    """
    Return service and database health status.
    """
    database_healthy = get_database().health_check()

    return {
        "status": (
            "ok"
            if database_healthy
            else "degraded"
        ),
        "service": settings.app_name,
        "version": settings.app_version,
        "database": database_healthy,
    }


@app.get("/ready", tags=["system"])
def ready() -> dict[str, str]:
    """
    Readiness probe — checks database connectivity.
    """
    database_healthy = get_database().health_check()
    if not database_healthy:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "rag-engine",
                "reason": "database_unavailable",
            },
        )
    return {"status": "ready", "service": "rag-engine"}
