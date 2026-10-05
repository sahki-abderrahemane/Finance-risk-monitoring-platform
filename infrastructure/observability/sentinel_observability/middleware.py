from __future__ import annotations

import time
from typing import Any

import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from sentinel_observability.config import ObservabilityConfig
from sentinel_observability.metrics import SentinelMetrics


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records Prometheus metrics for every HTTP request."""

    def __init__(self, app: Any, metrics: SentinelMetrics) -> None:
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path = request.url.path

        # Normalize path to avoid high-cardinality labels.
        endpoint = self._normalize_path(path)

        self.metrics.http_requests_in_progress.labels(
            service=self.metrics.service,
            method=method,
            endpoint=endpoint,
        ).inc()

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start_time
            self.metrics.http_requests_total.labels(
                service=self.metrics.service,
                method=method,
                endpoint=endpoint,
                status_code="500",
            ).inc()
            self.metrics.http_request_duration_seconds.labels(
                service=self.metrics.service,
                method=method,
                endpoint=endpoint,
            ).observe(duration)
            raise
        finally:
            self.metrics.http_requests_in_progress.labels(
                service=self.metrics.service,
                method=method,
                endpoint=endpoint,
            ).dec()

        duration = time.perf_counter() - start_time
        status_code = str(response.status_code)

        self.metrics.http_requests_total.labels(
            service=self.metrics.service,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        self.metrics.http_request_duration_seconds.labels(
            service=self.metrics.service,
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize URL paths to prevent high-cardinality labels.

        Replaces UUIDs, numeric IDs, and other variable path segments
        with placeholders.
        """
        segments = path.strip("/").split("/")
        normalized: list[str] = []
        for segment in segments:
            if segment.isdigit():
                normalized.append("{id}")
            elif len(segment) > 36:
                normalized.append("{uuid}")
            else:
                normalized.append(segment)
        return "/" + "/".join(normalized) if normalized else "/"


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that emits structured JSON logs for every request."""

    def __init__(self, app: Any, service: str) -> None:
        super().__init__(app)
        self.service = service
        self.logger = structlog.get_logger("http")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(
            "X-Request-Id", ""
        )
        trace_id = request.headers.get(
            "traceparent", ""
        )

        log_context: dict[str, Any] = {
            "request_id": request_id,
            "trace_id": trace_id,
            "method": request.method,
            "endpoint": request.url.path,
            "client": request.client.host if request.client else "unknown",
        }

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                "request_failed",
                duration_ms=round(duration_ms, 2),
                status_code=500,
                error_type=type(exc).__name__,
                **log_context,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_fn = (
                self.logger.warning
                if response.status_code >= 400
                else self.logger.info
            )
            log_fn(
                "request_completed",
                duration_ms=round(duration_ms, 2),
                status_code=response.status_code,
                **log_context,
            )

        return response


def add_observability_middleware(
    app: FastAPI,
    config: ObservabilityConfig | None = None,
    metrics: SentinelMetrics | None = None,
) -> None:
    """Add all observability middleware to a FastAPI application.

    This is the single entrypoint for instrumenting a service.
    """
    from sentinel_observability.config import get_observability_config

    if config is None:
        config = get_observability_config()

    if metrics is None:
        metrics = SentinelMetrics(service=config.service_name)

    # 1. CORS (must be outermost).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Correlation IDs (generates X-Request-Id if absent).
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name="X-Request-Id",
    )

    # 3. Structured logging.
    app.add_middleware(
        StructuredLoggingMiddleware,
        service=config.service_name,
    )

    # 4. Prometheus metrics.
    app.add_middleware(
        PrometheusMiddleware,
        metrics=metrics,
    )

    # 5. Mount /metrics endpoint for Prometheus scraping.
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Store metrics on app state for access in route handlers.
    app.state.metrics = metrics
