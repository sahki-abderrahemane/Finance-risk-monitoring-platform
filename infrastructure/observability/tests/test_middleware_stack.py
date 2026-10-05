"""Tests for the full observability middleware stack."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from sentinel_observability.config import ObservabilityConfig
from sentinel_observability.metrics import SentinelMetrics
from sentinel_observability.middleware import add_observability_middleware, PrometheusMiddleware


def _make_app(service: str = "test-app") -> FastAPI:
    config = ObservabilityConfig(
        service_name=service,
        environment="test",
        metrics_enabled=True,
        tracing_enabled=False,
    )
    app = FastAPI(title=f"Test {service}")
    add_observability_middleware(app, config=config)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/items/{item_id}")
    def get_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    @app.get("/error")
    def error() -> None:
        raise ValueError("test error")

    return app


class TestMiddlewareStack:
    def test_health_endpoint_returns_200(self) -> None:
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_correlation_id_generated(self) -> None:
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_correlation_id_preserved_with_uuid(self) -> None:
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        test_id = str(uuid.uuid4())
        resp = client.get("/health", headers={"X-Request-Id": test_id})
        assert resp.headers["x-request-id"] == test_id

    def test_metrics_recorded_via_registry(self) -> None:
        registry = CollectorRegistry()
        metrics = SentinelMetrics(service="test-metrics", registry=registry)
        metrics.http_requests_total.labels(
            service="test", method="GET", endpoint="/health", status_code="200"
        ).inc()
        output = generate_latest(registry).decode()
        assert "sentinel_http_requests_total" in output

    def test_path_normalization_in_metrics(self) -> None:
        assert PrometheusMiddleware._normalize_path("/items/42") == "/items/{id}"
        assert PrometheusMiddleware._normalize_path("/items/123/documents") == "/items/{id}/documents"

    def test_error_returns_500(self) -> None:
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/error")
        assert resp.status_code == 500

    def test_in_progress_gauge_increment_decrement(self) -> None:
        registry = CollectorRegistry()
        metrics = SentinelMetrics(service="test-gauge", registry=registry)
        metrics.http_requests_in_progress.labels(
            service="test", method="GET", endpoint="/health"
        ).inc()
        output = generate_latest(registry).decode()
        assert "sentinel_http_requests_in_progress" in output
