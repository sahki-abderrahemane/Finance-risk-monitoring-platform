from __future__ import annotations

from sentinel_observability.middleware import PrometheusMiddleware


class TestPrometheusMiddleware:
    def test_normalize_path_simple(self) -> None:
        assert PrometheusMiddleware._normalize_path("/health") == "/health"

    def test_normalize_path_with_id(self) -> None:
        assert PrometheusMiddleware._normalize_path("/api/v1/models/123") == "/api/v1/models/{id}"

    def test_normalize_path_with_uuid(self) -> None:
        long_id = "a" * 37
        assert PrometheusMiddleware._normalize_path(f"/api/{long_id}") == "/api/{uuid}"

    def test_normalize_path_root(self) -> None:
        assert PrometheusMiddleware._normalize_path("/") == "/"

    def test_normalize_path_mixed(self) -> None:
        assert PrometheusMiddleware._normalize_path("/api/v1/users/42/documents") == "/api/v1/users/{id}/documents"
