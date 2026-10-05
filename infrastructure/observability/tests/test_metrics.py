from __future__ import annotations

from prometheus_client import CollectorRegistry

from sentinel_observability.metrics import SentinelMetrics


class TestSentinelMetrics:
    def _make_metrics(self) -> SentinelMetrics:
        """Create metrics with an isolated registry to avoid duplicate timeseries."""
        return SentinelMetrics(service="test-service", registry=CollectorRegistry())

    def test_creates_all_metrics(self) -> None:
        metrics = self._make_metrics()
        assert metrics.service == "test-service"

        metrics.http_requests_total.labels(
            service="test", method="GET", endpoint="/health", status_code="200"
        ).inc()
        metrics.http_request_duration_seconds.labels(
            service="test", method="GET", endpoint="/health"
        ).observe(0.1)

        metrics.inference_requests_total.labels(
            service="test", model_name="gbm", model_version="1.0", status="success"
        ).inc()
        metrics.model_loaded.labels(
            service="test", model_name="gbm", model_version="1.0"
        ).set(1)

        metrics.retrieval_requests_total.labels(
            service="test", status="success"
        ).inc()
        metrics.retrieval_latency_seconds.labels(service="test").observe(0.5)

        metrics.risk_score_count.labels(
            service="test", model_name="gbm"
        ).inc()

    def test_metric_label_cardinality(self) -> None:
        metrics = self._make_metrics()
        assert "service" in metrics.http_requests_total._labelnames
        assert "method" in metrics.http_requests_total._labelnames
        assert "endpoint" in metrics.http_requests_total._labelnames
        assert "status_code" in metrics.http_requests_total._labelnames
