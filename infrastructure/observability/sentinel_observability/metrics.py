from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, Info, Gauge


class SentinelMetrics:
    """Prometheus metrics for Sentinel-AI services.

    All metrics use low-cardinality labels. High-cardinality values
    such as request_id, trace_id, user_id belong in logs/traces only.
    """

    def __init__(
        self, service: str, registry: CollectorRegistry | None = None
    ) -> None:
        self.service = service
        r = registry

        # ── HTTP request metrics ──────────────────────────────────
        self.http_requests_total = Counter(
            name="sentinel_http_requests_total",
            documentation="Total HTTP requests",
            labelnames=["service", "method", "endpoint", "status_code"],
            registry=r,
        )

        self.http_request_duration_seconds = Histogram(
            name="sentinel_http_request_duration_seconds",
            documentation="HTTP request latency in seconds",
            labelnames=["service", "method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=r,
        )

        self.http_requests_in_progress = Gauge(
            name="sentinel_http_requests_in_progress",
            documentation="Number of HTTP requests currently being processed",
            labelnames=["service", "method", "endpoint"],
            registry=r,
        )

        self.http_request_size_bytes = Histogram(
            name="sentinel_http_request_size_bytes",
            documentation="HTTP request body size in bytes",
            labelnames=["service", "method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=r,
        )

        self.http_response_size_bytes = Histogram(
            name="sentinel_http_response_size_bytes",
            documentation="HTTP response body size in bytes",
            labelnames=["service", "method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=r,
        )

        # ── ML inference metrics ──────────────────────────────────
        self.inference_requests_total = Counter(
            name="sentinel_inference_requests_total",
            documentation="Total ML inference requests",
            labelnames=["service", "model_name", "model_version", "status"],
            registry=r,
        )

        self.inference_latency_seconds = Histogram(
            name="sentinel_inference_latency_seconds",
            documentation="ML inference latency in seconds",
            labelnames=["service", "model_name", "model_version"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=r,
        )

        self.model_loaded = Gauge(
            name="sentinel_model_loaded",
            documentation="Whether a model is currently loaded (1=yes, 0=no)",
            labelnames=["service", "model_name", "model_version"],
            registry=r,
        )

        self.model_load_failures_total = Counter(
            name="sentinel_model_load_failures_total",
            documentation="Total model load failures",
            labelnames=["service", "model_name"],
            registry=r,
        )

        # ── Data quality metrics ──────────────────────────────────
        self.input_records_total = Counter(
            name="sentinel_input_records_total",
            documentation="Total input records processed",
            labelnames=["service", "source"],
            registry=r,
        )

        self.data_quality_failures_total = Counter(
            name="sentinel_data_quality_failures_total",
            documentation="Total data quality failures",
            labelnames=["service", "failure_type"],
            registry=r,
        )

        self.missing_feature_rate = Gauge(
            name="sentinel_missing_feature_rate",
            documentation="Rate of missing features in current batch",
            labelnames=["service", "feature_name"],
            registry=r,
        )

        # ── Risk model metrics ────────────────────────────────────
        self.risk_score_distribution = Histogram(
            name="sentinel_risk_score_distribution",
            documentation="Distribution of risk scores produced",
            labelnames=["service", "model_name"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=r,
        )

        self.risk_score_count = Counter(
            name="sentinel_risk_score_count",
            documentation="Total risk scores produced",
            labelnames=["service", "model_name"],
            registry=r,
        )

        # ── RAG / Retrieval metrics ───────────────────────────────
        self.retrieval_requests_total = Counter(
            name="sentinel_retrieval_requests_total",
            documentation="Total retrieval requests",
            labelnames=["service", "status"],
            registry=r,
        )

        self.retrieval_latency_seconds = Histogram(
            name="sentinel_retrieval_latency_seconds",
            documentation="Retrieval latency in seconds",
            labelnames=["service"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=r,
        )

        self.retrieved_documents_count = Histogram(
            name="sentinel_retrieved_documents_count",
            documentation="Number of documents returned per retrieval",
            labelnames=["service"],
            buckets=(0, 1, 2, 5, 10, 20, 50),
            registry=r,
        )

        self.retrieval_empty_results_total = Counter(
            name="sentinel_retrieval_empty_results_total",
            documentation="Total retrieval requests returning zero results",
            labelnames=["service"],
            registry=r,
        )

        # ── Explanation metrics ───────────────────────────────────
        self.explanation_requests_total = Counter(
            name="sentinel_explanation_requests_total",
            documentation="Total explanation requests",
            labelnames=["service", "status"],
            registry=r,
        )

        self.explanation_latency_seconds = Histogram(
            name="sentinel_explanation_latency_seconds",
            documentation="Explanation generation latency in seconds",
            labelnames=["service"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=r,
        )

        self.explanation_validation_failures_total = Counter(
            name="sentinel_explanation_validation_failures_total",
            documentation="Total explanation validation failures",
            labelnames=["service", "failure_reason"],
            registry=r,
        )

        self.explanation_groundedness = Gauge(
            name="sentinel_explanation_groundedness",
            documentation="Groundedness score of last explanation (0-1)",
            labelnames=["service"],
            registry=r,
        )

        # ── Service info ──────────────────────────────────────────
        self.service_info = Info(
            name="sentinel_service",
            documentation="Sentinel-AI service metadata",
            registry=r,
        )
