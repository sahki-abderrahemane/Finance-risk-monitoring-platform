from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

from sentinel_observability.config import get_observability_config


def configure_tracing(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    sample_rate: float | None = None,
) -> TracerProvider | None:
    """Configure OpenTelemetry distributed tracing.

    Returns the TracerProvider if tracing is enabled, None otherwise.
    The provider is set as the global tracer provider.
    """
    config = get_observability_config()

    if not config.tracing_enabled:
        return None

    _service_name = service_name or config.service_name
    _otlp_endpoint = otlp_endpoint or config.otlp_endpoint
    _sample_rate = sample_rate if sample_rate is not None else config.sample_rate

    resource = Resource.create(
        {
            SERVICE_NAME: _service_name,
            "deployment.environment": config.environment,
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=_otlp_endpoint,
        insecure=True,
    )

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    return provider


def get_tracer(name: str = "sentinel-ai") -> trace.Tracer:
    """Return the configured tracer."""
    return trace.get_tracer(name)


def get_propagator() -> TraceContextTextMapPropagator:
    """Return the W3C Trace Context propagator."""
    return TraceContextTextMapPropagator()
