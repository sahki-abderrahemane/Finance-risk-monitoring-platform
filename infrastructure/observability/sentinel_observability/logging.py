from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


_SENSITIVE_KEYS = frozenset({
    "password",
    "postgres_password",
    "secret",
    "api_key",
    "token",
    "authorization",
    "credentials",
    "langsmith_api_key",
    "access_token",
    "private_key",
})


def _redact_secrets(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that redacts sensitive values."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            value = event_dict[key]
            if isinstance(value, str) and len(value) > 0:
                event_dict[key] = "***REDACTED***"
            elif value is not None:
                event_dict[key] = "***REDACTED***"
    return event_dict


def _add_service_context(
    service: str, environment: str
) -> structlog.types.Processor:
    """Return a processor that injects service and environment."""

    def processor(
        logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        event_dict["service"] = service
        event_dict["environment"] = environment
        return event_dict

    return processor


def configure_logging(
    service: str = "sentinel-ai",
    environment: str = "development",
    log_level: str = "INFO",
    log_format: str = "json",
) -> None:
    """Configure structured JSON logging for a Sentinel-AI service.

    Parameters
    ----------
    service:
        The service name included in every log line.
    environment:
        The deployment environment (development, staging, production).
    log_level:
        Python logging level name.
    log_format:
        "json" for machine-readable logs, "console" for human-readable.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_service_context(service, environment),
        _redact_secrets,
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for noisy_logger in ("uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
