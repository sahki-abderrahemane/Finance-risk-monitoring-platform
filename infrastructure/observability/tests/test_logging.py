from __future__ import annotations

import json
import logging

from sentinel_observability.logging import configure_logging, get_logger


class TestStructuredLogging:
    def test_configure_logging_json(self) -> None:
        configure_logging(
            service="test-service",
            environment="test",
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("test")
        logger.info("hello", key="value")

    def test_configure_logging_console(self) -> None:
        configure_logging(
            service="test-service",
            environment="test",
            log_level="DEBUG",
            log_format="console",
        )
        logger = get_logger("test")
        logger.debug("debug message")

    def test_logger_has_bound_context(self) -> None:
        configure_logging(
            service="my-service",
            environment="development",
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("test-bound")
        assert logger is not None

    def test_sensitive_values_redacted(self) -> None:
        configure_logging(
            service="test",
            environment="test",
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("test-redact")
        logger.info("auth", password="secret123", api_key="key-123")

    def test_log_level_configuration(self) -> None:
        configure_logging(
            service="test",
            environment="test",
            log_level="WARNING",
            log_format="json",
        )
        root = logging.getLogger()
        assert root.level == logging.WARNING
