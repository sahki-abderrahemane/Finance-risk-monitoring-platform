"""Tests for the observability configuration."""

from __future__ import annotations

from sentinel_observability.config import ObservabilityConfig


class TestObservabilityConfig:
    def test_default_values(self) -> None:
        config = ObservabilityConfig()
        assert config.metrics_enabled is True
        assert config.log_format == "json"
        assert config.tracing_enabled is True
        assert config.metrics_port == 9091

    def test_custom_values(self) -> None:
        config = ObservabilityConfig(
            service_name="my-service",
            environment="staging",
            log_level="DEBUG",
            log_format="console",
            metrics_enabled=False,
            tracing_enabled=False,
        )
        assert config.service_name == "my-service"
        assert config.environment == "staging"
        assert config.log_level == "DEBUG"
        assert config.log_format == "console"
        assert config.metrics_enabled is False
        assert config.tracing_enabled is False

    def test_port_validation(self) -> None:
        config = ObservabilityConfig(metrics_port=8080)
        assert config.metrics_port == 8080
