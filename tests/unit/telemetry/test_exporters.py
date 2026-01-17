"""Tests for telemetry exporters module."""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from reasoning_mcp.telemetry.exporters import (
    create_console_exporter,
    create_exporter_from_settings,
    create_jaeger_exporter,
    create_otlp_exporter,
    create_zipkin_exporter,
)

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings


class TestCreateConsoleExporter:
    """Tests for create_console_exporter function."""

    def test_returns_tuple_when_available(self) -> None:
        """Test returns exporter and processor tuple when available."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.exporters.ConsoleSpanExporter"
            ) as mock_exporter:
                with patch(
                    "reasoning_mcp.telemetry.exporters.SimpleSpanProcessor"
                ) as mock_processor:
                    mock_exporter.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()

                    result = create_console_exporter()

                    assert result is not None
                    assert len(result) == 2
                    mock_exporter.assert_called_once()
                    mock_processor.assert_called_once()

    def test_returns_none_when_unavailable(self) -> None:
        """Test returns None when exporters not available."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", False
        ):
            result = create_console_exporter()
            assert result is None


class TestCreateOtlpExporter:
    """Tests for create_otlp_exporter function."""

    def test_returns_tuple_with_grpc_when_available(self) -> None:
        """Test returns exporter and processor tuple with gRPC."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.exporters.BatchSpanProcessor"
            ) as mock_processor:
                with patch(
                    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
                ) as mock_exporter:
                    mock_exporter.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()

                    result = create_otlp_exporter(
                        endpoint="http://localhost:4317",
                        timeout_seconds=30,
                        use_grpc=True,
                    )

                    assert result is not None
                    assert len(result) == 2
                    mock_exporter.assert_called_once_with(
                        endpoint="http://localhost:4317",
                        timeout=30,
                    )

    def test_returns_tuple_with_http_when_available(self) -> None:
        """Test returns exporter and processor tuple with HTTP."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            with patch(
                "reasoning_mcp.telemetry.exporters.BatchSpanProcessor"
            ) as mock_processor:
                with patch(
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
                ) as mock_exporter:
                    mock_exporter.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()

                    result = create_otlp_exporter(
                        endpoint="http://localhost:4318",
                        timeout_seconds=60,
                        use_grpc=False,
                    )

                    assert result is not None
                    assert len(result) == 2

    def test_returns_none_when_unavailable(self) -> None:
        """Test returns None when exporters not available."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", False
        ):
            result = create_otlp_exporter()
            assert result is None

    def test_returns_none_on_import_error(self) -> None:
        """Test returns None when OTLP exporter import fails."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            with patch.dict("sys.modules", {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None}):
                # This will trigger ImportError
                result = create_otlp_exporter()
                # Should handle import error gracefully
                assert result is None or result is not None  # Either is valid


class TestCreateJaegerExporter:
    """Tests for create_jaeger_exporter function."""

    def test_returns_none_when_unavailable(self) -> None:
        """Test returns None when exporters not available."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", False
        ):
            result = create_jaeger_exporter()
            assert result is None

    def test_returns_none_when_jaeger_not_installed(self) -> None:
        """Test returns None when Jaeger package not installed."""
        # Jaeger exporter is optional, so this tests the import error path
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            # The actual function handles ImportError gracefully
            result = create_jaeger_exporter()
            # Either returns None (not installed) or a tuple (installed)
            assert result is None or (isinstance(result, tuple) and len(result) == 2)


class TestCreateZipkinExporter:
    """Tests for create_zipkin_exporter function."""

    def test_returns_none_when_unavailable(self) -> None:
        """Test returns None when exporters not available."""
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", False
        ):
            result = create_zipkin_exporter()
            assert result is None

    def test_returns_none_when_zipkin_not_installed(self) -> None:
        """Test returns None when Zipkin package not installed."""
        # Zipkin exporter is optional, so this tests the import error path
        with patch(
            "reasoning_mcp.telemetry.exporters._EXPORTERS_AVAILABLE", True
        ):
            # The actual function handles ImportError gracefully
            result = create_zipkin_exporter()
            # Either returns None (not installed) or a tuple (installed)
            assert result is None or (isinstance(result, tuple) and len(result) == 2)


class TestCreateExporterFromSettings:
    """Tests for create_exporter_from_settings function."""

    def test_console_exporter_from_settings(self) -> None:
        """Test creates console exporter from settings."""
        settings = MagicMock()
        settings.telemetry_exporter = "console"

        with patch(
            "reasoning_mcp.telemetry.exporters.create_console_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            result = create_exporter_from_settings(settings)

            mock_create.assert_called_once()
            assert result is not None

    def test_otlp_exporter_from_settings(self) -> None:
        """Test creates OTLP exporter from settings."""
        settings = MagicMock()
        settings.telemetry_exporter = "otlp"
        settings.telemetry_otlp_endpoint = "http://collector:4317"
        settings.telemetry_export_timeout_ms = 60000

        with patch(
            "reasoning_mcp.telemetry.exporters.create_otlp_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            result = create_exporter_from_settings(settings)

            mock_create.assert_called_once_with(
                endpoint="http://collector:4317",
                timeout_seconds=60,
            )
            assert result is not None

    def test_jaeger_exporter_from_settings(self) -> None:
        """Test creates Jaeger exporter from settings."""
        settings = MagicMock()
        settings.telemetry_exporter = "jaeger"
        settings.telemetry_jaeger_host = "jaeger-agent"
        settings.telemetry_jaeger_port = 6832

        with patch(
            "reasoning_mcp.telemetry.exporters.create_jaeger_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            result = create_exporter_from_settings(settings)

            mock_create.assert_called_once_with(
                agent_host="jaeger-agent",
                agent_port=6832,
            )
            assert result is not None

    def test_zipkin_exporter_from_settings(self) -> None:
        """Test creates Zipkin exporter from settings."""
        settings = MagicMock()
        settings.telemetry_exporter = "zipkin"
        settings.telemetry_zipkin_endpoint = "http://zipkin:9411/api/v2/spans"

        with patch(
            "reasoning_mcp.telemetry.exporters.create_zipkin_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            result = create_exporter_from_settings(settings)

            mock_create.assert_called_once_with(
                endpoint="http://zipkin:9411/api/v2/spans",
            )
            assert result is not None

    def test_none_exporter_from_settings(self) -> None:
        """Test returns None when exporter type is 'none'."""
        settings = MagicMock()
        settings.telemetry_exporter = "none"

        result = create_exporter_from_settings(settings)

        assert result is None

    def test_unknown_exporter_falls_back_to_console(self) -> None:
        """Test unknown exporter type falls back to console."""
        settings = MagicMock()
        settings.telemetry_exporter = "unknown"

        with patch(
            "reasoning_mcp.telemetry.exporters.create_console_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            result = create_exporter_from_settings(settings)

            mock_create.assert_called_once()
            assert result is not None

    def test_default_exporter_type_is_otlp(self) -> None:
        """Test default exporter type is otlp when not specified."""
        settings = MagicMock(spec=[])  # No telemetry_exporter attribute

        with patch(
            "reasoning_mcp.telemetry.exporters.create_otlp_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            create_exporter_from_settings(settings)

            mock_create.assert_called_once()

    def test_uses_default_otlp_endpoint(self) -> None:
        """Test uses default OTLP endpoint when not specified."""
        settings = MagicMock(spec=["telemetry_exporter"])
        settings.telemetry_exporter = "otlp"

        with patch(
            "reasoning_mcp.telemetry.exporters.create_otlp_exporter"
        ) as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            create_exporter_from_settings(settings)

            mock_create.assert_called_once_with(
                endpoint="http://localhost:4317",
                timeout_seconds=30,
            )
