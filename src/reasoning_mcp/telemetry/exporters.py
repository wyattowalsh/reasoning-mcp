"""Exporter configuration helpers for OpenTelemetry.

This module provides utilities for configuring different span exporters
based on user settings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings

logger = logging.getLogger(__name__)

# Track availability
_EXPORTERS_AVAILABLE = False

try:
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        SpanExporter,
        SpanProcessor,
    )

    _EXPORTERS_AVAILABLE = True
except ImportError:
    SpanExporter = None  # type: ignore[assignment, misc]
    SpanProcessor = None  # type: ignore[assignment, misc]
    BatchSpanProcessor = None  # type: ignore[assignment, misc]
    SimpleSpanProcessor = None  # type: ignore[assignment, misc]
    ConsoleSpanExporter = None  # type: ignore[assignment, misc]


def create_console_exporter() -> tuple[Any, Any] | None:
    """Create a console span exporter with simple processor.

    Returns:
        Tuple of (exporter, processor) or None if unavailable.
    """
    if not _EXPORTERS_AVAILABLE:
        return None

    exporter = ConsoleSpanExporter()
    processor = SimpleSpanProcessor(exporter)
    return exporter, processor


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    timeout_seconds: int = 30,
    use_grpc: bool = True,
) -> tuple[Any, Any] | None:
    """Create an OTLP span exporter with batch processor.

    Args:
        endpoint: OTLP collector endpoint.
        timeout_seconds: Export timeout in seconds.
        use_grpc: Use gRPC transport (True) or HTTP (False).

    Returns:
        Tuple of (exporter, processor) or None if unavailable.
    """
    if not _EXPORTERS_AVAILABLE:
        return None

    try:
        if use_grpc:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            timeout=timeout_seconds,
        )
        processor = BatchSpanProcessor(exporter)
        return exporter, processor

    except ImportError as e:
        logger.warning(f"OTLP exporter not available: {e}")
        return None


def create_jaeger_exporter(
    agent_host: str = "localhost",
    agent_port: int = 6831,
) -> tuple[Any, Any] | None:
    """Create a Jaeger span exporter with batch processor.

    Args:
        agent_host: Jaeger agent hostname.
        agent_port: Jaeger agent UDP port.

    Returns:
        Tuple of (exporter, processor) or None if unavailable.
    """
    if not _EXPORTERS_AVAILABLE:
        return None

    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        exporter = JaegerExporter(
            agent_host_name=agent_host,
            agent_port=agent_port,
        )
        processor = BatchSpanProcessor(exporter)
        return exporter, processor

    except ImportError:
        logger.warning(
            "Jaeger exporter not available. "
            "Install opentelemetry-exporter-jaeger for Jaeger support."
        )
        return None


def create_zipkin_exporter(
    endpoint: str = "http://localhost:9411/api/v2/spans",
) -> tuple[Any, Any] | None:
    """Create a Zipkin span exporter with batch processor.

    Args:
        endpoint: Zipkin collector endpoint.

    Returns:
        Tuple of (exporter, processor) or None if unavailable.
    """
    if not _EXPORTERS_AVAILABLE:
        return None

    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter

        exporter = ZipkinExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        return exporter, processor

    except ImportError:
        logger.warning(
            "Zipkin exporter not available. "
            "Install opentelemetry-exporter-zipkin for Zipkin support."
        )
        return None


def create_exporter_from_settings(settings: Settings) -> tuple[Any, Any] | None:
    """Create appropriate exporter based on settings.

    Args:
        settings: Server settings with telemetry configuration.

    Returns:
        Tuple of (exporter, processor) or None if unavailable.
    """
    exporter_type = getattr(settings, "telemetry_exporter", "otlp")

    if exporter_type == "console":
        return create_console_exporter()

    elif exporter_type == "otlp":
        endpoint = getattr(settings, "telemetry_otlp_endpoint", "http://localhost:4317")
        timeout = getattr(settings, "telemetry_export_timeout_ms", 30000) // 1000
        return create_otlp_exporter(endpoint=endpoint, timeout_seconds=timeout)

    elif exporter_type == "jaeger":
        host = getattr(settings, "telemetry_jaeger_host", "localhost")
        port = getattr(settings, "telemetry_jaeger_port", 6831)
        return create_jaeger_exporter(agent_host=host, agent_port=port)

    elif exporter_type == "zipkin":
        endpoint = getattr(
            settings, "telemetry_zipkin_endpoint", "http://localhost:9411/api/v2/spans"
        )
        return create_zipkin_exporter(endpoint=endpoint)

    elif exporter_type == "none":
        return None

    else:
        logger.warning(f"Unknown exporter type: {exporter_type}, using console")
        return create_console_exporter()


__all__ = [
    "create_console_exporter",
    "create_otlp_exporter",
    "create_jaeger_exporter",
    "create_zipkin_exporter",
    "create_exporter_from_settings",
]
