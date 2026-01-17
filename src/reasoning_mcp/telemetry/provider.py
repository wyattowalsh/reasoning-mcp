"""TracerProvider setup and management for reasoning-mcp.

This module handles initialization and shutdown of OpenTelemetry
tracing. It gracefully handles missing dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings

logger = logging.getLogger(__name__)

# Track whether OpenTelemetry is available
_OTEL_AVAILABLE = False
_tracer_provider: Any = None
_initialized = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.semconv.resource import ResourceAttributes

    _OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment, misc]
    Resource = None  # type: ignore[assignment, misc]
    ResourceAttributes = None  # type: ignore[assignment, misc]
    BatchSpanProcessor = None  # type: ignore[assignment, misc]
    SimpleSpanProcessor = None  # type: ignore[assignment, misc]
    ConsoleSpanExporter = None  # type: ignore[assignment, misc]


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs: Any) -> NoOpSpan:
        """Return a no-op span."""
        return NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> NoOpSpanContext:
        """Return a no-op context manager."""
        return NoOpSpanContext()


class NoOpSpan:
    """No-op span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: BaseException) -> None:
        """No-op."""
        pass

    def end(self) -> None:
        """No-op."""
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass


class NoOpSpanContext:
    """No-op context manager for spans."""

    def __enter__(self) -> NoOpSpan:
        return NoOpSpan()

    def __exit__(self, *args: Any) -> None:
        pass


_noop_tracer = NoOpTracer()


def is_telemetry_available() -> bool:
    """Check if OpenTelemetry packages are installed.

    Returns:
        True if OpenTelemetry is available, False otherwise.
    """
    return _OTEL_AVAILABLE


def init_telemetry(settings: Settings) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        settings: Server settings containing telemetry configuration.

    Returns:
        True if telemetry was initialized, False if disabled or unavailable.
    """
    global _tracer_provider, _initialized

    if _initialized:
        logger.debug("Telemetry already initialized")
        return True

    if not getattr(settings, "enable_telemetry", False):
        logger.info("Telemetry disabled in settings")
        return False

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: pip install reasoning-mcp[observability]"
        )
        return False

    try:
        # Create resource with service info
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: settings.telemetry_service_name,
                ResourceAttributes.SERVICE_VERSION: settings.server_version,
            }
        )

        # Create TracerProvider
        _tracer_provider = TracerProvider(resource=resource)

        # Configure exporter based on settings
        exporter_type = getattr(settings, "telemetry_exporter", "otlp")

        if exporter_type == "console":
            # Console exporter for development
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            _tracer_provider.add_span_processor(processor)
            logger.info("Telemetry initialized with console exporter")

        elif exporter_type == "otlp":
            # OTLP exporter for production
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                endpoint = getattr(settings, "telemetry_otlp_endpoint", "http://localhost:4317")
                timeout = getattr(settings, "telemetry_export_timeout_ms", 30000)

                exporter = OTLPSpanExporter(
                    endpoint=endpoint,
                    timeout=timeout // 1000,  # Convert to seconds
                )
                processor = BatchSpanProcessor(exporter)
                _tracer_provider.add_span_processor(processor)
                logger.info(f"Telemetry initialized with OTLP exporter to {endpoint}")

            except ImportError:
                logger.warning(
                    "OTLP exporter not available, falling back to console. "
                    "Install opentelemetry-exporter-otlp for OTLP support."
                )
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                _tracer_provider.add_span_processor(processor)

        elif exporter_type == "none":
            logger.info("Telemetry initialized with no exporter (spans collected but not exported)")

        else:
            logger.warning(f"Unknown exporter type: {exporter_type}, using console")
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            _tracer_provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        _initialized = True

        logger.info("OpenTelemetry telemetry initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        return False


def shutdown_telemetry() -> None:
    """Shutdown OpenTelemetry tracing.

    Flushes any pending spans and releases resources.
    """
    global _tracer_provider, _initialized

    if not _initialized or _tracer_provider is None:
        return

    try:
        _tracer_provider.shutdown()
        logger.info("Telemetry shutdown complete")
    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")
    finally:
        _tracer_provider = None
        _initialized = False


def get_tracer(name: str = "reasoning-mcp") -> Any:
    """Get a tracer instance.

    Args:
        name: Name for the tracer, typically module path.

    Returns:
        OpenTelemetry Tracer if available, otherwise a NoOpTracer.
    """
    if not _OTEL_AVAILABLE or not _initialized:
        return _noop_tracer

    return trace.get_tracer(name)


__all__ = [
    "is_telemetry_available",
    "init_telemetry",
    "shutdown_telemetry",
    "get_tracer",
]
