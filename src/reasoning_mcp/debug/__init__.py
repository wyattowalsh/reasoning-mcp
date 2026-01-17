"""Debug and tracing module for reasoning-mcp.

This module provides comprehensive debugging and tracing capabilities for
monitoring reasoning execution, including:

- Trace collection (TraceCollector)
- Multiple storage backends (MemoryTraceStorage, FileTraceStorage, SQLiteTraceStorage)
- Export formats (JSON, HTML, Mermaid, OpenTelemetry)
- Trace analysis (TraceAnalyzer)
- Instrumentation utilities
"""

from reasoning_mcp.debug.analyzer import TraceAnalyzer
from reasoning_mcp.debug.collector import TraceCollector
from reasoning_mcp.debug.export import (
    HTMLExporter,
    JSONExporter,
    MermaidExporter,
    OpenTelemetryExporter,
    TraceExporter,
    get_exporter,
)
from reasoning_mcp.debug.instrumentation import (
    HasTraceCollector,
    InstrumentationConfig,
    InstrumentedExecutor,
    InstrumentedMethodExecutor,
    InstrumentedSequenceExecutor,
    instrument,
)
from reasoning_mcp.debug.storage import (
    FileTraceStorage,
    MemoryTraceStorage,
    SQLiteTraceStorage,
    TraceStorage,
    get_storage,
)

__all__ = [
    # Collector
    "TraceCollector",
    # Storage
    "TraceStorage",
    "MemoryTraceStorage",
    "FileTraceStorage",
    "SQLiteTraceStorage",
    "get_storage",
    # Export
    "TraceExporter",
    "JSONExporter",
    "MermaidExporter",
    "HTMLExporter",
    "OpenTelemetryExporter",
    "get_exporter",
    # Analysis
    "TraceAnalyzer",
    # Instrumentation
    "HasTraceCollector",
    "InstrumentationConfig",
    "instrument",
    "InstrumentedExecutor",
    "InstrumentedMethodExecutor",
    "InstrumentedSequenceExecutor",
]
