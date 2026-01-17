"""Test that all modules can be imported without circular import errors."""

import importlib
import re
from pathlib import Path

import pytest

MODULES_TO_TEST = [
    # streaming
    "reasoning_mcp.streaming",
    "reasoning_mcp.streaming.backpressure",
    "reasoning_mcp.streaming.buffer",
    "reasoning_mcp.streaming.context",
    "reasoning_mcp.streaming.emitter",
    "reasoning_mcp.streaming.events",
    "reasoning_mcp.streaming.metrics",
    "reasoning_mcp.streaming.serialization",
    "reasoning_mcp.streaming.transport",
    "reasoning_mcp.streaming.transport.sse",
    "reasoning_mcp.streaming.transport.websocket",
    # debug
    "reasoning_mcp.debug",
    "reasoning_mcp.debug.analyzer",
    "reasoning_mcp.debug.collector",
    "reasoning_mcp.debug.export",
    "reasoning_mcp.debug.instrumentation",
    "reasoning_mcp.debug.storage",
    # tools
    "reasoning_mcp.tools.debug",
]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module_imports(module_name: str) -> None:
    """Test that each module can be imported without errors."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_no_relative_imports_in_source() -> None:
    """Verify no relative imports exist in source files."""
    pattern = re.compile(r"^\s*from\s+\.+[a-zA-Z_]")
    src_dir = Path(__file__).parent.parent.parent / "src" / "reasoning_mcp"

    relative_imports = []
    for filepath in src_dir.rglob("*.py"):
        content = filepath.read_text()
        for i, line in enumerate(content.split("\n"), 1):
            if pattern.match(line):
                relative_path = filepath.relative_to(src_dir.parent.parent)
                relative_imports.append(f"{relative_path}:{i}: {line.strip()}")

    assert not relative_imports, f"Found {len(relative_imports)} relative imports:\n" + "\n".join(
        relative_imports[:20]
    )


def test_streaming_exports() -> None:
    """Test that streaming module exports all expected symbols."""
    from reasoning_mcp import streaming

    expected_exports = [
        "StreamingEventType",
        "BaseStreamEvent",
        "ThoughtEvent",
        "ProgressEvent",
        "StageEvent",
        "TokenEvent",
        "ErrorEvent",
        "CompleteEvent",
        "StreamEmitterProtocol",
        "AsyncStreamEmitter",
        "StreamingContext",
        "BackpressureStrategy",
        "BackpressureConfig",
        "BackpressureHandler",
        "BackpressureError",
        "EventBuffer",
        "SSESerializer",
        "JSONLSerializer",
        "StreamingMetrics",
        "MetricsCollector",
    ]

    for name in expected_exports:
        assert hasattr(streaming, name), f"Missing export: {name}"


def test_debug_exports() -> None:
    """Test that debug module exports all expected symbols."""
    from reasoning_mcp import debug

    expected_exports = [
        "TraceCollector",
        "TraceStorage",
        "MemoryTraceStorage",
        "FileTraceStorage",
        "SQLiteTraceStorage",
        "get_storage",
        "TraceExporter",
        "JSONExporter",
        "MermaidExporter",
        "HTMLExporter",
        "OpenTelemetryExporter",
        "get_exporter",
        "TraceAnalyzer",
        "HasTraceCollector",
        "InstrumentationConfig",
        "instrument",
        "InstrumentedExecutor",
        "InstrumentedMethodExecutor",
        "InstrumentedSequenceExecutor",
    ]

    for name in expected_exports:
        assert hasattr(debug, name), f"Missing export: {name}"


def test_no_import_cycles() -> None:
    """Test that importing modules doesn't cause circular import errors."""
    # Fresh import sequence that would expose cycles
    import importlib
    import sys

    # Save the current module state to restore after the test
    saved_modules = {m: sys.modules[m] for m in list(sys.modules.keys()) if m.startswith("reasoning_mcp.")}

    # Clear any cached imports for these modules
    modules_to_clear = [m for m in sys.modules if m.startswith("reasoning_mcp.")]
    for m in modules_to_clear:
        del sys.modules[m]

    # Now re-import in an order that would expose cycles
    # Start from leaf modules and work up
    try:
        importlib.import_module("reasoning_mcp.streaming.events")
        importlib.import_module("reasoning_mcp.streaming.metrics")
        importlib.import_module("reasoning_mcp.streaming.backpressure")
        importlib.import_module("reasoning_mcp.streaming.buffer")
        importlib.import_module("reasoning_mcp.streaming.emitter")
        importlib.import_module("reasoning_mcp.streaming.serialization")
        importlib.import_module("reasoning_mcp.streaming.context")
        importlib.import_module("reasoning_mcp.streaming.transport.sse")
        importlib.import_module("reasoning_mcp.streaming.transport.websocket")
        importlib.import_module("reasoning_mcp.streaming.transport")
        importlib.import_module("reasoning_mcp.streaming")

        importlib.import_module("reasoning_mcp.models.debug")
        importlib.import_module("reasoning_mcp.debug.analyzer")
        importlib.import_module("reasoning_mcp.debug.collector")
        importlib.import_module("reasoning_mcp.debug.export")
        importlib.import_module("reasoning_mcp.debug.instrumentation")
        importlib.import_module("reasoning_mcp.debug.storage")
        importlib.import_module("reasoning_mcp.debug")

        importlib.import_module("reasoning_mcp.tools.debug")
    except ImportError as e:
        pytest.fail(f"Circular import detected: {e}")
    finally:
        # Restore original modules to avoid polluting other tests
        # First clear any new modules that were imported during the test
        new_modules = [m for m in sys.modules if m.startswith("reasoning_mcp.")]
        for m in new_modules:
            del sys.modules[m]
        # Now restore the saved modules
        sys.modules.update(saved_modules)
