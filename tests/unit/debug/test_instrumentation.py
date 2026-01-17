from unittest.mock import Mock

from reasoning_mcp.debug.instrumentation import (
    InstrumentationConfig,
    InstrumentedExecutor,
    InstrumentedMethodExecutor,
    InstrumentedSequenceExecutor,
    instrument,
)
from reasoning_mcp.models.debug import TraceLevel


class TestInstrumentationConfig:
    def test_default_values(self):
        config = InstrumentationConfig()
        assert config.trace_level == TraceLevel.STANDARD
        assert config.capture_inputs is True
        assert config.capture_outputs is True
        assert config.capture_thoughts is True
        assert config.max_content_length == 1000

    def test_custom_values(self):
        config = InstrumentationConfig(
            trace_level=TraceLevel.VERBOSE, capture_inputs=False, max_content_length=500
        )
        assert config.trace_level == TraceLevel.VERBOSE
        assert config.capture_inputs is False
        assert config.max_content_length == 500


class TestInstrumentDecorator:
    def test_runs_function_without_collector(self):
        @instrument()
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_uses_custom_name(self):
        @instrument(name="custom_name")
        def my_func():
            return "result"

        assert my_func() == "result"


class TestInstrumentedExecutor:
    def test_init_without_collector(self):
        executor = InstrumentedExecutor()
        assert executor._trace_collector is None

    def test_init_with_collector(self):
        collector = Mock()
        executor = InstrumentedExecutor(trace_collector=collector)
        assert executor._trace_collector is collector

    def test_set_trace_collector(self):
        executor = InstrumentedExecutor()
        collector = Mock()
        executor.set_trace_collector(collector)
        assert executor._trace_collector is collector

    def test_start_span_without_collector(self):
        executor = InstrumentedExecutor()
        result = executor._start_span("test")
        assert result is None

    def test_start_span_with_collector(self):
        collector = Mock()
        collector.start_span.return_value = "span-123"
        executor = InstrumentedExecutor(trace_collector=collector)

        result = executor._start_span("test", key="value")

        assert result == "span-123"
        collector.start_span.assert_called_once()


class TestInstrumentedMethodExecutor:
    def test_init(self):
        executor = InstrumentedMethodExecutor("test_method")
        assert executor.method_name == "test_method"
        assert executor._trace_collector is None

    async def test_execute_without_collector(self):
        executor = InstrumentedMethodExecutor("test")

        async def my_func():
            return "result"

        result = await executor.execute_with_tracing(my_func)
        assert result == "result"

    async def test_execute_with_collector(self):
        collector = Mock()
        collector.start_span.return_value = "span-1"
        collector.add_step = Mock()
        collector.end_span = Mock()

        executor = InstrumentedMethodExecutor("test", trace_collector=collector)

        async def my_func():
            return "result"

        result = await executor.execute_with_tracing(my_func)

        assert result == "result"
        collector.start_span.assert_called_once()
        assert collector.add_step.call_count >= 2  # START and END
        collector.end_span.assert_called()


class TestInstrumentedSequenceExecutor:
    def test_init(self):
        executor = InstrumentedSequenceExecutor("my_sequence")
        assert executor.sequence_name == "my_sequence"

    async def test_execute_sequence_without_collector(self):
        executor = InstrumentedSequenceExecutor()

        async def step1():
            return 1

        async def step2():
            return 2

        results = await executor.execute_sequence_with_tracing([step1, step2])
        assert results == [1, 2]

    async def test_execute_sequence_with_collector(self):
        collector = Mock()
        collector.start_span.return_value = "span-1"
        collector.end_span = Mock()

        executor = InstrumentedSequenceExecutor("test", trace_collector=collector)

        async def step1():
            return 1

        async def step2():
            return 2

        results = await executor.execute_sequence_with_tracing([step1, step2])

        assert results == [1, 2]
        # Parent span + 2 step spans
        assert collector.start_span.call_count >= 3
