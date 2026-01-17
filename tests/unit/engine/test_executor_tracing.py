"""Tests for executor tracing support.

This module tests that engine executors properly integrate with TraceCollector
for debugging and monitoring purposes.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from reasoning_mcp.debug.collector import TraceCollector
from reasoning_mcp.engine.conditional import ConditionalExecutor
from reasoning_mcp.engine.executor import ExecutionContext, PipelineExecutor
from reasoning_mcp.engine.method import MethodExecutor
from reasoning_mcp.engine.parallel import ParallelExecutor
from reasoning_mcp.engine.sequence import SequenceExecutor
from reasoning_mcp.models.core import MethodIdentifier
from reasoning_mcp.models.debug import SpanStatus
from reasoning_mcp.models.pipeline import (
    Condition,
    ConditionalPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    SequencePipeline,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.registry import MethodRegistry


class TestExecutorTracing:
    """Test that executors accept and use trace_collector parameter."""

    def test_base_executor_accepts_trace_collector(self):
        """Test that PipelineExecutor base class accepts trace_collector parameter."""
        # Create a mock trace collector
        trace_collector = Mock(spec=TraceCollector)

        # Create a minimal concrete executor to test the base class
        class TestExecutor(PipelineExecutor):
            async def execute(self, context):
                return Mock()

            async def validate(self, stage):
                return []

        # Initialize with trace_collector
        executor = TestExecutor(trace_collector=trace_collector)

        # Verify it was stored
        assert executor._trace_collector is trace_collector

    def test_base_executor_works_without_trace_collector(self):
        """Test that PipelineExecutor works when trace_collector is not provided."""

        class TestExecutor(PipelineExecutor):
            async def execute(self, context):
                return Mock()

            async def validate(self, stage):
                return []

        # Initialize without trace_collector
        executor = TestExecutor()

        # Verify it's None
        assert executor._trace_collector is None

    def test_method_executor_accepts_trace_collector(self):
        """Test that MethodExecutor accepts trace_collector parameter."""
        stage = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="test_method")
        trace_collector = Mock(spec=TraceCollector)

        executor = MethodExecutor(pipeline=stage, trace_collector=trace_collector)

        assert executor._trace_collector is trace_collector

    async def test_method_executor_creates_span_when_tracing(self):
        """Test that MethodExecutor creates spans when trace_collector is present."""
        # Setup
        stage = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="test_method")
        trace_collector = Mock(spec=TraceCollector)
        span_id = str(uuid4())
        trace_collector.start_span.return_value = span_id

        executor = MethodExecutor(pipeline=stage, trace_collector=trace_collector)

        # Create mock session and registry
        session = Session().start()
        registry = Mock(spec=MethodRegistry)

        # Create mock method and thought
        from reasoning_mcp.models.core import ThoughtType
        from reasoning_mcp.models.thought import ThoughtNode

        mock_thought = ThoughtNode(
            id=str(uuid4()),
            content="Test thought",
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            confidence=0.9,
        )

        mock_method = Mock()
        mock_method.execute = AsyncMock(return_value=mock_thought)

        registry.get.return_value = mock_method
        registry.is_initialized.return_value = True

        context = ExecutionContext(
            session=session, registry=registry, input_data={"input": "test query"}
        )

        # Execute
        await executor.execute(context)

        # Verify span was created and ended
        trace_collector.start_span.assert_called_once()
        assert "MethodExecutor" in trace_collector.start_span.call_args[0][0]

        trace_collector.end_span.assert_called_once_with(span_id, SpanStatus.COMPLETED)

    async def test_method_executor_ends_span_on_error(self):
        """Test that MethodExecutor ends span with FAILED status on error."""
        stage = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="test_method")
        trace_collector = Mock(spec=TraceCollector)
        span_id = str(uuid4())
        trace_collector.start_span.return_value = span_id

        executor = MethodExecutor(pipeline=stage, trace_collector=trace_collector)

        session = Session().start()
        registry = Mock(spec=MethodRegistry)

        # Make registry.get raise an error
        registry.get.side_effect = RuntimeError("Test error")

        context = ExecutionContext(
            session=session, registry=registry, input_data={"input": "test query"}
        )

        # Execute (should handle error)
        result = await executor.execute(context)

        # Verify span was ended with FAILED status
        trace_collector.end_span.assert_called_once_with(span_id, SpanStatus.FAILED)
        assert result.success is False

    def test_sequence_executor_accepts_trace_collector(self):
        """Test that SequenceExecutor accepts trace_collector parameter."""
        pipeline = SequencePipeline(name="test_sequence", stages=[])
        trace_collector = Mock(spec=TraceCollector)

        executor = SequenceExecutor(pipeline=pipeline, trace_collector=trace_collector)

        assert executor._trace_collector is trace_collector

    def test_parallel_executor_accepts_trace_collector(self):
        """Test that ParallelExecutor accepts trace_collector parameter."""
        pipeline = ParallelPipeline(
            name="test_parallel", branches=[], merge_strategy=MergeStrategy(name="concat")
        )
        trace_collector = Mock(spec=TraceCollector)

        executor = ParallelExecutor(pipeline=pipeline, trace_collector=trace_collector)

        assert executor._trace_collector is trace_collector

    def test_conditional_executor_accepts_trace_collector(self):
        """Test that ConditionalExecutor accepts trace_collector parameter."""
        # Create a simple branch
        if_true_stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="if_true_branch"
        )

        pipeline = ConditionalPipeline(
            name="test_conditional",
            condition=Condition(name="test", expression="True"),
            if_true=if_true_stage,
        )
        trace_collector = Mock(spec=TraceCollector)

        executor = ConditionalExecutor(pipeline=pipeline, trace_collector=trace_collector)

        assert executor._trace_collector is trace_collector

    async def test_executor_works_without_trace_collector(self):
        """Test that executors work normally when trace_collector is None."""
        from reasoning_mcp.models.core import ThoughtType
        from reasoning_mcp.models.thought import ThoughtNode

        stage = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="test_method")

        # Create executor without trace_collector
        executor = MethodExecutor(pipeline=stage)

        session = Session().start()
        registry = Mock(spec=MethodRegistry)

        mock_thought = ThoughtNode(
            id=str(uuid4()),
            content="Test thought",
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            confidence=0.9,
        )

        mock_method = Mock()
        mock_method.execute = AsyncMock(return_value=mock_thought)

        registry.get.return_value = mock_method
        registry.is_initialized.return_value = True

        context = ExecutionContext(
            session=session, registry=registry, input_data={"input": "test query"}
        )

        # Execute should work without tracing
        result = await executor.execute(context)

        # Should succeed
        assert result.success is True
