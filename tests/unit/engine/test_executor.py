"""Tests for the ExecutionContext and base executor classes.

This module tests the core execution context and base executor functionality,
including variable management, thought accumulation, and context updates.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from reasoning_mcp.engine.executor import (
    ExecutionContext,
    ExecutorRegistry,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import StageMetrics, StageTrace
from reasoning_mcp.models.session import Session
from reasoning_mcp.registry import MethodRegistry

# ============================================================================
# ExecutionContext Tests
# ============================================================================


class TestExecutionContext:
    """Test suite for ExecutionContext class."""

    @pytest.fixture
    def session(self) -> Session:
        """Provide a started session for testing."""
        return Session().start()

    @pytest.fixture
    def registry(self) -> MethodRegistry:
        """Provide a mock registry for testing."""
        return Mock(spec=MethodRegistry)

    @pytest.fixture
    def context(self, session: Session, registry: MethodRegistry) -> ExecutionContext:
        """Provide an ExecutionContext for testing."""
        return ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": "test query"},
            variables={},
        )

    def test_initialization(self, session: Session, registry: MethodRegistry):
        """Test ExecutionContext initializes correctly."""
        ctx = ExecutionContext(session=session, registry=registry)

        assert ctx.session is session
        assert ctx.registry is registry
        assert ctx.input_data == {}
        assert ctx.variables == {}
        assert ctx.trace is None
        assert ctx.thought_ids == []
        assert ctx.metadata == {}
        assert ctx.ctx is None

    def test_initialization_with_all_parameters(self, session: Session, registry: MethodRegistry):
        """Test ExecutionContext with all parameters."""
        from reasoning_mcp.models.pipeline import PipelineTrace

        trace = PipelineTrace(
            pipeline_id="test",
            session_id=session.id,
            started_at=datetime.now(),
            status="running",
        )

        ctx = ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": "test"},
            variables={"var1": "value1"},
            trace=trace,
            thought_ids=["t1", "t2"],
            metadata={"key": "value"},
        )

        assert ctx.input_data == {"input": "test"}
        assert ctx.variables == {"var1": "value1"}
        assert ctx.trace is trace
        assert ctx.thought_ids == ["t1", "t2"]
        assert ctx.metadata == {"key": "value"}

    def test_can_sample_without_ctx(self, context: ExecutionContext):
        """Test can_sample returns False when ctx is not set."""
        assert context.can_sample is False

    def test_can_sample_with_ctx(self, session: Session, registry: MethodRegistry):
        """Test can_sample returns True when ctx is set."""
        mock_ctx = Mock()
        context = ExecutionContext(
            session=session,
            registry=registry,
            ctx=mock_ctx,
        )

        assert context.can_sample is True

    def test_with_update_returns_new_context(self, context: ExecutionContext):
        """Test with_update returns a new ExecutionContext."""
        new_context = context.with_update(variables={"new_var": "new_value"})

        assert new_context is not context
        assert new_context.variables == {"new_var": "new_value"}
        assert context.variables == {}  # Original unchanged

    def test_with_update_preserves_unchanged_values(self, context: ExecutionContext):
        """Test with_update preserves values not explicitly updated."""
        context.input_data["extra"] = "data"

        new_context = context.with_update(variables={"new_var": "value"})

        assert new_context.session is context.session
        assert new_context.registry is context.registry
        assert new_context.input_data == context.input_data
        assert new_context.thought_ids == context.thought_ids

    def test_with_update_all_parameters(self, session: Session, registry: MethodRegistry):
        """Test with_update with all parameters."""
        original = ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": "original"},
            variables={"var": "original"},
            thought_ids=["t1"],
            metadata={"key": "original"},
        )

        new_session = Session().start()
        new_registry = Mock(spec=MethodRegistry)

        updated = original.with_update(
            session=new_session,
            registry=new_registry,
            input_data={"input": "updated"},
            variables={"var": "updated"},
            thought_ids=["t2", "t3"],
            metadata={"key": "updated"},
        )

        assert updated.session is new_session
        assert updated.registry is new_registry
        assert updated.input_data == {"input": "updated"}
        assert updated.variables == {"var": "updated"}
        assert updated.thought_ids == ["t2", "t3"]
        assert updated.metadata == {"key": "updated"}


# ============================================================================
# StageResult Tests
# ============================================================================


class TestStageResult:
    """Test suite for StageResult dataclass."""

    def test_basic_initialization(self):
        """Test StageResult with minimal parameters."""
        result = StageResult(
            stage_id="stage-123",
            stage_type=PipelineStageType.METHOD,
            success=True,
        )

        assert result.stage_id == "stage-123"
        assert result.stage_type == PipelineStageType.METHOD
        assert result.success is True
        assert result.output_thought_ids == []
        assert result.output_data == {}
        assert result.trace is None
        assert result.error is None
        assert result.metadata == {}
        assert result.should_continue is True

    def test_full_initialization(self):
        """Test StageResult with all parameters."""
        result = StageResult(
            stage_id="stage-123",
            stage_type=PipelineStageType.SEQUENCE,
            success=False,
            output_thought_ids=["t1", "t2"],
            output_data={"key": "value"},
            error="Something went wrong",
            metadata={"extra": "data"},
            should_continue=False,
        )

        assert result.stage_id == "stage-123"
        assert result.stage_type == PipelineStageType.SEQUENCE
        assert result.success is False
        assert result.output_thought_ids == ["t1", "t2"]
        assert result.output_data == {"key": "value"}
        assert result.error == "Something went wrong"
        assert result.metadata == {"extra": "data"}
        assert result.should_continue is False


# ============================================================================
# PipelineExecutor Tests
# ============================================================================


class TestPipelineExecutor:
    """Test suite for PipelineExecutor base class."""

    @pytest.fixture
    def concrete_executor(self) -> type[PipelineExecutor]:
        """Create a concrete executor class for testing."""

        class TestExecutor(PipelineExecutor):
            async def execute(self, context: ExecutionContext) -> StageResult:
                return StageResult(
                    stage_id="test",
                    stage_type=PipelineStageType.METHOD,
                    success=True,
                )

            async def validate(self, stage) -> list[str]:
                return []

        return TestExecutor

    def test_executor_initialization(self, concrete_executor):
        """Test PipelineExecutor initialization."""
        executor = concrete_executor()

        assert executor.streaming_context is None
        assert executor._trace_collector is None

    def test_executor_with_trace_collector(self, concrete_executor):
        """Test PipelineExecutor with trace_collector parameter."""
        trace_collector = Mock()
        executor = concrete_executor(trace_collector=trace_collector)

        assert executor._trace_collector is trace_collector

    def test_create_metrics(self, concrete_executor):
        """Test create_metrics helper method."""
        executor = concrete_executor()
        start_time = datetime(2025, 1, 1, 10, 0, 0)
        end_time = datetime(2025, 1, 1, 10, 0, 5)

        metrics = executor.create_metrics(
            stage_id="stage-123",
            start_time=start_time,
            end_time=end_time,
            thoughts_generated=10,
            errors_count=1,
            retries_count=2,
            custom_metric="custom_value",
        )

        assert isinstance(metrics, StageMetrics)
        assert metrics.stage_id == "stage-123"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration_seconds == 5.0
        assert metrics.thoughts_generated == 10
        assert metrics.errors_count == 1
        assert metrics.retries_count == 2
        assert metrics.metadata["custom_metric"] == "custom_value"

    def test_create_metrics_auto_end_time(self, concrete_executor):
        """Test create_metrics with automatic end_time."""
        executor = concrete_executor()
        start_time = datetime.now()

        metrics = executor.create_metrics(
            stage_id="stage-123",
            start_time=start_time,
        )

        assert metrics.end_time is not None
        assert metrics.end_time >= start_time

    def test_create_trace(self, concrete_executor):
        """Test create_trace helper method."""
        executor = concrete_executor()

        trace = executor.create_trace(
            stage_id="stage-123",
            stage_type=PipelineStageType.METHOD,
            status="completed",
            input_thought_ids=["t1"],
            output_thought_ids=["t2", "t3"],
            error=None,
            children=[],
            custom_data="value",
        )

        assert isinstance(trace, StageTrace)
        assert trace.stage_id == "stage-123"
        assert trace.stage_type == PipelineStageType.METHOD
        assert trace.status == "completed"
        assert trace.input_thought_ids == ["t1"]
        assert trace.output_thought_ids == ["t2", "t3"]
        assert trace.error is None
        assert trace.children == []
        assert trace.metadata["custom_data"] == "value"


# ============================================================================
# ExecutorRegistry Tests
# ============================================================================


class TestExecutorRegistry:
    """Test suite for ExecutorRegistry class."""

    @pytest.fixture
    def registry(self) -> ExecutorRegistry:
        """Provide a fresh ExecutorRegistry for testing."""
        return ExecutorRegistry()

    @pytest.fixture
    def mock_executor(self) -> PipelineExecutor:
        """Provide a mock executor for testing."""

        class MockExecutor(PipelineExecutor):
            async def execute(self, context: ExecutionContext) -> StageResult:
                return StageResult(
                    stage_id="mock",
                    stage_type=PipelineStageType.METHOD,
                    success=True,
                )

            async def validate(self, stage) -> list[str]:
                return []

        return MockExecutor()

    def test_empty_registry(self, registry: ExecutorRegistry):
        """Test newly created registry is empty."""
        assert registry.executor_count == 0
        assert registry.list_registered() == []

    def test_register_executor(self, registry: ExecutorRegistry, mock_executor: PipelineExecutor):
        """Test registering an executor."""
        registry.register(PipelineStageType.METHOD, mock_executor)

        assert registry.executor_count == 1
        assert registry.is_registered(PipelineStageType.METHOD) is True
        assert registry.get(PipelineStageType.METHOD) is mock_executor

    def test_register_duplicate_raises_error(
        self, registry: ExecutorRegistry, mock_executor: PipelineExecutor
    ):
        """Test registering duplicate stage type raises ValueError."""
        registry.register(PipelineStageType.METHOD, mock_executor)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(PipelineStageType.METHOD, mock_executor)

    def test_register_with_replace(
        self, registry: ExecutorRegistry, mock_executor: PipelineExecutor
    ):
        """Test registering with replace=True."""
        registry.register(PipelineStageType.METHOD, mock_executor)

        # Create another executor
        class AnotherExecutor(PipelineExecutor):
            async def execute(self, context: ExecutionContext) -> StageResult:
                return StageResult(
                    stage_id="another",
                    stage_type=PipelineStageType.METHOD,
                    success=True,
                )

            async def validate(self, stage) -> list[str]:
                return []

        another = AnotherExecutor()
        registry.register(PipelineStageType.METHOD, another, replace=True)

        assert registry.get(PipelineStageType.METHOD) is another

    def test_register_invalid_type(self, registry: ExecutorRegistry):
        """Test registering non-executor raises TypeError."""
        with pytest.raises(TypeError, match="must inherit from PipelineExecutor"):
            registry.register(PipelineStageType.METHOD, "not an executor")  # type: ignore

    def test_unregister_executor(self, registry: ExecutorRegistry, mock_executor: PipelineExecutor):
        """Test unregistering an executor."""
        registry.register(PipelineStageType.METHOD, mock_executor)

        result = registry.unregister(PipelineStageType.METHOD)

        assert result is True
        assert registry.is_registered(PipelineStageType.METHOD) is False
        assert registry.executor_count == 0

    def test_unregister_non_existent(self, registry: ExecutorRegistry):
        """Test unregistering non-existent executor returns False."""
        result = registry.unregister(PipelineStageType.METHOD)

        assert result is False

    def test_get_non_existent(self, registry: ExecutorRegistry):
        """Test getting non-existent executor returns None."""
        result = registry.get(PipelineStageType.METHOD)

        assert result is None

    def test_list_registered(self, registry: ExecutorRegistry, mock_executor: PipelineExecutor):
        """Test listing registered stage types."""
        registry.register(PipelineStageType.METHOD, mock_executor)
        registry.register(PipelineStageType.SEQUENCE, mock_executor)

        registered = registry.list_registered()

        assert len(registered) == 2
        assert PipelineStageType.METHOD in registered
        assert PipelineStageType.SEQUENCE in registered
