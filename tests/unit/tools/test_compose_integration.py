"""Integration tests for the compose tool.

This module tests the compose() function with real pipeline execution,
verifying the end-to-end flow through the executor chain.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.pipeline import (
    Condition,
    ConditionalPipeline,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    SequencePipeline,
    SwitchPipeline,
)
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import ComposeOutput
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.tools.compose import compose, compose_background

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Provide a mock registry with pre-configured methods."""
    registry = Mock(spec=MethodRegistry)

    # Mock method that returns a thought
    def create_mock_method():
        method = Mock()
        method.execute = AsyncMock(
            return_value=ThoughtNode(
                id="thought-123",
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="Reasoning output",
                confidence=0.85,
            )
        )
        method.initialize = AsyncMock()
        return method

    registry.get.return_value = create_mock_method()
    registry.is_initialized.return_value = True
    registry.is_registered.return_value = True

    return registry


@pytest.fixture
def method_stage() -> MethodStage:
    """Provide a simple method stage."""
    return MethodStage(
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="test_stage",
    )


@pytest.fixture
def sequence_pipeline() -> SequencePipeline:
    """Provide a sequence pipeline."""
    return SequencePipeline(
        name="test_sequence",
        stages=[
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="stage_1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="stage_2"),
        ],
    )


@pytest.fixture
def parallel_pipeline() -> ParallelPipeline:
    """Provide a parallel pipeline."""
    return ParallelPipeline(
        name="test_parallel",
        branches=[
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="branch_1"),
            MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="branch_2"),
        ],
        merge_strategy=MergeStrategy(name="concat"),
    )


@pytest.fixture
def conditional_pipeline() -> ConditionalPipeline:
    """Provide a conditional pipeline."""
    return ConditionalPipeline(
        name="test_conditional",
        condition=Condition(name="check", expression="confidence > 0.5"),
        if_true=MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="true_path",
        ),
        if_false=MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="false_path",
        ),
    )


@pytest.fixture
def loop_pipeline() -> LoopPipeline:
    """Provide a loop pipeline."""
    return LoopPipeline(
        name="test_loop",
        body=MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="reflect"),
        condition=Condition(
            name="iteration_check",
            expression="iteration < max_iterations",
            operator="<",
        ),
        max_iterations=2,
    )


@pytest.fixture
def switch_pipeline() -> SwitchPipeline:
    """Provide a switch pipeline."""
    return SwitchPipeline(
        name="test_switch",
        expression="category",
        cases={
            "math": MethodStage(
                method_id=MethodIdentifier.MATHEMATICAL_REASONING,
                name="math_handler",
            ),
        },
        default=MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="default_handler",
        ),
    )


# ============================================================================
# Basic Compose Tests
# ============================================================================


class TestComposeBasic:
    """Basic tests for compose function."""

    async def test_compose_returns_compose_output(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose returns ComposeOutput."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert isinstance(result, ComposeOutput)

    async def test_compose_generates_session_id(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose generates session ID if not provided."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.session_id is not None
        assert len(result.session_id) > 0

    async def test_compose_uses_provided_session_id(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose uses provided session ID."""
        session_id = "custom-session-123"

        result = await compose(
            pipeline=method_stage,
            input="Test input",
            session_id=session_id,
            registry=mock_registry,
        )

        assert result.session_id == session_id

    async def test_compose_returns_pipeline_id(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose returns correct pipeline ID."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.pipeline_id == method_stage.id

    async def test_compose_includes_trace(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose includes execution trace."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.trace is not None
        assert result.trace.pipeline_id == method_stage.id


# ============================================================================
# Pipeline Type Tests
# ============================================================================


class TestComposePipelineTypes:
    """Test compose with different pipeline types."""

    async def test_compose_method_stage(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test composing a simple method stage."""
        result = await compose(
            pipeline=method_stage,
            input="Test query",
            registry=mock_registry,
        )

        assert result.success is True

    async def test_compose_sequence_pipeline(
        self, sequence_pipeline: SequencePipeline, mock_registry: MethodRegistry
    ):
        """Test composing a sequence pipeline."""
        result = await compose(
            pipeline=sequence_pipeline,
            input="Test query",
            registry=mock_registry,
        )

        # Sequence should execute even if internal errors occur
        assert result.trace is not None

    async def test_compose_parallel_pipeline(
        self, parallel_pipeline: ParallelPipeline, mock_registry: MethodRegistry
    ):
        """Test composing a parallel pipeline."""
        result = await compose(
            pipeline=parallel_pipeline,
            input="Test query",
            registry=mock_registry,
        )

        assert result.trace is not None

    async def test_compose_conditional_pipeline(
        self, conditional_pipeline: ConditionalPipeline, mock_registry: MethodRegistry
    ):
        """Test composing a conditional pipeline."""
        result = await compose(
            pipeline=conditional_pipeline,
            input="Test query",
            registry=mock_registry,
        )

        assert result.trace is not None

    async def test_compose_loop_pipeline(
        self, loop_pipeline: LoopPipeline, mock_registry: MethodRegistry
    ):
        """Test composing a loop pipeline."""
        result = await compose(
            pipeline=loop_pipeline,
            input="Test query",
            registry=mock_registry,
        )

        assert result.trace is not None

    async def test_compose_switch_pipeline(
        self, switch_pipeline: SwitchPipeline, mock_registry: MethodRegistry
    ):
        """Test composing a switch pipeline."""
        result = await compose(
            pipeline=switch_pipeline,
            input="Test query",
            registry=mock_registry,
        )

        assert result.trace is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestComposeErrorHandling:
    """Test compose error handling."""

    async def test_compose_handles_executor_exception(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose handles executor exceptions gracefully."""
        # Make registry raise an exception
        mock_registry.get.side_effect = RuntimeError("Registry error")

        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.success is False
        assert result.error is not None
        assert "Registry error" in result.error

    async def test_compose_handles_method_execution_failure(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test handling when method execution fails."""
        mock_method = Mock()
        mock_method.execute = AsyncMock(side_effect=ValueError("Method failed"))
        mock_method.initialize = AsyncMock()
        mock_registry.get.return_value = mock_method

        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.success is False
        assert "Method failed" in result.error


# ============================================================================
# Background Execution Tests
# ============================================================================


class TestComposeBackground:
    """Test compose_background function."""

    async def test_compose_background_returns_session_id(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose_background returns session ID."""
        session_id = await compose_background(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert session_id is not None
        assert len(session_id) > 0

    async def test_compose_background_uses_provided_session_id(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that compose_background uses provided session ID."""
        provided_session_id = "provided-session-456"

        session_id = await compose_background(
            pipeline=method_stage,
            input="Test input",
            session_id=provided_session_id,
            registry=mock_registry,
        )

        assert session_id == provided_session_id


# ============================================================================
# Trace Content Tests
# ============================================================================


class TestComposeTraceContent:
    """Test compose trace content."""

    async def test_compose_trace_has_timestamps(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that trace includes timestamps."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.trace.started_at is not None
        assert result.trace.completed_at is not None
        assert result.trace.completed_at >= result.trace.started_at

    async def test_compose_trace_has_status(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that trace includes status."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.trace.status in ["completed", "failed"]

    async def test_compose_trace_has_metadata(
        self, method_stage: MethodStage, mock_registry: MethodRegistry
    ):
        """Test that trace includes metadata."""
        result = await compose(
            pipeline=method_stage,
            input="Test input",
            registry=mock_registry,
        )

        assert result.trace.metadata is not None
        assert "pipeline_type" in result.trace.metadata
