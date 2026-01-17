"""
Comprehensive tests for the compose() tool function.

This module provides complete test coverage for the compose() function:
- Testing with different pipeline types (MethodStage, SequencePipeline, etc.)
- Testing with and without session_id
- Testing return type is ComposeOutput
- Testing field values and validation
- Testing async execution
"""

from datetime import datetime

import pytest

from reasoning_mcp.models.core import MethodIdentifier
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
from reasoning_mcp.models.tools import ComposeOutput
from reasoning_mcp.tools.compose import compose

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_method_stage() -> MethodStage:
    """Provide a simple MethodStage for testing."""
    return MethodStage(
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="test_stage",
        max_thoughts=10,
    )


@pytest.fixture
def sequence_pipeline() -> SequencePipeline:
    """Provide a SequencePipeline for testing."""
    return SequencePipeline(
        name="test_sequence",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="stage_1",
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="stage_2",
            ),
        ],
    )


@pytest.fixture
def parallel_pipeline() -> ParallelPipeline:
    """Provide a ParallelPipeline for testing."""
    return ParallelPipeline(
        name="test_parallel",
        branches=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="branch_1",
            ),
            MethodStage(
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                name="branch_2",
            ),
        ],
        merge_strategy=MergeStrategy(
            name="best",
            selection_criteria="highest_confidence",
        ),
    )


@pytest.fixture
def conditional_pipeline() -> ConditionalPipeline:
    """Provide a ConditionalPipeline for testing."""
    return ConditionalPipeline(
        name="test_conditional",
        condition=Condition(
            name="confidence_check",
            expression="confidence > 0.8",
            operator=">",
            threshold=0.8,
            field="confidence",
        ),
        if_true=MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="high_confidence_path",
        ),
        if_false=MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="low_confidence_path",
        ),
    )


@pytest.fixture
def loop_pipeline() -> LoopPipeline:
    """Provide a LoopPipeline for testing."""
    return LoopPipeline(
        name="test_loop",
        body=MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="loop_body",
        ),
        condition=Condition(
            name="iteration_limit",
            expression="iterations < 5",
            operator="<",
            threshold=5,
            field="iterations",
        ),
        max_iterations=5,
    )


@pytest.fixture
def switch_pipeline() -> SwitchPipeline:
    """Provide a SwitchPipeline for testing."""
    return SwitchPipeline(
        name="test_switch",
        expression="problem_type",
        cases={
            "ethical": MethodStage(
                method_id=MethodIdentifier.ETHICAL_REASONING,
                name="ethical_case",
            ),
            "code": MethodStage(
                method_id=MethodIdentifier.CODE_REASONING,
                name="code_case",
            ),
        },
        default=MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="default_case",
        ),
    )


# ============================================================================
# Test Basic Execution
# ============================================================================


class TestComposeBasic:
    """Test basic compose() function behavior."""

    @pytest.mark.asyncio
    async def test_compose_is_async(self, simple_method_stage: MethodStage):
        """Test that compose() is an async function."""
        result = compose(
            pipeline=simple_method_stage,
            input="Test input",
        )
        # Should return a coroutine
        assert hasattr(result, "__await__")
        # Await and verify it completes
        output = await result
        assert isinstance(output, ComposeOutput)

    @pytest.mark.asyncio
    async def test_compose_returns_compose_output(self, simple_method_stage: MethodStage):
        """Test that compose() returns ComposeOutput type."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert isinstance(output, ComposeOutput)

    @pytest.mark.asyncio
    async def test_compose_with_minimal_parameters(self, simple_method_stage: MethodStage):
        """Test compose() with only required parameters."""
        output = await compose(
            pipeline=simple_method_stage,
            input="What is the meaning of life?",
        )

        # Verify required fields are present
        assert output.session_id is not None
        assert output.pipeline_id is not None
        assert isinstance(output.success, bool)

    @pytest.mark.asyncio
    async def test_compose_without_session_id(self, simple_method_stage: MethodStage):
        """Test compose() without providing session_id (should create new)."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
            session_id=None,
        )

        # Should create a new session ID
        assert output.session_id is not None
        assert len(output.session_id) > 0

    @pytest.mark.asyncio
    async def test_compose_with_session_id(self, simple_method_stage: MethodStage):
        """Test compose() with explicit session_id."""
        test_session_id = "test-session-123"

        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
            session_id=test_session_id,
        )

        # Should use the provided session ID
        assert output.session_id == test_session_id

    @pytest.mark.asyncio
    async def test_compose_pipeline_id_matches(self, simple_method_stage: MethodStage):
        """Test that output pipeline_id matches input pipeline id."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert output.pipeline_id == simple_method_stage.id


# ============================================================================
# Test Different Pipeline Types
# ============================================================================


class TestComposePipelineTypes:
    """Test compose() with different pipeline stage types."""

    @pytest.mark.asyncio
    async def test_compose_with_method_stage(self, simple_method_stage: MethodStage):
        """Test compose() with MethodStage."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test problem",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == simple_method_stage.id

    @pytest.mark.asyncio
    async def test_compose_with_sequence_pipeline(self, sequence_pipeline: SequencePipeline):
        """Test compose() with SequencePipeline."""
        output = await compose(
            pipeline=sequence_pipeline,
            input="Test problem requiring multiple stages",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == sequence_pipeline.id

    @pytest.mark.asyncio
    async def test_compose_with_parallel_pipeline(self, parallel_pipeline: ParallelPipeline):
        """Test compose() with ParallelPipeline."""
        output = await compose(
            pipeline=parallel_pipeline,
            input="Test problem with parallel exploration",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == parallel_pipeline.id

    @pytest.mark.asyncio
    async def test_compose_with_conditional_pipeline(
        self, conditional_pipeline: ConditionalPipeline
    ):
        """Test compose() with ConditionalPipeline."""
        output = await compose(
            pipeline=conditional_pipeline,
            input="Test problem with conditional logic",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == conditional_pipeline.id

    @pytest.mark.asyncio
    async def test_compose_with_loop_pipeline(self, loop_pipeline: LoopPipeline):
        """Test compose() with LoopPipeline."""
        output = await compose(
            pipeline=loop_pipeline,
            input="Test problem requiring iteration",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == loop_pipeline.id

    @pytest.mark.asyncio
    async def test_compose_with_switch_pipeline(self, switch_pipeline: SwitchPipeline):
        """Test compose() with SwitchPipeline."""
        output = await compose(
            pipeline=switch_pipeline,
            input="Test problem with routing",
        )

        assert isinstance(output, ComposeOutput)
        assert output.pipeline_id == switch_pipeline.id


# ============================================================================
# Test Output Structure
# ============================================================================


class TestComposeOutput:
    """Test the structure and content of ComposeOutput."""

    @pytest.mark.asyncio
    async def test_compose_output_has_session_id(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has session_id field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "session_id")
        assert isinstance(output.session_id, str)

    @pytest.mark.asyncio
    async def test_compose_output_has_pipeline_id(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has pipeline_id field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "pipeline_id")
        assert isinstance(output.pipeline_id, str)

    @pytest.mark.asyncio
    async def test_compose_output_has_success_flag(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has success field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "success")
        assert isinstance(output.success, bool)

    @pytest.mark.asyncio
    async def test_compose_output_has_final_thoughts(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has final_thoughts field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "final_thoughts")
        assert isinstance(output.final_thoughts, list)

    @pytest.mark.asyncio
    async def test_compose_output_has_trace(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has trace field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "trace")
        # Trace can be None or PipelineTrace

    @pytest.mark.asyncio
    async def test_compose_output_has_error(self, simple_method_stage: MethodStage):
        """Test that ComposeOutput has error field."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        assert hasattr(output, "error")
        # Error can be None or str


# ============================================================================
# Test Input Validation
# ============================================================================


class TestComposeInputValidation:
    """Test input validation for compose() function."""

    @pytest.mark.asyncio
    async def test_compose_with_empty_input(self, simple_method_stage: MethodStage):
        """Test compose() with empty input string."""
        output = await compose(
            pipeline=simple_method_stage,
            input="",
        )

        # Should still execute (empty input is valid)
        assert isinstance(output, ComposeOutput)

    @pytest.mark.asyncio
    async def test_compose_with_long_input(self, simple_method_stage: MethodStage):
        """Test compose() with very long input string."""
        long_input = "Test problem. " * 1000

        output = await compose(
            pipeline=simple_method_stage,
            input=long_input,
        )

        assert isinstance(output, ComposeOutput)

    @pytest.mark.asyncio
    async def test_compose_with_special_characters(self, simple_method_stage: MethodStage):
        """Test compose() with special characters in input."""
        special_input = "Test with émojis 🎉 and speçial chàracters: @#$%^&*()"

        output = await compose(
            pipeline=simple_method_stage,
            input=special_input,
        )

        assert isinstance(output, ComposeOutput)

    @pytest.mark.asyncio
    async def test_compose_with_multiline_input(self, simple_method_stage: MethodStage):
        """Test compose() with multiline input."""
        multiline_input = """Line 1
        Line 2
        Line 3
        """

        output = await compose(
            pipeline=simple_method_stage,
            input=multiline_input,
        )

        assert isinstance(output, ComposeOutput)


# ============================================================================
# Test Trace Information
# ============================================================================


class TestComposeTrace:
    """Test trace information in ComposeOutput."""

    @pytest.mark.asyncio
    async def test_compose_trace_contains_pipeline_id(self, simple_method_stage: MethodStage):
        """Test that trace contains pipeline_id if present."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        if output.trace is not None:
            assert output.trace.pipeline_id == output.pipeline_id

    @pytest.mark.asyncio
    async def test_compose_trace_contains_session_id(self, simple_method_stage: MethodStage):
        """Test that trace contains session_id if present."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        if output.trace is not None:
            assert output.trace.session_id == output.session_id

    @pytest.mark.asyncio
    async def test_compose_trace_has_timestamps(self, simple_method_stage: MethodStage):
        """Test that trace has started_at timestamp if present."""
        output = await compose(
            pipeline=simple_method_stage,
            input="Test input",
        )

        if output.trace is not None:
            assert output.trace.started_at is not None
            assert isinstance(output.trace.started_at, datetime)


# ============================================================================
# Test Error Handling
# ============================================================================


class TestComposeErrorHandling:
    """Test error handling in compose() function."""

    @pytest.mark.asyncio
    async def test_compose_failure_sets_success_false(self):
        """Test that failed execution sets success=False."""
        # This test assumes the current placeholder implementation
        # which always returns success=False
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="test",
        )

        output = await compose(
            pipeline=stage,
            input="Test input",
        )

        # Current placeholder implementation returns False
        assert output.success is False

    @pytest.mark.asyncio
    async def test_compose_failure_provides_error_message(self):
        """Test that failed execution provides error message."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="test",
        )

        output = await compose(
            pipeline=stage,
            input="Test input",
        )

        # Current placeholder implementation provides an error
        if not output.success:
            assert output.error is not None
            assert isinstance(output.error, str)


# ============================================================================
# Test Idempotency
# ============================================================================


class TestComposeIdempotency:
    """Test idempotency and consistency of compose() function."""

    @pytest.mark.asyncio
    async def test_compose_multiple_calls_same_pipeline(self, simple_method_stage: MethodStage):
        """Test multiple calls with same pipeline create different sessions."""
        output1 = await compose(
            pipeline=simple_method_stage,
            input="Test input 1",
        )

        output2 = await compose(
            pipeline=simple_method_stage,
            input="Test input 2",
        )

        # Should create different sessions
        assert output1.session_id != output2.session_id
        # But use the same pipeline
        assert output1.pipeline_id == output2.pipeline_id

    @pytest.mark.asyncio
    async def test_compose_with_same_session_id(self, simple_method_stage: MethodStage):
        """Test multiple calls with same session_id."""
        session_id = "shared-session-123"

        output1 = await compose(
            pipeline=simple_method_stage,
            input="Test input 1",
            session_id=session_id,
        )

        output2 = await compose(
            pipeline=simple_method_stage,
            input="Test input 2",
            session_id=session_id,
        )

        # Should use the same session ID
        assert output1.session_id == session_id
        assert output2.session_id == session_id
