"""Tests for the SwitchExecutor.

This module tests switch/case pipeline execution, including case matching,
pattern types, and default handling.
"""

from unittest.mock import Mock, patch

import pytest

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.engine.switch import SwitchExecutor
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import MethodStage, SwitchPipeline
from reasoning_mcp.models.session import Session
from reasoning_mcp.registry import MethodRegistry

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session() -> Session:
    """Provide a started session for testing."""
    return Session().start()


@pytest.fixture
def registry() -> MethodRegistry:
    """Provide a mock registry for testing."""
    return Mock(spec=MethodRegistry)


@pytest.fixture
def context(session: Session, registry: MethodRegistry) -> ExecutionContext:
    """Provide an ExecutionContext for testing."""
    return ExecutionContext(
        session=session,
        registry=registry,
        input_data={"input": "test query"},
        variables={"category": "math", "difficulty": 5},
        thought_ids=[],
    )


@pytest.fixture
def switch_pipeline() -> SwitchPipeline:
    """Provide a switch pipeline for testing."""
    return SwitchPipeline(
        name="test_switch",
        expression="category",
        cases={
            "math": MethodStage(
                method_id=MethodIdentifier.MATHEMATICAL_REASONING,
                name="math_reasoning",
            ),
            "science": MethodStage(
                method_id=MethodIdentifier.CAUSAL_REASONING,
                name="scientific_reasoning",
            ),
        },
        default=MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="default_reasoning",
        ),
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSwitchExecutorInit:
    """Test SwitchExecutor initialization."""

    def test_basic_init(self, switch_pipeline: SwitchPipeline):
        """Test basic initialization."""
        executor = SwitchExecutor(pipeline=switch_pipeline)

        assert executor.pipeline is switch_pipeline


# ============================================================================
# Expression Evaluation Tests
# ============================================================================


class TestSwitchExpressionEvaluation:
    """Test switch expression evaluation."""

    @pytest.fixture
    def executor(self, switch_pipeline: SwitchPipeline) -> SwitchExecutor:
        """Provide a SwitchExecutor for testing."""
        return SwitchExecutor(pipeline=switch_pipeline)

    def test_evaluate_simple_variable(self, executor: SwitchExecutor, context: ExecutionContext):
        """Test evaluating a simple variable expression."""
        result = executor.evaluate_expression("category", context)

        assert result == "math"

    def test_evaluate_missing_variable(self, executor: SwitchExecutor, context: ExecutionContext):
        """Test evaluating missing variable returns None."""
        result = executor.evaluate_expression("missing_var", context)

        assert result is None

    def test_evaluate_input_data(self, executor: SwitchExecutor, context: ExecutionContext):
        """Test evaluating input_data access."""
        result = executor.evaluate_expression("input", context)

        assert result == "test query"


# ============================================================================
# Case Matching Tests
# ============================================================================


class TestSwitchCaseMatching:
    """Test switch case matching logic."""

    @pytest.fixture
    def executor(self, switch_pipeline: SwitchPipeline) -> SwitchExecutor:
        """Provide a SwitchExecutor for testing."""
        return SwitchExecutor(pipeline=switch_pipeline)

    def test_find_case_exact_match(self, executor: SwitchExecutor):
        """Test finding a matching case."""
        cases = executor.pipeline.cases
        matched_key, matched_case = executor.find_matching_case("math", cases)

        assert matched_case is not None
        assert matched_key == "math"
        assert matched_case.method_id == MethodIdentifier.MATHEMATICAL_REASONING

    def test_find_case_no_match(self, executor: SwitchExecutor):
        """Test when no case matches."""
        cases = executor.pipeline.cases
        matched_key, matched_case = executor.find_matching_case("history", cases)

        assert matched_case is None
        assert matched_key is None


# ============================================================================
# Execution Tests
# ============================================================================


class TestSwitchExecutorExecution:
    """Test SwitchExecutor execution behavior."""

    async def test_execute_matching_case(
        self, switch_pipeline: SwitchPipeline, context: ExecutionContext
    ):
        """Test executing with matching case."""
        executor = SwitchExecutor(pipeline=switch_pipeline)

        mock_result = StageResult(
            stage_id="math-case",
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=["thought-1"],
            output_data={"content": "math reasoning output"},
        )

        with patch.object(executor, "execute_case", return_value=mock_result):
            result = await executor.execute(context)

        assert result.success is True
        assert result.stage_type == PipelineStageType.SWITCH
        assert result.metadata.get("case_matched") is True
        assert result.metadata.get("matched_key") == "math"

    async def test_execute_default_case(
        self, switch_pipeline: SwitchPipeline, context: ExecutionContext
    ):
        """Test executing with default case when no match."""
        context.variables["category"] = "unknown"
        executor = SwitchExecutor(pipeline=switch_pipeline)

        mock_result = StageResult(
            stage_id="default-case",
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=["thought-1"],
            output_data={"content": "default reasoning output"},
        )

        with patch.object(executor, "execute_case", return_value=mock_result):
            result = await executor.execute(context)

        assert result.success is True
        assert result.metadata.get("case_matched") is False
        assert result.metadata.get("default_used") is True

    async def test_execute_no_match_no_default(self, context: ExecutionContext):
        """Test executing with no match and no default case."""
        context.variables["category"] = "unknown"
        pipeline = SwitchPipeline(
            name="test",
            expression="category",
            cases={
                "math": MethodStage(
                    method_id=MethodIdentifier.MATHEMATICAL_REASONING,
                    name="math",
                ),
            },
            default=None,
        )
        executor = SwitchExecutor(pipeline=pipeline)

        result = await executor.execute(context)

        assert result.success is False
        assert "No matching case" in result.error


# ============================================================================
# Validation Tests
# ============================================================================


class TestSwitchExecutorValidation:
    """Test SwitchExecutor validation."""

    async def test_validate_valid_switch(self, switch_pipeline: SwitchPipeline):
        """Test validating a valid switch pipeline."""
        executor = SwitchExecutor(pipeline=switch_pipeline)

        errors = await executor.validate(switch_pipeline)

        assert errors == []

    async def test_validate_wrong_type(self, switch_pipeline: SwitchPipeline):
        """Test validation fails for wrong pipeline type."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single",
        )
        executor = SwitchExecutor(pipeline=switch_pipeline)

        errors = await executor.validate(stage)  # Wrong type

        assert len(errors) == 1
        assert "Expected SwitchPipeline" in errors[0]

    async def test_validate_missing_expression(self):
        """Test validation fails when expression is missing."""
        pipeline = SwitchPipeline(
            name="test",
            expression="",  # Empty expression
            cases={
                "test": MethodStage(
                    method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                    name="test",
                ),
            },
        )
        executor = SwitchExecutor(pipeline=pipeline)

        errors = await executor.validate(pipeline)

        assert any("must have an expression" in e for e in errors)

    async def test_validate_empty_cases(self):
        """Test validation fails with no cases and no default."""
        pipeline = SwitchPipeline(
            name="test",
            expression="variable",
            cases={},
            default=None,
        )
        executor = SwitchExecutor(pipeline=pipeline)

        errors = await executor.validate(pipeline)

        assert any("at least one case" in e for e in errors)
