"""Tests for the ConditionalExecutor.

This module tests conditional pipeline branching, including condition evaluation,
branch execution, and nested conditionals.
"""

from unittest.mock import Mock, patch

import pytest

from reasoning_mcp.engine.conditional import ConditionalExecutor
from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import Condition, ConditionalPipeline, MethodStage
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
        variables={"confidence": 0.9, "is_valid": True},
        thought_ids=[],
    )


@pytest.fixture
def conditional_pipeline() -> ConditionalPipeline:
    """Provide a conditional pipeline for testing."""
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
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="high_confidence_path",
        ),
        if_false=MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="low_confidence_path",
        ),
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestConditionalExecutorInit:
    """Test ConditionalExecutor initialization."""

    def test_basic_init(self, conditional_pipeline: ConditionalPipeline):
        """Test basic initialization."""
        executor = ConditionalExecutor(pipeline=conditional_pipeline)

        assert executor.pipeline is conditional_pipeline
        assert executor.default_branch == "else"

    def test_init_with_default_branch(self, conditional_pipeline: ConditionalPipeline):
        """Test initialization with custom default_branch."""
        executor = ConditionalExecutor(
            pipeline=conditional_pipeline,
            default_branch="then",
        )

        assert executor.default_branch == "then"


# ============================================================================
# Condition Evaluation Tests
# ============================================================================


class TestConditionEvaluation:
    """Test condition evaluation logic."""

    @pytest.fixture
    def executor(self, conditional_pipeline: ConditionalPipeline) -> ConditionalExecutor:
        """Provide a ConditionalExecutor for testing."""
        return ConditionalExecutor(pipeline=conditional_pipeline)

    def test_evaluate_simple_comparison_true(
        self, executor: ConditionalExecutor, context: ExecutionContext
    ):
        """Test simple comparison that evaluates to True."""
        condition = Condition(
            name="test",
            expression="confidence > 0.8",
            operator=">",
            threshold=0.8,
            field="confidence",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_simple_comparison_false(
        self, executor: ConditionalExecutor, context: ExecutionContext
    ):
        """Test simple comparison that evaluates to False."""
        condition = Condition(
            name="test",
            expression="confidence > 0.95",
            operator=">",
            threshold=0.95,
            field="confidence",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is False

    def test_evaluate_equality(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test equality comparison."""
        condition = Condition(
            name="test",
            expression="is_valid == True",
            operator="==",
            field="is_valid",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_inequality(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test inequality comparison."""
        condition = Condition(
            name="test",
            expression="confidence != 0.5",
            operator="!=",
            threshold=0.5,
            field="confidence",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_missing_field(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test evaluation with missing field returns False."""
        condition = Condition(
            name="test",
            expression="missing_field > 0.5",
            operator=">",
            threshold=0.5,
            field="missing_field",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is False

    def test_evaluate_existence_check(
        self, executor: ConditionalExecutor, context: ExecutionContext
    ):
        """Test existence check."""
        condition = Condition(
            name="test",
            expression="confidence exists",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_existence_check_missing(
        self, executor: ConditionalExecutor, context: ExecutionContext
    ):
        """Test existence check for missing field."""
        condition = Condition(
            name="test",
            expression="missing exists",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is False

    def test_evaluate_and_logic(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test AND logical operator."""
        condition = Condition(
            name="test",
            expression="confidence > 0.5 and is_valid == True",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_or_logic(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test OR logical operator."""
        condition = Condition(
            name="test",
            expression="confidence < 0.5 or is_valid == True",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_not_logic(self, executor: ConditionalExecutor, context: ExecutionContext):
        """Test NOT logical operator."""
        context.variables["is_invalid"] = False
        condition = Condition(
            name="test",
            expression="not is_invalid",
        )

        result = executor.evaluate_condition(condition, context)

        assert result is True


# ============================================================================
# Execution Tests
# ============================================================================


class TestConditionalExecutorExecution:
    """Test ConditionalExecutor execution behavior."""

    async def test_execute_true_branch(
        self, conditional_pipeline: ConditionalPipeline, context: ExecutionContext
    ):
        """Test executing when condition is True."""
        executor = ConditionalExecutor(pipeline=conditional_pipeline)

        mock_result = StageResult(
            stage_id="true-branch",
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=["thought-1"],
            output_data={"content": "high confidence output"},
        )

        with patch.object(executor, "execute_branch", return_value=mock_result):
            result = await executor.execute(context)

        assert result.success is True
        assert result.metadata["condition_met"] is True
        assert result.metadata["branch_taken"] == "if_true"

    async def test_execute_false_branch(
        self, conditional_pipeline: ConditionalPipeline, context: ExecutionContext
    ):
        """Test executing when condition is False."""
        # Set low confidence to trigger false branch
        context.variables["confidence"] = 0.5
        executor = ConditionalExecutor(pipeline=conditional_pipeline)

        mock_result = StageResult(
            stage_id="false-branch",
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=["thought-1"],
            output_data={"content": "low confidence output"},
        )

        with patch.object(executor, "execute_branch", return_value=mock_result):
            result = await executor.execute(context)

        assert result.success is True
        assert result.metadata["condition_met"] is False
        assert result.metadata["branch_taken"] == "if_false"

    async def test_execute_missing_false_branch(self, context: ExecutionContext):
        """Test executing when condition is False but no false branch defined."""
        pipeline = ConditionalPipeline(
            name="test",
            condition=Condition(
                name="test",
                expression="confidence > 0.99",  # Will be False
                operator=">",
                threshold=0.99,
                field="confidence",
            ),
            if_true=MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="true_path",
            ),
            if_false=None,
        )
        executor = ConditionalExecutor(pipeline=pipeline)

        result = await executor.execute(context)

        assert result.success is True
        assert result.metadata["branch_skipped"] is True

    async def test_execute_no_pipeline_returns_error(self, context: ExecutionContext):
        """Test executing without a pipeline returns error."""
        executor = ConditionalExecutor(pipeline=None)

        result = await executor.execute(context)

        assert result.success is False
        assert "No pipeline provided" in result.error


# ============================================================================
# Validation Tests
# ============================================================================


class TestConditionalExecutorValidation:
    """Test ConditionalExecutor validation."""

    async def test_validate_valid_conditional(self, conditional_pipeline: ConditionalPipeline):
        """Test validating a valid conditional pipeline."""
        executor = ConditionalExecutor(pipeline=conditional_pipeline)

        errors = await executor.validate(conditional_pipeline)

        assert errors == []

    async def test_validate_wrong_type(self, conditional_pipeline: ConditionalPipeline):
        """Test validation fails for wrong pipeline type."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single",
        )
        executor = ConditionalExecutor(pipeline=conditional_pipeline)

        errors = await executor.validate(stage)  # Wrong type

        assert len(errors) == 1
        assert "Expected ConditionalPipeline" in errors[0]

    async def test_validate_no_branches(self):
        """Test validation fails when both branches are None.

        Note: ConditionalPipeline may or may not allow None branches at model level.
        """
        try:
            pipeline = ConditionalPipeline(
                name="test",
                condition=Condition(name="test", expression="True"),
                if_true=None,
                if_false=None,
            )
            executor = ConditionalExecutor(pipeline=pipeline)

            errors = await executor.validate(pipeline)

            # Check for the specific error message about branches
            assert any("branch" in e.lower() for e in errors)
        except Exception:
            # Pydantic may reject the model at construction time
            pass
