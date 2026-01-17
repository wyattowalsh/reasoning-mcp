"""Unit tests for ProgramOfThoughts reasoning method.

This module provides comprehensive unit tests for the ProgramOfThoughts class,
testing initialization, code generation, execution simulation, phase transitions,
retry logic, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.program_of_thoughts import (
    PROGRAM_OF_THOUGHTS_METADATA,
    ProgramOfThoughts,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> ProgramOfThoughts:
    """Create a ProgramOfThoughts instance for testing."""
    return ProgramOfThoughts()


@pytest.fixture
async def initialized_method() -> ProgramOfThoughts:
    """Create and initialize a ProgramOfThoughts instance."""
    method = ProgramOfThoughts()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Create an active session for testing."""
    return Session().start()


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability."""
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="def solve(x): return x * 2")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestProgramOfThoughtsMetadata:
    """Tests for ProgramOfThoughts metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert PROGRAM_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.PROGRAM_OF_THOUGHTS

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert PROGRAM_OF_THOUGHTS_METADATA.name == "Program of Thoughts"

    def test_metadata_category(self) -> None:
        """Test metadata is in SPECIALIZED category."""
        assert PROGRAM_OF_THOUGHTS_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates no branching support."""
        assert PROGRAM_OF_THOUGHTS_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert PROGRAM_OF_THOUGHTS_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test metadata indicates no context required."""
        assert PROGRAM_OF_THOUGHTS_METADATA.requires_context is False

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        tags = PROGRAM_OF_THOUGHTS_METADATA.tags
        # Check for relevant code-related tags
        assert any(
            tag in tags for tag in ["code-generation", "python", "computational", "executable"]
        )

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert PROGRAM_OF_THOUGHTS_METADATA.complexity == 6

    def test_metadata_min_thoughts(self) -> None:
        """Test metadata specifies minimum thoughts."""
        assert PROGRAM_OF_THOUGHTS_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test metadata specifies maximum thoughts."""
        assert PROGRAM_OF_THOUGHTS_METADATA.max_thoughts == 8


# ============================================================================
# Initialization Tests
# ============================================================================


class TestProgramOfThoughtsInitialization:
    """Tests for ProgramOfThoughts initialization."""

    def test_default_initialization(self, method: ProgramOfThoughts) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._generated_code == ""
        assert method._execution_result == ""
        assert method._retry_count == 0

    async def test_initialize_method(self, method: ProgramOfThoughts) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "analyze"

    async def test_health_check_before_initialize(self, method: ProgramOfThoughts) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(
        self, initialized_method: ProgramOfThoughts
    ) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True

    def test_max_retries_constant(self) -> None:
        """Test MAX_RETRIES constant value."""
        assert ProgramOfThoughts.MAX_RETRIES == 2


# ============================================================================
# Property Tests
# ============================================================================


class TestProgramOfThoughtsProperties:
    """Tests for ProgramOfThoughts properties."""

    def test_identifier_property(self, method: ProgramOfThoughts) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.PROGRAM_OF_THOUGHTS

    def test_name_property(self, method: ProgramOfThoughts) -> None:
        """Test name property returns correct value."""
        assert method.name == "Program of Thoughts"

    def test_description_property(self, method: ProgramOfThoughts) -> None:
        """Test description property returns correct value."""
        assert "code" in method.description.lower()
        assert "execut" in method.description.lower()

    def test_category_property(self, method: ProgramOfThoughts) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestProgramOfThoughtsExecution:
    """Tests for basic execution of ProgramOfThoughts method."""

    async def test_execute_without_initialization_fails(
        self, method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Calculate 5+5")

    async def test_execute_creates_initial_thought(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "Calculate 5+5")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.PROGRAM_OF_THOUGHTS

    async def test_execute_sets_step_number(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() sets phase to analyze."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "analyze"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.PROGRAM_OF_THOUGHTS

    async def test_execute_content_includes_problem(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute() content includes the problem text."""
        problem = "Calculate the factorial of 5"
        result = await initialized_method.execute(active_session, problem)
        assert problem in result.content


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestProgramOfThoughtsContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.PROGRAM_OF_THOUGHTS,
            content="Test",
            metadata={"phase": "analyze"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_analyze_to_generate(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test transition from analyze to generate phase."""
        initial = await initialized_method.execute(active_session, "Calculate 2*3")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "generate"
        assert result.type == ThoughtType.REASONING
        assert result.parent_id == initial.id

    async def test_continue_from_generate_to_execute(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test transition from generate to execute phase."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, generate)

        assert result.metadata["phase"] == "execute"
        assert result.type == ThoughtType.ACTION

    async def test_continue_from_execute_to_interpret(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test transition from execute to interpret phase."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)
        result = await initialized_method.continue_reasoning(active_session, execute)

        assert result.metadata["phase"] == "interpret"
        assert result.type == ThoughtType.SYNTHESIS

    async def test_continue_from_interpret_to_conclude(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test transition from interpret to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)
        interpret = await initialized_method.continue_reasoning(active_session, execute)
        result = await initialized_method.continue_reasoning(active_session, interpret)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.step_number == 1

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.depth == 0

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == 1


# ============================================================================
# Code Generation Tests
# ============================================================================


class TestProgramOfThoughtsCodeGeneration:
    """Tests for code generation functionality."""

    async def test_generate_phase_produces_code_content(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test generate phase produces code-related content."""
        initial = await initialized_method.execute(active_session, "Calculate 2+2")
        generate = await initialized_method.continue_reasoning(active_session, initial)

        # Generate phase should have code-related content
        assert generate.metadata["phase"] == "generate"
        assert generate.content
        assert "code" in generate.content.lower() or "python" in generate.content.lower()

    async def test_generate_phase_follows_analysis(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test generate phase follows analysis phase."""
        initial = await initialized_method.execute(active_session, "Calculate 10/2")
        generate = await initialized_method.continue_reasoning(active_session, initial)

        # Generate phase should follow analysis
        assert initial.metadata["phase"] == "analyze"
        assert generate.metadata["phase"] == "generate"
        assert generate.parent_id == initial.id

    async def test_generate_phase_metadata(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test generate phase has proper metadata."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)

        # Generate phase should have appropriate metadata
        assert generate.metadata["phase"] == "generate"
        assert generate.step_number == 2


# ============================================================================
# Execution Simulation Tests
# ============================================================================


class TestProgramOfThoughtsExecutionSimulation:
    """Tests for execution simulation functionality."""

    async def test_execute_phase_follows_generate(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute phase follows generate phase."""
        initial = await initialized_method.execute(active_session, "Calculate 5*5")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)

        # Execute phase should follow generate
        assert generate.metadata["phase"] == "generate"
        assert execute.metadata["phase"] == "execute"
        assert execute.parent_id == generate.id

    async def test_execute_phase_has_execution_content(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execute phase has execution-related content."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)

        # Execute phase should have execution content
        assert execute.metadata["phase"] == "execute"
        assert execute.content
        assert "execut" in execute.content.lower()

    async def test_execution_result_stored(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execution result is stored."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)

        # Execute phase should complete
        assert execute.metadata["phase"] == "execute"
        assert execute.content  # Content should exist


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestProgramOfThoughtsRetryLogic:
    """Tests for retry logic on execution failure."""

    async def test_retry_count_initial(self, method: ProgramOfThoughts) -> None:
        """Test retry count is initially 0."""
        assert method._retry_count == 0

    async def test_retry_count_reset_on_initialize(self, method: ProgramOfThoughts) -> None:
        """Test retry count is reset on initialize."""
        method._retry_count = 2
        await method.initialize()
        assert method._retry_count == 0


# ============================================================================
# Phase Integration Tests
# ============================================================================


class TestProgramOfThoughtsPhaseIntegration:
    """Tests for phase integration and content flow."""

    async def test_analysis_phase_content(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test analysis phase produces meaningful content."""
        result = await initialized_method.execute(active_session, "Sort a list")

        # Analysis phase should have analysis content
        assert result.metadata["phase"] == "analyze"
        assert result.content
        assert "analysi" in result.content.lower() or "problem" in result.content.lower()

    async def test_interpret_phase_content(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test interpretation phase produces meaningful content."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)
        interpret = await initialized_method.continue_reasoning(active_session, execute)

        # Interpret phase should have interpretation content
        assert interpret.metadata["phase"] == "interpret"
        assert interpret.content
        assert "result" in interpret.content.lower() or "interpret" in interpret.content.lower()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestProgramOfThoughtsEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "Calculate the sum of " + " ".join(str(i) for i in range(100))
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Calculate: @#$%^&*() + 测试"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None

    async def test_input_with_quotes(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test execution with input containing quotes."""
        input_text = "Calculate length of \"hello 'world'\""
        result = await initialized_method.execute(active_session, input_text)
        assert result is not None
        assert result.type == ThoughtType.INITIAL


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestProgramOfThoughtsWorkflow:
    """Tests for complete PoT workflows."""

    async def test_full_workflow(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test complete PoT workflow from analysis to conclusion."""
        # Analyze phase
        initial = await initialized_method.execute(active_session, "Calculate 5 * 5")
        assert initial.metadata["phase"] == "analyze"
        assert initial.type == ThoughtType.INITIAL

        # Generate phase
        generate = await initialized_method.continue_reasoning(active_session, initial)
        assert generate.metadata["phase"] == "generate"
        assert generate.type == ThoughtType.REASONING

        # Execute phase
        execute = await initialized_method.continue_reasoning(active_session, generate)
        assert execute.metadata["phase"] == "execute"
        assert execute.type == ThoughtType.ACTION

        # Interpret phase
        interpret = await initialized_method.continue_reasoning(active_session, execute)
        assert interpret.metadata["phase"] == "interpret"
        assert interpret.type == ThoughtType.SYNTHESIS

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, interpret)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 5
        assert active_session.current_method == MethodIdentifier.PROGRAM_OF_THOUGHTS

    async def test_confidence_progression(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test that confidence changes through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)

        # Confidence should generally increase through phases
        assert initial.confidence > 0
        assert generate.confidence >= initial.confidence
        assert execute.confidence >= 0.7  # Execution should have good confidence

    async def test_metadata_includes_code_and_result(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test that metadata includes code and execution result."""
        initial = await initialized_method.execute(active_session, "Test")
        generate = await initialized_method.continue_reasoning(active_session, initial)
        execute = await initialized_method.continue_reasoning(active_session, generate)
        interpret = await initialized_method.continue_reasoning(active_session, execute)
        conclusion = await initialized_method.continue_reasoning(active_session, interpret)

        # Conclusion metadata should include code generated
        assert "generated_code" in conclusion.metadata or initialized_method._generated_code

    async def test_mathematical_problem_workflow(
        self, initialized_method: ProgramOfThoughts, active_session: Session
    ) -> None:
        """Test workflow with mathematical problem."""
        problem = "Calculate the area of a circle with radius 5"
        initial = await initialized_method.execute(active_session, problem)

        assert "circle" in initial.content.lower() or "area" in initial.content.lower()
        assert initial.type == ThoughtType.INITIAL
