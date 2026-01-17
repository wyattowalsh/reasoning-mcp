"""Comprehensive tests for HintOfThought reasoning method.

This module provides complete test coverage for the HintOfThought method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Phase transitions (hint -> solution -> verify)
- Hint generation and application
- Solution development
- Verification against hints
- Configuration options (hint_types)
- Continue reasoning flow
- Edge cases

The tests aim for 90%+ coverage of the HintOfThought implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.hint_of_thought import (
    HINT_OF_THOUGHT_METADATA,
    HintOfThought,
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
def method() -> HintOfThought:
    """Provide a HintOfThought method instance for testing.

    Returns:
        HintOfThought instance (uninitialized).
    """
    return HintOfThought()


@pytest.fixture
async def initialized_method() -> HintOfThought:
    """Provide an initialized HintOfThought method instance.

    Returns:
        Initialized HintOfThought instance.
    """
    method = HintOfThought()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.fixture
def algorithm_input() -> str:
    """Provide an algorithm problem input.

    Returns:
        Algorithm problem for testing.
    """
    return "Implement quicksort algorithm"


@pytest.fixture
def coding_input() -> str:
    """Provide a coding problem input.

    Returns:
        Coding problem for testing.
    """
    return "Write a function to find the longest palindromic substring"


@pytest.fixture
def planning_input() -> str:
    """Provide a planning problem input.

    Returns:
        Planning problem for testing.
    """
    return "Design a system to process real-time data streams"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMetadata:
    """Test suite for HintOfThought metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert HINT_OF_THOUGHT_METADATA.identifier == MethodIdentifier.HINT_OF_THOUGHT

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert HINT_OF_THOUGHT_METADATA.name == "Hint of Thought"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert HINT_OF_THOUGHT_METADATA.category == MethodCategory.CORE

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert HINT_OF_THOUGHT_METADATA.complexity == 3
        assert 1 <= HINT_OF_THOUGHT_METADATA.complexity <= 10

    def test_metadata_no_revision(self):
        """Test that metadata indicates no revision support."""
        assert HINT_OF_THOUGHT_METADATA.supports_revision is False

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert HINT_OF_THOUGHT_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "zero-shot",
            "structural-hints",
            "guidance",
            "pseudocode",
            "decomposition",
        }
        assert expected_tags.issubset(HINT_OF_THOUGHT_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert HINT_OF_THOUGHT_METADATA.min_thoughts == 2

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert HINT_OF_THOUGHT_METADATA.max_thoughts == 5

    def test_metadata_best_for(self):
        """Test that metadata specifies best use cases."""
        assert "zero-shot problem solving" in HINT_OF_THOUGHT_METADATA.best_for
        assert "algorithm design" in HINT_OF_THOUGHT_METADATA.best_for
        assert "structured problem decomposition" in HINT_OF_THOUGHT_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test that metadata specifies not recommended cases."""
        assert "simple factual queries" in HINT_OF_THOUGHT_METADATA.not_recommended_for
        assert "creative writing" in HINT_OF_THOUGHT_METADATA.not_recommended_for


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test suite for HintOfThought initialization."""

    def test_create_method(self, method: HintOfThought):
        """Test creating a HintOfThought instance."""
        assert isinstance(method, HintOfThought)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: HintOfThought):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.HINT_OF_THOUGHT
        assert method.name == "Hint of Thought"
        assert method.category == MethodCategory.CORE
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: HintOfThought):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "hint"

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = HintOfThought()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "solution"

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._current_phase == "hint"

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: HintOfThought):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: HintOfThought):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestExecution:
    """Test suite for basic HintOfThought execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: HintOfThought, session: Session, algorithm_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=algorithm_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.HINT_OF_THOUGHT
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metadata(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == algorithm_input
        assert thought.metadata["phase"] == "hint"
        assert "structural_hints" in thought.metadata
        assert thought.metadata["reasoning_type"] == "hint_of_thought"

    @pytest.mark.asyncio
    async def test_execute_generates_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute generates structural hints."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        hints = thought.metadata["structural_hints"]
        assert isinstance(hints, list)
        assert len(hints) > 0
        assert all(isinstance(h, str) for h in hints)

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute sets initial confidence."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert thought.confidence == 0.7
        assert thought.quality_score == 0.7

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.HINT_OF_THOUGHT
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test execute with custom context."""
        context = {"hint_types": ["decomposition", "algorithm"], "custom_key": "value"}

        thought = await initialized_method.execute(
            session=session, input_text=algorithm_input, context=context
        )

        assert thought.metadata["hint_types"] == ["decomposition", "algorithm"]
        assert thought.metadata["context"]["custom_key"] == "value"

    @pytest.mark.asyncio
    async def test_execute_default_hint_types(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that execute uses default hint types."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        hint_types = thought.metadata["hint_types"]
        assert "decomposition" in hint_types
        assert "algorithm" in hint_types
        assert "ordering" in hint_types
        assert "edge_cases" in hint_types


# ============================================================================
# Hint Generation Tests
# ============================================================================


class TestHintGeneration:
    """Test suite for hint generation."""

    @pytest.mark.asyncio
    async def test_decomposition_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that decomposition hints are generated."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["decomposition"]},
        )

        hints = thought.metadata["structural_hints"]
        assert any("decomp" in h.lower() or "break" in h.lower() for h in hints)

    @pytest.mark.asyncio
    async def test_algorithm_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that algorithm pattern hints are generated."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["algorithm"]},
        )

        hints = thought.metadata["structural_hints"]
        assert any("algorithm" in h.lower() or "pattern" in h.lower() for h in hints)

    @pytest.mark.asyncio
    async def test_ordering_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that step ordering hints are generated."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["ordering"]},
        )

        hints = thought.metadata["structural_hints"]
        assert any("order" in h.lower() or "step" in h.lower() for h in hints)

    @pytest.mark.asyncio
    async def test_edge_case_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that edge case hints are generated."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["edge_cases"]},
        )

        hints = thought.metadata["structural_hints"]
        assert any("edge" in h.lower() or "case" in h.lower() for h in hints)

    @pytest.mark.asyncio
    async def test_multiple_hint_types(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test generating multiple hint types."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["decomposition", "algorithm", "edge_cases"]},
        )

        hints = thought.metadata["structural_hints"]
        # Should have hints from all types
        assert len(hints) >= 3

    @pytest.mark.asyncio
    async def test_hint_content_structure(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that hint content has proper structure."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        content = thought.content
        assert "Structural Hints" in content
        assert algorithm_input in content
        assert "Hint Types:" in content
        assert "Total Hints:" in content


# ============================================================================
# Solution Phase Tests
# ============================================================================


class TestSolutionPhase:
    """Test suite for solution phase."""

    @pytest.mark.asyncio
    async def test_solution_phase_after_hint(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that solution follows hint phase."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert solution.type == ThoughtType.CONTINUATION
        assert solution.metadata["phase"] == "solution"
        assert solution.parent_id == hint.id
        assert solution.step_number == 2

    @pytest.mark.asyncio
    async def test_solution_has_hints_applied(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that solution tracks hints applied."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert "hints_applied" in solution.metadata
        hints_applied = solution.metadata["hints_applied"]
        assert isinstance(hints_applied, list)
        assert len(hints_applied) > 0

    @pytest.mark.asyncio
    async def test_solution_references_original_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that solution references original hints."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert "original_hints" in solution.metadata
        original_hints = solution.metadata["original_hints"]
        assert original_hints == hint.metadata["structural_hints"]

    @pytest.mark.asyncio
    async def test_solution_with_guidance(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test solution phase with guidance."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        guidance_text = "Focus on time complexity optimization"
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint, guidance=guidance_text
        )

        assert "guidance" in solution.metadata
        assert solution.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_solution_confidence(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test solution phase confidence."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert solution.confidence == 0.8
        assert solution.quality_score == 0.8

    @pytest.mark.asyncio
    async def test_solution_content_structure(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that solution content has proper structure."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        content = solution.content
        assert "Solution" in content
        assert "Applying Hints" in content
        assert "Hints applied:" in content


# ============================================================================
# Verification Tests
# ============================================================================


class TestVerification:
    """Test suite for verification phase."""

    @pytest.mark.asyncio
    async def test_verification_phase_after_solution(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that verification follows solution phase."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert verify.type == ThoughtType.CONCLUSION
        assert verify.metadata["phase"] == "verify"
        assert verify.parent_id == solution.id
        assert verify.step_number == 3

    @pytest.mark.asyncio
    async def test_verification_has_result(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that verification includes result."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert "verification_result" in verify.metadata
        result = verify.metadata["verification_result"]
        assert isinstance(result, dict)
        assert "passed" in result

    @pytest.mark.asyncio
    async def test_verification_checks_coverage(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that verification checks hint coverage."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        result = verify.metadata["verification_result"]
        assert "coverage" in result
        assert 0.0 <= result["coverage"] <= 1.0

    @pytest.mark.asyncio
    async def test_verification_tracks_hints(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that verification tracks hints applied."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert "hints_applied" in verify.metadata
        hints_applied = verify.metadata["hints_applied"]
        assert hints_applied == solution.metadata["hints_applied"]

    @pytest.mark.asyncio
    async def test_verification_confidence(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test verification phase confidence."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert verify.confidence == 0.85
        assert verify.quality_score == 0.85

    @pytest.mark.asyncio
    async def test_verification_content_structure(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that verification content has proper structure."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        content = verify.content
        assert "Verification" in content
        assert "Hints to verify:" in content
        assert "Hints applied:" in content


# ============================================================================
# Phase Flow Tests
# ============================================================================


class TestPhaseFlow:
    """Test suite for complete phase flow."""

    @pytest.mark.asyncio
    async def test_complete_flow_hint_to_verify(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test complete flow: hint -> solution -> verify."""
        # Hint phase
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert hint.metadata["phase"] == "hint"

        # Solution phase
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        assert solution.metadata["phase"] == "solution"

        # Verify phase
        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )
        assert verify.metadata["phase"] == "verify"

    @pytest.mark.asyncio
    async def test_phase_transitions(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that phases transition correctly."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert initialized_method._current_phase == "hint"

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._current_phase == "solution"

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._current_phase == "verify"

    @pytest.mark.asyncio
    async def test_step_counter_increments(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that step counter increments through phases."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert hint.step_number == 1

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        assert solution.step_number == 2

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )
        assert verify.step_number == 3

    @pytest.mark.asyncio
    async def test_depth_increases(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that depth increases through phases."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert hint.depth == 0

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        assert solution.depth == 1

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )
        assert verify.depth == 2

    @pytest.mark.asyncio
    async def test_parent_child_relationships(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test parent-child relationships through phases."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert hint.parent_id is None

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        assert solution.parent_id == hint.id

        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )
        assert verify.parent_id == solution.id

    @pytest.mark.asyncio
    async def test_conclusion_after_verify(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that conclusion follows verification."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        conclude = await initialized_method.continue_reasoning(
            session=session, previous_thought=verify
        )

        assert conclude.type == ThoughtType.CONCLUSION
        assert conclude.metadata["phase"] == "conclude"
        assert conclude.metadata.get("final", False) is True


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: HintOfThought, session: Session, algorithm_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HINT_OF_THOUGHT,
            content="Test",
            metadata={"phase": "hint"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that continue_reasoning increments step counter."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert hint.step_number == 1

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        assert solution.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test continue_reasoning with guidance parameter."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        guidance_text = "Optimize for space complexity"
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint, guidance=guidance_text
        )

        assert "guidance" in solution.metadata
        assert solution.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test continue_reasoning with context parameter."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        context = {"additional_info": "test data"}
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint, context=context
        )

        assert "context" in solution.metadata
        assert solution.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that continue_reasoning adds thought to session."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        count_after_hint = session.thought_count

        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert session.thought_count == count_after_hint + 1
        assert solution.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_fallback_phase_handling(
        self, initialized_method: HintOfThought, session: Session
    ):
        """Test fallback handling for unknown phase."""
        # Create a thought with unknown phase
        hint = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        hint.metadata["phase"] = "unknown_phase"

        # Continue should fallback to conclusion
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert thought.type == ThoughtType.CONCLUSION
        assert thought.metadata["phase"] == "conclude"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: HintOfThought, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)
        assert len(thought.metadata["structural_hints"]) > 0

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: HintOfThought, session: Session):
        """Test handling of very long input."""
        long_input = "Implement " + "complex " * 100 + "algorithm"
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=algorithm_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=algorithm_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_hint_types(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test handling of empty hint types."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": []},
        )
        # Should still have metadata but with empty hints
        assert "structural_hints" in thought.metadata
        assert "hint_types" in thought.metadata

    @pytest.mark.asyncio
    async def test_custom_single_hint_type(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test with single custom hint type."""
        thought = await initialized_method.execute(
            session=session,
            input_text=algorithm_input,
            context={"hint_types": ["algorithm"]},
        )

        assert thought.metadata["hint_types"] == ["algorithm"]
        assert len(thought.metadata["structural_hints"]) >= 1

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_execute(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that step counter resets on new execute."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=algorithm_input)
        assert thought1.step_number == 1

        # Continue
        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )
        assert thought2.step_number == 2

        # New execution should reset
        session2 = Session().start()
        thought3 = await initialized_method.execute(session=session2, input_text=algorithm_input)
        assert thought3.step_number == 1

    @pytest.mark.asyncio
    async def test_phase_resets_on_execute(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that phase resets on new execute."""
        # First execution
        hint1 = await initialized_method.execute(session=session, input_text=algorithm_input)
        await initialized_method.continue_reasoning(session=session, previous_thought=hint1)
        assert initialized_method._current_phase == "solution"

        # New execution should reset phase
        session2 = Session().start()
        hint2 = await initialized_method.execute(session=session2, input_text=algorithm_input)
        assert initialized_method._current_phase == "hint"
        assert hint2.metadata["phase"] == "hint"


# ============================================================================
# Different Problem Types Tests
# ============================================================================


class TestDifferentProblemTypes:
    """Test suite for different problem types."""

    @pytest.mark.asyncio
    async def test_algorithm_problem(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test with algorithm problem."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert thought.type == ThoughtType.INITIAL
        assert len(thought.metadata["structural_hints"]) > 0
        assert "quicksort" in algorithm_input.lower()

    @pytest.mark.asyncio
    async def test_coding_problem(
        self,
        initialized_method: HintOfThought,
        session: Session,
        coding_input: str,
    ):
        """Test with coding problem."""
        thought = await initialized_method.execute(session=session, input_text=coding_input)

        assert thought.type == ThoughtType.INITIAL
        assert len(thought.metadata["structural_hints"]) > 0

    @pytest.mark.asyncio
    async def test_planning_problem(
        self,
        initialized_method: HintOfThought,
        session: Session,
        planning_input: str,
    ):
        """Test with system design problem."""
        thought = await initialized_method.execute(session=session, input_text=planning_input)

        assert thought.type == ThoughtType.INITIAL
        assert len(thought.metadata["structural_hints"]) > 0

    @pytest.mark.asyncio
    async def test_different_problems_same_session(
        self, initialized_method: HintOfThought, session: Session
    ):
        """Test multiple different problems in same session."""
        # Problem 1
        thought1 = await initialized_method.execute(session=session, input_text="Sort an array")
        assert thought1.step_number == 1

        # Problem 2 (new execution resets)
        thought2 = await initialized_method.execute(
            session=session, input_text="Find shortest path"
        )
        assert thought2.step_number == 1
        assert thought2.metadata["input"] != thought1.metadata["input"]


# ============================================================================
# Confidence and Quality Tests
# ============================================================================


class TestConfidenceAndQuality:
    """Test suite for confidence and quality scores."""

    @pytest.mark.asyncio
    async def test_hint_phase_scores(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test hint phase confidence and quality."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)

        assert hint.confidence == 0.7
        assert hint.quality_score == 0.7

    @pytest.mark.asyncio
    async def test_solution_phase_scores(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test solution phase confidence and quality."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )

        assert solution.confidence == 0.8
        assert solution.quality_score == 0.8

    @pytest.mark.asyncio
    async def test_verify_phase_scores(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test verify phase confidence and quality."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert verify.confidence == 0.85
        assert verify.quality_score == 0.85

    @pytest.mark.asyncio
    async def test_scores_increase_through_phases(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that confidence and quality increase through phases."""
        hint = await initialized_method.execute(session=session, input_text=algorithm_input)
        solution = await initialized_method.continue_reasoning(
            session=session, previous_thought=hint
        )
        verify = await initialized_method.continue_reasoning(
            session=session, previous_thought=solution
        )

        assert solution.confidence > hint.confidence
        assert verify.confidence > solution.confidence
        assert solution.quality_score > hint.quality_score
        assert verify.quality_score > solution.quality_score

    @pytest.mark.asyncio
    async def test_scores_bounded(
        self,
        initialized_method: HintOfThought,
        session: Session,
        algorithm_input: str,
    ):
        """Test that scores stay within valid bounds."""
        thought = await initialized_method.execute(session=session, input_text=algorithm_input)

        # Go through all phases
        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should stay within bounds
        assert 0.0 <= thought.confidence <= 1.0
        assert 0.0 <= thought.quality_score <= 1.0
