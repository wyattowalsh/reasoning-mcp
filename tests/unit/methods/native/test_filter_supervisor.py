"""Unit tests for Filter Supervisor (FS-C) reasoning method.

This module provides comprehensive tests for the FilterSupervisor method implementation,
covering initialization, execution, candidate generation, filtering, supervision,
self-correction, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.filter_supervisor import (
    FILTER_SUPERVISOR_METADATA,
    FilterSupervisor,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> FilterSupervisor:
    """Create a FilterSupervisor method instance for testing.

    Returns:
        A fresh FilterSupervisor instance
    """
    return FilterSupervisor()


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A new Session instance in ACTIVE status
    """
    return Session().start()


@pytest.fixture
def sample_problem() -> str:
    """Provide a sample problem for testing.

    Returns:
        A sample problem string
    """
    return "Design an algorithm to efficiently sort a large dataset with limited memory."


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Candidate A: Use external merge sort. Quality: 0.85")
    return ctx


class TestFilterSupervisorInitialization:
    """Tests for FilterSupervisor initialization and setup."""

    def test_create_method(self, method: FilterSupervisor) -> None:
        """Test that FilterSupervisor can be instantiated."""
        assert method is not None
        assert isinstance(method, FilterSupervisor)

    def test_initial_state(self, method: FilterSupervisor) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidates == []
        assert method._filtered_candidates == []

    async def test_initialize(self, method: FilterSupervisor) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidates == []
        assert method._filtered_candidates == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = FilterSupervisor()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"
        method._candidates = [{"id": "A", "content": "test"}]

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidates == []

    async def test_health_check_not_initialized(self, method: FilterSupervisor) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: FilterSupervisor) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestFilterSupervisorProperties:
    """Tests for FilterSupervisor property accessors."""

    def test_identifier_property(self, method: FilterSupervisor) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.FILTER_SUPERVISOR

    def test_name_property(self, method: FilterSupervisor) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Filter Supervisor"

    def test_description_property(self, method: FilterSupervisor) -> None:
        """Test that description returns the correct method description."""
        assert "filter" in method.description.lower()
        assert "supervision" in method.description.lower()

    def test_category_property(self, method: FilterSupervisor) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestFilterSupervisorMetadata:
    """Tests for FILTER_SUPERVISOR metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert FILTER_SUPERVISOR_METADATA.identifier == MethodIdentifier.FILTER_SUPERVISOR

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert FILTER_SUPERVISOR_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"filtering", "supervision", "self-correction", "candidates"}
        assert expected_tags.issubset(FILTER_SUPERVISOR_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert FILTER_SUPERVISOR_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert FILTER_SUPERVISOR_METADATA.supports_revision is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert FILTER_SUPERVISOR_METADATA.complexity == 7


class TestFilterSupervisorExecution:
    """Tests for FilterSupervisor execute() method."""

    async def test_execute_basic(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.FILTER_SUPERVISOR

    async def test_execute_without_initialization_raises(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_generate_phase(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets generate phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "generate"
        assert "candidates" in thought.metadata

    async def test_execute_generates_candidates(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute generates candidate solutions."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert len(method._candidates) > 0
        for candidate in method._candidates:
            assert "id" in candidate
            assert "content" in candidate
            assert "quality" in candidate

    async def test_execute_with_execution_context(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test execute with execution context for sampling."""
        await method.initialize()
        thought = await method.execute(
            session,
            sample_problem,
            execution_context=mock_execution_context,
        )

        assert thought is not None
        assert thought.content != ""


class TestFilterSupervisorContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_generate_to_filter(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from generate to filter phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "filter"
        assert continuation.type == ThoughtType.VERIFICATION

    async def test_continue_filter_to_supervise(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from filter to supervise phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)

        supervise = await method.continue_reasoning(session, filter_step)

        assert supervise is not None
        assert supervise.metadata.get("phase") == "supervise"
        assert supervise.type == ThoughtType.REASONING

    async def test_continue_supervise_to_correct(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from supervise to correct phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)

        correct = await method.continue_reasoning(session, supervise)

        assert correct is not None
        assert correct.metadata.get("phase") == "correct"
        assert correct.type == ThoughtType.REVISION

    async def test_continue_to_conclusion(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)
        correct = await method.continue_reasoning(session, supervise)

        conclusion = await method.continue_reasoning(session, correct)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: FilterSupervisor,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.FILTER_SUPERVISOR,
            content="Test",
            metadata={"phase": "generate"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestCandidateGeneration:
    """Tests for candidate generation."""

    async def test_heuristic_generates_multiple_candidates(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that heuristic generates multiple candidates."""
        await method.initialize()
        await method.execute(session, sample_problem)

        assert len(method._candidates) >= 3

    async def test_candidates_have_varying_quality(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that candidates have varying quality scores."""
        await method.initialize()
        await method.execute(session, sample_problem)

        qualities = [c["quality"] for c in method._candidates]
        # Should have at least some variation
        assert max(qualities) != min(qualities)


class TestQualityFiltering:
    """Tests for quality filtering."""

    async def test_filtering_applies_threshold(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that filtering applies quality threshold."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        # All filtered candidates should pass threshold
        for candidate in method._filtered_candidates:
            assert candidate["quality"] >= 0.75

    async def test_filtering_reduces_candidates(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that filtering can reduce number of candidates."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)

        # Filtered should be <= total candidates
        assert len(method._filtered_candidates) <= len(method._candidates)


class TestSupervision:
    """Tests for supervision functionality."""

    async def test_supervision_analyzes_candidates(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that supervision phase analyzes filtered candidates."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)

        # Content should reference supervision analysis
        assert "supervise" in supervise.content.lower() or "analysis" in supervise.content.lower()


class TestSelfCorrection:
    """Tests for self-correction functionality."""

    async def test_correction_phase_refines_solution(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that correction phase refines the solution."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)
        correct = await method.continue_reasoning(session, supervise)

        # Content should reference correction/refinement
        assert "correct" in correct.content.lower() or "refine" in correct.content.lower()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: FilterSupervisor,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: FilterSupervisor,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Design an algorithm: " + "requirement " * 500

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: FilterSupervisor,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Design: O(n log n) → O(n) optimization with α, β constraints"

        thought = await method.execute(session, problem)

        assert thought is not None

    async def test_unicode_in_problem(
        self,
        method: FilterSupervisor,
        session: Session,
    ) -> None:
        """Test execution with Unicode characters."""
        await method.initialize()
        problem = "设计一个算法来优化数据处理流程"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "generate"

        # Continue through all phases
        filter_step = await method.continue_reasoning(session, initial)
        assert filter_step.metadata.get("phase") == "filter"

        supervise = await method.continue_reasoning(session, filter_step)
        assert supervise.metadata.get("phase") == "supervise"

        correct = await method.continue_reasoning(session, supervise)
        assert correct.metadata.get("phase") == "correct"

        conclude = await method.continue_reasoning(session, correct)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION

    async def test_session_thought_count_updates(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_thought_parent_chain(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought parent chain is correct."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)

        assert initial.parent_id is None
        assert filter_step.parent_id == initial.id
        assert supervise.parent_id == filter_step.id

    async def test_thought_depth_increments(
        self,
        method: FilterSupervisor,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        supervise = await method.continue_reasoning(session, filter_step)

        assert initial.depth == 0
        assert filter_step.depth == 1
        assert supervise.depth == 2
