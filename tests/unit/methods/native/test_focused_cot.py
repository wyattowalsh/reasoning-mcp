"""Unit tests for Focused Chain-of-Thought (F-CoT) reasoning method.

This module provides comprehensive tests for the FocusedCot method implementation,
covering initialization, execution, condition identification, information filtering,
focused reasoning, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.focused_cot import (
    FOCUSED_COT_METADATA,
    FocusedCot,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> FocusedCot:
    """Create a FocusedCot method instance for testing.

    Returns:
        A fresh FocusedCot instance
    """
    return FocusedCot()


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
        A sample problem string with distractors
    """
    return (
        "A train travels at 60 mph for 2 hours. The train has 8 carriages, "
        "each painted blue. The weather was sunny. How far did the train travel?"
    )


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Key conditions:\n1. Speed: 60 mph\n2. Time: 2 hours")
    return ctx


class TestFocusedCotInitialization:
    """Tests for FocusedCot initialization and setup."""

    def test_create_method(self, method: FocusedCot) -> None:
        """Test that FocusedCot can be instantiated."""
        assert method is not None
        assert isinstance(method, FocusedCot)

    def test_initial_state(self, method: FocusedCot) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "identify_conditions"
        assert method._use_sampling is False

    async def test_initialize(self, method: FocusedCot) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "identify_conditions"

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = FocusedCot()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "identify_conditions"

    async def test_health_check_not_initialized(self, method: FocusedCot) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: FocusedCot) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestFocusedCotProperties:
    """Tests for FocusedCot property accessors."""

    def test_identifier_property(self, method: FocusedCot) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.FOCUSED_COT

    def test_name_property(self, method: FocusedCot) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Focused Chain-of-Thought"

    def test_description_property(self, method: FocusedCot) -> None:
        """Test that description returns the correct method description."""
        assert "condition" in method.description.lower()
        assert "distractor" in method.description.lower()

    def test_category_property(self, method: FocusedCot) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.HIGH_VALUE


class TestFocusedCotMetadata:
    """Tests for FOCUSED_COT metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert FOCUSED_COT_METADATA.identifier == MethodIdentifier.FOCUSED_COT

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert FOCUSED_COT_METADATA.category == MethodCategory.HIGH_VALUE

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"condition-first", "distractor-filtering", "focused-reasoning"}
        assert expected_tags.issubset(FOCUSED_COT_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert FOCUSED_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert FOCUSED_COT_METADATA.supports_revision is False

    def test_metadata_complexity(self) -> None:
        """Test that metadata has medium complexity rating."""
        assert FOCUSED_COT_METADATA.complexity == 4

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata has correct min_thoughts."""
        assert FOCUSED_COT_METADATA.min_thoughts == 5


class TestFocusedCotExecution:
    """Tests for FocusedCot execute() method."""

    async def test_execute_basic(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.FOCUSED_COT

    async def test_execute_without_initialization_raises(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_identify_conditions_phase(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets identify_conditions phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "identify_conditions"
        assert thought.metadata.get("reasoning_type") == "focused_cot"

    async def test_execute_identifies_key_conditions(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute identifies key conditions."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert "key_conditions" in thought.metadata
        assert len(thought.metadata["key_conditions"]) > 0

    async def test_execute_with_execution_context(
        self,
        method: FocusedCot,
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


class TestFocusedCotContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_identify_to_filter(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from identify_conditions to filter_relevant phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "filter_relevant"
        assert continuation.type == ThoughtType.CONTINUATION

    async def test_continue_filter_to_focus(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from filter_relevant to focus_reasoning phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)

        focus = await method.continue_reasoning(session, filter_step)

        assert focus is not None
        assert focus.metadata.get("phase") == "focus_reasoning"
        assert focus.type == ThoughtType.CONTINUATION

    async def test_continue_focus_to_derive(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from focus_reasoning to derive_answer phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        derive = await method.continue_reasoning(session, focus)

        assert derive is not None
        assert derive.metadata.get("phase") == "derive_answer"
        assert derive.type == ThoughtType.SYNTHESIS

    async def test_continue_to_conclusion(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)
        derive = await method.continue_reasoning(session, focus)

        conclusion = await method.continue_reasoning(session, derive)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.FOCUSED_COT,
            content="Test",
            metadata={"phase": "identify_conditions"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestConditionIdentification:
    """Tests for condition identification functionality."""

    async def test_conditions_extracted(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conditions are extracted from the problem."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        conditions = thought.metadata.get("key_conditions", [])
        assert len(conditions) > 0

    async def test_conditions_referenced_in_content(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conditions are referenced in thought content."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert "condition" in thought.content.lower()


class TestInformationFiltering:
    """Tests for information filtering functionality."""

    async def test_relevant_info_identified(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that relevant information is identified."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)

        assert "relevant_info" in filter_step.metadata
        assert len(filter_step.metadata["relevant_info"]) > 0

    async def test_irrelevant_info_identified(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that irrelevant information is identified."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)

        assert "irrelevant_info" in filter_step.metadata
        assert len(filter_step.metadata["irrelevant_info"]) > 0

    async def test_distractors_filtered(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that distractors are filtered out."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)

        assert "filtered_distractors" in filter_step.metadata
        assert len(filter_step.metadata["filtered_distractors"]) > 0


class TestFocusedReasoning:
    """Tests for focused reasoning functionality."""

    async def test_focus_reasoning_uses_relevant_info(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that focused reasoning uses relevant information."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        # Content should mention relevant information or focused reasoning
        content_lower = focus.content.lower()
        assert "relevant" in content_lower or "focus" in content_lower

    async def test_focus_reasoning_excludes_distractors(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that focused reasoning excludes distractors."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        # Content should mention filtering out distractors
        content_lower = focus.content.lower()
        assert "distractor" in content_lower or "filter" in content_lower


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_problem_without_distractors(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test execution with problem that has no distractors."""
        await method.initialize()
        clean_problem = "If x=5 and y=3, what is x + y?"

        thought = await method.execute(session, clean_problem)

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Calculate: " + "context " * 500 + "x + y = ?"

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None

    async def test_unicode_in_problem(
        self,
        method: FocusedCot,
        session: Session,
    ) -> None:
        """Test execution with Unicode characters."""
        await method.initialize()
        problem = "解决问题: 如果 α=5 且 β=3, 求 α+β"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "identify_conditions"

        # Continue through all phases
        filter_step = await method.continue_reasoning(session, initial)
        assert filter_step.metadata.get("phase") == "filter_relevant"

        focus = await method.continue_reasoning(session, filter_step)
        assert focus.metadata.get("phase") == "focus_reasoning"

        derive = await method.continue_reasoning(session, focus)
        assert derive.metadata.get("phase") == "derive_answer"

        conclude = await method.continue_reasoning(session, derive)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION

    async def test_session_thought_count_updates(
        self,
        method: FocusedCot,
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
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought parent chain is correct."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        assert initial.parent_id is None
        assert filter_step.parent_id == initial.id
        assert focus.parent_id == filter_step.id

    async def test_thought_depth_increments(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        assert initial.depth == 0
        assert filter_step.depth == 1
        assert focus.depth == 2

    async def test_metadata_preserved_through_chain(
        self,
        method: FocusedCot,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that key metadata is preserved through reasoning chain."""
        await method.initialize()

        initial = await method.execute(session, sample_problem)
        filter_step = await method.continue_reasoning(session, initial)
        focus = await method.continue_reasoning(session, filter_step)

        # Key conditions should be preserved
        assert "key_conditions" in initial.metadata
        assert "key_conditions" in filter_step.metadata
        assert "key_conditions" in focus.metadata
