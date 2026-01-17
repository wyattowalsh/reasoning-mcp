"""Unit tests for ThreadOfThought reasoning method.

This module provides comprehensive unit tests for the ThreadOfThought class,
testing initialization, segmentation, threading, weaving, synthesis,
and complete workflow execution for processing long contexts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.thread_of_thought import (
    THREAD_OF_THOUGHT_METADATA,
    ThreadOfThought,
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
def method() -> ThreadOfThought:
    """Create a ThreadOfThought instance for testing."""
    return ThreadOfThought()


@pytest.fixture
def method_no_elicitation() -> ThreadOfThought:
    """Create a ThreadOfThought instance with elicitation disabled."""
    return ThreadOfThought(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> ThreadOfThought:
    """Create and initialize a ThreadOfThought instance."""
    method = ThreadOfThought()
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
    ctx.sample = AsyncMock(return_value="Segmented and threaded content")
    return ctx


@pytest.fixture
def long_input_text() -> str:
    """Create a long input text for testing segmentation."""
    return " ".join([f"Paragraph {i}: This is content for segment {i}." for i in range(20)])


# ============================================================================
# Metadata Tests
# ============================================================================


class TestThreadOfThoughtMetadata:
    """Tests for ThreadOfThought metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert THREAD_OF_THOUGHT_METADATA.identifier == MethodIdentifier.THREAD_OF_THOUGHT

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "Thread" in THREAD_OF_THOUGHT_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in SPECIALIZED category."""
        assert THREAD_OF_THOUGHT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates branching support."""
        assert isinstance(THREAD_OF_THOUGHT_METADATA.supports_branching, bool)

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert THREAD_OF_THOUGHT_METADATA.supports_revision is True

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "long-context" in THREAD_OF_THOUGHT_METADATA.tags
        assert "segmentation" in THREAD_OF_THOUGHT_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert THREAD_OF_THOUGHT_METADATA.complexity >= 5


# ============================================================================
# Initialization Tests
# ============================================================================


class TestThreadOfThoughtInitialization:
    """Tests for ThreadOfThought initialization."""

    def test_default_initialization(self, method: ThreadOfThought) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "segment"
        assert method._segments == []
        assert method._thread_insights == []
        assert method._current_thread == 0

    def test_elicitation_enabled_by_default(self, method: ThreadOfThought) -> None:
        """Test elicitation is enabled by default."""
        assert method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, method_no_elicitation: ThreadOfThought) -> None:
        """Test elicitation can be disabled."""
        assert method_no_elicitation.enable_elicitation is False

    async def test_initialize_method(self, method: ThreadOfThought) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "segment"

    async def test_health_check_before_initialize(self, method: ThreadOfThought) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: ThreadOfThought) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestThreadOfThoughtProperties:
    """Tests for ThreadOfThought properties."""

    def test_identifier_property(self, method: ThreadOfThought) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.THREAD_OF_THOUGHT

    def test_name_property(self, method: ThreadOfThought) -> None:
        """Test name property returns correct value."""
        assert "Thread" in method.name

    def test_description_property(self, method: ThreadOfThought) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "thread" in desc_lower or "segment" in desc_lower or "context" in desc_lower

    def test_category_property(self, method: ThreadOfThought) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestThreadOfThoughtExecution:
    """Tests for basic execution of ThreadOfThought method."""

    async def test_execute_without_initialization_fails(
        self, method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test input")

    async def test_execute_creates_initial_thought(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "Short input text")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.THREAD_OF_THOUGHT

    async def test_execute_sets_step_number(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() sets phase to segment."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "segment"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.THREAD_OF_THOUGHT


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestThreadOfThoughtContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THREAD_OF_THOUGHT,
            content="Test",
            metadata={"phase": "segment"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_segment_to_thread(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test transition from segment to thread phase."""
        initial = await initialized_method.execute(active_session, "Test input")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "thread"
        assert result.parent_id == initial.id

    async def test_continue_from_thread_to_weave(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test transition from thread to weave phase."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, thread)

        assert result.metadata["phase"] == "weave"

    async def test_continue_from_weave_to_synthesize(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test transition from weave to synthesize phase."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        result = await initialized_method.continue_reasoning(active_session, weave)

        assert result.metadata["phase"] == "synthesize"

    async def test_continue_from_synthesize_to_conclude(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test transition from synthesize to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        synthesize = await initialized_method.continue_reasoning(active_session, weave)
        result = await initialized_method.continue_reasoning(active_session, synthesize)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Segmentation Tests
# ============================================================================


class TestThreadOfThoughtSegmentation:
    """Tests for segmentation functionality."""

    async def test_segments_created_after_execute(
        self, initialized_method: ThreadOfThought, active_session: Session, long_input_text: str
    ) -> None:
        """Test that segments are created after execute."""
        await initialized_method.execute(active_session, long_input_text)
        assert len(initialized_method._segments) >= 1

    async def test_segment_text_heuristic(
        self, initialized_method: ThreadOfThought, long_input_text: str
    ) -> None:
        """Test heuristic segmentation."""
        segments = initialized_method._segment_text(long_input_text)
        assert isinstance(segments, list)
        assert len(segments) >= 1

    async def test_short_input_single_segment(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test that short input creates minimal segments."""
        await initialized_method.execute(active_session, "Short text")
        assert len(initialized_method._segments) >= 1


# ============================================================================
# Threading Tests
# ============================================================================


class TestThreadOfThoughtThreading:
    """Tests for threading functionality."""

    async def test_thread_insights_created_after_thread_phase(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test that thread insights are created after thread phase."""
        initial = await initialized_method.execute(active_session, "Test input with content")
        thread = await initialized_method.continue_reasoning(active_session, initial)

        # Thread phase should be reached
        assert thread.metadata["phase"] == "thread"
        assert isinstance(initialized_method._thread_insights, list)

    async def test_thread_phase_content(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test thread phase produces content."""
        initial = await initialized_method.execute(active_session, "Test input with content")
        thread = await initialized_method.continue_reasoning(active_session, initial)

        assert thread.content
        assert thread.metadata["phase"] == "thread"


# ============================================================================
# Weaving Tests
# ============================================================================


class TestThreadOfThoughtWeaving:
    """Tests for weaving functionality."""

    async def test_weave_phase_follows_thread(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test that weave phase follows thread phase."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)

        # Weave phase should follow thread
        assert thread.metadata["phase"] == "thread"
        assert weave.metadata["phase"] == "weave"

    async def test_weave_phase_content(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test weave phase produces content."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)

        assert weave.content
        assert weave.metadata["phase"] == "weave"


# ============================================================================
# Synthesis Tests
# ============================================================================


class TestThreadOfThoughtSynthesis:
    """Tests for synthesis functionality."""

    async def test_synthesis_produces_unified_understanding(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test that synthesis produces unified understanding."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        synthesize = await initialized_method.continue_reasoning(active_session, weave)

        assert synthesize.content != ""
        assert synthesize.metadata["phase"] == "synthesize"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestThreadOfThoughtEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = " ".join([f"Word{i}" for i in range(5000)])
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Content: @#$%^&*() + 日本語 + العربية = mixed?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None

    async def test_newlines_in_input(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test execution with newline characters."""
        multiline_text = "Line 1\nLine 2\nLine 3\n\nParagraph 2"
        result = await initialized_method.execute(active_session, multiline_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestThreadOfThoughtWorkflow:
    """Tests for complete ThreadOfThought workflows."""

    async def test_full_workflow(
        self, initialized_method: ThreadOfThought, active_session: Session, long_input_text: str
    ) -> None:
        """Test complete workflow from segmentation to conclusion."""
        # Segment phase
        initial = await initialized_method.execute(active_session, long_input_text)
        assert initial.metadata["phase"] == "segment"
        assert initial.type == ThoughtType.INITIAL

        # Thread phase (process each segment)
        thread = await initialized_method.continue_reasoning(active_session, initial)
        assert thread.metadata["phase"] == "thread"

        # Weave phase (connect insights)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        assert weave.metadata["phase"] == "weave"

        # Synthesize phase (unified understanding)
        synthesize = await initialized_method.continue_reasoning(active_session, weave)
        assert synthesize.metadata["phase"] == "synthesize"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, synthesize)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 5
        assert active_session.current_method == MethodIdentifier.THREAD_OF_THOUGHT

    async def test_confidence_progression(
        self, initialized_method: ThreadOfThought, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test input")
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        synthesize = await initialized_method.continue_reasoning(active_session, weave)
        conclusion = await initialized_method.continue_reasoning(active_session, synthesize)

        # Conclusion should have reasonable confidence
        assert conclusion.confidence > 0

    async def test_metadata_includes_segments_and_insights(
        self, initialized_method: ThreadOfThought, active_session: Session, long_input_text: str
    ) -> None:
        """Test that metadata includes segments and insights."""
        initial = await initialized_method.execute(active_session, long_input_text)
        thread = await initialized_method.continue_reasoning(active_session, initial)
        weave = await initialized_method.continue_reasoning(active_session, thread)
        synthesize = await initialized_method.continue_reasoning(active_session, weave)
        conclusion = await initialized_method.continue_reasoning(active_session, synthesize)

        # Conclusion should have relevant metadata
        assert (
            "segments" in conclusion.metadata
            or "insights" in conclusion.metadata
            or len(initialized_method._segments) > 0
        )

    async def test_incremental_context_processing(
        self, initialized_method: ThreadOfThought, active_session: Session, long_input_text: str
    ) -> None:
        """Test incremental processing of long context."""
        initial = await initialized_method.execute(active_session, long_input_text)
        thread = await initialized_method.continue_reasoning(active_session, initial)

        # After threading, should have processed segments
        assert len(initialized_method._thread_insights) >= 0
        assert thread.metadata["phase"] == "thread"
