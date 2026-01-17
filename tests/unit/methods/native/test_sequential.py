"""
Comprehensive tests for SequentialThinking method.

This module provides complete test coverage for:
- SequentialThinking: Linear step-by-step reasoning method

Tests cover:
1. Initialization and health checks
2. Basic execution with proper thought sequence
3. Step configuration and formatting
4. Continue reasoning with step progression
5. Edge cases (empty input, long input, special characters)
6. Step labeling verification (Step 1, Step 2, etc.)
7. Completion detection
8. Graph structure validation (linear chain, no branching)
9. Metadata propagation
10. Error handling and validation

The test suite aims for 90%+ code coverage with at least 15 test cases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.sequential import (
    SEQUENTIAL_METADATA,
    SequentialThinking,
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
def method() -> SequentialThinking:
    """Provide an uninitialized SequentialThinking method instance.

    Returns:
        Fresh SequentialThinking instance for testing.
    """
    return SequentialThinking()


@pytest.fixture
async def initialized_method() -> SequentialThinking:
    """Provide an initialized SequentialThinking method instance.

    Returns:
        Initialized SequentialThinking instance ready for execution.
    """
    method = SequentialThinking()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide a fresh Session instance.

    Returns:
        New Session instance for testing.
    """
    return Session()


@pytest.fixture
def active_session() -> Session:
    """Provide an active Session instance.

    Returns:
        Session that has been started (status=ACTIVE).
    """
    session = Session()
    session.start()
    return session


@pytest.fixture
def sample_thought() -> ThoughtNode:
    """Provide a sample ThoughtNode for testing.

    Returns:
        ThoughtNode with minimal required fields.
    """
    return ThoughtNode(
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.SEQUENTIAL_THINKING,
        content="Step 1: Initial thought",
        step_number=1,
        confidence=0.7,
    )


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSequentialMetadata:
    """Test suite for Sequential Thinking metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SEQUENTIAL_METADATA.identifier == MethodIdentifier.SEQUENTIAL_THINKING

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SEQUENTIAL_METADATA.name == "Sequential Thinking"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert SEQUENTIAL_METADATA.category == MethodCategory.CORE

    def test_metadata_complexity(self):
        """Test that Sequential Thinking has minimal complexity."""
        assert SEQUENTIAL_METADATA.complexity == 1

    def test_metadata_no_branching_support(self):
        """Test that Sequential Thinking doesn't support branching."""
        assert SEQUENTIAL_METADATA.supports_branching is False

    def test_metadata_supports_revision(self):
        """Test that Sequential Thinking supports revision."""
        assert SEQUENTIAL_METADATA.supports_revision is True

    def test_metadata_no_context_required(self):
        """Test that Sequential Thinking doesn't require special context."""
        assert SEQUENTIAL_METADATA.requires_context is False


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSequentialThinkingInitialization:
    """Test suite for SequentialThinking initialization."""

    def test_create_uninitialized(self, method: SequentialThinking):
        """Test creating a new SequentialThinking instance."""
        assert method._initialized is False
        assert method._step_counter == 0

    @pytest.mark.asyncio
    async def test_initialize(self, method: SequentialThinking):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, method: SequentialThinking):
        """Test health check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, initialized_method: SequentialThinking):
        """Test health check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_initializations(self, method: SequentialThinking):
        """Test that method can be reinitialized."""
        await method.initialize()
        assert method._initialized is True

        # Initialize again - should reset step counter
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0


# ============================================================================
# Property Tests
# ============================================================================


class TestSequentialThinkingProperties:
    """Test suite for SequentialThinking properties."""

    def test_identifier_property(self, method: SequentialThinking):
        """Test that identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.SEQUENTIAL_THINKING

    def test_name_property(self, method: SequentialThinking):
        """Test that name property returns correct value."""
        assert method.name == "Sequential Thinking"

    def test_description_property(self, method: SequentialThinking):
        """Test that description property returns correct value."""
        assert "step-by-step" in method.description.lower()
        assert "linear" in method.description.lower()

    def test_category_property(self, method: SequentialThinking):
        """Test that category property returns correct value."""
        assert method.category == MethodCategory.CORE


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSequentialThinkingExecution:
    """Test suite for basic execution of SequentialThinking."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_fails(
        self, method: SequentialThinking, active_session: Session
    ):
        """Test that executing without initialization raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=active_session, input_text="Test problem")

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that execute creates an initial thought."""
        result = await initialized_method.execute(
            session=active_session, input_text="Solve 2x + 5 = 15"
        )

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.SEQUENTIAL_THINKING

    @pytest.mark.asyncio
    async def test_execute_sets_step_number_to_one(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that first execution sets step_number to 1."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    @pytest.mark.asyncio
    async def test_execute_sets_correct_depth(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that initial thought has depth 0."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert result.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that initial thought has moderate confidence."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert result.confidence == 0.7
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_execute_includes_input_in_metadata(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that input text is included in thought metadata."""
        input_text = "Analyze this problem"
        result = await initialized_method.execute(session=active_session, input_text=input_text)

        assert "input" in result.metadata
        assert result.metadata["input"] == input_text

    @pytest.mark.asyncio
    async def test_execute_includes_reasoning_type_in_metadata(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that reasoning type is included in metadata."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert "reasoning_type" in result.metadata
        assert result.metadata["reasoning_type"] == "sequential"

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that execute adds thought to session graph."""
        initial_count = active_session.thought_count

        await initialized_method.execute(session=active_session, input_text="Test problem")

        assert active_session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_sets_session_current_method(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that execute sets the session's current method."""
        await initialized_method.execute(session=active_session, input_text="Test problem")

        assert active_session.current_method == MethodIdentifier.SEQUENTIAL_THINKING

    @pytest.mark.asyncio
    async def test_execute_with_context(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execute with optional context parameter."""
        context = {"key": "value", "number": 42}
        result = await initialized_method.execute(
            session=active_session, input_text="Test problem", context=context
        )

        assert "context" in result.metadata
        assert result.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_without_context(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execute without context defaults to empty dict."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert "context" in result.metadata
        assert result.metadata["context"] == {}


# ============================================================================
# Step Labeling Tests
# ============================================================================


class TestSequentialThinkingStepLabeling:
    """Test suite for step labeling in Sequential Thinking."""

    @pytest.mark.asyncio
    async def test_initial_step_labeled_correctly(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that initial thought is labeled as Step 1."""
        result = await initialized_method.execute(session=active_session, input_text="Test problem")

        assert "Step 1" in result.content

    @pytest.mark.asyncio
    async def test_continuation_step_labeled_correctly(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continuation thoughts are labeled with correct step numbers."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert "Step 2" in second.content

    @pytest.mark.asyncio
    async def test_multiple_steps_labeled_sequentially(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that multiple steps are labeled sequentially."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for expected_step in range(2, 6):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
            assert f"Step {expected_step}" in thought.content
            assert thought.step_number == expected_step


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSequentialThinkingContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_fails(
        self, method: SequentialThinking, active_session: Session, sample_thought: ThoughtNode
    ):
        """Test that continue_reasoning without initialization raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=active_session, previous_thought=sample_thought)

    @pytest.mark.asyncio
    async def test_continue_reasoning_creates_continuation(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continue_reasoning creates a CONTINUATION thought."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        result = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert result.type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_number(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continue_reasoning increments step number."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.step_number == 2
        assert initialized_method._step_counter == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continuation thought has correct parent_id."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_depth(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continuation thought has incremented depth."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test continue_reasoning with guidance parameter."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        guidance = "Focus on the mathematical aspects"
        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first, guidance=guidance
        )

        assert "guidance" in second.metadata
        assert second.metadata["guidance"] == guidance
        assert guidance in second.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test continue_reasoning without guidance defaults to empty string."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert "guidance" in second.metadata
        assert second.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test continue_reasoning with context parameter."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        context = {"additional": "information"}
        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first, context=context
        )

        assert "context" in second.metadata
        assert second.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_continue_reasoning_updates_confidence(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that confidence decreases slightly with more steps."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        # Confidence should decrease based on step counter
        assert second.confidence < first.confidence
        assert second.confidence >= 0.5  # Minimum confidence threshold

    @pytest.mark.asyncio
    async def test_continue_reasoning_maintains_minimum_confidence(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that confidence never goes below 0.5."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        # Create many steps to test minimum confidence
        for _ in range(20):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert thought.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_continue_reasoning_includes_previous_step_in_metadata(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that previous step number is included in metadata."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert "previous_step" in second.metadata
        assert second.metadata["previous_step"] == first.step_number

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_to_session(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that continue_reasoning adds thought to session."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        initial_count = active_session.thought_count

        await initialized_method.continue_reasoning(session=active_session, previous_thought=first)

        assert active_session.thought_count == initial_count + 1


# ============================================================================
# Graph Structure Tests
# ============================================================================


class TestSequentialThinkingGraphStructure:
    """Test suite for validating linear graph structure."""

    @pytest.mark.asyncio
    async def test_creates_linear_chain(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that Sequential Thinking creates a linear chain (no branching)."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        # Create a chain of thoughts
        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Verify linear structure - each thought should have at most one child
        for node in active_session.graph.nodes.values():
            assert len(node.children_ids) <= 1

    @pytest.mark.asyncio
    async def test_no_branching(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that no thoughts have multiple children (no branching)."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Check for branching
        has_branching = any(
            len(node.children_ids) > 1 for node in active_session.graph.nodes.values()
        )
        assert has_branching is False

    @pytest.mark.asyncio
    async def test_parent_child_relationships(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that parent-child relationships are correctly established."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        third = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=second
        )

        # Verify relationships
        assert second.parent_id == first.id
        assert third.parent_id == second.id
        assert first.parent_id is None  # Root has no parent

    @pytest.mark.asyncio
    async def test_depth_increases_linearly(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that depth increases by 1 for each step."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        assert thought.depth == 0

        for expected_depth in range(1, 6):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
            assert thought.depth == expected_depth


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestSequentialThinkingEdgeCases:
    """Test suite for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_input_text(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with empty input text."""
        result = await initialized_method.execute(session=active_session, input_text="")

        assert result is not None
        assert result.metadata["input"] == ""
        assert "Step 1" in result.content

    @pytest.mark.asyncio
    async def test_very_long_input_text(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with very long input text."""
        long_text = "x" * 10000
        result = await initialized_method.execute(session=active_session, input_text=long_text)

        assert result is not None
        assert result.metadata["input"] == long_text

    @pytest.mark.asyncio
    async def test_special_characters_in_input(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with special characters in input."""
        special_text = "Test with émojis 🎉, ünîcödé, and symbols: @#$%^&*()"
        result = await initialized_method.execute(session=active_session, input_text=special_text)

        assert result is not None
        assert result.metadata["input"] == special_text

    @pytest.mark.asyncio
    async def test_newlines_in_input(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with newlines in input."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        result = await initialized_method.execute(session=active_session, input_text=multiline_text)

        assert result is not None
        assert result.metadata["input"] == multiline_text

    @pytest.mark.asyncio
    async def test_whitespace_only_input(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with whitespace-only input."""
        whitespace_text = "   \t\n   "
        result = await initialized_method.execute(
            session=active_session, input_text=whitespace_text
        )

        assert result is not None
        assert result.metadata["input"] == whitespace_text

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with explicitly None context."""
        result = await initialized_method.execute(
            session=active_session, input_text="Test", context=None
        )

        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_dict_context(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with empty dictionary context."""
        result = await initialized_method.execute(
            session=active_session, input_text="Test", context={}
        )

        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_complex_context_structure(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test execution with complex nested context."""
        context = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "mixed": {"a": [1, {"b": 2}]},
        }
        result = await initialized_method.execute(
            session=active_session, input_text="Test", context=context
        )

        assert result.metadata["context"] == context


# ============================================================================
# Metadata Propagation Tests
# ============================================================================


class TestSequentialThinkingMetadataPropagation:
    """Test suite for metadata propagation through reasoning chain."""

    @pytest.mark.asyncio
    async def test_reasoning_type_in_all_thoughts(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that reasoning_type metadata is in all thoughts."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Check all thoughts
        for node in active_session.graph.nodes.values():
            assert "reasoning_type" in node.metadata
            assert node.metadata["reasoning_type"] == "sequential"

    @pytest.mark.asyncio
    async def test_method_id_consistent(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that all thoughts have the same method_id."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Check all thoughts
        for node in active_session.graph.nodes.values():
            assert node.method_id == MethodIdentifier.SEQUENTIAL_THINKING

    @pytest.mark.asyncio
    async def test_step_numbers_unique_and_sequential(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that step numbers are unique and sequential."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(4):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Get all step numbers
        step_numbers = [node.step_number for node in active_session.graph.nodes.values()]
        step_numbers.sort()

        assert step_numbers == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_guidance_metadata_only_in_continuations(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that guidance metadata only appears in continuation thoughts."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first, guidance="Test guidance"
        )

        # Initial thought should not have guidance
        assert "guidance" not in first.metadata

        # Continuation should have guidance
        assert "guidance" in second.metadata

    @pytest.mark.asyncio
    async def test_previous_step_metadata_only_in_continuations(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that previous_step metadata only appears in continuation thoughts."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        # Initial thought should not have previous_step
        assert "previous_step" not in first.metadata

        # Continuation should have previous_step
        assert "previous_step" in second.metadata
        assert second.metadata["previous_step"] == 1


# ============================================================================
# Step Counter Reset Tests
# ============================================================================


class TestSequentialThinkingStepCounterReset:
    """Test suite for step counter reset behavior."""

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_new_execution(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that step counter resets when execute is called again."""
        # First execution
        first_thought = await initialized_method.execute(
            session=active_session, input_text="First problem"
        )
        assert first_thought.step_number == 1

        # Continue a few steps
        thought = first_thought
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert initialized_method._step_counter == 4

        # New execution should reset counter
        new_session = Session().start()
        new_thought = await initialized_method.execute(
            session=new_session, input_text="Second problem"
        )
        assert new_thought.step_number == 1
        assert initialized_method._step_counter == 1


# ============================================================================
# Session Integration Tests
# ============================================================================


class TestSequentialThinkingSessionIntegration:
    """Test suite for integration with Session."""

    @pytest.mark.asyncio
    async def test_updates_session_metrics(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that execution updates session metrics."""
        initial_thought_count = active_session.metrics.total_thoughts

        await initialized_method.execute(session=active_session, input_text="Test problem")

        assert active_session.metrics.total_thoughts == initial_thought_count + 1

    @pytest.mark.asyncio
    async def test_metrics_track_method_usage(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that metrics track Sequential Thinking usage."""
        await initialized_method.execute(session=active_session, input_text="Test problem")

        method_key = str(MethodIdentifier.SEQUENTIAL_THINKING)
        assert method_key in active_session.metrics.methods_used
        assert active_session.metrics.methods_used[method_key] >= 1

    @pytest.mark.asyncio
    async def test_metrics_track_thought_types(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that metrics track thought types."""
        first = await initialized_method.execute(session=active_session, input_text="Test problem")

        await initialized_method.continue_reasoning(session=active_session, previous_thought=first)

        initial_key = str(ThoughtType.INITIAL)
        continuation_key = str(ThoughtType.CONTINUATION)

        assert initial_key in active_session.metrics.thought_types
        assert continuation_key in active_session.metrics.thought_types
        assert active_session.metrics.thought_types[initial_key] >= 1
        assert active_session.metrics.thought_types[continuation_key] >= 1

    @pytest.mark.asyncio
    async def test_max_depth_tracked_in_metrics(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that maximum depth is tracked in metrics."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert active_session.metrics.max_depth_reached == 5

    @pytest.mark.asyncio
    async def test_average_confidence_calculated(
        self, initialized_method: SequentialThinking, active_session: Session
    ):
        """Test that average confidence is calculated correctly."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Test problem"
        )

        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Average confidence should be between 0 and 1
        assert 0.0 <= active_session.metrics.average_confidence <= 1.0
