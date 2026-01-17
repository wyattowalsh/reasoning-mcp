"""Unit tests for ZeroShotCoT reasoning method.

This module provides comprehensive unit tests for the ZeroShotCoT class,
testing initialization, the magic trigger phrase, step-by-step reasoning,
answer extraction, phase transitions, and complete workflow execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.zero_shot_cot import (
    ZERO_SHOT_COT_METADATA,
    ZeroShotCoT,
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
def method() -> ZeroShotCoT:
    """Create a ZeroShotCoT instance for testing."""
    return ZeroShotCoT()


@pytest.fixture
def method_no_elicitation() -> ZeroShotCoT:
    """Create a ZeroShotCoT instance with elicitation disabled."""
    return ZeroShotCoT(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> ZeroShotCoT:
    """Create and initialize a ZeroShotCoT instance."""
    method = ZeroShotCoT()
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
    ctx.sample = AsyncMock(return_value="Step-by-step reasoning response")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestZeroShotCoTMetadata:
    """Tests for ZeroShotCoT metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert ZERO_SHOT_COT_METADATA.identifier == MethodIdentifier.ZERO_SHOT_COT

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "Zero-Shot" in ZERO_SHOT_COT_METADATA.name
        assert "Chain-of-Thought" in ZERO_SHOT_COT_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in CORE category."""
        assert ZERO_SHOT_COT_METADATA.category == MethodCategory.CORE

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates no branching support."""
        assert ZERO_SHOT_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates no revision support."""
        assert ZERO_SHOT_COT_METADATA.supports_revision is False

    def test_metadata_requires_context(self) -> None:
        """Test metadata indicates no context required."""
        assert ZERO_SHOT_COT_METADATA.requires_context is False

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "zero-shot" in ZERO_SHOT_COT_METADATA.tags
        assert "simple" in ZERO_SHOT_COT_METADATA.tags
        assert "step-by-step" in ZERO_SHOT_COT_METADATA.tags
        assert "foundational" in ZERO_SHOT_COT_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has low complexity level."""
        assert ZERO_SHOT_COT_METADATA.complexity == 2

    def test_metadata_min_thoughts(self) -> None:
        """Test metadata specifies minimum thoughts."""
        assert ZERO_SHOT_COT_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self) -> None:
        """Test metadata specifies maximum thoughts."""
        assert ZERO_SHOT_COT_METADATA.max_thoughts == 5

    def test_metadata_best_for(self) -> None:
        """Test metadata specifies best use cases."""
        assert "quick reasoning" in ZERO_SHOT_COT_METADATA.best_for
        assert "simple problems" in ZERO_SHOT_COT_METADATA.best_for

    def test_metadata_not_recommended_for(self) -> None:
        """Test metadata specifies when not to use."""
        assert "complex multi-step" in ZERO_SHOT_COT_METADATA.not_recommended_for


# ============================================================================
# Trigger Phrase Tests
# ============================================================================


class TestTriggerPhrase:
    """Tests for the magic trigger phrase."""

    def test_trigger_phrase_constant(self) -> None:
        """Test TRIGGER_PHRASE constant value."""
        assert ZeroShotCoT.TRIGGER_PHRASE == "Let's think step by step."

    async def test_trigger_phrase_in_execute_content(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test trigger phrase appears in execute content."""
        result = await initialized_method.execute(active_session, "What is 2+2?")
        assert (
            ZeroShotCoT.TRIGGER_PHRASE in result.content or "step by step" in result.content.lower()
        )

    async def test_trigger_phrase_in_metadata(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test trigger phrase is stored in metadata."""
        result = await initialized_method.execute(active_session, "Test problem")
        assert result.metadata.get("trigger") == ZeroShotCoT.TRIGGER_PHRASE


# ============================================================================
# Initialization Tests
# ============================================================================


class TestZeroShotCoTInitialization:
    """Tests for ZeroShotCoT initialization."""

    def test_default_initialization(self, method: ZeroShotCoT) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "trigger"
        assert method._reasoning_steps == []
        assert method._use_sampling is True

    def test_elicitation_enabled_by_default(self, method: ZeroShotCoT) -> None:
        """Test elicitation is enabled by default."""
        assert method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, method_no_elicitation: ZeroShotCoT) -> None:
        """Test elicitation can be disabled."""
        assert method_no_elicitation.enable_elicitation is False

    async def test_initialize_method(self, method: ZeroShotCoT) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "trigger"
        assert method._reasoning_steps == []

    async def test_health_check_before_initialize(self, method: ZeroShotCoT) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: ZeroShotCoT) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestZeroShotCoTProperties:
    """Tests for ZeroShotCoT properties."""

    def test_identifier_property(self, method: ZeroShotCoT) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.ZERO_SHOT_COT

    def test_name_property(self, method: ZeroShotCoT) -> None:
        """Test name property returns correct value."""
        assert "Zero-Shot" in method.name
        assert "Chain-of-Thought" in method.name

    def test_description_property(self, method: ZeroShotCoT) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "step" in desc_lower or "think" in desc_lower

    def test_category_property(self, method: ZeroShotCoT) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.CORE


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestZeroShotCoTExecution:
    """Tests for basic execution of ZeroShotCoT method."""

    async def test_execute_without_initialization_fails(
        self, method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "What is 5+5?")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.ZERO_SHOT_COT

    async def test_execute_sets_step_number(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() sets phase to trigger."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "trigger"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.ZERO_SHOT_COT

    async def test_execute_confidence_is_moderate(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() sets moderate confidence for initial thought."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.confidence == 0.6

    async def test_execute_depth_is_zero(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execute() sets depth to 0 for initial thought."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.depth == 0


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestZeroShotCoTContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ZERO_SHOT_COT,
            content="Test",
            metadata={"phase": "trigger"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_trigger_to_reason(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test transition from trigger to reason phase."""
        initial = await initialized_method.execute(active_session, "What is 3+3?")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "reason"
        assert result.type == ThoughtType.REASONING
        assert result.parent_id == initial.id

    async def test_continue_from_reason_to_extract(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test transition from reason to extract phase."""
        initial = await initialized_method.execute(active_session, "Test")
        reason = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, reason)

        assert result.metadata["phase"] == "extract"
        assert result.type == ThoughtType.SYNTHESIS

    async def test_continue_from_extract_to_conclude(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test transition from extract to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        reason = await initialized_method.continue_reasoning(active_session, initial)
        extract = await initialized_method.continue_reasoning(active_session, reason)
        result = await initialized_method.continue_reasoning(active_session, extract)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.step_number == 1

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.depth == 0

        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == 1


# ============================================================================
# Reasoning Steps Tests
# ============================================================================


class TestZeroShotCoTReasoningSteps:
    """Tests for reasoning steps generation."""

    async def test_reasoning_steps_generated(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test that reasoning steps are generated after reason phase."""
        initial = await initialized_method.execute(active_session, "Test")
        await initialized_method.continue_reasoning(active_session, initial)

        assert len(initialized_method._reasoning_steps) >= 1

    async def test_reasoning_steps_in_metadata(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test that reasoning steps appear in metadata."""
        initial = await initialized_method.execute(active_session, "Test")
        reason = await initialized_method.continue_reasoning(active_session, initial)

        assert "steps" in reason.metadata


# ============================================================================
# Heuristic Generation Tests
# ============================================================================


class TestZeroShotCoTHeuristics:
    """Tests for heuristic generation methods."""

    def test_generate_reasoning_heuristic(self, initialized_method: ZeroShotCoT) -> None:
        """Test heuristic reasoning generation."""
        result = initialized_method._generate_reasoning_heuristic("What is 2+2?")
        assert isinstance(result, str)
        assert "What is 2+2?" in result
        assert ZeroShotCoT.TRIGGER_PHRASE in result

    def test_generate_continuation_heuristic_from_trigger(
        self, initialized_method: ZeroShotCoT
    ) -> None:
        """Test heuristic continuation from trigger phase."""
        content, thought_type, confidence = initialized_method._generate_continuation_heuristic(
            "trigger", None
        )
        assert isinstance(content, str)
        assert thought_type == ThoughtType.REASONING
        assert confidence == 0.75
        assert initialized_method._current_phase == "reason"

    def test_generate_continuation_heuristic_from_reason(
        self, initialized_method: ZeroShotCoT
    ) -> None:
        """Test heuristic continuation from reason phase."""
        # First transition to reason phase
        initialized_method._generate_continuation_heuristic("trigger", None)

        content, thought_type, confidence = initialized_method._generate_continuation_heuristic(
            "reason", None
        )
        assert isinstance(content, str)
        assert thought_type == ThoughtType.SYNTHESIS
        assert confidence == 0.8
        assert initialized_method._current_phase == "extract"

    def test_generate_continuation_heuristic_from_extract(
        self, initialized_method: ZeroShotCoT
    ) -> None:
        """Test heuristic continuation from extract phase."""
        content, thought_type, confidence = initialized_method._generate_continuation_heuristic(
            "extract", None
        )
        assert isinstance(content, str)
        assert thought_type == ThoughtType.CONCLUSION
        assert confidence == 0.8
        assert initialized_method._current_phase == "conclude"


# ============================================================================
# Phase Metadata Tests
# ============================================================================


class TestZeroShotCoTPhaseMetadata:
    """Tests for phase metadata methods."""

    def test_get_phase_metadata_from_trigger(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase metadata from trigger phase."""
        thought_type, confidence = initialized_method._get_phase_metadata("trigger")
        assert thought_type == ThoughtType.REASONING
        assert confidence == 0.75
        assert initialized_method._current_phase == "reason"

    def test_get_phase_metadata_from_reason(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase metadata from reason phase."""
        thought_type, confidence = initialized_method._get_phase_metadata("reason")
        assert thought_type == ThoughtType.SYNTHESIS
        assert confidence == 0.8
        assert initialized_method._current_phase == "extract"

    def test_get_phase_metadata_from_extract(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase metadata from extract phase."""
        thought_type, confidence = initialized_method._get_phase_metadata("extract")
        assert thought_type == ThoughtType.CONCLUSION
        assert confidence == 0.8
        assert initialized_method._current_phase == "conclude"

    def test_get_phase_name_from_trigger(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase name from trigger phase."""
        name = initialized_method._get_phase_name("trigger")
        assert name == "Generate Reasoning"

    def test_get_phase_name_from_reason(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase name from reason phase."""
        name = initialized_method._get_phase_name("reason")
        assert name == "Extract Answer"

    def test_get_phase_name_from_extract(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase name from extract phase."""
        name = initialized_method._get_phase_name("extract")
        assert name == "Final Answer"

    def test_get_phase_name_unknown(self, initialized_method: ZeroShotCoT) -> None:
        """Test phase name for unknown phase."""
        name = initialized_method._get_phase_name("unknown")
        assert name == "Continue"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestZeroShotCoTEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "Calculate: " + "number " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Solve: @#$%^&*() + 中文 + العربية = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None

    async def test_mathematical_expression(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test execution with mathematical expression."""
        math_text = "What is (15 * 4) + (27 / 3) - 8?"
        result = await initialized_method.execute(active_session, math_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestZeroShotCoTWorkflow:
    """Tests for complete ZeroShotCoT workflows."""

    async def test_full_workflow(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test complete workflow from trigger to conclusion."""
        # Trigger phase
        initial = await initialized_method.execute(active_session, "What is 7*8?")
        assert initial.metadata["phase"] == "trigger"
        assert initial.type == ThoughtType.INITIAL

        # Reason phase (generate step-by-step reasoning)
        reason = await initialized_method.continue_reasoning(active_session, initial)
        assert reason.metadata["phase"] == "reason"
        assert reason.type == ThoughtType.REASONING

        # Extract phase (derive answer)
        extract = await initialized_method.continue_reasoning(active_session, reason)
        assert extract.metadata["phase"] == "extract"
        assert extract.type == ThoughtType.SYNTHESIS

        # Conclude phase (present final answer)
        conclusion = await initialized_method.continue_reasoning(active_session, extract)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 4
        assert active_session.current_method == MethodIdentifier.ZERO_SHOT_COT

    async def test_confidence_progression(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test confidence progression through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.confidence == 0.6

        reason = await initialized_method.continue_reasoning(active_session, initial)
        assert reason.confidence == 0.75

        extract = await initialized_method.continue_reasoning(active_session, reason)
        assert extract.confidence == 0.8

        conclusion = await initialized_method.continue_reasoning(active_session, extract)
        assert conclusion.confidence == 0.8

    async def test_depth_progression(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test depth progression through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        assert initial.depth == 0

        reason = await initialized_method.continue_reasoning(active_session, initial)
        assert reason.depth == 1

        extract = await initialized_method.continue_reasoning(active_session, reason)
        assert extract.depth == 2

        conclusion = await initialized_method.continue_reasoning(active_session, extract)
        assert conclusion.depth == 3

    async def test_parent_child_relationship(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test parent-child relationships in thought graph."""
        initial = await initialized_method.execute(active_session, "Test")
        reason = await initialized_method.continue_reasoning(active_session, initial)
        extract = await initialized_method.continue_reasoning(active_session, reason)
        conclusion = await initialized_method.continue_reasoning(active_session, extract)

        assert reason.parent_id == initial.id
        assert extract.parent_id == reason.id
        assert conclusion.parent_id == extract.id

    async def test_no_examples_required(
        self, initialized_method: ZeroShotCoT, active_session: Session
    ) -> None:
        """Test that zero-shot CoT works without examples."""
        # This is the core concept of zero-shot CoT
        initial = await initialized_method.execute(active_session, "New problem without examples")
        reason = await initialized_method.continue_reasoning(active_session, initial)
        extract = await initialized_method.continue_reasoning(active_session, reason)
        conclusion = await initialized_method.continue_reasoning(active_session, extract)

        # Should complete successfully without any few-shot examples
        assert conclusion.type == ThoughtType.CONCLUSION
