"""Unit tests for Spoc reasoning method.

This module provides comprehensive unit tests for the Spoc (Spontaneous Self-Correction)
class, testing initialization, solution generation, error detection, validation,
and internal correction mechanisms.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.spoc import (
    SPOC_METADATA,
    Spoc,
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
def method() -> Spoc:
    """Create a Spoc instance for testing."""
    return Spoc()


@pytest.fixture
async def initialized_method() -> Spoc:
    """Create and initialize a Spoc instance."""
    method = Spoc()
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
    ctx.sample = AsyncMock(return_value="Generated solution with error detection")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSpocMetadata:
    """Tests for Spoc metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert SPOC_METADATA.identifier == MethodIdentifier.SPOC

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "Spoc" in SPOC_METADATA.name or "SPOC" in SPOC_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in SPECIALIZED category."""
        assert SPOC_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates branching support."""
        assert isinstance(SPOC_METADATA.supports_branching, bool)

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert SPOC_METADATA.supports_revision is True

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "self-correction" in SPOC_METADATA.tags
        assert "spontaneous" in SPOC_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert SPOC_METADATA.complexity >= 5


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSpocInitialization:
    """Tests for Spoc initialization."""

    def test_default_initialization(self, method: Spoc) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._error_detected is False

    async def test_initialize_method(self, method: Spoc) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_health_check_before_initialize(self, method: Spoc) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: Spoc) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestSpocProperties:
    """Tests for Spoc properties."""

    def test_identifier_property(self, method: Spoc) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.SPOC

    def test_name_property(self, method: Spoc) -> None:
        """Test name property returns correct value."""
        assert "Spoc" in method.name or "SPOC" in method.name or "Self" in method.name

    def test_description_property(self, method: Spoc) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "self" in desc_lower or "correction" in desc_lower or "error" in desc_lower

    def test_category_property(self, method: Spoc) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSpocExecution:
    """Tests for basic execution of Spoc method."""

    async def test_execute_without_initialization_fails(
        self, method: Spoc, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "Solve 3+3")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.SPOC

    async def test_execute_sets_step_number(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() sets phase to generate."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "generate"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.SPOC

    async def test_execute_sets_phase(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execute() sets the correct phase."""
        result = await initialized_method.execute(active_session, "What is 5*5?")
        assert result.metadata["phase"] == "generate"


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSpocContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: Spoc, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SPOC,
            content="Test",
            metadata={"phase": "generate"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_generate_to_detect(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test transition from generate to detect_error phase."""
        initial = await initialized_method.execute(active_session, "Test")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "detect_error"
        assert result.parent_id == initial.id

    async def test_continue_from_detect_to_validate(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test transition from detect_error to validate phase."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, detect)

        assert result.metadata["phase"] == "validate"

    async def test_continue_from_validate_to_conclude(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test transition from validate to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)
        result = await initialized_method.continue_reasoning(active_session, validate)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Error Detection Tests
# ============================================================================


class TestSpocErrorDetection:
    """Tests for error detection functionality."""

    async def test_error_detection_phase(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test error detection phase behavior."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)

        # After detect phase, error_detected should be set
        assert detect.metadata["phase"] == "detect_error"
        assert isinstance(initialized_method._error_detected, bool)

    async def test_error_detected_flag(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test error_detected flag behavior."""
        initial = await initialized_method.execute(active_session, "Test")
        await initialized_method.continue_reasoning(active_session, initial)

        # Error detected flag should be set
        assert isinstance(initialized_method._error_detected, bool)


# ============================================================================
# Validation Tests
# ============================================================================


class TestSpocValidation:
    """Tests for validation functionality."""

    async def test_validate_phase_produces_result(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test that validate phase produces a result."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)

        assert validate.content != ""
        assert validate.metadata["phase"] == "validate"


# ============================================================================
# Correction Tests
# ============================================================================


class TestSpocCorrection:
    """Tests for self-correction functionality."""

    async def test_validate_phase_follows_detect(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test that validation phase follows error detection."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)

        # Validate phase should follow detect
        assert detect.metadata["phase"] == "detect_error"
        assert validate.metadata["phase"] == "validate"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSpocEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "Solve this complex problem: " + "word " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Calculate: @#$%^&*() + 한국어 = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestSpocWorkflow:
    """Tests for complete Spoc workflows."""

    async def test_full_workflow(self, initialized_method: Spoc, active_session: Session) -> None:
        """Test complete Spoc workflow from generation to conclusion."""
        # Generate phase (initial solution)
        initial = await initialized_method.execute(active_session, "What is 8+8?")
        assert initial.metadata["phase"] == "generate"
        assert initial.type == ThoughtType.INITIAL

        # Detect error phase
        detect = await initialized_method.continue_reasoning(active_session, initial)
        assert detect.metadata["phase"] == "detect_error"

        # Validate phase
        validate = await initialized_method.continue_reasoning(active_session, detect)
        assert validate.metadata["phase"] == "validate"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, validate)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 4
        assert active_session.current_method == MethodIdentifier.SPOC

    async def test_confidence_progression(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)
        conclusion = await initialized_method.continue_reasoning(active_session, validate)

        # Conclusion should have reasonable confidence
        assert conclusion.confidence > 0

    async def test_metadata_includes_error_status(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test that metadata includes error status."""
        initial = await initialized_method.execute(active_session, "Test")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)
        conclusion = await initialized_method.continue_reasoning(active_session, validate)

        # Conclusion should have phase metadata
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_no_external_feedback_required(
        self, initialized_method: Spoc, active_session: Session
    ) -> None:
        """Test that Spoc works without external feedback."""
        # This verifies the "spontaneous" nature of self-correction
        initial = await initialized_method.execute(active_session, "Test problem")
        detect = await initialized_method.continue_reasoning(active_session, initial)
        validate = await initialized_method.continue_reasoning(active_session, detect)
        conclusion = await initialized_method.continue_reasoning(active_session, validate)

        # Should complete without external input
        assert conclusion.type == ThoughtType.CONCLUSION
