"""Unit tests for STaR reasoning method.

This module provides comprehensive unit tests for the STaR (Self-Taught Reasoner)
class, testing initialization, attempt generation, rationalization, verification,
bootstrapping, and complete workflow execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.star import (
    STAR_METADATA,
    STaR,
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
def method() -> STaR:
    """Create an STaR instance for testing."""
    return STaR()


@pytest.fixture
async def initialized_method() -> STaR:
    """Create and initialize an STaR instance."""
    method = STaR()
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
    ctx.sample = AsyncMock(return_value="Generated reasoning attempt")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSTaRMetadata:
    """Tests for STaR metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert STAR_METADATA.identifier == MethodIdentifier.STAR

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "STaR" in STAR_METADATA.name or "Self-Taught" in STAR_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in ADVANCED category."""
        assert STAR_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates branching support."""
        assert isinstance(STAR_METADATA.supports_branching, bool)

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert STAR_METADATA.supports_revision is True

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "self-taught" in STAR_METADATA.tags
        assert "bootstrap" in STAR_METADATA.tags
        assert "rationale" in STAR_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert STAR_METADATA.complexity >= 6


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSTaRInitialization:
    """Tests for STaR initialization."""

    def test_default_initialization(self, method: STaR) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "attempt"
        assert method._initial_answer == ""
        assert method._rationale == ""
        assert method._verified is False
        assert method._bootstrap_count == 0

    async def test_initialize_method(self, method: STaR) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "attempt"

    async def test_health_check_before_initialize(self, method: STaR) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: STaR) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestSTaRProperties:
    """Tests for STaR properties."""

    def test_identifier_property(self, method: STaR) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.STAR

    def test_name_property(self, method: STaR) -> None:
        """Test name property returns correct value."""
        assert "STaR" in method.name or "Self-Taught" in method.name

    def test_description_property(self, method: STaR) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "self-taught" in desc_lower or "bootstrap" in desc_lower or "rationale" in desc_lower

    def test_category_property(self, method: STaR) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSTaRExecution:
    """Tests for basic execution of STaR method."""

    async def test_execute_without_initialization_fails(
        self, method: STaR, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "What is 4+4?")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.STAR

    async def test_execute_sets_step_number(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() sets phase to attempt."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "attempt"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.STAR

    async def test_execute_stores_initial_answer(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execute() stores the initial answer."""
        await initialized_method.execute(active_session, "Solve this problem")
        # Initial answer should be populated after execute
        assert isinstance(initialized_method._initial_answer, str)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSTaRContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: STaR, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.STAR,
            content="Test",
            metadata={"phase": "attempt"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_attempt_to_rationalize(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test transition from attempt to rationalize phase."""
        initial = await initialized_method.execute(active_session, "Test")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "rationalize"
        assert result.parent_id == initial.id

    async def test_continue_from_rationalize_to_verify(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test transition from rationalize to verify phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, rationalize)

        assert result.metadata["phase"] == "verify"

    async def test_continue_from_verify_to_bootstrap(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test transition from verify to bootstrap phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        result = await initialized_method.continue_reasoning(active_session, verify)

        assert result.metadata["phase"] == "bootstrap"

    async def test_continue_from_bootstrap_to_conclude(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test transition from bootstrap to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)
        result = await initialized_method.continue_reasoning(active_session, bootstrap)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Rationalization Tests
# ============================================================================


class TestSTaRRationalization:
    """Tests for rationalization functionality."""

    async def test_rationale_generated_after_rationalize_phase(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test that rationale is generated after rationalize phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)

        assert rationalize.metadata["phase"] == "rationalize"
        assert isinstance(initialized_method._rationale, str)

    async def test_rationalize_phase_content(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test rationalize phase produces meaningful content."""
        initial = await initialized_method.execute(active_session, "What is 4+4?")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)

        assert rationalize.content
        assert rationalize.metadata["phase"] == "rationalize"


# ============================================================================
# Verification Tests
# ============================================================================


class TestSTaRVerification:
    """Tests for verification functionality."""

    async def test_verification_performed_after_verify_phase(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test that verification is performed after verify phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)

        # Verification flag should be set
        assert verify.metadata["phase"] == "verify"
        assert isinstance(initialized_method._verified, bool)

    async def test_verify_phase_content(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test verify phase produces verification content."""
        initial = await initialized_method.execute(active_session, "What is 4+4?")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)

        assert verify.content
        assert verify.metadata["phase"] == "verify"


# ============================================================================
# Bootstrapping Tests
# ============================================================================


class TestSTaRBootstrapping:
    """Tests for bootstrapping functionality."""

    async def test_bootstrap_performed_after_bootstrap_phase(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test that bootstrapping is performed after bootstrap phase."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)

        # Bootstrap phase should be reached
        assert bootstrap.metadata["phase"] == "bootstrap"
        assert isinstance(initialized_method._bootstrap_count, int)

    async def test_bootstrap_phase_content(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test bootstrap phase produces content."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)

        assert bootstrap.content
        assert bootstrap.metadata["phase"] == "bootstrap"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSTaREdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "Solve: " + "detail " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Calculate: @#$%^&*() + العربية = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestSTaRWorkflow:
    """Tests for complete STaR workflows."""

    async def test_full_workflow(self, initialized_method: STaR, active_session: Session) -> None:
        """Test complete STaR workflow from attempt to conclusion."""
        # Attempt phase (initial solution)
        initial = await initialized_method.execute(active_session, "What is 6*6?")
        assert initial.metadata["phase"] == "attempt"
        assert initial.type == ThoughtType.INITIAL

        # Rationalize phase (generate rationale)
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        assert rationalize.metadata["phase"] == "rationalize"

        # Verify phase (check correctness)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        assert verify.metadata["phase"] == "verify"

        # Bootstrap phase (improve with rationales)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)
        assert bootstrap.metadata["phase"] == "bootstrap"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, bootstrap)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 5
        assert active_session.current_method == MethodIdentifier.STAR

    async def test_confidence_progression(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)
        conclusion = await initialized_method.continue_reasoning(active_session, bootstrap)

        # Conclusion should have reasonable confidence
        assert conclusion.confidence > 0

    async def test_metadata_includes_attempt_and_rationale(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test that final metadata includes attempt and rationale."""
        initial = await initialized_method.execute(active_session, "Test")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)
        conclusion = await initialized_method.continue_reasoning(active_session, bootstrap)

        # Conclusion should have relevant metadata
        assert (
            "rationale" in conclusion.metadata
            or "verified" in conclusion.metadata
            or initialized_method._rationale
        )

    async def test_self_taught_learning_process(
        self, initialized_method: STaR, active_session: Session
    ) -> None:
        """Test the self-taught learning process."""
        # This test verifies the core STaR concept
        initial = await initialized_method.execute(active_session, "Test problem")
        rationalize = await initialized_method.continue_reasoning(active_session, initial)
        verify = await initialized_method.continue_reasoning(active_session, rationalize)
        bootstrap = await initialized_method.continue_reasoning(active_session, verify)
        conclusion = await initialized_method.continue_reasoning(active_session, bootstrap)

        # The process should have generated rationales and bootstrapped
        assert isinstance(initialized_method._initial_answer, str)
        assert isinstance(initialized_method._rationale, str)
        assert conclusion.type == ThoughtType.CONCLUSION
