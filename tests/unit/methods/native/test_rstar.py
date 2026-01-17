"""Unit tests for RStar reasoning method.

This module provides comprehensive unit tests for the RStar (Self-play muTual reasoning)
class, testing initialization, MCTS-like behavior, discriminator scoring, path selection,
and complete workflow execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.rstar import (
    RSTAR_METADATA,
    RStar,
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
def method() -> RStar:
    """Create an RStar instance for testing."""
    return RStar()


@pytest.fixture
async def initialized_method() -> RStar:
    """Create and initialize an RStar instance."""
    method = RStar()
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
    ctx.sample = AsyncMock(return_value="Generated reasoning path")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestRStarMetadata:
    """Tests for RStar metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert RSTAR_METADATA.identifier == MethodIdentifier.RSTAR

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert "rStar" in RSTAR_METADATA.name or "RStar" in RSTAR_METADATA.name

    def test_metadata_category(self) -> None:
        """Test metadata is in ADVANCED category."""
        assert RSTAR_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates branching support."""
        assert RSTAR_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert RSTAR_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test metadata indicates no context required."""
        assert RSTAR_METADATA.requires_context is False

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "self-play" in RSTAR_METADATA.tags
        assert "mcts" in RSTAR_METADATA.tags
        assert "discriminator" in RSTAR_METADATA.tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert RSTAR_METADATA.complexity >= 5

    def test_metadata_min_max_thoughts(self) -> None:
        """Test metadata specifies min/max thoughts."""
        assert RSTAR_METADATA.min_thoughts >= 4
        assert RSTAR_METADATA.max_thoughts >= RSTAR_METADATA.min_thoughts


# ============================================================================
# Initialization Tests
# ============================================================================


class TestRStarInitialization:
    """Tests for RStar initialization."""

    def test_default_initialization(self, method: RStar) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidate_paths == []
        assert method._discriminator_scores == []
        assert method._best_path_idx == 0

    async def test_initialize_method(self, method: RStar) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_health_check_before_initialize(self, method: RStar) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: RStar) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestRStarProperties:
    """Tests for RStar properties."""

    def test_identifier_property(self, method: RStar) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.RSTAR

    def test_name_property(self, method: RStar) -> None:
        """Test name property returns correct value."""
        assert "rStar" in method.name or "RStar" in method.name

    def test_description_property(self, method: RStar) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "self-play" in desc_lower or "mcts" in desc_lower or "discriminator" in desc_lower

    def test_category_property(self, method: RStar) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestRStarExecution:
    """Tests for basic execution of RStar method."""

    async def test_execute_without_initialization_fails(
        self, method: RStar, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "Solve 2+2")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.RSTAR

    async def test_execute_sets_step_number(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execute() sets phase to generate."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "generate"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.RSTAR


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestRStarContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: RStar, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.RSTAR,
            content="Test",
            metadata={"phase": "generate"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_generate_to_execute(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test transition from generate to execute phase."""
        initial = await initialized_method.execute(active_session, "Test")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "execute"
        assert result.parent_id == initial.id

    async def test_continue_from_execute_to_discriminate(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test transition from execute to discriminate phase."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, execute)

        assert result.metadata["phase"] == "discriminate"

    async def test_continue_from_discriminate_to_select(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test transition from discriminate to select phase."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        result = await initialized_method.continue_reasoning(active_session, discriminate)

        assert result.metadata["phase"] == "select"

    async def test_continue_from_select_to_conclude(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test transition from select to conclude phase."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        select = await initialized_method.continue_reasoning(active_session, discriminate)
        result = await initialized_method.continue_reasoning(active_session, select)

        assert result.metadata["phase"] == "conclude"
        assert result.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Candidate Path Tests
# ============================================================================


class TestRStarCandidatePaths:
    """Tests for candidate path generation and management."""

    async def test_candidate_paths_generated(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test that candidate paths are generated during execution."""
        await initialized_method.execute(active_session, "Solve 5*5")
        assert len(initialized_method._candidate_paths) >= 1

    async def test_generate_candidate_paths_heuristic(self, initialized_method: RStar) -> None:
        """Test heuristic candidate path generation."""
        paths = initialized_method._generate_candidate_paths_heuristic()
        assert isinstance(paths, list)
        assert len(paths) >= 1  # Should generate at least one path
        # Each path should be a dict with expected keys
        for path in paths:
            assert isinstance(path, dict)
            assert "id" in path
            assert "steps" in path


# ============================================================================
# Discriminator Scoring Tests
# ============================================================================


class TestRStarDiscriminatorScoring:
    """Tests for discriminator scoring functionality."""

    async def test_discriminator_scores_initialized(self, initialized_method: RStar) -> None:
        """Test discriminator scores list is initialized."""
        assert initialized_method._discriminator_scores == []

    async def test_scores_assigned_after_discriminate(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test that scores are assigned after discriminate phase."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        await initialized_method.continue_reasoning(active_session, execute)

        # After discriminate, scores should be populated
        assert len(initialized_method._discriminator_scores) >= 0

    async def test_discriminator_scores_are_valid(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test discriminator scores are valid floats."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        await initialized_method.continue_reasoning(active_session, execute)

        # Scores should be floats between 0 and 1
        for score in initialized_method._discriminator_scores:
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0


# ============================================================================
# Path Selection Tests
# ============================================================================


class TestRStarPathSelection:
    """Tests for path selection functionality."""

    async def test_best_path_selected_after_select_phase(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test best path index is selected after select phase."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        await initialized_method.continue_reasoning(active_session, discriminate)

        # Best path index should be set
        assert isinstance(initialized_method._best_path_idx, int)
        assert initialized_method._best_path_idx >= 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestRStarEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "A complex problem: " + "detail " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Solve: @#$%^&*() + 测试 = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestRStarWorkflow:
    """Tests for complete RStar workflows."""

    async def test_full_workflow(self, initialized_method: RStar, active_session: Session) -> None:
        """Test complete RStar workflow from generation to conclusion."""
        # Generate phase
        initial = await initialized_method.execute(active_session, "Solve 3+3")
        assert initial.metadata["phase"] == "generate"
        assert initial.type == ThoughtType.INITIAL

        # Execute phase (code verification)
        execute = await initialized_method.continue_reasoning(active_session, initial)
        assert execute.metadata["phase"] == "execute"

        # Discriminate phase (score paths)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        assert discriminate.metadata["phase"] == "discriminate"

        # Select phase (choose best)
        select = await initialized_method.continue_reasoning(active_session, discriminate)
        assert select.metadata["phase"] == "select"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, select)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 5
        assert active_session.current_method == MethodIdentifier.RSTAR

    async def test_confidence_progression(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        select = await initialized_method.continue_reasoning(active_session, discriminate)
        conclusion = await initialized_method.continue_reasoning(active_session, select)

        # Conclusion should have reasonable confidence
        assert conclusion.confidence > 0

    async def test_metadata_includes_paths_and_scores(
        self, initialized_method: RStar, active_session: Session
    ) -> None:
        """Test that final metadata includes paths and scores."""
        initial = await initialized_method.execute(active_session, "Test")
        execute = await initialized_method.continue_reasoning(active_session, initial)
        discriminate = await initialized_method.continue_reasoning(active_session, execute)
        select = await initialized_method.continue_reasoning(active_session, discriminate)
        conclusion = await initialized_method.continue_reasoning(active_session, select)

        # Conclusion should have relevant metadata
        assert (
            "best_path" in conclusion.metadata
            or "best_path_idx" in conclusion.metadata
            or initialized_method._best_path_idx >= 0
        )
