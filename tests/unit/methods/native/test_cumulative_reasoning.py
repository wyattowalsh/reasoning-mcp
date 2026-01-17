"""Unit tests for CumulativeReasoning method.

This module provides basic tests for the CumulativeReasoning method implementation,
covering initialization, execution, and basic functionality.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.cumulative_reasoning import (
    CUMULATIVE_REASONING_METADATA,
    CumulativeReasoning,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType


@pytest.fixture
def cumulative_method() -> CumulativeReasoning:
    """Create a CumulativeReasoning method instance for testing.

    Returns:
        A fresh CumulativeReasoning instance
    """
    return CumulativeReasoning()


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
    return "Prove that the sum of angles in a triangle is 180 degrees."


class TestCumulativeReasoningInitialization:
    """Tests for CumulativeReasoning initialization and setup."""

    def test_create_method(self, cumulative_method: CumulativeReasoning):
        """Test that CumulativeReasoning can be instantiated."""
        assert cumulative_method is not None
        assert isinstance(cumulative_method, CumulativeReasoning)

    def test_initial_state(self, cumulative_method: CumulativeReasoning):
        """Test that a new method starts in the correct initial state."""
        assert cumulative_method._initialized is False
        assert cumulative_method._step_counter == 0
        assert cumulative_method._current_phase == "initialize"
        assert len(cumulative_method._propositions) == 0
        assert cumulative_method._proposition_count == 0

    async def test_initialize(self, cumulative_method: CumulativeReasoning):
        """Test that initialize() sets up the method correctly."""
        await cumulative_method.initialize()
        assert cumulative_method._initialized is True
        assert cumulative_method._step_counter == 0
        assert cumulative_method._current_phase == "initialize"

    def test_metadata(self):
        """Test that metadata is properly configured."""
        assert CUMULATIVE_REASONING_METADATA.identifier == MethodIdentifier.CUMULATIVE_REASONING
        assert CUMULATIVE_REASONING_METADATA.name == "Cumulative Reasoning"
        assert CUMULATIVE_REASONING_METADATA.supports_branching is True
        assert CUMULATIVE_REASONING_METADATA.supports_revision is True


class TestCumulativeReasoningExecution:
    """Tests for CumulativeReasoning execution."""

    async def test_execute_without_initialization_raises_error(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that executing without initialization raises an error."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await cumulative_method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates an initial thought."""
        await cumulative_method.initialize()
        thought = await cumulative_method.execute(session, sample_problem)

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CUMULATIVE_REASONING
        assert thought.step_number == 1
        assert "Initialize" in thought.content or "initialize" in thought.content.lower()
        assert thought.metadata["phase"] == "initialize"

    async def test_execute_adds_thought_to_session(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute adds the thought to the session."""
        await cumulative_method.initialize()
        await cumulative_method.execute(session, sample_problem)

        assert session.thought_count == 1
        assert session.current_method == MethodIdentifier.CUMULATIVE_REASONING


class TestCumulativeReasoningContinuation:
    """Tests for CumulativeReasoning continuation."""

    async def test_continue_from_initialize_to_propose(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation from initialize moves to propose phase."""
        await cumulative_method.initialize()
        initial = await cumulative_method.execute(session, sample_problem)

        continued = await cumulative_method.continue_reasoning(session, initial)

        assert continued is not None
        assert continued.type == ThoughtType.HYPOTHESIS
        assert continued.metadata["phase"] == "propose"
        assert continued.parent_id == initial.id
        assert cumulative_method._proposition_count == 1

    async def test_continue_from_propose_to_verify(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation from propose moves to verify phase."""
        await cumulative_method.initialize()
        initial = await cumulative_method.execute(session, sample_problem)
        propose = await cumulative_method.continue_reasoning(session, initial)

        verify = await cumulative_method.continue_reasoning(session, propose)

        assert verify is not None
        assert verify.type == ThoughtType.VERIFICATION
        assert verify.metadata["phase"] == "verify"
        assert verify.parent_id == propose.id

    async def test_continue_from_verify_to_accumulate(
        self,
        cumulative_method: CumulativeReasoning,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation from verify moves to accumulate phase."""
        await cumulative_method.initialize()
        initial = await cumulative_method.execute(session, sample_problem)
        propose = await cumulative_method.continue_reasoning(session, initial)
        verify = await cumulative_method.continue_reasoning(session, propose)

        accumulate = await cumulative_method.continue_reasoning(session, verify)

        assert accumulate is not None
        assert accumulate.type == ThoughtType.SYNTHESIS
        assert accumulate.metadata["phase"] == "accumulate"
        assert len(cumulative_method._propositions) == 1


class TestCumulativeReasoningHealthCheck:
    """Tests for CumulativeReasoning health check."""

    async def test_health_check_before_initialization(
        self,
        cumulative_method: CumulativeReasoning,
    ):
        """Test that health check returns False before initialization."""
        is_healthy = await cumulative_method.health_check()
        assert is_healthy is False

    async def test_health_check_after_initialization(
        self,
        cumulative_method: CumulativeReasoning,
    ):
        """Test that health check returns True after initialization."""
        await cumulative_method.initialize()
        is_healthy = await cumulative_method.health_check()
        assert is_healthy is True
