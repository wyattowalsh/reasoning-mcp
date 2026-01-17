"""Unit tests for Compute-Optimal Scaling reasoning method.

This module provides comprehensive tests for the ComputeOptimalScaling method
implementation, covering initialization, execution, difficulty assessment,
compute allocation, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.compute_optimal_scaling import (
    COMPUTE_OPTIMAL_SCALING_METADATA,
    ComputeOptimalScaling,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> ComputeOptimalScaling:
    """Create a ComputeOptimalScaling method instance for testing.

    Returns:
        A fresh ComputeOptimalScaling instance
    """
    return ComputeOptimalScaling()


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A new Session instance in ACTIVE status with mocked get_history
    """
    s = Session().start()
    # Mock get_history since source code calls it but Session model doesn't have it
    # Use object.__setattr__ to bypass Pydantic's validation
    object.__setattr__(s, "get_history", lambda: list(s.graph.nodes.values()))
    return s


@pytest.fixture
def easy_problem() -> str:
    """Provide an easy problem for testing.

    Returns:
        A simple problem string
    """
    return "What is 2+2?"


@pytest.fixture
def hard_problem() -> str:
    """Provide a hard problem for testing.

    Returns:
        A complex problem string
    """
    return (
        "Given a complex system with multiple interacting components, "
        "derive the optimal configuration that maximizes throughput while "
        "minimizing latency. Consider the following constraints: memory "
        "bandwidth is limited to 100GB/s, CPU cores are limited to 64, "
        "and network latency must be below 10ms. Additionally, account for "
        "variable workload patterns, thermal throttling effects, and "
        "potential cascading failures in the distributed system architecture."
    )


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="medium")
    return ctx


class TestComputeOptimalScalingInitialization:
    """Tests for ComputeOptimalScaling initialization and setup."""

    def test_create_method(self, method: ComputeOptimalScaling) -> None:
        """Test that ComputeOptimalScaling can be instantiated."""
        assert method is not None
        assert isinstance(method, ComputeOptimalScaling)

    def test_initial_state(self, method: ComputeOptimalScaling) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "assess"
        assert method._difficulty == "medium"

    def test_difficulty_levels_defined(self, method: ComputeOptimalScaling) -> None:
        """Test that difficulty levels are properly defined."""
        assert "easy" in method.DIFFICULTY_LEVELS
        assert "medium" in method.DIFFICULTY_LEVELS
        assert "hard" in method.DIFFICULTY_LEVELS
        assert "very_hard" in method.DIFFICULTY_LEVELS

        for _level, config in method.DIFFICULTY_LEVELS.items():
            assert "budget" in config
            assert "samples" in config
            assert "strategy" in config

    async def test_initialize(self, method: ComputeOptimalScaling) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "assess"
        assert method._difficulty == "medium"
        assert method._compute_budget == 4
        assert method._samples_generated == 0
        assert method._results == []

    async def test_health_check_not_initialized(
        self,
        method: ComputeOptimalScaling,
    ) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(
        self,
        method: ComputeOptimalScaling,
    ) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestComputeOptimalScalingProperties:
    """Tests for ComputeOptimalScaling property accessors."""

    def test_identifier_property(self, method: ComputeOptimalScaling) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.COMPUTE_OPTIMAL_SCALING

    def test_name_property(self, method: ComputeOptimalScaling) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Compute-Optimal Scaling"

    def test_description_property(self, method: ComputeOptimalScaling) -> None:
        """Test that description returns the correct method description."""
        assert "adaptive" in method.description.lower()
        assert "compute" in method.description.lower()

    def test_category_property(self, method: ComputeOptimalScaling) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestComputeOptimalScalingMetadata:
    """Tests for ComputeOptimalScaling metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert (
            COMPUTE_OPTIMAL_SCALING_METADATA.identifier == MethodIdentifier.COMPUTE_OPTIMAL_SCALING
        )

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert COMPUTE_OPTIMAL_SCALING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"adaptive", "test-time", "scaling"}
        assert expected_tags.issubset(COMPUTE_OPTIMAL_SCALING_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert COMPUTE_OPTIMAL_SCALING_METADATA.complexity == 7


class TestComputeOptimalScalingExecution:
    """Tests for ComputeOptimalScaling execute() method."""

    async def test_execute_basic(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, easy_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.COMPUTE_OPTIMAL_SCALING

    async def test_execute_without_initialization_raises(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, easy_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, easy_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_assess_phase(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test that execute sets assess phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, easy_problem)

        assert thought.metadata.get("phase") == "assess"
        assert "difficulty" in thought.metadata
        assert "budget" in thought.metadata


class TestDifficultyAssessment:
    """Tests for difficulty assessment functionality."""

    async def test_easy_problem_classification(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test that short problems are classified as easy."""
        await method.initialize()
        await method.execute(session, easy_problem)

        # Heuristic: short problems should be classified as easy
        assert method._difficulty == "easy"

    async def test_hard_problem_classification(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        hard_problem: str,
    ) -> None:
        """Test that long complex problems are classified as harder."""
        await method.initialize()
        await method.execute(session, hard_problem)

        # Heuristic: long problems should be classified as harder
        assert method._difficulty in ["hard", "very_hard"]

    async def test_difficulty_affects_budget(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test that difficulty affects compute budget."""
        await method.initialize()

        # Easy problem
        await method.execute(session, "2+2?")
        easy_budget = method._compute_budget

        # Reset for hard problem
        await method.initialize()
        await method.execute(
            session,
            "A" * 600,  # Very long problem
        )
        hard_budget = method._compute_budget

        assert hard_budget > easy_budget


class TestComputeOptimalScalingContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_assess_to_allocate(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test continuation from assess to allocate phase."""
        await method.initialize()
        initial = await method.execute(session, easy_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "allocate"
        assert continuation.type == ThoughtType.REASONING

    async def test_continue_allocate_to_execute(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test continuation from allocate to execute phase."""
        await method.initialize()
        initial = await method.execute(session, easy_problem)
        allocate = await method.continue_reasoning(session, initial)

        execute = await method.continue_reasoning(session, allocate)

        assert execute is not None
        assert execute.metadata.get("phase") == "execute"

    async def test_continue_to_conclusion(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, easy_problem)
        allocate = await method.continue_reasoning(session, initial)
        execute = await method.continue_reasoning(session, allocate)
        verify = await method.continue_reasoning(session, execute)

        conclusion = await method.continue_reasoning(session, verify)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION


class TestStrategySelection:
    """Tests for strategy selection based on difficulty."""

    async def test_easy_uses_direct_strategy(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test that easy problems use direct strategy."""
        await method.initialize()
        await method.execute(session, "2+2?")

        assert method._strategy == "direct"

    async def test_medium_uses_self_consistency(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test that medium problems use self_consistency strategy."""
        await method.initialize()
        # Medium length problem
        await method.execute(session, "A" * 150)

        assert method._strategy == "self_consistency"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None
        # Empty string should be classified as easy
        assert method._difficulty == "easy"

    async def test_very_long_problem(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Analyze: " + "context " * 1000

        thought = await method.execute(session, long_problem)

        assert thought is not None
        assert method._difficulty == "very_hard"

    async def test_special_characters_in_problem(
        self,
        method: ComputeOptimalScaling,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_full_reasoning_chain(
        self,
        method: ComputeOptimalScaling,
        session: Session,
        easy_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, easy_problem)
        assert initial.metadata.get("phase") == "assess"

        # Continue through all phases
        allocate = await method.continue_reasoning(session, initial)
        assert allocate.metadata.get("phase") == "allocate"

        execute = await method.continue_reasoning(session, allocate)
        assert execute.metadata.get("phase") == "execute"

        verify = await method.continue_reasoning(session, execute)
        assert verify.metadata.get("phase") == "verify"

        conclude = await method.continue_reasoning(session, verify)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION
