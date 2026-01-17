"""Unit tests for SSR (Socratic Self-Refine) reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.ssr import (
    SSR,
    SSR_METADATA,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> SSR:
    """Create an SSR instance for testing."""
    return SSR()


class TestSSRInitialization:
    """Tests for SSR initialization."""

    def test_create_method(self, method: SSR):
        """Test that SSR can be instantiated."""
        assert method is not None
        assert isinstance(method, SSR)

    def test_initial_state(self, method: SSR):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"

    async def test_initialize(self, method: SSR):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: SSR):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: SSR):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestSSRProperties:
    """Tests for SSR property accessors."""

    def test_identifier_property(self, method: SSR):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.SSR

    def test_name_property(self, method: SSR):
        """Test that name returns the correct human-readable name."""
        assert "Socratic" in method.name

    def test_description_property(self, method: SSR):
        """Test that description returns the correct method description."""
        assert "socratic" in method.description.lower()

    def test_category_property(self, method: SSR):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestSSRMetadata:
    """Tests for SSR metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert SSR_METADATA.identifier == MethodIdentifier.SSR

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert SSR_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"socratic", "self-refine"}
        assert expected_tags.issubset(SSR_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= SSR_METADATA.complexity <= 10


class TestSSRExecution:
    """Tests for SSR execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: SSR, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: SSR, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Why is the sky blue?")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.SSR
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: SSR, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "generate"


class TestSSRContinuation:
    """Tests for SSR continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: SSR, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "generate"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "socratic_question"

    async def test_continue_sets_parent(
        self, method: SSR, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: SSR, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
