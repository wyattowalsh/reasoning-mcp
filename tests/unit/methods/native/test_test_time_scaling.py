"""Unit tests for TestTimeScaling reasoning method."""
from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.test_time_scaling import (
    TEST_TIME_SCALING_METADATA,
    TestTimeScaling,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> TestTimeScaling:
    """Create a TestTimeScaling instance for testing."""
    return TestTimeScaling()


class TestTestTimeScalingInitialization:
    """Tests for TestTimeScaling initialization."""

    def test_create_method(self, method: TestTimeScaling):
        """Test that TestTimeScaling can be instantiated."""
        assert method is not None
        assert isinstance(method, TestTimeScaling)

    def test_initial_state(self, method: TestTimeScaling):
        """Test initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "analyze"

    async def test_initialize(self, method: TestTimeScaling):
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0

    async def test_health_check_not_initialized(self, method: TestTimeScaling):
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: TestTimeScaling):
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestTestTimeScalingProperties:
    """Tests for TestTimeScaling property accessors."""

    def test_identifier_property(self, method: TestTimeScaling):
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.TEST_TIME_SCALING

    def test_name_property(self, method: TestTimeScaling):
        """Test that name returns the correct human-readable name."""
        assert "Test-Time" in method.name or "Scaling" in method.name

    def test_description_property(self, method: TestTimeScaling):
        """Test that description returns the correct method description."""
        assert "compute" in method.description.lower() or "scaling" in method.description.lower()

    def test_category_property(self, method: TestTimeScaling):
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.ADVANCED


class TestTestTimeScalingMetadata:
    """Tests for TestTimeScaling metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert TEST_TIME_SCALING_METADATA.identifier == MethodIdentifier.TEST_TIME_SCALING

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert TEST_TIME_SCALING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"test-time-compute", "extended-thinking"}
        assert expected_tags.issubset(TEST_TIME_SCALING_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= TEST_TIME_SCALING_METADATA.complexity <= 10


class TestTestTimeScalingExecution:
    """Tests for TestTimeScaling execute() method."""

    async def test_execute_without_initialization_raises(
        self, method: TestTimeScaling, session: Session
    ):
        """Test that execute raises if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    async def test_execute_basic(self, method: TestTimeScaling, session: Session):
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, "Solve complex problem")

        assert thought is not None
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.TEST_TIME_SCALING
        assert thought.type == ThoughtType.INITIAL

    async def test_execute_sets_metadata(self, method: TestTimeScaling, session: Session):
        """Test that execute sets correct metadata."""
        await method.initialize()
        thought = await method.execute(session, "Test problem")

        assert "phase" in thought.metadata
        assert thought.metadata["phase"] == "analyze"


class TestTestTimeScalingContinuation:
    """Tests for TestTimeScaling continue_reasoning() method."""

    async def test_continue_reasoning_advances_phase(
        self, method: TestTimeScaling, session: Session
    ):
        """Test that continue_reasoning advances through phases."""
        await method.initialize()
        initial = await method.execute(session, "Test problem")
        assert initial.metadata["phase"] == "analyze"

        step2 = await method.continue_reasoning(session, initial)
        assert step2.metadata["phase"] == "expand"

    async def test_continue_sets_parent(
        self, method: TestTimeScaling, session: Session
    ):
        """Test that continuation has correct parent_id."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self, method: TestTimeScaling, session: Session
    ):
        """Test that continuation increments depth."""
        await method.initialize()
        initial = await method.execute(session, "Test")
        continuation = await method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1
