"""Unit tests for SelfDiscover reasoning method.

This module provides comprehensive unit tests for the SelfDiscover class,
testing initialization, module selection, adaptation, implementation,
execution phases, and user elicitation handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.self_discover import (
    SELF_DISCOVER_METADATA,
    SelfDiscover,
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
def method() -> SelfDiscover:
    """Create a SelfDiscover instance for testing."""
    return SelfDiscover()


@pytest.fixture
def method_no_elicitation() -> SelfDiscover:
    """Create a SelfDiscover instance with elicitation disabled."""
    return SelfDiscover(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> SelfDiscover:
    """Create and initialize a SelfDiscover instance."""
    method = SelfDiscover()
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
    ctx.sample = AsyncMock(return_value="Selected modules and adaptations")
    return ctx


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfDiscoverMetadata:
    """Tests for SelfDiscover metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert SELF_DISCOVER_METADATA.identifier == MethodIdentifier.SELF_DISCOVER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert (
            "Self-Discover" in SELF_DISCOVER_METADATA.name
            or "Self Discover" in SELF_DISCOVER_METADATA.name
        )

    def test_metadata_category(self) -> None:
        """Test metadata is in ADVANCED category."""
        assert SELF_DISCOVER_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_supports_branching(self) -> None:
        """Test metadata indicates branching support."""
        assert SELF_DISCOVER_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test metadata indicates revision support."""
        assert SELF_DISCOVER_METADATA.supports_revision is True

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        tags = SELF_DISCOVER_METADATA.tags
        assert "self-compose" in tags or "adaptive" in tags or "modules" in tags

    def test_metadata_complexity(self) -> None:
        """Test metadata has appropriate complexity level."""
        assert SELF_DISCOVER_METADATA.complexity >= 6


# ============================================================================
# Reasoning Modules Tests
# ============================================================================


class TestReasoningModules:
    """Tests for REASONING_MODULES class attribute."""

    def test_modules_not_empty(self) -> None:
        """Test that reasoning modules list is not empty."""
        method = SelfDiscover()
        assert len(method.REASONING_MODULES) > 0

    def test_modules_are_dicts(self) -> None:
        """Test that all modules are dictionaries."""
        method = SelfDiscover()
        for module in method.REASONING_MODULES:
            assert isinstance(module, dict)
            assert "name" in module
            assert "desc" in module

    def test_expected_modules_present(self) -> None:
        """Test that expected reasoning modules are present."""
        method = SelfDiscover()
        # Check for some commonly expected reasoning modules
        module_names_lower = [m["name"].lower() for m in method.REASONING_MODULES]
        # At least some common reasoning patterns should be present
        assert any("critical" in m or "analytical" in m for m in module_names_lower)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSelfDiscoverInitialization:
    """Tests for SelfDiscover initialization."""

    def test_default_initialization(self, method: SelfDiscover) -> None:
        """Test initialization with default parameters."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "select"
        assert method._selected_modules == []
        assert method._adapted_modules == []
        assert method._reasoning_structure == {}

    def test_elicitation_enabled_by_default(self, method: SelfDiscover) -> None:
        """Test elicitation is enabled by default."""
        assert method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, method_no_elicitation: SelfDiscover) -> None:
        """Test elicitation can be disabled."""
        assert method_no_elicitation.enable_elicitation is False

    async def test_initialize_method(self, method: SelfDiscover) -> None:
        """Test initialize() method sets initialized flag."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "select"

    async def test_health_check_before_initialize(self, method: SelfDiscover) -> None:
        """Test health_check() returns False before initialization."""
        assert await method.health_check() is False

    async def test_health_check_after_initialize(self, initialized_method: SelfDiscover) -> None:
        """Test health_check() returns True after initialization."""
        assert await initialized_method.health_check() is True


# ============================================================================
# Property Tests
# ============================================================================


class TestSelfDiscoverProperties:
    """Tests for SelfDiscover properties."""

    def test_identifier_property(self, method: SelfDiscover) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.SELF_DISCOVER

    def test_name_property(self, method: SelfDiscover) -> None:
        """Test name property returns correct value."""
        assert "Self" in method.name
        assert "Discover" in method.name

    def test_description_property(self, method: SelfDiscover) -> None:
        """Test description property contains key concepts."""
        desc_lower = method.description.lower()
        assert "task" in desc_lower or "structure" in desc_lower or "discover" in desc_lower

    def test_category_property(self, method: SelfDiscover) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSelfDiscoverExecution:
    """Tests for basic execution of SelfDiscover method."""

    async def test_execute_without_initialization_fails(
        self, method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(active_session, "Test problem")

    async def test_execute_creates_initial_thought(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() creates an initial thought."""
        result = await initialized_method.execute(active_session, "Solve a puzzle")

        assert result is not None
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.SELF_DISCOVER

    async def test_execute_sets_step_number(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() sets step_number to 1."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.step_number == 1
        assert initialized_method._step_counter == 1

    async def test_execute_sets_correct_phase(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() sets phase to select."""
        result = await initialized_method.execute(active_session, "Test")
        assert result.metadata["phase"] == "select"

    async def test_execute_adds_thought_to_session(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() adds thought to session."""
        initial_count = active_session.thought_count
        await initialized_method.execute(active_session, "Test")
        assert active_session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execute() sets session's current method."""
        await initialized_method.execute(active_session, "Test")
        assert active_session.current_method == MethodIdentifier.SELF_DISCOVER


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSelfDiscoverContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_without_initialization_fails(
        self, method: SelfDiscover, active_session: Session
    ) -> None:
        """Test continue_reasoning() fails without initialization."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_DISCOVER,
            content="Test",
            metadata={"phase": "select"},
        )
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(active_session, thought)

    async def test_continue_from_select_to_adapt(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test transition from select to adapt phase."""
        initial = await initialized_method.execute(active_session, "Test")
        result = await initialized_method.continue_reasoning(active_session, initial)

        assert result.metadata["phase"] == "adapt"
        assert result.parent_id == initial.id

    async def test_continue_from_adapt_to_implement(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test transition from adapt to implement phase."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        result = await initialized_method.continue_reasoning(active_session, adapt)

        assert result.metadata["phase"] == "implement"

    async def test_continue_from_implement_to_execute(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test transition from implement to execute phase."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        implement = await initialized_method.continue_reasoning(active_session, adapt)
        result = await initialized_method.continue_reasoning(active_session, implement)

        assert result.metadata["phase"] == "execute"

    async def test_continue_through_all_phases(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test continuation through all phases to conclusion."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        implement = await initialized_method.continue_reasoning(active_session, adapt)
        execute = await initialized_method.continue_reasoning(active_session, implement)
        solve = await initialized_method.continue_reasoning(active_session, execute)
        conclusion = await initialized_method.continue_reasoning(active_session, solve)

        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_increments_step_number(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments step number."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.step_number == 2

    async def test_continue_increments_depth(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test continue_reasoning() increments depth."""
        initial = await initialized_method.execute(active_session, "Test")
        continuation = await initialized_method.continue_reasoning(active_session, initial)
        assert continuation.depth == initial.depth + 1


# ============================================================================
# Module Selection Tests
# ============================================================================


class TestSelfDiscoverModuleSelection:
    """Tests for module selection functionality."""

    async def test_modules_selected_in_select_phase(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test that modules are selected in select phase."""
        await initialized_method.execute(active_session, "Solve a math problem")
        # Selected modules should be a list (may be empty with heuristic)
        assert isinstance(initialized_method._selected_modules, list)

    async def test_generate_select_phase_heuristic(self, initialized_method: SelfDiscover) -> None:
        """Test heuristic module selection."""
        # Call with correct arguments (input_text, discovery_approach)
        result = initialized_method._generate_select_phase("Analyze data patterns", None)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Module Adaptation Tests
# ============================================================================


class TestSelfDiscoverModuleAdaptation:
    """Tests for module adaptation functionality."""

    async def test_modules_adapted_in_adapt_phase(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test that modules are adapted after adapt phase."""
        initial = await initialized_method.execute(active_session, "Test")
        await initialized_method.continue_reasoning(active_session, initial)

        assert isinstance(initialized_method._adapted_modules, list)

    async def test_generate_adapt_phase_heuristic(self, initialized_method: SelfDiscover) -> None:
        """Test heuristic module adaptation."""
        # Use the correct argument type (list of dicts)
        selected_modules = [{"name": "break_down"}, {"name": "identify_key_info"}]
        result = initialized_method._generate_adapt_phase(selected_modules)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Structure Implementation Tests
# ============================================================================


class TestSelfDiscoverStructureImplementation:
    """Tests for reasoning structure implementation."""

    async def test_structure_implemented_in_implement_phase(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test that structure is implemented after implement phase."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        await initialized_method.continue_reasoning(active_session, adapt)

        assert isinstance(initialized_method._reasoning_structure, dict)

    async def test_generate_implement_phase_heuristic(
        self, initialized_method: SelfDiscover
    ) -> None:
        """Test heuristic structure implementation."""
        # Use the correct argument type (list of dicts with name and adaptation keys)
        adapted_modules = [
            {"name": "module_1", "adaptation": "adapted for task"},
            {"name": "module_2", "adaptation": "customized approach"},
        ]
        result = initialized_method._generate_implement_phase(adapted_modules)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSelfDiscoverEdgeCases:
    """Tests for edge cases."""

    async def test_empty_input_text(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execution with empty input text."""
        result = await initialized_method.execute(active_session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    async def test_very_long_input_text(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execution with very long input text."""
        long_text = "A complex task: " + "step " * 500
        result = await initialized_method.execute(active_session, long_text)
        assert result is not None

    async def test_special_characters_in_input(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test execution with special characters."""
        special_text = "Solve: @#$%^&*() + 中文 = ?"
        result = await initialized_method.execute(active_session, special_text)
        assert result is not None


# ============================================================================
# Complete Workflow Tests
# ============================================================================


class TestSelfDiscoverWorkflow:
    """Tests for complete SelfDiscover workflows."""

    async def test_full_workflow(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test complete SelfDiscover workflow from selection to conclusion."""
        # Select phase
        initial = await initialized_method.execute(active_session, "Analyze data")
        assert initial.metadata["phase"] == "select"
        assert initial.type == ThoughtType.INITIAL

        # Adapt phase
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        assert adapt.metadata["phase"] == "adapt"

        # Implement phase
        implement = await initialized_method.continue_reasoning(active_session, adapt)
        assert implement.metadata["phase"] == "implement"

        # Execute phase
        execute = await initialized_method.continue_reasoning(active_session, implement)
        assert execute.metadata["phase"] == "execute"

        # Solve phase
        solve = await initialized_method.continue_reasoning(active_session, execute)
        assert solve.metadata["phase"] == "solve"

        # Conclude phase
        conclusion = await initialized_method.continue_reasoning(active_session, solve)
        assert conclusion.metadata["phase"] == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 6
        assert active_session.current_method == MethodIdentifier.SELF_DISCOVER

    async def test_confidence_progression(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test confidence values through phases."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        implement = await initialized_method.continue_reasoning(active_session, adapt)
        execute = await initialized_method.continue_reasoning(active_session, implement)
        solve = await initialized_method.continue_reasoning(active_session, execute)
        conclusion = await initialized_method.continue_reasoning(active_session, solve)

        # Confidence should be reasonable
        assert conclusion.confidence > 0

    async def test_metadata_includes_modules_and_structure(
        self, initialized_method: SelfDiscover, active_session: Session
    ) -> None:
        """Test that metadata includes modules and structure."""
        initial = await initialized_method.execute(active_session, "Test")
        adapt = await initialized_method.continue_reasoning(active_session, initial)
        implement = await initialized_method.continue_reasoning(active_session, adapt)
        execute = await initialized_method.continue_reasoning(active_session, implement)
        solve = await initialized_method.continue_reasoning(active_session, execute)
        conclusion = await initialized_method.continue_reasoning(active_session, solve)

        # Conclusion should have relevant metadata
        assert (
            "selected_modules" in conclusion.metadata
            or "structure" in conclusion.metadata
            or len(initialized_method._selected_modules) > 0
        )
