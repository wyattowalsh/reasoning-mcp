"""Unit tests for CognitiveTools reasoning method.

This module provides comprehensive tests for the CognitiveTools method
implementation, covering initialization, execution, tool selection,
application, evaluation, combination, and edge cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.cognitive_tools import (
    COGNITIVE_TOOLS_METADATA,
    CognitiveTools,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def ct_method() -> CognitiveTools:
    """Create a CognitiveTools method instance for testing.

    Returns:
        A fresh CognitiveTools instance
    """
    return CognitiveTools()


@pytest.fixture
def ct_no_elicitation() -> CognitiveTools:
    """Create a CognitiveTools method with elicitation disabled.

    Returns:
        A CognitiveTools instance with elicitation disabled
    """
    return CognitiveTools(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> CognitiveTools:
    """Create an initialized CognitiveTools method instance.

    Returns:
        An initialized CognitiveTools instance
    """
    method = CognitiveTools()
    await method.initialize()
    return method


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
    return "Why do birds migrate south for winter?"


@pytest.fixture
def deductive_problem() -> str:
    """Provide a deductive reasoning problem.

    Returns:
        A deductive problem string
    """
    return "If all birds can fly, and penguins are birds, can penguins fly?"


class TestCognitiveToolsMetadata:
    """Tests for COGNITIVE_TOOLS_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert COGNITIVE_TOOLS_METADATA.identifier == MethodIdentifier.COGNITIVE_TOOLS

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert COGNITIVE_TOOLS_METADATA.name == "Cognitive Tools"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = COGNITIVE_TOOLS_METADATA.description.lower()
        assert "cognitive" in desc
        assert "modular" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in ADVANCED category."""
        assert COGNITIVE_TOOLS_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has high complexity."""
        assert COGNITIVE_TOOLS_METADATA.complexity == 8
        assert 1 <= COGNITIVE_TOOLS_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that CognitiveTools supports branching."""
        assert COGNITIVE_TOOLS_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that CognitiveTools supports revision."""
        assert COGNITIVE_TOOLS_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that CognitiveTools doesn't require context."""
        assert COGNITIVE_TOOLS_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert COGNITIVE_TOOLS_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert COGNITIVE_TOOLS_METADATA.max_thoughts == 20

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "modular" in COGNITIVE_TOOLS_METADATA.tags
        assert "cognitive-operations" in COGNITIVE_TOOLS_METADATA.tags
        assert "agentic" in COGNITIVE_TOOLS_METADATA.tags
        assert "analogical" in COGNITIVE_TOOLS_METADATA.tags
        assert "deductive" in COGNITIVE_TOOLS_METADATA.tags
        assert "abductive" in COGNITIVE_TOOLS_METADATA.tags
        assert "inductive" in COGNITIVE_TOOLS_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(COGNITIVE_TOOLS_METADATA.best_for).lower()
        assert "complex" in best_for_text or "reasoning" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(COGNITIVE_TOOLS_METADATA.not_recommended_for).lower()
        assert "simple" in not_recommended


class TestCognitiveToolsInitialization:
    """Tests for CognitiveTools method initialization."""

    def test_create_instance(self, ct_method: CognitiveTools) -> None:
        """Test that we can create a CognitiveTools instance."""
        assert isinstance(ct_method, CognitiveTools)

    def test_initial_state(self, ct_method: CognitiveTools) -> None:
        """Test that initial state is correct before initialization."""
        assert ct_method._initialized is False
        assert ct_method._step_counter == 0
        assert ct_method._current_phase == "select_tool"
        assert ct_method._selected_tools == []
        assert ct_method._tool_results == {}

    def test_available_tools(self, ct_method: CognitiveTools) -> None:
        """Test that all cognitive tools are available."""
        assert CognitiveTools.ANALOGICAL in CognitiveTools.ALL_TOOLS
        assert CognitiveTools.DEDUCTIVE in CognitiveTools.ALL_TOOLS
        assert CognitiveTools.ABDUCTIVE in CognitiveTools.ALL_TOOLS
        assert CognitiveTools.INDUCTIVE in CognitiveTools.ALL_TOOLS

    def test_default_elicitation_enabled(self, ct_method: CognitiveTools) -> None:
        """Test that elicitation is enabled by default."""
        assert ct_method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, ct_no_elicitation: CognitiveTools) -> None:
        """Test that elicitation can be disabled."""
        assert ct_no_elicitation.enable_elicitation is False

    async def test_initialize(self, ct_method: CognitiveTools) -> None:
        """Test that initialize sets up the method correctly."""
        await ct_method.initialize()
        assert ct_method._initialized is True
        assert ct_method._step_counter == 0
        assert ct_method._current_phase == "select_tool"
        assert ct_method._selected_tools == []
        assert ct_method._tool_results == {}

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = CognitiveTools()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._selected_tools = ["analogical", "deductive"]
        method._tool_results = {"analogical": "result"}

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "select_tool"
        assert method._selected_tools == []
        assert method._tool_results == {}

    async def test_health_check_before_init(self, ct_method: CognitiveTools) -> None:
        """Test health_check returns False before initialization."""
        health = await ct_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: CognitiveTools) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestCognitiveToolsProperties:
    """Tests for CognitiveTools method properties."""

    def test_identifier_property(self, ct_method: CognitiveTools) -> None:
        """Test that identifier property returns correct value."""
        assert ct_method.identifier == MethodIdentifier.COGNITIVE_TOOLS

    def test_name_property(self, ct_method: CognitiveTools) -> None:
        """Test that name property returns correct value."""
        assert ct_method.name == "Cognitive Tools"

    def test_description_property(self, ct_method: CognitiveTools) -> None:
        """Test that description property returns correct value."""
        assert ct_method.description == COGNITIVE_TOOLS_METADATA.description

    def test_category_property(self, ct_method: CognitiveTools) -> None:
        """Test that category property returns correct value."""
        assert ct_method.category == MethodCategory.ADVANCED


class TestCognitiveToolsExecution:
    """Tests for basic execution of CognitiveTools reasoning."""

    async def test_execute_without_initialization_fails(
        self, ct_method: CognitiveTools, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ct_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates tool selection thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.COGNITIVE_TOOLS
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_select_tool(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to select_tool."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "select_tool"
        assert thought.metadata["phase"] == "select_tool"

    async def test_execute_selects_tools(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that execute selects appropriate tools."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._selected_tools) > 0
        # "Why" should trigger abductive reasoning
        assert "abductive" in initialized_method._selected_tools

    async def test_execute_adds_to_session(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.COGNITIVE_TOOLS

    async def test_execute_content_format(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "Tool Selection" in thought.content or "tool" in thought.content.lower()

    async def test_execute_metadata(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "select_tool"
        assert "selected_tools" in thought.metadata
        assert "selection_reasoning" in thought.metadata

    async def test_tool_preference_in_context(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that tool_preference in context is respected."""
        context: dict[str, Any] = {"tool_preference": "deductive"}

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
            context=context,
        )

        assert initialized_method._selected_tools == ["deductive"]


class TestToolSelection:
    """Tests for cognitive tool selection logic."""

    async def test_deductive_keywords_select_deductive(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that deductive keywords select deductive tool."""
        await initialized_method.execute(
            session=session,
            input_text="If A then B, therefore C must follow",
        )

        assert "deductive" in initialized_method._selected_tools

    async def test_analogical_keywords_select_analogical(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that analogical keywords select analogical tool."""
        await initialized_method.execute(
            session=session,
            input_text="This problem is similar to the one we solved before",
        )

        assert "analogical" in initialized_method._selected_tools

    async def test_inductive_keywords_select_inductive(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that inductive keywords select inductive tool."""
        await initialized_method.execute(
            session=session,
            input_text="What pattern can we generalize from these examples?",
        )

        assert "inductive" in initialized_method._selected_tools

    async def test_abductive_keywords_select_abductive(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that abductive keywords select abductive tool."""
        await initialized_method.execute(
            session=session,
            input_text="Why did this happen? What is the best explanation?",
        )

        assert "abductive" in initialized_method._selected_tools

    async def test_default_to_abductive_when_unclear(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that abductive is default when no clear keywords."""
        await initialized_method.execute(
            session=session,
            input_text="Consider the following scenario",
        )

        assert "abductive" in initialized_method._selected_tools


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, ct_method: CognitiveTools, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "select_tool", "selected_tools": []}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await ct_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_select_to_apply(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from select_tool to apply."""
        select_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        assert initialized_method._current_phase == "apply"
        assert apply_thought.metadata["phase"] == "apply"
        assert apply_thought.type == ThoughtType.REASONING

    async def test_phase_transition_apply_to_evaluate(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from apply to evaluate."""
        select_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        evaluate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=apply_thought,
        )

        assert initialized_method._current_phase == "evaluate"
        assert evaluate_thought.metadata["phase"] == "evaluate"
        assert evaluate_thought.type == ThoughtType.VERIFICATION

    async def test_single_tool_skips_combine(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that single tool selection skips combine phase."""
        # Use a simple problem that selects only one tool
        context: dict[str, Any] = {"tool_preference": "deductive"}
        select_thought = await initialized_method.execute(
            session=session,
            input_text="If A then B",
            context=context,
        )
        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        evaluate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=apply_thought,
        )

        # Should go to conclude, not combine
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=evaluate_thought,
        )

        assert conclude_thought.metadata["phase"] == "conclude"

    async def test_step_counter_increments(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        select_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert select_thought.step_number == 1

        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert apply_thought.step_number == 2

    async def test_parent_id_set_correctly(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        select_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )

        assert apply_thought.parent_id == select_thought.id

    async def test_depth_increases(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        select_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert select_thought.depth == 0

        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert apply_thought.depth == 1


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that confidence generally increases through phases."""
        context: dict[str, Any] = {"tool_preference": "abductive"}  # Single tool for simpler flow

        select_thought = await initialized_method.execute(
            session=session,
            input_text="Why did this happen?",
            context=context,
        )
        select_confidence = select_thought.confidence

        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        apply_confidence = apply_thought.confidence

        evaluate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=apply_thought,
        )
        evaluate_confidence = evaluate_thought.confidence

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=evaluate_thought,
        )
        conclude_confidence = conclude_thought.confidence

        # Confidence should generally increase
        assert select_confidence < apply_confidence
        assert apply_confidence <= evaluate_confidence
        assert evaluate_confidence <= conclude_confidence


class TestEdgeCases:
    """Tests for edge cases in CognitiveTools reasoning."""

    async def test_empty_query(self, initialized_method: CognitiveTools, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Analyze this problem: " + "why " * 100 + "explain the reason"
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="为什么鸟类要南飞过冬？",
        )

        assert thought is not None

    async def test_complete_reasoning_flow_single_tool(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test complete reasoning flow with single tool."""
        context: dict[str, Any] = {"tool_preference": "deductive"}

        # Phase 1: Select
        select_thought = await initialized_method.execute(
            session=session,
            input_text="If A then B",
            context=context,
        )
        assert select_thought.type == ThoughtType.INITIAL
        assert select_thought.metadata["phase"] == "select_tool"

        # Phase 2: Apply
        apply_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=select_thought,
        )
        assert apply_thought.type == ThoughtType.REASONING
        assert apply_thought.metadata["phase"] == "apply"

        # Phase 3: Evaluate
        evaluate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=apply_thought,
        )
        assert evaluate_thought.type == ThoughtType.VERIFICATION
        assert evaluate_thought.metadata["phase"] == "evaluate"

        # Phase 4: Conclude (skipping combine for single tool)
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=evaluate_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 4
        assert session.current_method == MethodIdentifier.COGNITIVE_TOOLS

    async def test_multiple_execution_cycles(
        self, initialized_method: CognitiveTools, session: Session
    ) -> None:
        """Test that method can handle multiple execution cycles."""
        # First execution
        thought1 = await initialized_method.execute(
            session=session,
            input_text="First problem",
        )
        assert thought1.step_number == 1

        # Reinitialize
        await initialized_method.initialize()

        # Second execution
        thought2 = await initialized_method.execute(
            session=session,
            input_text="Second problem",
        )
        assert thought2.step_number == 1


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.COGNITIVE_TOOLS)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        ct_thoughts = session.get_thoughts_by_method(MethodIdentifier.COGNITIVE_TOOLS)
        assert len(ct_thoughts) > 0


class TestElicitationBehavior:
    """Tests for elicitation-related behavior."""

    async def test_elicitation_disabled_skips_interactions(
        self, ct_no_elicitation: CognitiveTools, session: Session, sample_problem: str
    ) -> None:
        """Test that disabled elicitation skips user interactions."""
        await ct_no_elicitation.initialize()

        # Execute should work without any elicitation
        thought = await ct_no_elicitation.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
