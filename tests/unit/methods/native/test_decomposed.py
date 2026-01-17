"""Unit tests for DecomposedPrompting reasoning method.

This module contains comprehensive tests for the DecomposedPromptingMethod class,
covering initialization, execution, task decomposition, specialist routing,
configuration options, result aggregation, and edge cases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.decomposed import (
    DECOMPOSED_PROMPTING_METADATA,
    DecomposedPrompting,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session, SessionConfig


class TestDecomposedPromptingMetadata:
    """Tests for DecomposedPrompting metadata."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert DECOMPOSED_PROMPTING_METADATA.identifier == MethodIdentifier.DECOMPOSED_PROMPTING

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert DECOMPOSED_PROMPTING_METADATA.name == "Decomposed Prompting"

    def test_metadata_description(self):
        """Test metadata has non-empty description."""
        assert len(DECOMPOSED_PROMPTING_METADATA.description) > 0
        assert "multi-specialist" in DECOMPOSED_PROMPTING_METADATA.description.lower()

    def test_metadata_category(self):
        """Test metadata category is SPECIALIZED."""
        assert DECOMPOSED_PROMPTING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        tags = DECOMPOSED_PROMPTING_METADATA.tags
        assert "decomposition" in tags
        assert "specialists" in tags
        assert "multi-disciplinary" in tags
        assert "parallel" in tags
        assert "integration" in tags

    def test_metadata_complexity(self):
        """Test metadata complexity is appropriate (medium-high)."""
        assert DECOMPOSED_PROMPTING_METADATA.complexity == 6

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert DECOMPOSED_PROMPTING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert DECOMPOSED_PROMPTING_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context is not required."""
        assert DECOMPOSED_PROMPTING_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test metadata specifies minimum thoughts."""
        assert DECOMPOSED_PROMPTING_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test metadata specifies unlimited max thoughts."""
        assert DECOMPOSED_PROMPTING_METADATA.max_thoughts == 0

    def test_metadata_best_for(self):
        """Test metadata lists appropriate use cases."""
        best_for = DECOMPOSED_PROMPTING_METADATA.best_for
        assert len(best_for) > 0
        assert "multi-disciplinary problems" in best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata lists inappropriate use cases."""
        not_recommended = DECOMPOSED_PROMPTING_METADATA.not_recommended_for
        assert len(not_recommended) > 0
        assert "single-domain problems" in not_recommended


class TestDecomposedPromptingInitialization:
    """Tests for DecomposedPrompting initialization and health checks."""

    @pytest.mark.asyncio
    async def test_init_creates_instance(self):
        """Test that __init__ creates a DecomposedPrompting instance."""
        method = DecomposedPrompting()
        assert method is not None
        assert isinstance(method, DecomposedPrompting)

    @pytest.mark.asyncio
    async def test_init_sets_uninitialized_state(self):
        """Test that new instance starts in uninitialized state."""
        method = DecomposedPrompting()
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._specialists == []
        assert method._specialist_outputs == {}
        assert method._decomposition_complete is False
        assert method._specialists_assigned is False

    @pytest.mark.asyncio
    async def test_health_check_before_init(self):
        """Test health check returns False before initialization."""
        method = DecomposedPrompting()
        health = await method.health_check()
        assert health is False

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize method sets correct state."""
        method = DecomposedPrompting()
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._specialists == []
        assert method._specialist_outputs == {}
        assert method._decomposition_complete is False
        assert method._specialists_assigned is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self):
        """Test health check returns True after initialization."""
        method = DecomposedPrompting()
        await method.initialize()
        health = await method.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_reinitialize_resets_state(self):
        """Test that calling initialize again resets state."""
        method = DecomposedPrompting()
        await method.initialize()

        # Manually modify state
        method._step_counter = 5
        method._specialists = [{"role": "test"}]
        method._specialist_outputs = {"test": "output"}
        method._decomposition_complete = True
        method._specialists_assigned = True

        # Reinitialize
        await method.initialize()

        # Verify reset
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._specialists == []
        assert method._specialist_outputs == {}
        assert method._decomposition_complete is False
        assert method._specialists_assigned is False


class TestDecomposedPromptingProperties:
    """Tests for DecomposedPrompting property accessors."""

    @pytest.mark.asyncio
    async def test_identifier_property(self):
        """Test identifier property returns correct value."""
        method = DecomposedPrompting()
        assert method.identifier == MethodIdentifier.DECOMPOSED_PROMPTING

    @pytest.mark.asyncio
    async def test_name_property(self):
        """Test name property returns correct value."""
        method = DecomposedPrompting()
        assert method.name == "Decomposed Prompting"

    @pytest.mark.asyncio
    async def test_description_property(self):
        """Test description property returns correct value."""
        method = DecomposedPrompting()
        assert method.description == DECOMPOSED_PROMPTING_METADATA.description

    @pytest.mark.asyncio
    async def test_category_property(self):
        """Test category property returns correct value."""
        method = DecomposedPrompting()
        assert method.category == MethodCategory.SPECIALIZED


class TestDecomposedPromptingExecution:
    """Tests for DecomposedPrompting execute method."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(self):
        """Test execute fails without initialization."""
        method = DecomposedPrompting()
        session = Session().start()

        with pytest.raises(
            RuntimeError,
            match="Decomposed Prompting method must be initialized before execution",
        ):
            await method.execute(session, "Test problem")

    @pytest.mark.asyncio
    async def test_execute_returns_thought_node(self):
        """Test execute returns a ThoughtNode."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Design a sustainable smart city")

        assert thought is not None
        assert hasattr(thought, "content")
        assert hasattr(thought, "type")
        assert hasattr(thought, "method_id")

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought_type(self):
        """Test execute creates INITIAL thought type."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Analyze complex problem")

        assert thought.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_execute_sets_correct_method_id(self):
        """Test execute sets correct method_id."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test input")

        assert thought.method_id == MethodIdentifier.DECOMPOSED_PROMPTING

    @pytest.mark.asyncio
    async def test_execute_sets_step_number_one(self):
        """Test execute sets step_number to 1."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test input")

        assert thought.step_number == 1

    @pytest.mark.asyncio
    async def test_execute_sets_depth_zero(self):
        """Test execute sets depth to 0."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test input")

        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(self):
        """Test execute sets reasonable confidence."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test input")

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.75  # Expected initial confidence

    @pytest.mark.asyncio
    async def test_execute_generates_expertise_identification_content(self):
        """Test execute generates expertise identification content."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Build multi-platform application")

        assert "Expertise Identification" in thought.content
        assert "Problem Analysis" in thought.content
        assert "Build multi-platform application" in thought.content

    @pytest.mark.asyncio
    async def test_execute_identifies_specialists(self):
        """Test execute identifies specialist roles."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        # Verify specialists were identified
        assert len(method._specialists) > 0
        # Check for expected specialist structure
        specialist = method._specialists[0]
        assert "role" in specialist
        assert "domain" in specialist
        assert "focus" in specialist

    @pytest.mark.asyncio
    async def test_execute_includes_metadata(self):
        """Test execute includes appropriate metadata."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()
        input_text = "Test problem"

        thought = await method.execute(session, input_text)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == input_text
        assert "context" in thought.metadata
        assert "reasoning_type" in thought.metadata
        assert thought.metadata["reasoning_type"] == "decomposed_prompting"
        assert "stage" in thought.metadata
        assert thought.metadata["stage"] == "expertise_identification"
        assert "specialists_identified" in thought.metadata

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test execute accepts and stores context."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()
        context = {"priority": "high", "deadline": "2024-12-31"}

        thought = await method.execute(session, "Test problem", context=context)

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self):
        """Test execute adds thought to session."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        initial_count = session.thought_count
        await method.execute(session, "Test problem")

        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_sets_current_method_on_session(self):
        """Test execute sets current_method on session."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        assert session.current_method == MethodIdentifier.DECOMPOSED_PROMPTING

    @pytest.mark.asyncio
    async def test_execute_resets_state_on_new_execution(self):
        """Test execute resets state for new execution."""
        method = DecomposedPrompting()
        await method.initialize()
        session1 = Session().start()
        session2 = Session().start()

        # First execution
        await method.execute(session1, "Problem 1")
        len(method._specialists)

        # Modify state
        method._specialist_outputs = {"test": "output"}
        method._decomposition_complete = True

        # Second execution should reset
        await method.execute(session2, "Problem 2")

        assert method._step_counter == 1
        assert len(method._specialists) > 0
        assert method._specialist_outputs == {}
        assert method._decomposition_complete is False


class TestDecomposedPromptingContinueReasoning:
    """Tests for DecomposedPrompting continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(self):
        """Test continue_reasoning fails without initialization."""
        method = DecomposedPrompting()
        session = Session().start()

        # Create a mock previous thought
        from reasoning_mcp.models.thought import ThoughtNode

        previous_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DECOMPOSED_PROMPTING,
            content="Test",
            metadata={"stage": "expertise_identification"},
        )

        with pytest.raises(
            RuntimeError,
            match="Decomposed Prompting method must be initialized before continuation",
        ):
            await method.continue_reasoning(session, previous_thought)

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_counter(self):
        """Test continue_reasoning increments step counter."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        initial_step = method._step_counter

        await method.continue_reasoning(session, first_thought)

        assert method._step_counter == initial_step + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_after_expertise_identification(self):
        """Test continue_reasoning creates specialist assignment."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        second_thought = await method.continue_reasoning(session, first_thought)

        assert "Specialist Assignment" in second_thought.content
        assert "Sub-Task Creation" in second_thought.content
        assert method._specialists_assigned is True

    @pytest.mark.asyncio
    async def test_continue_reasoning_specialist_execution_phase(self):
        """Test continue_reasoning executes specialist tasks."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Initial thought
        first_thought = await method.execute(session, "Test problem")

        # Specialist assignment
        second_thought = await method.continue_reasoning(session, first_thought)

        # First specialist execution
        third_thought = await method.continue_reasoning(session, second_thought)

        assert "Analysis" in third_thought.content
        assert len(method._specialist_outputs) == 1
        assert third_thought.metadata["stage"] == "specialist_execution"

    @pytest.mark.asyncio
    async def test_continue_reasoning_multiple_specialists(self):
        """Test continue_reasoning handles multiple specialists."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Initial thought
        first_thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        # Assignment
        current_thought = await method.continue_reasoning(session, first_thought)

        # Execute each specialist
        for i in range(num_specialists):
            current_thought = await method.continue_reasoning(session, current_thought)
            assert len(method._specialist_outputs) == i + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_uses_branch_type_for_specialists(self):
        """Test continue_reasoning uses BRANCH type for specialist execution."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        second_thought = await method.continue_reasoning(session, first_thought)
        third_thought = await method.continue_reasoning(session, second_thought)

        # Specialist execution should use BRANCH type
        assert third_thought.type == ThoughtType.BRANCH

    @pytest.mark.asyncio
    async def test_continue_reasoning_integration_phase(self):
        """Test continue_reasoning creates integration after all specialists."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Initial
        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        # Assignment
        thought = await method.continue_reasoning(session, thought)

        # All specialist executions
        for _ in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)

        # Integration
        integration_thought = await method.continue_reasoning(session, thought)

        assert "Integration" in integration_thought.content
        assert "Synthesis" in integration_thought.content
        assert integration_thought.metadata["stage"] == "integration"

    @pytest.mark.asyncio
    async def test_continue_reasoning_uses_synthesis_type_for_integration(self):
        """Test continue_reasoning uses SYNTHESIS type for integration."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Execute through all phases
        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        thought = await method.continue_reasoning(session, thought)
        for _ in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)

        integration_thought = await method.continue_reasoning(session, thought)

        assert integration_thought.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_continue_reasoning_confidence_increases_with_progress(self):
        """Test continue_reasoning increases confidence as specialists complete."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        thought = await method.continue_reasoning(session, thought)
        initial_confidence = thought.confidence

        # Execute specialists and track confidence
        for i in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)
            if i < num_specialists - 1:
                # Earlier specialists should have lower confidence
                assert thought.confidence >= initial_confidence

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self):
        """Test continue_reasoning accepts guidance parameter."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        second_thought = await method.continue_reasoning(
            session, first_thought, guidance="Focus on technical aspects"
        )

        assert "guidance" in second_thought.metadata
        assert second_thought.metadata["guidance"] == "Focus on technical aspects"

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self):
        """Test continue_reasoning accepts context parameter."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        context = {"additional": "info"}
        second_thought = await method.continue_reasoning(session, first_thought, context=context)

        assert "context" in second_thought.metadata
        assert second_thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(self):
        """Test continue_reasoning sets parent_id correctly."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        second_thought = await method.continue_reasoning(session, first_thought)

        assert second_thought.parent_id == first_thought.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_depth(self):
        """Test continue_reasoning increases depth."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        first_thought = await method.execute(session, "Test problem")
        second_thought = await method.continue_reasoning(session, first_thought)

        assert second_thought.depth == first_thought.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_tracks_progress_in_metadata(self):
        """Test continue_reasoning tracks specialist progress in metadata."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        thought = await method.continue_reasoning(session, thought)

        for i in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)
            assert "specialists_total" in thought.metadata
            assert "specialists_completed" in thought.metadata
            assert "progress" in thought.metadata
            assert thought.metadata["specialists_total"] == num_specialists
            assert thought.metadata["specialists_completed"] == i + 1


class TestDecomposedPromptingSpecialistRouting:
    """Tests for specialist routing and assignment."""

    @pytest.mark.asyncio
    async def test_specialists_have_defined_roles(self):
        """Test that specialists have defined roles."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        for specialist in method._specialists:
            assert "role" in specialist
            assert isinstance(specialist["role"], str)
            assert len(specialist["role"]) > 0

    @pytest.mark.asyncio
    async def test_specialists_have_domains(self):
        """Test that specialists have defined domains."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        for specialist in method._specialists:
            assert "domain" in specialist
            assert isinstance(specialist["domain"], str)
            assert len(specialist["domain"]) > 0

    @pytest.mark.asyncio
    async def test_specialists_have_focus_areas(self):
        """Test that specialists have defined focus areas."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        for specialist in method._specialists:
            assert "focus" in specialist
            assert isinstance(specialist["focus"], str)
            assert len(specialist["focus"]) > 0

    @pytest.mark.asyncio
    async def test_default_specialists_are_multi_disciplinary(self):
        """Test that default specialists cover multiple disciplines."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        # Should have at least 3 specialists for comprehensive coverage
        assert len(method._specialists) >= 3

        # Check for typical specialist types
        roles = [s["role"] for s in method._specialists]
        assert "Technical Architect" in roles
        assert "Domain Expert" in roles
        assert "Implementation Specialist" in roles

    @pytest.mark.asyncio
    async def test_specialist_outputs_stored_by_role(self):
        """Test that specialist outputs are keyed by role."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")
        thought = await method.continue_reasoning(session, thought)
        thought = await method.continue_reasoning(session, thought)

        # Should have one specialist output
        assert len(method._specialist_outputs) == 1

        # Key should be a specialist role
        role = list(method._specialist_outputs.keys())[0]
        assert role in [s["role"] for s in method._specialists]


class TestDecomposedPromptingResultAggregation:
    """Tests for result aggregation and integration."""

    @pytest.mark.asyncio
    async def test_integration_combines_all_specialist_outputs(self):
        """Test integration phase references all specialist outputs."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Execute through all phases
        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        thought = await method.continue_reasoning(session, thought)
        for _ in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)

        integration = await method.continue_reasoning(session, thought)

        # Integration should reference all specialists
        assert len(method._specialist_outputs) == num_specialists
        assert "comprehensive" in integration.content.lower()
        assert "unified" in integration.content.lower()

    @pytest.mark.asyncio
    async def test_integration_produces_final_recommendation(self):
        """Test integration produces a final recommendation."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")
        num_specialists = len(method._specialists)

        thought = await method.continue_reasoning(session, thought)
        for _ in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)

        integration = await method.continue_reasoning(session, thought)

        assert (
            "Final Recommendation" in integration.content or "Recommendation" in integration.content
        )


class TestDecomposedPromptingEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_simple_task_still_decomposes(self):
        """Test that even simple tasks get decomposed."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Add two numbers")

        # Even simple tasks should identify specialists
        assert len(method._specialists) > 0

    @pytest.mark.asyncio
    async def test_complex_multi_specialist_task(self):
        """Test complex task with multiple specialist iterations."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        complex_problem = (
            "Design a comprehensive system for autonomous vehicles including "
            "hardware, software, safety, legal compliance, and user experience"
        )

        thought = await method.execute(session, complex_problem)
        num_specialists = len(method._specialists)

        # Should identify multiple specialists
        assert num_specialists >= 3

        # Complete full reasoning cycle
        thought = await method.continue_reasoning(session, thought)
        for i in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)
            assert thought.metadata["specialists_completed"] == i + 1

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test handling of empty input text."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Should handle empty input gracefully
        thought = await method.execute(session, "")

        assert thought is not None
        assert len(thought.content) > 0

    @pytest.mark.asyncio
    async def test_very_long_input_text(self):
        """Test handling of very long input text."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        long_input = "Test problem " * 1000
        thought = await method.execute(session, long_input)

        assert thought is not None
        assert long_input in thought.content

    @pytest.mark.asyncio
    async def test_special_characters_in_input(self):
        """Test handling of special characters in input."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        special_input = "Test with special chars: !@#$%^&*(){}[]<>?/\\|~`"
        thought = await method.execute(session, special_input)

        assert thought is not None
        assert special_input in thought.content

    @pytest.mark.asyncio
    async def test_unicode_input(self):
        """Test handling of unicode input."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        unicode_input = "Test with unicode: 你好世界 🌍 Ω α β γ"
        thought = await method.execute(session, unicode_input)

        assert thought is not None
        assert unicode_input in thought.content

    @pytest.mark.asyncio
    async def test_none_context_handled_gracefully(self):
        """Test that None context is handled as empty dict."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test", context=None)

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_multiple_executions_in_sequence(self):
        """Test multiple executions with same method instance."""
        method = DecomposedPrompting()
        await method.initialize()

        session1 = Session().start()
        thought1 = await method.execute(session1, "Problem 1")
        assert thought1.step_number == 1

        session2 = Session().start()
        thought2 = await method.execute(session2, "Problem 2")
        assert thought2.step_number == 1  # Should reset for new execution


class TestDecomposedPromptingIntegration:
    """Integration tests for complete reasoning flows."""

    @pytest.mark.asyncio
    async def test_complete_reasoning_flow(self):
        """Test a complete reasoning flow from start to integration."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Execute initial thought
        thought = await method.execute(session, "Design a multi-platform mobile application")
        assert thought.type == ThoughtType.INITIAL
        assert session.thought_count == 1

        # Specialist assignment
        thought = await method.continue_reasoning(session, thought)
        assert method._specialists_assigned is True
        assert session.thought_count == 2

        # Execute all specialists
        num_specialists = len(method._specialists)
        for i in range(num_specialists):
            thought = await method.continue_reasoning(session, thought)
            assert len(method._specialist_outputs) == i + 1
            assert thought.type == ThoughtType.BRANCH

        # Integration
        thought = await method.continue_reasoning(session, thought)
        assert thought.type == ThoughtType.SYNTHESIS
        assert "Integration" in thought.content

        # Verify session state
        assert session.thought_count == 2 + num_specialists + 1
        assert session.current_method == MethodIdentifier.DECOMPOSED_PROMPTING

    @pytest.mark.asyncio
    async def test_session_metrics_updated_correctly(self):
        """Test that session metrics are updated throughout reasoning."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        initial_metrics = session.metrics.total_thoughts

        # Execute and continue
        thought = await method.execute(session, "Test problem")
        assert session.metrics.total_thoughts == initial_metrics + 1

        thought = await method.continue_reasoning(session, thought)
        assert session.metrics.total_thoughts == initial_metrics + 2

        # Verify metrics consistency
        assert session.metrics.total_thoughts == session.thought_count

    @pytest.mark.asyncio
    async def test_graph_structure_maintained(self):
        """Test that thought graph structure is maintained correctly."""
        method = DecomposedPrompting()
        await method.initialize()
        session = Session().start()

        # Build reasoning chain
        thought1 = await method.execute(session, "Test problem")
        thought2 = await method.continue_reasoning(session, thought1)
        thought3 = await method.continue_reasoning(session, thought2)

        # Verify graph relationships
        assert thought2.parent_id == thought1.id
        assert thought3.parent_id == thought2.id

        # Verify thoughts are in graph
        assert thought1.id in session.graph.nodes
        assert thought2.id in session.graph.nodes
        assert thought3.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_configuration_options_respected(self):
        """Test that session configuration options are respected."""
        # Create session with custom config
        config = SessionConfig(
            max_depth=5,
            enable_branching=True,
            max_branches=10,
        )
        session = Session(config=config).start()

        method = DecomposedPrompting()
        await method.initialize()

        # Execute reasoning
        await method.execute(session, "Test problem")

        # Verify config is accessible
        assert session.config.max_depth == 5
        assert session.config.enable_branching is True
        assert session.config.max_branches == 10
