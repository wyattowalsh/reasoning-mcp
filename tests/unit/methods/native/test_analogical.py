"""Unit tests for Analogical Reasoning method."""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.analogical import ANALOGICAL_METADATA, Analogical
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestAnalogicalMetadata:
    """Tests for Analogical method metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert ANALOGICAL_METADATA.identifier == MethodIdentifier.ANALOGICAL

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert ANALOGICAL_METADATA.name == "Analogical Reasoning"

    def test_metadata_description(self):
        """Test that metadata has description."""
        assert "analog" in ANALOGICAL_METADATA.description.lower()  # analogous, analogy, etc.
        assert "mapping" in ANALOGICAL_METADATA.description.lower()

    def test_metadata_category(self):
        """Test that Analogical is in SPECIALIZED category."""
        assert ANALOGICAL_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that complexity is set appropriately."""
        assert ANALOGICAL_METADATA.complexity == 4
        assert 1 <= ANALOGICAL_METADATA.complexity <= 10

    def test_metadata_supports_branching(self):
        """Test that Analogical supports branching."""
        assert ANALOGICAL_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that Analogical supports revision."""
        assert ANALOGICAL_METADATA.supports_revision is True

    def test_metadata_tags(self):
        """Test that metadata has appropriate tags."""
        assert "analogical" in ANALOGICAL_METADATA.tags
        assert "analogy" in ANALOGICAL_METADATA.tags
        assert "mapping" in ANALOGICAL_METADATA.tags
        assert isinstance(ANALOGICAL_METADATA.tags, frozenset)

    def test_metadata_min_thoughts(self):
        """Test minimum thoughts requirement."""
        assert ANALOGICAL_METADATA.min_thoughts == 5

    def test_metadata_best_for(self):
        """Test that best_for includes appropriate use cases."""
        assert len(ANALOGICAL_METADATA.best_for) > 0
        assert any("creative" in use_case.lower() for use_case in ANALOGICAL_METADATA.best_for)

    def test_metadata_not_recommended_for(self):
        """Test that not_recommended_for is specified."""
        assert len(ANALOGICAL_METADATA.not_recommended_for) > 0


class TestAnalogicalInitialization:
    """Tests for Analogical method initialization."""

    @pytest.mark.asyncio
    async def test_init_creates_instance(self):
        """Test that __init__ creates an Analogical instance."""
        method = Analogical()
        assert isinstance(method, Analogical)
        assert method._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self):
        """Test that initialize() sets the initialized flag."""
        method = Analogical()
        await method.initialize()
        assert method._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_resets_step_counter(self):
        """Test that initialize() resets step counter."""
        method = Analogical()
        method._step_counter = 10
        await method.initialize()
        assert method._step_counter == 0

    @pytest.mark.asyncio
    async def test_initialize_sets_initial_stage(self):
        """Test that initialize() sets stage to target_analysis."""
        method = Analogical()
        await method.initialize()
        assert method._current_stage == "target_analysis"

    @pytest.mark.asyncio
    async def test_health_check_before_init(self):
        """Test health_check returns False before initialization."""
        method = Analogical()
        health = await method.health_check()
        assert health is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self):
        """Test health_check returns True after initialization."""
        method = Analogical()
        await method.initialize()
        health = await method.health_check()
        assert health is True

    def test_identifier_property(self):
        """Test that identifier property returns correct value."""
        method = Analogical()
        assert method.identifier == MethodIdentifier.ANALOGICAL

    def test_name_property(self):
        """Test that name property returns correct value."""
        method = Analogical()
        assert method.name == "Analogical Reasoning"

    def test_description_property(self):
        """Test that description property returns correct value."""
        method = Analogical()
        assert method.description == ANALOGICAL_METADATA.description

    def test_category_property(self):
        """Test that category property returns correct value."""
        method = Analogical()
        assert method.category == MethodCategory.SPECIALIZED


class TestAnalogicalExecution:
    """Tests for Analogical method execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(self):
        """Test that execute raises RuntimeError if not initialized."""
        method = Analogical()
        session = Session().start()

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, "Test problem")

    @pytest.mark.asyncio
    async def test_execute_returns_thought_node(self):
        """Test that execute returns a ThoughtNode."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "How to improve team collaboration?")

        assert isinstance(thought, ThoughtNode)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self):
        """Test that execute creates an INITIAL thought type."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert thought.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_execute_sets_method_id(self):
        """Test that execute sets correct method_id."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert thought.method_id == MethodIdentifier.ANALOGICAL

    @pytest.mark.asyncio
    async def test_execute_sets_step_number(self):
        """Test that execute sets step_number to 1."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert thought.step_number == 1

    @pytest.mark.asyncio
    async def test_execute_sets_depth_to_zero(self):
        """Test that execute sets depth to 0."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_confidence(self):
        """Test that execute sets initial confidence."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.6

    @pytest.mark.asyncio
    async def test_execute_stores_input_in_metadata(self):
        """Test that execute stores input text in metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()
        input_text = "How to design a new product?"

        thought = await method.execute(session, input_text)

        assert thought.metadata["input"] == input_text
        assert thought.metadata["target_problem"] == input_text

    @pytest.mark.asyncio
    async def test_execute_stores_stage_in_metadata(self):
        """Test that execute stores current stage in metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert thought.metadata["stage"] == "target_analysis"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test that execute accepts and stores context."""
        method = Analogical()
        await method.initialize()
        session = Session().start()
        context = {"preferred_domain": "nature", "similarity_threshold": 0.7}

        thought = await method.execute(session, "Test problem", context=context)

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self):
        """Test that execute adds thought to session."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test problem")

        assert session.thought_count == 1
        assert session.graph.get_node(thought.id) is not None

    @pytest.mark.asyncio
    async def test_execute_sets_current_method(self):
        """Test that execute sets current_method on session."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        await method.execute(session, "Test problem")

        assert session.current_method == MethodIdentifier.ANALOGICAL

    @pytest.mark.asyncio
    async def test_execute_generates_target_analysis_content(self):
        """Test that execute generates target analysis content."""
        method = Analogical()
        await method.initialize()
        session = Session().start()
        input_text = "How to reduce customer churn?"

        thought = await method.execute(session, input_text)

        assert "Target Problem" in thought.content
        assert input_text in thought.content
        assert "Step 1" in thought.content


class TestAnalogicalContinueReasoning:
    """Tests for Analogical continue_reasoning."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(self):
        """Test that continue_reasoning raises RuntimeError if not initialized."""
        method = Analogical()
        session = Session().start()

        # Create a mock previous thought
        previous = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.ANALOGICAL,
            content="Previous thought",
            metadata={"stage": "target_analysis"}
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, previous)

    @pytest.mark.asyncio
    async def test_continue_reasoning_returns_thought_node(self):
        """Test that continue_reasoning returns a ThoughtNode."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert isinstance(second, ThoughtNode)

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_counter(self):
        """Test that continue_reasoning increments step counter."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert second.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(self):
        """Test that continue_reasoning sets parent_id."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_depth(self):
        """Test that continue_reasoning increments depth."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_to_source_identification(self):
        """Test that continue_reasoning progresses from target_analysis to source_identification."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert second.metadata["stage"] == "source_identification"
        assert second.type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_to_structural_mapping(self):
        """Test that continue_reasoning progresses to structural_mapping."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)
        third = await method.continue_reasoning(session, second)

        assert third.metadata["stage"] == "structural_mapping"
        assert third.type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_to_insight_transfer(self):
        """Test that continue_reasoning progresses to insight_transfer."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        # Stage progression: target_analysis → source_identification → structural_mapping → insight_transfer
        # Need 3 continues to reach insight_transfer from initial execute
        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(3):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        assert thoughts[-1].metadata["stage"] == "insight_transfer"
        assert thoughts[-1].type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_to_validation(self):
        """Test that continue_reasoning progresses to validation."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        # Stage progression: target_analysis → source_identification → structural_mapping → insight_transfer → validation
        # Need 4 continues to reach validation from initial execute
        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        assert thoughts[-1].metadata["stage"] == "validation"
        assert thoughts[-1].type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self):
        """Test that continue_reasoning accepts guidance parameter."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        guidance = "Consider natural systems as source domain"
        second = await method.continue_reasoning(session, first, guidance=guidance)

        assert second.metadata["guidance"] == guidance

    @pytest.mark.asyncio
    async def test_continue_reasoning_branching_with_different_keyword(self):
        """Test that 'different' keyword triggers branching."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)
        branch = await method.continue_reasoning(
            session, second, guidance="Try a different source domain"
        )

        assert branch.type == ThoughtType.BRANCH
        assert branch.metadata["stage"] == "source_identification"

    @pytest.mark.asyncio
    async def test_continue_reasoning_branching_with_alternative_keyword(self):
        """Test that 'alternative' keyword triggers branching."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)
        branch = await method.continue_reasoning(
            session, second, guidance="Find an alternative analogy"
        )

        assert branch.type == ThoughtType.BRANCH

    @pytest.mark.asyncio
    async def test_continue_reasoning_branching_with_another_keyword(self):
        """Test that 'another' keyword triggers branching."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        branch = await method.continue_reasoning(
            session, first, guidance="another source domain"
        )

        assert branch.type == ThoughtType.BRANCH

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_confidence(self):
        """Test that confidence increases as reasoning progresses."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        # Confidence should generally increase
        assert thoughts[-1].confidence > thoughts[0].confidence

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_source_domain_metadata(self):
        """Test that source_identification stage adds source_domain metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        assert "source_domain" in second.metadata
        assert second.metadata["stage"] == "source_identification"

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_mappings_metadata(self):
        """Test that structural_mapping stage adds mappings metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(2):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        assert "mappings" in thoughts[-1].metadata
        assert thoughts[-1].metadata["stage"] == "structural_mapping"

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_transferred_insights_metadata(self):
        """Test that insight_transfer stage adds transferred_insights metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(3):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        assert "transferred_insights" in thoughts[-1].metadata
        assert thoughts[-1].metadata["stage"] == "insight_transfer"

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_validation_criteria_metadata(self):
        """Test that validation stage adds validation_criteria metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        assert "validation_criteria" in thoughts[-1].metadata
        assert thoughts[-1].metadata["stage"] == "validation"

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self):
        """Test that continue_reasoning accepts and stores context."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        context = {"analogy_depth": 3, "similarity_threshold": 0.8}
        second = await method.continue_reasoning(session, first, context=context)

        assert second.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_thought_to_session(self):
        """Test that continue_reasoning adds thought to session."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        initial_count = session.thought_count
        second = await method.continue_reasoning(session, first)

        assert session.thought_count == initial_count + 1
        assert session.graph.get_node(second.id) is not None


class TestAnalogicalStageProgression:
    """Tests for stage progression logic in Analogical method."""

    @pytest.mark.asyncio
    async def test_full_stage_progression_sequence(self):
        """Test complete progression through all stages."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        stages = []
        thoughts = [await method.execute(session, "Test problem")]
        stages.append(thoughts[0].metadata["stage"])

        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))
            stages.append(thoughts[-1].metadata["stage"])

        expected_stages = [
            "target_analysis",
            "source_identification",
            "structural_mapping",
            "insight_transfer",
            "validation",
        ]
        assert stages == expected_stages

    @pytest.mark.asyncio
    async def test_validation_stage_can_conclude(self):
        """Test that validation stage can create CONCLUSION thoughts."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        # Continue from validation stage
        conclusion = await method.continue_reasoning(session, thoughts[-1])

        assert conclusion.metadata["stage"] == "validation"
        assert conclusion.type == ThoughtType.CONCLUSION


class TestAnalogicalEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_execute_with_empty_input(self):
        """Test execute with empty string input."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "")

        assert isinstance(thought, ThoughtNode)
        assert thought.metadata["input"] == ""

    @pytest.mark.asyncio
    async def test_execute_with_very_long_input(self):
        """Test execute with very long input text."""
        method = Analogical()
        await method.initialize()
        session = Session().start()
        long_input = "How to solve this problem? " * 100

        thought = await method.execute(session, long_input)

        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_execute_with_none_context(self):
        """Test execute with explicit None context."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session, "Test", context=None)

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_none_guidance(self):
        """Test continue_reasoning with explicit None guidance."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test")
        second = await method.continue_reasoning(session, first, guidance=None)

        assert second.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_multiple_branches_from_same_thought(self):
        """Test creating multiple branches from the same thought."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session, "Test problem")
        second = await method.continue_reasoning(session, first)

        branch1 = await method.continue_reasoning(
            session, second, guidance="different approach"
        )
        branch2 = await method.continue_reasoning(
            session, second, guidance="alternative method"
        )

        assert branch1.type == ThoughtType.BRANCH
        assert branch2.type == ThoughtType.BRANCH
        assert branch1.parent_id == second.id
        assert branch2.parent_id == second.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_multiple_times_from_validation(self):
        """Test that continuing from validation maintains validation stage."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        # Continue multiple times from validation
        validation1 = await method.continue_reasoning(session, thoughts[-1])
        validation2 = await method.continue_reasoning(session, validation1)

        assert validation1.metadata["stage"] == "validation"
        assert validation2.metadata["stage"] == "validation"

    @pytest.mark.asyncio
    async def test_confidence_calculation_blending(self):
        """Test that confidence blends with previous confidence."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]

        # Track confidence progression
        confidences = [thoughts[0].confidence]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))
            confidences.append(thoughts[-1].confidence)

        # Each confidence should be between 0 and 1
        assert all(0.0 <= c <= 1.0 for c in confidences)
        # Later stages should generally have higher confidence
        assert confidences[-1] > confidences[0]

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self):
        """Test that session metrics are updated correctly."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thought1 = await method.execute(session, "Test problem")
        thought2 = await method.continue_reasoning(session, thought1)

        assert session.metrics.total_thoughts == 2
        assert session.metrics.methods_used[MethodIdentifier.ANALOGICAL] == 2
        assert session.metrics.thought_types[ThoughtType.INITIAL] == 1
        assert session.metrics.thought_types[ThoughtType.CONTINUATION] == 1

    @pytest.mark.asyncio
    async def test_reinitialization_resets_state(self):
        """Test that reinitializing resets internal state."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        # Execute and progress
        thought1 = await method.execute(session, "First problem")
        await method.continue_reasoning(session, thought1)

        # Reinitialize
        await method.initialize()

        # Should reset to initial state
        assert method._step_counter == 0
        assert method._current_stage == "target_analysis"

        # Execute again
        session2 = Session().start()
        thought2 = await method.execute(session2, "Second problem")
        assert thought2.step_number == 1
        assert thought2.metadata["stage"] == "target_analysis"

    @pytest.mark.asyncio
    async def test_content_generation_for_all_stages(self):
        """Test that all stages generate appropriate content."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(4):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        # Check content for each stage
        assert "Target Problem" in thoughts[0].content
        assert "Source Domain" in thoughts[1].content
        assert "Structural Mapping" in thoughts[2].content
        assert "Insight Transfer" in thoughts[3].content
        assert "Validation" in thoughts[4].content

        # All should have step numbers
        for i, thought in enumerate(thoughts):
            assert f"Step {i + 1}" in thought.content

    @pytest.mark.asyncio
    async def test_reasoning_type_metadata(self):
        """Test that all thoughts have reasoning_type metadata."""
        method = Analogical()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session, "Test problem")]
        for _ in range(3):
            thoughts.append(await method.continue_reasoning(session, thoughts[-1]))

        for thought in thoughts:
            assert thought.metadata.get("reasoning_type") == "analogical"
