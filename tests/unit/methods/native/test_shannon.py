"""Unit tests for ShannonThinking reasoning method.

This test suite provides comprehensive coverage of the ShannonThinking method,
testing all five phases (Problem Definition, Constraints, Model, Proof, Implementation),
phase transitions, configuration options, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.shannon import (
    SHANNON_THINKING_METADATA,
    ShannonPhase,
    ShannonThinking,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import Session


class TestShannonThinkingMetadata:
    """Tests for Shannon Thinking metadata configuration."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SHANNON_THINKING_METADATA.identifier == MethodIdentifier.SHANNON_THINKING

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SHANNON_THINKING_METADATA.name == "Shannon Thinking"

    def test_metadata_description(self):
        """Test that metadata includes description of 5-phase approach."""
        assert "Problem Definition" in SHANNON_THINKING_METADATA.description
        assert "Constraints" in SHANNON_THINKING_METADATA.description
        assert "Model" in SHANNON_THINKING_METADATA.description
        assert "Proof" in SHANNON_THINKING_METADATA.description
        assert "Implementation" in SHANNON_THINKING_METADATA.description

    def test_metadata_category(self):
        """Test that Shannon Thinking is in HIGH_VALUE category."""
        assert SHANNON_THINKING_METADATA.category == MethodCategory.HIGH_VALUE

    def test_metadata_tags(self):
        """Test that metadata includes appropriate tags."""
        expected_tags = {
            "shannon",
            "information-theory",
            "engineering",
            "systematic",
            "mathematical",
        }
        assert expected_tags.issubset(SHANNON_THINKING_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that complexity reflects rigorous engineering approach."""
        assert SHANNON_THINKING_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test that method supports branching."""
        assert SHANNON_THINKING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that method supports revision."""
        assert SHANNON_THINKING_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self):
        """Test that minimum thoughts matches 5 phases."""
        assert SHANNON_THINKING_METADATA.min_thoughts == 5

    def test_metadata_best_for(self):
        """Test that best_for includes technical and engineering problems."""
        assert "technical problems" in SHANNON_THINKING_METADATA.best_for
        assert "engineering challenges" in SHANNON_THINKING_METADATA.best_for
        assert "mathematical modeling" in SHANNON_THINKING_METADATA.best_for


class TestShannonPhase:
    """Tests for Shannon phase enumeration."""

    def test_phase_problem_definition(self):
        """Test problem definition phase identifier."""
        assert ShannonPhase.PROBLEM_DEFINITION == "problem_definition"

    def test_phase_constraints(self):
        """Test constraints phase identifier."""
        assert ShannonPhase.CONSTRAINTS == "constraints"

    def test_phase_model(self):
        """Test model phase identifier."""
        assert ShannonPhase.MODEL == "model"

    def test_phase_proof(self):
        """Test proof phase identifier."""
        assert ShannonPhase.PROOF == "proof"

    def test_phase_implementation(self):
        """Test implementation phase identifier."""
        assert ShannonPhase.IMPLEMENTATION == "implementation"


class TestShannonThinkingInitialization:
    """Tests for initialization and health checks."""

    def test_constructor_creates_uninitialized_instance(self):
        """Test that constructor creates uninitialized method."""
        method = ShannonThinking()
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == ShannonPhase.PROBLEM_DEFINITION

    def test_identifier_property(self):
        """Test that identifier property returns correct value."""
        method = ShannonThinking()
        assert method.identifier == MethodIdentifier.SHANNON_THINKING

    def test_name_property(self):
        """Test that name property returns correct value."""
        method = ShannonThinking()
        assert method.name == "Shannon Thinking"

    def test_description_property(self):
        """Test that description property returns metadata description."""
        method = ShannonThinking()
        assert method.description == SHANNON_THINKING_METADATA.description

    def test_category_property(self):
        """Test that category property returns HIGH_VALUE."""
        method = ShannonThinking()
        assert method.category == MethodCategory.HIGH_VALUE

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self):
        """Test that initialize sets the initialized flag."""
        method = ShannonThinking()
        await method.initialize()
        assert method._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets all state variables."""
        method = ShannonThinking()
        # Manually set some state
        method._step_counter = 10
        method._current_phase = ShannonPhase.IMPLEMENTATION
        method._phase_history = [ShannonPhase.CONSTRAINTS, ShannonPhase.MODEL]

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == ShannonPhase.PROBLEM_DEFINITION
        assert method._phase_history == []

    @pytest.mark.asyncio
    async def test_health_check_false_before_initialization(self):
        """Test that health check returns False before initialization."""
        method = ShannonThinking()
        assert await method.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_true_after_initialization(self):
        """Test that health check returns True after initialization."""
        method = ShannonThinking()
        await method.initialize()
        assert await method.health_check() is True


class TestShannonThinkingExecution:
    """Tests for execute method and initial thought generation."""

    @pytest.mark.asyncio
    async def test_execute_requires_initialization(self):
        """Test that execute raises error if not initialized."""
        method = ShannonThinking()
        session = Session().start()

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(
                session=session,
                input_text="Test problem"
            )

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self):
        """Test that execute creates an INITIAL thought."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Design a data compression algorithm"
        )

        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SHANNON_THINKING

    @pytest.mark.asyncio
    async def test_execute_starts_with_problem_definition_phase(self):
        """Test that execute starts with problem definition phase."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Optimize signal transmission"
        )

        assert thought.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION
        assert thought.metadata["phase_number"] == 1
        assert thought.metadata["total_phases"] == 5

    @pytest.mark.asyncio
    async def test_execute_sets_step_counter(self):
        """Test that execute sets step counter to 1."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert thought.step_number == 1
        assert method._step_counter == 1

    @pytest.mark.asyncio
    async def test_execute_initializes_phase_history(self):
        """Test that execute initializes phase history."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert method._phase_history == [ShannonPhase.PROBLEM_DEFINITION]

    @pytest.mark.asyncio
    async def test_execute_sets_initial_confidence(self):
        """Test that execute sets appropriate initial confidence."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert thought.confidence == 0.6

    @pytest.mark.asyncio
    async def test_execute_includes_input_in_metadata(self):
        """Test that execute includes input text in metadata."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()
        input_text = "Design error correction code"

        thought = await method.execute(
            session=session,
            input_text=input_text
        )

        assert thought.metadata["input"] == input_text

    @pytest.mark.asyncio
    async def test_execute_includes_context_in_metadata(self):
        """Test that execute includes context in metadata."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()
        context = {"domain": "communication systems", "priority": "high"}

        thought = await method.execute(
            session=session,
            input_text="Test problem",
            context=context
        )

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self):
        """Test that execute adds thought to session."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert session.thought_count == 1

    @pytest.mark.asyncio
    async def test_execute_sets_current_method_on_session(self):
        """Test that execute updates session's current method."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert session.current_method == MethodIdentifier.SHANNON_THINKING

    @pytest.mark.asyncio
    async def test_execute_content_mentions_problem_definition(self):
        """Test that execute generates content mentioning problem definition."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Test problem"
        )

        assert "Problem Definition" in thought.content
        assert "Shannon" in thought.content


class TestShannonThinkingContinueReasoning:
    """Tests for continue_reasoning method and phase transitions."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_requires_initialization(self):
        """Test that continue_reasoning raises error if not initialized."""
        method = ShannonThinking()
        session = Session().start()

        # Create a mock previous thought
        from reasoning_mcp.models.thought import ThoughtNode
        previous = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SHANNON_THINKING,
            content="Test",
            metadata={"phase": ShannonPhase.PROBLEM_DEFINITION}
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(
                session=session,
                previous_thought=previous
            )

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_counter(self):
        """Test that continue_reasoning increments step counter."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        await method.continue_reasoning(session=session, previous_thought=first)

        assert method._step_counter == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_advances_to_constraints_phase(self):
        """Test that continue_reasoning advances from problem definition to constraints."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test problem")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first
        )

        assert second.metadata["phase"] == ShannonPhase.CONSTRAINTS
        assert second.metadata["phase_number"] == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_through_all_phases(self):
        """Test that continue_reasoning progresses through all 5 phases in order."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thoughts = []
        thoughts.append(await method.execute(session=session, input_text="Test"))

        for _ in range(4):  # Advance through 4 more phases
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        expected_phases = [
            ShannonPhase.PROBLEM_DEFINITION,
            ShannonPhase.CONSTRAINTS,
            ShannonPhase.MODEL,
            ShannonPhase.PROOF,
            ShannonPhase.IMPLEMENTATION,
        ]

        for i, thought in enumerate(thoughts):
            assert thought.metadata["phase"] == expected_phases[i]

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(self):
        """Test that continue_reasoning sets parent_id correctly."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first
        )

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_depth(self):
        """Test that continue_reasoning increases depth by 1."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first
        )

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_updates_phase_history(self):
        """Test that continue_reasoning updates phase history."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first
        )

        assert ShannonPhase.CONSTRAINTS in method._phase_history
        assert second.metadata["phase_history"] == [
            ShannonPhase.PROBLEM_DEFINITION,
            ShannonPhase.CONSTRAINTS,
        ]

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance_to_specific_phase(self):
        """Test that guidance can specify a specific phase."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        # Skip directly to model phase with guidance
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="Jump to model phase"
        )

        assert second.metadata["phase"] == ShannonPhase.MODEL

    @pytest.mark.asyncio
    async def test_continue_reasoning_guidance_refine_stays_in_phase(self):
        """Test that 'refine' guidance stays in current phase."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="refine the problem definition"
        )

        assert second.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_thought_to_session(self):
        """Test that continue_reasoning adds thought to session."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        await method.continue_reasoning(session=session, previous_thought=first)

        assert session.thought_count == 2


class TestShannonThinkingPhaseContent:
    """Tests for phase-specific content generation."""

    @pytest.mark.asyncio
    async def test_problem_definition_content(self):
        """Test that problem definition phase includes appropriate content."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session,
            input_text="Design compression algorithm"
        )

        content = thought.content.lower()
        assert "problem" in content
        assert "definition" in content
        assert "shannon" in content

    @pytest.mark.asyncio
    async def test_constraints_phase_content(self):
        """Test that constraints phase mentions constraint identification."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first
        )

        content = second.content.lower()
        assert "constraint" in content
        assert "phase 2" in content

    @pytest.mark.asyncio
    async def test_model_phase_content(self):
        """Test that model phase mentions mathematical modeling."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to model phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(2):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        model_thought = thoughts[-1]
        content = model_thought.content.lower()
        assert "model" in content
        assert "phase 3" in content
        assert "mathematical" in content or "theoretical" in content

    @pytest.mark.asyncio
    async def test_proof_phase_content(self):
        """Test that proof phase mentions validation."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to proof phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(3):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        proof_thought = thoughts[-1]
        content = proof_thought.content.lower()
        assert "proof" in content or "validation" in content
        assert "phase 4" in content

    @pytest.mark.asyncio
    async def test_implementation_phase_content(self):
        """Test that implementation phase mentions practical design."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to implementation phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(4):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        impl_thought = thoughts[-1]
        content = impl_thought.content.lower()
        assert "implementation" in content
        assert "phase 5" in content


class TestShannonThinkingThoughtTypes:
    """Tests for thought type assignment based on phases."""

    @pytest.mark.asyncio
    async def test_model_phase_uses_hypothesis_type(self):
        """Test that model phase uses HYPOTHESIS thought type."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to model phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(2):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        model_thought = thoughts[-1]
        assert model_thought.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_proof_phase_uses_verification_type(self):
        """Test that proof phase uses VERIFICATION thought type."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to proof phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(3):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        proof_thought = thoughts[-1]
        assert proof_thought.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_implementation_phase_uses_synthesis_type(self):
        """Test that implementation phase uses SYNTHESIS thought type."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress to implementation phase
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(4):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        impl_thought = thoughts[-1]
        assert impl_thought.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_revision_guidance_creates_revision_type(self):
        """Test that revision guidance creates REVISION thought type."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="revise the problem statement"
        )

        assert second.type == ThoughtType.REVISION

    @pytest.mark.asyncio
    async def test_branch_guidance_creates_branch_type(self):
        """Test that branch guidance creates BRANCH thought type."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="branch to explore alternative approach"
        )

        assert second.type == ThoughtType.BRANCH


class TestShannonThinkingConfidence:
    """Tests for confidence calculation across phases."""

    @pytest.mark.asyncio
    async def test_confidence_increases_through_phases(self):
        """Test that confidence generally increases as phases progress."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(4):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        # Proof phase should have highest confidence
        proof_thought = thoughts[3]
        assert proof_thought.metadata["phase"] == ShannonPhase.PROOF
        assert proof_thought.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_verification_increases_confidence(self):
        """Test that VERIFICATION thought type increases confidence."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Get to proof phase which uses VERIFICATION type
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(3):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        verification_thought = thoughts[-1]
        assert verification_thought.type == ThoughtType.VERIFICATION
        assert verification_thought.confidence > 0.8

    @pytest.mark.asyncio
    async def test_revision_decreases_confidence(self):
        """Test that REVISION thought type decreases confidence."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        revision = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="revise the approach"
        )

        # Revision should have lower confidence than normal progression
        assert revision.type == ThoughtType.REVISION
        assert revision.confidence < 0.6


class TestShannonThinkingEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_staying_in_final_phase(self):
        """Test that continuing after final phase stays in implementation."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        # Progress through all phases
        thoughts = [await method.execute(session=session, input_text="Test")]
        for _ in range(4):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        # Continue one more time
        final = await method.continue_reasoning(
            session=session,
            previous_thought=thoughts[-1]
        )

        assert final.metadata["phase"] == ShannonPhase.IMPLEMENTATION

    @pytest.mark.asyncio
    async def test_multiple_thoughts_in_same_phase(self):
        """Test that multiple thoughts can be generated in the same phase."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")

        # Refine in same phase
        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance="refine problem definition"
        )

        third = await method.continue_reasoning(
            session=session,
            previous_thought=second,
            guidance="iterate on problem definition"
        )

        assert first.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION
        assert second.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION
        assert third.metadata["phase"] == ShannonPhase.PROBLEM_DEFINITION

    @pytest.mark.asyncio
    async def test_guidance_includes_context(self):
        """Test that guidance is included in metadata and influences content."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        first = await method.execute(session=session, input_text="Test")
        guidance_text = "Focus on noise constraints"

        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            guidance=guidance_text
        )

        assert second.metadata["guidance"] == guidance_text
        assert guidance_text in second.content

    @pytest.mark.asyncio
    async def test_context_preserved_across_continuation(self):
        """Test that context is preserved in thought metadata."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        context = {"domain": "information theory", "complexity": "high"}

        first = await method.execute(
            session=session,
            input_text="Test",
            context=context
        )

        second = await method.continue_reasoning(
            session=session,
            previous_thought=first,
            context=context
        )

        assert second.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_depth_increases_linearly(self):
        """Test that depth increases by 1 with each continuation."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        thoughts = [await method.execute(session=session, input_text="Test")]

        for i in range(5):
            thoughts.append(
                await method.continue_reasoning(
                    session=session,
                    previous_thought=thoughts[-1]
                )
            )

        for i, thought in enumerate(thoughts):
            assert thought.depth == i

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self):
        """Test that session metrics are updated correctly."""
        method = ShannonThinking()
        await method.initialize()
        session = Session().start()

        await method.execute(session=session, input_text="Test")

        assert session.metrics.total_thoughts == 1
        assert session.metrics.methods_used[MethodIdentifier.SHANNON_THINKING] == 1

        await method.continue_reasoning(
            session=session,
            previous_thought=session.get_recent_thoughts(n=1)[0]
        )

        assert session.metrics.total_thoughts == 2
        assert session.metrics.methods_used[MethodIdentifier.SHANNON_THINKING] == 2
