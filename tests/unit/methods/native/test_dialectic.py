"""Unit tests for Dialectic reasoning method.

This module contains comprehensive tests for the DialecticMethod class,
covering initialization, execution, dialectical phases (thesis-antithesis-synthesis),
configuration, continuation, and edge cases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.dialectic import DIALECTIC_METADATA, Dialectic
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def dialectic_method() -> Dialectic:
    """Create a fresh Dialectic method instance.

    Returns:
        Uninitialized Dialectic method instance
    """
    return Dialectic()


@pytest.fixture
async def initialized_method() -> Dialectic:
    """Create and initialize a Dialectic method.

    Returns:
        Initialized Dialectic method instance
    """
    method = Dialectic()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Create an active session.

    Returns:
        Session in ACTIVE status
    """
    return Session().start()


class TestDialecticMetadata:
    """Tests for Dialectic method metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert DIALECTIC_METADATA.identifier == MethodIdentifier.DIALECTIC

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert DIALECTIC_METADATA.name == "Dialectic"

    def test_metadata_description(self):
        """Test that metadata has a description."""
        assert len(DIALECTIC_METADATA.description) > 0
        assert "thesis" in DIALECTIC_METADATA.description.lower()

    def test_metadata_category(self):
        """Test that Dialectic is a high-value method."""
        assert DIALECTIC_METADATA.category == MethodCategory.HIGH_VALUE

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        assert "dialectic" in DIALECTIC_METADATA.tags
        assert "thesis" in DIALECTIC_METADATA.tags
        assert "antithesis" in DIALECTIC_METADATA.tags
        assert "synthesis" in DIALECTIC_METADATA.tags

    def test_metadata_complexity(self):
        """Test that complexity is at expected level."""
        assert DIALECTIC_METADATA.complexity == 6

    def test_metadata_supports_branching(self):
        """Test that Dialectic supports branching."""
        assert DIALECTIC_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that Dialectic supports revision."""
        assert DIALECTIC_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self):
        """Test that minimum thoughts is at least 3 (thesis, antithesis, synthesis)."""
        assert DIALECTIC_METADATA.min_thoughts == 3

    def test_metadata_best_for(self):
        """Test that best_for contains expected use cases."""
        best_for_str = " ".join(DIALECTIC_METADATA.best_for).lower()
        assert "philosophical" in best_for_str or "ethical" in best_for_str


class TestDialecticInitialization:
    """Tests for Dialectic method initialization."""

    def test_create_uninitialized(self, dialectic_method: Dialectic):
        """Test creating an uninitialized method."""
        assert dialectic_method._initialized is False
        assert dialectic_method._step_counter == 0
        assert dialectic_method._phase == "thesis"

    async def test_initialize(self, dialectic_method: Dialectic):
        """Test initializing the method."""
        await dialectic_method.initialize()
        assert dialectic_method._initialized is True
        assert dialectic_method._step_counter == 0
        assert dialectic_method._phase == "thesis"
        assert dialectic_method._thesis_id is None
        assert dialectic_method._antithesis_count == 0

    async def test_reinitialize(self, initialized_method: Dialectic):
        """Test that reinitializing resets state."""
        # Modify state
        initialized_method._step_counter = 5
        initialized_method._phase = "synthesis"
        initialized_method._antithesis_count = 2

        # Reinitialize
        await initialized_method.initialize()

        # Check state is reset
        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._phase == "thesis"
        assert initialized_method._antithesis_count == 0

    async def test_health_check_uninitialized(self, dialectic_method: Dialectic):
        """Test health check on uninitialized method."""
        assert await dialectic_method.health_check() is False

    async def test_health_check_initialized(self, initialized_method: Dialectic):
        """Test health check on initialized method."""
        assert await initialized_method.health_check() is True


class TestDialecticProperties:
    """Tests for Dialectic method properties."""

    def test_identifier_property(self, dialectic_method: Dialectic):
        """Test identifier property returns correct value."""
        assert dialectic_method.identifier == MethodIdentifier.DIALECTIC

    def test_name_property(self, dialectic_method: Dialectic):
        """Test name property returns correct value."""
        assert dialectic_method.name == "Dialectic"

    def test_description_property(self, dialectic_method: Dialectic):
        """Test description property returns correct value."""
        assert dialectic_method.description == DIALECTIC_METADATA.description

    def test_category_property(self, dialectic_method: Dialectic):
        """Test category property returns correct value."""
        assert dialectic_method.category == MethodCategory.HIGH_VALUE


class TestDialecticExecute:
    """Tests for Dialectic method execute() function."""

    async def test_execute_creates_thesis(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that execute creates a thesis thought."""
        input_text = "Should we prioritize economic growth over environmental protection?"

        thought = await initialized_method.execute(session=active_session, input_text=input_text)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)

    async def test_execute_thesis_properties(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that thesis has correct properties."""
        input_text = "Is artificial intelligence beneficial to humanity?"

        thought = await initialized_method.execute(session=active_session, input_text=input_text)

        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.DIALECTIC
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.confidence == 0.7
        assert thought.metadata["phase"] == "thesis"
        assert thought.metadata["input"] == input_text

    async def test_execute_adds_to_session(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that execute adds thought to session."""
        input_text = "Is democracy the best form of government?"

        thought = await initialized_method.execute(session=active_session, input_text=input_text)

        assert active_session.thought_count == 1
        assert active_session.current_method == MethodIdentifier.DIALECTIC
        assert thought.id in active_session.graph.nodes

    async def test_execute_updates_method_state(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that execute updates method internal state."""
        input_text = "Should we allow genetic engineering in humans?"

        thought = await initialized_method.execute(session=active_session, input_text=input_text)

        assert initialized_method._step_counter == 1
        assert initialized_method._phase == "antithesis"
        assert initialized_method._thesis_id == thought.id
        assert initialized_method._antithesis_count == 0

    async def test_execute_with_context(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test execute with additional context."""
        input_text = "Should we regulate social media platforms?"
        context = {"domain": "technology", "urgency": "high"}

        thought = await initialized_method.execute(
            session=active_session, input_text=input_text, context=context
        )

        assert thought.metadata["context"] == context

    async def test_execute_uninitialized_raises_error(
        self, dialectic_method: Dialectic, active_session: Session
    ):
        """Test that execute raises error when not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await dialectic_method.execute(
                session=active_session,
                input_text="Test question?",
            )

    async def test_execute_content_includes_thesis(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that thesis content indicates it's establishing a position."""
        thought = await initialized_method.execute(
            session=active_session,
            input_text="Should we adopt universal basic income?",
        )

        content_lower = thought.content.lower()
        assert "thesis" in content_lower
        assert "step 1" in content_lower


class TestDialecticContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_to_antithesis(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test continuing from thesis to antithesis."""
        # Create thesis
        thesis = await initialized_method.execute(
            session=active_session, input_text="Test question?"
        )

        # Continue to antithesis
        antithesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
            guidance="Develop the opposing viewpoint",
        )

        assert antithesis is not None
        assert antithesis.type == ThoughtType.BRANCH
        assert antithesis.metadata["phase"] == "antithesis"

    async def test_antithesis_properties(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that antithesis has correct properties."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        antithesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
            guidance="opposing view",
        )

        assert antithesis.step_number == 2
        assert antithesis.depth == 1
        assert antithesis.confidence == 0.7
        assert antithesis.branch_id == "antithesis_1"
        # Antithesis should branch from thesis
        assert antithesis.parent_id == thesis.id

    async def test_continue_to_synthesis(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test continuing from antithesis to synthesis."""
        # Create thesis
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        # Create antithesis
        antithesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
        )

        # Create synthesis
        synthesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=antithesis,
            guidance="Synthesize the perspectives",
        )

        assert synthesis.type == ThoughtType.SYNTHESIS
        assert synthesis.metadata["phase"] == "synthesis"

    async def test_synthesis_properties(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that synthesis has correct properties."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        assert synthesis.step_number == 3
        assert synthesis.depth == 2
        # Synthesis should have higher confidence
        assert synthesis.confidence >= antithesis.confidence
        assert synthesis.parent_id == antithesis.id

    async def test_multiple_antitheses(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test creating multiple antitheses (branching)."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        # Create first antithesis
        antithesis1 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
            guidance="First opposing view",
        )

        # Create second antithesis
        antithesis2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
            guidance="alternative opposing view",
        )

        assert antithesis1.branch_id == "antithesis_1"
        assert antithesis2.branch_id == "antithesis_2"
        assert initialized_method._antithesis_count == 2

    async def test_phase_detection_from_guidance(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that phase is correctly detected from guidance keywords."""
        await initialized_method.execute(session=active_session, input_text="Question?")

        # Test antithesis detection
        for keyword in ["antithesis", "opposing", "counter", "alternative"]:
            method = Dialectic()
            await method.initialize()
            sess = Session().start()
            t = await method.execute(session=sess, input_text="Q?")
            result = await method.continue_reasoning(
                session=sess,
                previous_thought=t,
                guidance=f"Consider the {keyword} view",
            )
            assert result.metadata["phase"] == "antithesis"

    async def test_phase_detection_synthesis_keywords(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test synthesis detection from guidance keywords."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        await initialized_method.continue_reasoning(session=active_session, previous_thought=thesis)

        # Test synthesis detection
        for keyword in ["synthesis", "synthesize", "integrate", "combine", "conclude"]:
            method = Dialectic()
            await method.initialize()
            sess = Session().start()
            t = await method.execute(session=sess, input_text="Q?")
            a = await method.continue_reasoning(session=sess, previous_thought=t)
            result = await method.continue_reasoning(
                session=sess,
                previous_thought=a,
                guidance=f"Now {keyword} the ideas",
            )
            assert result.metadata["phase"] == "synthesis"

    async def test_continuation_after_synthesis(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test continuing after synthesis creates continuation."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        continuation = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=synthesis,
            guidance="Further develop this idea",
        )

        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.metadata["phase"] == "continuation"

    async def test_continue_uninitialized_raises_error(
        self, dialectic_method: Dialectic, active_session: Session
    ):
        """Test that continue_reasoning raises error when not initialized."""
        thesis = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIALECTIC,
            content="Test",
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await dialectic_method.continue_reasoning(
                session=active_session, previous_thought=thesis
            )

    async def test_step_counter_increments(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that step counter increments with each continuation."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        assert thesis.step_number == 1

        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        assert antithesis.step_number == 2

        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )
        assert synthesis.step_number == 3


class TestDialecticThreePhaseStructure:
    """Tests for the three-phase dialectical structure."""

    async def test_complete_dialectic_cycle(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test a complete thesis-antithesis-synthesis cycle."""
        # Thesis
        thesis = await initialized_method.execute(
            session=active_session,
            input_text="Is renewable energy economically viable?",
        )
        assert thesis.metadata["phase"] == "thesis"
        assert thesis.type == ThoughtType.INITIAL

        # Antithesis
        antithesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
        )
        assert antithesis.metadata["phase"] == "antithesis"
        assert antithesis.type == ThoughtType.BRANCH

        # Synthesis
        synthesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=antithesis,
        )
        assert synthesis.metadata["phase"] == "synthesis"
        assert synthesis.type == ThoughtType.SYNTHESIS

    async def test_thesis_content_structure(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that thesis content has expected structure."""
        thesis = await initialized_method.execute(
            session=active_session,
            input_text="Should we colonize Mars?",
        )

        content = thesis.content.lower()
        assert "thesis" in content
        assert "position" in content or "perspective" in content

    async def test_antithesis_content_structure(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that antithesis content has expected structure."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )

        content = antithesis.content.lower()
        assert "antithesis" in content
        assert "opposing" in content or "opposition" in content

    async def test_synthesis_content_structure(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that synthesis content has expected structure."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        content = synthesis.content.lower()
        assert "synthesis" in content
        assert "integrat" in content or "perspect" in content

    async def test_confidence_progression(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that confidence increases through synthesis."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        # Thesis and antithesis should have equal confidence
        assert thesis.confidence == antithesis.confidence == 0.7
        # Synthesis should have higher confidence
        assert synthesis.confidence > thesis.confidence

    async def test_metadata_tracks_previous_phase(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that metadata tracks previous phase."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )

        assert antithesis.metadata["previous_phase"] == "thesis"

        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        assert synthesis.metadata["previous_phase"] == "antithesis"


class TestDialecticMultiRoundDialectic:
    """Tests for multiple rounds of dialectical reasoning."""

    async def test_second_dialectic_round(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that a second dialectic round can be initiated."""
        # First round
        thesis1 = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis1 = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis1
        )
        synthesis1 = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis1
        )

        # Second round - synthesis becomes new thesis
        antithesis2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=synthesis1,
            guidance="Now consider an alternative perspective to this synthesis",
        )

        # Should create new antithesis
        assert "antithesis" in antithesis2.metadata["phase"]

    async def test_complex_branching_structure(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test complex branching with multiple antitheses."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        # Create multiple antitheses
        antitheses = []
        for i in range(3):
            ant = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thesis,
                guidance=f"Alternative view {i + 1}",
            )
            antitheses.append(ant)

        # All should branch from thesis
        for ant in antitheses:
            assert ant.parent_id == thesis.id
            assert ant.depth == 1

        # Should have different branch IDs
        branch_ids = [ant.branch_id for ant in antitheses]
        assert len(set(branch_ids)) == 3


class TestDialecticEdgeCases:
    """Tests for edge cases and special scenarios."""

    async def test_uncontroversial_topic(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test with an uncontroversial topic."""
        thesis = await initialized_method.execute(
            session=active_session,
            input_text="Is 2+2 equal to 4?",
        )

        # Should still create thesis even for uncontroversial topic
        assert thesis.metadata["phase"] == "thesis"
        assert thesis.type == ThoughtType.INITIAL

        # Antithesis should still be generated
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        assert antithesis.metadata["phase"] == "antithesis"

    async def test_empty_guidance(self, initialized_method: Dialectic, active_session: Session):
        """Test continue_reasoning with empty guidance."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        # Empty guidance should still infer phase from previous thought
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis, guidance=""
        )

        assert antithesis.metadata["phase"] == "antithesis"

    async def test_none_guidance(self, initialized_method: Dialectic, active_session: Session):
        """Test continue_reasoning with None guidance."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        # None guidance should use phase inference
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis, guidance=None
        )

        assert antithesis.metadata["phase"] == "antithesis"

    async def test_empty_context(self, initialized_method: Dialectic, active_session: Session):
        """Test execute with empty context dict."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Question?", context={}
        )

        assert thought.metadata["context"] == {}

    async def test_none_context(self, initialized_method: Dialectic, active_session: Session):
        """Test execute with None context."""
        thought = await initialized_method.execute(
            session=active_session, input_text="Question?", context=None
        )

        assert thought.metadata["context"] == {}

    async def test_very_long_input(self, initialized_method: Dialectic, active_session: Session):
        """Test with very long input text."""
        long_input = "Should we " + "really " * 100 + "do this?"

        thought = await initialized_method.execute(session=active_session, input_text=long_input)

        assert thought.metadata["input"] == long_input

    async def test_session_metrics_updated(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that session metrics are updated correctly."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        assert active_session.metrics.total_thoughts == 1
        assert active_session.metrics.methods_used[str(MethodIdentifier.DIALECTIC)] == 1
        assert active_session.metrics.thought_types[str(ThoughtType.INITIAL)] == 1

        await initialized_method.continue_reasoning(session=active_session, previous_thought=thesis)

        assert active_session.metrics.total_thoughts == 2
        assert active_session.metrics.thought_types[str(ThoughtType.BRANCH)] == 1

    async def test_guidance_preserved_in_metadata(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that guidance is preserved in thought metadata."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")

        guidance_text = "Explore the counterargument thoroughly"
        antithesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thesis,
            guidance=guidance_text,
        )

        assert antithesis.metadata["guidance"] == guidance_text


class TestDialecticSessionIntegration:
    """Tests for integration with Session."""

    async def test_thought_graph_structure(
        self, initialized_method: Dialectic, active_session: Session
    ):
        """Test that thoughts form correct graph structure."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        # Check graph structure
        assert active_session.graph.node_count == 3
        assert antithesis.parent_id == thesis.id
        assert synthesis.parent_id == antithesis.id

    async def test_current_method_set(self, initialized_method: Dialectic, active_session: Session):
        """Test that current_method is set on session."""
        await initialized_method.execute(session=active_session, input_text="Question?")

        assert active_session.current_method == MethodIdentifier.DIALECTIC

    async def test_depth_tracking(self, initialized_method: Dialectic, active_session: Session):
        """Test that depth is correctly tracked."""
        thesis = await initialized_method.execute(session=active_session, input_text="Question?")
        antithesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thesis
        )
        synthesis = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=antithesis
        )

        assert thesis.depth == 0
        assert antithesis.depth == 1
        assert synthesis.depth == 2
        assert active_session.current_depth == 2

    async def test_multiple_methods_same_session(self, active_session: Session):
        """Test using multiple Dialectic instances in same session."""
        method1 = Dialectic()
        await method1.initialize()

        method2 = Dialectic()
        await method2.initialize()

        # Both should be able to add thoughts to the same session
        thought1 = await method1.execute(session=active_session, input_text="First question?")
        thought2 = await method2.execute(session=active_session, input_text="Second question?")

        assert active_session.thought_count == 2
        assert thought1.id != thought2.id
