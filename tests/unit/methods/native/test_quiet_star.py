"""Comprehensive tests for QuietStar reasoning method.

This module provides complete test coverage for the QuietStar method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Phase progression (rationale → integrate → output)
- Configuration options (max_rationale_tokens, integration_threshold)
- Continue reasoning flow
- Integration weight tracking
- Inner thoughts management
- Rationale token tracking
- Phase transitions
- Edge cases

The tests aim for 90%+ coverage of the QuietStar implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.quiet_star import (
    QUIET_STAR_METADATA,
    QuietStar,
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
def method() -> QuietStar:
    """Provide a QuietStar method instance for testing.

    Returns:
        QuietStar instance (uninitialized).
    """
    return QuietStar()


@pytest.fixture
async def initialized_method() -> QuietStar:
    """Provide an initialized QuietStar method instance.

    Returns:
        Initialized QuietStar instance.
    """
    method = QuietStar()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.fixture
def simple_input() -> str:
    """Provide a simple test input.

    Returns:
        Simple question for testing.
    """
    return "What is photosynthesis?"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex question requiring deeper reasoning.
    """
    return "Explain the relationship between entropy, information theory, and thermodynamics"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestQuietStarMetadata:
    """Test suite for QuietStar metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert QUIET_STAR_METADATA.identifier == MethodIdentifier.QUIET_STAR

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert QUIET_STAR_METADATA.name == "Quiet-STaR"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert QUIET_STAR_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert QUIET_STAR_METADATA.complexity == 7
        assert 1 <= QUIET_STAR_METADATA.complexity <= 10

    def test_metadata_no_revision(self):
        """Test that metadata indicates no revision support."""
        assert QUIET_STAR_METADATA.supports_revision is False

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert QUIET_STAR_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "internal-reasoning",
            "rationale",
            "self-taught",
            "inner-thoughts",
            "think-before-speaking",
        }
        assert expected_tags.issubset(QUIET_STAR_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert QUIET_STAR_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert QUIET_STAR_METADATA.max_thoughts == 5

    def test_metadata_best_for(self):
        """Test that metadata includes appropriate best_for cases."""
        assert "complex reasoning tasks" in QUIET_STAR_METADATA.best_for
        assert "quality-critical outputs" in QUIET_STAR_METADATA.best_for


# ============================================================================
# Initialization Tests
# ============================================================================


class TestQuietStarInitialization:
    """Test suite for QuietStar initialization."""

    def test_create_method(self, method: QuietStar):
        """Test creating a QuietStar instance."""
        assert isinstance(method, QuietStar)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: QuietStar):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.QUIET_STAR
        assert method.name == "Quiet-STaR"
        assert method.category == MethodCategory.ADVANCED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: QuietStar):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "rationale"
        assert method._rationale_tokens == 0
        assert method._integration_weight == 0.0
        assert method._inner_thoughts == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = QuietStar()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "output"
        method._rationale_tokens = 300
        method._integration_weight = 0.8
        method._inner_thoughts = ["thought1", "thought2"]

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._current_phase == "rationale"
        assert method._rationale_tokens == 0
        assert method._integration_weight == 0.0
        assert method._inner_thoughts == []

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: QuietStar):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: QuietStar):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestQuietStarExecution:
    """Test suite for basic QuietStar execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_reasoning_thought(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute creates a REASONING thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.REASONING
        assert thought.method_id == MethodIdentifier.QUIET_STAR
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_rationale_metadata(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["phase"] == "rationale"
        assert "rationale_tokens" in thought.metadata
        assert "max_rationale_tokens" in thought.metadata
        assert thought.metadata["reasoning_type"] == "quiet_star"

    @pytest.mark.asyncio
    async def test_execute_tracks_inner_thoughts(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute tracks inner thoughts."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "inner_thoughts_count" in thought.metadata
        assert thought.metadata["inner_thoughts_count"] == 1
        assert "inner_thought" in thought.metadata
        assert len(initialized_method._inner_thoughts) == 1

    @pytest.mark.asyncio
    async def test_execute_sets_quality_scores(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute sets initial quality and confidence."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert thought.quality_score == 0.7
        assert thought.confidence == 0.65

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.QUIET_STAR
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"max_rationale_tokens": 256, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["max_rationale_tokens"] == 256
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_max_rationale_tokens(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that max_rationale_tokens is clamped to valid range."""
        # Test upper bound
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_rationale_tokens": 5000},
        )
        assert thought1.metadata["max_rationale_tokens"] == 2048

        # Re-initialize for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test lower bound
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"max_rationale_tokens": 50},
        )
        assert thought2.metadata["max_rationale_tokens"] == 128

    @pytest.mark.asyncio
    async def test_execute_estimates_rationale_tokens(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that rationale tokens are estimated."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert initialized_method._rationale_tokens > 0
        assert thought.metadata["rationale_tokens"] > 0


# ============================================================================
# Phase Progression Tests
# ============================================================================


class TestPhaseProgression:
    """Test suite for the phase progression flow."""

    @pytest.mark.asyncio
    async def test_integrate_phase_after_rationale(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integrate follows rationale thought."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)

        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert integrate.type == ThoughtType.SYNTHESIS
        assert integrate.metadata["phase"] == "integrate"
        assert integrate.parent_id == rationale.id
        assert integrate.step_number == 2

    @pytest.mark.asyncio
    async def test_output_phase_after_integrate(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that output follows integrate."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        assert output.type == ThoughtType.CONCLUSION
        assert output.metadata["phase"] == "output"
        assert output.parent_id == integrate.id
        assert output.step_number == 3

    @pytest.mark.asyncio
    async def test_full_phase_progression(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test a complete phase progression: rationale → integrate → output."""
        # Rationale
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        assert rationale.metadata["phase"] == "rationale"
        assert rationale.type == ThoughtType.REASONING

        # Integrate
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        assert integrate.metadata["phase"] == "integrate"
        assert integrate.type == ThoughtType.SYNTHESIS

        # Output
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )
        assert output.metadata["phase"] == "output"
        assert output.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_optional_elaboration_after_output(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test optional elaboration after output."""
        # Complete full progression
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Optional elaboration
        elaborate = await initialized_method.continue_reasoning(
            session=session, previous_thought=output
        )

        assert elaborate.type == ThoughtType.CONTINUATION
        assert elaborate.metadata["phase"] == "elaborate"
        assert elaborate.step_number == 4


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    @pytest.mark.asyncio
    async def test_default_max_rationale_tokens(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test default max rationale tokens."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["max_rationale_tokens"] == QuietStar.MAX_RATIONALE_TOKENS
        assert thought.metadata["max_rationale_tokens"] == 512

    @pytest.mark.asyncio
    async def test_custom_max_rationale_tokens(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test custom max rationale tokens in context."""
        custom_max = 768

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_rationale_tokens": custom_max},
        )

        assert thought.metadata["max_rationale_tokens"] == custom_max

    @pytest.mark.asyncio
    async def test_integration_threshold_constant(self):
        """Test that INTEGRATION_THRESHOLD is properly defined."""
        assert QuietStar.INTEGRATION_THRESHOLD == 0.7
        assert isinstance(QuietStar.INTEGRATION_THRESHOLD, float)

    @pytest.mark.asyncio
    async def test_max_inner_thoughts_constant(self):
        """Test that MAX_INNER_THOUGHTS is properly defined."""
        assert QuietStar.MAX_INNER_THOUGHTS == 3
        assert isinstance(QuietStar.MAX_INNER_THOUGHTS, int)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: QuietStar, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.QUIET_STAR,
            content="Test",
            metadata={"phase": "rationale"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        assert rationale.step_number == 1

        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        assert integrate.step_number == 2

        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )
        assert output.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Focus on biological processes"
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale, guidance=guidance_text
        )

        assert "guidance" in integrate.metadata
        assert integrate.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)

        context = {"additional_info": "test data"}
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale, context=context
        )

        assert "context" in integrate.metadata
        assert integrate.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        count_after_rationale = session.thought_count

        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert session.thought_count == count_after_rationale + 1
        assert integrate.id in session.graph.nodes


# ============================================================================
# Integration Weight Tests
# ============================================================================


class TestIntegrationWeight:
    """Test suite for integration weight tracking."""

    @pytest.mark.asyncio
    async def test_integration_weight_computed(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integration weight is computed."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert "integration_weight" in integrate.metadata
        assert initialized_method._integration_weight > 0

    @pytest.mark.asyncio
    async def test_integration_weight_affects_confidence(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integration weight affects confidence."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        # Higher integration weight should lead to higher confidence
        integrate.metadata["integration_weight"]
        assert integrate.confidence >= 0.7
        assert integrate.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_integration_weight_affects_quality(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integration weight affects quality score."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        # Higher integration weight should lead to higher quality
        assert integrate.quality_score >= 0.75
        assert integrate.quality_score <= 0.95

    @pytest.mark.asyncio
    async def test_integration_weight_propagates_to_output(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integration weight propagates to output phase."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Output should reference integration weight
        assert "integration_weight" in output.metadata
        assert output.metadata["integration_weight"] == integrate.metadata["integration_weight"]


# ============================================================================
# Inner Thoughts Tests
# ============================================================================


class TestInnerThoughts:
    """Test suite for inner thoughts tracking."""

    @pytest.mark.asyncio
    async def test_inner_thoughts_created_on_execute(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that inner thoughts are created on execute."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert len(initialized_method._inner_thoughts) == 1
        assert thought.metadata["inner_thoughts_count"] == 1
        assert "inner_thought" in thought.metadata

    @pytest.mark.asyncio
    async def test_inner_thought_content_meaningful(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that inner thought has meaningful content."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        inner_thought = thought.metadata["inner_thought"]
        assert isinstance(inner_thought, str)
        assert len(inner_thought) > 0
        assert "rationale" in inner_thought.lower()

    @pytest.mark.asyncio
    async def test_inner_thoughts_count_tracked(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that inner thoughts count is tracked through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Count should be maintained
        assert rationale.metadata["inner_thoughts_count"] == 1
        assert integrate.metadata["inner_thoughts_count"] == 1
        assert output.metadata["inner_thoughts_count"] == 1


# ============================================================================
# Rationale Token Tracking Tests
# ============================================================================


class TestRationaleTokens:
    """Test suite for rationale token tracking."""

    @pytest.mark.asyncio
    async def test_rationale_tokens_tracked(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that rationale tokens are tracked."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert initialized_method._rationale_tokens > 0
        assert thought.metadata["rationale_tokens"] > 0

    @pytest.mark.asyncio
    async def test_rationale_tokens_in_metadata(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that rationale tokens appear in metadata."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert "rationale_tokens" in rationale.metadata
        assert "rationale_tokens" in integrate.metadata

    @pytest.mark.asyncio
    async def test_rationale_tokens_consistent(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that rationale tokens remain consistent through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        initial_tokens = rationale.metadata["rationale_tokens"]

        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        # Should be the same (from initial rationale)
        assert integrate.metadata["rationale_tokens"] == initial_tokens


# ============================================================================
# Quality Progression Tests
# ============================================================================


class TestQualityProgression:
    """Test suite for quality score progression through phases."""

    @pytest.mark.asyncio
    async def test_quality_improves_through_phases(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that quality generally improves through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Quality should generally improve
        assert integrate.quality_score >= rationale.quality_score
        assert output.quality_score >= integrate.quality_score

    @pytest.mark.asyncio
    async def test_confidence_improves_through_phases(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that confidence improves through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Confidence should improve
        assert integrate.confidence >= rationale.confidence
        assert output.confidence >= integrate.confidence

    @pytest.mark.asyncio
    async def test_quality_caps_at_one(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that quality score caps at 1.0."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # All scores should be <= 1.0
        assert rationale.quality_score <= 1.0
        assert integrate.quality_score <= 1.0
        assert output.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_output_has_high_quality(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that final output has high quality."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Output should have high quality (>= 0.85)
        assert output.quality_score >= 0.85
        assert output.confidence >= 0.8


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: QuietStar, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: QuietStar, session: Session):
        """Test handling of very long input."""
        long_input = "Explain " + "complex " * 1000 + "system?"
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_fallback_to_integrate_phase(
        self, initialized_method: QuietStar, session: Session
    ):
        """Test fallback to integrate for unknown phase."""
        # Create a thought with unknown phase
        rationale = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        rationale.metadata["phase"] = "unknown_phase"

        # Continue should fallback to integrate
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert thought.type == ThoughtType.SYNTHESIS
        assert thought.metadata["phase"] == "integrate"

    @pytest.mark.asyncio
    async def test_multiple_executions_reset_state(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that multiple executions reset state properly."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought1.step_number == 1

        # Second execution should reset
        session2 = Session().start()
        thought2 = await initialized_method.execute(session=session2, input_text=simple_input)
        assert thought2.step_number == 1

    @pytest.mark.asyncio
    async def test_extreme_max_rationale_tokens(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test handling of extreme max_rationale_tokens values."""
        # Test very high value
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_rationale_tokens": 999999},
        )
        assert thought1.metadata["max_rationale_tokens"] == 2048  # clamped

        # New session for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test very low value
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"max_rationale_tokens": 1},
        )
        assert thought2.metadata["max_rationale_tokens"] == 128  # clamped


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_rationale_content_structure(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that rationale has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Internal Rationale" in content
        assert simple_input in content

    @pytest.mark.asyncio
    async def test_integrate_content_structure(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that integrate has expected content structure."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        content = integrate.content
        assert isinstance(content, str)
        assert "Integration Phase" in content or "Integration" in content
        assert "Integration Weight" in content or "integration" in content.lower()

    @pytest.mark.asyncio
    async def test_output_content_structure(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that output has expected content structure."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        content = output.content
        assert isinstance(content, str)
        assert "Final Output" in content or "Output" in content
        assert "Internal" in content or "rationale" in content.lower()

    @pytest.mark.asyncio
    async def test_elaboration_content_structure(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that elaboration has expected content structure."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )
        elaborate = await initialized_method.continue_reasoning(
            session=session, previous_thought=output
        )

        content = elaborate.content
        assert isinstance(content, str)
        assert "Elaboration" in content or "Additional" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that guidance appears in generated content."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)

        guidance = "Focus on molecular details"
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale, guidance=guidance
        )

        # Guidance should appear in metadata
        assert guidance in integrate.metadata["guidance"]


# ============================================================================
# Metadata Consistency Tests
# ============================================================================


class TestMetadataConsistency:
    """Test suite for metadata consistency across phases."""

    @pytest.mark.asyncio
    async def test_previous_phase_tracked(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that previous phase is tracked in metadata."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )

        assert "previous_phase" in integrate.metadata
        assert integrate.metadata["previous_phase"] == "rationale"

    @pytest.mark.asyncio
    async def test_reasoning_type_consistent(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that reasoning_type is consistent through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        assert rationale.metadata["reasoning_type"] == "quiet_star"
        assert integrate.metadata["reasoning_type"] == "quiet_star"
        assert output.metadata["reasoning_type"] == "quiet_star"

    @pytest.mark.asyncio
    async def test_method_id_consistent(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that method_id is consistent through phases."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        assert rationale.method_id == MethodIdentifier.QUIET_STAR
        assert integrate.method_id == MethodIdentifier.QUIET_STAR
        assert output.method_id == MethodIdentifier.QUIET_STAR


# ============================================================================
# Complex Scenario Tests
# ============================================================================


class TestComplexScenarios:
    """Test suite for complex usage scenarios."""

    @pytest.mark.asyncio
    async def test_simple_problem_completes_normally(
        self, initialized_method: QuietStar, session: Session
    ):
        """Test that simple problems complete through all phases."""
        simple_question = "What is H2O?"

        rationale = await initialized_method.execute(session=session, input_text=simple_question)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        assert output.type == ThoughtType.CONCLUSION
        assert session.thought_count == 3

    @pytest.mark.asyncio
    async def test_complex_problem_completes(
        self, initialized_method: QuietStar, session: Session, complex_input: str
    ):
        """Test that complex problems complete through all phases."""
        rationale = await initialized_method.execute(session=session, input_text=complex_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        assert output.type == ThoughtType.CONCLUSION
        assert output.quality_score >= 0.85

    @pytest.mark.asyncio
    async def test_with_custom_configuration(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test execution with custom configuration."""
        context = {
            "max_rationale_tokens": 1024,
            "additional_context": "scientific domain",
        }

        rationale = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert rationale.metadata["max_rationale_tokens"] == 1024
        assert rationale.metadata["context"]["additional_context"] == "scientific domain"

    @pytest.mark.asyncio
    async def test_depth_increases_with_continuation(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that depth increases with each continuation."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        assert rationale.depth == 0

        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        assert integrate.depth == 1

        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )
        assert output.depth == 2

    @pytest.mark.asyncio
    async def test_parent_child_relationships(
        self, initialized_method: QuietStar, session: Session, simple_input: str
    ):
        """Test that parent-child relationships are established correctly."""
        rationale = await initialized_method.execute(session=session, input_text=simple_input)
        integrate = await initialized_method.continue_reasoning(
            session=session, previous_thought=rationale
        )
        output = await initialized_method.continue_reasoning(
            session=session, previous_thought=integrate
        )

        # Check parent relationships
        assert rationale.parent_id is None
        assert integrate.parent_id == rationale.id
        assert output.parent_id == integrate.id
