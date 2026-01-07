"""Unit tests for LateralThinking reasoning method.

This module provides comprehensive tests for the LateralThinkingMethod implementation,
covering initialization, execution, lateral thinking techniques, configuration,
continuation, creative exploration, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.lateral import (
    LATERAL_THINKING_METADATA,
    LateralThinkingMethod,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def lateral_method() -> LateralThinkingMethod:
    """Create a LateralThinkingMethod instance for testing.

    Returns:
        A fresh LateralThinkingMethod instance
    """
    return LateralThinkingMethod()


@pytest.fixture
def initialized_method() -> LateralThinkingMethod:
    """Create a LateralThinkingMethod instance for testing.

    Returns:
        A LateralThinkingMethod instance (tests must call initialize())
    """
    return LateralThinkingMethod()


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
        A sample creative problem string
    """
    return "How can we reduce traffic congestion in cities?"


@pytest.fixture
def concrete_problem() -> str:
    """Provide a concrete problem for testing.

    Returns:
        A concrete problem requiring creative solutions
    """
    return "How can we improve employee engagement in remote teams?"


@pytest.fixture
def abstract_problem() -> str:
    """Provide an abstract problem for testing.

    Returns:
        An abstract problem requiring lateral thinking
    """
    return "What is the nature of consciousness and how does it emerge?"


class TestLateralThinkingInitialization:
    """Tests for LateralThinkingMethod initialization and setup."""

    def test_create_method(self, lateral_method: LateralThinkingMethod):
        """Test that LateralThinkingMethod can be instantiated."""
        assert lateral_method is not None
        assert isinstance(lateral_method, LateralThinkingMethod)

    def test_initial_state(self, lateral_method: LateralThinkingMethod):
        """Test that a new method starts in the correct initial state."""
        assert lateral_method._initialized is False
        assert lateral_method._step_counter == 0
        assert lateral_method._phase == "framing"
        assert lateral_method._root_id is None
        assert len(lateral_method._techniques_used) == 0
        assert len(lateral_method._ideas_generated) == 0

    async def test_initialize(self, lateral_method: LateralThinkingMethod):
        """Test that initialize() sets up the method correctly."""
        await lateral_method.initialize()
        assert lateral_method._initialized is True
        assert lateral_method._step_counter == 0
        assert lateral_method._phase == "framing"
        assert lateral_method._root_id is None
        assert len(lateral_method._techniques_used) == 0
        assert len(lateral_method._ideas_generated) == 0

    async def test_initialize_resets_state(self):
        """Test that initialize() resets state even if called multiple times."""
        method = LateralThinkingMethod()
        await method.initialize()
        method._step_counter = 5
        method._phase = "synthesis"
        method._techniques_used = {"assumption_challenge", "random_entry"}
        method._ideas_generated = ["idea1", "idea2"]

        # Re-initialize
        await method.initialize()
        assert method._step_counter == 0
        assert method._phase == "framing"
        assert len(method._techniques_used) == 0
        assert len(method._ideas_generated) == 0
        assert method._initialized is True

    async def test_health_check_not_initialized(self, lateral_method: LateralThinkingMethod):
        """Test that health_check returns False before initialization."""
        result = await lateral_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, lateral_method: LateralThinkingMethod):
        """Test that health_check returns True after initialization."""
        await lateral_method.initialize()
        result = await lateral_method.health_check()
        assert result is True


class TestLateralThinkingProperties:
    """Tests for LateralThinkingMethod property accessors."""

    def test_identifier_property(self, lateral_method: LateralThinkingMethod):
        """Test that identifier returns the correct method identifier."""
        assert lateral_method.identifier == MethodIdentifier.LATERAL_THINKING

    def test_name_property(self, lateral_method: LateralThinkingMethod):
        """Test that name returns the correct human-readable name."""
        assert lateral_method.name == "Lateral Thinking"

    def test_description_property(self, lateral_method: LateralThinkingMethod):
        """Test that description returns the correct method description."""
        assert "creative" in lateral_method.description.lower()
        assert "divergent" in lateral_method.description.lower()

    def test_category_property(self, lateral_method: LateralThinkingMethod):
        """Test that category returns the correct method category."""
        assert lateral_method.category == MethodCategory.HOLISTIC


class TestLateralThinkingMetadata:
    """Tests for LateralThinking metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert LATERAL_THINKING_METADATA.identifier == MethodIdentifier.LATERAL_THINKING

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert LATERAL_THINKING_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"creative", "divergent", "innovation", "assumptions", "lateral"}
        assert expected_tags.issubset(LATERAL_THINKING_METADATA.tags)

    def test_metadata_supports_branching(self):
        """Test that metadata correctly indicates branching support."""
        assert LATERAL_THINKING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata correctly indicates revision support."""
        assert LATERAL_THINKING_METADATA.supports_revision is True

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= LATERAL_THINKING_METADATA.complexity <= 10
        # Lateral thinking should be moderately complex
        assert LATERAL_THINKING_METADATA.complexity >= 5

    def test_metadata_thought_bounds(self):
        """Test that metadata defines min/max thought counts."""
        assert LATERAL_THINKING_METADATA.min_thoughts >= 5
        assert LATERAL_THINKING_METADATA.max_thoughts == 0  # No limit


class TestLateralThinkingExecution:
    """Tests for LateralThinkingMethod execute() method."""

    async def test_execute_basic(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test basic execution creates a thought."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.LATERAL_THINKING

    async def test_execute_requires_initialization(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute raises error if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await lateral_method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates an INITIAL thought type."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_framing_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute starts with framing phase."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert thought.metadata["phase"] == "framing"
        assert "FRAMING" in thought.content

    async def test_execute_adds_to_session(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute adds thought to the session."""
        await lateral_method.initialize()
        initial_count = session.thought_count

        await lateral_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_execute_sets_session_method(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute updates session's current method."""
        await lateral_method.initialize()
        await lateral_method.execute(session, sample_problem)

        assert session.current_method == MethodIdentifier.LATERAL_THINKING

    async def test_execute_with_context(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test execute with custom context parameters."""
        await lateral_method.initialize()
        context: dict[str, Any] = {"technique_preference": "random_entry"}

        thought = await lateral_method.execute(session, sample_problem, context=context)

        assert thought is not None
        assert "input" in thought.metadata
        assert thought.metadata["input"] == sample_problem
        assert "context" in thought.metadata

    async def test_execute_sets_confidence(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets a confidence score."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert 0.0 <= thought.confidence <= 1.0
        # Initial framing should have moderate confidence
        assert thought.confidence > 0.5

    async def test_execute_sets_step_number(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets correct step number."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert thought.step_number == 1
        assert lateral_method._step_counter == 1

    async def test_execute_stores_root_id(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute stores root thought ID."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert lateral_method._root_id == thought.id

    async def test_execute_transitions_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute transitions to assumption_challenge phase."""
        await lateral_method.initialize()
        await lateral_method.execute(session, sample_problem)

        assert lateral_method._phase == "assumption_challenge"

    async def test_execute_includes_problem_reference(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that framing includes the problem statement."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert sample_problem in thought.content

    async def test_execute_lists_techniques(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that framing metadata lists available techniques."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert "techniques_available" in thought.metadata
        techniques = thought.metadata["techniques_available"]
        expected = [
            "assumption_challenge",
            "random_entry",
            "reversal",
            "analogy",
            "provocation",
            "alternative_perspective",
        ]
        assert all(tech in techniques for tech in expected)


class TestLateralThinkingContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_reasoning_basic(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test basic continuation of reasoning."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        continuation = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Challenge assumptions",
        )

        assert continuation is not None
        assert isinstance(continuation, ThoughtNode)

    async def test_continue_requires_initialization(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning raises error if not initialized."""
        # Create an initial thought with a different method
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LATERAL_THINKING,
            content="Test",
        )
        session.add_thought(thought)

        with pytest.raises(RuntimeError, match="must be initialized"):
            await lateral_method.continue_reasoning(session, thought)

    async def test_continue_increments_step(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation increments step counter."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        continuation = await lateral_method.continue_reasoning(session, initial)

        assert continuation.step_number == 2
        assert lateral_method._step_counter == 2

    async def test_continue_adds_to_session(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation is added to session."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)
        count_before = session.thought_count

        await lateral_method.continue_reasoning(session, initial)

        assert session.thought_count == count_before + 1

    async def test_continue_with_guidance(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation with explicit guidance."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        guidance = "Use random entry technique"
        continuation = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance=guidance,
        )

        assert "guidance" in continuation.metadata
        assert continuation.metadata["guidance"] == guidance

    async def test_continue_without_guidance(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation without explicit guidance."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        continuation = await lateral_method.continue_reasoning(session, initial)

        assert continuation.content != ""
        assert "phase" in continuation.metadata

    async def test_continue_tracks_techniques_used(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation tracks which techniques were used."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        continuation = await lateral_method.continue_reasoning(session, initial)

        assert "techniques_used" in continuation.metadata
        technique = continuation.metadata.get("technique")
        if technique:
            assert technique in lateral_method._techniques_used


class TestAssumptionChallenging:
    """Tests for assumption challenging technique."""

    async def test_assumption_challenge_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test assumption challenge phase generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        # First continuation should be assumption challenge
        challenge = await lateral_method.continue_reasoning(session, initial)

        assert challenge.metadata["phase"] == "assumption_challenge"
        assert challenge.metadata["technique"] == "assumption_challenge"

    async def test_assumption_challenge_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test assumption challenge includes questioning."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)
        challenge = await lateral_method.continue_reasoning(session, initial)

        content_lower = challenge.content.lower()
        assert "assumption" in content_lower
        assert "challenge" in content_lower or "question" in content_lower

    async def test_assumption_challenge_thought_type(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test assumption challenge uses EXPLORATION thought type."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)
        challenge = await lateral_method.continue_reasoning(session, initial)

        assert challenge.type == ThoughtType.EXPLORATION

    async def test_assumption_challenge_guidance(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test explicit guidance triggers assumption challenge."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        # Set phase to something else first
        lateral_method._phase = "divergent_exploration"

        challenge = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Challenge the core assumptions",
        )

        assert challenge.metadata["phase"] == "assumption_challenge"


class TestRandomEntryTechnique:
    """Tests for random entry technique."""

    async def test_random_entry_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test random entry technique generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        # First go through assumption_challenge (method looks at previous_thought.metadata["phase"])
        challenge = await lateral_method.continue_reasoning(session, initial)
        assert challenge.metadata["phase"] == "assumption_challenge"

        # Now continue from assumption_challenge - should progress to random_entry
        random_entry = await lateral_method.continue_reasoning(session, challenge)

        assert random_entry.metadata["phase"] == "divergent_exploration"
        assert random_entry.metadata["technique"] == "random_entry"

    async def test_random_entry_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test random entry includes random stimulus."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        random_entry = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Use random unrelated concept",
        )

        content_upper = random_entry.content
        assert "RANDOM" in content_upper or "random" in random_entry.content.lower()

    async def test_random_entry_thought_type(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test random entry uses BRANCH thought type."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        random_entry = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Random stimulus",
        )

        assert random_entry.type == ThoughtType.BRANCH
        assert random_entry.branch_id is not None

    async def test_random_entry_guidance(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test explicit guidance triggers random entry."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        random_entry = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Use random stimulus",
        )

        assert random_entry.metadata["technique"] == "random_entry"


class TestReversalTechnique:
    """Tests for reversal technique."""

    async def test_reversal_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test reversal technique generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        reversal = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Think about the opposite",
        )

        assert reversal.metadata["technique"] == "reversal"

    async def test_reversal_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test reversal includes flipping/opposite concepts."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        reversal = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Reverse the problem",
        )

        content_lower = reversal.content.lower()
        reversal_markers = ["reverse", "opposite", "flip", "worse"]
        assert any(marker in content_lower for marker in reversal_markers)

    async def test_reversal_thought_type(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test reversal uses BRANCH thought type."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        reversal = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Reversal technique",
        )

        assert reversal.type == ThoughtType.BRANCH


class TestProvocationTechnique:
    """Tests for provocation technique."""

    async def test_provocation_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test provocation technique generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        provocation = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="What if we had unlimited resources?",
        )

        assert provocation.metadata["technique"] == "provocation"

    async def test_provocation_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test provocation includes 'what if' scenarios."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        provocation = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Provocation: what if",
        )

        content_lower = provocation.content.lower()
        assert "what if" in content_lower or "provocation" in content_lower


class TestAlternativePerspective:
    """Tests for alternative perspective technique."""

    async def test_alternative_perspective_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test alternative perspective technique generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        perspective = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="View from different stakeholder perspective",
        )

        assert perspective.metadata["technique"] == "alternative_perspective"

    async def test_alternative_perspective_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test alternative perspective includes different viewpoints."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        perspective = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Different perspective",
        )

        content_lower = perspective.content.lower()
        perspective_markers = ["perspective", "viewpoint", "view", "stakeholder"]
        assert any(marker in content_lower for marker in perspective_markers)


class TestAnalogyTechnique:
    """Tests for analogy/connection making technique."""

    async def test_analogy_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test analogy technique generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        analogy = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Draw analogies from other domains",
        )

        assert analogy.metadata["technique"] == "analogy"

    async def test_analogy_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test analogy includes connection making."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        analogy = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Find analogies",
        )

        content_lower = analogy.content.lower()
        analogy_markers = ["analog", "similar", "like", "parallel", "connection"]
        assert any(marker in content_lower for marker in analogy_markers)

    async def test_analogy_thought_type(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test analogy uses HYPOTHESIS thought type."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        analogy = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Analogy",
        )

        assert analogy.type == ThoughtType.HYPOTHESIS


class TestSynthesisTechnique:
    """Tests for synthesis phase."""

    async def test_synthesis_phase(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test synthesis phase generation."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        synthesis = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Synthesize and combine insights",
        )

        assert synthesis.metadata["technique"] == "synthesis"

    async def test_synthesis_content(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test synthesis includes integration."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        synthesis = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Synthesis",
        )

        content_lower = synthesis.content.lower()
        synthesis_markers = ["synthesis", "synthesiz", "combine", "integrate", "harvest"]
        assert any(marker in content_lower for marker in synthesis_markers)

    async def test_synthesis_thought_type(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test synthesis uses SYNTHESIS thought type."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        synthesis = await lateral_method.continue_reasoning(
            session,
            initial,
            guidance="Synthesis solution",
        )

        assert synthesis.type == ThoughtType.SYNTHESIS


class TestConfiguration:
    """Tests for configuration options."""

    async def test_technique_selection(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that guidance can select specific techniques."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        # Test each technique can be selected via guidance
        techniques = [
            ("Challenge assumptions", "assumption_challenge"),
            ("Random entry", "random_entry"),
            ("Reversal", "reversal"),
            ("Analogy", "analogy"),
            ("Provocation", "provocation"),
            ("Alternative perspective", "alternative_perspective"),
        ]

        for guidance, expected_technique in techniques:
            thought = await lateral_method.continue_reasoning(
                session,
                initial,
                guidance=guidance,
            )
            assert thought.metadata["technique"] == expected_technique

    async def test_creativity_level_metadata(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that thoughts include creativity-related metadata."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, sample_problem)

        assert "reasoning_type" in thought.metadata
        assert thought.metadata["reasoning_type"] == "lateral_thinking"

    async def test_context_passed_through(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that context is passed through to metadata."""
        await lateral_method.initialize()
        context: dict[str, Any] = {"creativity_boost": True}

        thought = await lateral_method.execute(session, sample_problem, context=context)

        assert "context" in thought.metadata
        assert thought.metadata["context"] == context


class TestPhaseProgression:
    """Tests for phase progression and transition."""

    async def test_phase_progression_framing_to_challenge(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test progression from framing to assumption challenge."""
        await lateral_method.initialize()
        framing = await lateral_method.execute(session, sample_problem)

        assert framing.metadata["phase"] == "framing"

        challenge = await lateral_method.continue_reasoning(session, framing)
        assert challenge.metadata["phase"] == "assumption_challenge"
        assert challenge.metadata["previous_phase"] == "framing"

    async def test_phase_progression_challenge_to_exploration(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test progression from challenge to divergent exploration."""
        await lateral_method.initialize()
        framing = await lateral_method.execute(session, sample_problem)
        challenge = await lateral_method.continue_reasoning(session, framing)

        exploration = await lateral_method.continue_reasoning(session, challenge)

        assert exploration.metadata["phase"] == "divergent_exploration"
        assert exploration.metadata["previous_phase"] == "assumption_challenge"

    async def test_multiple_exploration_techniques(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test multiple exploration techniques can be used."""
        await lateral_method.initialize()
        framing = await lateral_method.execute(session, sample_problem)
        current = framing

        # Generate several exploration thoughts
        for _ in range(4):
            current = await lateral_method.continue_reasoning(session, current)

        # Should have used multiple techniques
        assert len(lateral_method._techniques_used) >= 2

    async def test_phase_metadata_tracking(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that phase metadata is properly tracked."""
        await lateral_method.initialize()
        thoughts = []

        current = await lateral_method.execute(session, sample_problem)
        thoughts.append(current)

        for _ in range(3):
            current = await lateral_method.continue_reasoning(session, current)
            thoughts.append(current)

        # Each thought should have phase metadata
        for thought in thoughts:
            assert "phase" in thought.metadata


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_concrete_problem(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        concrete_problem: str,
    ):
        """Test lateral thinking with concrete problem."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, concrete_problem)

        assert thought is not None
        assert concrete_problem in thought.content
        assert thought.metadata["phase"] == "framing"

    async def test_abstract_problem(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        abstract_problem: str,
    ):
        """Test lateral thinking with abstract problem."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, abstract_problem)

        assert thought is not None
        assert thought.content != ""

    async def test_empty_problem_string(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
    ):
        """Test execution with empty problem string."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, "")

        assert thought is not None
        assert thought.content != ""

    async def test_very_short_problem(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
    ):
        """Test execution with very short problem."""
        await lateral_method.initialize()
        thought = await lateral_method.execute(session, "Innovate?")

        assert thought is not None
        assert thought.content != ""

    async def test_creative_block_scenario(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
    ):
        """Test handling of creative blocks through techniques."""
        await lateral_method.initialize()
        problem = "We're stuck on this problem and need a breakthrough"
        thought = await lateral_method.execute(session, problem)

        # Lateral thinking should provide creative exploration
        assert thought is not None
        assert "creative" in thought.content.lower() or "assumption" in thought.content.lower()

    async def test_multiple_sessions(
        self,
        lateral_method: LateralThinkingMethod,
        sample_problem: str,
    ):
        """Test method can handle multiple sessions."""
        await lateral_method.initialize()

        session1 = Session().start()
        thought1 = await lateral_method.execute(session1, sample_problem)

        # Re-initialize for new session
        await lateral_method.initialize()
        session2 = Session().start()
        thought2 = await lateral_method.execute(session2, sample_problem)

        assert thought1.id != thought2.id
        assert lateral_method._step_counter == 1

    async def test_deep_reasoning_chain(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test deeply nested reasoning chain."""
        await lateral_method.initialize()
        current = await lateral_method.execute(session, sample_problem)

        # Create a chain of lateral thinking
        for i in range(6):
            current = await lateral_method.continue_reasoning(session, current)

        # Should have explored multiple techniques
        assert len(lateral_method._techniques_used) >= 3
        assert lateral_method._step_counter >= 7

    async def test_branching_support(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that exploration techniques create branches."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)

        # Create exploration branches
        branches = []
        for guidance in ["Random entry", "Reversal", "Provocation"]:
            branch = await lateral_method.continue_reasoning(
                session,
                initial,
                guidance=guidance,
            )
            branches.append(branch)

        # Branches should have branch_ids
        branch_count = sum(1 for b in branches if b.branch_id is not None)
        assert branch_count > 0

    async def test_unicode_in_problem(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
    ):
        """Test execution with Unicode characters."""
        await lateral_method.initialize()
        problem = "如何创新思考这个问题？ (How to think creatively about this?)"
        thought = await lateral_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_special_characters_in_problem(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
    ):
        """Test execution with special characters."""
        await lateral_method.initialize()
        problem = "How to solve: x² + y² = z² → creative approaches? @#$%"
        thought = await lateral_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session thought count updates correctly."""
        await lateral_method.initialize()
        initial_count = session.thought_count

        await lateral_method.execute(session, sample_problem)
        assert session.thought_count == initial_count + 1

        await lateral_method.continue_reasoning(
            session,
            session.get_recent_thoughts(1)[0],
        )
        assert session.thought_count == initial_count + 2

    async def test_session_metrics_update(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session metrics update after execution."""
        await lateral_method.initialize()
        await lateral_method.execute(session, sample_problem)

        assert session.metrics.total_thoughts > 0
        assert session.metrics.average_confidence > 0.0

    async def test_session_method_tracking(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session tracks method usage."""
        await lateral_method.initialize()
        await lateral_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.LATERAL_THINKING)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session can filter thoughts by method."""
        await lateral_method.initialize()
        await lateral_method.execute(session, sample_problem)

        lateral_thoughts = session.get_thoughts_by_method(MethodIdentifier.LATERAL_THINKING)
        assert len(lateral_thoughts) > 0

    async def test_session_graph_structure(
        self,
        lateral_method: LateralThinkingMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that thoughts are properly linked in session graph."""
        await lateral_method.initialize()
        initial = await lateral_method.execute(session, sample_problem)
        continuation = await lateral_method.continue_reasoning(session, initial)

        # Check graph structure
        assert session.graph.node_count >= 2
        assert continuation.id in session.graph.nodes
        assert initial.id in session.graph.nodes
