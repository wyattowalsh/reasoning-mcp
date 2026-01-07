"""
Comprehensive tests for Socratic Reasoning method.

This module provides complete test coverage for the SocraticReasoning class
from src/reasoning_mcp/methods/native/socratic.py.

Test coverage includes:
1. Initialization and health checks
2. Basic execution with questioning dialogue
3. Question types (clarifying, probing, challenging)
4. Configuration and phases
5. Continue reasoning with deeper questions
6. Assumption exposure and tracking
7. Definition refinement through questions
8. Contradiction discovery
9. Conclusion building and synthesis
10. Edge cases (clear/vague/controversial topics)

Target: 90%+ coverage with 15+ test cases
"""

from typing import Any

import pytest

from reasoning_mcp.methods.native.socratic import (
    SOCRATIC_METADATA,
    SocraticReasoning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def socratic_method() -> SocraticReasoning:
    """Provide a Socratic Reasoning method instance for testing.

    Returns:
        Uninitialized SocraticReasoning instance.
    """
    return SocraticReasoning()


@pytest.fixture
async def initialized_method() -> SocraticReasoning:
    """Provide an initialized Socratic Reasoning method for testing.

    Returns:
        Initialized SocraticReasoning instance ready for execution.
    """
    method = SocraticReasoning()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Provide an active session for testing.

    Returns:
        Session in ACTIVE status.
    """
    session = Session()
    session.start()
    return session


@pytest.fixture
def sample_claim() -> str:
    """Provide a sample philosophical claim for testing.

    Returns:
        A testable claim suitable for Socratic examination.
    """
    return "Democracy is the best form of government"


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Provide sample context for testing.

    Returns:
        Dictionary with test context data.
    """
    return {"domain": "political_philosophy", "depth": "moderate"}


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSocraticInitialization:
    """Test suite for SocraticReasoning initialization and setup."""

    def test_create_instance(self, socratic_method: SocraticReasoning):
        """Test creating a SocraticReasoning instance."""
        assert socratic_method is not None
        assert isinstance(socratic_method, SocraticReasoning)

    def test_initial_state_uninitialized(self, socratic_method: SocraticReasoning):
        """Test that new instance starts in uninitialized state."""
        assert socratic_method._initialized is False
        assert socratic_method._step_counter == 0
        assert socratic_method._current_phase == SocraticReasoning.PHASE_INITIAL
        assert socratic_method._identified_assumptions == []
        assert socratic_method._identified_contradictions == []

    @pytest.mark.asyncio
    async def test_initialize_method(self, socratic_method: SocraticReasoning):
        """Test that initialize() sets up the method correctly."""
        await socratic_method.initialize()

        assert socratic_method._initialized is True
        assert socratic_method._step_counter == 0
        assert socratic_method._current_phase == SocraticReasoning.PHASE_INITIAL
        assert socratic_method._identified_assumptions == []
        assert socratic_method._identified_contradictions == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize() resets state from previous execution."""
        method = SocraticReasoning()

        # Manually modify state to simulate previous execution
        method._initialized = True
        method._step_counter = 5
        method._current_phase = SocraticReasoning.PHASE_SYNTHESIS
        method._identified_assumptions = ["assumption1", "assumption2"]
        method._identified_contradictions = ["contradiction1"]

        # Re-initialize
        await method.initialize()

        # All state should be reset
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == SocraticReasoning.PHASE_INITIAL
        assert method._identified_assumptions == []
        assert method._identified_contradictions == []

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, socratic_method: SocraticReasoning):
        """Test health_check returns False when not initialized."""
        result = await socratic_method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: SocraticReasoning):
        """Test health_check returns True when initialized."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Properties Tests
# ============================================================================


class TestSocraticProperties:
    """Test suite for SocraticReasoning properties."""

    def test_identifier_property(self, socratic_method: SocraticReasoning):
        """Test that identifier property returns correct value."""
        assert socratic_method.identifier == MethodIdentifier.SOCRATIC

    def test_name_property(self, socratic_method: SocraticReasoning):
        """Test that name property returns correct value."""
        assert socratic_method.name == "Socratic Reasoning"
        assert socratic_method.name == SOCRATIC_METADATA.name

    def test_description_property(self, socratic_method: SocraticReasoning):
        """Test that description property returns correct value."""
        assert "question-driven" in socratic_method.description.lower()
        assert socratic_method.description == SOCRATIC_METADATA.description

    def test_category_property(self, socratic_method: SocraticReasoning):
        """Test that category property returns correct value."""
        assert socratic_method.category == MethodCategory.ADVANCED


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSocraticMetadata:
    """Test suite for SOCRATIC_METADATA constant."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert SOCRATIC_METADATA.identifier == MethodIdentifier.SOCRATIC

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert SOCRATIC_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test metadata contains expected tags."""
        expected_tags = {
            "socratic",
            "questioning",
            "critical-thinking",
            "assumptions",
            "contradictions",
            "discovery",
            "dialogue",
        }
        assert expected_tags.issubset(SOCRATIC_METADATA.tags)

    def test_metadata_complexity(self):
        """Test metadata has appropriate complexity level."""
        assert SOCRATIC_METADATA.complexity == 4  # Medium-high

    def test_metadata_branching_support(self):
        """Test metadata correctly indicates no branching support."""
        assert SOCRATIC_METADATA.supports_branching is False

    def test_metadata_revision_support(self):
        """Test metadata correctly indicates revision support."""
        assert SOCRATIC_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self):
        """Test metadata specifies minimum thoughts."""
        assert SOCRATIC_METADATA.min_thoughts == 3


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSocraticExecution:
    """Test suite for basic SocraticReasoning execution."""

    @pytest.mark.asyncio
    async def test_execute_not_initialized_raises_error(
        self, socratic_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError) as exc_info:
            await socratic_method.execute(session=active_session, input_text=sample_claim)

        assert "must be initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SOCRATIC
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_contains_questioning(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() creates content with questioning dialogue."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        content = thought.content.lower()
        # Should contain question indicators
        assert "?" in thought.content
        assert any(
            word in content
            for word in ["question", "ask", "examine", "explore", "what", "why", "how"]
        )

    @pytest.mark.asyncio
    async def test_execute_includes_input_claim(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() includes the input claim in the content."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        assert sample_claim in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_metadata(
        self,
        initialized_method: SocraticReasoning,
        active_session: Session,
        sample_claim: str,
        sample_context: dict[str, Any],
    ):
        """Test that execute() sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim, context=sample_context
        )

        assert thought.metadata is not None
        assert thought.metadata["input"] == sample_claim
        assert thought.metadata["context"] == sample_context
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_INITIAL
        assert thought.metadata["reasoning_type"] == "socratic"
        assert "questioning_focus" in thought.metadata

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() adds thought to session."""
        initial_count = active_session.thought_count

        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        assert active_session.thought_count == initial_count + 1
        assert thought.id in active_session.graph.nodes
        assert active_session.current_method == MethodIdentifier.SOCRATIC

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() sets initial confidence level."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Initial confidence should be relatively low (questioning process)
        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.6  # Expected initial confidence

    @pytest.mark.asyncio
    async def test_execute_resets_step_counter(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that execute() resets step counter for new execution."""
        # Manually set step counter to simulate previous execution
        initialized_method._step_counter = 10

        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Step counter should be reset to 1
        assert thought.step_number == 1
        assert initialized_method._step_counter == 1


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSocraticContinueReasoning:
    """Test suite for continuing Socratic reasoning."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_not_initialized_raises_error(
        self, socratic_method: SocraticReasoning, active_session: Session
    ):
        """Test that continue_reasoning() raises error when not initialized."""
        # Create a dummy thought
        previous = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SOCRATIC,
            content="Test",
            step_number=1,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await socratic_method.continue_reasoning(
                session=active_session, previous_thought=previous
            )

        assert "must be initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() increments step counter."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.step_number == 2
        assert second.step_number == first.step_number + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() sets parent relationship."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_depth(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() increases depth."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() incorporates guidance."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        guidance = "What do you mean by 'best'?"
        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first, guidance=guidance
        )

        assert guidance in second.content
        assert second.metadata["guidance"] == guidance

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_confidence(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() progressively increases confidence."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )
        initial_confidence = first.confidence

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first
        )

        # Confidence should increase as understanding deepens
        assert second.confidence >= initial_confidence

    @pytest.mark.asyncio
    async def test_continue_reasoning_progresses_through_phases(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that continue_reasoning() progresses through questioning phases."""
        thoughts = []

        # Execute initial thought
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )
        thoughts.append(thought)

        # Continue through multiple steps to see phase progression
        for i in range(10):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=f"Response {i}",
            )
            thoughts.append(thought)

        # Verify phases progress
        phases_seen = {t.metadata["phase"] for t in thoughts}

        # Should progress through different phases
        assert SocraticReasoning.PHASE_INITIAL in phases_seen
        assert SocraticReasoning.PHASE_ASSUMPTIONS in phases_seen
        assert SocraticReasoning.PHASE_CONTRADICTIONS in phases_seen
        assert SocraticReasoning.PHASE_EXPLORATION in phases_seen
        assert SocraticReasoning.PHASE_SYNTHESIS in phases_seen


# ============================================================================
# Phase Progression Tests
# ============================================================================


class TestSocraticPhases:
    """Test suite for Socratic questioning phase progression."""

    @pytest.mark.asyncio
    async def test_phase_initial_for_steps_1_2(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that steps 1-2 use PHASE_INITIAL."""
        thought1 = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )
        assert thought1.metadata["phase"] == SocraticReasoning.PHASE_INITIAL

        thought2 = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought1
        )
        assert thought2.metadata["phase"] == SocraticReasoning.PHASE_INITIAL

    @pytest.mark.asyncio
    async def test_phase_assumptions_for_steps_3_4(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that steps 3-4 use PHASE_ASSUMPTIONS."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to step 3
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert thought.step_number == 3
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_ASSUMPTIONS

    @pytest.mark.asyncio
    async def test_phase_contradictions_for_steps_5_6(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that steps 5-6 use PHASE_CONTRADICTIONS."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to step 5
        for _ in range(4):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert thought.step_number == 5
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_CONTRADICTIONS

    @pytest.mark.asyncio
    async def test_phase_exploration_for_steps_7_8(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that steps 7-8 use PHASE_EXPLORATION."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to step 7
        for _ in range(6):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert thought.step_number == 7
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_EXPLORATION

    @pytest.mark.asyncio
    async def test_phase_synthesis_for_step_9_plus(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that steps 9+ use PHASE_SYNTHESIS."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to step 9
        for _ in range(8):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        assert thought.step_number == 9
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_SYNTHESIS
        assert thought.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_thought_types_match_phases(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that thought types are appropriate for each phase."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Step 1: Initial phase -> INITIAL type
        assert thought.type == ThoughtType.INITIAL

        # Step 2: Initial phase -> CONTINUATION type
        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )
        assert thought.type == ThoughtType.CONTINUATION

        # Step 3-4: Assumptions phase -> HYPOTHESIS type
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
        assert thought.type == ThoughtType.HYPOTHESIS

        # Step 5-6: Contradictions phase -> VERIFICATION type
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
        assert thought.type == ThoughtType.VERIFICATION

        # Step 9+: Synthesis phase -> SYNTHESIS type
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )
        assert thought.type == ThoughtType.SYNTHESIS


# ============================================================================
# Assumption Tracking Tests
# ============================================================================


class TestAssumptionExposure:
    """Test suite for assumption identification and tracking."""

    @pytest.mark.asyncio
    async def test_tracks_assumptions_in_metadata(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that assumptions are tracked in thought metadata."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to assumptions phase
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance="Democratic participation is inherently good",
            )

        # Check metadata tracks assumptions
        assert "assumptions_identified" in thought.metadata
        assert isinstance(thought.metadata["assumptions_identified"], int)

    @pytest.mark.asyncio
    async def test_assumptions_phase_asks_about_assumptions(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that assumptions phase contains assumption-related questions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to assumptions phase (step 3)
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        content_lower = thought.content.lower()
        assert any(
            word in content_lower
            for word in ["assumption", "assuming", "take for granted", "presume"]
        )

    @pytest.mark.asyncio
    async def test_assumption_count_increments(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that assumption count increments with guidance."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to assumptions phase
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        initial_count = thought.metadata["assumptions_identified"]

        # Add guidance that represents an assumption
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="People always want to participate in government",
        )

        # Count should increment
        assert thought.metadata["assumptions_identified"] > initial_count


# ============================================================================
# Question Type Tests
# ============================================================================


class TestQuestionTypes:
    """Test suite for different types of Socratic questions."""

    @pytest.mark.asyncio
    async def test_clarifying_questions_in_initial_phase(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that initial phase asks clarifying questions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        content_lower = thought.content.lower()
        # Should ask about definitions and meaning
        assert any(
            phrase in content_lower
            for phrase in ["what do you mean", "define", "clarify", "mean by"]
        )

    @pytest.mark.asyncio
    async def test_probing_questions_dig_deeper(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that probing questions dig deeper into concepts."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        thought = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=thought
        )

        # Second thought should probe deeper
        content_lower = thought.content.lower()
        assert "?" in thought.content  # Contains questions
        assert any(
            word in content_lower
            for word in ["deeper", "probe", "why", "how", "explain"]
        )

    @pytest.mark.asyncio
    async def test_challenging_questions_find_contradictions(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that contradiction phase asks challenging questions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to contradictions phase (step 5)
        for _ in range(4):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        content_lower = thought.content.lower()
        assert any(
            word in content_lower
            for word in ["contradict", "conflict", "inconsist", "challenge", "counterexample"]
        )

    @pytest.mark.asyncio
    async def test_exploratory_questions_seek_implications(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that exploration phase asks about implications."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to exploration phase (step 7)
        for _ in range(6):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        content_lower = thought.content.lower()
        assert any(
            word in content_lower
            for word in ["implication", "consequence", "if this is true", "relate"]
        )


# ============================================================================
# Synthesis and Conclusion Tests
# ============================================================================


class TestSynthesisAndConclusion:
    """Test suite for synthesis phase and conclusion building."""

    @pytest.mark.asyncio
    async def test_synthesis_phase_summarizes_insights(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that synthesis phase summarizes insights gained."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to synthesis phase (step 9)
        for _ in range(8):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        content_lower = thought.content.lower()
        assert any(
            word in content_lower
            for word in ["insight", "discover", "understanding", "wisdom", "learn"]
        )

    @pytest.mark.asyncio
    async def test_synthesis_includes_assumption_count(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that synthesis references identified assumptions."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress through dialogue with assumptions
        for i in range(8):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=f"Assumption {i}",
            )

        # Synthesis should reference assumptions examined
        assert "assumption" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_synthesis_references_questioning_process(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that synthesis references the Socratic questioning process."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress to synthesis
        for _ in range(8):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        content_lower = thought.content.lower()
        assert any(
            word in content_lower for word in ["question", "dialogue", "inquiry", "socratic"]
        )


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_execute_with_clear_simple_topic(
        self, initialized_method: SocraticReasoning, active_session: Session
    ):
        """Test execution with a clear, simple topic."""
        simple_claim = "Water is essential for life"

        thought = await initialized_method.execute(
            session=active_session, input_text=simple_claim
        )

        assert thought is not None
        assert simple_claim in thought.content
        assert "?" in thought.content  # Still asks questions

    @pytest.mark.asyncio
    async def test_execute_with_vague_topic(
        self, initialized_method: SocraticReasoning, active_session: Session
    ):
        """Test execution with a vague, unclear topic."""
        vague_claim = "Things are happening"

        thought = await initialized_method.execute(
            session=active_session, input_text=vague_claim
        )

        assert thought is not None
        assert vague_claim in thought.content
        # Should ask clarifying questions
        content_lower = thought.content.lower()
        assert "clarif" in content_lower or "mean" in content_lower

    @pytest.mark.asyncio
    async def test_execute_with_controversial_topic(
        self, initialized_method: SocraticReasoning, active_session: Session
    ):
        """Test execution with a controversial topic."""
        controversial_claim = "All taxation is theft"

        thought = await initialized_method.execute(
            session=active_session, input_text=controversial_claim
        )

        assert thought is not None
        assert controversial_claim in thought.content
        # Should ask probing questions
        assert "?" in thought.content

    @pytest.mark.asyncio
    async def test_execute_with_empty_context(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test execution with None context."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim, context=None
        )

        assert thought is not None
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_continue_without_guidance(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test continue_reasoning without guidance parameter."""
        first = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        second = await initialized_method.continue_reasoning(
            session=active_session, previous_thought=first, guidance=None
        )

        assert second is not None
        assert second.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_long_dialogue_chain(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test a long dialogue chain of 15+ thoughts."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Create chain of 15 thoughts
        for i in range(15):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=f"Response to step {i + 1}",
            )

        # Should reach step 16
        assert thought.step_number == 16
        # Should be in synthesis phase (steps 9+)
        assert thought.metadata["phase"] == SocraticReasoning.PHASE_SYNTHESIS
        # Confidence should have increased
        assert thought.confidence > 0.6

    @pytest.mark.asyncio
    async def test_execute_resets_between_sessions(
        self, initialized_method: SocraticReasoning, sample_claim: str
    ):
        """Test that execute() properly resets state between different sessions."""
        # First session
        session1 = Session().start()
        thought1 = await initialized_method.execute(session=session1, input_text=sample_claim)

        # Continue a few steps
        for _ in range(5):
            thought1 = await initialized_method.continue_reasoning(
                session=session1, previous_thought=thought1
            )

        # New session - should reset
        session2 = Session().start()
        different_claim = "Science is the best path to knowledge"
        thought2 = await initialized_method.execute(session=session2, input_text=different_claim)

        # Should start fresh
        assert thought2.step_number == 1
        assert thought2.metadata["phase"] == SocraticReasoning.PHASE_INITIAL
        assert different_claim in thought2.content

    @pytest.mark.asyncio
    async def test_contradiction_tracking(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that contradictions are tracked in metadata."""
        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )

        # Progress through dialogue
        for _ in range(8):
            thought = await initialized_method.continue_reasoning(
                session=active_session, previous_thought=thought
            )

        # Check contradiction tracking
        assert "contradictions_found" in thought.metadata
        assert isinstance(thought.metadata["contradictions_found"], int)
        assert thought.metadata["contradictions_found"] >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestSocraticIntegration:
    """Integration tests for complete Socratic dialogue flows."""

    @pytest.mark.asyncio
    async def test_complete_socratic_dialogue(
        self, initialized_method: SocraticReasoning, active_session: Session
    ):
        """Test a complete Socratic dialogue from start to synthesis."""
        claim = "Knowledge is power"

        # Initial question
        thought = await initialized_method.execute(session=active_session, input_text=claim)
        assert thought.type == ThoughtType.INITIAL
        assert "?" in thought.content

        # Clarification (step 2)
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Power to influence others and make informed decisions",
        )
        assert thought.type == ThoughtType.CONTINUATION

        # Assumption identification (steps 3-4)
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="I assume knowledge always leads to positive outcomes",
        )
        assert thought.type == ThoughtType.HYPOTHESIS

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Knowledge is universally accessible",
        )

        # Contradiction examination (steps 5-6)
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Knowledge can be misused for harmful purposes",
        )
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Not all knowledge is equally empowering",
        )

        # Exploration (steps 7-8)
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="Different types of knowledge provide different types of power",
        )
        assert thought.type == ThoughtType.CONTINUATION

        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="The relationship between knowledge and power is complex",
        )

        # Synthesis (step 9)
        thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=thought,
            guidance="We need to qualify what kind of knowledge and what kind of power",
        )
        assert thought.type == ThoughtType.SYNTHESIS
        assert "insight" in thought.content.lower() or "understanding" in thought.content.lower()

        # Verify session state
        assert active_session.thought_count == 9  # 1 initial + 8 continuations (steps 2-9)
        assert active_session.current_method == MethodIdentifier.SOCRATIC

    @pytest.mark.asyncio
    async def test_confidence_progression_through_dialogue(
        self, initialized_method: SocraticReasoning, active_session: Session, sample_claim: str
    ):
        """Test that confidence increases appropriately through dialogue."""
        confidences = []

        thought = await initialized_method.execute(
            session=active_session, input_text=sample_claim
        )
        confidences.append(thought.confidence)

        # Progress through 10 steps
        for i in range(10):
            thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
                guidance=f"Thoughtful response {i}",
            )
            confidences.append(thought.confidence)

        # Confidence should generally increase
        assert confidences[-1] > confidences[0]
        # But should cap at reasonable level (not exceed 0.9)
        assert all(c <= 0.9 for c in confidences)
        # All confidences should be valid
        assert all(0.0 <= c <= 1.0 for c in confidences)
