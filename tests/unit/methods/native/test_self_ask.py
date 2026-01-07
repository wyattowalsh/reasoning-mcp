"""Comprehensive tests for SelfAsk reasoning method.

This module provides complete test coverage for the SelfAskMethod class,
testing all aspects of the self-ask reasoning pattern including:
- Initialization and health checks
- Basic execution and question generation
- Configuration options
- Continue reasoning and question decomposition
- Answer integration and synthesis
- Recursive questioning behavior
- Termination conditions
- Edge cases
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.self_ask import SELF_ASK_METADATA, SelfAsk
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def self_ask_method() -> SelfAsk:
    """Provide an uninitialized SelfAsk instance.

    Returns:
        Fresh SelfAsk instance for testing.
    """
    return SelfAsk()


@pytest.fixture
async def initialized_method() -> SelfAsk:
    """Provide an initialized SelfAsk instance.

    Returns:
        Initialized SelfAsk instance ready for execution.
    """
    method = SelfAsk()
    await method.initialize()
    return method


@pytest.fixture
def active_session() -> Session:
    """Provide an active Session for testing.

    Returns:
        Session that has been started (status=ACTIVE).
    """
    session = Session()
    session.start()
    return session


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfAskMetadata:
    """Test suite for SELF_ASK_METADATA configuration."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SELF_ASK_METADATA.identifier == MethodIdentifier.SELF_ASK

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SELF_ASK_METADATA.name == "Self-Ask"

    def test_metadata_description(self):
        """Test that metadata has a description."""
        assert len(SELF_ASK_METADATA.description) > 0
        assert "subquestion" in SELF_ASK_METADATA.description.lower()

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity."""
        assert SELF_ASK_METADATA.complexity == 4
        assert 1 <= SELF_ASK_METADATA.complexity <= 10

    def test_metadata_branching_support(self):
        """Test that metadata correctly indicates no branching support."""
        assert SELF_ASK_METADATA.supports_branching is False

    def test_metadata_revision_support(self):
        """Test that metadata correctly indicates revision support."""
        assert SELF_ASK_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self):
        """Test that metadata has appropriate minimum thoughts."""
        assert SELF_ASK_METADATA.min_thoughts == 3

    def test_metadata_tags(self):
        """Test that metadata includes appropriate tags."""
        assert "self-ask" in SELF_ASK_METADATA.tags
        assert "decomposition" in SELF_ASK_METADATA.tags
        assert "subquestions" in SELF_ASK_METADATA.tags


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSelfAskInitialization:
    """Test suite for SelfAsk initialization."""

    def test_create_instance(self, self_ask_method):
        """Test creating a SelfAsk instance."""
        assert isinstance(self_ask_method, SelfAsk)
        assert self_ask_method._initialized is False

    def test_properties_before_initialization(self, self_ask_method):
        """Test that properties work before initialization."""
        assert self_ask_method.identifier == MethodIdentifier.SELF_ASK
        assert self_ask_method.name == "Self-Ask"
        assert len(self_ask_method.description) > 0
        assert self_ask_method.category == "specialized"

    @pytest.mark.asyncio
    async def test_initialize(self, self_ask_method):
        """Test initializing the method."""
        await self_ask_method.initialize()
        assert self_ask_method._initialized is True
        assert self_ask_method._step_counter == 0
        assert self_ask_method._question_stack == []
        assert self_ask_method._answered_questions == {}

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, self_ask_method):
        """Test health check returns False before initialization."""
        result = await self_ask_method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method):
        """Test health check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_reinitialize_resets_state(self, initialized_method):
        """Test that re-initializing resets internal state."""
        # Add some state
        initialized_method._step_counter = 5
        initialized_method._question_stack = ["question1", "question2"]
        initialized_method._answered_questions = {"q": "a"}

        # Re-initialize
        await initialized_method.initialize()

        # Verify reset
        assert initialized_method._step_counter == 0
        assert initialized_method._question_stack == []
        assert initialized_method._answered_questions == {}


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestSelfAskExecution:
    """Test suite for basic execute() method."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, self_ask_method, active_session
    ):
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await self_ask_method.execute(
                session=active_session,
                input_text="How does photosynthesis work?",
            )

    @pytest.mark.asyncio
    async def test_execute_simple_question(self, initialized_method, active_session):
        """Test executing with a simple question."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="Why is the sky blue?",
        )

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL
        assert result.method_id == MethodIdentifier.SELF_ASK
        assert result.step_number == 1
        assert result.depth == 0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method, active_session
    ):
        """Test that execute creates an INITIAL thought."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="How do airplanes fly?",
        )

        assert result.type == ThoughtType.INITIAL
        assert "Initial Question Analysis" in result.content
        assert "How do airplanes fly?" in result.content

    @pytest.mark.asyncio
    async def test_execute_sets_metadata(self, initialized_method, active_session):
        """Test that execute sets appropriate metadata."""
        input_text = "What causes earthquakes?"
        result = await initialized_method.execute(
            session=active_session,
            input_text=input_text,
        )

        assert result.metadata["main_question"] == input_text
        assert result.metadata["reasoning_type"] == "self_ask"
        assert result.metadata["phase"] == "initial_analysis"
        assert "pending_questions" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_with_context(self, initialized_method, active_session):
        """Test execute with optional context parameter."""
        context = {"domain": "physics", "difficulty": "advanced"}
        result = await initialized_method.execute(
            session=active_session,
            input_text="What is quantum entanglement?",
            context=context,
        )

        assert result.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(
        self, initialized_method, active_session
    ):
        """Test that execute adds the thought to the session."""
        initial_count = active_session.thought_count

        await initialized_method.execute(
            session=active_session,
            input_text="How does gravity work?",
        )

        assert active_session.thought_count == initial_count + 1
        assert active_session.current_method == MethodIdentifier.SELF_ASK

    @pytest.mark.asyncio
    async def test_execute_resets_internal_state(self, initialized_method, active_session):
        """Test that execute resets internal counters and stacks."""
        # Set some state
        initialized_method._step_counter = 5
        initialized_method._question_stack = ["old question"]
        initialized_method._answered_questions = {"old": "answer"}

        # Execute
        await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Verify reset (step_counter should be 1 after execute)
        assert initialized_method._step_counter == 1
        assert initialized_method._question_stack == []
        assert initialized_method._answered_questions == {}


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestSelfAskContinueReasoning:
    """Test suite for continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization_raises_error(
        self, self_ask_method, active_session
    ):
        """Test that continue_reasoning raises RuntimeError if not initialized."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_ASK,
            content="Test",
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await self_ask_method.continue_reasoning(
                session=active_session,
                previous_thought=thought,
            )

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method, active_session
    ):
        """Test that continue_reasoning increments the step counter."""
        # Execute first
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do magnets work?",
        )

        # Continue
        second_thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Generate first sub-question",
        )

        assert second_thought.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_links_to_parent(self, initialized_method, active_session):
        """Test that continue_reasoning links the new thought to its parent."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="What causes lightning?",
        )

        second_thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert second_thought.parent_id == first_thought.id
        assert second_thought.depth == first_thought.depth + 1

    @pytest.mark.asyncio
    async def test_continue_adds_thought_to_session(
        self, initialized_method, active_session
    ):
        """Test that continue_reasoning adds the thought to the session."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do vaccines work?",
        )

        initial_count = active_session.thought_count

        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert active_session.thought_count == initial_count + 1


# ============================================================================
# Question Generation Tests
# ============================================================================


class TestSelfAskQuestionGeneration:
    """Test suite for sub-question generation."""

    @pytest.mark.asyncio
    async def test_generate_subquestion_creates_hypothesis(
        self, initialized_method, active_session
    ):
        """Test that generating a sub-question creates HYPOTHESIS thought."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does the internet work?",
        )

        # Request question generation
        question_thought = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Generate question",
        )

        assert question_thought.type == ThoughtType.HYPOTHESIS
        assert "Sub-Question" in question_thought.content

    @pytest.mark.asyncio
    async def test_subquestion_added_to_stack(self, initialized_method, active_session):
        """Test that generated sub-questions are added to the question stack."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do rockets reach space?",
        )

        # Generate a question
        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="ask question",
        )

        # Stack should have one question
        assert len(initialized_method._question_stack) == 1

    @pytest.mark.asyncio
    async def test_multiple_subquestions_generation(
        self, initialized_method, active_session
    ):
        """Test generating multiple sub-questions."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does climate change affect ecosystems?",
        )

        # Generate first question
        q1 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Generate question",
        )

        # Generate second question
        q2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=q1,
            guidance="ask another question",
        )

        assert q1.type == ThoughtType.HYPOTHESIS
        assert q2.type == ThoughtType.HYPOTHESIS
        assert q2.step_number > q1.step_number


# ============================================================================
# Answer Integration Tests
# ============================================================================


class TestSelfAskAnswerIntegration:
    """Test suite for answering sub-questions."""

    @pytest.mark.asyncio
    async def test_answer_subquestion_creates_verification(
        self, initialized_method, active_session
    ):
        """Test that answering a sub-question creates VERIFICATION thought."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does photosynthesis work?",
        )

        # Generate a question
        question = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="ask question",
        )

        # Answer the question
        answer = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=question,
            guidance="answer",
        )

        assert answer.type == ThoughtType.VERIFICATION
        assert "Answer" in answer.content

    @pytest.mark.asyncio
    async def test_answer_stores_qa_pair(self, initialized_method, active_session):
        """Test that answering stores the Q&A pair."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do batteries work?",
        )

        # Generate and answer a question
        question = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Generate question",
        )

        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=question,
            guidance="answer",
        )

        # Should have one Q&A pair
        assert len(initialized_method._answered_questions) == 1

    @pytest.mark.asyncio
    async def test_answer_removes_from_stack(self, initialized_method, active_session):
        """Test that answering a question removes it from the stack."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does DNA replication work?",
        )

        # Generate question
        question = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="ask question",
        )

        stack_size_before = len(initialized_method._question_stack)

        # Answer the question
        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=question,
            guidance="answer",
        )

        # Stack should be smaller
        assert len(initialized_method._question_stack) < stack_size_before


# ============================================================================
# Synthesis Tests
# ============================================================================


class TestSelfAskSynthesis:
    """Test suite for synthesizing answers."""

    @pytest.mark.asyncio
    async def test_synthesis_creates_synthesis_thought(
        self, initialized_method, active_session
    ):
        """Test that synthesis creates a SYNTHESIS thought type."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does the brain store memories?",
        )

        # Request synthesis
        synthesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="synthesize",
        )

        assert synthesis.type == ThoughtType.SYNTHESIS
        assert "Synthesis" in synthesis.content

    @pytest.mark.asyncio
    async def test_synthesis_includes_answered_questions(
        self, initialized_method, active_session
    ):
        """Test that synthesis includes information about answered questions."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="What causes tides?",
        )

        # Add some answered questions manually
        initialized_method._answered_questions = {
            "What is the moon's role?": "The moon's gravity pulls on Earth's oceans",
            "What is the sun's role?": "The sun also exerts gravitational force",
        }

        synthesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="synthesize",
        )

        # Should mention the number of answered questions
        assert str(len(initialized_method._answered_questions)) in synthesis.content


# ============================================================================
# Conclusion Tests
# ============================================================================


class TestSelfAskConclusion:
    """Test suite for generating conclusions."""

    @pytest.mark.asyncio
    async def test_conclusion_creates_conclusion_thought(
        self, initialized_method, active_session
    ):
        """Test that conclusion creates a CONCLUSION thought type."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do plants grow?",
        )

        conclusion = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="conclude",
        )

        assert conclusion.type == ThoughtType.CONCLUSION
        assert "Final Answer" in conclusion.content

    @pytest.mark.asyncio
    async def test_conclusion_includes_main_question(
        self, initialized_method, active_session
    ):
        """Test that conclusion includes the original main question."""
        main_question = "How does evolution work?"
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text=main_question,
        )

        conclusion = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="final answer",
        )

        # Main question should appear in metadata or content
        assert (
            main_question in conclusion.content
            or conclusion.metadata.get("main_question") == main_question
        )


# ============================================================================
# Phase Determination Tests
# ============================================================================


class TestSelfAskPhaseDetermination:
    """Test suite for phase determination logic."""

    @pytest.mark.asyncio
    async def test_guidance_overrides_default_phase_question(
        self, initialized_method, active_session
    ):
        """Test that guidance with 'question' triggers question generation."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Generate a question",
        )

        assert result.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_guidance_overrides_default_phase_answer(
        self, initialized_method, active_session
    ):
        """Test that guidance with 'answer' triggers answer generation."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Add a question to the stack
        initialized_method._question_stack.append("Test sub-question")

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="Provide an answer",
        )

        assert result.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_guidance_overrides_default_phase_synthesize(
        self, initialized_method, active_session
    ):
        """Test that guidance with 'synthesize' triggers synthesis."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="combine the answers",
        )

        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_guidance_overrides_default_phase_conclude(
        self, initialized_method, active_session
    ):
        """Test that guidance with 'conclude' triggers conclusion."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="final conclusion",
        )

        assert result.type == ThoughtType.CONCLUSION


# ============================================================================
# Confidence Calculation Tests
# ============================================================================


class TestSelfAskConfidenceCalculation:
    """Test suite for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_initial_confidence_moderate(self, initialized_method, active_session):
        """Test that initial thought has moderate confidence."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_confidence_increases_with_answered_questions(
        self, initialized_method, active_session
    ):
        """Test that confidence increases as questions are answered."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Add answered questions
        initialized_method._answered_questions = {"q1": "a1", "q2": "a2"}

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        # Confidence should be higher with answered questions
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_confidence_within_valid_range(
        self, initialized_method, active_session
    ):
        """Test that confidence is always within valid range [0, 1]."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Test with many answered questions
        initialized_method._answered_questions = {
            f"q{i}": f"a{i}" for i in range(20)
        }

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# Metadata Tests
# ============================================================================


class TestSelfAskMetadataTracking:
    """Test suite for metadata tracking in thoughts."""

    @pytest.mark.asyncio
    async def test_metadata_tracks_pending_questions(
        self, initialized_method, active_session
    ):
        """Test that metadata tracks the number of pending questions."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Add questions to stack
        initialized_method._question_stack = ["q1", "q2", "q3"]

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert "pending_questions" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_tracks_answered_count(
        self, initialized_method, active_session
    ):
        """Test that metadata tracks the number of answered questions."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Add answered questions
        initialized_method._answered_questions = {"q1": "a1", "q2": "a2"}

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert result.metadata["answered_count"] == 2

    @pytest.mark.asyncio
    async def test_metadata_includes_phase(self, initialized_method, active_session):
        """Test that metadata includes the current phase."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="ask question",
        )

        assert "phase" in result.metadata
        assert result.metadata["phase"] == "generate_subquestion"

    @pytest.mark.asyncio
    async def test_metadata_includes_guidance(
        self, initialized_method, active_session
    ):
        """Test that metadata includes the guidance provided."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        guidance_text = "Focus on physics concepts"
        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance=guidance_text,
        )

        assert result.metadata["guidance"] == guidance_text


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSelfAskEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_question(self, initialized_method, active_session):
        """Test handling an empty question string."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="",
        )

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_very_long_question(self, initialized_method, active_session):
        """Test handling a very long question."""
        long_question = "How " + "very " * 100 + "long question?"
        result = await initialized_method.execute(
            session=active_session,
            input_text=long_question,
        )

        assert isinstance(result, ThoughtNode)
        assert long_question in result.content or long_question in str(
            result.metadata.get("main_question", "")
        )

    @pytest.mark.asyncio
    async def test_answer_with_empty_stack(self, initialized_method, active_session):
        """Test answering when the question stack is empty."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        # Ensure stack is empty
        initialized_method._question_stack = []

        # Try to answer - should not crash
        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance="answer",
        )

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_none_context(self, initialized_method, active_session):
        """Test execute with None context."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
            context=None,
        )

        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context_dict(self, initialized_method, active_session):
        """Test execute with empty context dictionary."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
            context={},
        )

        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_continue_with_none_guidance(
        self, initialized_method, active_session
    ):
        """Test continue_reasoning with None guidance."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        result = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
            guidance=None,
        )

        assert result.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_simple_yes_no_question(self, initialized_method, active_session):
        """Test handling a simple yes/no question."""
        result = await initialized_method.execute(
            session=active_session,
            input_text="Is water wet?",
        )

        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_compound_question(self, initialized_method, active_session):
        """Test handling a compound question with multiple parts."""
        compound = "How does photosynthesis work and what are its byproducts?"
        result = await initialized_method.execute(
            session=active_session,
            input_text=compound,
        )

        assert isinstance(result, ThoughtNode)
        assert compound in result.content

    @pytest.mark.asyncio
    async def test_multiple_continue_calls(self, initialized_method, active_session):
        """Test multiple sequential continue_reasoning calls."""
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How does the solar system work?",
        )

        # Multiple continues
        thoughts = [first_thought]
        for i in range(5):
            next_thought = await initialized_method.continue_reasoning(
                session=active_session,
                previous_thought=thoughts[-1],
            )
            thoughts.append(next_thought)

        # Verify step numbers increase
        for i, thought in enumerate(thoughts):
            assert thought.step_number == i + 1

    @pytest.mark.asyncio
    async def test_context_preserved_across_continues(
        self, initialized_method, active_session
    ):
        """Test that context from execute is available in metadata."""
        context = {"domain": "biology", "level": "advanced"}
        first_thought = await initialized_method.execute(
            session=active_session,
            input_text="How do cells divide?",
            context=context,
        )

        # Context should be in the first thought's metadata
        assert first_thought.metadata["context"] == context


# ============================================================================
# Integration Tests
# ============================================================================


class TestSelfAskIntegration:
    """Test suite for end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_reasoning_flow(self, initialized_method, active_session):
        """Test a complete reasoning flow from start to finish."""
        # Execute initial analysis
        initial = await initialized_method.execute(
            session=active_session,
            input_text="How do plants convert sunlight into energy?",
        )
        assert initial.type == ThoughtType.INITIAL

        # Generate first sub-question
        q1 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=initial,
            guidance="ask question",
        )
        assert q1.type == ThoughtType.HYPOTHESIS

        # Answer first question
        a1 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=q1,
            guidance="answer",
        )
        assert a1.type == ThoughtType.VERIFICATION

        # Generate second sub-question
        q2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=a1,
            guidance="ask another question",
        )
        assert q2.type == ThoughtType.HYPOTHESIS

        # Answer second question
        a2 = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=q2,
            guidance="answer",
        )
        assert a2.type == ThoughtType.VERIFICATION

        # Synthesize answers
        synthesis = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=a2,
            guidance="synthesize",
        )
        assert synthesis.type == ThoughtType.SYNTHESIS

        # Generate conclusion
        conclusion = await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=synthesis,
            guidance="conclude",
        )
        assert conclusion.type == ThoughtType.CONCLUSION

        # Verify session state
        assert active_session.thought_count == 7  # initial + q1 + a1 + q2 + a2 + syn + con

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self, initialized_method, active_session):
        """Test that session metrics are properly updated."""
        await initialized_method.execute(
            session=active_session,
            input_text="Test question",
        )

        initial_metrics = active_session.metrics.total_thoughts
        assert initial_metrics == 1

        # Continue reasoning
        first_thought = list(active_session.graph.nodes.values())[0]
        await initialized_method.continue_reasoning(
            session=active_session,
            previous_thought=first_thought,
        )

        assert active_session.metrics.total_thoughts == 2

    @pytest.mark.asyncio
    async def test_multiple_executions_in_same_session(
        self, initialized_method, active_session
    ):
        """Test executing multiple times resets state correctly."""
        # First execution
        result1 = await initialized_method.execute(
            session=active_session,
            input_text="First question",
        )
        assert result1.step_number == 1

        # Second execution - should reset
        result2 = await initialized_method.execute(
            session=active_session,
            input_text="Second question",
        )
        assert result2.step_number == 1

        # Both should be in the session
        assert active_session.thought_count == 2
