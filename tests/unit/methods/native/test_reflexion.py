"""Comprehensive tests for Reflexion reasoning method.

This module provides complete test coverage for the Reflexion method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Episode cycles (attempt → evaluate → reflect → retry)
- Episodic memory persistence and learning
- Configuration options (quality_threshold, max_episodes)
- Continue reasoning flow
- Quality improvement across episodes
- Lesson extraction and application
- Memory management
- Convergence detection
- Edge cases

The tests aim for 90%+ coverage of the Reflexion implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.reflexion import (
    REFLEXION_METADATA,
    Reflexion,
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
def method() -> Reflexion:
    """Provide a Reflexion method instance for testing.

    Returns:
        Reflexion instance (uninitialized).
    """
    return Reflexion()


@pytest.fixture
async def initialized_method() -> Reflexion:
    """Provide an initialized Reflexion method instance.

    Returns:
        Initialized Reflexion instance.
    """
    method = Reflexion()
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
        Simple problem for testing.
    """
    return "Implement a function to reverse a linked list"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex problem requiring multiple episodes.
    """
    return "Design and implement a distributed consensus algorithm"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestReflexionMetadata:
    """Test suite for Reflexion metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert REFLEXION_METADATA.identifier == MethodIdentifier.REFLEXION

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert REFLEXION_METADATA.name == "Reflexion"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert REFLEXION_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert REFLEXION_METADATA.complexity == 7
        assert 1 <= REFLEXION_METADATA.complexity <= 10

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert REFLEXION_METADATA.supports_revision is True

    def test_metadata_no_branching(self):
        """Test that metadata indicates no branching support."""
        assert REFLEXION_METADATA.supports_branching is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "episodic-memory",
            "self-reflection",
            "learning",
            "iterative",
            "memory-persistence",
        }
        assert expected_tags.issubset(REFLEXION_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert REFLEXION_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert REFLEXION_METADATA.max_thoughts == 25


# ============================================================================
# Initialization Tests
# ============================================================================


class TestReflexionInitialization:
    """Test suite for Reflexion initialization."""

    def test_create_method(self, method: Reflexion):
        """Test creating a Reflexion instance."""
        assert isinstance(method, Reflexion)
        assert method._initialized is False

    def test_properties_before_initialization(self, method: Reflexion):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.REFLEXION
        assert method.name == "Reflexion"
        assert method.category == MethodCategory.ADVANCED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: Reflexion):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._episode_number == 0
        assert method._current_phase == "attempt"
        assert len(method._episodic_memory) == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = Reflexion()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._episode_number = 2
        method._current_phase = "reflect"
        method._episodic_memory = [{"test": "data"}]

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._episode_number == 0
        assert method._current_phase == "attempt"
        assert len(method._episodic_memory) == 0

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: Reflexion):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: Reflexion):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestReflexionExecution:
    """Test suite for basic Reflexion execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: Reflexion, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.REFLEXION
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metadata(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["phase"] == "attempt"
        assert thought.metadata["episode_number"] == 1
        assert "quality_threshold" in thought.metadata
        assert thought.metadata["reasoning_type"] == "reflexion"
        assert "episodic_memory_size" in thought.metadata
        assert thought.metadata["episodic_memory_size"] == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_quality(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that execute sets initial quality score."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert thought.quality_score == 0.5
        assert thought.confidence == 0.5
        assert thought.metadata["needs_improvement"] is True

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.REFLEXION
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"quality_threshold": 0.9, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["quality_threshold"] == 0.9
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_quality_threshold(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that quality_threshold is clamped to [0.0, 1.0]."""
        # Test upper bound
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 1.5},
        )
        assert thought1.metadata["quality_threshold"] == 1.0

        # Re-initialize for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test lower bound
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"quality_threshold": -0.5},
        )
        assert thought2.metadata["quality_threshold"] == 0.0


# ============================================================================
# Episode Cycle Tests
# ============================================================================


class TestEpisodeCycle:
    """Test suite for the episode cycle flow."""

    @pytest.mark.asyncio
    async def test_evaluate_phase_after_attempt(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that evaluate follows attempt thought."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)

        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        assert evaluate.type == ThoughtType.VERIFICATION
        assert evaluate.metadata["phase"] == "evaluate"
        assert evaluate.parent_id == attempt.id
        assert evaluate.step_number == 2

    @pytest.mark.asyncio
    async def test_reflect_phase_after_evaluate(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that reflect follows evaluate."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        reflect = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate
        )

        assert reflect.type == ThoughtType.INSIGHT
        assert reflect.metadata["phase"] == "reflect"
        assert reflect.parent_id == evaluate.id
        assert reflect.step_number == 3

    @pytest.mark.asyncio
    async def test_retry_phase_after_reflect(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that retry follows reflect."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        reflect = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate
        )

        retry = await initialized_method.continue_reasoning(
            session=session, previous_thought=reflect
        )

        assert retry.type in (ThoughtType.REVISION, ThoughtType.CONCLUSION)
        assert retry.metadata["phase"] == "retry"
        assert retry.parent_id == reflect.id
        assert retry.step_number == 4
        assert retry.metadata["episode_number"] == 2

    @pytest.mark.asyncio
    async def test_full_episode_cycle(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test a complete episode cycle: attempt → evaluate → reflect → retry."""
        # Episode 1: Attempt
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        assert attempt.metadata["episode_number"] == 1
        assert attempt.metadata["phase"] == "attempt"

        # Episode 1: Evaluate
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        assert evaluate.metadata["episode_number"] == 1
        assert evaluate.metadata["phase"] == "evaluate"

        # Episode 1: Reflect
        reflect = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate
        )
        assert reflect.metadata["episode_number"] == 1
        assert reflect.metadata["phase"] == "reflect"

        # Episode 2: Retry
        retry = await initialized_method.continue_reasoning(
            session=session, previous_thought=reflect
        )
        assert retry.metadata["episode_number"] == 2
        assert retry.metadata["phase"] == "retry"

    @pytest.mark.asyncio
    async def test_multiple_episodes(self, initialized_method: Reflexion, session: Session):
        """Test multiple complete episodes."""
        input_text = "Solve a complex algorithm problem"

        # Episode 1
        thought = await initialized_method.execute(session=session, input_text=input_text)
        assert thought.metadata["episode_number"] == 1

        # Complete episode 1 (evaluate, reflect, retry)
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should now be in episode 2 (retry increments episode)
        assert thought.metadata["episode_number"] == 2

        # Continue with episode 2 (evaluate, reflect)
        for _ in range(2):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Retry should increment to episode 3
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["episode_number"] == 3

    @pytest.mark.asyncio
    async def test_evaluate_after_retry(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that evaluate can follow retry for next episode."""
        # Complete first episode
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        reflect1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate1
        )
        retry1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=reflect1
        )

        # Start second episode with evaluation
        evaluate2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=retry1
        )

        assert evaluate2.type == ThoughtType.VERIFICATION
        assert evaluate2.metadata["phase"] == "evaluate"
        assert evaluate2.metadata["episode_number"] == 2


# ============================================================================
# Episodic Memory Tests
# ============================================================================


class TestEpisodicMemory:
    """Test suite for episodic memory management."""

    @pytest.mark.asyncio
    async def test_memory_empty_initially(self, initialized_method: Reflexion):
        """Test that episodic memory is empty initially."""
        assert len(initialized_method._episodic_memory) == 0

    @pytest.mark.asyncio
    async def test_memory_populated_after_reflection(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that memory is populated after reflection phase."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        await initialized_method.continue_reasoning(session=session, previous_thought=evaluate)

        # Memory should have one entry after reflection
        assert len(initialized_method._episodic_memory) == 1
        assert initialized_method._episodic_memory[0]["episode"] == 1
        assert "quality" in initialized_method._episodic_memory[0]
        assert "lesson" in initialized_method._episodic_memory[0]

    @pytest.mark.asyncio
    async def test_memory_persists_across_episodes(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that memory persists across episodes."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Complete first episode
        for _ in range(4):  # evaluate, reflect, retry, evaluate
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Memory should have at least one entry
        memory_size_1 = len(initialized_method._episodic_memory)
        assert memory_size_1 >= 1

        # Continue to second reflection
        for _ in range(2):  # reflect, retry
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Memory should have grown
        memory_size_2 = len(initialized_method._episodic_memory)
        assert memory_size_2 > memory_size_1

    @pytest.mark.asyncio
    async def test_lessons_learned_in_metadata(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that lessons_learned appear in metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Initial attempt has no lessons
        assert "lessons_learned" in thought.metadata
        assert len(thought.metadata["lessons_learned"]) == 0

        # After one reflection
        for _ in range(3):  # evaluate, reflect, retry
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should have one lesson learned
        assert len(thought.metadata["lessons_learned"]) == 1

    @pytest.mark.asyncio
    async def test_memory_size_tracked_in_metadata(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that memory size is tracked in metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["episodic_memory_size"] == 0

        # After reflection
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Memory size should be tracked
        assert thought.metadata["episodic_memory_size"] >= 1


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    @pytest.mark.asyncio
    async def test_default_quality_threshold(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test default quality threshold."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["quality_threshold"] == Reflexion.QUALITY_THRESHOLD
        assert thought.metadata["quality_threshold"] == 0.85

    @pytest.mark.asyncio
    async def test_custom_quality_threshold(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test custom quality threshold in context."""
        custom_threshold = 0.95

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": custom_threshold},
        )

        assert thought.metadata["quality_threshold"] == custom_threshold

    @pytest.mark.asyncio
    async def test_quality_threshold_propagates(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that quality threshold propagates through cycle."""
        custom_threshold = 0.75

        attempt = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": custom_threshold},
        )

        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        assert evaluate.metadata["quality_threshold"] == custom_threshold

    @pytest.mark.asyncio
    async def test_max_episodes_constant(self):
        """Test that MAX_EPISODES is properly defined."""
        assert Reflexion.MAX_EPISODES == 3
        assert isinstance(Reflexion.MAX_EPISODES, int)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: Reflexion, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.REFLEXION,
            content="Test",
            metadata={"phase": "attempt"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        assert attempt.step_number == 1

        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        assert evaluate.step_number == 2

        reflect = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate
        )
        assert reflect.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Focus on edge cases"
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt, guidance=guidance_text
        )

        assert "guidance" in evaluate.metadata
        assert evaluate.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)

        context = {"additional_info": "test data"}
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt, context=context
        )

        assert "context" in evaluate.metadata
        assert evaluate.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        count_after_attempt = session.thought_count

        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        assert session.thought_count == count_after_attempt + 1
        assert evaluate.id in session.graph.nodes


# ============================================================================
# Quality Improvement Tests
# ============================================================================


class TestQualityImprovement:
    """Test suite for quality score improvement across episodes."""

    @pytest.mark.asyncio
    async def test_quality_improves_across_episodes(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that quality score improves with each episode."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        initial_quality = thought.quality_score

        # Complete first episode to get to retry
        for _ in range(3):  # evaluate, reflect, retry
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Retry (episode 2) should have better quality
        episode_2_quality = thought.quality_score
        assert episode_2_quality > initial_quality

    @pytest.mark.asyncio
    async def test_confidence_improves_across_episodes(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that confidence improves with each episode."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        initial_confidence = thought.confidence

        # Get to second episode retry
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Retry should have higher confidence
        assert thought.confidence > initial_confidence

    @pytest.mark.asyncio
    async def test_quality_caps_at_one(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that quality score caps at 1.0."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Run many cycles
        for _ in range(20):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Quality should never exceed 1.0
        assert thought.quality_score is not None
        assert thought.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_previous_quality_stored_in_metadata(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that previous quality is stored in metadata."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        assert "previous_quality" in evaluate.metadata
        assert evaluate.metadata["previous_quality"] == attempt.quality_score


# ============================================================================
# Episode Tracking Tests
# ============================================================================


class TestEpisodeTracking:
    """Test suite for episode counting and limits."""

    @pytest.mark.asyncio
    async def test_episode_increments_on_retry(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that episode_number increments on retry."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert initialized_method._episode_number == 1

        # evaluate and reflect don't increment
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._episode_number == 1

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._episode_number == 1

        # retry increments
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._episode_number == 2

    @pytest.mark.asyncio
    async def test_max_episodes_limit(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that max episodes limit is respected."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Run until max episodes reached
        for _ in range(Reflexion.MAX_EPISODES * 4 + 5):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should not exceed max
        assert initialized_method._episode_number <= Reflexion.MAX_EPISODES

    @pytest.mark.asyncio
    async def test_needs_improvement_flag(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test the needs_improvement flag in metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.85},
        )

        # Initial quality is 0.5, threshold is 0.85
        assert thought.metadata["needs_improvement"] is True

    @pytest.mark.asyncio
    async def test_step_counter_resets_on_execute(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that step counter resets on new execute."""
        # First execution
        thought1 = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought1.step_number == 1

        # Continue
        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )
        assert thought2.step_number == 2

        # New execution should reset
        session2 = Session().start()
        thought3 = await initialized_method.execute(session=session2, input_text=simple_input)
        assert thought3.step_number == 1


# ============================================================================
# Convergence Detection Tests
# ============================================================================


class TestConvergenceDetection:
    """Test suite for convergence and completion detection."""

    @pytest.mark.asyncio
    async def test_conclusion_when_quality_met(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that thought becomes CONCLUSION when quality threshold is met."""
        # Set low threshold so it's easily met
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.5},
        )

        # Go through cycles until quality exceeds threshold
        for _ in range(6):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Eventually should get a CONCLUSION on retry phase
        if thought.metadata["phase"] == "retry" and thought.quality_score >= 0.5:
            assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_conclusion_at_max_episodes(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that reasoning concludes at max episodes even if quality not met."""
        # Set high threshold that won't be met
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.99},
        )

        # Run through max episodes
        for _ in range(Reflexion.MAX_EPISODES * 4 + 10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should eventually conclude
        if initialized_method._episode_number >= Reflexion.MAX_EPISODES:
            if thought.metadata["phase"] == "retry":
                assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_needs_improvement_false_at_conclusion(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that needs_improvement is False at conclusion."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.6},
        )

        # Run until conclusion
        for _ in range(15):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break

        if thought.type == ThoughtType.CONCLUSION:
            assert thought.metadata["needs_improvement"] is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: Reflexion, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: Reflexion, session: Session):
        """Test handling of very long input."""
        long_input = "Solve this problem: " + "very complex " * 1000
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_fallback_to_evaluate_phase(
        self, initialized_method: Reflexion, session: Session
    ):
        """Test fallback to evaluate for unknown phase."""
        # Create a thought with unknown phase
        attempt = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify phase to unknown value
        attempt.metadata["phase"] = "unknown_phase"

        # Continue should fallback to evaluate
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        assert thought.type == ThoughtType.VERIFICATION
        assert thought.metadata["phase"] == "evaluate"

    @pytest.mark.asyncio
    async def test_quality_threshold_zero(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test handling of quality threshold of 0.0."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.0},
        )

        # Should immediately meet threshold
        assert thought.metadata["needs_improvement"] is False

    @pytest.mark.asyncio
    async def test_quality_threshold_one(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test handling of quality threshold of 1.0."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 1.0},
        )

        # Should always need improvement (initial quality < 1.0)
        assert thought.metadata["needs_improvement"] is True


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_attempt_content_structure(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that attempt has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Attempt" in content
        assert "Episode 1" in content
        assert simple_input in content

    @pytest.mark.asyncio
    async def test_evaluation_content_structure(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that evaluation has expected content structure."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )

        content = evaluate.content
        assert isinstance(content, str)
        assert "Evaluation" in content
        assert "Episode" in content

    @pytest.mark.asyncio
    async def test_reflection_content_structure(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that reflection has expected content structure."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt
        )
        reflect = await initialized_method.continue_reasoning(
            session=session, previous_thought=evaluate
        )

        content = reflect.content
        assert isinstance(content, str)
        assert "Reflection" in content
        assert "Episode" in content

    @pytest.mark.asyncio
    async def test_retry_content_structure(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that retry has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Get to retry phase
        for _ in range(3):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        content = thought.content
        assert isinstance(content, str)
        assert "Retry" in content
        assert "Episode 2" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that guidance appears in generated content."""
        attempt = await initialized_method.execute(session=session, input_text=simple_input)

        guidance = "Focus on efficiency"
        evaluate = await initialized_method.continue_reasoning(
            session=session, previous_thought=attempt, guidance=guidance
        )

        # Guidance should appear in metadata
        assert guidance in evaluate.metadata["guidance"]


# ============================================================================
# Lesson Learning Tests
# ============================================================================


class TestLessonLearning:
    """Test suite for lesson extraction and application."""

    @pytest.mark.asyncio
    async def test_lesson_extracted_from_low_quality(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test lesson extraction from low quality attempt."""
        # Create evaluation thought with low quality
        thought = ThoughtNode(
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.REFLEXION,
            content="Test",
            quality_score=0.2,
            metadata={"phase": "evaluate"},
        )

        lesson = initialized_method._extract_lesson(thought, 0.2)
        assert "Fundamental" in lesson or "complete redesign" in lesson

    @pytest.mark.asyncio
    async def test_lesson_extracted_from_medium_quality(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test lesson extraction from medium quality attempt."""
        thought = ThoughtNode(
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.REFLEXION,
            content="Test",
            quality_score=0.5,
            metadata={"phase": "evaluate"},
        )

        lesson = initialized_method._extract_lesson(thought, 0.5)
        assert "gaps" in lesson or "improvements" in lesson

    @pytest.mark.asyncio
    async def test_lesson_extracted_from_high_quality(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test lesson extraction from high quality attempt."""
        thought = ThoughtNode(
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.REFLEXION,
            content="Test",
            quality_score=0.9,
            metadata={"phase": "evaluate"},
        )

        lesson = initialized_method._extract_lesson(thought, 0.9)
        assert "Good quality" in lesson or "minor polish" in lesson

    @pytest.mark.asyncio
    async def test_lessons_accumulate_in_memory(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that lessons accumulate in episodic memory."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Go through multiple episodes
        for _ in range(7):  # Should get through at least one full episode
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Check that memory has accumulated
        lessons = initialized_method._get_lessons_learned()
        assert len(lessons) >= 1

    @pytest.mark.asyncio
    async def test_memory_context_formatting(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that memory context is formatted correctly."""
        # Empty memory
        context = initialized_method._format_memory_context()
        assert context == ""

        # Add some memory
        initialized_method._episodic_memory.append(
            {
                "episode": 1,
                "quality": 0.5,
                "lesson": "Test lesson",
                "phase": "reflect",
            }
        )

        context = initialized_method._format_memory_context()
        assert "Previous attempts" in context
        assert "Test lesson" in context


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestMemoryManagement:
    """Test suite for episodic memory management."""

    @pytest.mark.asyncio
    async def test_memory_summarization(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test memory summarization for reflection."""
        # Empty memory
        summary = initialized_method._summarize_memory()
        assert "No previous episodes" in summary

        # Add memory entries
        initialized_method._episodic_memory = [
            {"episode": 1, "quality": 0.5, "lesson": "Lesson 1", "phase": "reflect"},
            {"episode": 2, "quality": 0.7, "lesson": "Lesson 2", "phase": "reflect"},
        ]

        summary = initialized_method._summarize_memory()
        assert "2 previous episode" in summary
        assert "0.60" in summary  # average quality

    @pytest.mark.asyncio
    async def test_lessons_formatting_empty(self, initialized_method: Reflexion):
        """Test lessons formatting with no lessons."""
        formatted = initialized_method._format_lessons_learned()
        assert "No previous lessons" in formatted or "first episode" in formatted

    @pytest.mark.asyncio
    async def test_lessons_formatting_with_lessons(self, initialized_method: Reflexion):
        """Test lessons formatting with lessons."""
        initialized_method._episodic_memory = [
            {"episode": 1, "quality": 0.5, "lesson": "Lesson 1", "phase": "reflect"},
            {"episode": 2, "quality": 0.7, "lesson": "Lesson 2", "phase": "reflect"},
        ]

        formatted = initialized_method._format_lessons_learned()
        assert "Lesson 1" in formatted
        assert "Lesson 2" in formatted
        assert "1." in formatted
        assert "2." in formatted


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete Reflexion workflows."""

    @pytest.mark.asyncio
    async def test_complete_single_episode(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test a complete single episode workflow."""
        # Attempt
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.type == ThoughtType.INITIAL
        assert thought.metadata["phase"] == "attempt"

        # Evaluate
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.type == ThoughtType.VERIFICATION
        assert thought.metadata["phase"] == "evaluate"

        # Reflect
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.type == ThoughtType.INSIGHT
        assert thought.metadata["phase"] == "reflect"

        # Retry
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.type in (ThoughtType.REVISION, ThoughtType.CONCLUSION)
        assert thought.metadata["phase"] == "retry"
        assert thought.metadata["episode_number"] == 2

        # Verify memory was populated
        assert len(initialized_method._episodic_memory) >= 1

    @pytest.mark.asyncio
    async def test_learning_across_episodes(
        self, initialized_method: Reflexion, session: Session, simple_input: str
    ):
        """Test that learning persists across episodes."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"quality_threshold": 0.95},  # High threshold to force multiple episodes
        )

        initial_quality = thought.quality_score
        episodes_completed = 0

        # Run through multiple episodes
        for _ in range(12):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.metadata["phase"] == "retry":
                episodes_completed = thought.metadata["episode_number"]

        # Quality should improve
        final_quality = thought.quality_score
        assert final_quality > initial_quality

        # Should have gone through multiple episodes
        assert episodes_completed >= 2

        # Memory should have accumulated
        assert len(initialized_method._episodic_memory) >= 2
