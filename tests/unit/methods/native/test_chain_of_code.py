"""Unit tests for ChainOfCode reasoning method.

This module provides comprehensive tests for the ChainOfCode method
implementation, covering initialization, execution, code generation,
LM emulation, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.chain_of_code import (
    CHAIN_OF_CODE_METADATA,
    ChainOfCode,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def coc_method() -> ChainOfCode:
    """Create a ChainOfCode method instance for testing.

    Returns:
        A fresh ChainOfCode instance
    """
    return ChainOfCode()


@pytest.fixture
def coc_no_elicitation() -> ChainOfCode:
    """Create a ChainOfCode method with elicitation disabled.

    Returns:
        A ChainOfCode instance with elicitation disabled
    """
    return ChainOfCode(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> ChainOfCode:
    """Create an initialized ChainOfCode method instance.

    Returns:
        An initialized ChainOfCode instance
    """
    method = ChainOfCode()
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
    return "Given x=5, y=3, and z=2, compute x*y + z"


class TestChainOfCodeMetadata:
    """Tests for CHAIN_OF_CODE_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert CHAIN_OF_CODE_METADATA.identifier == MethodIdentifier.CHAIN_OF_CODE

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert CHAIN_OF_CODE_METADATA.name == "Chain of Code"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = CHAIN_OF_CODE_METADATA.description.lower()
        assert "code" in desc
        assert "emulation" in desc or "semantic" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in SPECIALIZED category."""
        assert CHAIN_OF_CODE_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity."""
        assert CHAIN_OF_CODE_METADATA.complexity == 7
        assert 1 <= CHAIN_OF_CODE_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that CoC doesn't support branching."""
        assert CHAIN_OF_CODE_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that CoC supports revision."""
        assert CHAIN_OF_CODE_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that CoC doesn't require context."""
        assert CHAIN_OF_CODE_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert CHAIN_OF_CODE_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert CHAIN_OF_CODE_METADATA.max_thoughts == 8

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "code" in CHAIN_OF_CODE_METADATA.tags
        assert "emulation" in CHAIN_OF_CODE_METADATA.tags
        assert "semantic" in CHAIN_OF_CODE_METADATA.tags
        assert "hybrid" in CHAIN_OF_CODE_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(CHAIN_OF_CODE_METADATA.best_for).lower()
        assert "semantic" in best_for_text or "code" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(CHAIN_OF_CODE_METADATA.not_recommended_for).lower()
        assert "simple" in not_recommended or "factual" in not_recommended


class TestChainOfCodeInitialization:
    """Tests for ChainOfCode method initialization."""

    def test_create_instance(self, coc_method: ChainOfCode) -> None:
        """Test that we can create a ChainOfCode instance."""
        assert isinstance(coc_method, ChainOfCode)

    def test_initial_state(self, coc_method: ChainOfCode) -> None:
        """Test that initial state is correct before initialization."""
        assert coc_method._initialized is False
        assert coc_method._step_counter == 0
        assert coc_method._current_phase == "understand"
        assert coc_method._code_segments == []
        assert coc_method._semantic_segments == []
        assert coc_method._execution_trace == []
        assert coc_method._final_result is None

    def test_default_elicitation_enabled(self, coc_method: ChainOfCode) -> None:
        """Test that elicitation is enabled by default."""
        assert coc_method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, coc_no_elicitation: ChainOfCode) -> None:
        """Test that elicitation can be disabled."""
        assert coc_no_elicitation.enable_elicitation is False

    async def test_initialize(self, coc_method: ChainOfCode) -> None:
        """Test that initialize sets up the method correctly."""
        await coc_method.initialize()
        assert coc_method._initialized is True
        assert coc_method._step_counter == 0
        assert coc_method._current_phase == "understand"
        assert coc_method._code_segments == []
        assert coc_method._semantic_segments == []
        assert coc_method._execution_trace == []
        assert coc_method._final_result is None

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = ChainOfCode()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._code_segments = [{"id": 1}]
        method._semantic_segments = [{"id": 1}]
        method._execution_trace = [{"step": 1}]
        method._final_result = "Result"

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "understand"
        assert method._code_segments == []
        assert method._semantic_segments == []
        assert method._execution_trace == []
        assert method._final_result is None

    async def test_health_check_before_init(self, coc_method: ChainOfCode) -> None:
        """Test health_check returns False before initialization."""
        health = await coc_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: ChainOfCode) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestChainOfCodeProperties:
    """Tests for ChainOfCode method properties."""

    def test_identifier_property(self, coc_method: ChainOfCode) -> None:
        """Test that identifier property returns correct value."""
        assert coc_method.identifier == MethodIdentifier.CHAIN_OF_CODE

    def test_name_property(self, coc_method: ChainOfCode) -> None:
        """Test that name property returns correct value."""
        assert coc_method.name == "Chain of Code"

    def test_description_property(self, coc_method: ChainOfCode) -> None:
        """Test that description property returns correct value."""
        assert coc_method.description == CHAIN_OF_CODE_METADATA.description

    def test_category_property(self, coc_method: ChainOfCode) -> None:
        """Test that category property returns correct value."""
        assert coc_method.category == MethodCategory.SPECIALIZED


class TestChainOfCodeExecution:
    """Tests for basic execution of ChainOfCode reasoning."""

    async def test_execute_without_initialization_fails(
        self, coc_method: ChainOfCode, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await coc_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates understand phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.CHAIN_OF_CODE
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_understand(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to understand."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "understand"
        assert thought.metadata["phase"] == "understand"

    async def test_execute_populates_segments(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute populates code and semantic segments."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._code_segments) > 0
        assert len(initialized_method._semantic_segments) > 0

    async def test_execute_adds_to_session(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.CHAIN_OF_CODE

    async def test_execute_content_format(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert "Code" in thought.content or "code" in thought.content.lower()

    async def test_execute_confidence_level(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for understand phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.6

    async def test_execute_metadata(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "understand"
        assert "code_segments" in thought.metadata
        assert "semantic_segments" in thought.metadata


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, coc_method: ChainOfCode, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "understand"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await coc_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_understand_to_generate(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from understand to generate."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )

        assert initialized_method._current_phase == "generate"
        assert generate_thought.metadata["phase"] == "generate"
        assert generate_thought.type == ThoughtType.REASONING

    async def test_phase_transition_generate_to_emulate(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from generate to emulate."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )

        emulate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert initialized_method._current_phase == "emulate"
        assert emulate_thought.metadata["phase"] == "emulate"
        assert emulate_thought.type == ThoughtType.REASONING

    async def test_execution_trace_populated(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execution trace is populated during emulate phase."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        assert len(initialized_method._execution_trace) > 0

    async def test_execution_trace_types(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that execution trace contains both code and lm_emulate types."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        trace_types = [t["type"] for t in initialized_method._execution_trace]
        assert "code" in trace_types
        assert "lm_emulate" in trace_types

    async def test_phase_transition_emulate_to_synthesize(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from emulate to synthesize."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )
        emulate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )

        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=emulate_thought,
        )

        assert initialized_method._current_phase == "synthesize"
        assert synthesize_thought.metadata["phase"] == "synthesize"
        assert synthesize_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_synthesize_to_conclude(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from synthesize to conclude."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )
        emulate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=emulate_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=synthesize_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert understand_thought.step_number == 1

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )
        assert generate_thought.step_number == 2

    async def test_parent_id_set_correctly(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )

        assert generate_thought.parent_id == understand_thought.id


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        understand_confidence = understand_thought.confidence

        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )
        generate_confidence = generate_thought.confidence

        emulate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        emulate_confidence = emulate_thought.confidence

        # Confidence should generally increase
        assert understand_confidence < generate_confidence
        assert generate_confidence < emulate_confidence


class TestEdgeCases:
    """Tests for edge cases in ChainOfCode reasoning."""

    async def test_empty_query(self, initialized_method: ChainOfCode, session: Session) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(self, initialized_method: ChainOfCode, session: Session) -> None:
        """Test handling of very long query."""
        long_query = "Compute: " + "x + y + " * 100 + "z"
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None

    async def test_special_characters(
        self, initialized_method: ChainOfCode, session: Session
    ) -> None:
        """Test handling of special characters."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_query(self, initialized_method: ChainOfCode, session: Session) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="计算 x=5, y=3 的和",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Understand
        understand_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert understand_thought.type == ThoughtType.INITIAL
        assert understand_thought.metadata["phase"] == "understand"

        # Phase 2: Generate
        generate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=understand_thought,
        )
        assert generate_thought.type == ThoughtType.REASONING
        assert generate_thought.metadata["phase"] == "generate"

        # Phase 3: Emulate
        emulate_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=generate_thought,
        )
        assert emulate_thought.type == ThoughtType.REASONING
        assert emulate_thought.metadata["phase"] == "emulate"

        # Phase 4: Synthesize
        synthesize_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=emulate_thought,
        )
        assert synthesize_thought.type == ThoughtType.SYNTHESIS
        assert synthesize_thought.metadata["phase"] == "synthesize"

        # Phase 5: Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=synthesize_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 5
        assert session.current_method == MethodIdentifier.CHAIN_OF_CODE


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.CHAIN_OF_CODE)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        coc_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_CODE)
        assert len(coc_thoughts) > 0


class TestElicitationBehavior:
    """Tests for elicitation-related behavior."""

    async def test_elicitation_disabled_skips_interactions(
        self, coc_no_elicitation: ChainOfCode, session: Session, sample_problem: str
    ) -> None:
        """Test that disabled elicitation skips user interactions."""
        await coc_no_elicitation.initialize()

        # Execute should work without any elicitation
        thought = await coc_no_elicitation.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
