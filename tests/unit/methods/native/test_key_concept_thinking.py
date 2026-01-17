"""Unit tests for Key-Concept Thinking (KCT) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Extract phase (concept identification)
- Define phase (concept definitions)
- Apply phase (concept application)
- Solve phase (solution derivation)
- Conclude phase (final answer)
- LLM sampling with fallbacks
- Concept management
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.key_concept_thinking import (
    KEY_CONCEPT_THINKING_METADATA,
    KeyConceptThinking,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestKeyConceptThinkingMetadata:
    """Tests for Key-Concept Thinking metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert KEY_CONCEPT_THINKING_METADATA.identifier == MethodIdentifier.KEY_CONCEPT_THINKING

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert KEY_CONCEPT_THINKING_METADATA.name == "Key-Concept Thinking"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert KEY_CONCEPT_THINKING_METADATA.description is not None
        assert "concept" in KEY_CONCEPT_THINKING_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert KEY_CONCEPT_THINKING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {
            "concept-extraction",
            "domain-knowledge",
            "structured-reasoning",
            "knowledge-grounding",
        }
        assert expected_tags.issubset(KEY_CONCEPT_THINKING_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert KEY_CONCEPT_THINKING_METADATA.complexity == 6

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert KEY_CONCEPT_THINKING_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert KEY_CONCEPT_THINKING_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert KEY_CONCEPT_THINKING_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert KEY_CONCEPT_THINKING_METADATA.max_thoughts == 8

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "domain-specific problems" in KEY_CONCEPT_THINKING_METADATA.best_for


class TestKeyConceptThinking:
    """Test suite for Key-Concept Thinking reasoning method."""

    @pytest.fixture
    def method(self) -> KeyConceptThinking:
        """Create method instance."""
        return KeyConceptThinking()

    @pytest.fixture
    async def initialized_method(self) -> KeyConceptThinking:
        """Create an initialized method instance."""
        method = KeyConceptThinking()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> MagicMock:
        """Create a mock session for testing."""
        mock_sess = MagicMock(spec=Session)
        mock_sess.current_method = None
        mock_sess.thought_count = 0
        mock_sess._thoughts = []

        def add_thought(thought: ThoughtNode) -> None:
            mock_sess._thoughts.append(thought)
            mock_sess.thought_count = len(mock_sess._thoughts)

        mock_sess.add_thought = add_thought

        return mock_sess

    @pytest.fixture
    def sample_problem(self) -> str:
        """Provide a sample problem for testing."""
        return "Explain the difference between supervised and unsupervised learning"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = (
            "Key concepts:\n1. Supervised learning\n2. Unsupervised learning\n3. Training data"
        )
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: KeyConceptThinking) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, KeyConceptThinking)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "extract"
        assert method._extracted_concepts == []
        assert method._concept_definitions == {}
        assert method._use_sampling is True
        assert method._execution_context is None

    def test_max_concepts_constant(self, method: KeyConceptThinking) -> None:
        """Test that MAX_CONCEPTS constant is defined."""
        assert KeyConceptThinking.MAX_CONCEPTS == 5

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: KeyConceptThinking) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "extract"
        assert method._extracted_concepts == []
        assert method._concept_definitions == {}

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: KeyConceptThinking) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._extracted_concepts = ["Concept 1", "Concept 2"]
        initialized_method._concept_definitions = {"Concept 1": "Definition 1"}

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "extract"
        assert initialized_method._extracted_concepts == []
        assert initialized_method._concept_definitions == {}

    # === Property Tests ===

    def test_identifier_property(self, method: KeyConceptThinking) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.KEY_CONCEPT_THINKING

    def test_name_property(self, method: KeyConceptThinking) -> None:
        """Test name property returns correct value."""
        assert method.name == "Key-Concept Thinking"

    def test_description_property(self, method: KeyConceptThinking) -> None:
        """Test description property returns correct value."""
        assert method.description == KEY_CONCEPT_THINKING_METADATA.description

    def test_category_property(self, method: KeyConceptThinking) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: KeyConceptThinking) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: KeyConceptThinking) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Extract Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: KeyConceptThinking, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.KEY_CONCEPT_THINKING
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "extract"

    @pytest.mark.asyncio
    async def test_execute_extracts_concepts(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() extracts concepts."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._extracted_concepts) > 0

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.KEY_CONCEPT_THINKING

    @pytest.mark.asyncio
    async def test_execute_stores_metadata(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() stores proper metadata."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought.metadata["input"] == sample_problem
        assert thought.metadata["reasoning_type"] == "key_concept_thinking"
        assert "concepts" in thought.metadata

    @pytest.mark.asyncio
    async def test_execute_content_includes_extraction(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes extraction info."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Key Concept Extraction" in thought.content
        assert sample_problem in thought.content

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: KeyConceptThinking, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "extract"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Define Phase Tests ===

    @pytest.mark.asyncio
    async def test_define_phase(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that define phase creates definitions."""
        extract_thought = await initialized_method.execute(session, sample_problem)
        define_thought = await initialized_method.continue_reasoning(session, extract_thought)

        assert define_thought.metadata["phase"] == "define"
        assert define_thought.type == ThoughtType.REASONING
        assert "Concept Definitions" in define_thought.content

    @pytest.mark.asyncio
    async def test_define_stores_definitions(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that define phase stores definitions."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        assert len(initialized_method._concept_definitions) > 0

    # === Apply Phase Tests ===

    @pytest.mark.asyncio
    async def test_apply_phase(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that apply phase structures the reasoning."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply

        assert thought.metadata["phase"] == "apply"
        assert thought.type == ThoughtType.SYNTHESIS
        assert "Applying Concepts" in thought.content

    @pytest.mark.asyncio
    async def test_apply_includes_framework(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that apply phase includes concept framework."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply

        assert "Framework" in thought.content or "Problem →" in thought.content

    # === Solve Phase Tests ===

    @pytest.mark.asyncio
    async def test_solve_phase(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that solve phase derives solution."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply
        thought = await initialized_method.continue_reasoning(session, thought)  # solve

        assert thought.metadata["phase"] == "solve"
        assert thought.type == ThoughtType.REASONING
        assert "Solution" in thought.content or "Deriving" in thought.content

    @pytest.mark.asyncio
    async def test_solve_includes_reasoning_chain(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that solve phase includes reasoning chain."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply
        thought = await initialized_method.continue_reasoning(session, thought)  # solve

        assert "Reasoning Chain" in thought.content or "inference" in thought.content.lower()

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply
        thought = await initialized_method.continue_reasoning(session, thought)  # solve
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Final Answer" in thought.content

    @pytest.mark.asyncio
    async def test_conclude_includes_summary(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude includes summary."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # define
        thought = await initialized_method.continue_reasoning(session, thought)  # apply
        thought = await initialized_method.continue_reasoning(session, thought)  # solve
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert "Summary" in thought.content
        assert "Concepts extracted" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "extract"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "define"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "apply"
        assert thought.type == ThoughtType.SYNTHESIS

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "solve"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sampling_fallback_on_timeout_error(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling times out."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=TimeoutError("LLM timeout"))

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=failing_ctx
        )

        # Should use fallback content for expected errors
        assert "Key Concept Extraction" in thought.content

    @pytest.mark.asyncio
    async def test_sampling_raises_on_unexpected_error(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that unexpected errors are re-raised."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        # Unexpected errors should be re-raised, not caught
        with pytest.raises(RuntimeError, match="Unexpected error"):
            await initialized_method.execute(
                session, sample_problem, execution_context=failing_ctx
            )

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "Key Concept Extraction" in thought.content

    @pytest.mark.asyncio
    async def test_definitions_with_sampling(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that definitions use LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        extract_thought = session._thoughts[0]

        await initialized_method.continue_reasoning(session, extract_thought)

        # Should have called sample again for definitions
        assert mock_execution_context.sample.call_count == 2

    # === Helper Method Tests ===

    def test_generate_extraction(self, initialized_method: KeyConceptThinking) -> None:
        """Test _generate_extraction heuristic."""
        initialized_method._step_counter = 1
        content = initialized_method._generate_extraction("Test problem", None)

        assert "Key Concept Extraction" in content
        assert "Test problem" in content
        assert len(initialized_method._extracted_concepts) == 3

    def test_generate_definitions(self, initialized_method: KeyConceptThinking) -> None:
        """Test _generate_definitions heuristic."""
        initialized_method._step_counter = 2
        initialized_method._extracted_concepts = ["Concept 1", "Concept 2"]
        content = initialized_method._generate_definitions(None, None)

        assert "Concept Definitions" in content
        assert len(initialized_method._concept_definitions) > 0

    def test_generate_application(self, initialized_method: KeyConceptThinking) -> None:
        """Test _generate_application heuristic."""
        initialized_method._step_counter = 3
        content = initialized_method._generate_application(None, None)

        assert "Applying Concepts" in content

    def test_generate_solution(self, initialized_method: KeyConceptThinking) -> None:
        """Test _generate_solution heuristic."""
        initialized_method._step_counter = 4
        content = initialized_method._generate_solution(None, None)

        assert "Deriving Solution" in content
        assert "Reasoning Chain" in content

    def test_generate_conclusion(self, initialized_method: KeyConceptThinking) -> None:
        """Test _generate_conclusion heuristic."""
        initialized_method._step_counter = 5
        initialized_method._extracted_concepts = ["C1", "C2", "C3"]
        initialized_method._concept_definitions = {"C1": "D1", "C2": "D2"}
        content = initialized_method._generate_conclusion(None, None)

        assert "Final Answer" in content
        assert "Concepts extracted: 3" in content

    @pytest.mark.asyncio
    async def test_sample_extraction(
        self,
        initialized_method: KeyConceptThinking,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_extraction method."""
        initialized_method._execution_context = mock_execution_context

        result = await initialized_method._sample_extraction("Test problem", None)

        assert result is not None
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_definitions(
        self,
        initialized_method: KeyConceptThinking,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_definitions method."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._extracted_concepts = ["Concept 1", "Concept 2"]

        result = await initialized_method._sample_definitions(None, None)

        assert result is not None
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_application(
        self,
        initialized_method: KeyConceptThinking,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_application method."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._concept_definitions = {"C1": "D1"}

        result = await initialized_method._sample_application(None, None)

        assert result is not None
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_solution(
        self,
        initialized_method: KeyConceptThinking,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_solution method."""
        initialized_method._execution_context = mock_execution_context

        result = await initialized_method._sample_solution(None, None)

        assert result is not None
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_conclusion(
        self,
        initialized_method: KeyConceptThinking,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_conclusion method."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._extracted_concepts = ["C1"]
        initialized_method._concept_definitions = {"C1": "D1"}

        result = await initialized_method._sample_conclusion(None, None)

        assert result is not None
        mock_execution_context.sample.assert_called_once()

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that step counter increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        await initialized_method.continue_reasoning(session, thought1)
        assert initialized_method._step_counter == 2

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_concepts(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks concepts."""
        thought = await initialized_method.execute(session, sample_problem)
        assert "concepts" in thought.metadata

    @pytest.mark.asyncio
    async def test_metadata_tracks_definitions(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks definitions after define phase."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "definitions" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)  # define
        assert thought.confidence == 0.75

        thought = await initialized_method.continue_reasoning(session, thought)  # apply
        assert thought.confidence == 0.8

        thought = await initialized_method.continue_reasoning(session, thought)  # solve
        assert thought.confidence == 0.85

        thought = await initialized_method.continue_reasoning(session, thought)  # conclude
        assert thought.confidence == 0.9

    @pytest.mark.asyncio
    async def test_fallback_phase_handling(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test fallback phase handling for unknown phases."""
        await initialized_method.execute(session, sample_problem)

        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 5
        mock_thought.metadata = {"phase": "unknown_phase"}

        thought = await initialized_method.continue_reasoning(session, mock_thought)
        assert thought.metadata["phase"] == "solve"

    @pytest.mark.asyncio
    async def test_metadata_tracks_sampled_flag(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that metadata tracks whether sampling was used."""
        thought = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        assert thought.metadata["sampled"] is True

    @pytest.mark.asyncio
    async def test_metadata_tracks_sampled_false(
        self,
        initialized_method: KeyConceptThinking,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks when sampling was not used."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert thought.metadata["sampled"] is False


__all__ = [
    "TestKeyConceptThinkingMetadata",
    "TestKeyConceptThinking",
]
