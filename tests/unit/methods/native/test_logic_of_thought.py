"""Unit tests for Logic of Thought (LogiCoT) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Formalize phase (premise extraction)
- Infer phase (logical inference)
- Validate phase (validity checking)
- Conclude phase (logical conclusion)
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.logic_of_thought import (
    LOGIC_OF_THOUGHT_METADATA,
    LogicOfThought,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestLogicOfThoughtMetadata:
    """Tests for Logic of Thought metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert LOGIC_OF_THOUGHT_METADATA.identifier == MethodIdentifier.LOGIC_OF_THOUGHT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert LOGIC_OF_THOUGHT_METADATA.name == "Logic of Thought"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert LOGIC_OF_THOUGHT_METADATA.description is not None
        assert "logic" in LOGIC_OF_THOUGHT_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert LOGIC_OF_THOUGHT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"formal-logic", "premises", "inferences", "validity", "deduction"}
        assert expected_tags.issubset(LOGIC_OF_THOUGHT_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert LOGIC_OF_THOUGHT_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert LOGIC_OF_THOUGHT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert LOGIC_OF_THOUGHT_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert LOGIC_OF_THOUGHT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert LOGIC_OF_THOUGHT_METADATA.max_thoughts == 10

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "logical puzzles" in LOGIC_OF_THOUGHT_METADATA.best_for
        assert "deductive reasoning" in LOGIC_OF_THOUGHT_METADATA.best_for


class TestLogicOfThought:
    """Test suite for Logic of Thought reasoning method."""

    @pytest.fixture
    def method(self) -> LogicOfThought:
        """Create method instance."""
        return LogicOfThought()

    @pytest.fixture
    async def initialized_method(self) -> LogicOfThought:
        """Create an initialized method instance."""
        method = LogicOfThought()
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
        return "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = "P1: All mammals are warm-blooded\nP2: Whales are mammals"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: LogicOfThought) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, LogicOfThought)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "formalize"
        assert method._premises == []
        assert method._inferences == []
        assert method._inference_count == 0
        assert method._execution_context is None

    def test_max_inferences_constant(self, method: LogicOfThought) -> None:
        """Test that MAX_INFERENCES constant is defined."""
        assert LogicOfThought.MAX_INFERENCES == 5

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: LogicOfThought) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "formalize"
        assert method._premises == []
        assert method._inferences == []
        assert method._inference_count == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: LogicOfThought) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._premises = ["P1", "P2"]
        initialized_method._inferences = ["I1"]
        initialized_method._inference_count = 3

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "formalize"
        assert initialized_method._premises == []
        assert initialized_method._inferences == []
        assert initialized_method._inference_count == 0

    # === Property Tests ===

    def test_identifier_property(self, method: LogicOfThought) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.LOGIC_OF_THOUGHT

    def test_name_property(self, method: LogicOfThought) -> None:
        """Test name property returns correct value."""
        assert method.name == "Logic of Thought"

    def test_description_property(self, method: LogicOfThought) -> None:
        """Test description property returns correct value."""
        assert method.description == LOGIC_OF_THOUGHT_METADATA.description

    def test_category_property(self, method: LogicOfThought) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: LogicOfThought) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: LogicOfThought) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Formalize Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: LogicOfThought, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.LOGIC_OF_THOUGHT
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "formalize"

    @pytest.mark.asyncio
    async def test_execute_formalizes_problem(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() formalizes the problem."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Logical Formalization" in thought.content
        assert "Premises" in thought.content or "premises" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.LOGIC_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() stores the execution context."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        assert initialized_method._execution_context is mock_execution_context

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: LogicOfThought, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "formalize"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Infer Phase Tests ===

    @pytest.mark.asyncio
    async def test_infer_phase_produces_inference(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that infer phase produces logical inference."""
        formalize_thought = await initialized_method.execute(session, sample_problem)
        infer_thought = await initialized_method.continue_reasoning(session, formalize_thought)

        assert infer_thought is not None
        assert infer_thought.metadata["phase"] == "infer"
        assert infer_thought.type == ThoughtType.REASONING
        assert "Inference" in infer_thought.content

    @pytest.mark.asyncio
    async def test_infer_phase_tracks_inference_count(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that infer phase tracks inference count."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert initialized_method._inference_count == 1

    @pytest.mark.asyncio
    async def test_multiple_inferences(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that multiple inferences are performed."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # infer 1
        thought = await initialized_method.continue_reasoning(session, thought)  # infer 2

        assert initialized_method._inference_count >= 2

    # === Validate Phase Tests ===

    @pytest.mark.asyncio
    async def test_validate_phase_uses_verification_type(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that validate phase correctly uses ThoughtType.VERIFICATION.

        The source file was fixed to use ThoughtType.VERIFICATION instead of
        the non-existent ThoughtType.EVALUATION.
        """
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # infer 1
        thought = await initialized_method.continue_reasoning(session, thought)  # infer 2

        # Validate phase should work and produce VERIFICATION type thought
        validate_thought = await initialized_method.continue_reasoning(session, thought)
        assert validate_thought.type == ThoughtType.VERIFICATION
        assert initialized_method._current_phase == "validate"

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_generate_conclusion_heuristic(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_conclusion heuristic method directly."""
        initialized_method._step_counter = 5
        initialized_method._inference_count = 3
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.content = "Validation content"

        content = initialized_method._generate_conclusion(mock_thought, None, None)

        assert "Logical Conclusion" in content
        assert "∴" in content or "Conclusion" in content

    # === Done Phase Tests ===
    # NOTE: Cannot test done phase due to ThoughtType.EVALUATION bug in source.

    @pytest.mark.asyncio
    async def test_generate_final_synthesis_heuristic(
        self, initialized_method: LogicOfThought
    ) -> None:
        """Test _generate_final_synthesis heuristic method directly."""
        initialized_method._step_counter = 6
        initialized_method._inference_count = 3
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.content = "Conclusion content"

        content = initialized_method._generate_final_synthesis(mock_thought, None, None)

        assert "Final Answer" in content

    # === Full Pipeline Tests ===
    # NOTE: Full pipeline cannot be tested due to ThoughtType.EVALUATION bug.
    # Testing early phases only.

    @pytest.mark.asyncio
    async def test_reasoning_pipeline_early_phases(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test reasoning pipeline through formalize and infer phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "formalize"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "infer"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        # Second inference - still in infer phase
        assert thought.metadata["phase"] == "infer"
        assert thought.type == ThoughtType.REASONING

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: LogicOfThought,
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
    async def test_sampling_fallback_on_error(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=failing_ctx
        )

        # Should use fallback formalization
        assert "Logical Formalization" in thought.content

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "Logical Formalization" in thought.content

    # === Heuristic Method Tests ===

    def test_generate_formalization(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_formalization heuristic."""
        initialized_method._step_counter = 1
        content = initialized_method._generate_formalization("Test problem", None)

        assert "Logical Formalization" in content
        assert "P1:" in content or "Premises" in content
        assert len(initialized_method._premises) == 2

    def test_generate_inference(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_inference heuristic."""
        initialized_method._step_counter = 2
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.content = "Previous content"
        mock_thought.metadata = {"phase": "formalize"}

        content = initialized_method._generate_inference(mock_thought, 1, None, None)

        assert "Logical Inference" in content
        assert "Modus Ponens" in content or "Rule" in content

    def test_generate_validation(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_validation heuristic."""
        initialized_method._step_counter = 4
        content = initialized_method._generate_validation(None, None)

        assert "Logical Validation" in content
        assert "✓" in content or "VALID" in content

    def test_generate_conclusion(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_conclusion heuristic."""
        initialized_method._step_counter = 5
        initialized_method._inference_count = 3
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.content = "Validation content"

        content = initialized_method._generate_conclusion(mock_thought, None, None)

        assert "Logical Conclusion" in content
        assert "∴" in content or "Conclusion" in content

    def test_generate_final_synthesis(self, initialized_method: LogicOfThought) -> None:
        """Test _generate_final_synthesis heuristic."""
        initialized_method._step_counter = 6
        initialized_method._inference_count = 3
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.content = "Conclusion content"

        content = initialized_method._generate_final_synthesis(mock_thought, None, None)

        assert "Final Answer" in content

    # === Premise Extraction Tests ===

    def test_extract_premises_from_content(self, initialized_method: LogicOfThought) -> None:
        """Test _extract_premises_from_content."""
        content = "P1: All cats are animals\nP2: Whiskers is a cat\nP3: Therefore..."
        initialized_method._extract_premises_from_content(content)

        assert len(initialized_method._premises) == 3
        assert "P1:" in initialized_method._premises[0]

    def test_extract_premises_limits_to_five(self, initialized_method: LogicOfThought) -> None:
        """Test that premise extraction limits to 5 premises."""
        content = "\n".join([f"P{i}: Premise {i}" for i in range(1, 10)])
        initialized_method._extract_premises_from_content(content)

        assert len(initialized_method._premises) <= 5

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: LogicOfThought,
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
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_phase_history(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks previous phase."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "previous_phase" in thought.metadata
        assert thought.metadata["previous_phase"] == "formalize"

    @pytest.mark.asyncio
    async def test_metadata_tracks_inference_count(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks inference count."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "inference_count" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.75

    @pytest.mark.asyncio
    async def test_max_inferences_triggers_validation(
        self,
        initialized_method: LogicOfThought,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that max inferences triggers validation phase.

        When max inferences is reached, the method should transition to
        the validation phase with ThoughtType.VERIFICATION.
        """
        await initialized_method.execute(session, sample_problem)

        # Force high inference count
        initialized_method._inference_count = LogicOfThought.MAX_INFERENCES

        # Create inference phase thought
        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.LOGIC_OF_THOUGHT,
            content="Inference content",
            step_number=2,
            depth=1,
            confidence=0.75,
            metadata={"phase": "infer"},
        )

        # Should transition to validation phase with VERIFICATION type
        validate_thought = await initialized_method.continue_reasoning(session, thought)
        assert validate_thought.type == ThoughtType.VERIFICATION
        assert initialized_method._current_phase == "validate"


__all__ = [
    "TestLogicOfThoughtMetadata",
    "TestLogicOfThought",
]
