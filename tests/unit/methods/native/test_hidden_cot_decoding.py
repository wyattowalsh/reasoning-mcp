"""Unit tests for Hidden CoT Decoding reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Hidden reasoning phase
- Decode phase
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.hidden_cot_decoding import (
    HIDDEN_COT_DECODING_METADATA,
    HiddenCotDecoding,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestHiddenCotDecodingMetadata:
    """Tests for Hidden CoT Decoding metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert HIDDEN_COT_DECODING_METADATA.identifier == MethodIdentifier.HIDDEN_COT_DECODING

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert HIDDEN_COT_DECODING_METADATA.name == "Hidden CoT Decoding"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert HIDDEN_COT_DECODING_METADATA.description is not None
        assert "hidden" in HIDDEN_COT_DECODING_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert HIDDEN_COT_DECODING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"hidden", "efficient", "decoding", "implicit"}
        assert expected_tags.issubset(HIDDEN_COT_DECODING_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert HIDDEN_COT_DECODING_METADATA.complexity == 5

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert HIDDEN_COT_DECODING_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates no revision support."""
        assert HIDDEN_COT_DECODING_METADATA.supports_revision is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert HIDDEN_COT_DECODING_METADATA.min_thoughts == 2

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert HIDDEN_COT_DECODING_METADATA.max_thoughts == 4

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "token efficiency" in HIDDEN_COT_DECODING_METADATA.best_for

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies what method is not recommended for."""
        assert "explainability" in HIDDEN_COT_DECODING_METADATA.not_recommended_for


class TestHiddenCotDecoding:
    """Test suite for Hidden CoT Decoding reasoning method."""

    @pytest.fixture
    def method(self) -> HiddenCotDecoding:
        """Create method instance."""
        return HiddenCotDecoding()

    @pytest.fixture
    async def initialized_method(self) -> HiddenCotDecoding:
        """Create an initialized method instance."""
        method = HiddenCotDecoding()
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
        return "What is 5 + 3 * 4?"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = "Internal reasoning proceeding..."
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: HiddenCotDecoding) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, HiddenCotDecoding)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "hidden_reason"
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: HiddenCotDecoding) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "hidden_reason"

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: HiddenCotDecoding) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "decode"

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "hidden_reason"

    # === Property Tests ===

    def test_identifier_property(self, method: HiddenCotDecoding) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.HIDDEN_COT_DECODING

    def test_name_property(self, method: HiddenCotDecoding) -> None:
        """Test name property returns correct value."""
        assert method.name == "Hidden CoT Decoding"

    def test_description_property(self, method: HiddenCotDecoding) -> None:
        """Test description property returns correct value."""
        assert method.description == HIDDEN_COT_DECODING_METADATA.description

    def test_category_property(self, method: HiddenCotDecoding) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: HiddenCotDecoding) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: HiddenCotDecoding) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Hidden Reasoning Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: HiddenCotDecoding, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.HIDDEN_COT_DECODING
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "hidden_reason"

    @pytest.mark.asyncio
    async def test_execute_content_indicates_hidden_reasoning(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content indicates hidden state reasoning."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Hidden" in thought.content or "hidden" in thought.content
        assert sample_problem in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.HIDDEN_COT_DECODING

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() stores the execution context."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        assert initialized_method._execution_context is mock_execution_context

    # === Continue Reasoning Tests (Decode Phase) ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: HiddenCotDecoding, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "hidden_reason"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    @pytest.mark.asyncio
    async def test_decode_phase_produces_answer(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that decode phase produces an answer."""
        hidden_thought = await initialized_method.execute(session, sample_problem)
        decode_thought = await initialized_method.continue_reasoning(session, hidden_thought)

        assert decode_thought is not None
        assert decode_thought.metadata["phase"] == "decode"
        assert decode_thought.type == ThoughtType.CONCLUSION
        assert "Answer" in decode_thought.content or "answer" in decode_thought.content

    @pytest.mark.asyncio
    async def test_decode_phase_shows_efficiency(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that decode phase emphasizes efficiency."""
        hidden_thought = await initialized_method.execute(session, sample_problem)
        decode_thought = await initialized_method.continue_reasoning(session, hidden_thought)

        assert (
            "efficiency" in decode_thought.content.lower()
            or "token" in decode_thought.content.lower()
        )

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "hidden_reason"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "decode"
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_two_step_completion(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that method completes in two steps."""
        thought = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        thought = await initialized_method.continue_reasoning(session, thought)
        assert initialized_method._step_counter == 2

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: HiddenCotDecoding,
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
    async def test_continue_with_sampling(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that continue_reasoning uses LLM sampling when available."""
        thought = await initialized_method.execute(session, sample_problem)

        await initialized_method.continue_reasoning(
            session, thought, execution_context=mock_execution_context
        )

        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sampling_fallback_on_expected_error(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails with expected exceptions."""
        # Test with TimeoutError (an expected exception that triggers fallback)
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=TimeoutError("Request timed out"))

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=failing_ctx
        )

        # Should use fallback content
        assert "Hidden state reasoning" in thought.content or "hidden" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_sampling_reraises_unexpected_error(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that unexpected exceptions are re-raised, not swallowed."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=RuntimeError("Unexpected bug"))

        with pytest.raises(RuntimeError, match="Unexpected bug"):
            await initialized_method.execute(
                session, sample_problem, execution_context=failing_ctx
            )

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "hidden" in thought.content.lower()

    # === Heuristic Method Tests ===

    def test_generate_hidden_reasoning(self, initialized_method: HiddenCotDecoding) -> None:
        """Test _generate_hidden_reasoning heuristic."""
        initialized_method._step_counter = 1
        content = initialized_method._generate_hidden_reasoning("Test problem")

        assert "Hidden Reasoning" in content
        assert "Test problem" in content
        assert "no explicit tokens" in content

    def test_generate_decode_answer(self, initialized_method: HiddenCotDecoding) -> None:
        """Test _generate_decode_answer heuristic."""
        initialized_method._step_counter = 2
        content = initialized_method._generate_decode_answer()

        assert "Decode Answer" in content
        assert "Final Answer" in content
        assert "Maximum efficiency" in content

    # === Sampling Method Tests ===

    @pytest.mark.asyncio
    async def test_sample_hidden_reasoning(
        self,
        initialized_method: HiddenCotDecoding,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_hidden_reasoning with context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 1

        content = await initialized_method._sample_hidden_reasoning("Test problem")

        assert "Hidden Reasoning" in content
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_decode_answer(
        self,
        initialized_method: HiddenCotDecoding,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_decode_answer with context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 2

        content = await initialized_method._sample_decode_answer("Previous content", None)

        assert "Decode Answer" in content
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_decode_answer_with_guidance(
        self,
        initialized_method: HiddenCotDecoding,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_decode_answer with guidance."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 2

        await initialized_method._sample_decode_answer("Previous content", "Show work")

        # Check that guidance was included in the call
        call_args = mock_execution_context.sample.call_args
        assert "Show work" in call_args[0][0]

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: HiddenCotDecoding,
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
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_sampling_status(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks whether sampling was used."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "sampled" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_values(
        self,
        initialized_method: HiddenCotDecoding,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence values are set correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.confidence == 0.75

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.confidence == 0.88


__all__ = [
    "TestHiddenCotDecodingMetadata",
    "TestHiddenCotDecoding",
]
