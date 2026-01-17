"""Unit tests for HybridCoT reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Encode latent phase
- Text steps phase
- Decode phase
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.hybrid_cot import (
    HYBRID_COT_METADATA,
    HybridCot,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestHybridCotMetadata:
    """Tests for HybridCoT metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert HYBRID_COT_METADATA.identifier == MethodIdentifier.HYBRID_COT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert HYBRID_COT_METADATA.name == "HybridCoT"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert HYBRID_COT_METADATA.description is not None
        assert "latent" in HYBRID_COT_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert HYBRID_COT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"latent", "hybrid", "efficient", "interleaved"}
        assert expected_tags.issubset(HYBRID_COT_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert HYBRID_COT_METADATA.complexity == 6

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert HYBRID_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert HYBRID_COT_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert HYBRID_COT_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert HYBRID_COT_METADATA.max_thoughts == 7

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "balanced efficiency" in HYBRID_COT_METADATA.best_for


class TestHybridCot:
    """Test suite for HybridCoT reasoning method."""

    @pytest.fixture
    def method(self) -> HybridCot:
        """Create method instance."""
        return HybridCot()

    @pytest.fixture
    async def initialized_method(self) -> HybridCot:
        """Create an initialized method instance."""
        method = HybridCot()
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
        mock_response = (
            "Step 1: Encode Latent (HybridCoT)\n\n"
            "Problem: Test\n\n"
            "Latent Tokens:\n"
            "  <L1: parse_input>\n"
            "  <L2: compute>\n\n"
            "Next: Add text steps."
        )
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: HybridCot) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, HybridCot)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "encode_latent"
        assert method._latent_tokens == []
        assert method._text_steps == []
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: HybridCot) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "encode_latent"
        assert method._latent_tokens == []
        assert method._text_steps == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: HybridCot) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "decode"
        initialized_method._latent_tokens = ["<L1: test>"]
        initialized_method._text_steps = ["Step 1"]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "encode_latent"
        assert initialized_method._latent_tokens == []
        assert initialized_method._text_steps == []

    # === Property Tests ===

    def test_identifier_property(self, method: HybridCot) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.HYBRID_COT

    def test_name_property(self, method: HybridCot) -> None:
        """Test name property returns correct value."""
        assert method.name == "HybridCoT"

    def test_description_property(self, method: HybridCot) -> None:
        """Test description property returns correct value."""
        assert method.description == HYBRID_COT_METADATA.description

    def test_category_property(self, method: HybridCot) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: HybridCot) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: HybridCot) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Encode Latent Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: HybridCot, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.HYBRID_COT
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "encode_latent"

    @pytest.mark.asyncio
    async def test_execute_content_includes_latent_tokens(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes latent tokens."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Latent" in thought.content
        assert "<L" in thought.content or "latent" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.HYBRID_COT

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: HybridCot,
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
        self, method: HybridCot, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "encode_latent"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Text Steps Phase Tests ===

    @pytest.mark.asyncio
    async def test_text_steps_phase(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that text steps phase provides explicit reasoning."""
        latent_thought = await initialized_method.execute(session, sample_problem)
        text_thought = await initialized_method.continue_reasoning(session, latent_thought)

        assert text_thought is not None
        assert text_thought.metadata["phase"] == "text_steps"
        assert text_thought.type == ThoughtType.REASONING
        assert "Explicit" in text_thought.content or "Text" in text_thought.content

    # === Decode Phase Tests ===

    @pytest.mark.asyncio
    async def test_decode_phase_produces_answer(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that decode phase produces an answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # text_steps
        thought = await initialized_method.continue_reasoning(session, thought)  # decode

        assert thought.metadata["phase"] == "decode"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Answer" in thought.content or "answer" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_decode_phase_shows_efficiency(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that decode phase mentions token efficiency."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "token" in thought.content.lower() or "HybridCoT" in thought.content

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_after_decode(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase can be reached after decode."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "encode_latent"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "text_steps"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "decode"
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_three_step_completion(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that method completes in three steps."""
        thought = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        thought = await initialized_method.continue_reasoning(session, thought)
        assert initialized_method._step_counter == 2

        thought = await initialized_method.continue_reasoning(session, thought)
        assert initialized_method._step_counter == 3

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: HybridCot,
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
        initialized_method: HybridCot,
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
    async def test_sampling_fallback_on_error(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails with expected errors."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        # Use ConnectionError which is an expected error type that triggers fallback
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection error"))

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=failing_ctx
        )

        # Should use fallback content with latent tokens
        assert "<L" in thought.content

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "<L" in thought.content

    # === Heuristic Method Tests ===

    def test_generate_encode_latent(self, initialized_method: HybridCot) -> None:
        """Test _generate_encode_latent heuristic."""
        initialized_method._step_counter = 1
        content = initialized_method._generate_encode_latent("Test problem")

        assert "Encode Latent" in content
        assert "Test problem" in content
        assert "<L1:" in content
        assert len(initialized_method._latent_tokens) == 3

    def test_generate_text_steps(self, initialized_method: HybridCot) -> None:
        """Test _generate_text_steps heuristic."""
        initialized_method._step_counter = 2
        content = initialized_method._generate_text_steps()

        assert "Text Steps" in content
        assert "Explicit" in content
        assert len(initialized_method._text_steps) == 2

    def test_generate_decode(self, initialized_method: HybridCot) -> None:
        """Test _generate_decode heuristic."""
        initialized_method._step_counter = 3
        content = initialized_method._generate_decode()

        assert "Decode" in content
        assert "Final Answer" in content
        assert "token reduction" in content

    # === Sampling Method Tests ===

    @pytest.mark.asyncio
    async def test_sample_encode_latent(
        self,
        initialized_method: HybridCot,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_encode_latent with context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 1

        content = await initialized_method._sample_encode_latent("Test problem")

        assert "Latent" in content or "<L" in content
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_text_steps(
        self,
        initialized_method: HybridCot,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_text_steps with context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 2

        await initialized_method._sample_text_steps("Previous content", None)

        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_text_steps_with_guidance(
        self,
        initialized_method: HybridCot,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_text_steps with guidance."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 2

        await initialized_method._sample_text_steps("Previous content", "Focus on math")

        # Check that guidance was included in the call
        call_args = mock_execution_context.sample.call_args
        assert "Focus on math" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_sample_decode(
        self,
        initialized_method: HybridCot,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_decode with context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 3

        await initialized_method._sample_decode("Previous content")

        mock_execution_context.sample.assert_called_once()

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that step counter increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert initialized_method._step_counter == 2

        await initialized_method.continue_reasoning(session, thought2)
        assert initialized_method._step_counter == 3

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.depth == 2

    @pytest.mark.asyncio
    async def test_metadata_tracks_sampling_status(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks whether sampling was used."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "sampled" in thought.metadata

    @pytest.mark.asyncio
    async def test_latent_count_tracked_in_metadata(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that latent token count is tracked in metadata."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "latent_count" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_values(
        self,
        initialized_method: HybridCot,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence values are set correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.confidence == 0.65

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.confidence == 0.78

        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.confidence == 0.88


class TestHybridCotLatentTokenParsing:
    """Tests for HybridCoT latent token parsing."""

    @pytest.fixture
    async def initialized_method(self) -> HybridCot:
        """Create initialized method."""
        method = HybridCot()
        await method.initialize()
        return method

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Create mock context with token-containing response."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = (
            "Step 1: Encode Latent\n\n"
            "Latent Tokens:\n"
            "  <L1: parse_problem>\n"
            "  <L2: identify_operations>\n"
            "  <L3: compute_result>\n"
            "  <L4: verify>\n\n"
            "Next: Add text steps."
        )
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    @pytest.mark.asyncio
    async def test_latent_tokens_extracted_from_response(
        self, initialized_method: HybridCot, mock_execution_context: MagicMock
    ) -> None:
        """Test that latent tokens are extracted from LLM response."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._step_counter = 1

        await initialized_method._sample_encode_latent("Test problem")

        assert len(initialized_method._latent_tokens) == 4
        assert "<L1: parse_problem>" in initialized_method._latent_tokens

    @pytest.mark.asyncio
    async def test_fallback_tokens_used_when_none_found(
        self, initialized_method: HybridCot
    ) -> None:
        """Test that fallback tokens are used when extraction fails."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.sample = AsyncMock(return_value="No tokens here")

        initialized_method._execution_context = mock_ctx
        initialized_method._step_counter = 1

        await initialized_method._sample_encode_latent("Test")

        # Should use fallback tokens
        assert len(initialized_method._latent_tokens) == 3
        assert initialized_method._latent_tokens[0] == "<L1: parse>"


__all__ = [
    "TestHybridCotMetadata",
    "TestHybridCot",
    "TestHybridCotLatentTokenParsing",
]
