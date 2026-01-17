"""Unit tests for LightThinker reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Compress phase (gist token generation)
- Reason phase (processing gist tokens)
- Expand phase (final answer)
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.light_thinker import (
    LIGHT_THINKER_METADATA,
    LightThinker,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestLightThinkerMetadata:
    """Tests for LightThinker metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert LIGHT_THINKER_METADATA.identifier == MethodIdentifier.LIGHT_THINKER

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert LIGHT_THINKER_METADATA.name == "LightThinker"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert LIGHT_THINKER_METADATA.description is not None
        assert "gist" in LIGHT_THINKER_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert LIGHT_THINKER_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"compression", "gist", "efficient", "lightweight"}
        assert expected_tags.issubset(LIGHT_THINKER_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert LIGHT_THINKER_METADATA.complexity == 5

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert LIGHT_THINKER_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert LIGHT_THINKER_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert LIGHT_THINKER_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert LIGHT_THINKER_METADATA.max_thoughts == 5

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "token efficiency" in LIGHT_THINKER_METADATA.best_for


class TestLightThinker:
    """Test suite for LightThinker reasoning method."""

    @pytest.fixture
    def method(self) -> LightThinker:
        """Create method instance."""
        return LightThinker()

    @pytest.fixture
    async def initialized_method(self) -> LightThinker:
        """Create an initialized method instance."""
        method = LightThinker()
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

        def get_recent_thoughts(n: int) -> list[ThoughtNode]:
            return mock_sess._thoughts[-n:] if mock_sess._thoughts else []

        mock_sess.add_thought = add_thought
        mock_sess.get_recent_thoughts = get_recent_thoughts

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
        mock_response = "[G1:parse]\n[G2:compute]\n[G3:verify]"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: LightThinker) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, LightThinker)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "compress"
        assert method._gist_tokens == []
        assert method._use_sampling is True
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: LightThinker) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "compress"
        assert method._gist_tokens == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: LightThinker) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "expand"
        initialized_method._gist_tokens = ["[G1:test]"]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "compress"
        assert initialized_method._gist_tokens == []

    # === Property Tests ===

    def test_identifier_property(self, method: LightThinker) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.LIGHT_THINKER

    def test_name_property(self, method: LightThinker) -> None:
        """Test name property returns correct value."""
        assert method.name == "LightThinker"

    def test_description_property(self, method: LightThinker) -> None:
        """Test description property returns correct value."""
        assert method.description == LIGHT_THINKER_METADATA.description

    def test_category_property(self, method: LightThinker) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: LightThinker) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: LightThinker) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Compress Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: LightThinker, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.LIGHT_THINKER
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "compress"

    @pytest.mark.asyncio
    async def test_execute_content_includes_gist_tokens(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes gist tokens."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Gist" in thought.content
        assert "[G" in thought.content

    @pytest.mark.asyncio
    async def test_execute_generates_gist_tokens(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates gist tokens."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._gist_tokens) == 3
        assert initialized_method._gist_tokens[0] == "[G1:parse]"

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.LIGHT_THINKER

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: LightThinker, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "compress"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Reason Phase Tests ===

    @pytest.mark.asyncio
    async def test_reason_phase(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that reason phase processes gist tokens."""
        compress_thought = await initialized_method.execute(session, sample_problem)
        reason_thought = await initialized_method.continue_reasoning(session, compress_thought)

        assert reason_thought is not None
        assert reason_thought.metadata["phase"] == "reason"
        assert reason_thought.type == ThoughtType.REASONING
        assert "Processing gist tokens" in reason_thought.content

    # === Expand Phase Tests ===

    @pytest.mark.asyncio
    async def test_expand_phase_produces_answer(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that expand phase produces an answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reason
        thought = await initialized_method.continue_reasoning(session, thought)  # expand

        assert thought.metadata["phase"] == "expand"
        assert thought.type == ThoughtType.CONCLUSION
        assert "LightThinker Complete" in thought.content
        assert "Final Answer" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "compress"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "reason"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "expand"
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_three_step_completion(
        self,
        initialized_method: LightThinker,
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
        initialized_method: LightThinker,
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
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails with expected errors."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        # Use ConnectionError (an expected exception that triggers fallback)
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection error"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)

        # Should use fallback gist tokens
        assert initialized_method._gist_tokens == ["[G1:parse]", "[G2:compute]", "[G3:verify]"]

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        await initialized_method.execute(session, sample_problem, execution_context=no_sample_ctx)

        assert initialized_method._gist_tokens == ["[G1:parse]", "[G2:compute]", "[G3:verify]"]

    # === Heuristic Method Tests ===

    def test_heuristic_compress_to_gist(self, initialized_method: LightThinker) -> None:
        """Test _heuristic_compress_to_gist."""
        tokens = initialized_method._heuristic_compress_to_gist("Test problem")

        assert len(tokens) == 3
        assert tokens[0] == "[G1:parse]"
        assert tokens[1] == "[G2:compute]"
        assert tokens[2] == "[G3:verify]"

    def test_heuristic_reason_with_gist(self, initialized_method: LightThinker) -> None:
        """Test _heuristic_reason_with_gist."""
        result = initialized_method._heuristic_reason_with_gist()

        assert result == "5×3=15, 15+2=17"

    def test_heuristic_expand_from_gist(self, initialized_method: LightThinker) -> None:
        """Test _heuristic_expand_from_gist."""
        result = initialized_method._heuristic_expand_from_gist()

        assert result == "17"

    # === Sampling Method Tests ===

    @pytest.mark.asyncio
    async def test_sample_compress_to_gist(
        self,
        initialized_method: LightThinker,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_compress_to_gist with context."""
        initialized_method._execution_context = mock_execution_context

        tokens = await initialized_method._sample_compress_to_gist("Test problem")

        assert len(tokens) == 3
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_compress_fallback_no_tokens(
        self, initialized_method: LightThinker
    ) -> None:
        """Test _sample_compress_to_gist fallback when no tokens found."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.sample = AsyncMock(return_value="No tokens here")

        initialized_method._execution_context = mock_ctx

        tokens = await initialized_method._sample_compress_to_gist("Test")

        # Should use fallback tokens
        assert tokens == ["[G1:parse]", "[G2:compute]", "[G3:verify]"]

    @pytest.mark.asyncio
    async def test_sample_reason_with_gist(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_reason_with_gist with context."""
        # Add a thought to the session first
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LIGHT_THINKER,
            content="Test content",
            step_number=1,
            depth=0,
            confidence=0.7,
        )
        session.add_thought(thought)

        initialized_method._execution_context = mock_execution_context
        initialized_method._gist_tokens = ["[G1:parse]", "[G2:compute]"]

        await initialized_method._sample_reason_with_gist(session)

        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_expand_from_gist(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_expand_from_gist with context."""
        # Add a thought to the session first
        thought = ThoughtNode(
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.LIGHT_THINKER,
            content="Reasoning content",
            step_number=2,
            depth=1,
            confidence=0.8,
        )
        session.add_thought(thought)

        initialized_method._execution_context = mock_execution_context

        await initialized_method._sample_expand_from_gist(session)

        mock_execution_context.sample.assert_called_once()

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: LightThinker,
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
        initialized_method: LightThinker,
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
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks whether sampling was used."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "sampled" in thought.metadata
        assert thought.metadata["sampled"] is False  # No execution context provided

    @pytest.mark.asyncio
    async def test_gist_count_tracked_in_metadata(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that gist token count is tracked in metadata."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "gist_count" in thought.metadata
        assert thought.metadata["gist_count"] == 3

    @pytest.mark.asyncio
    async def test_confidence_values(
        self,
        initialized_method: LightThinker,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence values are set correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.confidence == 0.7

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.confidence == 0.8

        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.confidence == 0.88


__all__ = [
    "TestLightThinkerMetadata",
    "TestLightThinker",
]
