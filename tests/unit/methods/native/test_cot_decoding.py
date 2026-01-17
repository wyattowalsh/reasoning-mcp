"""Unit tests for CoT Decoding reasoning method.

This module provides comprehensive tests for the CoTDecoding method implementation,
covering initialization, execution, decoding phases, token exploration, path discovery,
and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.cot_decoding import (
    COT_DECODING_METADATA,
    CoTDecoding,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def method() -> CoTDecoding:
    """Create a CoTDecoding method instance for testing.

    Returns:
        A fresh CoTDecoding instance
    """
    return CoTDecoding()


@pytest.fixture
def method_with_top_k() -> CoTDecoding:
    """Create a CoTDecoding instance with custom top_k.

    Returns:
        A CoTDecoding instance with top_k=10
    """
    return CoTDecoding(top_k=10)


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
    return "What is 15 multiplied by 8?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Sample response with reasoning steps")
    return ctx


class TestCoTDecodingInitialization:
    """Tests for CoTDecoding initialization and setup."""

    def test_create_method(self, method: CoTDecoding) -> None:
        """Test that CoTDecoding can be instantiated."""
        assert method is not None
        assert isinstance(method, CoTDecoding)

    def test_initial_state(self, method: CoTDecoding) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "decode"

    def test_default_top_k(self, method: CoTDecoding) -> None:
        """Test that default top_k is set correctly."""
        assert method._top_k == CoTDecoding.DEFAULT_TOP_K
        assert method._top_k == 5

    def test_custom_top_k(self, method_with_top_k: CoTDecoding) -> None:
        """Test that custom top_k is set correctly."""
        assert method_with_top_k._top_k == 10

    async def test_initialize(self, method: CoTDecoding) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "decode"
        assert method._token_alternatives == []
        assert method._discovered_paths == []
        assert method._path_confidences == []

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize() resets state even if called multiple times."""
        method = CoTDecoding()
        await method.initialize()
        method._step_counter = 5
        method._current_phase = "conclude"

        await method.initialize()
        assert method._step_counter == 0
        assert method._current_phase == "decode"

    async def test_health_check_not_initialized(self, method: CoTDecoding) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: CoTDecoding) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestCoTDecodingProperties:
    """Tests for CoTDecoding property accessors."""

    def test_identifier_property(self, method: CoTDecoding) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == MethodIdentifier.COT_DECODING

    def test_name_property(self, method: CoTDecoding) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "CoT Decoding"

    def test_description_property(self, method: CoTDecoding) -> None:
        """Test that description returns the correct method description."""
        assert "decoding" in method.description.lower()
        assert "chain-of-thought" in method.description.lower()

    def test_category_property(self, method: CoTDecoding) -> None:
        """Test that category returns the correct method category."""
        assert method.category == MethodCategory.SPECIALIZED


class TestCoTDecodingMetadata:
    """Tests for CoTDecoding metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert COT_DECODING_METADATA.identifier == MethodIdentifier.COT_DECODING

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert COT_DECODING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"decoding", "implicit", "token-level"}
        assert expected_tags.issubset(COT_DECODING_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert COT_DECODING_METADATA.supports_branching is True

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity rating."""
        assert 1 <= COT_DECODING_METADATA.complexity <= 10


class TestCoTDecodingExecution:
    """Tests for CoTDecoding execute() method."""

    async def test_execute_basic(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.COT_DECODING

    async def test_execute_without_initialization_raises(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute creates an INITIAL thought type."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_sets_decode_phase(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute sets decode phase in metadata."""
        await method.initialize()
        thought = await method.execute(session, sample_problem)

        assert thought.metadata.get("phase") == "decode"

    async def test_execute_adds_to_session(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute adds thought to the session."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_execute_with_execution_context(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test execute with execution context for sampling."""
        await method.initialize()
        thought = await method.execute(
            session,
            sample_problem,
            execution_context=mock_execution_context,
        )

        assert thought is not None
        assert thought.content != ""


class TestCoTDecodingContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_decode_to_discover(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from decode to discover phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)

        continuation = await method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.metadata.get("phase") == "discover"
        assert continuation.type == ThoughtType.REASONING

    async def test_continue_discover_to_score(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from discover to score phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        discover = await method.continue_reasoning(session, initial)

        score = await method.continue_reasoning(session, discover)

        assert score is not None
        assert score.metadata.get("phase") == "score"
        assert score.type == ThoughtType.VERIFICATION

    async def test_continue_score_to_select(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation from score to select phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        discover = await method.continue_reasoning(session, initial)
        score = await method.continue_reasoning(session, discover)

        select = await method.continue_reasoning(session, score)

        assert select is not None
        assert select.metadata.get("phase") == "select"
        assert select.type == ThoughtType.SYNTHESIS

    async def test_continue_to_conclusion(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test continuation to conclusion phase."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        discover = await method.continue_reasoning(session, initial)
        score = await method.continue_reasoning(session, discover)
        select = await method.continue_reasoning(session, score)

        conclusion = await method.continue_reasoning(session, select)

        assert conclusion is not None
        assert conclusion.metadata.get("phase") == "conclude"
        assert conclusion.type == ThoughtType.CONCLUSION

    async def test_continue_without_initialization_raises(
        self,
        method: CoTDecoding,
        session: Session,
    ) -> None:
        """Test that continue_reasoning raises if not initialized."""
        # Create a mock previous thought
        prev_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COT_DECODING,
            content="Test",
            metadata={"phase": "decode"},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, prev_thought)


class TestTokenAlternatives:
    """Tests for token alternative generation."""

    async def test_fallback_token_alternatives(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that fallback generates token alternatives."""
        await method.initialize()
        await method.execute(session, sample_problem)

        # Check that token alternatives were generated
        assert len(method._token_alternatives) > 0
        for alt in method._token_alternatives:
            assert "position" in alt
            assert "top_token" in alt
            assert "alternatives" in alt


class TestPathDiscovery:
    """Tests for CoT path discovery."""

    async def test_fallback_discovers_paths(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that fallback discovers CoT paths."""
        await method.initialize()
        initial = await method.execute(session, sample_problem)
        await method.continue_reasoning(session, initial)  # discover phase

        # Check that paths were discovered
        assert len(method._discovered_paths) > 0
        for path in method._discovered_paths:
            assert "id" in path
            assert "is_cot" in path
            assert "steps" in path


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: CoTDecoding,
        session: Session,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        thought = await method.execute(session, "")

        assert thought is not None

    async def test_very_long_problem(
        self,
        method: CoTDecoding,
        session: Session,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Analyze this: " + "test " * 500

        thought = await method.execute(session, long_problem)

        assert thought is not None

    async def test_special_characters_in_problem(
        self,
        method: CoTDecoding,
        session: Session,
    ) -> None:
        """Test execution with special characters."""
        await method.initialize()
        problem = "Calculate: √(x² + y²) where x=3 & y=4 → result?"

        thought = await method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_unicode_in_problem(
        self,
        method: CoTDecoding,
        session: Session,
    ) -> None:
        """Test execution with Unicode characters."""
        await method.initialize()
        problem = "解决这个问题: 如何优化算法?"

        thought = await method.execute(session, problem)

        assert thought is not None


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that session thought count updates correctly."""
        await method.initialize()
        initial_count = session.thought_count

        await method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_full_reasoning_chain(
        self,
        method: CoTDecoding,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test a full reasoning chain through all phases."""
        await method.initialize()

        # Execute initial
        initial = await method.execute(session, sample_problem)
        assert initial.metadata.get("phase") == "decode"

        # Continue through all phases
        discover = await method.continue_reasoning(session, initial)
        assert discover.metadata.get("phase") == "discover"

        score = await method.continue_reasoning(session, discover)
        assert score.metadata.get("phase") == "score"

        select = await method.continue_reasoning(session, score)
        assert select.metadata.get("phase") == "select"

        conclude = await method.continue_reasoning(session, select)
        assert conclude.metadata.get("phase") == "conclude"

        # Verify chain structure
        assert session.thought_count >= 5
        assert conclude.type == ThoughtType.CONCLUSION
