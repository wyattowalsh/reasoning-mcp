"""Unit tests for Meta Chain-of-Thought (Meta-CoT) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Analyze phase (problem classification)
- Strategize phase (strategy selection)
- Execute phase (strategy application)
- Reflect phase (meta-evaluation)
- Conclude phase
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.meta_cot import (
    META_COT_METADATA,
    MetaCoT,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestMetaCotMetadata:
    """Tests for Meta-CoT metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert META_COT_METADATA.identifier == MethodIdentifier.META_COT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert META_COT_METADATA.name == "Meta Chain-of-Thought"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert META_COT_METADATA.description is not None
        assert "meta" in META_COT_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert META_COT_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"meta-reasoning", "strategy", "adaptive", "metacognitive"}
        assert expected_tags.issubset(META_COT_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert META_COT_METADATA.complexity == 8

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert META_COT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert META_COT_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert META_COT_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert META_COT_METADATA.max_thoughts == 10

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "complex reasoning" in META_COT_METADATA.best_for


class TestMetaCoT:
    """Test suite for Meta-CoT reasoning method."""

    @pytest.fixture
    def method(self) -> MetaCoT:
        """Create method instance."""
        return MetaCoT()

    @pytest.fixture
    async def initialized_method(self) -> MetaCoT:
        """Create an initialized method instance."""
        method = MetaCoT()
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
        return "Solve a complex multi-step problem involving logical reasoning."

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: MetaCoT) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, MetaCoT)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._problem_type == ""
        assert method._complexity == ""
        assert method._selected_strategy == ""
        assert method._strategy_rationale == ""
        assert method._execution_context is None

    def test_strategies_constant(self, method: MetaCoT) -> None:
        """Test that STRATEGIES constant is defined."""
        assert len(MetaCoT.STRATEGIES) == 5
        assert "decomposition" in MetaCoT.STRATEGIES
        assert "analogy" in MetaCoT.STRATEGIES
        assert "abstraction" in MetaCoT.STRATEGIES

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: MetaCoT) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "analyze"
        assert method._problem_type == ""
        assert method._complexity == ""
        assert method._selected_strategy == ""

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: MetaCoT) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "reflect"
        initialized_method._problem_type = "mathematical"
        initialized_method._selected_strategy = "decomposition"

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "analyze"
        assert initialized_method._problem_type == ""
        assert initialized_method._selected_strategy == ""

    # === Property Tests ===

    def test_identifier_property(self, method: MetaCoT) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.META_COT

    def test_name_property(self, method: MetaCoT) -> None:
        """Test name property returns correct value."""
        assert method.name == "Meta Chain-of-Thought"

    def test_description_property(self, method: MetaCoT) -> None:
        """Test description property returns correct value."""
        assert method.description == META_COT_METADATA.description

    def test_category_property(self, method: MetaCoT) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: MetaCoT) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: MetaCoT) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Analyze Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: MetaCoT, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.META_COT
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "analyze"

    @pytest.mark.asyncio
    async def test_execute_analyzes_problem(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() analyzes the problem."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Analyze Problem" in thought.content
        assert "Meta-Analysis" in thought.content
        assert "Problem Type" in thought.content
        assert "Complexity" in thought.content

    @pytest.mark.asyncio
    async def test_execute_lists_strategies(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() lists available strategies."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Available Strategies" in thought.content
        for strategy in MetaCoT.STRATEGIES:
            assert strategy in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.META_COT

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: MetaCoT, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "analyze"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Strategize Phase Tests ===

    @pytest.mark.asyncio
    async def test_strategize_phase(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that strategize phase selects a strategy."""
        analyze_thought = await initialized_method.execute(session, sample_problem)
        strategize_thought = await initialized_method.continue_reasoning(session, analyze_thought)

        assert strategize_thought is not None
        assert strategize_thought.metadata["phase"] == "strategize"
        assert strategize_thought.type == ThoughtType.REASONING
        assert "Select Meta-Strategy" in strategize_thought.content

    @pytest.mark.asyncio
    async def test_strategize_sets_selected_strategy(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that strategize phase sets the selected strategy."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        assert initialized_method._selected_strategy != ""
        assert initialized_method._strategy_rationale != ""

    # === Execute Phase Tests ===

    @pytest.mark.asyncio
    async def test_execute_phase_applies_strategy(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute phase applies the selected strategy."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # strategize
        thought = await initialized_method.continue_reasoning(session, thought)  # execute

        assert thought.metadata["phase"] == "execute"
        assert thought.type == ThoughtType.SYNTHESIS
        assert "Execute Meta-Strategy" in thought.content

    # === Reflect Phase Tests ===

    @pytest.mark.asyncio
    async def test_reflect_phase_evaluates_effectiveness(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that reflect phase evaluates effectiveness."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect

        assert thought.metadata["phase"] == "reflect"
        assert thought.type == ThoughtType.VERIFICATION
        assert (
            "Reflect on Meta-Reasoning" in thought.content or "Meta-Evaluation" in thought.content
        )

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_produces_final_answer(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Meta Chain-of-Thought Complete" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "analyze"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "strategize"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "execute"
        assert thought.type == ThoughtType.SYNTHESIS

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "reflect"
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === Helper Method Tests ===

    def test_extract_field_success(self, initialized_method: MetaCoT) -> None:
        """Test _extract_field extracts field value."""
        text = "Problem Type: mathematical\nComplexity: high"
        result = initialized_method._extract_field(text, "Problem Type:", "default")

        assert result == "mathematical"

    def test_extract_field_default(self, initialized_method: MetaCoT) -> None:
        """Test _extract_field returns default when field not found."""
        text = "Some other text"
        result = initialized_method._extract_field(text, "Problem Type:", "default")

        assert result == "default"

    def test_extract_field_empty_value(self, initialized_method: MetaCoT) -> None:
        """Test _extract_field returns default for empty value."""
        text = "Problem Type: \nComplexity: high"
        result = initialized_method._extract_field(text, "Problem Type:", "default")

        assert result == "default"

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: MetaCoT,
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
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_strategy(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks strategy."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # strategize
        thought = await initialized_method.continue_reasoning(session, thought)  # execute

        assert "strategy" in thought.metadata
        assert thought.metadata["strategy"] == initialized_method._selected_strategy

    @pytest.mark.asyncio
    async def test_problem_type_tracked_in_metadata(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that problem type is tracked in metadata."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "problem_type" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: MetaCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.6

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.8

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.85

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.87


__all__ = [
    "TestMetaCotMetadata",
    "TestMetaCoT",
]
