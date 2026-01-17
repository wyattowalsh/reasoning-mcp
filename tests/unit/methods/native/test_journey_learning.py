"""Unit tests for Journey Learning reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Explore phase (initial journey)
- Reflect phase (insight gathering)
- Adjust phase (course corrections)
- Synthesize phase (combining learnings)
- Conclude phase (final answer)
- LLM sampling with fallbacks
- Journey log management
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.journey_learning import (
    JOURNEY_LEARNING_METADATA,
    JourneyLearning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestJourneyLearningMetadata:
    """Tests for Journey Learning metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert JOURNEY_LEARNING_METADATA.identifier == MethodIdentifier.JOURNEY_LEARNING

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert JOURNEY_LEARNING_METADATA.name == "Journey Learning"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert JOURNEY_LEARNING_METADATA.description is not None
        assert "journey" in JOURNEY_LEARNING_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert JOURNEY_LEARNING_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"journey", "exploration", "learning", "reflection", "process"}
        assert expected_tags.issubset(JOURNEY_LEARNING_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert JOURNEY_LEARNING_METADATA.complexity == 6

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert JOURNEY_LEARNING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert JOURNEY_LEARNING_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert JOURNEY_LEARNING_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert JOURNEY_LEARNING_METADATA.max_thoughts == 10

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "learning from process" in JOURNEY_LEARNING_METADATA.best_for


class TestJourneyLearning:
    """Test suite for Journey Learning reasoning method."""

    @pytest.fixture
    def method(self) -> JourneyLearning:
        """Create method instance."""
        return JourneyLearning()

    @pytest.fixture
    async def initialized_method(self) -> JourneyLearning:
        """Create an initialized method instance."""
        method = JourneyLearning()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> MagicMock:
        """Create a mock session for testing."""
        mock_sess = MagicMock(spec=Session)
        mock_sess.current_method = None
        mock_sess.thought_count = 0
        mock_sess._thoughts = []
        mock_sess.thoughts = []

        def add_thought(thought: ThoughtNode) -> None:
            mock_sess._thoughts.append(thought)
            mock_sess.thoughts.append(thought)
            mock_sess.thought_count = len(mock_sess._thoughts)

        mock_sess.add_thought = add_thought

        return mock_sess

    @pytest.fixture
    def sample_problem(self) -> str:
        """Provide a sample problem for testing."""
        return "Explore the factors affecting climate change"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = (
            "Insight 1: Hidden dependencies\nInsight 2: Need revision\nInsight 3: Multiple paths"
        )
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: JourneyLearning) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, JourneyLearning)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "explore"
        assert method._journey_log == []
        assert method._insights == []
        assert method._adjustments == []
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: JourneyLearning) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "explore"
        assert method._journey_log == []
        assert method._insights == []
        assert method._adjustments == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: JourneyLearning) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._journey_log = [{"step": 1}]
        initialized_method._insights = ["Insight 1"]
        initialized_method._adjustments = ["Adjustment 1"]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "explore"
        assert initialized_method._journey_log == []
        assert initialized_method._insights == []
        assert initialized_method._adjustments == []

    # === Property Tests ===

    def test_identifier_property(self, method: JourneyLearning) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.JOURNEY_LEARNING

    def test_name_property(self, method: JourneyLearning) -> None:
        """Test name property returns correct value."""
        assert method.name == "Journey Learning"

    def test_description_property(self, method: JourneyLearning) -> None:
        """Test description property returns correct value."""
        assert method.description == JOURNEY_LEARNING_METADATA.description

    def test_category_property(self, method: JourneyLearning) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.HOLISTIC

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: JourneyLearning) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: JourneyLearning) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Explore Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: JourneyLearning, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.EXPLORATION
        assert thought.method_id == MethodIdentifier.JOURNEY_LEARNING
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "explore"

    @pytest.mark.asyncio
    async def test_execute_starts_journey_log(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() starts the journey log."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._journey_log) == 1
        assert initialized_method._journey_log[0]["step"] == 1
        assert initialized_method._journey_log[0]["action"] == "Begin exploration"

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.JOURNEY_LEARNING

    @pytest.mark.asyncio
    async def test_execute_content_includes_journey_start(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes journey start elements."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Begin Journey" in thought.content
        assert sample_problem in thought.content
        assert "Journey Log Entry" in thought.content

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: JourneyLearning, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "explore"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Reflect Phase Tests ===

    @pytest.mark.asyncio
    async def test_reflect_phase(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that reflect phase gathers insights."""
        explore_thought = await initialized_method.execute(session, sample_problem)
        reflect_thought = await initialized_method.continue_reasoning(session, explore_thought)

        assert reflect_thought.metadata["phase"] == "reflect"
        assert reflect_thought.type == ThoughtType.INSIGHT
        assert "Reflect on Journey" in reflect_thought.content

    @pytest.mark.asyncio
    async def test_reflect_generates_insights(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that reflect phase generates insights."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        assert len(initialized_method._insights) > 0

    @pytest.mark.asyncio
    async def test_reflect_adds_to_journey_log(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that reflect phase adds to journey log."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        assert len(initialized_method._journey_log) == 2
        assert initialized_method._journey_log[1]["action"] == "Deeper exploration"

    # === Adjust Phase Tests ===

    @pytest.mark.asyncio
    async def test_adjust_phase(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that adjust phase makes course corrections."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust

        assert thought.metadata["phase"] == "adjust"
        assert thought.type == ThoughtType.REVISION
        assert "Adjust Course" in thought.content

    @pytest.mark.asyncio
    async def test_adjust_generates_adjustments(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that adjust phase generates adjustments."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        await initialized_method.continue_reasoning(session, thought)  # adjust

        assert len(initialized_method._adjustments) > 0

    @pytest.mark.asyncio
    async def test_adjust_adds_to_journey_log(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that adjust phase adds to journey log."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        await initialized_method.continue_reasoning(session, thought)  # adjust

        assert len(initialized_method._journey_log) == 3
        assert initialized_method._journey_log[2]["action"] == "Course correction"

    # === Synthesize Phase Tests ===

    @pytest.mark.asyncio
    async def test_synthesize_phase(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that synthesize phase combines learnings."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        thought = await initialized_method.continue_reasoning(session, thought)  # synthesize

        assert thought.metadata["phase"] == "synthesize"
        assert thought.type == ThoughtType.SYNTHESIS
        assert "Synthesize Journey" in thought.content

    @pytest.mark.asyncio
    async def test_synthesize_includes_journey_summary(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that synthesize phase includes journey summary."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        thought = await initialized_method.continue_reasoning(session, thought)  # synthesize

        assert "Journey Summary" in thought.content
        assert "Key Takeaways" in thought.content

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        thought = await initialized_method.continue_reasoning(session, thought)  # synthesize
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Final Answer" in thought.content

    @pytest.mark.asyncio
    async def test_conclude_includes_journey_stats(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude includes journey statistics."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        thought = await initialized_method.continue_reasoning(session, thought)  # synthesize
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert "Journey steps" in thought.content
        assert "Insights gained" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "explore"
        assert thought.type == ThoughtType.EXPLORATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "reflect"
        assert thought.type == ThoughtType.INSIGHT

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "adjust"
        assert thought.type == ThoughtType.REVISION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "synthesize"
        assert thought.type == ThoughtType.SYNTHESIS

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_insights_with_sampling(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that insights use LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        explore_thought = session._thoughts[0]

        await initialized_method.continue_reasoning(session, explore_thought)

        # Should have called sample for insights
        mock_execution_context.sample.assert_called()

    @pytest.mark.asyncio
    async def test_insights_fallback_on_error(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback insights are used when sampling fails with expected errors."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        # Use ConnectionError which is an expected exception type that triggers fallback
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection error"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)
        explore_thought = session._thoughts[0]

        await initialized_method.continue_reasoning(session, explore_thought)

        # Should use fallback insights
        assert "hidden dependencies" in initialized_method._insights[0].lower()

    @pytest.mark.asyncio
    async def test_adjustments_with_sampling(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that adjustments use LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        thought = session._thoughts[0]

        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        await initialized_method.continue_reasoning(session, thought)  # adjust

        # Should have called sample for adjustments
        assert mock_execution_context.sample.call_count >= 2

    @pytest.mark.asyncio
    async def test_synthesis_with_sampling(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that synthesis uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        thought = session._thoughts[0]

        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        await initialized_method.continue_reasoning(session, thought)  # synthesize

        # Should have called sample for synthesis
        assert mock_execution_context.sample.call_count >= 3

    # === Helper Method Tests ===

    @pytest.mark.asyncio
    async def test_generate_insights_fallback(self, initialized_method: JourneyLearning) -> None:
        """Test _generate_insights fallback without context."""
        initialized_method._execution_context = None
        initialized_method._journey_log = [{"step": 1, "action": "Test", "observation": "Test obs"}]

        insights = await initialized_method._generate_insights(
            "Test problem", initialized_method._journey_log
        )

        assert len(insights) == 3
        assert "hidden dependencies" in insights[0].lower()

    @pytest.mark.asyncio
    async def test_generate_adjustments_fallback(self, initialized_method: JourneyLearning) -> None:
        """Test _generate_adjustments fallback without context."""
        initialized_method._execution_context = None
        initialized_method._insights = ["Test insight"]

        adjustments = await initialized_method._generate_adjustments(
            "Test problem", initialized_method._insights
        )

        assert len(adjustments) == 3
        assert "Broaden" in adjustments[0]

    @pytest.mark.asyncio
    async def test_synthesize_solution_fallback(self, initialized_method: JourneyLearning) -> None:
        """Test _synthesize_solution fallback without context."""
        initialized_method._execution_context = None

        result = await initialized_method._synthesize_solution(
            "Test problem",
            [{"step": 1, "action": "Test", "observation": "Test"}],
            ["Insight 1"],
            ["Adjustment 1"],
        )

        assert "journey experience" in result.lower()

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: JourneyLearning,
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
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_journey_log(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks journey log."""
        thought = await initialized_method.execute(session, sample_problem)
        assert "journey_log" in thought.metadata
        assert len(thought.metadata["journey_log"]) == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_insights(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks insights after reflect."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "insights" in thought.metadata
        assert len(thought.metadata["insights"]) > 0

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.5

        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        assert thought.confidence == 0.65

        thought = await initialized_method.continue_reasoning(session, thought)  # adjust
        assert thought.confidence == 0.75

        thought = await initialized_method.continue_reasoning(session, thought)  # synthesize
        assert thought.confidence == 0.8

        thought = await initialized_method.continue_reasoning(session, thought)  # conclude
        assert thought.confidence == 0.85

    @pytest.mark.asyncio
    async def test_journey_log_persists_through_phases(
        self,
        initialized_method: JourneyLearning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that journey log accumulates through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # reflect
        thought = await initialized_method.continue_reasoning(session, thought)  # adjust

        assert len(initialized_method._journey_log) == 3

    @pytest.mark.asyncio
    async def test_fallback_phase_handling(
        self,
        initialized_method: JourneyLearning,
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
        assert thought.metadata["phase"] == "conclude"


__all__ = [
    "TestJourneyLearningMetadata",
    "TestJourneyLearning",
]
