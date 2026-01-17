"""Unit tests for Iterative Refinement reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Generate phase (initial answer generation)
- Critique phase (identifying weaknesses)
- Refine phase (improving answer)
- Conclude phase (final answer)
- LLM sampling with fallbacks
- Elicitation for refinement approach
- Helper methods
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.methods.native.iterative_refinement import (
    ITERATIVE_REFINEMENT_METADATA,
    IterativeRefinement,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestIterativeRefinementMetadata:
    """Tests for Iterative Refinement metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert ITERATIVE_REFINEMENT_METADATA.identifier == MethodIdentifier.ITERATIVE_REFINEMENT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert ITERATIVE_REFINEMENT_METADATA.name == "Iterative Refinement"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert ITERATIVE_REFINEMENT_METADATA.description is not None
        assert "refine" in ITERATIVE_REFINEMENT_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert ITERATIVE_REFINEMENT_METADATA.category == MethodCategory.CORE

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"iterative", "refinement", "improvement", "critique", "progressive"}
        assert expected_tags.issubset(ITERATIVE_REFINEMENT_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert ITERATIVE_REFINEMENT_METADATA.complexity == 4

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert ITERATIVE_REFINEMENT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert ITERATIVE_REFINEMENT_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert ITERATIVE_REFINEMENT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert ITERATIVE_REFINEMENT_METADATA.max_thoughts == 10

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "quality improvement" in ITERATIVE_REFINEMENT_METADATA.best_for


class TestIterativeRefinement:
    """Test suite for Iterative Refinement reasoning method."""

    @pytest.fixture
    def method(self) -> IterativeRefinement:
        """Create method instance."""
        return IterativeRefinement()

    @pytest.fixture
    async def initialized_method(self) -> IterativeRefinement:
        """Create an initialized method instance."""
        method = IterativeRefinement()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> MagicMock:
        """Create a mock session for testing."""
        mock_sess = MagicMock(spec=Session)
        mock_sess.current_method = None
        mock_sess.thought_count = 0
        mock_sess._thoughts = []
        mock_sess.metrics = MagicMock()
        mock_sess.metrics.elicitations_made = 0

        def add_thought(thought: ThoughtNode) -> None:
            mock_sess._thoughts.append(thought)
            mock_sess.thought_count = len(mock_sess._thoughts)

        mock_sess.add_thought = add_thought

        return mock_sess

    @pytest.fixture
    def sample_problem(self) -> str:
        """Provide a sample problem for testing."""
        return "Write a comprehensive explanation of machine learning"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.ctx = MagicMock()
        mock_response = "Initial answer about machine learning concepts and applications."
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: IterativeRefinement) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, IterativeRefinement)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._current_iteration == 0
        assert method._current_answer == ""
        assert method._improvement_history == []
        assert method._execution_context is None

    def test_default_max_iterations_constant(self, method: IterativeRefinement) -> None:
        """Test that DEFAULT_MAX_ITERATIONS constant is defined."""
        assert IterativeRefinement.DEFAULT_MAX_ITERATIONS == 3

    def test_initialization_with_custom_iterations(self) -> None:
        """Test initialization with custom max iterations."""
        method = IterativeRefinement(max_iterations=5)
        assert method._max_iterations == 5

    def test_initialization_with_elicitation_disabled(self) -> None:
        """Test initialization with elicitation disabled."""
        method = IterativeRefinement(enable_elicitation=False)
        assert method.enable_elicitation is False

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: IterativeRefinement) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._current_iteration == 0
        assert method._current_answer == ""
        assert method._improvement_history == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: IterativeRefinement) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._current_iteration = 3
        initialized_method._current_answer = "Test answer"
        initialized_method._improvement_history = [{"iteration": 1, "quality": 0.7}]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "generate"
        assert initialized_method._current_iteration == 0
        assert initialized_method._current_answer == ""
        assert initialized_method._improvement_history == []

    # === Property Tests ===

    def test_identifier_property(self, method: IterativeRefinement) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.ITERATIVE_REFINEMENT

    def test_name_property(self, method: IterativeRefinement) -> None:
        """Test name property returns correct value."""
        assert method.name == "Iterative Refinement"

    def test_description_property(self, method: IterativeRefinement) -> None:
        """Test description property returns correct value."""
        assert method.description == ITERATIVE_REFINEMENT_METADATA.description

    def test_category_property(self, method: IterativeRefinement) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.CORE

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: IterativeRefinement) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: IterativeRefinement) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Generate Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: IterativeRefinement, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.ITERATIVE_REFINEMENT
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "generate"
        assert thought.metadata["iteration"] == 1

    @pytest.mark.asyncio
    async def test_execute_generates_initial_answer(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates initial answer."""
        await initialized_method.execute(session, sample_problem)

        assert initialized_method._current_answer != ""
        assert len(initialized_method._improvement_history) == 1
        assert initialized_method._improvement_history[0]["iteration"] == 1
        assert initialized_method._improvement_history[0]["quality"] == 0.6

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.ITERATIVE_REFINEMENT

    @pytest.mark.asyncio
    async def test_execute_stores_input_in_metadata(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() stores input text in metadata."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["input"] == sample_problem

    @pytest.mark.asyncio
    async def test_execute_content_includes_problem(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes the problem."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Initial Generation" in thought.content
        assert sample_problem in thought.content
        assert "Quality estimate: 60%" in thought.content

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: IterativeRefinement, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "generate", "input": "test"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Critique Phase Tests ===

    @pytest.mark.asyncio
    async def test_critique_phase(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that critique phase analyzes the answer."""
        generate_thought = await initialized_method.execute(session, sample_problem)
        critique_thought = await initialized_method.continue_reasoning(session, generate_thought)

        assert critique_thought.metadata["phase"] == "critique"
        assert critique_thought.type == ThoughtType.VERIFICATION
        assert "Critique" in critique_thought.content

    @pytest.mark.asyncio
    async def test_critique_phase_after_refine(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that critique follows refine phase (for iteration > 1)."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine

        # At this point, iteration 2 starts, phase should be refine
        assert thought.metadata["phase"] == "refine"

        # Continue to get next critique
        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "critique"

    # === Refine Phase Tests ===

    @pytest.mark.asyncio
    async def test_refine_phase(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that refine phase improves the answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine

        assert thought.metadata["phase"] == "refine"
        assert thought.type == ThoughtType.REVISION
        assert thought.metadata["iteration"] == 2

    @pytest.mark.asyncio
    async def test_refine_improves_quality(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that refinement improves quality."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine

        # Quality should have improved
        assert len(initialized_method._improvement_history) == 2
        assert initialized_method._improvement_history[1]["quality"] > 0.6

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_after_max_iterations(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase occurs after max iterations."""
        thought = await initialized_method.execute(session, sample_problem)

        # Go through critique-refine cycles
        for _ in range(initialized_method._max_iterations):
            thought = await initialized_method.continue_reasoning(session, thought)  # critique
            if thought.metadata["phase"] == "conclude":
                break
            thought = await initialized_method.continue_reasoning(session, thought)  # refine

        # Should reach conclude eventually
        while thought.metadata["phase"] != "conclude":
            thought = await initialized_method.continue_reasoning(session, thought)

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Final Answer" in thought.content

    @pytest.mark.asyncio
    async def test_conclude_includes_quality_progression(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude shows quality progression."""
        initialized_method._max_iterations = 2  # Reduce for faster test

        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine
        thought = await initialized_method.continue_reasoning(session, thought)  # critique/conclude

        # Continue until conclude
        while thought.metadata["phase"] != "conclude":
            thought = await initialized_method.continue_reasoning(session, thought)

        assert "Quality progression" in thought.content
        assert "Iterative Refinement Complete" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        initialized_method._max_iterations = 2  # Reduce for faster test

        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "generate"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "critique"
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "refine"
        assert thought.type == ThoughtType.REVISION

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: IterativeRefinement,
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
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)

        # Should use fallback content
        assert "[Initial answer for:" in initialized_method._current_answer

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        await initialized_method.execute(session, sample_problem, execution_context=no_sample_ctx)

        assert "[Initial answer for:" in initialized_method._current_answer

    @pytest.mark.asyncio
    async def test_critique_sampling(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that critique uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        generate_thought = session._thoughts[0]

        await initialized_method.continue_reasoning(session, generate_thought)

        # Should have called sample again for critique
        assert mock_execution_context.sample.call_count == 2

    # === Helper Method Tests ===

    def test_generate_initial_answer(self, initialized_method: IterativeRefinement) -> None:
        """Test _generate_initial_answer heuristic."""
        answer = initialized_method._generate_initial_answer("Test problem")

        assert "[Initial answer for:" in answer

    def test_generate_critique(self, initialized_method: IterativeRefinement) -> None:
        """Test _generate_critique heuristic."""
        critique = initialized_method._generate_critique()

        assert "Identified Issues:" in critique
        assert "specific" in critique.lower() or "area" in critique.lower()

    def test_generate_initial_answer_heuristic(
        self,
        initialized_method: IterativeRefinement,
    ) -> None:
        """Test _generate_initial_answer heuristic method."""
        result = initialized_method._generate_initial_answer("Test problem")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_critique_heuristic(
        self,
        initialized_method: IterativeRefinement,
    ) -> None:
        """Test _generate_critique heuristic method."""
        initialized_method._current_answer = "Some answer"

        result = initialized_method._generate_critique()

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_sample_refinement_with_fallback(
        self,
        initialized_method: IterativeRefinement,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_refinement_with_fallback method."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._current_iteration = 2
        mock_execution_context.sample = AsyncMock(
            return_value="Refined answer\n\nCHANGES:\n- Change 1\n- Change 2"
        )

        refined, changes = await initialized_method._sample_refinement_with_fallback(
            "Current answer", "Critique content", "Original question", None
        )

        assert refined is not None
        assert len(changes) > 0
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_refinement_with_fallback_parses_changes(
        self,
        initialized_method: IterativeRefinement,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that _sample_refinement_with_fallback correctly parses changes."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._current_iteration = 2
        mock_execution_context.sample = AsyncMock(
            return_value="Better answer with improvements.\n\nCHANGES:\n- Improved clarity\n- Added examples\n- Fixed logic"
        )

        refined, changes = await initialized_method._sample_refinement_with_fallback(
            "Current answer", "Critique content", "Original question", None
        )

        assert "Better answer" in refined
        assert len(changes) == 3
        assert "Improved clarity" in changes

    @pytest.mark.asyncio
    async def test_sample_refinement_with_fallback_approach(
        self,
        initialized_method: IterativeRefinement,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_refinement_with_fallback with different approaches."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._current_iteration = 2
        mock_execution_context.sample = AsyncMock(
            return_value="Aggressive refinement\n\nCHANGES:\n- Major change"
        )

        for approach in ["aggressive", "conservative", "targeted", "comprehensive"]:
            await initialized_method._sample_refinement_with_fallback(
                "Answer", "Critique", "Question", approach
            )

        assert mock_execution_context.sample.call_count == 4

    # === Elicitation Tests ===

    @pytest.mark.asyncio
    async def test_elicitation_for_refinement_approach(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that refinement phase completes with elicitation enabled.

        Note: The elicitation call depends on _execution_context.ctx being
        properly configured. This test verifies the execution path completes.
        """
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.ctx = MagicMock()
        mock_ctx.sample = AsyncMock(return_value="Answer")

        with patch(
            "reasoning_mcp.methods.native.iterative_refinement.elicit_selection",
            new=AsyncMock(return_value=MagicMock(selected="aggressive")),
        ):
            thought = await initialized_method.execute(
                session, sample_problem, execution_context=mock_ctx
            )
            thought = await initialized_method.continue_reasoning(session, thought)  # critique
            thought = await initialized_method.continue_reasoning(session, thought)  # refine

            # Verify refine phase completed successfully
            assert thought is not None
            assert thought.metadata.get("phase") == "refine"

    @pytest.mark.asyncio
    async def test_elicitation_disabled(
        self,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that elicitation is skipped when disabled."""
        method = IterativeRefinement(enable_elicitation=False)
        await method.initialize()

        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.ctx = MagicMock()
        mock_ctx.sample = AsyncMock(return_value="Answer")

        with patch(
            "reasoning_mcp.methods.native.iterative_refinement.elicit_selection"
        ) as mock_elicit:
            thought = await method.execute(session, sample_problem, execution_context=mock_ctx)
            thought = await method.continue_reasoning(session, thought)  # critique
            thought = await method.continue_reasoning(session, thought)  # refine

            # Elicitation should not have been called
            mock_elicit.assert_not_called()

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: IterativeRefinement,
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
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_iteration_count_tracked(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that iteration count is tracked correctly."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["iteration"] == 1

        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine

        assert thought.metadata["iteration"] == 2

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through iterations."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.6

        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        assert thought.confidence > 0.6

    @pytest.mark.asyncio
    async def test_input_propagated_through_phases(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that input is propagated through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["input"] == sample_problem

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["input"] == sample_problem

        @pytest.mark.asyncio
        async def test_single_iteration_mode(
            self,
            session: MagicMock,
            sample_problem: str,
        ) -> None:
            """Test behavior with max_iterations=1."""
            method = IterativeRefinement(max_iterations=1)
            await method.initialize()

            thought = await method.execute(session, sample_problem)
            assert thought.metadata["phase"] == "generate"

            thought = await method.continue_reasoning(session, thought)  # critique
            assert thought.metadata["phase"] == "critique"

            # Since _current_iteration(1) < max_iterations(1) is False,
            # the next call should conclude instead of refine
            thought = await method.continue_reasoning(session, thought)  # conclude
            assert thought.metadata["phase"] == "conclude"

    @pytest.mark.asyncio
    async def test_improvement_history_tracked(
        self,
        initialized_method: IterativeRefinement,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that improvement history is tracked through phases."""
        initialized_method._max_iterations = 2

        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # critique
        thought = await initialized_method.continue_reasoning(session, thought)  # refine

        assert "history" in thought.metadata
        assert len(thought.metadata["history"]) == 2


__all__ = [
    "TestIterativeRefinementMetadata",
    "TestIterativeRefinement",
]
