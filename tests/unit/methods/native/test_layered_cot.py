"""Unit tests for Layered Chain-of-Thought (Layered CoT) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Layer 1 phase (initial reasoning)
- Layer 2 phase (review and refine)
- Layer 3 phase (validate and adjust)
- Synthesize phase (combine layers)
- Conclude phase (final answer)
- LLM sampling with fallbacks
- Helper methods
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.layered_cot import (
    LAYERED_COT_METADATA,
    LayeredCoT,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestLayeredCoTMetadata:
    """Tests for Layered CoT metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert LAYERED_COT_METADATA.identifier == MethodIdentifier.LAYERED_COT

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert LAYERED_COT_METADATA.name == "Layered Chain-of-Thought"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert LAYERED_COT_METADATA.description is not None
        assert "multi-pass" in LAYERED_COT_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert LAYERED_COT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"multi-pass", "layered", "review", "refinement", "high-stakes"}
        assert expected_tags.issubset(LAYERED_COT_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert LAYERED_COT_METADATA.complexity == 6

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert LAYERED_COT_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert LAYERED_COT_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert LAYERED_COT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert LAYERED_COT_METADATA.max_thoughts == 8

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "high-stakes decisions" in LAYERED_COT_METADATA.best_for

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies what method is not recommended for."""
        assert "simple tasks" in LAYERED_COT_METADATA.not_recommended_for


class TestLayeredCoT:
    """Test suite for Layered Chain-of-Thought reasoning method."""

    @pytest.fixture
    def method(self) -> LayeredCoT:
        """Create method instance."""
        return LayeredCoT()

    @pytest.fixture
    async def initialized_method(self) -> LayeredCoT:
        """Create an initialized method instance."""
        method = LayeredCoT()
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
        return "Should this patient undergo surgery given the current diagnostic results?"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = "Layer analysis with comprehensive reasoning about the problem."
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: LayeredCoT) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, LayeredCoT)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "layer_1"
        assert method._current_layer == 0
        assert method._layer_outputs == []
        assert method._execution_context is None

    def test_default_layers_constant(self, method: LayeredCoT) -> None:
        """Test that DEFAULT_LAYERS constant is defined."""
        assert LayeredCoT.DEFAULT_LAYERS == 3

    def test_initialization_with_custom_layers(self) -> None:
        """Test initialization with custom number of layers."""
        method = LayeredCoT(num_layers=5)
        assert method._num_layers == 5

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: LayeredCoT) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "layer_1"
        assert method._current_layer == 0
        assert method._layer_outputs == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: LayeredCoT) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._current_layer = 3
        initialized_method._layer_outputs = [{"layer": 1, "confidence": 0.7}]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "layer_1"
        assert initialized_method._current_layer == 0
        assert initialized_method._layer_outputs == []

    # === Property Tests ===

    def test_identifier_property(self, method: LayeredCoT) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.LAYERED_COT

    def test_name_property(self, method: LayeredCoT) -> None:
        """Test name property returns correct value."""
        assert method.name == "Layered Chain-of-Thought"

    def test_description_property(self, method: LayeredCoT) -> None:
        """Test description property returns correct value."""
        assert method.description == LAYERED_COT_METADATA.description

    def test_category_property(self, method: LayeredCoT) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: LayeredCoT) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: LayeredCoT) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Layer 1 - Initial Reasoning) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: LayeredCoT, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.LAYERED_COT
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "layer_1"
        assert thought.metadata["layer"] == 1

    @pytest.mark.asyncio
    async def test_execute_generates_layer_output(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates layer output."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._layer_outputs) == 1
        layer_output = initialized_method._layer_outputs[0]
        assert layer_output["layer"] == 1
        assert "reasoning" in layer_output
        assert "conclusion" in layer_output
        assert "confidence" in layer_output
        assert "issues" in layer_output

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.LAYERED_COT

    @pytest.mark.asyncio
    async def test_execute_content_includes_layer_info(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes layer information."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "Layer 1" in thought.content
        assert "Initial Reasoning" in thought.content
        assert sample_problem in thought.content
        assert "Issues Identified" in thought.content
        assert "Next: Layer 2" in thought.content

    @pytest.mark.asyncio
    async def test_execute_confidence_from_layer_output(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute sets confidence from layer output."""
        thought = await initialized_method.execute(session, sample_problem)

        # Heuristic fallback sets confidence to 0.7
        assert thought.confidence == 0.7

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: LayeredCoT, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "layer_1", "layer": 1}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Layer 2 Tests (Review and Refine) ===

    @pytest.mark.asyncio
    async def test_layer_2_phase(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer 2 phase reviews and refines."""
        layer1_thought = await initialized_method.execute(session, sample_problem)
        layer2_thought = await initialized_method.continue_reasoning(session, layer1_thought)

        assert layer2_thought.metadata["phase"] == "layer_2"
        assert layer2_thought.metadata["layer"] == 2
        assert layer2_thought.type == ThoughtType.REVISION
        assert "Layer 2" in layer2_thought.content
        assert "Review" in layer2_thought.content

    @pytest.mark.asyncio
    async def test_layer_2_includes_adjustments(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer 2 includes adjustments."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "Adjustments Made" in thought.content

    @pytest.mark.asyncio
    async def test_layer_2_confidence_increases(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer 2 confidence increases."""
        thought = await initialized_method.execute(session, sample_problem)
        layer1_confidence = thought.confidence

        thought = await initialized_method.continue_reasoning(session, thought)

        assert thought.confidence > layer1_confidence

    # === Layer 3 Tests (Validate and Adjust) ===

    @pytest.mark.asyncio
    async def test_layer_3_phase(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer 3 phase validates and adjusts."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3

        assert thought.metadata["phase"] == "layer_3"
        assert thought.metadata["layer"] == 3
        assert thought.type == ThoughtType.VERIFICATION
        assert "Layer 3" in thought.content
        assert "Validate" in thought.content

    @pytest.mark.asyncio
    async def test_layer_3_notes_synthesis_next(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer 3 notes synthesis as next step."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3

        assert "Synthesize" in thought.content

    # === Synthesize and Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_synthesize_and_conclude_phase(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that synthesize transitions directly to conclude."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3
        thought = await initialized_method.continue_reasoning(
            session, thought
        )  # synthesize → conclude

        # Note: The source code transitions synthesize → conclude in same call
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_conclude_includes_layer_progression(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude includes layer progression info."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert "Layers processed" in thought.content
        assert "Initial confidence" in thought.content
        assert "Final confidence" in thought.content
        assert "Improvement" in thought.content

    @pytest.mark.asyncio
    async def test_conclude_includes_final_answer(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude includes final answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert "Final Answer" in thought.content
        assert "Layered Chain-of-Thought Complete" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "layer_1"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "layer_2"
        assert thought.type == ThoughtType.REVISION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "layer_3"
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_layers_complete_tracked(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layers_complete is tracked in metadata."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3

        assert thought.metadata["layers_complete"] == 3

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: LayeredCoT,
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
        initialized_method: LayeredCoT,
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

        # Should use fallback content
        assert "Layer 1" in thought.content
        assert initialized_method._layer_outputs[0]["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "Layer 1" in thought.content

    @pytest.mark.asyncio
    async def test_layer_refinement_with_sampling(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that layer refinement uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        layer1_thought = session._thoughts[0]

        await initialized_method.continue_reasoning(
            session, layer1_thought, execution_context=mock_execution_context
        )

        # Should have called sample twice (layer 1 + layer 2)
        assert mock_execution_context.sample.call_count == 2

    # === Helper Method Tests ===

    def test_generate_layer_reasoning_heuristic(self, initialized_method: LayeredCoT) -> None:
        """Test _generate_layer_reasoning_heuristic method."""
        result = initialized_method._generate_layer_reasoning_heuristic("Test problem", layer_num=1)

        assert result["layer"] == 1
        assert "reasoning" in result
        assert "conclusion" in result
        assert result["confidence"] == 0.7
        assert len(result["issues"]) > 0

    def test_generate_layer_reasoning_heuristic_different_layer(
        self, initialized_method: LayeredCoT
    ) -> None:
        """Test _generate_layer_reasoning_heuristic with different layer."""
        result = initialized_method._generate_layer_reasoning_heuristic("Test problem", layer_num=2)

        assert result["layer"] == 2

    def test_generate_layer_refinement_heuristic(self, initialized_method: LayeredCoT) -> None:
        """Test _generate_layer_refinement_heuristic method."""
        result = initialized_method._generate_layer_refinement_heuristic(
            prev_layer=1, current_layer=2, prev_confidence=0.7
        )

        assert result["layer"] == 2
        assert "reasoning" in result
        assert "conclusion" in result
        assert result["confidence"] == pytest.approx(0.78)  # 0.7 + 0.08
        assert len(result["adjustments"]) > 0

    def test_generate_layer_refinement_heuristic_confidence_cap(
        self, initialized_method: LayeredCoT
    ) -> None:
        """Test that refinement heuristic caps confidence at 0.95."""
        result = initialized_method._generate_layer_refinement_heuristic(
            prev_layer=2, current_layer=3, prev_confidence=0.92
        )

        # Should be capped at 0.95
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_sample_layer_reasoning(
        self,
        initialized_method: LayeredCoT,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_layer_reasoning method."""
        initialized_method._execution_context = mock_execution_context

        result = await initialized_method._sample_layer_reasoning("Test problem", layer_num=1)

        assert result is not None
        assert result["layer"] == 1
        assert "reasoning" in result
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_layer_reasoning_fallback_on_error(
        self,
        initialized_method: LayeredCoT,
    ) -> None:
        """Test _sample_layer_reasoning falls back on error."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))
        initialized_method._execution_context = failing_ctx

        result = await initialized_method._sample_layer_reasoning("Test problem", layer_num=1)

        # Should return heuristic fallback
        assert result["confidence"] == 0.7
        assert result["layer"] == 1

    @pytest.mark.asyncio
    async def test_sample_layer_refinement(
        self,
        initialized_method: LayeredCoT,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_layer_refinement method."""
        initialized_method._execution_context = mock_execution_context
        prev_output = {
            "layer": 1,
            "confidence": 0.7,
            "issues": ["Issue 1", "Issue 2"],
        }

        result = await initialized_method._sample_layer_refinement(
            previous_content="Layer 1 content",
            prev_layer=1,
            current_layer=2,
            prev_output=prev_output,
        )

        assert result is not None
        assert result["layer"] == 2
        assert result["confidence"] > 0.7
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_layer_refinement_with_guidance(
        self,
        initialized_method: LayeredCoT,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _sample_layer_refinement with guidance."""
        initialized_method._execution_context = mock_execution_context
        prev_output = {"layer": 1, "confidence": 0.7, "issues": []}

        await initialized_method._sample_layer_refinement(
            previous_content="Layer 1 content",
            prev_layer=1,
            current_layer=2,
            prev_output=prev_output,
            guidance="Focus on risk assessment",
        )

        # Check that guidance was included in the prompt
        call_args = mock_execution_context.sample.call_args
        assert "Focus on risk assessment" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_sample_layer_refinement_fallback_on_error(
        self,
        initialized_method: LayeredCoT,
    ) -> None:
        """Test _sample_layer_refinement falls back on error."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))
        initialized_method._execution_context = failing_ctx
        prev_output = {"layer": 1, "confidence": 0.7, "issues": []}

        result = await initialized_method._sample_layer_refinement(
            previous_content="Layer 1 content",
            prev_layer=1,
            current_layer=2,
            prev_output=prev_output,
        )

        # Should return heuristic fallback
        assert result["layer"] == 2
        assert result["confidence"] == pytest.approx(0.78)

    # === Custom Layer Count Tests ===

    @pytest.mark.asyncio
    async def test_two_layer_pipeline(
        self,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test pipeline with only 2 layers."""
        method = LayeredCoT(num_layers=2)
        await method.initialize()

        thought = await method.execute(session, sample_problem)
        assert thought.metadata["layer"] == 1

        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["layer"] == 2

        # Next should be synthesize → conclude
        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"

    @pytest.mark.asyncio
    async def test_four_layer_pipeline(
        self,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test pipeline with 4 layers."""
        method = LayeredCoT(num_layers=4)
        await method.initialize()

        thought = await method.execute(session, sample_problem)
        assert thought.metadata["layer"] == 1

        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["layer"] == 2

        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["layer"] == 3

        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["layer"] == 4

        thought = await method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that step counter increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1
        assert thought1.step_number == 1

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert initialized_method._step_counter == 2
        assert thought2.step_number == 2

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_layer_outputs_accumulate(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that layer outputs accumulate through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert len(initialized_method._layer_outputs) == 1

        thought = await initialized_method.continue_reasoning(session, thought)
        assert len(initialized_method._layer_outputs) == 2

        thought = await initialized_method.continue_reasoning(session, thought)
        assert len(initialized_method._layer_outputs) == 3

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through layers."""
        thought = await initialized_method.execute(session, sample_problem)
        conf1 = thought.confidence

        thought = await initialized_method.continue_reasoning(session, thought)
        conf2 = thought.confidence

        thought = await initialized_method.continue_reasoning(session, thought)
        conf3 = thought.confidence

        assert conf2 > conf1
        assert conf3 > conf2

    @pytest.mark.asyncio
    async def test_parent_id_tracked(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that parent_id is tracked in continued thoughts."""
        thought1 = await initialized_method.execute(session, sample_problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)

        assert thought2.parent_id == thought1.id

    @pytest.mark.asyncio
    async def test_quality_score_matches_confidence(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that quality_score matches confidence for continued thoughts."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert thought.quality_score == thought.confidence

    @pytest.mark.asyncio
    async def test_empty_layer_outputs_causes_index_error(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that empty layer_outputs causes IndexError (source code bug).

        This test documents a bug in the source code where the conclude phase
        tries to access self._layer_outputs[0]['confidence'] without checking
        if the list is empty. Since we cannot modify source code, we document
        the behavior here.
        """
        # Execute normally to set up state
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 2
        thought = await initialized_method.continue_reasoning(session, thought)  # layer 3

        # Clear layer outputs to simulate edge case
        initialized_method._layer_outputs = []

        # Manually set phase to synthesize to trigger edge case
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 3
        mock_thought.metadata = {"phase": "layer_3", "layer": 3}
        mock_thought.content = "Previous content"

        # Source code has a bug: tries to access self._layer_outputs[0] in conclude
        # phase without checking if list is empty
        with pytest.raises(IndexError, match="list index out of range"):
            await initialized_method.continue_reasoning(session, mock_thought)

    @pytest.mark.asyncio
    async def test_thought_added_to_session(
        self,
        initialized_method: LayeredCoT,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thoughts are added to session."""
        await initialized_method.execute(session, sample_problem)
        assert len(session._thoughts) == 1

        await initialized_method.continue_reasoning(session, session._thoughts[0])
        assert len(session._thoughts) == 2


__all__ = [
    "TestLayeredCoTMetadata",
    "TestLayeredCoT",
]
