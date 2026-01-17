"""Unit tests for Indirect Reasoning (Proof by Contradiction) method.

Tests cover:
- Metadata validation
- Initialization and state management
- State phase (claim formulation)
- Negate phase (assumption of negation)
- Derive phase (consequence derivation)
- Contradict phase (contradiction discovery)
- Conclude phase (proof completion)
- LLM sampling with fallbacks
- Helper methods
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.indirect_reasoning import (
    INDIRECT_REASONING_METADATA,
    IndirectReasoning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestIndirectReasoningMetadata:
    """Tests for Indirect Reasoning metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert INDIRECT_REASONING_METADATA.identifier == MethodIdentifier.INDIRECT_REASONING

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert INDIRECT_REASONING_METADATA.name == "Indirect Reasoning"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert INDIRECT_REASONING_METADATA.description is not None
        assert "contradiction" in INDIRECT_REASONING_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert INDIRECT_REASONING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {
            "proof-by-contradiction",
            "reductio-ad-absurdum",
            "indirect-proof",
            "logical",
        }
        assert expected_tags.issubset(INDIRECT_REASONING_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert INDIRECT_REASONING_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates no branching support."""
        assert INDIRECT_REASONING_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert INDIRECT_REASONING_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert INDIRECT_REASONING_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert INDIRECT_REASONING_METADATA.max_thoughts == 10

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "mathematical proofs" in INDIRECT_REASONING_METADATA.best_for


class TestIndirectReasoning:
    """Test suite for Indirect Reasoning method."""

    @pytest.fixture
    def method(self) -> IndirectReasoning:
        """Create method instance."""
        return IndirectReasoning()

    @pytest.fixture
    async def initialized_method(self) -> IndirectReasoning:
        """Create an initialized method instance."""
        method = IndirectReasoning()
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
        return "Prove that √2 is irrational"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = "Claim: √2 is irrational\nAssume: √2 = p/q in lowest terms"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: IndirectReasoning) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, IndirectReasoning)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "state"
        assert method._claim == ""
        assert method._negation == ""
        assert method._derivation_count == 0
        assert method._execution_context is None

    def test_max_derivations_constant(self, method: IndirectReasoning) -> None:
        """Test that MAX_DERIVATIONS constant is defined."""
        assert IndirectReasoning.MAX_DERIVATIONS == 5

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: IndirectReasoning) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "state"
        assert method._claim == ""
        assert method._negation == ""
        assert method._derivation_count == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: IndirectReasoning) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._claim = "Test claim"
        initialized_method._negation = "Test negation"
        initialized_method._derivation_count = 3

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "state"
        assert initialized_method._claim == ""
        assert initialized_method._negation == ""
        assert initialized_method._derivation_count == 0

    # === Property Tests ===

    def test_identifier_property(self, method: IndirectReasoning) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.INDIRECT_REASONING

    def test_name_property(self, method: IndirectReasoning) -> None:
        """Test name property returns correct value."""
        assert method.name == "Indirect Reasoning"

    def test_description_property(self, method: IndirectReasoning) -> None:
        """Test description property returns correct value."""
        assert method.description == INDIRECT_REASONING_METADATA.description

    def test_category_property(self, method: IndirectReasoning) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.SPECIALIZED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: IndirectReasoning) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: IndirectReasoning) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (State Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: IndirectReasoning, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.INDIRECT_REASONING
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "state"

    @pytest.mark.asyncio
    async def test_execute_stores_claim(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() stores the claim."""
        await initialized_method.execute(session, sample_problem)
        assert initialized_method._claim == sample_problem

    @pytest.mark.asyncio
    async def test_execute_content_includes_claim(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute content includes the claim."""
        thought = await initialized_method.execute(session, sample_problem)

        assert "State the Claim" in thought.content
        assert "Reductio ad Absurdum" in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.INDIRECT_REASONING

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: IndirectReasoning, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "state"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Negate Phase Tests ===

    @pytest.mark.asyncio
    async def test_negate_phase(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that negate phase assumes the negation."""
        state_thought = await initialized_method.execute(session, sample_problem)
        negate_thought = await initialized_method.continue_reasoning(session, state_thought)

        assert negate_thought.metadata["phase"] == "negate"
        assert negate_thought.type == ThoughtType.HYPOTHESIS
        assert "Assume the Negation" in negate_thought.content

    @pytest.mark.asyncio
    async def test_negate_stores_negation(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that negate phase stores the negation."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        assert initialized_method._negation != ""

    # === Derive Phase Tests ===

    @pytest.mark.asyncio
    async def test_derive_phase_starts(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that derive phase starts after negate."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive

        assert thought.metadata["phase"] == "derive"
        assert thought.type == ThoughtType.REASONING
        assert initialized_method._derivation_count == 1

    @pytest.mark.asyncio
    async def test_derive_phase_increments_count(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that derivation count increments."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 1

        assert initialized_method._derivation_count == 1

        thought = await initialized_method.continue_reasoning(session, thought)  # derive 2

        assert initialized_method._derivation_count == 2

    # === Contradict Phase Tests ===

    @pytest.mark.asyncio
    async def test_contradict_phase_after_derivations(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that contradict phase occurs after enough derivations."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 1
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 2
        thought = await initialized_method.continue_reasoning(session, thought)  # contradict

        assert thought.metadata["phase"] == "contradict"
        assert thought.type == ThoughtType.VERIFICATION
        assert "Contradiction" in thought.content

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase completes the proof."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 1
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 2
        thought = await initialized_method.continue_reasoning(session, thought)  # contradict
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "Proof Complete" in thought.content or "Conclusion" in thought.content

    # === Done Phase Tests ===

    @pytest.mark.asyncio
    async def test_done_phase_final_synthesis(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that done phase provides final synthesis."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 1
        thought = await initialized_method.continue_reasoning(session, thought)  # derive 2
        thought = await initialized_method.continue_reasoning(session, thought)  # contradict
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude
        thought = await initialized_method.continue_reasoning(session, thought)  # done

        assert thought.metadata["phase"] == "done"
        assert thought.type == ThoughtType.SYNTHESIS
        assert "Final Answer" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "state"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "negate"
        assert thought.type == ThoughtType.HYPOTHESIS

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "derive"
        assert thought.type == ThoughtType.REASONING

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: IndirectReasoning,
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
        initialized_method: IndirectReasoning,
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
        assert "State the Claim" in thought.content

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_ctx
        )

        assert "State the Claim" in thought.content

    # === Heuristic Method Tests ===

    def test_generate_statement_heuristic(self, initialized_method: IndirectReasoning) -> None:
        """Test _generate_statement_heuristic."""
        initialized_method._step_counter = 1
        content = initialized_method._generate_statement_heuristic("Test claim", None)

        assert "State the Claim" in content
        assert "Reductio ad Absurdum" in content

    def test_generate_negation_heuristic(self, initialized_method: IndirectReasoning) -> None:
        """Test _generate_negation_heuristic."""
        initialized_method._step_counter = 2
        content = initialized_method._generate_negation_heuristic(None, None)

        assert "Assume the Negation" in content
        assert initialized_method._negation != ""

    def test_generate_derivation_heuristic(self, initialized_method: IndirectReasoning) -> None:
        """Test _generate_derivation_heuristic."""
        initialized_method._step_counter = 3
        content = initialized_method._generate_derivation_heuristic(1, None, None)

        assert "Derivation #1" in content
        assert "Given: ¬P" in content

    def test_generate_contradiction_heuristic(self, initialized_method: IndirectReasoning) -> None:
        """Test _generate_contradiction_heuristic."""
        initialized_method._step_counter = 4
        content = initialized_method._generate_contradiction_heuristic(None, None)

        assert "Contradiction Found" in content
        assert "Q ∧ ¬Q" in content

    def test_generate_conclusion_heuristic(self, initialized_method: IndirectReasoning) -> None:
        """Test _generate_conclusion_heuristic."""
        initialized_method._step_counter = 5
        initialized_method._derivation_count = 3
        content = initialized_method._generate_conclusion_heuristic(None, None)

        assert "Conclusion by Contradiction" in content
        assert "Proof Complete" in content

    def test_generate_final_synthesis_heuristic(
        self, initialized_method: IndirectReasoning
    ) -> None:
        """Test _generate_final_synthesis_heuristic."""
        initialized_method._step_counter = 6
        initialized_method._derivation_count = 3
        content = initialized_method._generate_final_synthesis_heuristic(None, None)

        assert "Final Answer" in content
        assert "Confidence: Very High" in content

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: IndirectReasoning,
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
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_claim(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks the claim."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["claim"] == sample_problem

    @pytest.mark.asyncio
    async def test_metadata_tracks_derivation_count(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks derivation count."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        thought = await initialized_method.continue_reasoning(session, thought)  # derive

        assert "derivation_count" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: IndirectReasoning,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)  # negate
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)  # derive
        assert thought.confidence == 0.75

    @pytest.mark.asyncio
    async def test_fallback_phase_handling(
        self,
        initialized_method: IndirectReasoning,
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
        assert thought.metadata["phase"] == "contradict"


__all__ = [
    "TestIndirectReasoningMetadata",
    "TestIndirectReasoning",
]
