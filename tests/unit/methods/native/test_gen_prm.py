"""Unit tests for Generative Process Reward Model (GenPRM) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Generate phase (reasoning step generation)
- Verify phase (verification chain creation)
- Score phase (process reward computation)
- Select phase (best path selection)
- Conclude phase
- LLM sampling with fallbacks
- Helper methods
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.gen_prm import (
    GEN_PRM_METADATA,
    GenPRM,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestGenPRMMetadata:
    """Tests for GenPRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert GEN_PRM_METADATA.identifier == MethodIdentifier.GEN_PRM

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert GEN_PRM_METADATA.name == "Generative Process Reward Model"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert GEN_PRM_METADATA.description is not None
        assert "verification" in GEN_PRM_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert GEN_PRM_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"process-reward", "verification", "generative", "test-time", "scaling"}
        assert expected_tags.issubset(GEN_PRM_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert GEN_PRM_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert GEN_PRM_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert GEN_PRM_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert GEN_PRM_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert GEN_PRM_METADATA.max_thoughts == 12

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "mathematical reasoning" in GEN_PRM_METADATA.best_for


class TestGenPRM:
    """Test suite for GenPRM reasoning method."""

    @pytest.fixture
    def method(self) -> GenPRM:
        """Create method instance."""
        return GenPRM()

    @pytest.fixture
    async def initialized_method(self) -> GenPRM:
        """Create an initialized method instance."""
        method = GenPRM()
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
        return "Solve the equation: 2x + 5 = 13"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = "1. Subtract 5 from both sides\n2. Divide by 2\n3. x = 4"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: GenPRM) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, GenPRM)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._reasoning_steps == []
        assert method._verification_chains == []
        assert method._process_rewards == []
        assert method._execution_context is None

    def test_default_candidates_constant(self, method: GenPRM) -> None:
        """Test that DEFAULT_CANDIDATES constant is defined."""
        assert GenPRM.DEFAULT_CANDIDATES == 3

    def test_initialization_with_custom_candidates(self) -> None:
        """Test initialization with custom number of candidates."""
        method = GenPRM(num_candidates=5)
        assert method._num_candidates == 5

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: GenPRM) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._reasoning_steps == []
        assert method._verification_chains == []
        assert method._process_rewards == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: GenPRM) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._reasoning_steps = [{"id": 1, "content": "test"}]
        initialized_method._verification_chains = [{"step_id": 1}]
        initialized_method._process_rewards = [0.8, 0.9]

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "generate"
        assert initialized_method._reasoning_steps == []
        assert initialized_method._verification_chains == []
        assert initialized_method._process_rewards == []

    # === Property Tests ===

    def test_identifier_property(self, method: GenPRM) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.GEN_PRM

    def test_name_property(self, method: GenPRM) -> None:
        """Test name property returns correct value."""
        assert method.name == "Generative Process Reward Model"

    def test_description_property(self, method: GenPRM) -> None:
        """Test description property returns correct value."""
        assert method.description == GEN_PRM_METADATA.description

    def test_category_property(self, method: GenPRM) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: GenPRM) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: GenPRM) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Generate Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: GenPRM, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.GEN_PRM
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "generate"

    @pytest.mark.asyncio
    async def test_execute_generates_reasoning_steps(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates reasoning steps."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._reasoning_steps) == 4
        for step in initialized_method._reasoning_steps:
            assert "id" in step
            assert "content" in step
            assert "confidence" in step

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.GEN_PRM

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: GenPRM,
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
        self, method: GenPRM, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "generate"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Verify Phase Tests ===

    @pytest.mark.asyncio
    async def test_verify_phase_creates_verification_chains(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that verify phase creates verification chains."""
        generate_thought = await initialized_method.execute(session, sample_problem)
        verify_thought = await initialized_method.continue_reasoning(session, generate_thought)

        assert verify_thought is not None
        assert verify_thought.metadata["phase"] == "verify"
        assert verify_thought.type == ThoughtType.VERIFICATION
        assert len(initialized_method._verification_chains) == 4

    @pytest.mark.asyncio
    async def test_verification_chains_have_verdicts(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that verification chains include verdicts."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        for chain in initialized_method._verification_chains:
            assert "step_id" in chain
            assert "verification" in chain
            assert "verdict" in chain

    # === Score Phase Tests ===

    @pytest.mark.asyncio
    async def test_score_phase_computes_rewards(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that score phase computes process rewards."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # verify
        thought = await initialized_method.continue_reasoning(session, thought)  # score

        assert thought.metadata["phase"] == "score"
        assert thought.type == ThoughtType.REASONING
        assert len(initialized_method._process_rewards) == 4

    @pytest.mark.asyncio
    async def test_process_rewards_are_valid(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that process rewards are in valid range."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        for reward in initialized_method._process_rewards:
            assert 0.0 <= reward <= 1.0

    # === Select Phase Tests ===

    @pytest.mark.asyncio
    async def test_select_phase_chooses_best_path(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that select phase chooses best path."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # verify
        thought = await initialized_method.continue_reasoning(session, thought)  # score
        thought = await initialized_method.continue_reasoning(session, thought)  # select

        assert thought.metadata["phase"] == "select"
        assert thought.type == ThoughtType.SYNTHESIS
        assert "SELECTED" in thought.content

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_produces_final_answer(
        self,
        initialized_method: GenPRM,
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
        assert "GenPRM Complete" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "generate"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "verify"
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "score"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "select"
        assert thought.type == ThoughtType.SYNTHESIS

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: GenPRM,
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
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails with expected exceptions."""
        # Use ValueError which is an expected exception type that triggers fallback
        # (along with TimeoutError, ConnectionError, OSError)
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ValueError("LLM error"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)

        # Should use fallback steps
        assert len(initialized_method._reasoning_steps) == 4

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        await initialized_method.execute(session, sample_problem, execution_context=no_sample_ctx)

        assert len(initialized_method._reasoning_steps) == 4

    # === Helper Method Tests ===

    def test_parse_reasoning_steps(self, initialized_method: GenPRM) -> None:
        """Test _parse_reasoning_steps helper."""
        response = "1. First step\n2. Second step\n3. Third step\n4. Fourth step"
        steps = initialized_method._parse_reasoning_steps(response, 4)

        assert len(steps) == 4
        assert all("id" in s for s in steps)
        assert all("content" in s for s in steps)
        assert all("confidence" in s for s in steps)

    def test_parse_reasoning_steps_pads_to_num_steps(self, initialized_method: GenPRM) -> None:
        """Test that parsing pads to requested number of steps."""
        response = "1. Only one step"
        steps = initialized_method._parse_reasoning_steps(response, 4)

        assert len(steps) == 4

    def test_generate_fallback_steps(self, initialized_method: GenPRM) -> None:
        """Test _generate_fallback_steps helper."""
        steps = initialized_method._generate_fallback_steps("Test problem", 4)

        assert len(steps) == 4
        for i, step in enumerate(steps):
            assert step["id"] == i + 1
            assert "Test problem" in step["content"]

    def test_generate_fallback_verifications(self, initialized_method: GenPRM) -> None:
        """Test _generate_fallback_verifications helper."""
        initialized_method._reasoning_steps = [
            {"id": 1, "content": "Step 1", "confidence": 0.7},
            {"id": 2, "content": "Step 2", "confidence": 0.75},
        ]

        verifications = initialized_method._generate_fallback_verifications()

        assert len(verifications) == 2
        for v in verifications:
            assert "step_id" in v
            assert "verification" in v
            assert "verdict" in v

    def test_generate_fallback_rewards(self, initialized_method: GenPRM) -> None:
        """Test _generate_fallback_rewards helper."""
        initialized_method._reasoning_steps = [
            {"id": 1, "content": "Step 1", "confidence": 0.7},
            {"id": 2, "content": "Step 2", "confidence": 0.75},
        ]

        rewards = initialized_method._generate_fallback_rewards()

        assert len(rewards) == 2
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_extract_verdict_correct(self, initialized_method: GenPRM) -> None:
        """Test _extract_verdict extracts 'correct' verdict."""
        text = "This step is correct because..."
        assert initialized_method._extract_verdict(text) == "correct"

    def test_extract_verdict_incorrect(self, initialized_method: GenPRM) -> None:
        """Test _extract_verdict extracts 'incorrect' verdict."""
        text = "This step is incorrect because..."
        assert initialized_method._extract_verdict(text) == "incorrect"

    def test_extract_verdict_uncertain(self, initialized_method: GenPRM) -> None:
        """Test _extract_verdict extracts 'uncertain' verdict."""
        text = "The correctness is uncertain..."
        assert initialized_method._extract_verdict(text) == "uncertain"

    def test_extract_verdict_default(self, initialized_method: GenPRM) -> None:
        """Test _extract_verdict defaults to 'correct'."""
        text = "Some unrelated text"
        assert initialized_method._extract_verdict(text) == "correct"

    def test_extract_score_decimal(self, initialized_method: GenPRM) -> None:
        """Test _extract_score extracts decimal score."""
        text = "The score is 0.85"
        assert initialized_method._extract_score(text) == 0.85

    def test_extract_score_percentage(self, initialized_method: GenPRM) -> None:
        """Test _extract_score extracts percentage."""
        text = "I rate this 75%"
        assert initialized_method._extract_score(text) == 0.75

    def test_extract_score_clamped(self, initialized_method: GenPRM) -> None:
        """Test _extract_score clamps to valid range."""
        text = "The score is 1.5"  # Invalid, but should parse 1.5 and clamp
        score = initialized_method._extract_score(text)
        assert 0.0 <= score <= 1.0

    def test_extract_score_default(self, initialized_method: GenPRM) -> None:
        """Test _extract_score returns default on failure."""
        text = "No numbers here"
        assert initialized_method._extract_score(text) == 0.85

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: GenPRM,
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
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_rewards(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks rewards."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "rewards" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.6

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.75

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.85

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.88

    @pytest.mark.asyncio
    async def test_empty_rewards_handled(
        self,
        initialized_method: GenPRM,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that empty rewards are handled gracefully."""
        await initialized_method.execute(session, sample_problem)

        # Set up verification chains to match reasoning steps (required for zip)
        initialized_method._verification_chains = [
            {"step_id": s["id"], "verification": "test", "verdict": "correct"}
            for s in initialized_method._reasoning_steps
        ]
        # Manually clear rewards
        initialized_method._process_rewards = []

        # Create a mock thought for score phase
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 1
        mock_thought.metadata = {"phase": "verify"}

        # Should not raise error
        thought = await initialized_method.continue_reasoning(session, mock_thought)
        assert "Cumulative reward" in thought.content


__all__ = [
    "TestGenPRMMetadata",
    "TestGenPRM",
]
