"""Unit tests for GRPO (Group Relative Policy Optimization) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Sample group phase (candidate generation)
- Compute relative rewards phase
- Optimize policy phase
- Conclude phase
- LLM sampling with fallbacks
- Helper methods
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.grpo import (
    GRPO_METADATA,
    Grpo,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestGrpoMetadata:
    """Tests for GRPO metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert GRPO_METADATA.identifier == MethodIdentifier.GRPO

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert GRPO_METADATA.name == "GRPO"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert GRPO_METADATA.description is not None
        assert "policy" in GRPO_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert GRPO_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {
            "reinforcement-learning",
            "policy-optimization",
            "critic-free",
            "group-comparison",
        }
        assert expected_tags.issubset(GRPO_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert GRPO_METADATA.complexity == 8

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert GRPO_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert GRPO_METADATA.supports_revision is True

    def test_metadata_requires_context(self) -> None:
        """Test that metadata indicates context requirement."""
        assert GRPO_METADATA.requires_context is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert GRPO_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert GRPO_METADATA.max_thoughts == 8

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "complex reasoning" in GRPO_METADATA.best_for


class TestGrpo:
    """Test suite for GRPO reasoning method."""

    @pytest.fixture
    def method(self) -> Grpo:
        """Create method instance."""
        return Grpo()

    @pytest.fixture
    async def initialized_method(self) -> Grpo:
        """Create an initialized method instance."""
        method = Grpo()
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
        mock_response = "Strategy: Direct calculation\nSteps: 3\nConfidence: 0.85"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: Grpo) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, Grpo)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "sample_group"
        assert method._group_candidates == []
        assert method._group_size == 5
        assert method._rewards == []
        assert method._relative_rewards == []
        assert method._baseline_reward == 0.0
        assert method._selected_candidate is None
        assert method._policy_update_step == 0
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: Grpo) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "sample_group"
        assert method._group_candidates == []
        assert method._rewards == []
        assert method._relative_rewards == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: Grpo) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "conclude"
        initialized_method._group_candidates = [{"id": 1}]
        initialized_method._rewards = [0.8]
        initialized_method._relative_rewards = [0.1]
        initialized_method._baseline_reward = 0.7
        initialized_method._selected_candidate = {"id": 1}

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "sample_group"
        assert initialized_method._group_candidates == []
        assert initialized_method._rewards == []
        assert initialized_method._selected_candidate is None

    # === Property Tests ===

    def test_identifier_property(self, method: Grpo) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.GRPO

    def test_name_property(self, method: Grpo) -> None:
        """Test name property returns correct value."""
        assert method.name == "GRPO"

    def test_description_property(self, method: Grpo) -> None:
        """Test description property returns correct value."""
        assert method.description == GRPO_METADATA.description

    def test_category_property(self, method: Grpo) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: Grpo) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: Grpo) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Sample Group Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: Grpo, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.GRPO
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "sample_group"

    @pytest.mark.asyncio
    async def test_execute_generates_candidates(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates candidate group."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._group_candidates) == 5
        for candidate in initialized_method._group_candidates:
            assert "id" in candidate
            assert "reasoning" in candidate
            assert "steps" in candidate
            assert "confidence" in candidate

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.GRPO

    @pytest.mark.asyncio
    async def test_execute_stores_input_in_metadata(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() stores input text in metadata."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["input_text"] == sample_problem

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: Grpo, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "sample_group"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Compute Relative Rewards Phase Tests ===

    @pytest.mark.asyncio
    async def test_compute_rewards_phase(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that compute relative rewards phase computes rewards."""
        sample_thought = await initialized_method.execute(session, sample_problem)
        reward_thought = await initialized_method.continue_reasoning(session, sample_thought)

        assert reward_thought.metadata["phase"] == "compute_relative_rewards"
        assert reward_thought.type == ThoughtType.VERIFICATION
        assert len(initialized_method._rewards) == 5
        assert len(initialized_method._relative_rewards) == 5

    @pytest.mark.asyncio
    async def test_baseline_reward_is_group_average(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that baseline reward is computed as group average."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        expected_baseline = sum(initialized_method._rewards) / len(initialized_method._rewards)
        assert initialized_method._baseline_reward == expected_baseline

    @pytest.mark.asyncio
    async def test_relative_rewards_computed_correctly(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that relative rewards are computed as reward - baseline."""
        thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, thought)

        for i, rel_reward in enumerate(initialized_method._relative_rewards):
            expected = initialized_method._rewards[i] - initialized_method._baseline_reward
            assert abs(rel_reward - expected) < 0.001

    # === Optimize Policy Phase Tests ===

    @pytest.mark.asyncio
    async def test_optimize_policy_phase(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that optimize policy phase selects best candidate."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # rewards
        thought = await initialized_method.continue_reasoning(session, thought)  # optimize

        assert thought.metadata["phase"] == "optimize_policy"
        assert thought.type == ThoughtType.REASONING
        assert initialized_method._selected_candidate is not None

    @pytest.mark.asyncio
    async def test_best_candidate_selected(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that the best candidate is selected based on rewards."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        best_idx = initialized_method._rewards.index(max(initialized_method._rewards))
        expected_id = initialized_method._group_candidates[best_idx]["id"]
        assert initialized_method._selected_candidate["id"] == expected_id

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_produces_final_answer(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)  # conclude

        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION
        assert "GRPO" in thought.content

    @pytest.mark.asyncio
    async def test_conclude_includes_performance_summary(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase includes performance summary."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "Performance Summary" in thought.content
        assert "Baseline reward" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "sample_group"
        assert thought.type == ThoughtType.INITIAL

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "compute_relative_rewards"
        assert thought.type == ThoughtType.VERIFICATION

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "optimize_policy"
        assert thought.type == ThoughtType.REASONING

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "conclude"
        assert thought.type == ThoughtType.CONCLUSION

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        # Should call sample for each candidate
        assert mock_execution_context.sample.call_count == 5

    @pytest.mark.asyncio
    async def test_sampling_fallback_on_error(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)

        # Should use fallback candidates
        assert len(initialized_method._group_candidates) == 5

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        await initialized_method.execute(session, sample_problem, execution_context=no_sample_ctx)

        assert len(initialized_method._group_candidates) == 5

    # === Helper Method Tests ===

    def test_compute_std_empty(self, initialized_method: Grpo) -> None:
        """Test _compute_std with empty list."""
        assert initialized_method._compute_std([]) == 0.0

    def test_compute_std_single_value(self, initialized_method: Grpo) -> None:
        """Test _compute_std with single value."""
        assert initialized_method._compute_std([5.0]) == 0.0

    def test_compute_std_multiple_values(self, initialized_method: Grpo) -> None:
        """Test _compute_std with multiple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = initialized_method._compute_std(values)
        # Expected std for [1,2,3,4,5] is ~1.414
        assert 1.4 < std < 1.5

    @pytest.mark.asyncio
    async def test_generate_candidate_group_fallback(self, initialized_method: Grpo) -> None:
        """Test _generate_candidate_group fallback without context."""
        candidates = await initialized_method._generate_candidate_group("Test problem", None)

        assert len(candidates) == 5
        for c in candidates:
            assert "id" in c
            assert "reasoning" in c
            assert "steps" in c
            assert "confidence" in c

    @pytest.mark.asyncio
    async def test_compute_candidate_rewards_fallback(self, initialized_method: Grpo) -> None:
        """Test _compute_candidate_rewards fallback without context."""
        candidates = [{"id": i, "confidence": 0.7 + i * 0.05} for i in range(5)]

        rewards = await initialized_method._compute_candidate_rewards(
            "Test problem", candidates, None
        )

        assert len(rewards) == 5
        # Fallback uses confidence values
        for i, reward in enumerate(rewards):
            assert reward == candidates[i]["confidence"]

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: Grpo,
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
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_metadata_tracks_baseline_reward(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that metadata tracks baseline reward."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "baseline_reward" in thought.metadata

    @pytest.mark.asyncio
    async def test_confidence_progression(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that confidence increases through phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.confidence == 0.7

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.75

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.confidence == 0.82

    @pytest.mark.asyncio
    async def test_unknown_phase_fallback(
        self,
        initialized_method: Grpo,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that unknown phase falls back to conclude."""
        await initialized_method.execute(session, sample_problem)

        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 3
        mock_thought.metadata = {"phase": "unknown_phase"}

        thought = await initialized_method.continue_reasoning(session, mock_thought)
        assert thought.metadata["phase"] == "conclude"


__all__ = [
    "TestGrpoMetadata",
    "TestGrpo",
]
