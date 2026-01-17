"""Unit tests for GAR (Generator-Adversarial Reasoning) method.

Tests cover:
- Metadata validation
- Initialization and state management
- Generation phase (candidate creation)
- Discrimination phase (scoring)
- Update phase (refinement)
- Conclusion phase (final answer)
- Heuristic fallbacks
- Sampling methods
- Error handling and edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.gar import GAR_METADATA, Gar
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestGarMetadata:
    """Tests for GAR metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert GAR_METADATA.identifier == MethodIdentifier.GAR

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert GAR_METADATA.name == "GAR"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert GAR_METADATA.description is not None
        assert "adversarial" in GAR_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert GAR_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"adversarial", "generator", "discriminator", "iterative"}
        assert expected_tags.issubset(GAR_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert GAR_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert GAR_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert GAR_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert GAR_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert GAR_METADATA.max_thoughts == 8

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "robustness" in GAR_METADATA.best_for
        assert "candidate selection" in GAR_METADATA.best_for

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies what method is not recommended for."""
        assert "simple queries" in GAR_METADATA.not_recommended_for


class TestGar:
    """Test suite for Gar reasoning method."""

    @pytest.fixture
    def method(self) -> Gar:
        """Create method instance."""
        return Gar()

    @pytest.fixture
    async def initialized_method(self) -> Gar:
        """Create an initialized Gar method instance."""
        method = Gar()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> Session:
        """Create an active session for testing."""
        return Session().start()

    @pytest.fixture
    def sample_problem(self) -> str:
        """Provide a sample problem for testing."""
        return "What is 5 multiplied by 3, then add 2?"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_ctx.sample = AsyncMock(
            side_effect=[
                # For _sample_generate_candidates
                "[C1] Direct: 5*3+2=17 -> 17\n[C2] Step-wise: 5*3=15, then +2=17 -> 17\n[C3] Incorrect: 5+3*2=16 -> 16",
                # For _sample_discriminate_candidates
                "[C1] 0.8\n[C2] 0.9\n[C3] 0.3",
                # For _sample_update_candidate
                "Refined: Candidate 2 is stronger. It clearly breaks down the multiplication before addition.",
                # For _sample_final_answer
                "17",
            ]
        )
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_state(self, method: Gar) -> None:
        """Test method initializes with correct default state."""
        assert method is not None
        assert isinstance(method, Gar)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidates == []
        assert method._scores == []
        assert method._iteration == 0
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: Gar) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._candidates == []
        assert method._scores == []
        assert method._iteration == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: Gar) -> None:
        """Test that initialize() resets state when called multiple times."""
        # Modify state
        initialized_method._step_counter = 5
        initialized_method._current_phase = "discriminate"
        initialized_method._candidates = [{"id": "C1", "reasoning": "r", "answer": "a"}]
        initialized_method._scores = [0.5]
        initialized_method._iteration = 2

        # Re-initialize
        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "generate"
        assert initialized_method._candidates == []
        assert initialized_method._scores == []
        assert initialized_method._iteration == 0

    # === Property Tests ===

    def test_identifier_property(self, method: Gar) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.GAR

    def test_name_property(self, method: Gar) -> None:
        """Test name property returns correct value."""
        assert method.name == "GAR"

    def test_description_property(self, method: Gar) -> None:
        """Test description property returns correct value."""
        assert method.description == GAR_METADATA.description

    def test_category_property(self, method: Gar) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: Gar) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: Gar) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Generate Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: Gar, session: Session, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized before execution"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought in generate phase."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.GAR
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "generate"
        assert thought.metadata["iteration"] == 1
        assert sample_problem in thought.content
        assert "Candidates:" in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.GAR

    @pytest.mark.asyncio
    async def test_execute_increments_thought_count(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() increments the session thought count."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)
        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_generates_candidates_heuristic(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates candidates using heuristic when no context."""
        thought = await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._candidates) >= 2
        assert all(
            "id" in c and "reasoning" in c and "answer" in c for c in initialized_method._candidates
        )
        assert "C1" in thought.content
        assert "C2" in thought.content

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() uses LLM sampling when context available."""
        thought = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        assert thought.metadata["sampled"] is True
        mock_execution_context.sample.assert_called_once()
        assert len(initialized_method._candidates) >= 2

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_heuristic_on_sampling_error(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() falls back to heuristic if sampling fails."""
        mock_execution_context.sample.side_effect = ConnectionError("LLM connection failed")

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        assert thought is not None
        # Should still have candidates from heuristic
        assert len(initialized_method._candidates) >= 2

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() stores the execution context for later use."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        assert initialized_method._execution_context is mock_execution_context

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: Gar, session: Session
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.metadata = {"phase": "generate"}

        with pytest.raises(RuntimeError, match="must be initialized before continuation"):
            await method.continue_reasoning(session, mock_thought)

    # === Discriminate Phase Tests ===

    @pytest.mark.asyncio
    async def test_discriminate_phase_scores_candidates(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that discrimination phase scores candidates."""
        # Execute to create generate phase thought
        initial_thought = await initialized_method.execute(session, sample_problem)
        assert initial_thought.metadata["phase"] == "generate"

        # Continue to discriminate phase
        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)

        assert discriminate_thought is not None
        assert discriminate_thought.metadata["phase"] == "discriminate"
        assert "Scores:" in discriminate_thought.content
        assert "Best:" in discriminate_thought.content
        assert len(initialized_method._scores) == len(initialized_method._candidates)
        assert discriminate_thought.type == ThoughtType.VERIFICATION
        assert discriminate_thought.parent_id == initial_thought.id

    @pytest.mark.asyncio
    async def test_discriminate_phase_with_sampling(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test discrimination phase uses LLM sampling when available."""
        initial_thought = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        discriminate_thought = await initialized_method.continue_reasoning(
            session, initial_thought, execution_context=mock_execution_context
        )

        assert discriminate_thought.metadata["sampled"] is True
        assert len(initialized_method._scores) > 0

    @pytest.mark.asyncio
    async def test_discriminate_phase_identifies_best_candidate(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that discrimination phase identifies the best candidate."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)

        best_idx = initialized_method._scores.index(max(initialized_method._scores))
        best_id = initialized_method._candidates[best_idx]["id"]
        best_score = max(initialized_method._scores)

        assert best_id in discriminate_thought.content
        assert f"{best_score:.2f}" in discriminate_thought.content

    # === Update Phase Tests ===

    @pytest.mark.asyncio
    async def test_update_phase_refines_candidate(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that update phase refines the best candidate."""
        # Generate phase
        initial_thought = await initialized_method.execute(session, sample_problem)
        # Discriminate phase
        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)
        # Update phase
        update_thought = await initialized_method.continue_reasoning(session, discriminate_thought)

        assert update_thought is not None
        assert update_thought.metadata["phase"] == "update"
        assert update_thought.type == ThoughtType.REVISION
        assert update_thought.parent_id == discriminate_thought.id
        assert update_thought.confidence == 0.85

    @pytest.mark.asyncio
    async def test_update_phase_with_sampling(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test update phase uses LLM sampling when available."""
        initial_thought = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        discriminate_thought = await initialized_method.continue_reasoning(
            session, initial_thought, execution_context=mock_execution_context
        )
        update_thought = await initialized_method.continue_reasoning(
            session, discriminate_thought, execution_context=mock_execution_context
        )

        assert update_thought.metadata["sampled"] is True
        assert "Refined" in update_thought.content or "refined" in update_thought.content.lower()

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_produces_final_answer(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        # Generate phase
        initial_thought = await initialized_method.execute(session, sample_problem)
        # Discriminate phase
        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)
        # Update phase
        update_thought = await initialized_method.continue_reasoning(session, discriminate_thought)
        # Conclude phase
        conclude_thought = await initialized_method.continue_reasoning(session, update_thought)

        assert conclude_thought is not None
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.parent_id == update_thought.id
        assert "Final Answer" in conclude_thought.content
        assert "GAR Complete" in conclude_thought.content

    @pytest.mark.asyncio
    async def test_conclude_phase_confidence_from_best_score(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test conclude phase confidence matches best score."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)
        update_thought = await initialized_method.continue_reasoning(session, discriminate_thought)
        conclude_thought = await initialized_method.continue_reasoning(session, update_thought)

        expected_confidence = (
            max(initialized_method._scores) if initialized_method._scores else 0.91
        )
        assert conclude_thought.confidence == expected_confidence

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        # Generate
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.metadata["phase"] == "generate"
        assert thought1.step_number == 1

        # Discriminate
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.metadata["phase"] == "discriminate"
        assert thought2.step_number == 2
        assert thought2.depth == 1

        # Update
        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.metadata["phase"] == "update"
        assert thought3.step_number == 3
        assert thought3.depth == 2

        # Conclude
        thought4 = await initialized_method.continue_reasoning(session, thought3)
        assert thought4.metadata["phase"] == "conclude"
        assert thought4.step_number == 4
        assert thought4.depth == 3

        # Verify chain
        assert thought2.parent_id == thought1.id
        assert thought3.parent_id == thought2.id
        assert thought4.parent_id == thought3.id

        # Verify session has all thoughts
        assert session.thought_count == 4

    @pytest.mark.asyncio
    async def test_full_pipeline_with_sampling(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test complete pipeline with LLM sampling."""
        thought1 = await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        thought2 = await initialized_method.continue_reasoning(
            session, thought1, execution_context=mock_execution_context
        )
        thought3 = await initialized_method.continue_reasoning(
            session, thought2, execution_context=mock_execution_context
        )
        thought4 = await initialized_method.continue_reasoning(
            session, thought3, execution_context=mock_execution_context
        )

        # All thoughts should be marked as sampled
        assert all(t.metadata["sampled"] for t in [thought1, thought2, thought3, thought4])

    # === Heuristic Method Tests ===

    def test_generate_candidates_heuristic(self, method: Gar) -> None:
        """Test heuristic candidate generation."""
        candidates = method._generate_candidates_heuristic("test problem")

        assert len(candidates) == 2
        assert all("id" in c for c in candidates)
        assert all("reasoning" in c for c in candidates)
        assert all("answer" in c for c in candidates)
        assert candidates[0]["id"] == "C1"
        assert candidates[1]["id"] == "C2"

    def test_discriminate_candidates_heuristic(self, method: Gar) -> None:
        """Test heuristic candidate scoring."""
        method._candidates = [
            {"id": "C1", "reasoning": "Short", "answer": "A"},
            {"id": "C2", "reasoning": "This is a much longer reasoning path", "answer": "B"},
        ]

        scores = method._discriminate_candidates_heuristic()

        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Longer reasoning should score higher
        assert scores[1] > scores[0]

    def test_discriminate_candidates_heuristic_empty(self, method: Gar) -> None:
        """Test heuristic scoring with empty candidates returns defaults."""
        method._candidates = []
        scores = method._discriminate_candidates_heuristic()
        assert scores == [0.82, 0.91]

    # === Sampling Method Tests ===

    @pytest.mark.asyncio
    async def test_sample_generate_candidates_parses_response(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_generate_candidates parses LLM response correctly."""
        initialized_method._execution_context = mock_execution_context
        mock_execution_context.sample = AsyncMock(
            return_value="[C1] First approach -> answer1\n[C2] Second approach -> answer2"
        )

        candidates = await initialized_method._sample_generate_candidates("test")

        assert len(candidates) >= 2
        assert all("id" in c and "reasoning" in c and "answer" in c for c in candidates)

    @pytest.mark.asyncio
    async def test_sample_generate_candidates_fallback_on_insufficient(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test fallback when sampling returns insufficient candidates."""
        initialized_method._execution_context = mock_execution_context
        mock_execution_context.sample = AsyncMock(return_value="[C1] Only one -> single")

        candidates = await initialized_method._sample_generate_candidates("test")

        # Should fall back to heuristic which returns 2
        assert len(candidates) >= 2

    @pytest.mark.asyncio
    async def test_sample_discriminate_candidates_parses_scores(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_discriminate_candidates parses scores correctly."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._candidates = [
            {"id": "C1", "reasoning": "R1", "answer": "A1"},
            {"id": "C2", "reasoning": "R2", "answer": "A2"},
        ]
        mock_execution_context.sample = AsyncMock(return_value="[C1] 0.75\n[C2] 0.85")

        scores = await initialized_method._sample_discriminate_candidates()

        assert len(scores) == 2
        assert scores[0] == 0.75
        assert scores[1] == 0.85

    @pytest.mark.asyncio
    async def test_sample_discriminate_clamps_scores(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test that scores are clamped to [0, 1] range."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._candidates = [
            {"id": "C1", "reasoning": "R1", "answer": "A1"},
            {"id": "C2", "reasoning": "R2", "answer": "A2"},
        ]
        mock_execution_context.sample = AsyncMock(return_value="[C1] 1.5\n[C2] -0.5")

        scores = await initialized_method._sample_discriminate_candidates()

        assert scores[0] == 1.0  # Clamped from 1.5
        assert scores[1] == 0.0  # Clamped from -0.5

    @pytest.mark.asyncio
    async def test_sample_update_candidate_returns_refinement(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_update_candidate returns refinement text."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._candidates = [{"id": "C1", "reasoning": "R1", "answer": "A1"}]
        initialized_method._scores = [0.8]
        mock_execution_context.sample = AsyncMock(return_value="Improved reasoning here")

        result = await initialized_method._sample_update_candidate()

        assert result == "Improved reasoning here"

    @pytest.mark.asyncio
    async def test_sample_final_answer_returns_answer(
        self, initialized_method: Gar, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_final_answer returns final answer."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._candidates = [{"id": "C1", "reasoning": "R1", "answer": "42"}]
        initialized_method._scores = [0.9]
        mock_execution_context.sample = AsyncMock(return_value="  42  ")

        result = await initialized_method._sample_final_answer()

        assert result == "42"  # Should be stripped

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_updates_execution_context(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that continue_reasoning can update execution context."""
        initial_thought = await initialized_method.execute(session, sample_problem)

        new_context = MagicMock()
        new_context.can_sample = True
        new_context.sample = AsyncMock(return_value="[C1] 0.9\n[C2] 0.8")

        await initialized_method.continue_reasoning(
            session, initial_thought, execution_context=new_context
        )

        assert initialized_method._execution_context is new_context

    @pytest.mark.asyncio
    async def test_empty_candidates_during_discriminate(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test handling of empty candidates during discrimination."""
        # Create a generate phase thought manually
        initial_thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GAR,
            content="Test",
            step_number=1,
            depth=0,
            metadata={"phase": "generate", "iteration": 1},
        )
        session.add_thought(initial_thought)
        initialized_method._step_counter = 1
        initialized_method._candidates = []
        initialized_method._scores = []

        discriminate_thought = await initialized_method.continue_reasoning(session, initial_thought)

        # Should handle gracefully
        assert discriminate_thought is not None
        assert discriminate_thought.metadata["phase"] == "discriminate"

    @pytest.mark.asyncio
    async def test_sampling_failure_at_each_phase(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that sampling failures are handled gracefully at each phase."""
        failing_context = MagicMock()
        failing_context.can_sample = True
        failing_context.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        # Generate phase should fall back to heuristic
        thought1 = await initialized_method.execute(
            session, sample_problem, execution_context=failing_context
        )
        assert thought1 is not None

        # Discriminate phase should fall back to heuristic
        thought2 = await initialized_method.continue_reasoning(
            session, thought1, execution_context=failing_context
        )
        assert thought2 is not None

        # Update phase should fall back to default message
        thought3 = await initialized_method.continue_reasoning(
            session, thought2, execution_context=failing_context
        )
        assert thought3 is not None

        # Conclude phase should fall back to default answer
        thought4 = await initialized_method.continue_reasoning(
            session, thought3, execution_context=failing_context
        )
        assert thought4 is not None
        assert "Final Answer" in thought4.content

    @pytest.mark.asyncio
    async def test_context_with_can_sample_false(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that method uses heuristic when can_sample is False."""
        no_sample_context = MagicMock()
        no_sample_context.can_sample = False

        thought = await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_context
        )

        assert thought.metadata["sampled"] is False
        # Should still work with heuristic
        assert len(initialized_method._candidates) >= 2

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that step counter increments correctly through phases."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert initialized_method._step_counter == 2

        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert initialized_method._step_counter == 3

        await initialized_method.continue_reasoning(session, thought3)
        assert initialized_method._step_counter == 4

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: Gar,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.depth == 2

        thought4 = await initialized_method.continue_reasoning(session, thought3)
        assert thought4.depth == 3


class TestGarAdversarialDynamics:
    """Tests specifically for GAR adversarial dynamics (L3.1.4)."""

    @pytest.fixture
    def method(self) -> Gar:
        """Create method instance."""
        return Gar()

    @pytest.fixture
    async def initialized_method(self) -> Gar:
        """Create an initialized method."""
        method = Gar()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> Session:
        """Create an active session."""
        return Session().start()

    @pytest.mark.asyncio
    async def test_generator_discriminator_interaction(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test generator-discriminator adversarial interaction."""
        problem = "What is 10 divided by 2?"
        thought1 = await initialized_method.execute(session, problem)

        # Generator produces candidates
        assert len(initialized_method._candidates) >= 2

        await initialized_method.continue_reasoning(session, thought1)

        # Discriminator scores them
        assert len(initialized_method._scores) == len(initialized_method._candidates)
        assert all(0 <= s <= 1 for s in initialized_method._scores)

    @pytest.mark.asyncio
    async def test_best_candidate_selection(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test that best candidate is correctly selected."""
        problem = "Test problem"
        thought1 = await initialized_method.execute(session, problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)

        best_idx = initialized_method._scores.index(max(initialized_method._scores))
        best_id = initialized_method._candidates[best_idx]["id"]

        assert best_id in thought2.content

    @pytest.mark.asyncio
    async def test_refinement_targets_best_candidate(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test that update phase refines the best candidate."""
        problem = "Test problem"
        thought1 = await initialized_method.execute(session, problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        thought3 = await initialized_method.continue_reasoning(session, thought2)

        # Update phase should refine best candidate
        assert thought3.type == ThoughtType.REVISION
        assert thought3.confidence > thought1.confidence


class TestGarConvergence:
    """Tests for GAR convergence criteria (L3.1.5)."""

    @pytest.fixture
    def method(self) -> Gar:
        """Create method instance."""
        return Gar()

    @pytest.fixture
    async def initialized_method(self) -> Gar:
        """Create an initialized method."""
        method = Gar()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> Session:
        """Create an active session."""
        return Session().start()

    @pytest.mark.asyncio
    async def test_pipeline_converges_to_conclusion(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test that pipeline converges to a conclusion."""
        problem = "Simple test"
        thought1 = await initialized_method.execute(session, problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        thought3 = await initialized_method.continue_reasoning(session, thought2)
        thought4 = await initialized_method.continue_reasoning(session, thought3)

        assert thought4.type == ThoughtType.CONCLUSION
        assert thought4.metadata["phase"] == "conclude"

    @pytest.mark.asyncio
    async def test_final_confidence_based_on_scores(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test that final confidence is based on best discriminator score."""
        problem = "Test"
        thought1 = await initialized_method.execute(session, problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        thought3 = await initialized_method.continue_reasoning(session, thought2)
        thought4 = await initialized_method.continue_reasoning(session, thought3)

        expected_confidence = max(initialized_method._scores)
        assert thought4.confidence == expected_confidence

    @pytest.mark.asyncio
    async def test_iteration_tracking(self, initialized_method: Gar, session: Session) -> None:
        """Test that iteration is tracked throughout."""
        problem = "Test"
        thought1 = await initialized_method.execute(session, problem)
        assert thought1.metadata["iteration"] == 1

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.metadata["iteration"] == 1  # Same iteration

    @pytest.mark.asyncio
    async def test_quality_score_progression(
        self, initialized_method: Gar, session: Session
    ) -> None:
        """Test that quality scores progress through phases."""
        problem = "Test"
        thought1 = await initialized_method.execute(session, problem)
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        thought3 = await initialized_method.continue_reasoning(session, thought2)
        await initialized_method.continue_reasoning(session, thought3)

        # Quality should generally improve through phases
        assert thought1.quality_score == 0.65  # Generate
        assert thought2.quality_score == 0.78  # Discriminate
        assert thought3.quality_score == 0.85  # Update
        # Conclude uses best score


__all__ = [
    "TestGarMetadata",
    "TestGar",
    "TestGarAdversarialDynamics",
    "TestGarConvergence",
]
