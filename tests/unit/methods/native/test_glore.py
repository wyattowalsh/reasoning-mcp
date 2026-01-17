"""Unit tests for GLoRe (Global and Local Refinements) reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Generate phase (solution step creation)
- Global phase (overall ORM assessment)
- Local phase (step-level scoring)
- Refine phase (targeted refinements)
- Conclusion phase (final answer)
- Heuristic fallbacks
- Sampling methods
- Error handling and edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.glore import GLORE_METADATA, GLoRe
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestGloreMetadata:
    """Tests for GLoRe metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert GLORE_METADATA.identifier == MethodIdentifier.GLORE

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert GLORE_METADATA.name == "GLoRe"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert GLORE_METADATA.description is not None
        assert "refinement" in GLORE_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert GLORE_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"refinement", "orm", "global", "local", "process-supervision"}
        assert expected_tags.issubset(GLORE_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert GLORE_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert GLORE_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert GLORE_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert GLORE_METADATA.min_thoughts == 5

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert GLORE_METADATA.max_thoughts == 9

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "math reasoning" in GLORE_METADATA.best_for
        assert "step verification" in GLORE_METADATA.best_for
        assert "process supervision" in GLORE_METADATA.best_for

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies what method is not recommended for."""
        assert "simple queries" in GLORE_METADATA.not_recommended_for


class TestGlore:
    """Test suite for GLoRe reasoning method."""

    @pytest.fixture
    def method(self) -> GLoRe:
        """Create method instance."""
        return GLoRe()

    @pytest.fixture
    async def initialized_method(self) -> GLoRe:
        """Create an initialized GLoRe method instance."""
        method = GLoRe()
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
                # For _sample_solution_steps
                "Step 1: Parse problem\nStep 2: Compute multiplication\nStep 3: Add",
                # For _sample_global_orm_score
                "0.85",
                # For _sample_local_orm_scores
                "Step 1: 0.9, Step 2: 0.8, Step 3: 0.75",
                # For _sample_refinements
                "Refinement for step 3: verified calculation",
            ]
        )
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_state(self, method: GLoRe) -> None:
        """Test method initializes with correct default state."""
        assert method is not None
        assert isinstance(method, GLoRe)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._solution_steps == []
        assert method._global_score == 0.0
        assert method._local_scores == []
        assert method._refinements == []
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: GLoRe) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "generate"
        assert method._solution_steps == []
        assert method._global_score == 0.0
        assert method._local_scores == []
        assert method._refinements == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: GLoRe) -> None:
        """Test that initialize() resets state when called multiple times."""
        # Modify state
        initialized_method._step_counter = 5
        initialized_method._current_phase = "local"
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]
        initialized_method._global_score = 0.9
        initialized_method._local_scores = [{"step": 1, "score": 0.9, "issue": None}]
        initialized_method._refinements = [
            {"step": 1, "original": "o", "refined": "r", "improvement": 0.1}
        ]

        # Re-initialize
        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "generate"
        assert initialized_method._solution_steps == []
        assert initialized_method._global_score == 0.0
        assert initialized_method._local_scores == []
        assert initialized_method._refinements == []

    # === Property Tests ===

    def test_identifier_property(self, method: GLoRe) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.GLORE

    def test_name_property(self, method: GLoRe) -> None:
        """Test name property returns correct value."""
        assert method.name == "GLoRe"

    def test_description_property(self, method: GLoRe) -> None:
        """Test description property returns correct value."""
        assert method.description == GLORE_METADATA.description

    def test_category_property(self, method: GLoRe) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: GLoRe) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: GLoRe) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Generate Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: GLoRe, session: Session, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized before execution"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought in generate phase."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.GLORE
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "generate"
        assert thought.metadata["steps"] >= 1
        assert sample_problem in thought.content
        assert "Solution Steps:" in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.GLORE

    @pytest.mark.asyncio
    async def test_execute_increments_thought_count(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() increments the session thought count."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)
        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_generates_solution_steps_heuristic(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that execute() generates solution steps using heuristic when no context."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._solution_steps) >= 5
        assert all(
            "step" in s and "content" in s and "type" in s
            for s in initialized_method._solution_steps
        )
        # Verify step types are valid
        valid_types = {"setup", "computation", "verification", "conclusion"}
        assert all(s["type"] in valid_types for s in initialized_method._solution_steps)

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: GLoRe,
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
        self, method: GLoRe, session: Session
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.metadata = {"phase": "generate"}

        with pytest.raises(RuntimeError, match="must be initialized before continuation"):
            await method.continue_reasoning(session, mock_thought)

    # === Global Phase Tests ===

    @pytest.mark.asyncio
    async def test_global_phase_assesses_solution(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that global phase assesses overall solution quality."""
        # Execute to create generate phase thought
        initial_thought = await initialized_method.execute(session, sample_problem)
        assert initial_thought.metadata["phase"] == "generate"

        # Continue to global phase
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)

        assert global_thought is not None
        assert global_thought.metadata["phase"] == "global"
        assert "Global ORM Assessment" in global_thought.content
        assert "ORM Global Score:" in global_thought.content
        assert initialized_method._global_score > 0.0
        assert global_thought.type == ThoughtType.VERIFICATION
        assert global_thought.parent_id == initial_thought.id

    @pytest.mark.asyncio
    async def test_global_phase_score_is_valid(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that global phase produces a valid score."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        await initialized_method.continue_reasoning(session, initial_thought)

        assert 0.0 <= initialized_method._global_score <= 1.0

    # === Local Phase Tests ===

    @pytest.mark.asyncio
    async def test_local_phase_scores_steps(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that local phase scores individual steps."""
        # Generate
        initial_thought = await initialized_method.execute(session, sample_problem)
        # Global
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        # Local
        local_thought = await initialized_method.continue_reasoning(session, global_thought)

        assert local_thought is not None
        assert local_thought.metadata["phase"] == "local"
        assert "Local Step-Level ORM" in local_thought.content
        assert "Step-Level Scores:" in local_thought.content
        assert len(initialized_method._local_scores) == len(initialized_method._solution_steps)
        assert local_thought.type == ThoughtType.VERIFICATION
        assert local_thought.parent_id == global_thought.id

    @pytest.mark.asyncio
    async def test_local_phase_identifies_issues(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that local phase identifies steps with issues."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)

        # Should have at least one step with an issue (heuristic always flags step 3)
        issues = [s for s in initialized_method._local_scores if s["issue"]]
        assert len(issues) >= 1

        # Content should mention issues
        assert "issue" in local_thought.content.lower() or "⚠" in local_thought.content

    @pytest.mark.asyncio
    async def test_local_phase_computes_statistics(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that local phase computes step statistics."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)

        assert "Total steps:" in local_thought.content
        assert "Average score:" in local_thought.content
        assert "Steps with issues:" in local_thought.content
        assert "Min score step:" in local_thought.content

    # === Refine Phase Tests ===

    @pytest.mark.asyncio
    async def test_refine_phase_applies_refinements(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that refine phase applies targeted refinements."""
        # Generate
        initial_thought = await initialized_method.execute(session, sample_problem)
        # Global
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        # Local
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        # Refine
        refine_thought = await initialized_method.continue_reasoning(session, local_thought)

        assert refine_thought is not None
        assert refine_thought.metadata["phase"] == "refine"
        assert "Apply Refinements" in refine_thought.content
        assert refine_thought.type == ThoughtType.REVISION
        assert refine_thought.parent_id == local_thought.id
        assert refine_thought.confidence == 0.85

    @pytest.mark.asyncio
    async def test_refine_phase_targets_issue_steps(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that refinements target steps with issues."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        await initialized_method.continue_reasoning(session, local_thought)

        # Refinements should match issue steps
        issue_steps = {s["step"] for s in initialized_method._local_scores if s["issue"]}
        refined_steps = {r["step"] for r in initialized_method._refinements}

        assert refined_steps == issue_steps

    @pytest.mark.asyncio
    async def test_refine_phase_shows_improvements(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that refinements show original and refined content."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        await initialized_method.continue_reasoning(session, local_thought)

        for refinement in initialized_method._refinements:
            assert "original" in refinement
            assert "refined" in refinement
            assert "improvement" in refinement
            assert refinement["improvement"] > 0.0

    # === Conclude Phase Tests ===

    @pytest.mark.asyncio
    async def test_conclude_phase_produces_final_answer(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conclude phase produces final answer."""
        # Generate
        initial_thought = await initialized_method.execute(session, sample_problem)
        # Global
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        # Local
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        # Refine
        refine_thought = await initialized_method.continue_reasoning(session, local_thought)
        # Conclude
        conclude_thought = await initialized_method.continue_reasoning(session, refine_thought)

        assert conclude_thought is not None
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.parent_id == refine_thought.id
        assert "Final Answer" in conclude_thought.content
        assert "GLoRe Complete" in conclude_thought.content

    @pytest.mark.asyncio
    async def test_conclude_phase_includes_summary(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conclusion includes a summary of the process."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        refine_thought = await initialized_method.continue_reasoning(session, local_thought)
        conclude_thought = await initialized_method.continue_reasoning(session, refine_thought)

        assert "Solution steps:" in conclude_thought.content
        assert "Global ORM score:" in conclude_thought.content
        assert "Average local score:" in conclude_thought.content
        assert "Refinements applied:" in conclude_thought.content

    @pytest.mark.asyncio
    async def test_conclude_phase_confidence_combines_scores(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that conclusion confidence combines global and local scores."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        refine_thought = await initialized_method.continue_reasoning(session, local_thought)
        conclude_thought = await initialized_method.continue_reasoning(session, refine_thought)

        # Confidence should be capped at 0.92
        assert conclude_thought.confidence <= 0.92

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        # Generate
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.metadata["phase"] == "generate"
        assert thought1.step_number == 1

        # Global
        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.metadata["phase"] == "global"
        assert thought2.step_number == 2
        assert thought2.depth == 1

        # Local
        thought3 = await initialized_method.continue_reasoning(session, thought2)
        assert thought3.metadata["phase"] == "local"
        assert thought3.step_number == 3
        assert thought3.depth == 2

        # Refine
        thought4 = await initialized_method.continue_reasoning(session, thought3)
        assert thought4.metadata["phase"] == "refine"
        assert thought4.step_number == 4
        assert thought4.depth == 3

        # Conclude
        thought5 = await initialized_method.continue_reasoning(session, thought4)
        assert thought5.metadata["phase"] == "conclude"
        assert thought5.step_number == 5
        assert thought5.depth == 4

        # Verify chain
        assert thought2.parent_id == thought1.id
        assert thought3.parent_id == thought2.id
        assert thought4.parent_id == thought3.id
        assert thought5.parent_id == thought4.id

        # Verify session has all thoughts
        assert session.thought_count == 5

    @pytest.mark.asyncio
    async def test_full_pipeline_with_sampling(
        self,
        initialized_method: GLoRe,
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
        thought5 = await initialized_method.continue_reasoning(
            session, thought4, execution_context=mock_execution_context
        )

        # All phases should complete
        assert thought1.metadata["phase"] == "generate"
        assert thought2.metadata["phase"] == "global"
        assert thought3.metadata["phase"] == "local"
        assert thought4.metadata["phase"] == "refine"
        assert thought5.metadata["phase"] == "conclude"

    # === Heuristic Method Tests ===

    def test_generate_solution_steps_heuristic(self, method: GLoRe) -> None:
        """Test heuristic solution step generation."""
        steps = method._generate_solution_steps_heuristic("test problem")

        assert len(steps) == 5
        assert all("step" in s for s in steps)
        assert all("content" in s for s in steps)
        assert all("type" in s for s in steps)

        # Check step numbers are sequential
        assert [s["step"] for s in steps] == [1, 2, 3, 4, 5]

        # Check types distribution
        types = [s["type"] for s in steps]
        assert "setup" in types
        assert "computation" in types
        assert "verification" in types
        assert "conclusion" in types

    def test_compute_global_score_heuristic(self, method: GLoRe) -> None:
        """Test heuristic global score computation."""
        score = method._compute_global_score_heuristic()

        assert score == 0.82
        assert 0.0 <= score <= 1.0

    def test_compute_local_scores_heuristic(self, method: GLoRe) -> None:
        """Test heuristic local score computation."""
        scores = method._compute_local_scores_heuristic()

        assert len(scores) == 5
        assert all("step" in s for s in scores)
        assert all("score" in s for s in scores)
        assert all("issue" in s for s in scores)
        assert all(0.0 <= s["score"] <= 1.0 for s in scores)

        # Should have at least one issue
        issues = [s for s in scores if s["issue"]]
        assert len(issues) >= 1

    def test_generate_refinements_heuristic(self, method: GLoRe) -> None:
        """Test heuristic refinement generation."""
        method._solution_steps = [{"step": 1, "content": "Original content", "type": "computation"}]
        issue_steps = [{"step": 1, "score": 0.5, "issue": "Some issue"}]

        refinements = method._generate_refinements_heuristic(issue_steps)

        assert len(refinements) == 1
        assert refinements[0]["step"] == 1
        assert refinements[0]["original"] == "Original content"
        assert "Verified" in refinements[0]["refined"]
        assert refinements[0]["improvement"] == 0.10

    def test_generate_refinements_heuristic_empty(self, method: GLoRe) -> None:
        """Test heuristic refinement with no issues returns empty list."""
        refinements = method._generate_refinements_heuristic([])
        assert refinements == []

    # === Sampling Method Tests ===

    @pytest.mark.asyncio
    async def test_sample_solution_steps_fallback(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_solution_steps falls back to heuristic."""
        initialized_method._execution_context = mock_execution_context

        steps = await initialized_method._sample_solution_steps("test")

        # Should return heuristic result since parsing is not implemented
        assert len(steps) >= 5

    @pytest.mark.asyncio
    async def test_sample_global_orm_score_parses_float(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_global_orm_score parses float response."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]
        mock_execution_context.sample = AsyncMock(return_value="0.87")

        score = await initialized_method._sample_global_orm_score()

        assert score == 0.87

    @pytest.mark.asyncio
    async def test_sample_global_orm_score_clamps_value(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test that scores are clamped to [0, 1] range."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]

        # Test clamping high value
        mock_execution_context.sample = AsyncMock(return_value="1.5")
        score = await initialized_method._sample_global_orm_score()
        assert score == 1.0

        # Test clamping low value
        mock_execution_context.sample = AsyncMock(return_value="-0.3")
        score = await initialized_method._sample_global_orm_score()
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_sample_global_orm_score_fallback_on_connection_error(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test fallback when sampling fails with ConnectionError."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]
        mock_execution_context.sample = AsyncMock(side_effect=ConnectionError("LLM error"))

        score = await initialized_method._sample_global_orm_score()

        assert score == 0.82  # Heuristic default

    @pytest.mark.asyncio
    async def test_sample_global_orm_score_fallback_on_timeout(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test fallback when sampling fails with TimeoutError."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]
        mock_execution_context.sample = AsyncMock(side_effect=TimeoutError("Timeout"))

        score = await initialized_method._sample_global_orm_score()

        assert score == 0.82  # Heuristic default

    @pytest.mark.asyncio
    async def test_sample_global_orm_score_reraises_unexpected_errors(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test that unexpected errors are re-raised."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._solution_steps = [{"step": 1, "content": "test", "type": "setup"}]
        mock_execution_context.sample = AsyncMock(side_effect=RuntimeError("Unexpected"))

        with pytest.raises(RuntimeError, match="Unexpected"):
            await initialized_method._sample_global_orm_score()

    @pytest.mark.asyncio
    async def test_sample_refinements_empty_issues(
        self, initialized_method: GLoRe, mock_execution_context: MagicMock
    ) -> None:
        """Test that _sample_refinements returns empty for no issues."""
        initialized_method._execution_context = mock_execution_context

        refinements = await initialized_method._sample_refinements([])

        assert refinements == []

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_updates_execution_context(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that continue_reasoning can update execution context."""
        initial_thought = await initialized_method.execute(session, sample_problem)

        new_context = MagicMock()
        new_context.can_sample = True
        new_context.sample = AsyncMock(return_value="0.9")

        await initialized_method.continue_reasoning(
            session, initial_thought, execution_context=new_context
        )

        assert initialized_method._execution_context is new_context

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: GLoRe,
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

        thought4 = await initialized_method.continue_reasoning(session, thought3)
        assert initialized_method._step_counter == 4

        await initialized_method.continue_reasoning(session, thought4)
        assert initialized_method._step_counter == 5

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: GLoRe,
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

        thought5 = await initialized_method.continue_reasoning(session, thought4)
        assert thought5.depth == 4

    @pytest.mark.asyncio
    async def test_metadata_includes_global_score(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that metadata includes global score after global phase."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)

        assert "global_score" in global_thought.metadata
        assert global_thought.metadata["global_score"] == initialized_method._global_score

    @pytest.mark.asyncio
    async def test_metadata_includes_refinement_count(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that metadata includes refinement count."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        refine_thought = await initialized_method.continue_reasoning(session, local_thought)

        assert "refinements" in refine_thought.metadata
        assert refine_thought.metadata["refinements"] == len(initialized_method._refinements)

    @pytest.mark.asyncio
    async def test_sampling_failure_at_each_phase(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that sampling failures are handled gracefully at each phase."""
        failing_context = MagicMock()
        failing_context.can_sample = True
        failing_context.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        # All phases should work with heuristic fallbacks
        thought1 = await initialized_method.execute(
            session, sample_problem, execution_context=failing_context
        )
        assert thought1 is not None

        thought2 = await initialized_method.continue_reasoning(
            session, thought1, execution_context=failing_context
        )
        assert thought2 is not None

        thought3 = await initialized_method.continue_reasoning(
            session, thought2, execution_context=failing_context
        )
        assert thought3 is not None

        thought4 = await initialized_method.continue_reasoning(
            session, thought3, execution_context=failing_context
        )
        assert thought4 is not None

        thought5 = await initialized_method.continue_reasoning(
            session, thought4, execution_context=failing_context
        )
        assert thought5 is not None

    @pytest.mark.asyncio
    async def test_context_with_can_sample_false(
        self,
        initialized_method: GLoRe,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that method uses heuristic when can_sample is False."""
        no_sample_context = MagicMock()
        no_sample_context.can_sample = False

        await initialized_method.execute(
            session, sample_problem, execution_context=no_sample_context
        )

        # Should still work with heuristic
        assert len(initialized_method._solution_steps) >= 5

    @pytest.mark.asyncio
    async def test_empty_local_scores_during_conclude(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test handling of empty local scores during conclusion."""
        # Create a refine phase thought manually
        refine_thought = ThoughtNode(
            type=ThoughtType.REVISION,
            method_id=MethodIdentifier.GLORE,
            content="Test",
            step_number=4,
            depth=3,
            metadata={"phase": "refine", "global_score": 0.8, "refinements": 0},
        )
        session.add_thought(refine_thought)
        initialized_method._step_counter = 4
        initialized_method._global_score = 0.8
        initialized_method._local_scores = []  # Empty
        initialized_method._refinements = []

        conclude_thought = await initialized_method.continue_reasoning(session, refine_thought)

        # Should handle gracefully with default avg_local = 0.8
        assert conclude_thought is not None
        assert conclude_thought.metadata["phase"] == "conclude"


class TestGloreOrmIntegration:
    """Tests specifically for GLoRe ORM (Outcome Reward Model) integration."""

    @pytest.fixture
    def method(self) -> GLoRe:
        """Create method instance."""
        return GLoRe()

    @pytest.fixture
    async def initialized_method(self) -> GLoRe:
        """Create an initialized method."""
        method = GLoRe()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> Session:
        """Create an active session."""
        return Session().start()

    @pytest.mark.asyncio
    async def test_global_orm_assessment_structure(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test that global ORM assessment has proper structure."""
        problem = "Solve: 2 + 2 * 3"
        initial_thought = await initialized_method.execute(session, problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)

        assert "Global Analysis:" in global_thought.content
        assert "ORM Global Score:" in global_thought.content
        assert "Score Interpretation:" in global_thought.content

    @pytest.mark.asyncio
    async def test_local_orm_scoring_structure(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test that local ORM scoring has proper structure."""
        problem = "Solve: 2 + 2 * 3"
        initial_thought = await initialized_method.execute(session, problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)

        assert "Step-Level Scores:" in local_thought.content
        assert "Step Analysis:" in local_thought.content
        # Should show individual step evaluations
        for score_entry in initialized_method._local_scores:
            assert f"Step {score_entry['step']}" in local_thought.content

    @pytest.mark.asyncio
    async def test_orm_scores_influence_refinements(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test that low ORM scores trigger refinements."""
        problem = "Solve: 2 + 2 * 3"
        initial_thought = await initialized_method.execute(session, problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        await initialized_method.continue_reasoning(session, local_thought)

        # Steps with issues should have refinements
        issue_steps = [s for s in initialized_method._local_scores if s["issue"]]
        assert len(initialized_method._refinements) == len(issue_steps)


class TestGloreRefinementProcess:
    """Tests specifically for GLoRe refinement process."""

    @pytest.fixture
    def method(self) -> GLoRe:
        """Create method instance."""
        return GLoRe()

    @pytest.fixture
    async def initialized_method(self) -> GLoRe:
        """Create an initialized method."""
        method = GLoRe()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> Session:
        """Create an active session."""
        return Session().start()

    @pytest.mark.asyncio
    async def test_refinement_preserves_correct_steps(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test that refinement only targets steps with issues."""
        problem = "Solve: 5 * 3 + 2"
        initial_thought = await initialized_method.execute(session, problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        await initialized_method.continue_reasoning(session, local_thought)

        # Steps without issues should not be refined
        good_steps = {s["step"] for s in initialized_method._local_scores if not s["issue"]}
        refined_steps = {r["step"] for r in initialized_method._refinements}

        assert good_steps.isdisjoint(refined_steps)

    @pytest.mark.asyncio
    async def test_refinement_improvement_tracking(
        self, initialized_method: GLoRe, session: Session
    ) -> None:
        """Test that refinements track improvement scores."""
        problem = "Solve: 5 * 3 + 2"
        initial_thought = await initialized_method.execute(session, problem)
        global_thought = await initialized_method.continue_reasoning(session, initial_thought)
        local_thought = await initialized_method.continue_reasoning(session, global_thought)
        await initialized_method.continue_reasoning(session, local_thought)

        for refinement in initialized_method._refinements:
            assert "improvement" in refinement
            assert refinement["improvement"] > 0.0


__all__ = [
    "TestGloreMetadata",
    "TestGlore",
    "TestGloreOrmIntegration",
    "TestGloreRefinementProcess",
]
