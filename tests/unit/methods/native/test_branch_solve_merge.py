"""Unit tests for BranchSolveMerge reasoning method.

This module provides comprehensive tests for the BranchSolveMerge method
implementation, covering initialization, execution, branching, solving,
merging phases, and elicitation behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning_mcp.methods.native.branch_solve_merge import (
    BRANCH_SOLVE_MERGE_METADATA,
    BranchSolveMerge,
)
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def bsm_method() -> BranchSolveMerge:
    """Create a BranchSolveMerge method instance for testing.

    Returns:
        A fresh BranchSolveMerge instance
    """
    return BranchSolveMerge()


@pytest.fixture
def bsm_no_elicitation() -> BranchSolveMerge:
    """Create a BranchSolveMerge method with elicitation disabled.

    Returns:
        A BranchSolveMerge instance with elicitation disabled
    """
    return BranchSolveMerge(enable_elicitation=False)


@pytest.fixture
async def initialized_method() -> BranchSolveMerge:
    """Create an initialized BranchSolveMerge method instance.

    Returns:
        An initialized BranchSolveMerge instance
    """
    method = BranchSolveMerge()
    await method.initialize()
    return method


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
    return "Design a user authentication system for a web application"


@pytest.fixture
def complex_problem() -> str:
    """Provide a complex multi-faceted problem for testing.

    Returns:
        A complex problem string
    """
    return (
        "Create a comprehensive data pipeline that ingests data from multiple sources, "
        "performs real-time transformations, ensures data quality, and stores results "
        "in multiple output formats while maintaining auditability"
    )


class TestBranchSolveMergeMetadata:
    """Tests for BRANCH_SOLVE_MERGE_METADATA configuration."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert BRANCH_SOLVE_MERGE_METADATA.identifier == MethodIdentifier.BRANCH_SOLVE_MERGE

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert BRANCH_SOLVE_MERGE_METADATA.name == "Branch-Solve-Merge"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description mentioning key concepts."""
        desc = BRANCH_SOLVE_MERGE_METADATA.description.lower()
        assert "decompose" in desc or "parallel" in desc
        assert "merge" in desc

    def test_metadata_category(self) -> None:
        """Test that metadata is in ADVANCED category."""
        assert BRANCH_SOLVE_MERGE_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self) -> None:
        """Test that metadata has reasonable complexity."""
        assert BRANCH_SOLVE_MERGE_METADATA.complexity == 6
        assert 1 <= BRANCH_SOLVE_MERGE_METADATA.complexity <= 10

    def test_metadata_supports_branching(self) -> None:
        """Test that BSM supports branching."""
        assert BRANCH_SOLVE_MERGE_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that BSM doesn't support revision."""
        assert BRANCH_SOLVE_MERGE_METADATA.supports_revision is False

    def test_metadata_requires_context(self) -> None:
        """Test that BSM doesn't require context."""
        assert BRANCH_SOLVE_MERGE_METADATA.requires_context is False

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert BRANCH_SOLVE_MERGE_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert BRANCH_SOLVE_MERGE_METADATA.max_thoughts == 8

    def test_metadata_tags(self) -> None:
        """Test that metadata has appropriate tags."""
        assert "decomposition" in BRANCH_SOLVE_MERGE_METADATA.tags
        assert "parallel" in BRANCH_SOLVE_MERGE_METADATA.tags
        assert "merge" in BRANCH_SOLVE_MERGE_METADATA.tags
        assert "planning" in BRANCH_SOLVE_MERGE_METADATA.tags

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(BRANCH_SOLVE_MERGE_METADATA.best_for).lower()
        assert "multi-faceted" in best_for_text or "complex" in best_for_text

    def test_metadata_not_recommended_for(self) -> None:
        """Test that metadata specifies inappropriate use cases."""
        not_recommended = " ".join(BRANCH_SOLVE_MERGE_METADATA.not_recommended_for).lower()
        assert "simple" in not_recommended


class TestBranchSolveMergeInitialization:
    """Tests for BranchSolveMerge method initialization."""

    def test_create_instance(self, bsm_method: BranchSolveMerge) -> None:
        """Test that we can create a BranchSolveMerge instance."""
        assert isinstance(bsm_method, BranchSolveMerge)

    def test_initial_state(self, bsm_method: BranchSolveMerge) -> None:
        """Test that initial state is correct before initialization."""
        assert bsm_method._initialized is False
        assert bsm_method._step_counter == 0
        assert bsm_method._current_phase == "branch"
        assert bsm_method._branches == []
        assert bsm_method._solutions == []
        assert bsm_method._merged_result is None

    def test_default_elicitation_enabled(self, bsm_method: BranchSolveMerge) -> None:
        """Test that elicitation is enabled by default."""
        assert bsm_method.enable_elicitation is True

    def test_elicitation_can_be_disabled(self, bsm_no_elicitation: BranchSolveMerge) -> None:
        """Test that elicitation can be disabled."""
        assert bsm_no_elicitation.enable_elicitation is False

    async def test_initialize(self, bsm_method: BranchSolveMerge) -> None:
        """Test that initialize sets up the method correctly."""
        await bsm_method.initialize()
        assert bsm_method._initialized is True
        assert bsm_method._step_counter == 0
        assert bsm_method._current_phase == "branch"
        assert bsm_method._branches == []
        assert bsm_method._solutions == []
        assert bsm_method._merged_result is None

    async def test_initialize_resets_state(self) -> None:
        """Test that initialize resets state from previous executions."""
        method = BranchSolveMerge()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_phase = "conclude"
        method._branches = [{"id": 1}]
        method._solutions = [{"branch_id": 1}]
        method._merged_result = "Some result"

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._current_phase == "branch"
        assert method._branches == []
        assert method._solutions == []
        assert method._merged_result is None

    async def test_health_check_before_init(self, bsm_method: BranchSolveMerge) -> None:
        """Test health_check returns False before initialization."""
        health = await bsm_method.health_check()
        assert health is False

    async def test_health_check_after_init(self, initialized_method: BranchSolveMerge) -> None:
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestBranchSolveMergeProperties:
    """Tests for BranchSolveMerge method properties."""

    def test_identifier_property(self, bsm_method: BranchSolveMerge) -> None:
        """Test that identifier property returns correct value."""
        assert bsm_method.identifier == MethodIdentifier.BRANCH_SOLVE_MERGE

    def test_name_property(self, bsm_method: BranchSolveMerge) -> None:
        """Test that name property returns correct value."""
        assert bsm_method.name == "Branch-Solve-Merge"

    def test_description_property(self, bsm_method: BranchSolveMerge) -> None:
        """Test that description property returns correct value."""
        assert bsm_method.description == BRANCH_SOLVE_MERGE_METADATA.description

    def test_category_property(self, bsm_method: BranchSolveMerge) -> None:
        """Test that category property returns correct value."""
        assert bsm_method.category == MethodCategory.ADVANCED


class TestBranchSolveMergeExecution:
    """Tests for basic execution of BranchSolveMerge reasoning."""

    async def test_execute_without_initialization_fails(
        self, bsm_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await bsm_method.execute(
                session=session,
                input_text="Test problem",
            )

    async def test_execute_basic(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test basic execution creates branch phase thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.BRANCH_SOLVE_MERGE
        assert thought.step_number == 1
        assert thought.depth == 0

    async def test_execute_sets_phase_to_branch(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets phase to branch."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._current_phase == "branch"
        assert thought.metadata["phase"] == "branch"

    async def test_execute_generates_branches(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates branches."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert len(initialized_method._branches) >= 3
        for branch in initialized_method._branches:
            assert "id" in branch
            assert "task" in branch
            assert "criteria" in branch

    async def test_execute_adds_to_session(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.BRANCH_SOLVE_MERGE

    async def test_execute_content_format(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert "Step 1" in thought.content
        assert sample_problem in thought.content
        assert "Branch" in thought.content

    async def test_execute_confidence_level(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate confidence for branch phase."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.confidence == 0.6

    async def test_execute_metadata(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought.metadata["phase"] == "branch"
        assert thought.metadata["branches"] >= 3
        assert thought.metadata["input_text"] == sample_problem


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    async def test_continue_without_initialization_fails(
        self, bsm_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test that continue_reasoning fails if not initialized."""
        mock_thought = MagicMock()
        mock_thought.metadata = {"phase": "branch", "input_text": "Test"}
        mock_thought.id = "test-id"
        mock_thought.depth = 0

        with pytest.raises(RuntimeError, match="must be initialized"):
            await bsm_method.continue_reasoning(
                session=session,
                previous_thought=mock_thought,
            )

    async def test_phase_transition_branch_to_solve(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from branch to solve."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        assert initialized_method._current_phase == "solve"
        assert solve_thought.metadata["phase"] == "solve"
        assert solve_thought.type == ThoughtType.REASONING

    async def test_phase_transition_solve_to_merge(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from solve to merge."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )

        assert initialized_method._current_phase == "merge"
        assert merge_thought.metadata["phase"] == "merge"
        assert merge_thought.type == ThoughtType.SYNTHESIS

    async def test_phase_transition_merge_to_conclude(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test phase transition from merge to conclude."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )

        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=merge_thought,
        )

        assert initialized_method._current_phase == "conclude"
        assert conclude_thought.metadata["phase"] == "conclude"
        assert conclude_thought.type == ThoughtType.CONCLUSION

    async def test_step_counter_increments(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that step counter increments with each continuation."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert branch_thought.step_number == 1

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        assert solve_thought.step_number == 2

        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )
        assert merge_thought.step_number == 3

    async def test_parent_id_set_correctly(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that parent_id is set correctly in continuation."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        assert solve_thought.parent_id == branch_thought.id

    async def test_depth_increases(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that depth increases with each continuation."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert branch_thought.depth == 0

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        assert solve_thought.depth == 1


class TestBranchingSolving:
    """Tests for branching and solving behavior."""

    async def test_branches_have_criteria(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that each branch has evaluation criteria."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        for branch in initialized_method._branches:
            assert "criteria" in branch
            assert branch["criteria"] is not None

    async def test_solutions_generated_for_each_branch(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that solutions are generated for each branch."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        assert len(initialized_method._solutions) == len(initialized_method._branches)

    async def test_solutions_have_required_fields(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that solutions have required fields."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        for solution in initialized_method._solutions:
            assert "branch_id" in solution
            assert "task" in solution
            assert "result" in solution
            assert "score" in solution

    async def test_solution_scores_in_valid_range(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that solution scores are in valid range."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        for solution in initialized_method._solutions:
            assert 0.0 <= solution["score"] <= 1.0


class TestMergingBehavior:
    """Tests for merging behavior."""

    async def test_merged_result_generated(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that merged result is generated."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )

        assert initialized_method._merged_result is not None

    async def test_merge_thought_contains_integration_details(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that merge thought contains integration details."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )

        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )

        assert "Merge" in merge_thought.content
        assert "Integration" in merge_thought.content or "Integrated" in merge_thought.content


class TestConfidenceProgression:
    """Tests for confidence score progression through phases."""

    async def test_confidence_increases_through_phases(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that confidence generally increases through phases."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        branch_confidence = branch_thought.confidence

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        solve_confidence = solve_thought.confidence

        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )
        merge_confidence = merge_thought.confidence

        # Confidence should generally increase
        assert branch_confidence <= solve_confidence
        assert solve_confidence <= merge_confidence


class TestEdgeCases:
    """Tests for edge cases in BranchSolveMerge reasoning."""

    async def test_empty_query(
        self, initialized_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test handling of empty query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="",
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_very_long_query(
        self, initialized_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test handling of very long query."""
        long_query = "Design a system: " + "requirement " * 500
        thought = await initialized_method.execute(
            session=session,
            input_text=long_query,
        )

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    async def test_special_characters_in_query(
        self, initialized_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test handling of special characters in query."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test with émojis 🎉 and spëcial chars! @#$%",
        )

        assert thought is not None

    async def test_unicode_in_query(
        self, initialized_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test handling of unicode content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="设计一个用户认证系统",
        )

        assert thought is not None

    async def test_complete_reasoning_flow(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test complete reasoning flow from start to finish."""
        # Phase 1: Branch
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert branch_thought.type == ThoughtType.INITIAL
        assert branch_thought.metadata["phase"] == "branch"

        # Phase 2: Solve
        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        assert solve_thought.type == ThoughtType.REASONING
        assert solve_thought.metadata["phase"] == "solve"

        # Phase 3: Merge
        merge_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=solve_thought,
        )
        assert merge_thought.type == ThoughtType.SYNTHESIS
        assert merge_thought.metadata["phase"] == "merge"

        # Phase 4: Conclude
        conclude_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=merge_thought,
        )
        assert conclude_thought.type == ThoughtType.CONCLUSION
        assert conclude_thought.metadata["phase"] == "conclude"

        # Verify session state
        assert session.thought_count == 4
        assert session.current_method == MethodIdentifier.BRANCH_SOLVE_MERGE

    async def test_multiple_execution_cycles(
        self, initialized_method: BranchSolveMerge, session: Session
    ) -> None:
        """Test that method can handle multiple execution cycles."""
        # First execution
        thought1 = await initialized_method.execute(
            session=session,
            input_text="First problem",
        )
        assert thought1.step_number == 1

        # Reinitialize
        await initialized_method.initialize()

        # Second execution
        thought2 = await initialized_method.execute(
            session=session,
            input_text="Second problem",
        )
        assert thought2.step_number == 1
        assert initialized_method._branches != []


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that session thought count updates correctly."""
        initial_count = session.thought_count
        await initialized_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_method_tracking(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that session tracks method usage."""
        await initialized_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.BRANCH_SOLVE_MERGE)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that session can filter thoughts by method."""
        await initialized_method.execute(session, sample_problem)

        bsm_thoughts = session.get_thoughts_by_method(MethodIdentifier.BRANCH_SOLVE_MERGE)
        assert len(bsm_thoughts) > 0

    async def test_input_text_preserved_through_phases(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that input_text is preserved in metadata through phases."""
        branch_thought = await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )
        assert branch_thought.metadata["input_text"] == sample_problem

        solve_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=branch_thought,
        )
        assert solve_thought.metadata["input_text"] == sample_problem


class TestElicitationBehavior:
    """Tests for elicitation-related behavior."""

    async def test_elicitation_disabled_skips_interactions(
        self, bsm_no_elicitation: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that disabled elicitation skips user interactions."""
        await bsm_no_elicitation.initialize()

        # Execute should work without any elicitation
        thought = await bsm_no_elicitation.execute(
            session=session,
            input_text=sample_problem,
        )

        assert thought is not None

    async def test_elicitation_context_not_set_by_default(
        self, initialized_method: BranchSolveMerge, session: Session, sample_problem: str
    ) -> None:
        """Test that elicitation context is not set without execution context."""
        await initialized_method.execute(
            session=session,
            input_text=sample_problem,
        )

        assert initialized_method._ctx is None
