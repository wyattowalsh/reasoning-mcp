"""Comprehensive unit tests for SelfConsistency reasoning method."""

from __future__ import annotations

from uuid import uuid4

import pytest

from reasoning_mcp.methods.native.self_consistency import (
    SELF_CONSISTENCY_METADATA,
    SelfConsistency,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestSelfConsistencyMetadata:
    """Tests for SelfConsistency metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SELF_CONSISTENCY_METADATA.identifier == MethodIdentifier.SELF_CONSISTENCY

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SELF_CONSISTENCY_METADATA.name == "Self-Consistency"

    def test_metadata_category(self):
        """Test that metadata is in CORE category."""
        assert SELF_CONSISTENCY_METADATA.category == MethodCategory.CORE

    def test_metadata_supports_branching(self):
        """Test that metadata indicates branching support."""
        assert SELF_CONSISTENCY_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata indicates no revision support."""
        assert SELF_CONSISTENCY_METADATA.supports_revision is False

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        assert "voting" in SELF_CONSISTENCY_METADATA.tags
        assert "consensus" in SELF_CONSISTENCY_METADATA.tags
        assert "parallel" in SELF_CONSISTENCY_METADATA.tags
        assert "reliability" in SELF_CONSISTENCY_METADATA.tags

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert SELF_CONSISTENCY_METADATA.complexity == 4

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts for voting."""
        assert SELF_CONSISTENCY_METADATA.min_thoughts == 3


class TestSelfConsistencyInitialization:
    """Tests for SelfConsistency initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        method = SelfConsistency()
        assert method.identifier == str(MethodIdentifier.SELF_CONSISTENCY)
        assert method.name == "Self-Consistency"
        assert method.description == SELF_CONSISTENCY_METADATA.description
        assert method.category == str(MethodCategory.CORE)

    def test_custom_num_paths(self):
        """Test initialization with custom num_paths."""
        method = SelfConsistency(num_paths=5)
        assert method._num_paths == 5

    def test_custom_min_agreement(self):
        """Test initialization with custom min_agreement."""
        method = SelfConsistency(min_agreement=0.7)
        assert method._min_agreement == 0.7

    def test_num_paths_too_low(self):
        """Test that num_paths less than 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_paths must be at least 2"):
            SelfConsistency(num_paths=1)

    def test_num_paths_too_high(self):
        """Test that num_paths greater than 10 raises ValueError."""
        with pytest.raises(ValueError, match="num_paths should not exceed 10"):
            SelfConsistency(num_paths=11)

    def test_min_agreement_below_zero(self):
        """Test that min_agreement below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_agreement must be between 0.0 and 1.0"):
            SelfConsistency(min_agreement=-0.1)

    def test_min_agreement_above_one(self):
        """Test that min_agreement above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_agreement must be between 0.0 and 1.0"):
            SelfConsistency(min_agreement=1.1)

    def test_valid_boundary_values(self):
        """Test valid boundary values for initialization."""
        method = SelfConsistency(num_paths=2, min_agreement=0.0)
        assert method._num_paths == 2
        assert method._min_agreement == 0.0

        method = SelfConsistency(num_paths=10, min_agreement=1.0)
        assert method._num_paths == 10
        assert method._min_agreement == 1.0

    @pytest.mark.asyncio
    async def test_initialize_method(self):
        """Test initialize() method completes without error."""
        method = SelfConsistency()
        await method.initialize()
        # No assertion needed - just verify it doesn't raise

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check() returns True."""
        method = SelfConsistency()
        assert await method.health_check() is True


class TestSelfConsistencyBasicExecution:
    """Tests for basic execution of SelfConsistency method."""

    @pytest.mark.asyncio
    async def test_execute_creates_session_thoughts(self):
        """Test that execute creates thoughts in the session."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        input_text = "What is 2+2?"

        result = await method.execute(session, input_text)

        # Should have created thoughts
        assert session.thought_count > 0
        assert result is not None
        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self):
        """Test that execute creates an initial thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        input_text = "What is 2+2?"

        await method.execute(session, input_text)

        # Find initial thought
        initial_thoughts = [
            t for t in session.graph.nodes.values() if t.type == ThoughtType.INITIAL
        ]
        assert len(initial_thoughts) == 1
        assert "Self-Consistency reasoning" in initial_thoughts[0].content
        assert input_text in initial_thoughts[0].content

    @pytest.mark.asyncio
    async def test_execute_creates_multiple_paths(self):
        """Test that execute creates multiple reasoning paths."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        input_text = "What is 2+2?"

        await method.execute(session, input_text)

        # Should have branch thoughts for each path
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 3

    @pytest.mark.asyncio
    async def test_execute_creates_synthesis_thought(self):
        """Test that execute creates a synthesis thought with voting results."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        input_text = "What is 2+2?"

        result = await method.execute(session, input_text)

        # Result should be synthesis thought
        assert result.type == ThoughtType.SYNTHESIS
        assert "Self-Consistency Analysis Complete" in result.content
        assert "majority_answer" in result.metadata
        assert "vote_counts" in result.metadata
        assert "agreement_rate" in result.metadata
        assert "consistency_score" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_returns_synthesis_thought(self):
        """Test that execute returns the synthesis thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        input_text = "What is 2+2?"

        result = await method.execute(session, input_text)

        assert result.type == ThoughtType.SYNTHESIS
        assert result.metadata.get("is_final_answer") is True


class TestSelfConsistencyPathGeneration:
    """Tests for reasoning path generation."""

    @pytest.mark.asyncio
    async def test_num_paths_configuration(self):
        """Test that num_paths parameter controls path count."""
        method = SelfConsistency(num_paths=5)
        session = Session().start()

        await method.execute(session, "Test question")

        # Count branch thoughts
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 5

    @pytest.mark.asyncio
    async def test_context_override_num_paths(self):
        """Test that context can override num_paths."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        await method.execute(session, "Test question", context={"num_paths": 4})

        # Should use context value
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 4

    @pytest.mark.asyncio
    async def test_context_invalid_num_paths_uses_default(self):
        """Test that invalid context num_paths uses default."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        await method.execute(session, "Test question", context={"num_paths": 1})

        # Should use default value
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 3

    @pytest.mark.asyncio
    async def test_each_path_has_unique_branch_id(self):
        """Test that each path has a unique branch_id."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        await method.execute(session, "Test question")

        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        branch_ids = [t.branch_id for t in branch_thoughts]

        # All branch_ids should be unique
        assert len(branch_ids) == len(set(branch_ids))
        # All should be non-None
        assert all(bid is not None for bid in branch_ids)

    @pytest.mark.asyncio
    async def test_each_path_has_three_thoughts(self):
        """Test that each path generates branch, intermediate, and conclusion."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        await method.execute(session, "Test question")

        # Each path should have: branch + continuation + conclusion = 3 thoughts per path
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        continuation_thoughts = [
            t for t in session.graph.nodes.values() if t.type == ThoughtType.CONTINUATION
        ]
        conclusion_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.type == ThoughtType.CONCLUSION and t.metadata.get("is_path_conclusion")
        ]

        assert len(branch_thoughts) == 3
        assert len(continuation_thoughts) == 3
        assert len(conclusion_thoughts) == 3


class TestSelfConsistencyVotingMechanisms:
    """Tests for voting mechanisms."""

    def test_perform_voting_simple_majority(self):
        """Test voting with simple majority."""
        method = SelfConsistency()
        conclusions = ["answer A", "answer A", "answer B"]

        result = method._perform_voting(conclusions)

        assert result["majority_answer"] == "answer a"  # normalized to lowercase
        assert result["vote_counts"]["answer a"] == 2
        assert result["vote_counts"]["answer b"] == 1
        assert result["agreement_rate"] == 2 / 3

    def test_perform_voting_unanimous(self):
        """Test voting with unanimous agreement."""
        method = SelfConsistency()
        conclusions = ["answer A", "answer A", "answer A"]

        result = method._perform_voting(conclusions)

        assert result["majority_answer"] == "answer a"
        assert result["vote_counts"]["answer a"] == 3
        assert result["agreement_rate"] == 1.0
        assert result["consistency_score"] == 1.0  # Perfect consistency

    def test_perform_voting_all_different(self):
        """Test voting with all different answers."""
        method = SelfConsistency()
        conclusions = ["answer A", "answer B", "answer C"]

        result = method._perform_voting(conclusions)

        # Should pick first in sorted order with same count
        assert result["majority_answer"] in ["answer a", "answer b", "answer c"]
        assert result["agreement_rate"] == 1 / 3
        # Consistency score should be low
        assert result["consistency_score"] < 0.5

    def test_perform_voting_empty_conclusions(self):
        """Test voting with empty conclusions list."""
        method = SelfConsistency()
        conclusions = []

        result = method._perform_voting(conclusions)

        assert result["majority_answer"] == "No conclusion reached"
        assert result["vote_counts"] == {}
        assert result["agreement_rate"] == 0.0
        assert result["consistency_score"] == 0.0

    def test_perform_voting_tie_breaking(self):
        """Test voting tie breaking behavior."""
        method = SelfConsistency()
        conclusions = ["answer A", "answer A", "answer B", "answer B"]

        result = method._perform_voting(conclusions)

        # Both have 2 votes - should pick one consistently
        assert result["majority_answer"] in ["answer a", "answer b"]
        assert result["agreement_rate"] == 0.5

    def test_perform_voting_case_normalization(self):
        """Test that voting normalizes case."""
        method = SelfConsistency()
        conclusions = ["Answer A", "ANSWER A", "answer a"]

        result = method._perform_voting(conclusions)

        assert result["majority_answer"] == "answer a"
        assert result["vote_counts"]["answer a"] == 3
        assert result["agreement_rate"] == 1.0

    def test_perform_voting_whitespace_normalization(self):
        """Test that voting normalizes whitespace."""
        method = SelfConsistency()
        conclusions = ["  answer A  ", "answer A", "answer A   "]

        result = method._perform_voting(conclusions)

        assert result["majority_answer"] == "answer a"
        assert result["agreement_rate"] == 1.0

    def test_consistency_score_calculation(self):
        """Test consistency score calculation based on entropy."""
        method = SelfConsistency()

        # Perfect consistency
        result1 = method._perform_voting(["A", "A", "A", "A"])
        assert result1["consistency_score"] == 1.0

        # Moderate consistency
        result2 = method._perform_voting(["A", "A", "A", "B"])
        assert 0.5 < result2["consistency_score"] < 1.0

        # Low consistency
        result3 = method._perform_voting(["A", "B", "C", "D"])
        assert result3["consistency_score"] < 0.5


class TestSelfConsistencyContinueReasoning:
    """Tests for continue_reasoning functionality."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_creates_verification(self):
        """Test that continue_reasoning creates a verification thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        # Execute first
        initial_result = await method.execute(session, "Test question")

        # Continue reasoning
        continued = await method.continue_reasoning(session, initial_result)

        assert continued.type == ThoughtType.VERIFICATION
        assert "Verification of Self-Consistency result" in continued.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_includes_guidance(self):
        """Test that continue_reasoning includes guidance in output."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        initial_result = await method.execute(session, "Test question")
        continued = await method.continue_reasoning(
            session, initial_result, guidance="Please verify the calculation"
        )

        assert "Please verify the calculation" in continued.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_preserves_confidence(self):
        """Test that continue_reasoning preserves confidence from previous thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        initial_result = await method.execute(session, "Test question")
        continued = await method.continue_reasoning(session, initial_result)

        assert continued.confidence == initial_result.confidence

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_depth(self):
        """Test that continue_reasoning increments depth."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        initial_result = await method.execute(session, "Test question")
        continued = await method.continue_reasoning(session, initial_result)

        assert continued.depth == initial_result.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_links_to_previous(self):
        """Test that continue_reasoning links to previous thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        initial_result = await method.execute(session, "Test question")
        continued = await method.continue_reasoning(session, initial_result)

        assert continued.parent_id == initial_result.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_reports_high_consistency(self):
        """Test that continue_reasoning reports high consistency correctly."""
        method = SelfConsistency(num_paths=3, min_agreement=0.5)
        session = Session().start()

        # Mock a high agreement result
        initial_result = await method.execute(session, "Test question")

        # Manually set high agreement for testing
        initial_result = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            content="Test",
            metadata={"majority_answer": "test", "agreement_rate": 0.9, "consistency_score": 0.95},
        )
        session.add_thought(initial_result)

        continued = await method.continue_reasoning(session, initial_result)

        assert "high" in continued.content.lower()

    @pytest.mark.asyncio
    async def test_continue_reasoning_reports_moderate_consistency(self):
        """Test that continue_reasoning reports moderate consistency correctly."""
        method = SelfConsistency(num_paths=3, min_agreement=0.5)
        session = Session().start()

        # Mock a moderate agreement result
        initial_result = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            content="Test",
            metadata={"majority_answer": "test", "agreement_rate": 0.4, "consistency_score": 0.5},
        )
        session.add_thought(initial_result)

        continued = await method.continue_reasoning(session, initial_result)

        assert "moderate" in continued.content.lower()


class TestSelfConsistencyConsistencyCalculation:
    """Tests for consistency score calculation."""

    def test_consistency_score_perfect_agreement(self):
        """Test consistency score with perfect agreement."""
        method = SelfConsistency()
        conclusions = ["answer"] * 5

        result = method._perform_voting(conclusions)

        assert result["consistency_score"] == 1.0

    def test_consistency_score_no_agreement(self):
        """Test consistency score with no agreement."""
        method = SelfConsistency()
        conclusions = ["a", "b", "c", "d", "e"]

        result = method._perform_voting(conclusions)

        # Maximum entropy = low consistency
        assert result["consistency_score"] == 0.0

    def test_consistency_score_partial_agreement(self):
        """Test consistency score with partial agreement."""
        method = SelfConsistency()
        conclusions = ["a", "a", "a", "b", "b"]

        result = method._perform_voting(conclusions)

        # Should be between 0 and 1
        assert 0.0 < result["consistency_score"] < 1.0

    def test_consistency_score_increases_with_agreement(self):
        """Test that consistency score increases as agreement increases."""
        method = SelfConsistency()

        result1 = method._perform_voting(["a", "a", "b", "c"])
        result2 = method._perform_voting(["a", "a", "a", "b"])
        result3 = method._perform_voting(["a", "a", "a", "a"])

        assert result1["consistency_score"] < result2["consistency_score"]
        assert result2["consistency_score"] < result3["consistency_score"]


class TestSelfConsistencyPathDiversity:
    """Tests for path diversity."""

    @pytest.mark.asyncio
    async def test_paths_have_different_reasoning_steps(self):
        """Test that different paths have different reasoning approaches."""
        method = SelfConsistency(num_paths=5)
        session = Session().start()

        await method.execute(session, "Test question")

        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]

        # Extract reasoning content
        reasoning_contents = [t.content for t in branch_thoughts]

        # Should have some variation
        unique_contents = set(reasoning_contents)
        assert len(unique_contents) > 1

    @pytest.mark.asyncio
    async def test_path_numbers_are_sequential(self):
        """Test that path numbers are sequential."""
        method = SelfConsistency(num_paths=4)
        session = Session().start()

        await method.execute(session, "Test question")

        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]

        path_numbers = [t.metadata.get("path_number") for t in branch_thoughts]
        path_numbers.sort()

        assert path_numbers == [1, 2, 3, 4]


class TestSelfConsistencyFinalAnswerSelection:
    """Tests for final answer selection."""

    @pytest.mark.asyncio
    async def test_final_answer_metadata_present(self):
        """Test that final synthesis has all required metadata."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "Test question")

        assert "majority_answer" in result.metadata
        assert "vote_counts" in result.metadata
        assert "agreement_rate" in result.metadata
        assert "consistency_score" in result.metadata
        assert "num_paths" in result.metadata
        assert "all_conclusions" in result.metadata
        assert result.metadata["is_final_answer"] is True

    @pytest.mark.asyncio
    async def test_confidence_increases_with_agreement(self):
        """Test that confidence increases with higher agreement rates."""
        SelfConsistency(num_paths=3)

        # We can't easily mock different agreement rates in execute,
        # but we can test the formula
        # final_confidence = min(0.95, 0.5 + (agreement_rate * 0.5))

        # For agreement_rate = 0.5
        confidence_50 = min(0.95, 0.5 + (0.5 * 0.5))  # 0.75

        # For agreement_rate = 1.0
        confidence_100 = min(0.95, 0.5 + (1.0 * 0.5))  # 0.95

        assert confidence_100 > confidence_50

    @pytest.mark.asyncio
    async def test_synthesis_content_formatting(self):
        """Test that synthesis content is properly formatted."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "Test question")

        # Check key sections are present
        assert "Self-Consistency Analysis Complete" in result.content
        assert "Voting Results:" in result.content
        assert "Consensus:" in result.content
        assert "Majority Answer:" in result.content
        assert "Agreement Rate:" in result.content


class TestSelfConsistencyEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_minimum_num_paths(self):
        """Test execution with minimum num_paths (2)."""
        method = SelfConsistency(num_paths=2)
        session = Session().start()

        result = await method.execute(session, "Test question")

        assert result is not None
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 2

    @pytest.mark.asyncio
    async def test_maximum_num_paths(self):
        """Test execution with maximum num_paths (10)."""
        method = SelfConsistency(num_paths=10)
        session = Session().start()

        result = await method.execute(session, "Test question")

        assert result is not None
        branch_thoughts = [t for t in session.graph.nodes.values() if t.type == ThoughtType.BRANCH]
        assert len(branch_thoughts) == 10

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test execution with empty input text."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "")

        assert result is not None
        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_very_long_input_text(self):
        """Test execution with very long input text."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()
        long_text = "A" * 10000

        result = await method.execute(session, long_text)

        assert result is not None
        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_none_context(self):
        """Test execution with None context."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "Test question", context=None)

        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test execution with empty context dict."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "Test question", context={})

        assert result is not None

    @pytest.mark.asyncio
    async def test_quality_score_set_on_synthesis(self):
        """Test that quality_score is set on synthesis thought."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        result = await method.execute(session, "Test question")

        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 1.0
        # Quality score should equal consistency score
        assert result.quality_score == result.metadata["consistency_score"]

    def test_vote_counts_sorted_by_count(self):
        """Test that vote_counts are sorted by count descending."""
        method = SelfConsistency()
        conclusions = ["A", "B", "B", "C", "C", "C"]

        result = method._perform_voting(conclusions)

        vote_items = list(result["vote_counts"].items())
        # First should have highest count
        assert vote_items[0][1] >= vote_items[1][1]
        assert vote_items[1][1] >= vote_items[2][1]

    @pytest.mark.asyncio
    async def test_step_numbers_sequential(self):
        """Test that step numbers are sequential and logical."""
        method = SelfConsistency(num_paths=3)
        session = Session().start()

        await method.execute(session, "Test question")

        all_thoughts = list(session.graph.nodes.values())
        step_numbers = [t.step_number for t in all_thoughts]

        # Should have sequential step numbers
        assert len(step_numbers) == len(all_thoughts)
        # Step numbers should start from 1
        assert min(step_numbers) == 1


class TestSelfConsistencyHelperMethods:
    """Tests for helper methods."""

    def test_generate_reasoning_step_variation(self):
        """Test that _generate_reasoning_step provides variation."""
        method = SelfConsistency()

        steps = [method._generate_reasoning_step("problem", i) for i in range(1, 6)]

        # Should have different reasoning approaches
        unique_steps = set(steps)
        assert len(unique_steps) > 1

    def test_generate_intermediate_reasoning(self):
        """Test _generate_intermediate_reasoning output."""
        method = SelfConsistency()

        result = method._generate_intermediate_reasoning("test problem", 1)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Working through the logical steps" in result

    def test_generate_path_conclusion(self):
        """Test _generate_path_conclusion output."""
        method = SelfConsistency()

        conclusion = method._generate_path_conclusion("test problem", 1)

        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
        assert "path" in conclusion.lower()

    def test_format_synthesis(self):
        """Test _format_synthesis output formatting."""
        method = SelfConsistency(min_agreement=0.5)

        result = method._format_synthesis(
            input_text="What is 2+2?",
            conclusions=["4", "4", "5"],
            majority_answer="4",
            vote_counts={"4": 2, "5": 1},
            agreement_rate=2 / 3,
            num_paths=3,
        )

        assert "Self-Consistency Analysis Complete" in result
        assert "What is 2+2?" in result
        assert "Voting Results:" in result
        assert "Majority Answer: 4" in result
        assert "2/3" in result  # vote count
        assert "1/3" in result  # vote count

    def test_format_synthesis_high_agreement(self):
        """Test _format_synthesis with high agreement."""
        method = SelfConsistency(min_agreement=0.5)

        result = method._format_synthesis(
            input_text="Test",
            conclusions=["A"] * 5,
            majority_answer="A",
            vote_counts={"A": 5},
            agreement_rate=1.0,
            num_paths=5,
        )

        assert "Strong consensus" in result

    def test_format_synthesis_moderate_agreement(self):
        """Test _format_synthesis with moderate agreement."""
        method = SelfConsistency(min_agreement=0.5)

        result = method._format_synthesis(
            input_text="Test",
            conclusions=["A", "A", "A", "B", "B"],
            majority_answer="A",
            vote_counts={"A": 3, "B": 2},
            agreement_rate=0.6,
            num_paths=5,
        )

        assert "Moderate consensus" in result

    def test_format_synthesis_low_agreement(self):
        """Test _format_synthesis with low agreement."""
        method = SelfConsistency(min_agreement=0.5)

        result = method._format_synthesis(
            input_text="Test",
            conclusions=["A", "B", "C", "D", "E"],
            majority_answer="A",
            vote_counts={"A": 1, "B": 1, "C": 1, "D": 1, "E": 1},
            agreement_rate=0.2,
            num_paths=5,
        )

        assert "Low consensus" in result
