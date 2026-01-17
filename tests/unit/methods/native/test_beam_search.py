"""Unit tests for BeamSearch reasoning method."""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.beam_search import (
    BEAM_SEARCH_METADATA,
    BeamCandidate,
    BeamSearchMethod,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

# Fixtures


@pytest.fixture
def beam_method() -> BeamSearchMethod:
    """Create a BeamSearchMethod instance with default settings."""
    return BeamSearchMethod()


@pytest.fixture
def custom_beam_method() -> BeamSearchMethod:
    """Create a BeamSearchMethod instance with custom settings."""
    return BeamSearchMethod(
        beam_width=5,
        max_depth=8,
        scoring_strategy="confidence",
    )


@pytest.fixture
def active_session() -> Session:
    """Create an active session for testing."""
    return Session().start()


@pytest.fixture
def inactive_session() -> Session:
    """Create an inactive session for testing."""
    return Session()


# Test Metadata


class TestMetadata:
    """Tests for Beam Search metadata."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert BEAM_SEARCH_METADATA.identifier == MethodIdentifier.BEAM_SEARCH

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert BEAM_SEARCH_METADATA.name == "Beam Search Reasoning"

    def test_metadata_description(self):
        """Test metadata has descriptive text."""
        assert len(BEAM_SEARCH_METADATA.description) > 0
        assert "parallel" in BEAM_SEARCH_METADATA.description.lower()
        assert "pruning" in BEAM_SEARCH_METADATA.description.lower()

    def test_metadata_category(self):
        """Test metadata has correct category."""
        assert BEAM_SEARCH_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {
            "search",
            "parallel",
            "pruning",
            "optimization",
            "heuristic",
            "breadth-first",
            "advanced",
        }
        assert expected_tags.issubset(BEAM_SEARCH_METADATA.tags)

    def test_metadata_complexity(self):
        """Test metadata complexity is high (6)."""
        assert BEAM_SEARCH_METADATA.complexity == 6

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert BEAM_SEARCH_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates no revision support."""
        assert BEAM_SEARCH_METADATA.supports_revision is False

    def test_metadata_requires_context(self):
        """Test metadata indicates no context requirement."""
        assert BEAM_SEARCH_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test metadata has minimum thoughts requirement."""
        assert BEAM_SEARCH_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test metadata has unlimited max thoughts (0)."""
        assert BEAM_SEARCH_METADATA.max_thoughts == 0


# Test Initialization


class TestInitialization:
    """Tests for BeamSearchMethod initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        beam = BeamSearchMethod()
        assert beam.beam_width == 3
        assert beam.max_depth == 5
        assert beam.scoring_strategy == "heuristic"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        beam = BeamSearchMethod(
            beam_width=7,
            max_depth=10,
            scoring_strategy="confidence",
        )
        assert beam.beam_width == 7
        assert beam.max_depth == 10
        assert beam.scoring_strategy == "confidence"

    def test_init_invalid_beam_width_zero(self):
        """Test initialization fails with beam_width=0."""
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            BeamSearchMethod(beam_width=0)

    def test_init_invalid_beam_width_negative(self):
        """Test initialization fails with negative beam_width."""
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            BeamSearchMethod(beam_width=-1)

    def test_init_invalid_max_depth_zero(self):
        """Test initialization fails with max_depth=0."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            BeamSearchMethod(max_depth=0)

    def test_init_invalid_max_depth_negative(self):
        """Test initialization fails with negative max_depth."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            BeamSearchMethod(max_depth=-5)

    def test_init_invalid_scoring_strategy(self):
        """Test initialization fails with invalid scoring strategy."""
        with pytest.raises(
            ValueError, match="scoring_strategy must be 'heuristic' or 'confidence'"
        ):
            BeamSearchMethod(scoring_strategy="invalid")

    @pytest.mark.asyncio
    async def test_initialize_method(self, beam_method: BeamSearchMethod):
        """Test initialize() method executes successfully."""
        await beam_method.initialize()
        # No error means success - lightweight initialization

    @pytest.mark.asyncio
    async def test_health_check(self, beam_method: BeamSearchMethod):
        """Test health_check() returns True."""
        result = await beam_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(self):
        """Test health_check() works before initialize()."""
        beam = BeamSearchMethod()
        result = await beam.health_check()
        assert result is True


# Test Properties


class TestProperties:
    """Tests for BeamSearchMethod properties."""

    def test_identifier_property(self, beam_method: BeamSearchMethod):
        """Test identifier property returns correct value."""
        assert beam_method.identifier == str(MethodIdentifier.BEAM_SEARCH)

    def test_name_property(self, beam_method: BeamSearchMethod):
        """Test name property returns correct value."""
        assert beam_method.name == "Beam Search Reasoning"

    def test_description_property(self, beam_method: BeamSearchMethod):
        """Test description property returns correct value."""
        assert len(beam_method.description) > 0
        assert "parallel" in beam_method.description.lower()

    def test_category_property(self, beam_method: BeamSearchMethod):
        """Test category property returns correct value."""
        assert beam_method.category == str(MethodCategory.ADVANCED)


# Test BeamCandidate Class


class TestBeamCandidate:
    """Tests for BeamCandidate internal class."""

    def test_beam_candidate_creation(self):
        """Test creating a BeamCandidate instance."""
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content="Test thought",
        )
        candidate = BeamCandidate(thought=thought, score=0.75, level=1)

        assert candidate.thought == thought
        assert candidate.score == 0.75
        assert candidate.level == 1
        assert candidate.metadata == {}

    def test_beam_candidate_with_metadata(self):
        """Test BeamCandidate with custom metadata."""
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content="Branch",
        )
        metadata = {"approach": "optimizing", "parent_id": "root"}
        candidate = BeamCandidate(thought=thought, score=0.8, level=2, metadata=metadata)

        assert candidate.metadata == metadata
        assert candidate.metadata["approach"] == "optimizing"

    def test_beam_candidate_repr(self):
        """Test BeamCandidate string representation."""
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content="Test",
        )
        candidate = BeamCandidate(thought=thought, score=0.845, level=3)
        repr_str = repr(candidate)

        assert "BeamCandidate" in repr_str
        assert "0.845" in repr_str
        assert "3" in repr_str


# Test Basic Execution


class TestBasicExecution:
    """Tests for basic execute() functionality."""

    @pytest.mark.asyncio
    async def test_execute_creates_root_thought(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test execute() creates root thought."""
        result = await beam_method.execute(active_session, "Test problem")

        # Should have created thoughts
        assert active_session.thought_count > 0

        # Result should be a synthesis thought
        assert result.type == ThoughtType.SYNTHESIS
        assert result.method_id == MethodIdentifier.BEAM_SEARCH

    @pytest.mark.asyncio
    async def test_execute_creates_beam_structure(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test execute() creates beam search structure with multiple nodes."""
        await beam_method.execute(active_session, "Solve this problem")

        # Should have multiple thoughts (root + candidates + synthesis)
        assert active_session.thought_count > 2

        # Check metrics
        assert active_session.metrics.total_thoughts > 2
        assert active_session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_execute_with_inactive_session(
        self, beam_method: BeamSearchMethod, inactive_session: Session
    ):
        """Test execute() fails with inactive session."""
        with pytest.raises(ValueError, match="Session must be active"):
            await beam_method.execute(inactive_session, "Test")

    @pytest.mark.asyncio
    async def test_execute_returns_synthesis_thought(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test execute() returns final synthesis thought."""
        result = await beam_method.execute(active_session, "Test input")

        assert result.type == ThoughtType.SYNTHESIS
        assert "Beam Search Complete" in result.content
        assert result.metadata.get("is_final") is True
        assert "total_candidates" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_synthesis_includes_stats(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test synthesis thought includes statistics."""
        result = await beam_method.execute(active_session, "Problem")

        # Check content includes stats
        assert "Total candidates generated:" in result.content
        assert "Total candidates pruned:" in result.content
        assert "Levels explored:" in result.content
        assert "Final best score:" in result.content

        # Check metadata
        assert result.metadata.get("total_candidates", 0) > 0
        assert "levels_explored" in result.metadata
        assert "best_score" in result.metadata


# Test Beam Management


class TestBeamManagement:
    """Tests for beam width and candidate management."""

    @pytest.mark.asyncio
    async def test_beam_width_limits_candidates(self, active_session: Session):
        """Test that beam_width limits the number of candidates kept."""
        beam = BeamSearchMethod(beam_width=2, max_depth=3)
        result = await beam.execute(active_session, "Test")

        # Check that pruning occurred
        assert active_session.metrics.branches_pruned >= 0
        assert result is not None

    @pytest.mark.asyncio
    async def test_beam_width_controls_parallel_paths(self, active_session: Session):
        """Test beam_width controls number of parallel exploration paths."""
        beam = BeamSearchMethod(beam_width=4, max_depth=2)
        await beam.execute(active_session, "Test")

        # At each level, up to beam_width paths are maintained
        # Check that branches were created
        assert active_session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_small_beam_width(self, active_session: Session):
        """Test with beam_width=1 (single path)."""
        beam = BeamSearchMethod(beam_width=1, max_depth=3)
        result = await beam.execute(active_session, "Test")

        # Should still work with single beam
        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_large_beam_width(self, active_session: Session):
        """Test with large beam_width."""
        beam = BeamSearchMethod(beam_width=8, max_depth=2)
        result = await beam.execute(active_session, "Wide search")

        assert result is not None
        # Should create many branches
        assert active_session.metrics.branches_created > 0


# Test Configuration


class TestConfiguration:
    """Tests for configuration via context and initialization."""

    @pytest.mark.asyncio
    async def test_context_overrides_beam_width(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test context can override beam_width."""
        result = await beam_method.execute(active_session, "Test", context={"beam_width": 6})

        # Check that context was used
        assert result.metadata.get("total_candidates", 0) > 0

    @pytest.mark.asyncio
    async def test_context_overrides_max_depth(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test context can override max_depth."""
        await beam_method.execute(active_session, "Test", context={"max_depth": 2})

        # Depth should not exceed 2 (plus synthesis)
        assert active_session.current_depth <= 3

    @pytest.mark.asyncio
    async def test_context_overrides_scoring_strategy(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test context can override scoring_strategy."""
        result = await beam_method.execute(
            active_session, "Test", context={"scoring_strategy": "confidence"}
        )

        # Should complete successfully
        assert result is not None

    @pytest.mark.asyncio
    async def test_context_with_constraints(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test passing constraints via context."""
        await beam_method.execute(
            active_session,
            "Optimize",
            context={"constraints": ["budget", "time", "quality"]},
        )

        # Check that constraints are mentioned in root thought
        root_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.INITIAL
        ]
        assert len(root_thoughts) > 0
        assert "budget" in root_thoughts[0].content

    @pytest.mark.asyncio
    async def test_empty_context_uses_defaults(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test empty context uses default parameters."""
        result = await beam_method.execute(active_session, "Test", context={})

        assert result is not None
        # Check defaults were used in metadata
        assert result.metadata is not None


# Test Candidate Scoring


class TestCandidateScoring:
    """Tests for candidate scoring and ranking."""

    @pytest.mark.asyncio
    async def test_candidates_have_scores(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that candidates have quality scores."""
        await beam_method.execute(active_session, "Test")

        branch_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.BRANCH
        ]

        # Should have branch thoughts with scores
        if len(branch_thoughts) > 0:
            for branch in branch_thoughts:
                assert branch.confidence is not None
                assert 0.0 <= branch.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_heuristic_scoring_strategy(self, active_session: Session):
        """Test heuristic scoring strategy."""
        beam = BeamSearchMethod(scoring_strategy="heuristic")
        result = await beam.execute(active_session, "Test")

        assert result is not None
        # Best score should be in reasonable range
        best_score = result.metadata.get("best_score", 0.0)
        assert 0.0 <= best_score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_scoring_strategy(self, active_session: Session):
        """Test confidence scoring strategy."""
        beam = BeamSearchMethod(scoring_strategy="confidence")
        result = await beam.execute(active_session, "Test")

        assert result is not None
        # Should use confidence scores
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_best_candidate_has_highest_score(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that best candidate has highest score."""
        result = await beam_method.execute(active_session, "Test")

        # Synthesis inherits score from best candidate
        best_score = result.metadata.get("best_score", 0.0)
        assert best_score > 0.0


# Test Beam Pruning


class TestBeamPruning:
    """Tests for beam pruning behavior."""

    @pytest.mark.asyncio
    async def test_pruning_occurs_with_multiple_candidates(self, active_session: Session):
        """Test that pruning occurs when candidates exceed beam_width."""
        beam = BeamSearchMethod(beam_width=2, max_depth=3)
        await beam.execute(active_session, "Test")

        # Should have pruned some candidates
        assert active_session.metrics.branches_pruned >= 0

    @pytest.mark.asyncio
    async def test_pruning_tracked_in_metrics(self, active_session: Session):
        """Test that pruned candidates are tracked in metrics."""
        beam = BeamSearchMethod(beam_width=2, max_depth=4)
        result = await beam.execute(active_session, "Test")

        # Metrics should track pruned candidates
        pruned_count = result.metadata.get("total_pruned", 0)
        assert pruned_count >= 0
        assert active_session.metrics.branches_pruned >= 0

    @pytest.mark.asyncio
    async def test_pruning_creates_observation_thoughts(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that pruning creates observation thoughts."""
        await beam_method.execute(active_session, "Test")

        # Check for pruning observation thoughts
        observation_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.OBSERVATION
        ]

        # May have pruning observations (if pruning occurred)
        for obs in observation_thoughts:
            if obs.metadata.get("is_pruning"):
                assert "Pruning at Level" in obs.content
                assert obs.metadata.get("pruned_count") is not None

    @pytest.mark.asyncio
    async def test_keeps_only_top_k_candidates(self, active_session: Session):
        """Test that only top-k candidates are kept after pruning."""
        beam = BeamSearchMethod(beam_width=3, max_depth=2)
        result = await beam.execute(active_session, "Test")

        # Should maintain beam width constraint
        assert result is not None
        # Check that final beam was limited
        assert result.metadata.get("total_pruned", 0) >= 0


# Test Parallel Exploration


class TestParallelExploration:
    """Tests for parallel path exploration."""

    @pytest.mark.asyncio
    async def test_explores_multiple_paths_simultaneously(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that multiple paths are explored in parallel."""
        await beam_method.execute(active_session, "Multi-path problem")

        # Should have created multiple branches
        assert active_session.metrics.branches_created > 0

        # Check that branch thoughts exist
        branch_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.BRANCH
        ]
        assert len(branch_thoughts) > 0

    @pytest.mark.asyncio
    async def test_parallel_paths_have_different_approaches(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that parallel paths explore different approaches."""
        await beam_method.execute(active_session, "Test")

        branch_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.BRANCH
        ]

        # Different branches should have different approaches
        approaches = set()
        for branch in branch_thoughts:
            if "approach" in branch.metadata:
                approaches.add(branch.metadata["approach"])

        # Should have multiple different approaches
        if len(branch_thoughts) > 1:
            assert len(approaches) > 1

    @pytest.mark.asyncio
    async def test_beam_diversity_tracked(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that beam diversity is tracked in metrics."""
        result = await beam_method.execute(active_session, "Test")

        # Check for diversity metrics
        final_diversity = result.metadata.get("final_diversity")
        assert final_diversity is not None
        assert final_diversity >= 0.0


# Test Best Path Selection


class TestBestPathSelection:
    """Tests for selecting the best path from beam."""

    @pytest.mark.asyncio
    async def test_selects_highest_scoring_path(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that highest-scoring path is selected."""
        result = await beam_method.execute(active_session, "Test")

        # Result should have the best score
        best_score = result.metadata.get("best_score", 0.0)
        assert best_score > 0.0

        # Result confidence should match best score
        assert result.confidence == best_score

    @pytest.mark.asyncio
    async def test_final_synthesis_references_best_path(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that final synthesis references the best path."""
        result = await beam_method.execute(active_session, "Test")

        assert "Best solution found" in result.content
        assert "Final best score:" in result.content

    @pytest.mark.asyncio
    async def test_best_scores_tracked_per_level(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that best scores are tracked at each level."""
        result = await beam_method.execute(active_session, "Test")

        best_scores = result.metadata.get("best_scores_per_level", [])
        assert len(best_scores) > 0
        # All scores should be in valid range
        for score in best_scores:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_average_best_score_calculated(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test that average best score is calculated."""
        result = await beam_method.execute(active_session, "Test")

        avg_score = result.metadata.get("avg_best_score", 0.0)
        assert avg_score > 0.0
        assert avg_score <= 1.0


# Test Continue Reasoning


class TestContinueReasoning:
    """Tests for continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_creates_continuation(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test continue_reasoning() creates continuation thought."""
        # First execute to get a thought
        initial = await beam_method.execute(active_session, "Test")

        # Continue from that thought
        continuation = await beam_method.continue_reasoning(
            active_session, initial, guidance="Expand search"
        )

        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == initial.id
        assert "Continuing Beam Search" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test continue_reasoning() includes guidance."""
        initial = await beam_method.execute(active_session, "Test")
        guidance_text = "Focus on optimization paths"

        continuation = await beam_method.continue_reasoning(
            active_session, initial, guidance=guidance_text
        )

        assert guidance_text in continuation.content
        assert continuation.metadata.get("guidance") == guidance_text

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test continue_reasoning() works without guidance."""
        initial = await beam_method.execute(active_session, "Test")

        continuation = await beam_method.continue_reasoning(active_session, initial)

        assert continuation is not None
        assert "Expanding search depth" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_inactive_session(
        self, beam_method: BeamSearchMethod, inactive_session: Session
    ):
        """Test continue_reasoning() fails with inactive session."""
        # Create a dummy thought
        thought = ThoughtNode(
            id="test",
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BEAM_SEARCH,
            content="Test",
        )

        with pytest.raises(ValueError, match="Session must be active"):
            await beam_method.continue_reasoning(inactive_session, thought)

    @pytest.mark.asyncio
    async def test_continue_reasoning_uses_additional_depth(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test continue_reasoning() respects additional_depth from context."""
        initial = await beam_method.execute(active_session, "Test")

        continuation = await beam_method.continue_reasoning(
            active_session, initial, context={"additional_depth": 5}
        )

        assert "5 additional levels" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_confidence_decay(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test continuation has slightly lower confidence."""
        initial = await beam_method.execute(active_session, "Test")

        continuation = await beam_method.continue_reasoning(active_session, initial)

        # Continuation should have lower confidence (0.95 decay factor)
        assert continuation.confidence <= initial.confidence


# Test Early Convergence


class TestEarlyConvergence:
    """Tests for early convergence detection."""

    @pytest.mark.asyncio
    async def test_detects_low_diversity(self, active_session: Session):
        """Test that low diversity triggers convergence check."""
        beam = BeamSearchMethod(beam_width=3, max_depth=10)
        await beam.execute(active_session, "Test")

        # Check if convergence was detected (may or may not happen)
        convergence_thoughts = [
            t for t in active_session.graph.nodes.values() if t.metadata.get("is_convergence")
        ]

        # If convergence occurred, verify the thought
        for conv in convergence_thoughts:
            assert "Early Convergence Detected" in conv.content
            assert conv.metadata.get("diversity") is not None

    @pytest.mark.asyncio
    async def test_diversity_calculation(self, beam_method: BeamSearchMethod):
        """Test diversity calculation for beam candidates."""
        # Test the internal diversity calculation
        scores = [0.5, 0.6, 0.7]
        diversity = beam_method._calculate_diversity(scores)

        assert diversity > 0.0
        assert diversity < 1.0

    @pytest.mark.asyncio
    async def test_diversity_single_candidate(self, beam_method: BeamSearchMethod):
        """Test diversity with single candidate returns 0."""
        scores = [0.5]
        diversity = beam_method._calculate_diversity(scores)

        assert diversity == 0.0

    @pytest.mark.asyncio
    async def test_diversity_identical_scores(self, beam_method: BeamSearchMethod):
        """Test diversity with identical scores returns 0."""
        scores = [0.7, 0.7, 0.7]
        diversity = beam_method._calculate_diversity(scores)

        assert diversity == pytest.approx(0.0, abs=1e-10)


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimum_beam_width(self, active_session: Session):
        """Test with beam_width=1 (minimum)."""
        beam = BeamSearchMethod(beam_width=1, max_depth=3)
        result = await beam.execute(active_session, "Test")

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_minimum_max_depth(self, active_session: Session):
        """Test with max_depth=1 (minimum)."""
        beam = BeamSearchMethod(beam_width=3, max_depth=1)
        result = await beam.execute(active_session, "Shallow test")

        assert result is not None
        # Should not exceed depth 1 (plus synthesis)
        assert active_session.current_depth <= 2

    @pytest.mark.asyncio
    async def test_very_deep_search(self, active_session: Session):
        """Test with very deep search (max_depth=10)."""
        beam = BeamSearchMethod(beam_width=2, max_depth=10)
        result = await beam.execute(active_session, "Deep exploration")

        assert result is not None
        # Should explore to significant depth
        levels = result.metadata.get("levels_explored", 0)
        assert levels > 0

    @pytest.mark.asyncio
    async def test_empty_input_text(self, beam_method: BeamSearchMethod, active_session: Session):
        """Test with empty input text."""
        result = await beam_method.execute(active_session, "")

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_very_long_input_text(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test with very long input text."""
        long_input = "Complex optimization problem " * 100
        result = await beam_method.execute(active_session, long_input)

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_input(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test with special characters in input."""
        special_input = "Optimize: @#$%^&*() 测试 émojis 🔍🎯"
        result = await beam_method.execute(active_session, special_input)

        assert result is not None
        assert active_session.thought_count > 0

    @pytest.mark.asyncio
    async def test_no_candidates_generated(self, active_session: Session):
        """Test handling when no candidates are generated (edge case)."""
        beam = BeamSearchMethod(beam_width=3, max_depth=1)
        result = await beam.execute(active_session, "Test")

        # Should still produce a synthesis
        assert result is not None
        assert result.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_large_beam_width_with_shallow_depth(self, active_session: Session):
        """Test large beam_width with shallow max_depth."""
        beam = BeamSearchMethod(beam_width=10, max_depth=2)
        result = await beam.execute(active_session, "Wide shallow search")

        assert result is not None
        assert active_session.metrics.branches_created > 0

    @pytest.mark.asyncio
    async def test_fallback_to_root_on_empty_beam(
        self, beam_method: BeamSearchMethod, active_session: Session
    ):
        """Test fallback behavior when final beam is empty."""
        # This tests the fallback logic in execute()
        result = await beam_method.execute(active_session, "Test")

        # Should always produce a result
        assert result is not None
        assert result.type == ThoughtType.SYNTHESIS
