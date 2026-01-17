"""Unit tests for MoA ensembler.

Tests Mixture of Agents with layered model collaboration.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.moa import MOA_METADATA, Moa
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestMoaMetadata:
    """Tests for MoA metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert MOA_METADATA.identifier == EnsemblerIdentifier.MOA

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert MOA_METADATA.name == "MoA"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "mixture" in MOA_METADATA.tags
        assert "agents" in MOA_METADATA.tags
        assert "layered" in MOA_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert MOA_METADATA.min_models >= 2
        assert MOA_METADATA.max_models >= MOA_METADATA.min_models

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert MOA_METADATA.supports_weighted_voting is True
        assert MOA_METADATA.supports_dynamic_selection is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= MOA_METADATA.complexity <= 10


class TestMoaInitialization:
    """Tests for MoA initialization."""

    def test_create_instance(self) -> None:
        """Test creating Moa instance."""
        ensembler = Moa()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = Moa()
        assert ensembler.identifier == EnsemblerIdentifier.MOA

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = Moa()
        assert ensembler.name == "MoA"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = Moa()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_clears_layers(self) -> None:
        """Test initialize clears layers list."""
        ensembler = Moa()
        ensembler._layers = [["layer1"]]
        await ensembler.initialize()
        assert len(ensembler._layers) == 0


class TestMoaSolutionScoring:
    """Tests for MoA solution scoring."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Moa:
        """Create an initialized Moa."""
        ensembler = Moa()
        await ensembler.initialize()
        return ensembler

    def test_score_solution_base(self, initialized_ensembler: Moa) -> None:
        """Test base solution score."""
        score = initialized_ensembler._score_solution("Hi")
        assert score == 0.5

    def test_score_solution_length_bonus(self, initialized_ensembler: Moa) -> None:
        """Test longer solutions get bonus."""
        short_score = initialized_ensembler._score_solution("Hi")
        long_score = initialized_ensembler._score_solution("Longer text")
        assert long_score > short_score

    def test_score_solution_digit_bonus(self, initialized_ensembler: Moa) -> None:
        """Test solutions with digits get bonus."""
        no_digit = initialized_ensembler._score_solution("No nums")
        with_digit = initialized_ensembler._score_solution("Answer 42")
        assert with_digit > no_digit

    def test_score_solution_equals_bonus(self, initialized_ensembler: Moa) -> None:
        """Test solutions with equals sign get bonus."""
        # Use longer solutions so length bonus applies to both, isolate equals bonus
        no_equals = initialized_ensembler._score_solution("value five here")
        with_equals = initialized_ensembler._score_solution("x = five here")
        assert with_equals > no_equals

    def test_score_solution_capped(self, initialized_ensembler: Moa) -> None:
        """Test solution score is capped at 1.0."""
        score = initialized_ensembler._score_solution("x = 42 answer")
        assert score <= 1.0


class TestMoaAggregation:
    """Tests for MoA solution aggregation."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Moa:
        """Create an initialized Moa."""
        ensembler = Moa()
        await ensembler.initialize()
        return ensembler

    def test_aggregate_empty_solutions(self, initialized_ensembler: Moa) -> None:
        """Test aggregation with empty solutions."""
        result = initialized_ensembler._aggregate_solutions([])
        assert result == ""

    def test_aggregate_single_solution(self, initialized_ensembler: Moa) -> None:
        """Test aggregation with single solution."""
        result = initialized_ensembler._aggregate_solutions(["only one"])
        assert result == "only one"

    def test_aggregate_returns_best(self, initialized_ensembler: Moa) -> None:
        """Test aggregation returns best scored solution."""
        solutions = ["short", "x = 42 answer", "medium text"]
        result = initialized_ensembler._aggregate_solutions(solutions)
        # Solution with best score (digits + equals) should be selected
        assert result == "x = 42 answer"


class TestMoaRefinement:
    """Tests for MoA solution refinement."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Moa:
        """Create an initialized Moa."""
        ensembler = Moa()
        await ensembler.initialize()
        return ensembler

    def test_refine_solution_returns_solution(self, initialized_ensembler: Moa) -> None:
        """Test refinement returns the solution."""
        result = initialized_ensembler._refine_solution("test solution")
        assert result == "test solution"


class TestMoaEnsemble:
    """Tests for MoA ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Moa:
        """Create an initialized Moa."""
        ensembler = Moa()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = Moa()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(self, initialized_ensembler: Moa) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(self, initialized_ensembler: Moa) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_multiple_solutions(self, initialized_ensembler: Moa) -> None:
        """Test ensemble with multiple solutions."""
        solutions = ["solution 1", "x = 42", "solution 3"]
        result = await initialized_ensembler.ensemble("query", solutions)
        # Should return best aggregated and refined solution
        assert result == "x = 42"

    async def test_ensemble_with_context(self, initialized_ensembler: Moa) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestMoaSelectModels:
    """Tests for MoA model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Moa:
        """Create an initialized Moa."""
        ensembler = Moa()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = Moa()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(self, initialized_ensembler: Moa) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)

    async def test_select_models_respects_max(self, initialized_ensembler: Moa) -> None:
        """Test select_models respects max_models limit."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) <= MOA_METADATA.max_models

    async def test_select_models_with_context(self, initialized_ensembler: Moa) -> None:
        """Test select_models with context parameter."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2"],
            context={"domain": "math"},
        )
        assert len(models) >= 0


class TestMoaHealthCheck:
    """Tests for MoA health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = Moa()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = Moa()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
