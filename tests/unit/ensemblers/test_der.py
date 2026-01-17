"""Unit tests for DER ensembler.

Tests Dynamic Ensemble Reasoning - models ensemble as MDP.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.der import DER_METADATA, Der
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestDerMetadata:
    """Tests for DER metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert DER_METADATA.identifier == EnsemblerIdentifier.DER

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert DER_METADATA.name == "DER"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "mdp" in DER_METADATA.tags
        assert "dynamic" in DER_METADATA.tags
        assert "ensemble" in DER_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert DER_METADATA.min_models >= 2
        assert DER_METADATA.max_models >= DER_METADATA.min_models

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= DER_METADATA.complexity <= 10


class TestDerInitialization:
    """Tests for DER initialization."""

    def test_create_instance(self) -> None:
        """Test creating Der instance."""
        ensembler = Der()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = Der()
        assert ensembler.identifier == EnsemblerIdentifier.DER

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = Der()
        assert ensembler.name == "DER"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = Der()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_sets_state(self) -> None:
        """Test initialize sets initial state."""
        ensembler = Der()
        await ensembler.initialize()
        assert "step" in ensembler._state
        assert "confidence" in ensembler._state
        assert ensembler._state["step"] == 0
        assert ensembler._state["confidence"] == 0.0


class TestDerQualityEstimation:
    """Tests for DER quality estimation."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Der:
        """Create an initialized Der."""
        ensembler = Der()
        await ensembler.initialize()
        return ensembler

    def test_estimate_quality_base(self, initialized_ensembler: Der) -> None:
        """Test base quality estimation."""
        score = initialized_ensembler._estimate_quality("Hi")
        assert score == 0.5

    def test_estimate_quality_length_bonus(self, initialized_ensembler: Der) -> None:
        """Test longer solutions get bonus."""
        short_score = initialized_ensembler._estimate_quality("Hi")
        long_score = initialized_ensembler._estimate_quality("This is a longer solution")
        assert long_score > short_score

    def test_estimate_quality_digit_bonus(self, initialized_ensembler: Der) -> None:
        """Test solutions with digits get bonus."""
        no_digit = initialized_ensembler._estimate_quality("No numbers here")
        with_digit = initialized_ensembler._estimate_quality("The answer is 42")
        assert with_digit > no_digit

    def test_estimate_quality_equals_bonus(self, initialized_ensembler: Der) -> None:
        """Test solutions with equals sign get bonus."""
        # Use same length solutions to isolate equals bonus
        no_equals = initialized_ensembler._estimate_quality("value is five")
        with_equals = initialized_ensembler._estimate_quality("the value = 5")
        assert with_equals > no_equals

    def test_estimate_quality_keyword_bonus(self, initialized_ensembler: Der) -> None:
        """Test solutions with answer keywords get bonus."""
        basic = initialized_ensembler._estimate_quality("Something here")
        therefore = initialized_ensembler._estimate_quality("Therefore, we have")
        answer = initialized_ensembler._estimate_quality("The answer is clear")
        result = initialized_ensembler._estimate_quality("The result follows")

        assert therefore > basic
        assert answer > basic
        assert result > basic

    def test_estimate_quality_capped(self, initialized_ensembler: Der) -> None:
        """Test quality score is capped at 1.0."""
        # Solution with all bonuses
        score = initialized_ensembler._estimate_quality(
            "Therefore, the answer = 42, result confirmed"
        )
        assert score <= 1.0


class TestDerEnsemble:
    """Tests for DER ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Der:
        """Create an initialized Der."""
        ensembler = Der()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = Der()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(self, initialized_ensembler: Der) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(self, initialized_ensembler: Der) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_returns_best(self, initialized_ensembler: Der) -> None:
        """Test ensemble returns best quality solution."""
        solutions = [
            "Short",
            "The answer = 42, therefore correct",  # Best quality
            "Medium length text",
        ]
        result = await initialized_ensembler.ensemble("query", solutions)
        assert result == solutions[1]

    async def test_ensemble_updates_state(self, initialized_ensembler: Der) -> None:
        """Test ensemble updates MDP state."""
        await initialized_ensembler.ensemble("query", ["solution 1", "solution 2"])
        assert initialized_ensembler._state["step"] == 1
        assert initialized_ensembler._state["confidence"] > 0.0

    async def test_ensemble_with_context(self, initialized_ensembler: Der) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestDerSelectModels:
    """Tests for DER model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> Der:
        """Create an initialized Der."""
        ensembler = Der()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = Der()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(self, initialized_ensembler: Der) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)
        assert len(models) >= 2

    async def test_select_models_complexity_based(self, initialized_ensembler: Der) -> None:
        """Test model selection is based on query complexity."""
        short_query = "Hi"
        long_query = "x" * 300  # Complex query

        available = ["m1", "m2", "m3", "m4", "m5"]

        short_selection = await initialized_ensembler.select_models(short_query, available)
        long_selection = await initialized_ensembler.select_models(long_query, available)

        # Long query should select more models
        assert len(long_selection) >= len(short_selection)

    async def test_select_models_with_context(self, initialized_ensembler: Der) -> None:
        """Test select_models with context parameter."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2"],
            context={"domain": "math"},
        )
        assert len(models) >= 2


class TestDerHealthCheck:
    """Tests for DER health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = Der()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = Der()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
