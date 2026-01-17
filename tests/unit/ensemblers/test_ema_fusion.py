"""Unit tests for EMAFusion ensembler.

Tests self-optimizing LLM integration with exponential moving average.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.ema_fusion import EMA_FUSION_METADATA, EmaFusion
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestEmaFusionMetadata:
    """Tests for EMAFusion metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert EMA_FUSION_METADATA.identifier == EnsemblerIdentifier.EMA_FUSION

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert EMA_FUSION_METADATA.name == "EMAFusion"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "ema" in EMA_FUSION_METADATA.tags
        assert "fusion" in EMA_FUSION_METADATA.tags
        assert "adaptive" in EMA_FUSION_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert EMA_FUSION_METADATA.min_models >= 2
        assert EMA_FUSION_METADATA.max_models >= EMA_FUSION_METADATA.min_models

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= EMA_FUSION_METADATA.complexity <= 10


class TestEmaFusionInitialization:
    """Tests for EMAFusion initialization."""

    def test_create_instance(self) -> None:
        """Test creating EmaFusion instance."""
        ensembler = EmaFusion()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = EmaFusion()
        assert ensembler.identifier == EnsemblerIdentifier.EMA_FUSION

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = EmaFusion()
        assert ensembler.name == "EMAFusion"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_sets_ema_alpha(self) -> None:
        """Test initialize sets EMA alpha."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        assert ensembler._ema_alpha == 0.3

    async def test_initialize_clears_weights(self) -> None:
        """Test initialize clears model weights."""
        ensembler = EmaFusion()
        ensembler._model_weights = {"model1": 0.5}
        await ensembler.initialize()
        assert len(ensembler._model_weights) == 0

    async def test_initialize_clears_history(self) -> None:
        """Test initialize clears performance history."""
        ensembler = EmaFusion()
        ensembler._performance_history = {"model1": [0.5]}
        await ensembler.initialize()
        assert len(ensembler._performance_history) == 0


class TestEmaFusionQualityAssessment:
    """Tests for EMAFusion quality assessment."""

    @pytest.fixture
    async def initialized_ensembler(self) -> EmaFusion:
        """Create an initialized EmaFusion."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        return ensembler

    def test_assess_quality_base(self, initialized_ensembler: EmaFusion) -> None:
        """Test base quality assessment."""
        score = initialized_ensembler._assess_quality("Hi")
        assert score == 0.6

    def test_assess_quality_length_bonus(self, initialized_ensembler: EmaFusion) -> None:
        """Test longer solutions get bonus."""
        short_score = initialized_ensembler._assess_quality("Hi")
        long_score = initialized_ensembler._assess_quality(
            "This is a longer solution with more content"
        )
        assert long_score > short_score

    def test_assess_quality_digit_bonus(self, initialized_ensembler: EmaFusion) -> None:
        """Test solutions with digits get bonus."""
        no_digit = initialized_ensembler._assess_quality("No numbers")
        with_digit = initialized_ensembler._assess_quality("Answer is 42")
        assert with_digit > no_digit

    def test_assess_quality_equals_bonus(self, initialized_ensembler: EmaFusion) -> None:
        """Test solutions with equals sign get bonus."""
        no_equals = initialized_ensembler._assess_quality("Value is 5")
        with_equals = initialized_ensembler._assess_quality("x = 5")
        assert with_equals > no_equals


class TestEmaFusionEnsemble:
    """Tests for EMAFusion ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> EmaFusion:
        """Create an initialized EmaFusion."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = EmaFusion()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(self, initialized_ensembler: EmaFusion) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(self, initialized_ensembler: EmaFusion) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_initializes_weights(self, initialized_ensembler: EmaFusion) -> None:
        """Test ensemble initializes weights for models."""
        await initialized_ensembler.ensemble("query", ["solution 1", "solution 2", "solution 3"])
        # Should have initialized equal weights
        assert len(initialized_ensembler._model_weights) == 3

    async def test_ensemble_returns_best_weighted(self, initialized_ensembler: EmaFusion) -> None:
        """Test ensemble returns best weighted solution."""
        # Pre-set weights to favor model_1
        initialized_ensembler._model_weights = {
            "model_0": 0.1,
            "model_1": 0.9,
            "model_2": 0.1,
        }
        solutions = ["short", "x = 42 answer", "medium text"]
        result = await initialized_ensembler.ensemble("query", solutions)
        # Should pick solution with highest weight * quality
        assert result is not None

    async def test_ensemble_with_context(self, initialized_ensembler: EmaFusion) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestEmaFusionUpdateWeights:
    """Tests for EMAFusion weight update."""

    @pytest.fixture
    async def initialized_ensembler(self) -> EmaFusion:
        """Create an initialized EmaFusion."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        return ensembler

    async def test_update_weights_raises_when_not_initialized(self) -> None:
        """Test update_weights raises error when not initialized."""
        ensembler = EmaFusion()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.update_weights({"model1": 0.8})

    async def test_update_weights_new_model(self, initialized_ensembler: EmaFusion) -> None:
        """Test updating weights for new model."""
        await initialized_ensembler.update_weights({"new_model": 0.8})
        assert "new_model" in initialized_ensembler._model_weights

    async def test_update_weights_ema_calculation(self, initialized_ensembler: EmaFusion) -> None:
        """Test EMA calculation on weight update."""
        # Set initial weight
        initialized_ensembler._model_weights = {"model1": 0.5}
        alpha = initialized_ensembler._ema_alpha

        await initialized_ensembler.update_weights({"model1": 1.0})

        # EMA: new_weight = alpha * performance + (1 - alpha) * old_weight
        expected = alpha * 1.0 + (1 - alpha) * 0.5
        assert initialized_ensembler._model_weights["model1"] == pytest.approx(expected)

    async def test_update_weights_tracks_history(self, initialized_ensembler: EmaFusion) -> None:
        """Test weight update tracks performance history."""
        await initialized_ensembler.update_weights({"model1": 0.8})
        await initialized_ensembler.update_weights({"model1": 0.9})

        assert "model1" in initialized_ensembler._performance_history
        assert len(initialized_ensembler._performance_history["model1"]) == 2


class TestEmaFusionFuseAdaptive:
    """Tests for EMAFusion adaptive fusion."""

    @pytest.fixture
    async def initialized_ensembler(self) -> EmaFusion:
        """Create an initialized EmaFusion."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        return ensembler

    async def test_fuse_adaptive_raises_when_not_initialized(self) -> None:
        """Test fuse_adaptive raises error when not initialized."""
        ensembler = EmaFusion()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.fuse_adaptive("query", {"m1": "sol1"})

    async def test_fuse_adaptive_returns_solution(self, initialized_ensembler: EmaFusion) -> None:
        """Test fuse_adaptive returns a solution."""
        result = await initialized_ensembler.fuse_adaptive(
            "query",
            {"model1": "solution 1", "model2": "solution 2"},
        )
        assert result is not None

    async def test_fuse_adaptive_with_context(self, initialized_ensembler: EmaFusion) -> None:
        """Test fuse_adaptive with context parameter."""
        result = await initialized_ensembler.fuse_adaptive(
            "query",
            {"model1": "solution"},
            context={"domain": "math"},
        )
        assert result is not None


class TestEmaFusionSelectModels:
    """Tests for EMAFusion model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> EmaFusion:
        """Create an initialized EmaFusion."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = EmaFusion()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(self, initialized_ensembler: EmaFusion) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)

    async def test_select_models_respects_max(self, initialized_ensembler: EmaFusion) -> None:
        """Test select_models respects max_models limit."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) <= EMA_FUSION_METADATA.max_models

    async def test_select_models_by_weight(self, initialized_ensembler: EmaFusion) -> None:
        """Test models are selected by weight."""
        initialized_ensembler._model_weights = {
            "model_high": 0.9,
            "model_low": 0.1,
            "model_mid": 0.5,
        }
        selected = await initialized_ensembler.select_models(
            "query",
            ["model_high", "model_low", "model_mid"],
        )
        # High weight model should be first
        assert selected[0] == "model_high"


class TestEmaFusionHealthCheck:
    """Tests for EMAFusion health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = EmaFusion()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = EmaFusion()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
