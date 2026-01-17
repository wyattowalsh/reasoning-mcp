"""Unit tests for SLM-MUX ensembler.

Tests small language model multiplexing.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.slm_mux import SLM_MUX_METADATA, SlmMux
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestSlmMuxMetadata:
    """Tests for SLM-MUX metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert SLM_MUX_METADATA.identifier == EnsemblerIdentifier.SLM_MUX

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert SLM_MUX_METADATA.name == "SLM-MUX"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "slm" in SLM_MUX_METADATA.tags
        assert "multiplexing" in SLM_MUX_METADATA.tags
        assert "efficient" in SLM_MUX_METADATA.tags
        assert "orchestration" in SLM_MUX_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert SLM_MUX_METADATA.min_models >= 2
        assert SLM_MUX_METADATA.max_models >= SLM_MUX_METADATA.min_models

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= SLM_MUX_METADATA.complexity <= 10


class TestSlmMuxInitialization:
    """Tests for SLM-MUX initialization."""

    def test_create_instance(self) -> None:
        """Test creating SlmMux instance."""
        ensembler = SlmMux()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = SlmMux()
        assert ensembler.identifier == EnsemblerIdentifier.SLM_MUX

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = SlmMux()
        assert ensembler.name == "SLM-MUX"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = SlmMux()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_clears_active_slms(self) -> None:
        """Test initialize clears active SLMs."""
        ensembler = SlmMux()
        ensembler._active_slms = ["slm1", "slm2"]
        await ensembler.initialize()
        assert ensembler._active_slms == []

    async def test_initialize_clears_weights(self) -> None:
        """Test initialize clears SLM weights."""
        ensembler = SlmMux()
        ensembler._slm_weights = {"slm1": 0.5}
        await ensembler.initialize()
        assert ensembler._slm_weights == {}


class TestSlmMuxWeightedSelection:
    """Tests for SLM-MUX weighted selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> SlmMux:
        """Create an initialized SlmMux."""
        ensembler = SlmMux()
        await ensembler.initialize()
        return ensembler

    def test_weighted_selection_empty(self, initialized_ensembler: SlmMux) -> None:
        """Test weighted selection with empty list."""
        result = initialized_ensembler._weighted_selection([])
        assert result == ""

    def test_weighted_selection_single(self, initialized_ensembler: SlmMux) -> None:
        """Test weighted selection with single solution."""
        result = initialized_ensembler._weighted_selection([("solution", 1.0)])
        assert result == "solution"

    def test_weighted_selection_prefers_quality(self, initialized_ensembler: SlmMux) -> None:
        """Test weighted selection prefers quality indicators."""
        result = initialized_ensembler._weighted_selection(
            [
                ("short", 1.0),
                ("x = 42 longer solution", 1.0),
            ]
        )
        # Solution with quality indicators should be preferred
        assert "42" in result

    def test_weighted_selection_length_bonus(self, initialized_ensembler: SlmMux) -> None:
        """Test weighted selection gives length bonus."""
        result = initialized_ensembler._weighted_selection(
            [
                ("hi", 1.0),
                ("longer solution text here", 1.0),
            ]
        )
        # Longer solution should be preferred (more than 10 chars)
        assert len(result) > 2


class TestSlmMuxEnsemble:
    """Tests for SLM-MUX ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> SlmMux:
        """Create an initialized SlmMux."""
        ensembler = SlmMux()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = SlmMux()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(self, initialized_ensembler: SlmMux) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(self, initialized_ensembler: SlmMux) -> None:
        """Test ensemble with single solution returns that solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_returns_best_quality(self, initialized_ensembler: SlmMux) -> None:
        """Test ensemble returns solution with best quality."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["short", "x = 42 longer answer", "medium text"],
        )
        assert "42" in result

    async def test_ensemble_with_context(self, initialized_ensembler: SlmMux) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestSlmMuxOrchestrateSlms:
    """Tests for SLM-MUX orchestrate_slms method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> SlmMux:
        """Create an initialized SlmMux."""
        ensembler = SlmMux()
        await ensembler.initialize()
        return ensembler

    async def test_orchestrate_raises_when_not_initialized(self) -> None:
        """Test orchestrate_slms raises error when not initialized."""
        ensembler = SlmMux()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.orchestrate_slms("query", {"slm1": "solution1"})

    async def test_orchestrate_updates_active_slms(self, initialized_ensembler: SlmMux) -> None:
        """Test orchestrate_slms updates active SLMs list."""
        await initialized_ensembler.orchestrate_slms(
            "query",
            {"slm1": "solution1", "slm2": "solution2"},
        )
        assert "slm1" in initialized_ensembler._active_slms
        assert "slm2" in initialized_ensembler._active_slms

    async def test_orchestrate_initializes_weights(self, initialized_ensembler: SlmMux) -> None:
        """Test orchestrate_slms initializes equal weights."""
        await initialized_ensembler.orchestrate_slms(
            "query",
            {"slm1": "solution1", "slm2": "solution2"},
        )
        assert "slm1" in initialized_ensembler._slm_weights
        assert "slm2" in initialized_ensembler._slm_weights
        # Equal weights
        assert initialized_ensembler._slm_weights["slm1"] == 0.5
        assert initialized_ensembler._slm_weights["slm2"] == 0.5

    async def test_orchestrate_returns_best_solution(self, initialized_ensembler: SlmMux) -> None:
        """Test orchestrate_slms returns best solution."""
        result = await initialized_ensembler.orchestrate_slms(
            "query",
            {
                "slm1": "short",
                "slm2": "x = 42 longer answer",
            },
        )
        assert "42" in result

    async def test_orchestrate_with_context(self, initialized_ensembler: SlmMux) -> None:
        """Test orchestrate_slms with context parameter."""
        result = await initialized_ensembler.orchestrate_slms(
            "query",
            {"slm1": "solution"},
            context={"domain": "math"},
        )
        assert result is not None


class TestSlmMuxSelectModels:
    """Tests for SLM-MUX model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> SlmMux:
        """Create an initialized SlmMux."""
        ensembler = SlmMux()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = SlmMux()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(self, initialized_ensembler: SlmMux) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)

    async def test_select_models_respects_max(self, initialized_ensembler: SlmMux) -> None:
        """Test select_models respects max_models limit."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) <= SLM_MUX_METADATA.max_models


class TestSlmMuxHealthCheck:
    """Tests for SLM-MUX health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = SlmMux()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = SlmMux()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
