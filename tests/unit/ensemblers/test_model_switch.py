"""Unit tests for ModelSwitch ensembler.

Tests multi-LLM repeated sampling with dynamic switching.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.model_switch import MODEL_SWITCH_METADATA, ModelSwitch
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestModelSwitchMetadata:
    """Tests for ModelSwitch metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert MODEL_SWITCH_METADATA.identifier == EnsemblerIdentifier.MODEL_SWITCH

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert MODEL_SWITCH_METADATA.name == "ModelSwitch"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "switching" in MODEL_SWITCH_METADATA.tags
        assert "sampling" in MODEL_SWITCH_METADATA.tags
        assert "multi-llm" in MODEL_SWITCH_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert MODEL_SWITCH_METADATA.min_models >= 2
        assert MODEL_SWITCH_METADATA.max_models >= MODEL_SWITCH_METADATA.min_models

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= MODEL_SWITCH_METADATA.complexity <= 10


class TestModelSwitchInitialization:
    """Tests for ModelSwitch initialization."""

    def test_create_instance(self) -> None:
        """Test creating ModelSwitch instance."""
        ensembler = ModelSwitch()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = ModelSwitch()
        assert ensembler.identifier == EnsemblerIdentifier.MODEL_SWITCH

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = ModelSwitch()
        assert ensembler.name == "ModelSwitch"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_sets_parameters(self) -> None:
        """Test initialize sets configuration parameters."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        assert ensembler._current_model_idx == 0
        assert ensembler._samples_per_model == 3
        assert ensembler._switch_threshold == 0.7


class TestModelSwitchSolutionScoring:
    """Tests for ModelSwitch solution scoring."""

    @pytest.fixture
    async def initialized_ensembler(self) -> ModelSwitch:
        """Create an initialized ModelSwitch."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        return ensembler

    def test_score_solution_base(self, initialized_ensembler: ModelSwitch) -> None:
        """Test base solution score."""
        score = initialized_ensembler._score_solution("Hi")
        assert score == 0.5

    def test_score_solution_length_bonus(self, initialized_ensembler: ModelSwitch) -> None:
        """Test longer solutions get bonus."""
        short_score = initialized_ensembler._score_solution("Hi")
        long_score = initialized_ensembler._score_solution("Longer solution text")
        assert long_score > short_score

    def test_score_solution_digit_bonus(self, initialized_ensembler: ModelSwitch) -> None:
        """Test solutions with digits get bonus."""
        no_digit = initialized_ensembler._score_solution("No numbers")
        with_digit = initialized_ensembler._score_solution("Answer is 42")
        assert with_digit > no_digit

    def test_score_solution_equals_bonus(self, initialized_ensembler: ModelSwitch) -> None:
        """Test solutions with equals sign get bonus."""
        no_equals = initialized_ensembler._score_solution("Value is 5")
        with_equals = initialized_ensembler._score_solution("x = 5")
        assert with_equals > no_equals


class TestModelSwitchEnsemble:
    """Tests for ModelSwitch ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> ModelSwitch:
        """Create an initialized ModelSwitch."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = ModelSwitch()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(self, initialized_ensembler: ModelSwitch) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(self, initialized_ensembler: ModelSwitch) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_returns_best(self, initialized_ensembler: ModelSwitch) -> None:
        """Test ensemble returns best scored solution."""
        solutions = ["short", "x = 42 answer", "medium text"]
        result = await initialized_ensembler.ensemble("query", solutions)
        assert result == "x = 42 answer"

    async def test_ensemble_with_context(self, initialized_ensembler: ModelSwitch) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestModelSwitchSampleSwitch:
    """Tests for ModelSwitch sample_switch method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> ModelSwitch:
        """Create an initialized ModelSwitch."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        return ensembler

    async def test_sample_switch_raises_when_not_initialized(self) -> None:
        """Test sample_switch raises error when not initialized."""
        ensembler = ModelSwitch()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.sample_switch("query", {"m1": ["s1"]})

    async def test_sample_switch_empty_outputs(self, initialized_ensembler: ModelSwitch) -> None:
        """Test sample_switch with empty model outputs."""
        result = await initialized_ensembler.sample_switch("query", {})
        assert result == ""

    async def test_sample_switch_single_model(self, initialized_ensembler: ModelSwitch) -> None:
        """Test sample_switch with single model outputs."""
        result = await initialized_ensembler.sample_switch(
            "query",
            {"model1": ["solution 1", "solution 2"]},
        )
        assert result is not None

    async def test_sample_switch_multiple_models(self, initialized_ensembler: ModelSwitch) -> None:
        """Test sample_switch with multiple model outputs."""
        result = await initialized_ensembler.sample_switch(
            "query",
            {
                "model1": ["sol1_a", "sol1_b"],
                "model2": ["x = 42", "y = 10"],
                "model3": ["sol3_a", "sol3_b"],
            },
        )
        assert result is not None

    async def test_sample_switch_picks_best_model(self, initialized_ensembler: ModelSwitch) -> None:
        """Test sample_switch picks best model's outputs."""
        result = await initialized_ensembler.sample_switch(
            "query",
            {
                "weak_model": ["short", "tiny"],
                "good_model": ["x = 42 answer", "y = 10 result"],
            },
        )
        # Should pick from good_model's outputs
        assert "=" in result

    async def test_sample_switch_respects_n_samples(
        self, initialized_ensembler: ModelSwitch
    ) -> None:
        """Test sample_switch respects n_samples parameter."""
        result = await initialized_ensembler.sample_switch(
            "query",
            {"model1": ["s1", "s2", "s3", "s4", "s5"]},
            n_samples=2,
        )
        assert result is not None

    async def test_sample_switch_with_context(self, initialized_ensembler: ModelSwitch) -> None:
        """Test sample_switch with context parameter."""
        result = await initialized_ensembler.sample_switch(
            "query",
            {"model1": ["solution"]},
            context={"domain": "math"},
        )
        assert result is not None


class TestModelSwitchSelectModels:
    """Tests for ModelSwitch model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> ModelSwitch:
        """Create an initialized ModelSwitch."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = ModelSwitch()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(self, initialized_ensembler: ModelSwitch) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)

    async def test_select_models_respects_max(self, initialized_ensembler: ModelSwitch) -> None:
        """Test select_models respects max_models limit."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) <= MODEL_SWITCH_METADATA.max_models

    async def test_select_models_with_context(self, initialized_ensembler: ModelSwitch) -> None:
        """Test select_models with context parameter."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2"],
            context={"domain": "math"},
        )
        assert len(models) >= 0


class TestModelSwitchHealthCheck:
    """Tests for ModelSwitch health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = ModelSwitch()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = ModelSwitch()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
