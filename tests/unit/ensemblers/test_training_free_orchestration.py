"""Unit tests for TrainingFreeOrchestration ensembler.

Tests training-free multi-model orchestration.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.training_free_orchestration import (
    TRAINING_FREE_ORCHESTRATION_METADATA,
    TrainingFreeOrchestration,
)
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestTrainingFreeOrchestrationMetadata:
    """Tests for TrainingFreeOrchestration metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert (
            TRAINING_FREE_ORCHESTRATION_METADATA.identifier
            == EnsemblerIdentifier.TRAINING_FREE_ORCHESTRATION
        )

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert TRAINING_FREE_ORCHESTRATION_METADATA.name == "Training-Free Orchestration"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "training-free" in TRAINING_FREE_ORCHESTRATION_METADATA.tags
        assert "orchestration" in TRAINING_FREE_ORCHESTRATION_METADATA.tags
        assert "controller" in TRAINING_FREE_ORCHESTRATION_METADATA.tags
        assert "routing" in TRAINING_FREE_ORCHESTRATION_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert TRAINING_FREE_ORCHESTRATION_METADATA.min_models >= 2
        assert (
            TRAINING_FREE_ORCHESTRATION_METADATA.max_models
            >= TRAINING_FREE_ORCHESTRATION_METADATA.min_models
        )

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= TRAINING_FREE_ORCHESTRATION_METADATA.complexity <= 10


class TestTrainingFreeOrchestrationInitialization:
    """Tests for TrainingFreeOrchestration initialization."""

    def test_create_instance(self) -> None:
        """Test creating TrainingFreeOrchestration instance."""
        ensembler = TrainingFreeOrchestration()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = TrainingFreeOrchestration()
        assert ensembler.identifier == EnsemblerIdentifier.TRAINING_FREE_ORCHESTRATION

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = TrainingFreeOrchestration()
        assert ensembler.name == "Training-Free Orchestration"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_sets_specialist_registry(self) -> None:
        """Test initialize sets specialist registry."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        assert "math" in ensembler._specialist_registry
        assert "code" in ensembler._specialist_registry
        assert "reasoning" in ensembler._specialist_registry
        assert "knowledge" in ensembler._specialist_registry


class TestTrainingFreeOrchestrationTaskClassification:
    """Tests for TrainingFreeOrchestration task classification."""

    @pytest.fixture
    async def initialized_ensembler(self) -> TrainingFreeOrchestration:
        """Create an initialized TrainingFreeOrchestration."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        return ensembler

    def test_classify_math_task(self, initialized_ensembler: TrainingFreeOrchestration) -> None:
        """Test classification of math tasks."""
        assert initialized_ensembler._classify_task("calculate 2+2") == "math"
        assert initialized_ensembler._classify_task("solve this equation") == "math"

    def test_classify_code_task(self, initialized_ensembler: TrainingFreeOrchestration) -> None:
        """Test classification of code tasks."""
        assert initialized_ensembler._classify_task("write a function") == "code"
        assert initialized_ensembler._classify_task("fix this bug") == "code"

    def test_classify_reasoning_task(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test classification of reasoning tasks."""
        assert initialized_ensembler._classify_task("explain why this works") == "reasoning"
        assert initialized_ensembler._classify_task("how does this happen") == "reasoning"

    def test_classify_knowledge_task_default(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test classification defaults to knowledge."""
        assert initialized_ensembler._classify_task("what is Python") == "knowledge"


class TestTrainingFreeOrchestrationScoring:
    """Tests for TrainingFreeOrchestration task scoring."""

    @pytest.fixture
    async def initialized_ensembler(self) -> TrainingFreeOrchestration:
        """Create an initialized TrainingFreeOrchestration."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        return ensembler

    def test_score_math_solution_digits(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test math solutions with digits score higher."""
        score_no_digit = initialized_ensembler._score_for_task("no numbers", "math")
        score_with_digit = initialized_ensembler._score_for_task("answer is 42", "math")
        assert score_with_digit > score_no_digit

    def test_score_math_solution_equals(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test math solutions with equals sign score higher."""
        score_no_equals = initialized_ensembler._score_for_task("value 5", "math")
        score_with_equals = initialized_ensembler._score_for_task("x = 5", "math")
        assert score_with_equals > score_no_equals

    def test_score_code_solution_keywords(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test code solutions with keywords score higher."""
        score_basic = initialized_ensembler._score_for_task("some text", "code")
        score_code = initialized_ensembler._score_for_task("def foo(): return 1", "code")
        assert score_code > score_basic

    def test_score_reasoning_solution_keywords(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test reasoning solutions with keywords score higher."""
        score_basic = initialized_ensembler._score_for_task("some text", "reasoning")
        score_reasoning = initialized_ensembler._score_for_task(
            "therefore, this is true because of that", "reasoning"
        )
        assert score_reasoning > score_basic

    def test_score_capped_at_one(self, initialized_ensembler: TrainingFreeOrchestration) -> None:
        """Test scores are capped at 1.0."""
        score = initialized_ensembler._score_for_task("x = 42 answer", "math")
        assert score <= 1.0


class TestTrainingFreeOrchestrationEnsemble:
    """Tests for TrainingFreeOrchestration ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> TrainingFreeOrchestration:
        """Create an initialized TrainingFreeOrchestration."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = TrainingFreeOrchestration()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_math_query_prefers_digits(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test ensemble for math query prefers solutions with digits."""
        result = await initialized_ensembler.ensemble(
            "calculate the sum",
            ["no numbers here", "the answer is 42"],
        )
        assert "42" in result

    async def test_ensemble_code_query_prefers_keywords(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test ensemble for code query prefers solutions with code keywords."""
        result = await initialized_ensembler.ensemble(
            "write a function",
            ["some text", "def foo(): return 1"],
        )
        assert "def" in result

    async def test_ensemble_with_context(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestTrainingFreeOrchestrationOrchestrate:
    """Tests for TrainingFreeOrchestration orchestrate method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> TrainingFreeOrchestration:
        """Create an initialized TrainingFreeOrchestration."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        return ensembler

    async def test_orchestrate_raises_when_not_initialized(self) -> None:
        """Test orchestrate raises error when not initialized."""
        ensembler = TrainingFreeOrchestration()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.orchestrate("query", {"spec1": "output1"})

    async def test_orchestrate_empty_outputs(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test orchestrate with empty specialist outputs."""
        result = await initialized_ensembler.orchestrate("query", {})
        assert result == ""

    async def test_orchestrate_single_specialist(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test orchestrate with single specialist."""
        result = await initialized_ensembler.orchestrate(
            "query",
            {"specialist": "output"},
        )
        assert result == "output"

    async def test_orchestrate_prefers_relevant_specialist(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test orchestrate prefers relevant specialist for task."""
        result = await initialized_ensembler.orchestrate(
            "calculate the sum",
            {
                "calculator": "x = 42",
                "generic": "some text",
            },
        )
        # Calculator is registered for math, should be preferred
        assert "42" in result

    async def test_orchestrate_with_context(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test orchestrate with context parameter."""
        result = await initialized_ensembler.orchestrate(
            "query",
            {"specialist": "output"},
            context={"domain": "math"},
        )
        assert result is not None


class TestTrainingFreeOrchestrationSelectModels:
    """Tests for TrainingFreeOrchestration model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> TrainingFreeOrchestration:
        """Create an initialized TrainingFreeOrchestration."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = TrainingFreeOrchestration()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3"],
        )
        assert isinstance(models, list)

    async def test_select_models_prioritizes_relevant(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test select_models prioritizes relevant specialists."""
        models = await initialized_ensembler.select_models(
            "calculate the sum",
            ["calculator", "generic", "solver"],
        )
        # Math specialists should be prioritized
        assert "calculator" in models or "solver" in models

    async def test_select_models_respects_max(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test select_models respects max_models limit."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) <= TRAINING_FREE_ORCHESTRATION_METADATA.max_models

    async def test_select_models_with_context(
        self, initialized_ensembler: TrainingFreeOrchestration
    ) -> None:
        """Test select_models with context parameter."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2"],
            context={"domain": "math"},
        )
        assert len(models) >= 0


class TestTrainingFreeOrchestrationHealthCheck:
    """Tests for TrainingFreeOrchestration health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = TrainingFreeOrchestration()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = TrainingFreeOrchestration()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
