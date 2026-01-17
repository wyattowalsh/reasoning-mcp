"""Unit tests for MultiAgentVerification ensembler.

Tests multi-agent cross-verification.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemblers.multi_agent_verification import (
    MULTI_AGENT_VERIFICATION_METADATA,
    MultiAgentVerification,
)
from reasoning_mcp.models.core import EnsemblerIdentifier


class TestMultiAgentVerificationMetadata:
    """Tests for MultiAgentVerification metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert (
            MULTI_AGENT_VERIFICATION_METADATA.identifier
            == EnsemblerIdentifier.MULTI_AGENT_VERIFICATION
        )

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert MULTI_AGENT_VERIFICATION_METADATA.name == "Multi-Agent Verification"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "verification" in MULTI_AGENT_VERIFICATION_METADATA.tags
        assert "multi-agent" in MULTI_AGENT_VERIFICATION_METADATA.tags
        assert "cross-check" in MULTI_AGENT_VERIFICATION_METADATA.tags
        assert "consensus" in MULTI_AGENT_VERIFICATION_METADATA.tags

    def test_metadata_model_limits(self) -> None:
        """Test metadata has valid model limits."""
        assert MULTI_AGENT_VERIFICATION_METADATA.min_models >= 2
        assert (
            MULTI_AGENT_VERIFICATION_METADATA.max_models
            >= MULTI_AGENT_VERIFICATION_METADATA.min_models
        )

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= MULTI_AGENT_VERIFICATION_METADATA.complexity <= 10


class TestMultiAgentVerificationInitialization:
    """Tests for MultiAgentVerification initialization."""

    def test_create_instance(self) -> None:
        """Test creating MultiAgentVerification instance."""
        ensembler = MultiAgentVerification()
        assert ensembler is not None
        assert ensembler._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        ensembler = MultiAgentVerification()
        assert ensembler.identifier == EnsemblerIdentifier.MULTI_AGENT_VERIFICATION

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        ensembler = MultiAgentVerification()
        assert ensembler.name == "Multi-Agent Verification"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        assert ensembler._initialized is True

    async def test_initialize_sets_parameters(self) -> None:
        """Test initialize sets configuration parameters."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        assert ensembler._verification_rounds == 2
        assert ensembler._consensus_threshold == 0.6


class TestMultiAgentVerificationAgreementComputation:
    """Tests for MultiAgentVerification agreement computation."""

    @pytest.fixture
    async def initialized_ensembler(self) -> MultiAgentVerification:
        """Create an initialized MultiAgentVerification."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        return ensembler

    def test_compute_agreement_identical_solutions(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test agreement between identical solutions."""
        agreement = initialized_ensembler._compute_agreement("hello world", "hello world")
        assert agreement == 1.0

    def test_compute_agreement_no_overlap(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test agreement between solutions with no overlap."""
        agreement = initialized_ensembler._compute_agreement("hello world", "foo bar")
        assert agreement == 0.0

    def test_compute_agreement_partial_overlap(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test agreement between solutions with partial overlap."""
        agreement = initialized_ensembler._compute_agreement("hello world", "hello foo")
        assert 0.0 < agreement < 1.0

    def test_compute_agreement_empty_solutions(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test agreement with empty solutions."""
        agreement = initialized_ensembler._compute_agreement("", "")
        assert agreement == 0.5  # Default for empty


class TestMultiAgentVerificationCrossVerify:
    """Tests for MultiAgentVerification cross_verify."""

    @pytest.fixture
    async def initialized_ensembler(self) -> MultiAgentVerification:
        """Create an initialized MultiAgentVerification."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        return ensembler

    async def test_cross_verify_returns_list(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test cross_verify returns list of tuples."""
        result = await initialized_ensembler._cross_verify(["solution 1", "solution 2"])
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    async def test_cross_verify_scores_between_0_and_1(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test cross_verify scores are between 0 and 1."""
        result = await initialized_ensembler._cross_verify(["hello world", "hello foo", "bar baz"])
        for _, score in result:
            assert 0.0 <= score <= 1.0


class TestMultiAgentVerificationEnsemble:
    """Tests for MultiAgentVerification ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> MultiAgentVerification:
        """Create an initialized MultiAgentVerification."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        return ensembler

    async def test_ensemble_raises_when_not_initialized(self) -> None:
        """Test ensemble raises error when not initialized."""
        ensembler = MultiAgentVerification()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.ensemble("query", ["sol1", "sol2"])

    async def test_ensemble_empty_solutions(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test ensemble with empty solutions list."""
        result = await initialized_ensembler.ensemble("query", [])
        assert result == ""

    async def test_ensemble_single_solution(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test ensemble with single solution."""
        result = await initialized_ensembler.ensemble("query", ["only solution"])
        assert result == "only solution"

    async def test_ensemble_returns_best_verified(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test ensemble returns best verified solution."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["hello world", "hello foo", "hello world"],  # Similar solutions
        )
        assert result is not None

    async def test_ensemble_with_context(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test ensemble with context parameter."""
        result = await initialized_ensembler.ensemble(
            "query",
            ["solution 1", "solution 2"],
            context={"domain": "math"},
        )
        assert result is not None


class TestMultiAgentVerificationVerifyEnsemble:
    """Tests for MultiAgentVerification verify_ensemble method."""

    @pytest.fixture
    async def initialized_ensembler(self) -> MultiAgentVerification:
        """Create an initialized MultiAgentVerification."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        return ensembler

    async def test_verify_ensemble_raises_when_not_initialized(self) -> None:
        """Test verify_ensemble raises error when not initialized."""
        ensembler = MultiAgentVerification()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.verify_ensemble(["sol1", "sol2"])

    async def test_verify_ensemble_empty_solutions(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test verify_ensemble with empty solutions."""
        result, confidence = await initialized_ensembler.verify_ensemble([])
        assert result == ""
        assert confidence == 0.0

    async def test_verify_ensemble_returns_tuple(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test verify_ensemble returns tuple of solution and confidence."""
        result, confidence = await initialized_ensembler.verify_ensemble(
            ["solution 1", "solution 2"]
        )
        assert isinstance(result, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestMultiAgentVerificationSelectModels:
    """Tests for MultiAgentVerification model selection."""

    @pytest.fixture
    async def initialized_ensembler(self) -> MultiAgentVerification:
        """Create an initialized MultiAgentVerification."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        return ensembler

    async def test_select_models_raises_when_not_initialized(self) -> None:
        """Test select_models raises error when not initialized."""
        ensembler = MultiAgentVerification()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await ensembler.select_models("query", ["model1", "model2"])

    async def test_select_models_returns_list(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test select_models returns list of models."""
        models = await initialized_ensembler.select_models(
            "query",
            ["model1", "model2", "model3", "model4"],
        )
        assert isinstance(models, list)

    async def test_select_models_respects_limits(
        self, initialized_ensembler: MultiAgentVerification
    ) -> None:
        """Test select_models respects min/max limits."""
        many_models = [f"model{i}" for i in range(20)]
        selected = await initialized_ensembler.select_models("query", many_models)
        assert len(selected) >= MULTI_AGENT_VERIFICATION_METADATA.min_models
        assert len(selected) <= MULTI_AGENT_VERIFICATION_METADATA.max_models


class TestMultiAgentVerificationHealthCheck:
    """Tests for MultiAgentVerification health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        ensembler = MultiAgentVerification()
        assert await ensembler.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        ensembler = MultiAgentVerification()
        await ensembler.initialize()
        assert await ensembler.health_check() is True
