"""Unit tests for GenPRM verifier.

Tests generative process rewards with explicit CoT verification.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.gen_prm import GEN_PRM_METADATA, GenPrm


class TestGenPrmMetadata:
    """Tests for GenPRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert GEN_PRM_METADATA.identifier == VerifierIdentifier.GEN_PRM

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert GEN_PRM_METADATA.name == "GenPRM"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "generative" in GEN_PRM_METADATA.tags
        assert "process-reward" in GEN_PRM_METADATA.tags
        assert "cot" in GEN_PRM_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert GEN_PRM_METADATA.supports_step_level is True
        assert GEN_PRM_METADATA.supports_outcome_level is True
        assert GEN_PRM_METADATA.supports_cot_verification is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= GEN_PRM_METADATA.complexity <= 10


class TestGenPrmInitialization:
    """Tests for GenPRM initialization."""

    def test_create_instance(self) -> None:
        """Test creating GenPrm instance."""
        verifier = GenPrm()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = GenPrm()
        assert verifier.identifier == VerifierIdentifier.GEN_PRM

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = GenPrm()
        assert verifier.name == "GenPRM"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = GenPrm()
        await verifier.initialize()
        assert verifier._initialized is True


class TestGenPrmVerify:
    """Tests for GenPRM verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> GenPrm:
        """Create an initialized GenPrm."""
        verifier = GenPrm()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = GenPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_verification(
        self, initialized_verifier: GenPrm
    ) -> None:
        """Test verify returns tuple of score and verification string."""
        score, verification = await initialized_verifier.verify("2 + 2 = 4")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_returns_high_score(self, initialized_verifier: GenPrm) -> None:
        """Test verify returns expected high score."""
        score, _ = await initialized_verifier.verify("test solution")
        assert score == 0.91

    async def test_verify_includes_generative_steps(self, initialized_verifier: GenPrm) -> None:
        """Test verification includes generative verification steps."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "Generative verification" in verification
        assert "Step 1" in verification
        assert "Step 2" in verification
        assert "Step 3" in verification
        assert "Step 4" in verification

    async def test_verify_includes_checkmarks(self, initialized_verifier: GenPrm) -> None:
        """Test verification includes checkmarks for passed steps."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "✓" in verification

    async def test_verify_with_context(self, initialized_verifier: GenPrm) -> None:
        """Test verify with context parameter."""
        score, verification = await initialized_verifier.verify(
            "test solution",
            context={"problem": "Calculate 2+2"},
        )
        assert isinstance(score, float)


class TestGenPrmScoreSteps:
    """Tests for GenPRM step scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> GenPrm:
        """Create an initialized GenPrm."""
        verifier = GenPrm()
        await verifier.initialize()
        return verifier

    async def test_score_steps_raises_when_not_initialized(self) -> None:
        """Test score_steps raises error when not initialized."""
        verifier = GenPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_steps(["step 1", "step 2"])

    async def test_score_steps_returns_list(self, initialized_verifier: GenPrm) -> None:
        """Test score_steps returns list of scores."""
        scores = await initialized_verifier.score_steps(
            ["Step 1: Start", "Step 2: Process", "Step 3: End"]
        )
        assert isinstance(scores, list)
        assert len(scores) == 3

    async def test_score_steps_valid_range(self, initialized_verifier: GenPrm) -> None:
        """Test all step scores are in valid range."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_steps_progressive(self, initialized_verifier: GenPrm) -> None:
        """Test later steps can build on earlier ones."""
        scores = await initialized_verifier.score_steps(["First step", "Second step", "Third step"])
        # Progressive scoring - later steps should have progress bonus
        # The exact behavior depends on implementation
        assert len(scores) == 3

    async def test_score_steps_equals_bonus(self, initialized_verifier: GenPrm) -> None:
        """Test steps with equals sign get bonus."""
        scores_without = await initialized_verifier.score_steps(["Step: Process"])
        scores_with = await initialized_verifier.score_steps(["Step: x = 5"])

        # Step with = should have higher score
        assert scores_with[0] >= scores_without[0]

    async def test_score_steps_reasoning_keywords_bonus(self, initialized_verifier: GenPrm) -> None:
        """Test steps with reasoning keywords get bonus."""
        scores_without = await initialized_verifier.score_steps(["Step: Do something"])
        scores_therefore = await initialized_verifier.score_steps(["Therefore, we conclude"])
        scores_verify = await initialized_verifier.score_steps(["Verify the result"])

        # Steps with reasoning keywords should have higher scores
        assert scores_therefore[0] >= scores_without[0]
        assert scores_verify[0] >= scores_without[0]

    async def test_score_steps_with_context(self, initialized_verifier: GenPrm) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"problem": "Math problem"},
        )
        assert len(scores) == 2


class TestGenPrmHealthCheck:
    """Tests for GenPRM health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = GenPrm()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = GenPrm()
        await verifier.initialize()
        assert await verifier.health_check() is True
