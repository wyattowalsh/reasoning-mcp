"""Unit tests for GAR Discriminator verifier.

Tests adversarial discriminator for reasoning verification.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.gar_discriminator import (
    GAR_DISCRIMINATOR_METADATA,
    GarDiscriminator,
)


class TestGarDiscriminatorMetadata:
    """Tests for GAR Discriminator metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert GAR_DISCRIMINATOR_METADATA.identifier == VerifierIdentifier.GAR_DISCRIMINATOR

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert GAR_DISCRIMINATOR_METADATA.name == "GAR-Discriminator"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "adversarial" in GAR_DISCRIMINATOR_METADATA.tags
        assert "discriminator" in GAR_DISCRIMINATOR_METADATA.tags
        assert "gar" in GAR_DISCRIMINATOR_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert GAR_DISCRIMINATOR_METADATA.supports_step_level is True
        assert GAR_DISCRIMINATOR_METADATA.supports_outcome_level is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= GAR_DISCRIMINATOR_METADATA.complexity <= 10


class TestGarDiscriminatorInitialization:
    """Tests for GAR Discriminator initialization."""

    def test_create_instance(self) -> None:
        """Test creating GarDiscriminator instance."""
        verifier = GarDiscriminator()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = GarDiscriminator()
        assert verifier.identifier == VerifierIdentifier.GAR_DISCRIMINATOR

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = GarDiscriminator()
        assert verifier.name == "GAR-Discriminator"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        assert verifier._initialized is True

    async def test_initialize_sets_threshold(self) -> None:
        """Test initialize sets discrimination threshold."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        assert verifier._discrimination_threshold == 0.5


class TestGarDiscriminatorDiscrimination:
    """Tests for GAR Discriminator discrimination logic."""

    @pytest.fixture
    async def initialized_verifier(self) -> GarDiscriminator:
        """Create an initialized GarDiscriminator."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        return verifier

    def test_discriminate_empty_solution(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination of empty solution."""
        score = initialized_verifier._discriminate("")
        assert score == 0.5  # Base score

    def test_discriminate_short_solution(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination of short solution."""
        score = initialized_verifier._discriminate("Hi")
        assert score == 0.5  # No length bonus

    def test_discriminate_long_solution_bonus(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination gives bonus for longer solutions."""
        score = initialized_verifier._discriminate("This is a longer solution text")
        assert score >= 0.6  # Base + length bonus

    def test_discriminate_with_digits_bonus(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination gives bonus for solutions with digits."""
        score = initialized_verifier._discriminate("The answer is 42")
        assert score >= 0.6  # Should have digit bonus

    def test_discriminate_with_equals_bonus(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination gives bonus for solutions with equals sign."""
        score = initialized_verifier._discriminate("x = 5")
        assert score >= 0.6  # Should have equals bonus

    def test_discriminate_reasoning_keywords_bonus(
        self, initialized_verifier: GarDiscriminator
    ) -> None:
        """Test discrimination gives bonus for reasoning keywords."""
        score_therefore = initialized_verifier._discriminate("Therefore, the conclusion is correct")
        score_because = initialized_verifier._discriminate("Because of this, we can deduce that...")
        score_thus = initialized_verifier._discriminate("Thus, the solution follows naturally")

        # All should have reasoning keyword bonus
        assert score_therefore >= 0.6
        assert score_because >= 0.6
        assert score_thus >= 0.6

    def test_discriminate_max_score_capped(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discrimination score is capped at 1.0."""
        # Solution with all bonuses
        score = initialized_verifier._discriminate("Therefore, the answer = 42, because...")
        assert score <= 1.0


class TestGarDiscriminatorVerify:
    """Tests for GAR Discriminator verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> GarDiscriminator:
        """Create an initialized GarDiscriminator."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = GarDiscriminator()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_reasoning(
        self, initialized_verifier: GarDiscriminator
    ) -> None:
        """Test verify returns tuple of score and reasoning."""
        score, reasoning = await initialized_verifier.verify("test solution text here")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    async def test_verify_reasoning_contains_classification(
        self, initialized_verifier: GarDiscriminator
    ) -> None:
        """Test verify reasoning contains classification decision."""
        _, reasoning = await initialized_verifier.verify("Therefore, answer = 42")
        assert "ACCEPT" in reasoning or "REJECT" in reasoning

    async def test_verify_accept_high_quality(self, initialized_verifier: GarDiscriminator) -> None:
        """Test verify accepts high-quality solutions."""
        score, reasoning = await initialized_verifier.verify(
            "Therefore, the answer = 42 because..."
        )
        assert score > initialized_verifier._discrimination_threshold
        assert "ACCEPT" in reasoning

    async def test_verify_with_context(self, initialized_verifier: GarDiscriminator) -> None:
        """Test verify with context parameter."""
        score, reasoning = await initialized_verifier.verify(
            "test solution",
            context={"domain": "math"},
        )
        assert isinstance(score, float)


class TestGarDiscriminatorDiscriminateBatch:
    """Tests for GAR Discriminator batch discrimination."""

    @pytest.fixture
    async def initialized_verifier(self) -> GarDiscriminator:
        """Create an initialized GarDiscriminator."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        return verifier

    async def test_discriminate_raises_when_not_initialized(self) -> None:
        """Test discriminate raises error when not initialized."""
        verifier = GarDiscriminator()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.discriminate(["real"], ["generated"])

    async def test_discriminate_returns_two_lists(
        self, initialized_verifier: GarDiscriminator
    ) -> None:
        """Test discriminate returns two score lists."""
        real_scores, gen_scores = await initialized_verifier.discriminate(
            ["real solution 1", "real solution 2"],
            ["generated solution 1"],
        )
        assert len(real_scores) == 2
        assert len(gen_scores) == 1

    async def test_discriminate_real_scores_higher(
        self, initialized_verifier: GarDiscriminator
    ) -> None:
        """Test real solutions get bias bonus."""
        real_scores, gen_scores = await initialized_verifier.discriminate(
            ["test solution"],
            ["test solution"],  # Same content
        )
        # Real should be higher due to +0.1 bias
        assert real_scores[0] > gen_scores[0]

    async def test_discriminate_with_context(self, initialized_verifier: GarDiscriminator) -> None:
        """Test discriminate with context parameter."""
        real_scores, gen_scores = await initialized_verifier.discriminate(
            ["real"],
            ["generated"],
            context={"domain": "math"},
        )
        assert len(real_scores) == 1
        assert len(gen_scores) == 1


class TestGarDiscriminatorScoreSteps:
    """Tests for GAR Discriminator step scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> GarDiscriminator:
        """Create an initialized GarDiscriminator."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        return verifier

    async def test_score_steps_raises_when_not_initialized(self) -> None:
        """Test score_steps raises error when not initialized."""
        verifier = GarDiscriminator()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_steps(["step 1", "step 2"])

    async def test_score_steps_returns_list(self, initialized_verifier: GarDiscriminator) -> None:
        """Test score_steps returns list of scores."""
        scores = await initialized_verifier.score_steps(
            ["Step 1: Analyze", "Step 2: Solve", "Step 3: Verify"]
        )
        assert isinstance(scores, list)
        assert len(scores) == 3

    async def test_score_steps_valid_range(self, initialized_verifier: GarDiscriminator) -> None:
        """Test all step scores are in valid range."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_steps_with_context(self, initialized_verifier: GarDiscriminator) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"domain": "math"},
        )
        assert len(scores) == 2


class TestGarDiscriminatorHealthCheck:
    """Tests for GAR Discriminator health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = GarDiscriminator()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = GarDiscriminator()
        await verifier.initialize()
        assert await verifier.health_check() is True
